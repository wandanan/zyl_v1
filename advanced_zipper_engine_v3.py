# advanced_zipper_engine_v3.py

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import time
import logging
from dataclasses import dataclass

# 依赖项检查
try:
    from FlagEmbedding import FlagModel
except ImportError:
    raise ImportError("FlagEmbedding未安装。请运行: pip install FlagEmbedding")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25未安装。请运行: pip install rank_bm25")

# --- 日志和设备配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- V3 配置定义 (更丰富，可控制) ---

@dataclass
class ZipperV3Config:
    bge_model_path: str = "models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620"
    embedding_dim: int = 512
    
    bm25_top_n: int = 100
    final_top_k: int = 10
    
    # 优化策略开关
    use_hybrid_search: bool = True
    bm25_weight: float = 1.0
    colbert_weight: float = 1.0

    use_length_penalty: bool = True
    length_penalty_alpha: float = 0.05

    use_multi_head: bool = True
    num_heads: int = 8

    use_stateful_reranking: bool = True
    context_memory_decay: float = 0.8
    context_influence: float = 0.3


@dataclass
class ZipperV3State:
    original_query: str
    context_vector: torch.Tensor
    
class TokenLevelEncoder:
    def __init__(self, model_path: str, use_fp16: bool = True):
        logger.info(f"正在加载BGE模型用于Token级编码: {model_path}")
        self.model = FlagModel(model_path, 
                               query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                               use_fp16=use_fp16 if device.type == 'cuda' else False)
        self.tokenizer = self.model.tokenizer
        logger.info("Token级编码器加载成功。")

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def encode_query(self, query: str) -> torch.Tensor:
        # BGE模型建议对查询添加指令
        query_embeddings = self.model.encode_queries([query])
        return torch.tensor(query_embeddings, device=device)

    def encode_documents(self, documents: List[str]) -> torch.Tensor:
        doc_embeddings = self.model.encode(documents, batch_size=64)
        return torch.tensor(doc_embeddings, device=device)

    def encode_tokens(self, text: str) -> torch.Tensor:
        # 使用encode_corpus获取每个token的向量
        # 注意：这需要FlagEmbedding的一个较新版本
        try:
            # `return_dense` 返回篇章向量，`return_token_embeddings`返回token向量
            output = self.model.encode(text, return_dense=False, return_token_embeddings=True)
            return torch.tensor(output['token_embeddings'], device=device)
        except Exception:
            # 回退到手动方法（效率较低）
            tokens = self.tokenize(text)
            if not tokens: return torch.empty(0, self.model.model.config.hidden_size, device=device)
            inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True).to(device)
            with torch.no_grad():
                outputs = self.model.model(**inputs, output_hidden_states=True)
                token_embeddings = outputs.hidden_states[-1].squeeze(0)
            # 移除[CLS]和[SEP]
            if token_embeddings.size(0) > 2:
                return token_embeddings[1:-1]
            return token_embeddings

# --- V3 主引擎 ---
class AdvancedZipperQueryEngineV3:
    def __init__(self, config: ZipperV3Config):
        self.config = config
        self.encoder = TokenLevelEncoder(config.bge_model_path)
        
        if config.use_multi_head and config.embedding_dim % config.num_heads != 0:
            raise ValueError("embedding_dim 必须能被 num_heads 整除")
        
        self.documents: Dict[int, str] = {}
        self.doc_token_embeddings: Dict[int, torch.Tensor] = {}
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_idx_to_pid: Dict[int, int] = {}
        
        logger.info("AdvancedZipperQueryEngine V3 初始化完成 (多策略优化版)")

    def build_document_index(self, documents: Dict[int, str]):
        logger.info(f"开始构建V3索引，共 {len(documents)} 个文档...")
        start_time = time.time()
        self.documents = documents
        
        doc_ids_sorted = sorted(self.documents.keys())
        corpus_list = [self.documents[pid] for pid in doc_ids_sorted]
        
        logger.info("构建BM25稀疏索引...")
        tokenized_corpus = [self.encoder.tokenize(doc) for doc in corpus_list]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.bm25_idx_to_pid = {i: pid for i, pid in enumerate(doc_ids_sorted)}
        
        logger.info("构建Token级稠密索引...")
        for i, pid in enumerate(doc_ids_sorted):
            self.doc_token_embeddings[pid] = self.encoder.encode_tokens(self.documents[pid])
            if (i + 1) % 100 == 0: logger.info(f"  已处理 {i+1}/{len(doc_ids_sorted)} 个文档的Token向量化")

        logger.info(f"V3索引构建完成，总耗时: {time.time() - start_time:.3f}秒")

    def _calculate_colbert_score(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> float:
        if query_emb.nelement() == 0 or doc_emb.nelement() == 0: return 0.0

        if self.config.use_multi_head:
            num_heads = self.config.num_heads
            head_dim = self.config.embedding_dim // num_heads
            
            # 检查维度兼容性
            q_size = query_emb.size(-1)
            d_size = doc_emb.size(-1)
            
            if q_size != self.config.embedding_dim or d_size != self.config.embedding_dim:
                # 如果维度不匹配，回退到单头模式
                logger.warning(f"维度不匹配，回退到单头模式: query_dim={q_size}, doc_dim={d_size}, expected={self.config.embedding_dim}")
                sim_matrix = F.cosine_similarity(query_emb.unsqueeze(1), doc_emb.unsqueeze(0), dim=-1)
                max_sim = sim_matrix.max(dim=1).values
                score = max_sim.sum().item()
            else:
                # 安全的reshape操作
                try:
                    q_heads = query_emb.view(-1, num_heads, head_dim)
                    d_heads = doc_emb.view(-1, num_heads, head_dim)
                    head_scores = []
                    for i in range(num_heads):
                        sim_matrix_head = F.cosine_similarity(q_heads[:, i, :].unsqueeze(1), d_heads[:, i, :].unsqueeze(0), dim=-1)
                        max_sim_head = sim_matrix_head.max(dim=1).values
                        head_scores.append(max_sim_head.sum())
                    score = sum(head_scores).item()
                except RuntimeError as e:
                    logger.warning(f"多头计算失败，回退到单头模式: {e}")
                    sim_matrix = F.cosine_similarity(query_emb.unsqueeze(1), doc_emb.unsqueeze(0), dim=-1)
                    max_sim = sim_matrix.max(dim=1).values
                    score = max_sim.sum().item()
        else:
            sim_matrix = F.cosine_similarity(query_emb.unsqueeze(1), doc_emb.unsqueeze(0), dim=-1)
            max_sim = sim_matrix.max(dim=1).values
            score = max_sim.sum().item()
            
        if self.config.use_length_penalty:
            num_doc_tokens = doc_emb.size(0)
            penalty = 1.0 + self.config.length_penalty_alpha * np.log(num_doc_tokens + 1)
            score /= penalty
            
        return score

    def retrieve(self, query: str, state: Optional[ZipperV3State] = None) -> List[Tuple[int, float, str]]:
        if self.bm25_index is None: return []

        logger.info(f"V3 开始检索，查询: '{query[:50]}...'")
        
        # 1. BM25召回
        query_tokens_list = self.encoder.tokenize(query)
        bm25_raw_scores = self.bm25_index.get_scores(query_tokens_list)
        bm25_candidate_indices = np.argsort(bm25_raw_scores)[::-1][:self.config.bm25_top_n]
        candidate_pids = [self.bm25_idx_to_pid[idx] for idx in bm25_candidate_indices]
        bm25_scores_map = {pid: bm25_raw_scores[idx] for idx, pid in zip(bm25_candidate_indices, candidate_pids)}

        # 2. 准备查询向量 (可能被状态化调整)
        query_tokens_emb = self.encoder.encode_tokens(query)
        if state and self.config.use_stateful_reranking and state.context_vector.sum() != 0:
            logger.info("应用状态化上下文调整查询...")
            influence = self.config.context_influence
            dynamic_adjustment = state.context_vector.unsqueeze(0) * influence
            query_tokens_emb += dynamic_adjustment

        # 3. 计算ColBERT分数
        colbert_scores_map = {pid: self._calculate_colbert_score(query_tokens_emb, self.doc_token_embeddings.get(pid, torch.empty(0))) for pid in candidate_pids}

        # 4. 分数融合
        fused_scores = {}
        bm25_vals = np.array(list(bm25_scores_map.values()))
        colbert_vals = np.array(list(colbert_scores_map.values()))
        bm25_min, bm25_max = (bm25_vals.min(), bm25_vals.max()) if bm25_vals.size > 1 else (0, 1)
        colbert_min, colbert_max = (colbert_vals.min(), colbert_vals.max()) if colbert_vals.size > 1 else (0, 1)
        
        for pid in candidate_pids:
            norm_bm25 = (bm25_scores_map.get(pid, 0) - bm25_min) / (bm25_max - bm25_min + 1e-9)
            norm_colbert = (colbert_scores_map.get(pid, 0) - colbert_min) / (colbert_max - colbert_min + 1e-9)
            
            if self.config.use_hybrid_search:
                fused_scores[pid] = (self.config.bm25_weight * norm_bm25 + self.config.colbert_weight * norm_colbert)
            else:
                fused_scores[pid] = norm_colbert
        
        sorted_pids = sorted(candidate_pids, key=lambda pid: fused_scores.get(pid, 0), reverse=True)
        
        return [(pid, fused_scores.get(pid, 0), self.documents[pid]) for pid in sorted_pids[:self.config.final_top_k]]

    def update_state(self, state: ZipperV3State, results: List[Tuple[int, float, str]]) -> ZipperV3State:
        if not results or not self.config.use_stateful_reranking: return state
        top_doc_pid = results[0][0]
        top_doc_tokens_emb = self.doc_token_embeddings.get(top_doc_pid)
        if top_doc_tokens_emb is not None and top_doc_tokens_emb.nelement() > 0:
            top_doc_avg_emb = top_doc_tokens_emb.mean(dim=0)
            decay = self.config.context_memory_decay
            state.context_vector = decay * state.context_vector + (1-decay) * top_doc_avg_emb
        return state