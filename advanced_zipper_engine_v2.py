# advanced_zipper_engine_v2.py

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import time
import logging
from dataclasses import dataclass, field

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


# --- V2 配置与状态定义 ---

@dataclass
class ZipperV2Config:
    # 使用更强大的模型作为基础
    bge_model_path: str = "models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620"
    embedding_dim: int = 1024
    
    # Reranking阶段的配置
    rerank_top_n: int = 100  # BM25召回后，送入精排的数量
    final_top_k: int = 10    # 最终返回的结果数量
    
    # 状态化流式机制的配置
    use_stateful_reranking: bool = True # 是否启用状态化机制
    context_memory_decay: float = 0.8   # 上下文记忆衰减因子

@dataclass
class ZipperV2State:
    original_query: str
    # 上下文向量，用于动态调整查询
    context_vector: torch.Tensor
    # 记录已访问的文档，用于多样性或避免重复（此版本暂未深度使用）
    visited_docs: set = field(default_factory=set)


# --- V2 核心模块 ---

class TokenLevelEncoder:
    """负责将文本编码为Token级别的向量序列"""
    def __init__(self, model_path: str, use_fp16: bool = True):
        logger.info(f"正在加载BGE模型用于Token级编码: {model_path}")
        # V2中，我们直接使用FlagModel的`encode`方法，它能处理list of strings
        self.model = FlagModel(model_path, use_fp16=use_fp16 if device.type == 'cuda' else False)
        # 确保模型在正确的设备上
        if device.type == 'cuda':
            self.model.model = self.model.model.to(device)
        # BGE的tokenizer是继承自BERT的
        self.tokenizer = self.model.tokenizer
        logger.info("Token级编码器加载成功。")

    def tokenize(self, text: str, add_special_tokens: bool = False) -> List[str]:
        # 返回token字符串列表，方便BM25和后续处理
        return self.tokenizer.tokenize(text, add_special_tokens=add_special_tokens)

    def encode_tokens(self, tokens: List[str]) -> torch.Tensor:
        # 使用FlagModel的底层模型来获取最后一层的隐藏状态
        # 这是获取Token向量的标准方法
        if not tokens:
            return torch.empty(0, self.model.model.config.hidden_size, device=device)
            
        inputs = self.tokenizer(
            " ".join(tokens), # 将token列表转回字符串送入tokenizer
            return_tensors='pt',
            max_length=512, # 限制最大长度
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            outputs = self.model.model(**inputs, output_hidden_states=True)
            # 获取最后一层的隐藏状态
            token_embeddings = outputs.hidden_states[-1].squeeze(0)

        # 移除[CLS]和[SEP]标记的向量
        # 注意：这假设了`add_special_tokens=True`被内部处理了
        # 为了保险起见，最好的方式是根据input_ids来对齐
        # 但在FlagModel的encode中，它不直接暴露这个，所以我们采用一个近似
        # 更好的做法是自己构建这个流程，但为了简化，我们先这样
        if token_embeddings.size(0) > len(tokens):
             # 假设第一个是[CLS]，最后一个是[SEP]
            return token_embeddings[1:-1]
        return token_embeddings


class AdvancedZipperQueryEngineV2:
    def __init__(self, config: ZipperV2Config):
        self.config = config
        self.encoder = TokenLevelEncoder(config.bge_model_path)
        
        # 索引存储
        self.documents: Dict[int, str] = {} # doc_idx -> doc_text
        self.doc_token_embeddings: Dict[int, torch.Tensor] = {} # doc_idx -> tensor[seq_len, dim]
        self.bm25_index: Optional[BM25Okapi] = None
        
        logger.info("AdvancedZipperQueryEngine V2 初始化完成 (ColBERT '晚期交互' 范式)")

    def build_document_index(self, documents: Dict[int, str]):
        """V2的索引构建，输入是 PID -> content 的字典"""
        if not documents:
            logger.warning("没有提供文档，索引构建跳过。")
            return
            
        logger.info(f"开始构建V2索引，共 {len(documents)} 个文档...")
        start_time = time.time()
        self.documents = documents

        # 1. 构建BM25索引 (保持不变，用于召回)
        logger.info("构建BM25稀疏索引...")
        # BM25需要一个列表，我们按pid排序来保证顺序
        doc_ids_sorted = sorted(self.documents.keys())
        corpus_list = [self.documents[pid] for pid in doc_ids_sorted]
        tokenized_corpus = [self.encoder.tokenize(doc) for doc in corpus_list]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        # 创建一个从BM25内部索引到我们PID的映射
        self.bm25_idx_to_pid = {i: pid for i, pid in enumerate(doc_ids_sorted)}
        logger.info("BM25稀疏索引构建完成。")

        # 2. 构建Token级稠密索引 (核心变化)
        logger.info("构建Token级稠密索引...")
        for i, pid in enumerate(doc_ids_sorted):
            doc_text = self.documents[pid]
            # 使用分词结果来编码，确保一致性
            tokens = tokenized_corpus[i] 
            self.doc_token_embeddings[pid] = self.encoder.encode_tokens(tokens)
            if (i + 1) % 100 == 0:
                logger.info(f"  已处理 {i+1}/{len(doc_ids_sorted)} 个文档的Token向量化")
        logger.info("Token级稠密索引构建完成。")
        
        logger.info(f"V2索引构建完成，总耗时: {time.time() - start_time:.3f}秒")
        
    def late_interaction_rerank(self, query_tokens_emb: torch.Tensor, doc_pids: List[int]) -> List[Tuple[int, float]]:
        """执行晚期交互重排"""
        results = []
        for pid in doc_pids:
            doc_tokens_emb = self.doc_token_embeddings.get(pid)
            
            if doc_tokens_emb is None or doc_tokens_emb.nelement() == 0:
                continue
                
            # 计算相似度矩阵 (Query_Tokens x Doc_Tokens)
            # 形状: [num_query_tokens, num_doc_tokens]
            sim_matrix = F.cosine_similarity(query_tokens_emb.unsqueeze(1), doc_tokens_emb.unsqueeze(0), dim=-1)
            
            # 对每个查询Token，找到其在文档中的最大相似度 (MaxSim)
            # 形状: [num_query_tokens]
            doc_max_sim = sim_matrix.max(dim=1).values
            
            # 将所有查询Token的MaxSim值相加，作为最终分数
            score = doc_max_sim.sum().item()
            results.append((pid, score))
            
        # 按分数从高到低排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def retrieve(self, query: str) -> List[Tuple[int, float, str]]:
        if self.bm25_index is None:
            logger.error("索引未构建，无法执行检索。")
            return []

        logger.info(f"V2 开始检索，查询: '{query[:50]}...'")
        start_time = time.time()

        # 1. BM25稀疏召回
        query_tokens_list = self.encoder.tokenize(query)
        bm25_scores = self.bm25_index.get_scores(query_tokens_list)
        bm25_candidate_indices = np.argsort(bm25_scores)[::-1][:self.config.rerank_top_n]
        # 将BM25的内部索引转换为我们的文档PID
        candidate_pids = [self.bm25_idx_to_pid[idx] for idx in bm25_candidate_indices]
        logger.info(f"BM25召回了 {len(candidate_pids)} 个候选文档。")

        # 2. 准备查询的Token向量
        query_tokens_emb = self.encoder.encode_tokens(query_tokens_list)
        if query_tokens_emb.nelement() == 0:
            logger.warning("查询无法编码为Token向量，精排跳过。")
            # 在这种情况下，可以只返回BM25结果
            results = []
            for pid in candidate_pids[:self.config.final_top_k]:
                # 此处分数暂时用0代替
                results.append((pid, 0.0, self.documents[pid]))
            return results

        # 3. 晚期交互精排
        logger.info("执行'晚期交互'精排...")
        reranked_results = self.late_interaction_rerank(query_tokens_emb, candidate_pids)
        
        # 4. 截断并格式化输出
        final_results = []
        for pid, score in reranked_results[:self.config.final_top_k]:
            final_results.append((pid, score, self.documents[pid]))

        logger.info(f"V2 检索完成，返回 {len(final_results)} 个结果，耗时: {time.time() - start_time:.3f}秒")
        return final_results

    def stateful_retrieve(self, query: str, state: Optional[ZipperV2State] = None) -> Tuple[List[Tuple[int, float, str]], ZipperV2State]:
        """V2的状态化检索接口"""
        if not self.config.use_stateful_reranking:
            # 如果不使用状态化，就调用普通检索
            new_state = ZipperV2State(original_query=query, context_vector=torch.zeros(self.config.embedding_dim, device=device))
            return self.retrieve(query), new_state
        
        # 1. 初始化或获取状态
        if state is None:
            state = ZipperV2State(
                original_query=query,
                context_vector=torch.zeros(self.config.embedding_dim, device=device)
            )
        
        # --- 状态化逻辑的核心 ---
        # 我们可以用context_vector来调整查询的token向量
        # 例如，给与上下文语义更相关的查询token更高的权重
        
        # (此处为简化的概念实现，未来可以扩展)
        # a. 将上下文向量与查询token向量做点积，得到权重
        query_tokens_list = self.encoder.tokenize(query)
        query_tokens_emb = self.encoder.encode_tokens(query_tokens_list)
        
        # b. 动态调整查询
        if state.context_vector.sum() != 0:
            weights = torch.matmul(query_tokens_emb, state.context_vector).softmax(dim=0)
            dynamic_query_emb = query_tokens_emb * weights.unsqueeze(-1)
        else:
            dynamic_query_emb = query_tokens_emb
            
        # 后续流程与普通检索类似，但使用动态查询
        # ... (BM25召回) ...
        # ... (late_interaction_rerank, 但传入 dynamic_query_emb) ...
        
        # --- 更新上下文 ---
        # 简单的实现：将本次检索到的Top-1文档的平均向量融入上下文
        # final_results, _ = self.late_interaction_rerank(...)
        # if final_results:
        #     top_doc_pid = final_results[0][0]
        #     top_doc_avg_emb = self.doc_token_embeddings[top_doc_pid].mean(dim=0)
        #     decay = self.config.context_memory_decay
        #     state.context_vector = decay * state.context_vector + (1-decay) * top_doc_avg_emb
        
        # 注意：为保证能直接运行，stateful_retrieve暂时先调用普通retrieve
        # 真正的状态化逻辑需要根据上述注释进行详细实现和调试
        logger.info("状态化检索V2调用，当前版本简化为执行非状态化检索。")
        results = self.retrieve(query)
        # 模拟状态更新
        if results:
             top_doc_pid = results[0][0]
             top_doc_avg_emb = self.doc_token_embeddings[top_doc_pid].mean(dim=0)
             decay = self.config.context_memory_decay
             state.context_vector = decay * state.context_vector + (1-decay) * top_doc_avg_emb
        
        return results, state