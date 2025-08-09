# advanced_zipper_engine.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import math
import time
import logging
from dataclasses import dataclass, field
from queue import PriorityQueue

# 尝试导入依赖，并在失败时提供明确的安装指导
try:
    from FlagEmbedding import FlagModel
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False
    print("警告: FlagEmbedding未安装，将跳过BGE编码功能。如需完整功能请运行: pip install FlagEmbedding")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25未安装。请运行: pip install rank_bm25")

# --- 日志和设备配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- 数据类定义 ---

@dataclass
class AdvancedZipperConfig:
    use_bge_embedding: bool = True
    bge_model_path: str = "models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620"
    bge_embedding_dim: int = 512
    num_heads: int = 8
    # hidden_size将由引擎自动设为bge_embedding_dim
    hidden_size: int = 512
    top_k: int = 10
    sparse_top_n: int = 100
    context_memory_decay: float = 0.8

@dataclass
class ZipperState:
    original_query: str
    query_embedding: torch.Tensor
    context_vector: torch.Tensor
    top_k_candidates: PriorityQueue = field(default_factory=PriorityQueue)
    visited_blocks: Set[int] = field(default_factory=set)


# --- 模型组件 ---

class BGEEmbeddingModel:
    def __init__(self, model_path: str, use_fp16: bool = True):
        self.model = self._load_model(model_path, use_fp16)

    def _load_model(self, model_path, use_fp16):
        try:
            logger.info(f"正在加载BGE模型: {model_path}")
            
            # 检查是否是本地路径
            import os
            if os.path.exists(model_path):
                logger.info(f"使用本地模型路径: {model_path}")
            else:
                logger.warning(f"本地路径不存在: {model_path}，尝试从Hugging Face下载")
            
            model = FlagModel(
                model_path,
                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                use_fp16=use_fp16 if device.type == 'cuda' else False
            )
            logger.info("BGE模型加载成功")
            return model
        except Exception as e:
            logger.error(f"BGE模型加载失败: {e}")
            logger.error("请确保模型路径正确，或者检查网络连接和Hugging Face访问权限")
            raise

    def encode_query(self, query: str) -> torch.Tensor:
        embedding = self.model.encode_queries([query])
        return torch.tensor(embedding, dtype=torch.float32).to(device)

    def encode_documents(self, documents: List[str]) -> torch.Tensor:
        """批量编码文档 - 优化版本"""
        if not documents:
            return torch.empty(0, self.model.embedding_dim, device=device)
        
        logger.info(f"开始批量编码 {len(documents)} 个文档...")
        start_time = time.time()
        
        # 批处理编码
        batch_size = 64  # 增大批处理大小以提高效率
        all_embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_docs)
            all_embeddings.append(batch_embeddings)
            
            # 显示进度
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"  已编码 {i + len(batch_docs)}/{len(documents)} 个文档")
        
        # 合并所有批次的结果
        embeddings = np.vstack(all_embeddings)
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        
        elapsed_time = time.time() - start_time
        logger.info(f"文档编码完成，耗时: {elapsed_time:.3f}秒，平均: {elapsed_time/len(documents)*1000:.1f}ms/文档")
        
        return embeddings_tensor

class StatefulZipperAttention(nn.Module):
    def __init__(self, config: AdvancedZipperConfig):
        super().__init__()
        self.config = config
        # 确保使用正确的 hidden_size
        hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        
        # 【修复】确保 head_dim 计算正确
        if hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) 必须能被 num_heads ({self.num_heads}) 整除")
        self.head_dim = hidden_size // self.num_heads

        self.query_projection = nn.Linear(hidden_size, hidden_size).to(device)
        self.key_projection = nn.Linear(hidden_size, hidden_size).to(device)
        self.value_projection = nn.Linear(hidden_size, hidden_size).to(device)
        self.output_projection = nn.Linear(hidden_size, hidden_size).to(device)
        
        # 【修复】确保 relevance_scorer 的输入维度正确
        self.relevance_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, 1), nn.Sigmoid()
        ).to(device)

    def forward(self, state: ZipperState, block_embedding: torch.Tensor) -> Tuple[float, torch.Tensor]:
        # 1. 动态生成Query
        adaptive_query = state.query_embedding + state.context_vector * 0.5
        
        # 2. 多头注意力计算
        Q = self.query_projection(adaptive_query).view(1, self.num_heads, self.head_dim)
        K = self.key_projection(block_embedding).view(1, self.num_heads, self.head_dim)
        V = self.value_projection(block_embedding).view(1, self.num_heads, self.head_dim)

        attention_scores = torch.einsum('bhd,bhd->bh', Q, K) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 3. 聚合信息
        weighted_v = torch.einsum('bh,bhd->bhd', attention_weights, V)
        
        # 【关键修复】将多头结果重新拼接成 hidden_size
        context_contribution = weighted_v.reshape(1, self.config.hidden_size)
        
        # 4. 计算相关性分数
        combined_features = torch.cat([adaptive_query, context_contribution], dim=-1)
        relevance_score = self.relevance_scorer(combined_features).squeeze().item()

        # 5. 返回经过输出层投影的贡献向量
        final_contribution = self.output_projection(context_contribution)

        return relevance_score, final_contribution


# --- 主引擎 ---

class AdvancedZipperQueryEngine(nn.Module):
    def __init__(self, config: AdvancedZipperConfig):
        super().__init__()
        # 【关键修复】在传递配置之前，就强制同步维度
        config.hidden_size = config.bge_embedding_dim
        self.config = config

        # 检查BGE模型是否可用
        if config.use_bge_embedding and not FLAG_EMBEDDING_AVAILABLE:
            logger.warning("FlagEmbedding不可用，禁用BGE编码功能")
            config.use_bge_embedding = False
        
        self.bge_model = BGEEmbeddingModel(config.bge_model_path) if config.use_bge_embedding else None
        # 现在传递给Attention的config是维度同步后的
        self.zipper_attention = StatefulZipperAttention(config)

        self.documents: List[str] = []
        self.doc_embeddings: Optional[torch.Tensor] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.sliding_stats = {}
        
        logger.info("状态化流式拉链查询引擎初始化完成")

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        return list(text)

    def build_document_index(self, documents: List[str]):
        if not documents:
            logger.warning("没有提供文档，索引构建跳过。")
            self.documents, self.doc_embeddings, self.bm25_index = [], None, None
            return
            
        logger.info(f"开始构建索引，共 {len(documents)} 个文档片段...")
        start_time = time.time()
        self.documents = documents

        # 构建BM25索引
        logger.info("构建BM25稀疏索引...")
        tokenized_corpus = [self._tokenize_for_bm25(doc) for doc in self.documents]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        logger.info("BM25稀疏索引构建完成。")

        # 构建BGE索引（如果启用）
        if self.bge_model:
            logger.info("构建BGE稠密索引...")
            self.doc_embeddings = self.bge_model.encode_documents(self.documents)
            logger.info("BGE稠密索引构建完成。")
        else:
            logger.info("跳过BGE索引构建（仅使用BM25）")
            self.doc_embeddings = None
        
        logger.info(f"索引构建完成，总耗时: {time.time() - start_time:.3f}秒")

    def zipper_retrieve(self, query: str, top_k: int = None) -> List[Tuple[int, float, str]]:
        if top_k is None: top_k = self.config.top_k
            
        if not self.documents or self.bm25_index is None:
            logger.error("索引未构建，无法执行检索。")
            return []

        logger.info(f"开始检索，查询: '{query[:50]}...'")
        start_time = time.time()

        # BM25稀疏检索
        tokenized_query = self._tokenize_for_bm25(query)
        candidate_scores = self.bm25_index.get_scores(tokenized_query)
        num_candidates = min(len(self.documents), self.config.sparse_top_n)
        candidate_ids = np.argsort(candidate_scores)[::-1][:num_candidates]
        logger.info(f"BM25召回了 {len(candidate_ids)} 个候选片段。")

        # 如果启用了BGE，进行混合检索
        if self.bge_model and self.doc_embeddings is not None:
            logger.info("执行混合检索（BM25 + BGE）...")
            query_embedding = self.bge_model.encode_query(query)
            state = ZipperState(
                original_query=query,
                query_embedding=query_embedding,
                context_vector=torch.zeros_like(query_embedding).to(device),
            )

            with torch.no_grad():
                for doc_id in candidate_ids:
                    if doc_id in state.visited_blocks: continue
                    
                    current_block_embedding = self.doc_embeddings[doc_id].unsqueeze(0)
                    relevance_score, context_contribution = self.zipper_attention(
                        state, current_block_embedding
                    )
                    
                    state.top_k_candidates.put((-relevance_score, doc_id))
                    
                    decay = self.config.context_memory_decay
                    state.context_vector = decay * state.context_vector + (1 - decay) * context_contribution.squeeze(0)
                    state.visited_blocks.add(doc_id)
            
            final_results = []
            while not state.top_k_candidates.empty() and len(final_results) < top_k:
                score, doc_id = state.top_k_candidates.get()
                final_results.append((doc_id, -score, self.documents[doc_id]))
            
            final_results.sort(key=lambda x: x[1], reverse=True)
        else:
            # 仅使用BM25检索
            logger.info("仅使用BM25检索...")
            final_results = []
            for i, doc_id in enumerate(candidate_ids[:top_k]):
                score = candidate_scores[doc_id]
                final_results.append((doc_id, score, self.documents[doc_id]))

        logger.info(f"检索完成，返回 {len(final_results)} 个结果，耗时: {time.time() - start_time:.3f}秒")
        
        return final_results

    def get_cache_stats(self) -> Dict:
        return {'total_documents': len(self.documents), 'info': '本引擎为状态化流式处理。'}