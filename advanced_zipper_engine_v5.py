# advanced_zipper_engine_v5.py

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import time
import logging
from dataclasses import dataclass, field
import os

# 依赖项检查与导入
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


# --- 数据类定义 ---
@dataclass
class AdvancedZipperConfig:
    """V5动态自适应引擎配置"""
    # 【路径修复】使用您指定的本地快照路径
    bge_model_path: str = "models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620"
    
    # 检索参数
    top_k: int = 3
    sparse_top_n: int = 100
    
    # 【V5核心】定义两种权重策略
    # 策略1: 用于有明确意图的复杂问题
    intent_fusion_weights: Tuple[float, float, float] = (0.7, 0.1, 0.2)
    # 策略2: 用于没有意图的简单事实问题
    default_fusion_weights: Tuple[float, float, float] = (0.95, 0.05, 0.0)

@dataclass
class ZipperState:
    """V5引擎的状态，只存储向量，不做复杂结构"""
    original_query_embedding: torch.Tensor
    enhanced_query_embedding: torch.Tensor
    context_vector: torch.Tensor


# --- 模型组件 ---
class BGEEmbeddingModel:
    """BGE模型封装，确保模型只加载一次"""
    def __init__(self, model_path: str, use_fp16: bool = True):
        self.model = self._load_model(model_path, use_fp16)

    def _load_model(self, model_path, use_fp16):
        try:
            if not os.path.exists(model_path):
                 logger.error(f"指定的本地模型路径不存在: {model_path}")
                 logger.error("请确认模型文件已放置在正确位置，或者修改AdvancedZipperConfig中的路径。")
                 raise FileNotFoundError(f"BGE模型路径未找到: {model_path}")

            logger.info(f"正在从本地路径加载BGE模型: {model_path}")
            # 使用FlagModel加载模型
            model = FlagModel(
                model_path,
                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                use_fp16=use_fp16 if device.type == 'cuda' else False
            )
            logger.info("BGE模型加载成功")
            return model
        except Exception as e:
            logger.error(f"BGE模型加载失败: {e}")
            raise

    def encode(self, texts: List[str], is_query=False) -> torch.Tensor:
        """对文本进行编码"""
        if is_query:
            embeddings = self.model.encode_queries(texts)
        else:
            embeddings = self.model.encode(texts)
        return torch.tensor(embeddings, dtype=torch.float32).to(device)


# --- 主引擎 (V5 动态自适应版) ---
class AdvancedZipperQueryEngine:
    """
    V5动态自适应拉链查询引擎。
    核心思想是根据查询的类型（简单事实vs复杂意图）动态调整评分策略，
    以求在两类问题上都达到最佳性能。
    """
    def __init__(self, config: AdvancedZipperConfig):
        self.config = config
        self.bge_model = BGEEmbeddingModel(config.bge_model_path)
        
        # 文档存储与索引
        self.documents: List[str] = []
        self.doc_embeddings: Optional[torch.Tensor] = None
        self.bm25_index: Optional[BM25Okapi] = None
        
        # 意图理解关键词库
        self.intent_keywords = {
            "如何": "方法 步骤", "怎样": "方法 步骤", "为什么": "原因 理由",
            "看法": "观点 态度", "观点": "看法 态度", "道理": "启示 感悟"
        }
        logger.info("V5动态自适应引擎初始化完成: [动态权重融合]")

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """简单的中文分词器，用于BM25"""
        return list(text)
    
    def _get_enhanced_query_text(self, query: str) -> Tuple[str, bool]:
        """
        分析查询，如果包含意图关键词，则返回增强后的查询文本，并标记意图被找到。
        返回: (增强后的文本, 是否找到意图的布尔值)
        """
        for keyword, expansion_terms in self.intent_keywords.items():
            if keyword in query:
                # 找到了意图，返回增强后的文本和True
                return query + " " + expansion_terms, True
        # 未找到意图，返回原始文本和False
        return query, False

    def build_document_index(self, documents: List[str]):
        """构建BM25稀疏索引和BGE稠密索引"""
        if not documents:
            logger.warning("没有提供文档，索引构建跳过。")
            return
            
        logger.info(f"构建双重索引，共 {len(documents)} 个片段...")
        start_time = time.time()
        self.documents = documents
        
        # 1. 构建BM25稀疏索引
        tokenized_corpus = [self._tokenize_for_bm25(doc) for doc in self.documents]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # 2. 构建BGE稠密索引，并进行L2归一化以方便计算余弦相似度
        raw_embeddings = self.bge_model.encode(self.documents)
        self.doc_embeddings = F.normalize(raw_embeddings, p=2, dim=1)
        
        logger.info(f"双重索引构建完成，耗时: {time.time() - start_time:.3f}秒")

    def zipper_retrieve(self, query: str, top_k: int = None) -> List[Tuple[int, float, str]]:
        """
        执行V5版的动态自适应流式检索。
        """
        if top_k is None: top_k = self.config.top_k
        if not self.documents: return []

        # --- 阶段一: 稀疏召回 ---
        tokenized_query = self._tokenize_for_bm25(query)
        candidate_scores = self.bm25_index.get_scores(tokenized_query)
        num_candidates = min(len(self.documents), self.config.sparse_top_n)
        candidate_ids = np.argsort(candidate_scores)[::-1][:num_candidates]

        # --- 阶段二: 状态化流式精排 (V5) ---
        
        # 1. 动态选择权重策略
        enhanced_query_text, intent_found = self._get_enhanced_query_text(query)
        if intent_found:
            w_orig, w_ctx, w_intent = self.config.intent_fusion_weights
            logger.debug(f"检测到意图，使用意图权重: {self.config.intent_fusion_weights}")
        else:
            w_orig, w_ctx, w_intent = self.config.default_fusion_weights
            logger.debug(f"未检测到意图，使用默认权重: {self.config.default_fusion_weights}")

        # 2. 初始化状态，计算并归一化所有需要的查询向量
        embeddings = self.bge_model.encode([query, enhanced_query_text], is_query=True)
        state = ZipperState(
            original_query_embedding=F.normalize(embeddings[0:1], p=2, dim=1),
            enhanced_query_embedding=F.normalize(embeddings[1:2], p=2, dim=1),
            context_vector=torch.zeros_like(embeddings[0:1]).to(device),
        )

        scores_and_ids = []
        
        # 3. 在候选集上进行流式扫描
        with torch.no_grad():
            for doc_id in candidate_ids:
                current_block_embedding = self.doc_embeddings[doc_id:doc_id+1]
                
                # a. 【V5核心逻辑】在分数层面安全地融合信号
                score_orig = (state.original_query_embedding @ current_block_embedding.T).item()
                score_ctx = (state.context_vector @ current_block_embedding.T).item()
                score_intent = (state.enhanced_query_embedding @ current_block_embedding.T).item()
                
                # 加权融合得到最终分数
                final_score = w_orig * score_orig + w_ctx * score_ctx + w_intent * score_intent
                scores_and_ids.append((final_score, doc_id))

                # b. 【V5记忆更新】用原始文档嵌入安全地更新记忆
                # 策略：用当前最高分片段的向量作为下一个上下文
                if len(scores_and_ids) == 1 or final_score > max(s[0] for s in scores_and_ids[:-1]):
                    state.context_vector = current_block_embedding

        # 4. 整理最终结果
        scores_and_ids.sort(key=lambda x: x[0], reverse=True)
        final_results = []
        for score, doc_id in scores_and_ids[:top_k]:
            final_results.append((int(doc_id), score, self.documents[doc_id]))
        
        return final_results

    def get_cache_stats(self) -> Dict:
        """为兼容测试脚本提供一个空的统计函数"""
        return {'total_documents': len(self.documents), 'info': 'V5动态自适应引擎'}