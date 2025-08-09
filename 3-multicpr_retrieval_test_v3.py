# multicpr_retrieval_test_v3.py

import torch
import numpy as np
from typing import List, Tuple, Dict, Any
import time
import json
import logging
import sys
import os
import pandas as pd
from collections import defaultdict

# 导入V3引擎！
from advanced_zipper_engine_v3 import AdvancedZipperQueryEngineV3, ZipperV3Config, ZipperV3State

# --- 配置参数 ---
MAX_DOCS = 5000   # 限制总文档数，加快测试
DOMAIN = 'video'   # 测试域: 'ecom', 'medical', 'video'
MAX_QUERIES = 100 # 可以适当增加测试查询数，以获得更稳定的结果
K_VALUES = [1, 3, 5, 10]  # 评估的K值

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('multicpr_retrieval_test_v3.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# --- 数据加载函数 ---
def load_multicpr_data(data_dir: str, domain: str = 'ecom', max_docs: int = None) -> Dict[str, Any]:
    logger.info(f"正在加载Multi-CPR {domain}域数据...")
    corpus_file = os.path.join(data_dir, domain, 'corpus.tsv')
    dev_query_file = os.path.join(data_dir, domain, 'dev.query.txt')
    qrels_file = os.path.join(data_dir, domain, 'qrels.dev.tsv')
    
    # 检查文件是否存在
    for file_path in [corpus_file, dev_query_file, qrels_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 预加载所有相关PID
    required_pids = set()
    df_qrels_full = pd.read_csv(qrels_file, sep='\t', header=None, names=['qid', 'iter', 'pid', 'relevance'])
    qrels_pids = set(df_qrels_full[df_qrels_full['relevance'] == 1]['pid'].unique())
    logger.info(f"需要优先加载的相关PID数量: {len(qrels_pids)}")
    
    # 使用更稳健的逐行解析方式读取 TSV（仅按第一个制表符分割，容忍内容中的引号/制表符）
    malformed_corpus_lines = 0
    total_corpus_lines = 0
    required_rows: List[Tuple[int, str]] = []
    other_rows: List[Tuple[int, str]] = []

    def parse_two_column_tsv(filepath: str):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                yield line.rstrip('\n')

    for line in parse_two_column_tsv(corpus_file):
        total_corpus_lines += 1
        if not line:
            continue
        if '\t' not in line:
            malformed_corpus_lines += 1
            continue
        pid_str, content = line.split('\t', 1)
        try:
            pid = int(pid_str)
        except ValueError:
            malformed_corpus_lines += 1
            continue
        if pid in qrels_pids:
            required_rows.append((pid, content))
        else:
            other_rows.append((pid, content))

    if max_docs and len(required_rows) < max_docs:
        needed = max_docs - len(required_rows)
        df_final_rows = required_rows + other_rows[:max(0, needed)]
    elif max_docs:
        df_final_rows = required_rows[:max_docs]
    else:
        df_final_rows = required_rows + other_rows

    corpus = {pid: content for pid, content in df_final_rows}

    # 读取查询，同样采用稳健解析
    malformed_query_lines = 0
    total_query_lines = 0
    queries: Dict[int, str] = {}
    for line in parse_two_column_tsv(dev_query_file):
        total_query_lines += 1
        if not line or '\t' not in line:
            malformed_query_lines += 1
            continue
        qid_str, query_text = line.split('\t', 1)
        try:
            qid = int(qid_str)
        except ValueError:
            malformed_query_lines += 1
            continue
        queries[qid] = query_text
    
    # 构建 qrels 字典
    qrels = defaultdict(list)
    for _, row in df_qrels_full[df_qrels_full['relevance'] == 1].iterrows():
        qrels[row['qid']].append(row['pid'])
    
    logger.info(f"数据加载完成，语料库: {len(corpus)}（原始行数: {total_corpus_lines}，异常行: {malformed_corpus_lines}），查询: {len(queries)}（原始行数: {total_query_lines}，异常行: {malformed_query_lines}），标注: {len(qrels)}")
    return {'corpus': corpus, 'queries': queries, 'qrels': dict(qrels), 'domain': domain}

# --- 评估指标计算 ---
def calculate_metrics(expected: List[int], retrieved: List[int], k_values: List[int]) -> Dict[str, float]:
    metrics = {}
    for k in k_values:
        retrieved_k = retrieved[:k]
        hit = len(set(expected) & set(retrieved_k)) > 0
        metrics[f'hit_rate_at_{k}'] = 1.0 if hit else 0.0
        metrics[f'recall_at_{k}'] = len(set(expected) & set(retrieved_k)) / len(expected) if expected else 0.0
        
        rr = 0.0
        for i, doc_id in enumerate(retrieved_k):
            if doc_id in expected:
                rr = 1.0 / (i + 1)
                break
        metrics[f'mrr_at_{k}'] = rr
    return metrics

# --- V3 核心评估逻辑 ---
def evaluate_retrieval_performance_v3(engine: AdvancedZipperQueryEngineV3, dataset: Dict[str, Any]):
    config = engine.config
    logger.info("=" * 25 + f" 开始评估V3引擎在Multi-CPR上的性能 " + "=" * 25)
    
    engine.build_document_index(dataset['corpus'])
    
    all_results = []
    # 筛选出有标注的查询进行测试
    queries_to_test = {k: v for k, v in dataset['queries'].items() if k in dataset['qrels']}
    
    # 模拟会话状态
    session_state = ZipperV3State(original_query="", context_vector=torch.zeros(config.embedding_dim, device=device))

    test_count = 0
    for qid, query in queries_to_test.items():
        if test_count >= MAX_QUERIES:
            logger.info(f"已达到最大查询数限制 ({MAX_QUERIES})，停止测试。")
            break
        
        expected_pids = dataset['qrels'][qid]
        logger.info(f"--- ({test_count+1}/{MAX_QUERIES}) QID:{qid} -> {query[:50]}...")
        
        results = engine.retrieve(query, state=session_state)
        session_state = engine.update_state(session_state, results) # 更新状态
        
        retrieved_pids = [pid for pid, score, content in results]
        
        result_entry = {
            'qid': qid,
            'query': query,
            'expected_pids': expected_pids,
            'retrieved_pids': retrieved_pids,
            'metrics': calculate_metrics(expected_pids, retrieved_pids, K_VALUES)
        }
        all_results.append(result_entry)
        test_count += 1
    
    # 计算总体统计
    overall_stats = {'total_queries': len(all_results), 'metrics': {}}
    if not all_results:
        logger.warning("没有可评估的结果。")
        return overall_stats
        
    for k in K_VALUES:
        for metric_name in [f'mrr_at_{k}', f'hit_rate_at_{k}', f'recall_at_{k}']:
            avg_metric_name = f'average_{metric_name}'
            overall_stats['metrics'][avg_metric_name] = np.mean([r['metrics'][metric_name] for r in all_results])
            
    print_evaluation_results(overall_stats, config)
    return overall_stats

# --- 结果展示 ---
def print_evaluation_results(overall_stats: Dict[str, Any], config: ZipperV3Config):
    if not overall_stats.get('total_queries'): return
    logger.info("\n" + "="*30 + " V3引擎 Multi-CPR 测试结果 " + "="*30)
    logger.info(f"\n📊 总体性能 (查询数: {overall_stats['total_queries']}):")
    logger.info(f"   - 策略: {'Hybrid' if config.use_hybrid_search else 'ColBERT only'}, {'Multi-Head' if config.use_multi_head else 'Single-Head'}, {'Stateful' if config.use_stateful_reranking else 'Stateless'}")
    
    logger.info("\n📈 各K值平均性能:")
    header = f"   {'K':<5}{'MRR':<10}{'Hit Rate':<12}{'Recall':<10}"
    logger.info(header + "\n" + "   " + "-"*len(header))
    for k in K_VALUES:
        mrr = overall_stats['metrics'].get(f'average_mrr_at_{k}', 0)
        hit_rate = overall_stats['metrics'].get(f'average_hit_rate_at_{k}', 0)
        recall = overall_stats['metrics'].get(f'average_recall_at_{k}', 0)
        logger.info(f"   {k:<5}{mrr:<10.3f}{hit_rate:<12.1%}{recall:<10.1%}")
    logger.info("="*80)

# --- 主函数 ---
def main():
    logger.info("="*80)
    logger.info("=== V3引擎在Multi-CPR上的最终性能评估脚本 ===")
    
    # --- 在这里配置你的实验策略 ---
    # 这是一套推荐的、在之前测试中表现出色的“全功能”配置
    config = ZipperV3Config(
        bge_model_path="models--BAAI--bge-large-zh-v1.5/snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116", # 确保使用强大的large模型
        embedding_dim=1024,
        use_hybrid_search=True,
        bm25_weight=1.0,
        colbert_weight=1.5,      # 给予语义模型更高的权重
        use_multi_head=True,
        use_length_penalty=True,
        use_stateful_reranking=True, # 尽管在单轮评估中作用有限，但我们保持架构完整性
        context_influence=0.3,
        # --- 全量预编码与AMP自适应 ---
        precompute_doc_tokens=False,           # False=按需编码, True=全量预编码
        enable_amp_if_beneficial=True          # 自动根据显卡选择是否启用 AMP 及其精度
    )
    
    engine = AdvancedZipperQueryEngineV3(config)
    
    data_dir = "Multi-CPR/data"
    dataset = load_multicpr_data(data_dir, DOMAIN, max_docs=MAX_DOCS)

    evaluate_retrieval_performance_v3(engine, dataset)

if __name__ == "__main__":
    main()