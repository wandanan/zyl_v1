# multicpr_retrieval_test_v2.py

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

# 导入V2引擎！
from advanced_zipper_engine_v2 import AdvancedZipperQueryEngineV2, ZipperV2Config

# --- 配置参数 ---
MAX_DOCS = 5000   # 限制总文档数，加快测试
DOMAIN = 'ecom'   # 测试域: 'ecom', 'medical', 'video'
MAX_QUERIES = 50  # 减少测试查询数，加快测试
K_VALUES = [1, 3, 5, 10]  # 评估的K值

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('multicpr_retrieval_test_v2.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# --- 数据加载函数 (与之前版本相同) ---
def load_multicpr_data(data_dir: str, domain: str = 'ecom', max_docs: int = None) -> Dict[str, Any]:
    logger.info(f"正在加载Multi-CPR {domain}域数据...")
    corpus_file = os.path.join(data_dir, domain, 'corpus.tsv')
    dev_query_file = os.path.join(data_dir, domain, 'dev.query.txt')
    qrels_file = os.path.join(data_dir, domain, 'qrels.dev.tsv')
    
    for file_path in [corpus_file, dev_query_file, qrels_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
    
    required_pids = set()
    if qrels_file and os.path.exists(qrels_file):
        with open(qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4 and int(parts[3]) == 1:
                    required_pids.add(int(parts[2]))
    
    df_full = pd.read_csv(corpus_file, sep='\t', header=None, names=['pid', 'content'], dtype={'pid': int, 'content': str}, engine='python')
    df_required = df_full[df_full['pid'].isin(required_pids)]
    df_other = df_full[~df_full['pid'].isin(required_pids)]
    if max_docs:
        num_other_needed = max_docs - len(df_required)
        if num_other_needed > 0:
            df_final = pd.concat([df_required, df_other.head(num_other_needed)], ignore_index=True)
        else:
            df_final = df_required
    else:
        df_final = df_full
    corpus = dict(zip(df_final['pid'], df_final['content']))

    df_queries = pd.read_csv(dev_query_file, sep='\t', header=None, names=['qid', 'query'], dtype={'qid': int, 'query': str})
    queries = dict(zip(df_queries['qid'], df_queries['query']))
    
    df_qrels = pd.read_csv(qrels_file, sep='\t', header=None, names=['qid', 'iter', 'pid', 'relevance'], dtype={'qid': int, 'iter': int, 'pid': int, 'relevance': int})
    df_qrels = df_qrels[df_qrels['relevance'] == 1]
    qrels = defaultdict(list)
    for _, row in df_qrels.iterrows():
        qrels[row['qid']].append(row['pid'])
    
    logger.info(f"数据加载完成，语料库: {len(corpus)}，查询: {len(queries)}，标注: {len(dict(qrels))}")
    return {'corpus': corpus, 'queries': queries, 'qrels': dict(qrels), 'domain': domain}

# --- 评估指标计算 (与之前版本相同) ---
def calculate_mrr_at_k(expected: List[int], retrieved: List[int], k: int) -> float:
    retrieved_k = retrieved[:k]
    for i, doc_id in enumerate(retrieved_k):
        if doc_id in expected:
            return 1.0 / (i + 1)
    return 0.0

def calculate_hit_rate_at_k(expected: List[int], retrieved: List[int], k: int) -> float:
    return 1.0 if len(set(expected) & set(retrieved[:k])) > 0 else 0.0

def calculate_recall_at_k(expected: List[int], retrieved: List[int], k: int) -> float:
    if not expected: return 0.0
    retrieved_k = retrieved[:k]
    intersection = set(expected) & set(retrieved_k)
    return len(intersection) / len(expected)

def calculate_precision_at_k(expected: List[int], retrieved: List[int], k: int) -> float:
    if k == 0: return 0.0
    retrieved_k = retrieved[:k]
    intersection = set(expected) & set(retrieved_k)
    return len(intersection) / k

# --- V2 核心评估逻辑 (已修复并简化) ---
def evaluate_retrieval_performance_v2(engine: AdvancedZipperQueryEngineV2, dataset: Dict[str, Any], k_values: List[int], max_queries: int):
    logger.info("=" * 25 + " 开始评估V2引擎性能 " + "=" * 25)
    queries, qrels, corpus = dataset['queries'], dataset['qrels'], dataset['corpus']

    # V2直接使用PID进行索引，不再需要复杂的ID映射
    engine.build_document_index(corpus)
    
    all_results = []
    total_queries = 0
    queries_to_test = {k: v for k, v in queries.items() if k in qrels}

    for qid, query in queries_to_test.items():
        total_queries += 1
        if max_queries and total_queries > max_queries:
            logger.info(f"已达到最大查询数限制 ({max_queries})，停止测试")
            break
            
        expected_pids = qrels[qid]
        logger.info("-" * 80)
        logger.info(f"({total_queries}/{len(queries_to_test)}) -> 测试查询 {qid}: {query}")
        logger.info(f"     期望文档PID: {expected_pids}")
        
        start_time = time.time()
        results = engine.retrieve(query)
        retrieval_time = time.time() - start_time
        
        # V2引擎返回的doc_id已经是PID，直接使用即可
        retrieved_pids = [int(doc_id) for doc_id, score, doc_content in results]
        
        logger.info(f"     [V2日志] 引擎返回PID列表: {retrieved_pids}")
        logger.info(f"     [V2日志] 对应分数: {[f'{score:.4f}' for _, score, _ in results]}")

        # 检查是否命中
        is_hit = False
        for rank, pid in enumerate(retrieved_pids):
            if pid in expected_pids:
                logger.info(f"     ✅✅✅ 命中! Rank {rank+1}, PID: {pid}")
                is_hit = True
        if not is_hit:
             logger.info("     ❌ MISS")
        
        result = {
            'qid': qid,
            'query': query,
            'expected_pids': expected_pids,
            'retrieved_pids': retrieved_pids,
            'retrieval_time': retrieval_time,
            'metrics': {}
        }
        
        for k in k_values:
            result['metrics'][f'mrr_at_{k}'] = calculate_mrr_at_k(expected_pids, retrieved_pids, k)
            result['metrics'][f'hit_rate_at_{k}'] = calculate_hit_rate_at_k(expected_pids, retrieved_pids, k)
            result['metrics'][f'recall_at_{k}'] = calculate_recall_at_k(expected_pids, retrieved_pids, k)
            result['metrics'][f'precision_at_{k}'] = calculate_precision_at_k(expected_pids, retrieved_pids, k)
        
        all_results.append(result)
    
    # 计算总体统计
    overall_stats = {'total_queries': total_queries, 'domain': dataset['domain'], 'k_values': k_values, 'metrics': {}}
    if not all_results:
        return {'overall_stats': overall_stats, 'all_results': []}

    for k in k_values:
        overall_stats['metrics'][f'average_mrr_at_{k}'] = np.mean([r['metrics'][f'mrr_at_{k}'] for r in all_results])
        overall_stats['metrics'][f'average_hit_rate_at_{k}'] = np.mean([r['metrics'][f'hit_rate_at_{k}'] for r in all_results])
        overall_stats['metrics'][f'average_recall_at_{k}'] = np.mean([r['metrics'][f'recall_at_{k}'] for r in all_results])
        overall_stats['metrics'][f'average_precision_at_{k}'] = np.mean([r['metrics'][f'precision_at_{k}'] for r in all_results])
    
    overall_stats['average_retrieval_time'] = np.mean([r['retrieval_time'] for r in all_results])
    return {'overall_stats': overall_stats, 'all_results': all_results}

# --- 结果展示与保存 (与之前版本相同) ---
def print_evaluation_results(eval_results: Dict[str, Any]):
    if not eval_results or not eval_results['overall_stats'].get('total_queries'):
        logger.error("评估结果为空，无法打印。")
        return

    overall = eval_results['overall_stats']
    logger.info("\n" + "="*30 + f" Multi-CPR {overall['domain']}域检索测试结果 (V2引擎) " + "="*30)
    logger.info(f"\n📊 总体性能:")
    logger.info(f"   - 测试查询总数: {overall['total_queries']}")
    logger.info(f"   - 平均检索时间: {overall['average_retrieval_time']:.3f} 秒/查询")
    
    logger.info("\n📈 各K值平均性能:")
    header = f"   {'K':<5}{'MRR':<10}{'Hit Rate':<12}{'Recall':<10}{'Precision':<10}"
    logger.info(header)
    logger.info("   " + "-"*len(header))
    for k in overall['k_values']:
        mrr = overall['metrics'].get(f'average_mrr_at_{k}', 0)
        hit_rate = overall['metrics'].get(f'average_hit_rate_at_{k}', 0)
        recall = overall['metrics'].get(f'average_recall_at_{k}', 0)
        precision = overall['metrics'].get(f'average_precision_at_{k}', 0)
        row = f"   {k:<5}{mrr:<10.3f}{hit_rate:<12.1%}{recall:<10.1%}{precision:<10.1%}"
        logger.info(row)
    logger.info("="*80)

def save_results(eval_results: Dict[str, Any], config: ZipperV2Config, output_file: str):
    logger.info(f"正在保存详细测试结果到: {output_file}")
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    save_data = {'model_info': {'engine': 'AdvancedZipperQueryEngineV2', 'config': config.__dict__}, 'evaluation_summary': eval_results['overall_stats'], 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'details': eval_results['all_results']}
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        logger.info(f"结果已成功保存。")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")

# --- 主函数 ---
def main():
    logger.info("="*80)
    logger.info("=== V2引擎最终评估脚本 ===")
    
    config = ZipperV2Config()
    engine = AdvancedZipperQueryEngineV2(config)
    
    data_dir = "Multi-CPR/data"
    dataset = load_multicpr_data(data_dir, DOMAIN, max_docs=MAX_DOCS)

    evaluation_results = evaluate_retrieval_performance_v2(engine, dataset, k_values=K_VALUES, max_queries=MAX_QUERIES)
    
    print_evaluation_results(evaluation_results)
    save_results(evaluation_results, config, f"multicpr_{DOMAIN}_retrieval_results_v2.json")
    
    logger.info("\n" + "="*80)
    logger.info("=== V2测试完成 ===")
    logger.info("="*80)

if __name__ == "__main__":
    main()