# story_retrieval_test.py

import torch
import numpy as np
from typing import List, Tuple, Dict, Any
import time
import json
import logging

# 【确认导入V5】
from advanced_zipper_engine_v5 import AdvancedZipperQueryEngine, AdvancedZipperConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO, # INFO只看关键步骤, DEBUG可看权重选择等细节
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('story_retrieval_test.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 检查CUDA可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU型号: {torch.cuda.get_device_name()}")

def load_story_dataset(json_file_path: str) -> Dict[str, Any]:
    """加载故事数据集"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"成功加载数据集: {json_file_path}")
        logger.info(f"包含 {len(dataset['stories'])} 个故事")
        return dataset
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        raise

def calculate_mrr_and_hit_rate(expected: List[int], retrieved: List[int], k: int) -> Tuple[float, float]:
    """计算MRR@k和HitRate@k"""
    if not expected or not retrieved:
        return 0.0, 0.0
    
    mrr_score, hit = 0.0, 0.0
    retrieved_at_k = retrieved[:k]
    
    for i, doc_id in enumerate(retrieved_at_k):
        if doc_id in expected:
            mrr_score = 1.0 / (i + 1)
            break
            
    if len(set(expected) & set(retrieved_at_k)) > 0:
        hit = 1.0
        
    return mrr_score, hit


def evaluate_retrieval_performance(engine: AdvancedZipperQueryEngine, dataset: Dict[str, Any], k_val: int) -> Dict[str, Any]:
    """评估检索性能"""
    logger.info("开始评估检索性能...")
    all_results = []
    
    for story in dataset['stories']:
        story_id, story_title, story_content = story['story_id'], story['title'], story['content']
        logger.info(f"\n--- 处理故事: {story_title} ({story_id}) ---")
        engine.build_document_index(story_content)
        
        for question_data in story['questions']:
            question, expected_fragments = question_data['question'], question_data['expected_fragments']
            logger.info(f"  -> 测试问题: {question}")
            
            start_time = time.time()
            results = engine.zipper_retrieve(question, top_k=k_val)
            retrieval_time = time.time() - start_time
            
            retrieved_fragments = [doc_id for doc_id, score, doc in results]
            
            mrr_score, hit_rate = calculate_mrr_and_hit_rate(expected_fragments, retrieved_fragments, k=k_val)
            
            result = {
                'story_id': story_id, 'story_title': story_title, 'question': question,
                'expected_fragments': expected_fragments, 'retrieved_fragments': retrieved_fragments,
                'mrr_score': mrr_score, 'hit_rate': hit_rate, 'retrieval_time': retrieval_time,
                'difficulty': question_data['difficulty'], 'type': question_data['type'],
            }
            
            all_results.append(result)
            
            logger.info(f"     期望片段: {expected_fragments}")
            logger.info(f"     检索片段: {retrieved_fragments}")
            logger.info(f"     Hit@{k_val}: {'✅' if hit_rate > 0 else '❌'}")
            logger.info(f"     MRR@{k_val}: {mrr_score:.3f}")
            logger.info(f"     耗时: {retrieval_time:.3f}s")
            
    return {'all_results': all_results}

def analyze_and_print_results(evaluation_results: Dict[str, Any], k_val: int):
    """分析并打印最终的统计结果"""
    all_results = evaluation_results['all_results']
    if not all_results:
        logger.warning("没有可供分析的结果。")
        return

    total_questions = len(all_results)
    avg_mrr = sum(r['mrr_score'] for r in all_results) / total_questions
    avg_hit_rate = sum(r['hit_rate'] for r in all_results) / total_questions
    avg_time = sum(r['retrieval_time'] for r in all_results) / total_questions

    type_stats, difficulty_stats = {}, {}
    for r in all_results:
        q_type, q_diff = r['type'], r['difficulty']
        if q_type not in type_stats: type_stats[q_type] = []
        if q_diff not in difficulty_stats: difficulty_stats[q_diff] = []
        type_stats[q_type].append(r)
        difficulty_stats[q_diff].append(r)

    logger.info("\n" + "="*80)
    logger.info(f"=== 检索测试结果总结 (k={k_val}) ===")
    logger.info("="*80)
    
    logger.info(f"\n📊 总体性能:\n   - 总问题数: {total_questions}\n   - 平均倒数排名 (MRR@{k_val}): {avg_mrr:.3f}\n   - 总体命中率 (HitRate@{k_val}): {avg_hit_rate:.1%}\n   - 平均检索时间: {avg_time:.3f} 秒/查询")
    
    logger.info("\n📝 按问题类型分析:")
    for q_type, results in sorted(type_stats.items()):
        count = len(results)
        mrr = sum(r['mrr_score'] for r in results) / count if count > 0 else 0
        hit = sum(r['hit_rate'] for r in results) / count if count > 0 else 0
        logger.info(f"   - {q_type.capitalize():<10} ({count:>2}个): MRR@{k_val}={mrr:.3f}, HitRate@{k_val}={hit:.1%}")

    logger.info("\n📈 按难度分析:")
    for q_diff, results in sorted(difficulty_stats.items()):
        count = len(results)
        mrr = sum(r['mrr_score'] for r in results) / count if count > 0 else 0
        hit = sum(r['hit_rate'] for r in results) / count if count > 0 else 0
        logger.info(f"   - {q_diff.capitalize():<10} ({count:>2}个): MRR@{k_val}={mrr:.3f}, HitRate@{k_val}={hit:.1%}")
    
    logger.info("="*80)

def save_results(evaluation_results: Dict[str, Any], output_file: str = "retrieval_results.json"):
    """保存测试结果"""
    logger.info(f"\n正在保存详细测试结果到: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已成功保存。")
    except Exception as e:
        logger.error(f"保存结果失败: {e}", exc_info=True)

def main():
    """主函数"""
    logger.info("="*80 + "\n=== V5 动态自适应引擎测试开始 ===\n" + "="*80)
    
    # 加载数据集
    dataset = load_story_dataset('story_qa_dataset.json')
    
    # 创建V5配置
    config = AdvancedZipperConfig()
    
    # 初始化引擎
    engine = AdvancedZipperQueryEngine(config)
    
    # 设定评估参数
    k_value_for_eval = 3
    
    # 执行评估
    evaluation_results = evaluate_retrieval_performance(engine, dataset, k_val=k_value_for_eval)
    
    # 分析并打印结果
    analyze_and_print_results(evaluation_results, k_val=k_value_for_eval)
    
    # 保存结果
    save_results(evaluation_results)
    
    logger.info("\n" + "="*80 + "\n=== 测试完成 ===\n" + "="*80)

if __name__ == "__main__":
    main()