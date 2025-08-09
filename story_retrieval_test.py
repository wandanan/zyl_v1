# story_retrieval_test.py

import torch
import numpy as np
from typing import List, Tuple, Dict, Any
import time
import json
import logging
import sys

# 导入主引擎
# 假设 advanced_zipper_engine.py 在同一目录下
from advanced_zipper_engine import AdvancedZipperQueryEngine, AdvancedZipperConfig

# --- 日志配置 ---
# 配置日志，同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('story_retrieval_test.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# --- 辅助函数 ---

def load_story_dataset(json_file_path: str) -> Dict[str, Any]:
    """加载故事数据集"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"成功加载数据集: {json_file_path}")
        logger.info(f"包含 {dataset['metadata']['total_stories']} 个故事，{dataset['metadata']['total_questions']} 个问题")
        return dataset
    except FileNotFoundError:
        logger.error(f"错误：数据集文件未找到 -> {json_file_path}")
        logger.error("请确保 story_qa_dataset.json 文件与本脚本在同一目录下。")
        sys.exit(1)
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        raise

def calculate_mrr_at_k(expected: List[int], retrieved: List[int], k: int) -> float:
    """计算 Mean Reciprocal Rank @ K"""
    retrieved_k = retrieved[:k]
    for i, doc_id in enumerate(retrieved_k):
        if doc_id in expected:
            return 1.0 / (i + 1)
    return 0.0

def calculate_hit_rate_at_k(expected: List[int], retrieved: List[int], k: int) -> float:
    """计算 Hit Rate @ K"""
    return 1.0 if len(set(expected) & set(retrieved[:k])) > 0 else 0.0


# --- 核心评估逻辑 ---

def evaluate_retrieval_performance(engine: AdvancedZipperQueryEngine, dataset: Dict[str, Any], k: int) -> Dict[str, Any]:
    """评估检索性能"""
    logger.info("=" * 25 + " 开始评估检索性能 " + "=" * 25)
    
    all_results = []
    total_questions = 0
    
    # 按难度和类型统计
    difficulty_stats = {"easy": [], "medium": [], "hard": []}
    type_stats = {"factual": [], "reasoning": [], "summary": []}
    
    for story in dataset['stories']:
        story_id = story['story_id']
        story_title = story['title']
        story_content = story['content']
        
        logger.info(f"\n--- 处理故事: {story_title} ({story_id}) ---")
        
        # 【重要】为每个故事重新构建索引，避免数据污染
        engine.build_document_index(story_content)
        
        # 测试每个问题
        for question_data in story['questions']:
            total_questions += 1
            question_id = question_data['question_id']
            question = question_data['question']
            expected_fragments = question_data['expected_fragments']
            answer = question_data['answer']
            difficulty = question_data['difficulty']
            question_type = question_data['type']
            
            logger.info(f"  -> 测试问题: {question} (ID: {question_id})")
            
            # 执行检索
            start_time = time.time()
            results = engine.zipper_retrieve(question, top_k=k)
            retrieval_time = time.time() - start_time
            
            # 提取检索到的片段ID
            retrieved_fragments = [doc_id for doc_id, score, doc in results]
            
            # 计算评估指标
            hit_rate = calculate_hit_rate_at_k(expected_fragments, retrieved_fragments, k)
            mrr = calculate_mrr_at_k(expected_fragments, retrieved_fragments, k)
            
            # 记录结果
            result = {
                'story_id': story_id,
                'question': question,
                'expected_fragments': expected_fragments,
                'retrieved_fragments': retrieved_fragments,
                'hit_rate_at_k': hit_rate,
                'mrr_at_k': mrr,
                'retrieval_time': retrieval_time,
                'difficulty': difficulty,
                'type': question_type,
            }
            all_results.append(result)
            
            # 按难度和类型分组统计
            difficulty_stats[difficulty].append(result)
            type_stats[question_type].append(result)
            
            logger.info(f"     期望片段: {expected_fragments}")
            logger.info(f"     检索片段: {retrieved_fragments}")
            logger.info(f"     Hit@{k}: {'✅' if hit_rate > 0 else '❌'}")
            logger.info(f"     MRR@{k}: {mrr:.3f}")
            logger.info(f"     耗时: {retrieval_time:.3f}s")
    
    # 计算总体统计
    overall_stats = {
        'total_questions': total_questions,
        'average_mrr_at_k': sum(r['mrr_at_k'] for r in all_results) / len(all_results) if all_results else 0,
        'overall_hit_rate_at_k': sum(r['hit_rate_at_k'] for r in all_results) / len(all_results) if all_results else 0,
        'average_retrieval_time': sum(r['retrieval_time'] for r in all_results) / len(all_results) if all_results else 0,
        'k': k
    }
    
    # 分组统计函数
    def analyze_group(results_group):
        if not results_group: return None
        return {
            'count': len(results_group),
            'average_mrr_at_k': sum(r['mrr_at_k'] for r in results_group) / len(results_group),
            'hit_rate_at_k': sum(r['hit_rate_at_k'] for r in results_group) / len(results_group)
        }

    return {
        'overall_stats': overall_stats,
        'difficulty_analysis': {k: analyze_group(v) for k, v in difficulty_stats.items()},
        'type_analysis': {k: analyze_group(v) for k, v in type_stats.items()},
        'all_results': all_results
    }


# --- 结果展示与保存 ---

def print_evaluation_results(eval_results: Dict[str, Any]):
    """打印格式化的评估结果"""
    overall = eval_results['overall_stats']
    k = overall['k']
    
    logger.info("\n" + "="*30 + " 检索测试结果总结 " + "="*30)
    
    # 总体情况
    logger.info("\n📊 总体性能:")
    logger.info(f"   - 总问题数: {overall['total_questions']}")
    logger.info(f"   - 平均倒数排名 (MRR@{k}): {overall['average_mrr_at_k']:.3f}")
    logger.info(f"   - 总体命中率 (HitRate@{k}): {overall['overall_hit_rate_at_k']:.1%}")
    logger.info(f"   - 平均检索时间: {overall['average_retrieval_time']:.3f} 秒/查询")
    
    # 按问题类型分析
    logger.info("\n📝 按问题类型分析:")
    for q_type, stats in eval_results['type_analysis'].items():
        if stats:
            logger.info(f"   - {q_type.capitalize()} ({stats['count']}个): MRR@{k}={stats['average_mrr_at_k']:.3f}, HitRate@{k}={stats['hit_rate_at_k']:.1%}")
    
    # 按难度分析
    logger.info("\n📈 按难度分析:")
    for difficulty, stats in eval_results['difficulty_analysis'].items():
        if stats:
            logger.info(f"   - {difficulty.capitalize()} ({stats['count']}个): MRR@{k}={stats['average_mrr_at_k']:.3f}, HitRate@{k}={stats['hit_rate_at_k']:.1%}")

    logger.info("="*80)

def save_results(eval_results: Dict[str, Any], config: AdvancedZipperConfig, output_file: str):
    """保存测试结果到JSON文件"""
    logger.info(f"正在保存详细测试结果到: {output_file}")
    
    save_data = {
        'model_info': {
            'engine': 'AdvancedZipperQueryEngine',
            'config': config.__dict__ 
        },
        'evaluation_summary': eval_results['overall_stats'],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'details': {
            'difficulty_analysis': eval_results['difficulty_analysis'],
            'type_analysis': eval_results['type_analysis'],
            'individual_results': eval_results['all_results']
        }
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已成功保存。")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")


# --- 主函数 ---

def main():
    """主函数，执行整个测试流程"""
    logger.info("="*80)
    logger.info("=== 状态化流式拉链检索器 - 故事QA测试脚本 ===")
    
    # --- 系统与环境检查 ---
    logger.info("\n--- 环境检查 ---")
    logger.info(f"Python 版本: {sys.version.split()[0]}")
    logger.info(f"PyTorch 版本: {torch.__version__}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # --- 配置与初始化 ---
    logger.info("\n--- 初始化引擎 ---")
    
    # 【重要】使用与新引擎匹配的配置
    config = AdvancedZipperConfig(
        num_heads=8,
        sparse_top_n=50,  # BM25召回50个候选，可以调整
        context_memory_decay=0.8, # 记忆衰减率
        use_bge_embedding=True
    )
    # top_k 将在评估函数中指定，这里无需配置
    
    engine = AdvancedZipperQueryEngine(config)
    
    # --- 数据加载 ---
    logger.info("\n--- 加载数据集 ---")
    dataset = load_story_dataset('story_qa_dataset.json')
    
    # --- 性能评估 ---
    k_for_evaluation = 3 # 我们关心Top-3的结果
    evaluation_results = evaluate_retrieval_performance(engine, dataset, k=k_for_evaluation)
    
    # --- 结果展示与保存 ---
    print_evaluation_results(evaluation_results)
    save_results(evaluation_results, config, "retrieval_results.json")
    
    logger.info("\n" + "="*80)
    logger.info("=== 测试完成 ===")
    logger.info("="*80)

if __name__ == "__main__":
    main()