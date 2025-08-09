#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def analyze_qrels_file():
    """分析qrels.dev.tsv文件的结构"""
    qrels_file = "Multi-CPR/data/ecom/qrels.dev.tsv"
    
    try:
        with open(qrels_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"文件总行数: {len(lines)}")
        
        # 分析前几行
        print("\n前5行内容:")
        for i, line in enumerate(lines[:5]):
            parts = line.strip().split('\t')
            print(f"第{i+1}行: {parts}")
        
        # 统计唯一的文档ID数量
        unique_pids = set()
        unique_qids = set()
        
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                qid = int(parts[0])
                pid = int(parts[2])
                relevance = int(parts[3])
                
                unique_qids.add(qid)
                if relevance == 1:  # 只统计相关文档
                    unique_pids.add(pid)
        
        print(f"\n统计结果:")
        print(f"唯一查询ID数量: {len(unique_qids)}")
        print(f"唯一相关文档ID数量: {len(unique_pids)}")
        print(f"总标注行数: {len(lines)}")
        
        # 显示一些示例
        if unique_qids:
            print(f"查询ID范围: {min(unique_qids)} - {max(unique_qids)}")
        if unique_pids:
            print(f"文档ID范围: {min(unique_pids)} - {max(unique_pids)}")
        
        # 分析每个查询的相关文档数量
        qid_to_pids = {}
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                qid = int(parts[0])
                pid = int(parts[2])
                relevance = int(parts[3])
                
                if relevance == 1:
                    if qid not in qid_to_pids:
                        qid_to_pids[qid] = []
                    qid_to_pids[qid].append(pid)
        
        if qid_to_pids:
            print(f"\n每个查询的相关文档数量统计:")
            doc_counts = [len(pids) for pids in qid_to_pids.values()]
            print(f"平均每个查询的相关文档数: {sum(doc_counts) / len(doc_counts):.2f}")
            print(f"最少相关文档数: {min(doc_counts)}")
            print(f"最多相关文档数: {max(doc_counts)}")
        
        print("\n分析完成!")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")

if __name__ == "__main__":
    analyze_qrels_file() 