# Multi-CPR 检索测试指南

本目录包含了使用Multi-CPR数据集进行检索性能测试的脚本。

## 数据集介绍

Multi-CPR是一个多领域中文文档检索数据集，包含三个领域：
- **E-commerce (ecom)**: 电商领域，约100万文档
- **Medical**: 医疗领域，约100万文档  
- **Video**: 视频娱乐领域，约100万文档

每个领域包含：
- `corpus.tsv`: 文档集合 (pid, content)
- `dev.query.txt`: 开发集查询 (qid, query)
- `qrels.dev.tsv`: 相关性标注 (qid, 0, pid, relevance)

## 测试脚本

### 1. 数据加载测试
```bash
python test_multicpr_data_loading.py
```
验证Multi-CPR数据集是否正确加载。

### 2. 单域测试
```bash
# 使用默认参数
python multicpr_retrieval_test.py

# 自定义参数
python multicpr_retrieval_test.py --max_docs 500 --domain medical --max_queries 50 --k_values "1,3,5"
```
测试单个域的检索性能。

### 3. 多域测试
```bash
# 使用默认参数
python multicpr_all_domains_test.py

# 自定义参数
python multicpr_all_domains_test.py --max_docs 500 --max_queries 50 --domains "ecom,medical"
```
测试多个域的检索性能，并生成对比结果。

### 4. 快速测试
```bash
# 使用默认参数
python quick_multicpr_test.py

# 自定义参数
python quick_multicpr_test.py --max_docs 200 --max_queries 5 --domain video
```
快速测试脚本，用于验证功能。

## 评估指标

测试脚本会计算以下指标：
- **MRR@K**: Mean Reciprocal Rank at K
- **HitRate@K**: Hit Rate at K  
- **Recall@K**: Recall at K
- **Precision@K**: Precision at K

默认测试K值：1, 3, 5, 10

## 命令行参数

所有测试脚本都支持以下命令行参数：

### 单域测试 (multicpr_retrieval_test.py)
- `--max_docs`: 最大测试文档数 (默认: 1000)
- `--domain`: 测试域，可选 ecom/medical/video (默认: ecom)
- `--max_queries`: 最大测试查询数 (默认: 100)
- `--k_values`: 评估的K值，用逗号分隔 (默认: "1,3,5,10")

### 多域测试 (multicpr_all_domains_test.py)
- `--max_docs`: 每个域最大测试文档数 (默认: 1000)
- `--max_queries`: 每个域最大测试查询数 (默认: 100)
- `--k_values`: 评估的K值，用逗号分隔 (默认: "1,3,5,10")
- `--domains`: 测试的域，用逗号分隔 (默认: "ecom,medical,video")

### 快速测试 (quick_multicpr_test.py)
- `--max_docs`: 最大测试文档数 (默认: 500)
- `--max_queries`: 最大测试查询数 (默认: 10)
- `--domain`: 测试域，可选 ecom/medical/video (默认: ecom)
- `--k_values`: 评估的K值，用逗号分隔 (默认: "1,3,5")

## 引擎配置参数

在脚本中可以调整以下引擎参数：

```python
config = AdvancedZipperConfig(
    num_heads=8,                    # 注意力头数
    sparse_top_n=50,                # BM25召回候选数
    context_memory_decay=0.8,       # 记忆衰减率
    use_bge_embedding=True          # 使用BGE嵌入
)
```

## 输出文件

测试完成后会生成：
- `multicpr_retrieval_test.log`: 单域测试日志
- `multicpr_all_domains_test.log`: 多域测试日志
- `multicpr_ecom_retrieval_results.json`: ecom域测试结果
- `multicpr_all_domains_retrieval_results.json`: 多域测试结果

## 结果示例

```
Multi-CPR ecom域检索测试结果
==============================

📊 总体性能 (ecom域):
   - 总查询数: 100
   - 平均检索时间: 0.045 秒/查询

📈 各K值性能:
   - K= 1: MRR=0.123, HitRate=12.3%, Recall=12.3%, Precision=12.3%
   - K= 3: MRR=0.156, HitRate=18.7%, Recall=15.6%, Precision=5.2%
   - K= 5: MRR=0.167, HitRate=22.1%, Recall=16.7%, Precision=3.3%
   - K=10: MRR=0.178, HitRate=25.4%, Recall=17.8%, Precision=1.8%
```

## 性能基准

根据Multi-CPR论文，BERT-base DPR模型在各域的MRR@10性能：
- E-commerce: 0.2704
- Medical: 0.3270  
- Video: 0.2537

## 注意事项

1. **内存使用**: 医疗域数据较大(334MB)，脚本已限制只加载前1000个文档以减少内存使用
2. **测试时间**: 完整测试可能需要较长时间，建议先用少量查询测试
3. **GPU加速**: 如果有GPU，会自动使用GPU加速嵌入计算
4. **编码问题**: 数据文件使用UTF-8编码，确保系统支持中文显示
5. **文档限制**: 默认只使用前1000个文档进行测试，可通过修改`max_docs`参数调整

## 故障排除

### 常见问题

1. **文件不存在错误**
   - 确保Multi-CPR数据目录结构正确
   - 检查文件路径和权限

2. **内存不足**
   - 减少`max_queries_per_domain`参数
   - 使用更小的`sparse_top_n`值

3. **编码错误**
   - 确保Python环境支持UTF-8
   - 检查文件编码格式

4. **模型加载失败**
   - 确保BGE模型已正确下载
   - 检查网络连接和模型路径

## 扩展功能

可以根据需要扩展测试脚本：
- 添加更多评估指标（如NDCG@K）
- 支持自定义查询和文档
- 添加模型对比功能
- 生成可视化图表 