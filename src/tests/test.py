#!/usr/bin/env python
"""
Embedding模型评估脚本
用于评估不同embedding模型在中文和中英文混合查询场景下的性能
"""

import os
import json
import time
from dotenv import load_dotenv
import numpy as np
from tabulate import tabulate

# 修复导入路径
from src.core.embedding_selector import EmbeddingSelector
from src.tests.test_data import COMMIT_MESSAGES, TEST_QUERIES

# 加载环境变量
load_dotenv()

def evaluate_model(model_name, api_base=None):
    """
    评估指定模型的性能
    
    Args:
        model_name: 模型名称
        api_base: API端点URL，如果为None则使用环境变量中的配置
        
    Returns:
        包含评估结果的字典
    """
    print(f"\n正在评估模型: {model_name}")
    try:
        # 初始化模型
        selector = EmbeddingSelector(model_name=model_name, api_base=api_base)
        
        # 记录开始时间
        start_time = time.time()
        
        # 为所有commit messages生成embedding
        # 这里不需要传递max_samples，get_batch_embeddings方法已经实现了分批处理
        messages_embeddings = selector.get_batch_embeddings(COMMIT_MESSAGES)
        
        # 评估查询结果
        total_top1_hits = 0
        total_top3_hits = 0
        total_top5_hits = 0
        
        results = []
        
        for query in TEST_QUERIES:
            # 获取查询的embedding
            query_embedding = selector.get_embedding(query)
            
            # 找到最相似的commit messages
            similar_messages = selector.find_most_similar(
                query_embedding, 
                messages_embeddings, 
                COMMIT_MESSAGES
            )
            
            # 记录Top-N结果
            top_results = []
            for i, item in enumerate(similar_messages[:5]):
                # 适配新的find_most_similar返回格式
                if isinstance(item, dict):
                    message = item["text"]
                    score = item["score"]
                else:
                    # 旧格式兼容
                    message, score = item
                
                top_results.append({
                    "rank": i+1,
                    "message": message,
                    "score": float(score)
                })
                
                # 简单的相关性判断（基于关键词匹配）
                # 在实际应用中，你可能需要更复杂的相关性判断方法
                is_relevant = False
                query_keywords = set(query.lower().split())
                message_keywords = set(message.lower().split())
                
                # 检查是否有关键词重叠
                if len(query_keywords.intersection(message_keywords)) > 0:
                    is_relevant = True
                
                if is_relevant:
                    if i == 0:  # Top 1
                        total_top1_hits += 1
                    if i < 3:   # Top 3
                        total_top3_hits += 1
                    if i < 5:   # Top 5
                        total_top5_hits += 1
            
            results.append({
                "query": query,
                "top_results": top_results
            })
        
        # 计算指标
        elapsed_time = time.time() - start_time
        precision_at_1 = total_top1_hits / len(TEST_QUERIES)
        precision_at_3 = total_top3_hits / (len(TEST_QUERIES) * 3)
        precision_at_5 = total_top5_hits / (len(TEST_QUERIES) * 5)
        
        evaluation_result = {
            "model_name": model_name,
            "precision@1": precision_at_1,
            "precision@3": precision_at_3,
            "precision@5": precision_at_5,
            "processing_time": elapsed_time,
            "query_results": results,
            "data_size": len(COMMIT_MESSAGES)
        }
        
        return evaluation_result
    
    except Exception as e:
        print(f"评估模型 {model_name} 时出错: {str(e)}")
        return {
            "model_name": model_name,
            "error": str(e)
        }

def save_results(results, output_file="results.json"):
    """保存评估结果到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到 {output_file}")

def display_results_table(all_results):
    """以表格形式显示评估结果"""
    table_data = []
    headers = ["模型", "Precision@1", "Precision@3", "Precision@5", "处理时间(秒)", "数据量"]
    
    for result in all_results:
        if "error" in result:
            row = [
                result["model_name"],
                "错误",
                "错误",
                "错误",
                "错误",
                "N/A"
            ]
        else:
            row = [
                result["model_name"],
                f"{result['precision@1']:.4f}",
                f"{result['precision@3']:.4f}",
                f"{result['precision@5']:.4f}",
                f"{result['processing_time']:.2f}",
                result.get("data_size", "全部")
            ]
        table_data.append(row)
    
    print("\n模型评估结果:")
    print(tabulate(table_data, headers=headers, tablefmt="grid")) 