#!/usr/bin/env python
"""
快速测试脚本，用于评估LM Studio本地提供的三个embedding模型
"""

import os
import sys
from pathlib import Path
from src.embedding_selector import EmbeddingSelector
from src.test import evaluate_model, display_results_table, save_results

def main():
    print("== Embedding模型评估 ==")
    print("正在评估LM Studio本地模型...")
    
    # 检查LM Studio服务是否运行
    try:
        selector = EmbeddingSelector()
        # 简单测试
        selector.get_embedding("测试")
        print("✓ 成功连接到LM Studio服务")
    except Exception as e:
        print(f"错误: 无法连接到LM Studio服务。请确保LM Studio已启动并且模型已加载。")
        print(f"详细错误: {str(e)}")
        sys.exit(1)
    
    # 要评估的LM Studio本地模型列表
    models_to_evaluate = [
        "text-embedding-gte-large-zh",
        "text-embedding-bge-large-zh-v1.5",
        "text-embedding-m3e-base"
    ]
    
    # 保存所有结果
    all_results = []
    
    # 评估每个模型
    for model in models_to_evaluate:
        try:
            result = evaluate_model(model)
            all_results.append(result)
        except Exception as e:
            print(f"评估模型 {model} 时出错: {str(e)}")
            all_results.append({
                "model_name": model,
                "error": str(e)
            })
    
    # 显示结果表格
    display_results_table(all_results)
    
    # 保存详细结果
    save_results(all_results)
    
    print("\n评估已完成！详细结果已保存到 results.json 文件")

if __name__ == "__main__":
    main() 