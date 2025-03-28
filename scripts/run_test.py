#!/usr/bin/env python
"""
快速测试脚本，用于评估LM Studio本地提供的embedding模型
"""

import os
import sys
import argparse
from pathlib import Path
import json

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.core.embedding_selector import EmbeddingSelector
from src.models.model_configs import list_available_models, AVAILABLE_MODELS
from src.tests.test import evaluate_model, display_results_table, save_results

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='评估embedding模型')
    parser.add_argument('--output', '-o', 
                       type=str, 
                       default=os.path.join(project_root, 'results', 'results.json'),
                       help='输出结果的JSON文件路径 (默认: results/results.json)')
    parser.add_argument('--models', '-m',
                      type=str,
                      nargs='+',
                      help='要评估的模型列表 (空格分隔的模型名称)')
    parser.add_argument('--list', '-l',
                      action='store_true',
                      help='列出所有可用的模型')
    parser.add_argument('--provider', '-p',
                      type=str,
                      choices=['openai', 'local', 'custom'],
                      help='仅列出指定提供商的模型')
    args = parser.parse_args()
    
    # 如果指定了--list参数，列出所有模型并退出
    if args.list:
        provider = args.provider if args.provider else None
        models = list_available_models(provider)
        print("\n可用的embedding模型:")
        print("------------------------------------------------")
        print(f"{'模型名称':<30} {'维度':<10} {'提供商':<10} {'描述':<40}")
        print("------------------------------------------------")
        
        current_provider = None
        for name, desc, dims, provider in models:
            if current_provider is not None and current_provider != provider:
                print("------------------------------------------------")
                print(f"\n== {provider.upper()} 提供商模型 ==")
                print("------------------------------------------------")
            elif current_provider is None:
                print(f"\n== {provider.upper()} 提供商模型 ==")
                print("------------------------------------------------")
                
            print(f"{name:<30} {dims:<10} {provider:<10} {desc:<40}")
            current_provider = provider
            
        print("------------------------------------------------")
        sys.exit(0)
    
    # 如果没有指定模型，使用默认的本地模型列表
    models_to_evaluate = args.models if args.models else [
        "text-embedding-gte-large-zh",
        "text-embedding-bge-large-zh-v1.5",
        "text-embedding-m3e-base",
        "text-embedding-granite-embedding-278m-multilingual"
    ]
    
    # 验证模型是否存在
    invalid_models = [m for m in models_to_evaluate if m not in AVAILABLE_MODELS]
    if invalid_models:
        print(f"错误: 以下模型不存在: {', '.join(invalid_models)}")
        print("使用 --list 或 -l 查看所有可用模型")
        sys.exit(1)
    
    print("== Embedding模型评估 ==")
    print(f"将评估以下模型: {', '.join(models_to_evaluate)}")
    
    # 检查输出目录是否存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")
    
    # 检查LM Studio服务是否运行
    try:
        # 测试第一个模型的连接
        selector = EmbeddingSelector(models_to_evaluate[0])
        # 简单测试
        selector.get_embedding("测试")
        print("✓ 成功连接到API服务")
    except Exception as e:
        print(f"错误: 无法连接到API服务。")
        print(f"详细错误: {str(e)}")
        sys.exit(1)
    
    # 保存所有结果
    all_results = []
    
    # 评估每个模型
    for i, model in enumerate(models_to_evaluate):
        print(f"\n[{i+1}/{len(models_to_evaluate)}] 评估模型: {model}")
        try:
            result = evaluate_model(model)
            all_results.append(result)
            print(f"✓ {model} 评估完成")
        except Exception as e:
            print(f"✗ 评估模型 {model} 时出错: {str(e)}")
            all_results.append({
                "model_name": model,
                "error": str(e)
            })
    
    # 显示结果表格
    display_results_table(all_results)
    
    # 保存详细结果
    save_results(all_results, args.output)
    
    print(f"\n评估已完成！详细结果已保存到 {args.output} 文件")

if __name__ == "__main__":
    main() 