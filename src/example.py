#!/usr/bin/env python
"""
Embedding选择器示例用法
展示如何使用EmbeddingSelector来切换不同的模型和端点
"""

from embedding_selector import EmbeddingSelector
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def main():
    print("Embedding选择器示例")
    print("==================")
    
    # 创建一个使用默认模型的选择器
    print("\n1. 使用默认模型 (text-embedding-gte-large-zh)")
    selector = EmbeddingSelector()
    
    # 获取一些文本的embedding
    texts = [
        "这是一个测试文本",
        "这是另一个相似的测试文本",
        "这个文本与前两个不太相似"
    ]
    
    # 批量获取embedding
    print("为文本生成embedding...")
    embeddings = selector.get_batch_embeddings(texts)
    
    # 计算相似度
    print("\n计算相似度:")
    print(f"文本1和文本2的相似度: {selector.calculate_similarity(embeddings[0], embeddings[1]):.4f}")
    print(f"文本1和文本3的相似度: {selector.calculate_similarity(embeddings[0], embeddings[2]):.4f}")
    print(f"文本2和文本3的相似度: {selector.calculate_similarity(embeddings[1], embeddings[2]):.4f}")
    
    # 切换到另一个模型
    print("\n2. 切换到另一个模型 (text-embedding-bge-large-zh-v1.5)")
    selector.set_model("text-embedding-bge-large-zh-v1.5")
    
    # 再次获取embedding并计算相似度
    print("为文本生成embedding...")
    embeddings = selector.get_batch_embeddings(texts)
    
    print("\n计算相似度:")
    print(f"文本1和文本2的相似度: {selector.calculate_similarity(embeddings[0], embeddings[1]):.4f}")
    print(f"文本1和文本3的相似度: {selector.calculate_similarity(embeddings[0], embeddings[2]):.4f}")
    print(f"文本2和文本3的相似度: {selector.calculate_similarity(embeddings[1], embeddings[2]):.4f}")
    
    # 切换到第三个模型
    print("\n3. 切换到第三个模型 (text-embedding-m3e-base)")
    selector.set_model("text-embedding-m3e-base")
    
    # 再次获取embedding并计算相似度
    print("为文本生成embedding...")
    embeddings = selector.get_batch_embeddings(texts)
    
    print("\n计算相似度:")
    print(f"文本1和文本2的相似度: {selector.calculate_similarity(embeddings[0], embeddings[1]):.4f}")
    print(f"文本1和文本3的相似度: {selector.calculate_similarity(embeddings[0], embeddings[2]):.4f}")
    print(f"文本2和文本3的相似度: {selector.calculate_similarity(embeddings[1], embeddings[2]):.4f}")
    
    # 相似度搜索示例
    print("\n4. 相似度搜索示例")
    # 使用一个查询文本
    query = "如何优化性能"
    print(f"查询: '{query}'")
    
    # 候选文本列表
    candidates = [
        "优化代码执行性能",
        "修复登录页面的bug",
        "添加新的用户界面",
        "提高数据库查询性能",
        "解决UI渲染慢的问题"
    ]
    print("候选文本:")
    for i, text in enumerate(candidates):
        print(f"{i+1}. {text}")
    
    # 获取查询和候选文本的embedding
    query_embedding = selector.get_embedding(query)
    candidate_embeddings = selector.get_batch_embeddings(candidates)
    
    # 找到最相似的候选文本
    similar_texts = selector.find_most_similar(
        query_embedding, 
        candidate_embeddings, 
        candidates
    )
    
    # 打印结果
    print("\n最相似的文本:")
    for i, (text, score) in enumerate(similar_texts):
        print(f"{i+1}. {text} (相似度: {score:.4f})")

if __name__ == "__main__":
    main() 