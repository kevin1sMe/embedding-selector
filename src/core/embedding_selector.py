import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

from src.models.model_configs import get_model_info, get_provider_config, AVAILABLE_MODELS
from src.utils.api_utils import create_openai_client, batched_api_call

class EmbeddingSelector:
    """
    用于选择和使用不同的embedding模型的类
    支持切换不同的模型和API端点
    """
    
    def __init__(self, model_name, api_base=None, api_key=None):
        """
        初始化embedding选择器
        
        Args:
            model_name: 要使用的模型名称
            api_base: API端点URL，如果为None则使用环境变量中的配置
            api_key: API密钥，如果为None则使用环境变量中的配置
            
        Raises:
            ValueError: 当模型名称无效或API配置缺失时抛出
        """
        # 设置模型
        self.set_model(model_name)
        
        # 获取模型对应的提供商配置
        model_info = get_model_info(model_name)
        provider = model_info["provider"]
        provider_config = get_provider_config(provider)
        
        # 设置API密钥
        self.api_key = api_key or provider_config["api_key"]
        if not self.api_key:
            raise ValueError(f"未设置API密钥，请在初始化时提供或确保环境变量已正确配置")
        
        # 设置API端点
        self.api_base = api_base or provider_config["base_url"]
        if not self.api_base:
            raise ValueError(f"未设置API端点，请在初始化时提供或确保环境变量已正确配置")
            
        # 配置OpenAI客户端
        self.client = create_openai_client(self.api_key, self.api_base)
    
    def set_model(self, model_name):
        """设置要使用的embedding模型"""
        model_info = get_model_info(model_name)
        
        self.model_name = model_name
        # 为兼容性起见，使用模型的真实名称，而不是键
        self.model_api_name = model_info["name"]
        self.dimensions = model_info["dimensions"]
        self.batch_size = model_info["batch_size"]
        print(f"已设置模型为: {model_name} (维度: {self.dimensions})")
    
    def set_api_endpoint(self, api_base):
        """设置API端点"""
        self.api_base = api_base
        # 更新客户端配置
        self.client = create_openai_client(self.api_key, self.api_base)
        print(f"已设置API端点为: {api_base}")
    
    def get_embedding(self, text):
        """
        获取文本的embedding向量
        
        Args:
            text: 输入文本
            
        Returns:
            embedding向量的numpy数组
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_api_name,
                input=text
            )
            
            if response and response.data and len(response.data) > 0 and response.data[0].embedding:
                return np.array(response.data[0].embedding)
            else:
                print(f"警告: 模型 {self.model_name} 返回了空的embedding")
                # 返回一个全零向量作为替代
                return np.zeros(self.dimensions)
        except Exception as e:
            print(f"获取embedding时出错: {str(e)}")
            # 返回一个全零向量作为替代
            return np.zeros(self.dimensions)
    
    def get_batch_embeddings(self, texts):
        """
        批量获取文本的embedding向量
        
        Args:
            texts: 文本列表
            
        Returns:
            embedding向量列表的numpy数组
        """
        try:
            # 使用通用批处理函数
            embeddings = batched_api_call(
                client=self.client,
                model_name=self.model_api_name,
                items=texts,
                batch_size=self.batch_size
            )
            
            # 转换为numpy数组
            return np.array(embeddings)
        except Exception as e:
            print(f"批量获取embeddings时出错: {str(e)}")
            # 返回全零向量矩阵作为替代
            return np.zeros((len(texts), self.dimensions))
    
    def calculate_similarity(self, embedding1, embedding2):
        """计算两个embedding向量之间的余弦相似度"""
        return cosine_similarity([embedding1], [embedding2])[0][0]
    
    def find_most_similar(self, query_embedding, candidate_embeddings, candidate_texts=None, top_k=5):
        """
        找到与查询embedding最相似的候选embeddings
        
        Args:
            query_embedding: 查询embedding向量
            candidate_embeddings: 候选embedding向量列表
            candidate_texts: 候选文本列表，可选
            top_k: 返回的最相似项目数量
            
        Returns:
            如果提供了candidate_texts，则返回[{"text": text, "score": score}]的列表
            否则返回[(index, score)]的列表
        """
        # 确保查询embedding是二维数组
        query_embedding_2d = query_embedding.reshape(1, -1)
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_embedding_2d, candidate_embeddings)[0]
        
        # 获取top_k个最相似的索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        if candidate_texts:
            # 如果提供了文本，返回文本和分数
            results = [
                {"text": candidate_texts[idx], "score": float(similarities[idx])}
                for idx in top_indices
            ]
        else:
            # 否则返回索引和分数
            results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results 