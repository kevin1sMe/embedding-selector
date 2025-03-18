import os
import openai
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 加载环境变量
load_dotenv()

class EmbeddingSelector:
    """
    用于选择和使用不同的embedding模型的类
    支持切换不同的模型和API端点
    """
    
    # 预设可用的模型列表
    AVAILABLE_MODELS = {
        # OpenAI官方模型
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "description": "OpenAI第二代Ada嵌入模型"
        },
        "text-embedding-3-small": {
            "dimensions": 1536,
            "description": "OpenAI第三代小型嵌入模型"
        },
        "text-embedding-3-large": {
            "dimensions": 3072,
            "description": "OpenAI第三代大型嵌入模型"
        },
        
        # LM Studio本地模型
        "text-embedding-gte-large-zh": {
            "dimensions": 1024,
            "description": "GTE大型中文嵌入模型（本地）"
        },
        "text-embedding-bge-large-zh-v1.5": {
            "dimensions": 1024, 
            "description": "百度开源的中英双语大型嵌入模型（本地）"
        },
        "text-embedding-m3e-base": {
            "dimensions": 768,
            "description": "M3E基础嵌入模型（本地）"
        },
        
        # 其他可能通过自定义API端点使用的模型
        "m3e-large": {
            "dimensions": 1024,
            "description": "Moka开源的中英双语嵌入模型"
        }
    }
    
    def __init__(self, model_name="text-embedding-gte-large-zh", api_base=None, api_key=None):
        """
        初始化embedding选择器
        
        Args:
            model_name: 要使用的模型名称
            api_base: API端点URL，如果为None则使用环境变量中的配置
            api_key: API密钥，如果为None则使用环境变量中的配置
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("未设置API密钥，请在.env文件中设置OPENAI_API_KEY或在初始化时提供")
        
        # 设置API端点
        self.api_base = api_base
        if not self.api_base:
            self.api_base = os.getenv("CUSTOM_API_BASE") or os.getenv("DEFAULT_API_BASE")
            
        # 设置模型
        self.set_model(model_name)
        
        # 配置OpenAI客户端
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
    
    def set_model(self, model_name):
        """设置要使用的embedding模型"""
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"不支持的模型: {model_name}。可用模型: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_name = model_name
        self.dimensions = self.AVAILABLE_MODELS[model_name]["dimensions"]
        print(f"已设置模型为: {model_name} (维度: {self.dimensions})")
    
    def set_api_endpoint(self, api_base):
        """设置API端点"""
        self.api_base = api_base
        # 更新客户端配置
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
        print(f"已设置API端点为: {api_base}")
    
    def get_embedding(self, text):
        """
        获取文本的embedding向量
        
        Args:
            text: 输入文本
            
        Returns:
            embedding向量的numpy数组
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def get_batch_embeddings(self, texts):
        """
        批量获取文本的embedding向量
        
        Args:
            texts: 文本列表
            
        Returns:
            embedding向量列表的numpy数组
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return np.array([data.embedding for data in response.data])
    
    def calculate_similarity(self, embedding1, embedding2):
        """计算两个embedding之间的余弦相似度"""
        return cosine_similarity([embedding1], [embedding2])[0][0]
    
    def find_most_similar(self, query_embedding, candidate_embeddings, candidate_texts):
        """
        找出与查询embedding最相似的候选文本
        
        Args:
            query_embedding: 查询文本的embedding向量
            candidate_embeddings: 候选文本的embedding向量列表
            candidate_texts: 候选文本列表
            
        Returns:
            按相似度排序的(文本, 相似度)元组列表
        """
        similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
        
        # 将相似度与文本配对并排序
        result_pairs = list(zip(candidate_texts, similarities))
        return sorted(result_pairs, key=lambda x: x[1], reverse=True) 