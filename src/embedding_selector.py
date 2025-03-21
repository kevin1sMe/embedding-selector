import os
import openai
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# 加载环境变量
load_dotenv()

class EmbeddingSelector:
    """
    用于选择和使用不同的embedding模型的类
    支持切换不同的模型和API端点
    """
    
    # 提供商配置
    PROVIDER_CONFIGS = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "local": {
            "base_url": os.getenv("LOCAL_API_BASE"), 
            "api_key": os.getenv("LOCAL_API_KEY")
        },
        "custom": {
            "base_url": os.getenv("CUSTOM_API_BASE"),
            "api_key": os.getenv("CUSTOM_API_KEY")
        }
    }
    
    # 预设可用的模型列表
    AVAILABLE_MODELS = {
        # OpenAI官方模型
        "text-embedding-ada-002": {
            "description": "OpenAI第二代Ada嵌入模型",
            "provider": "openai",
            "name": "text-embedding-ada-002",
            "dimensions": 1536,
            "batch_size": 2048
        },
        "text-embedding-3-small": {
            "description": "OpenAI第三代小型嵌入模型",
            "provider": "openai",
            "name": "text-embedding-3-small",
            "dimensions": 1536,
            "batch_size": 2048
        },
        "text-embedding-3-large": {
            "description": "OpenAI第三代大型嵌入模型",
            "provider": "openai",
            "name": "text-embedding-3-large",
            "dimensions": 3072,
            "batch_size": 2048
        },
        
        # LM Studio本地模型
        "text-embedding-gte-large-zh": {
            "description": "GTE大型中文嵌入模型（本地）",
            "provider": "local",
            "name": "text-embedding-gte-large-zh",
            "dimensions": 1024,
            "batch_size": 512
        },
        "text-embedding-bge-large-zh-v1.5": {
            "description": "百度开源的中英双语大型嵌入模型（本地）",
            "provider": "local",
            "name": "text-embedding-bge-large-zh-v1.5",
            "dimensions": 1024,
            "batch_size": 512
        },
        "text-embedding-m3e-base": {
            "description": "M3E基础嵌入模型（本地）",
            "provider": "local",
            "name": "text-embedding-m3e-base",
            "dimensions": 768,
            "batch_size": 512
        },
        
        # 其他可能通过自定义API端点使用的模型
        "m3e-large": {
            "description": "Moka开源的中英双语嵌入模型",
            "provider": "custom",
            "name": "m3e-large",
            "dimensions": 1024,
            "batch_size": 512
        },

        # 新增模型
        "hunyuan": {
            "description": "腾讯混元嵌入模型",
            "provider": "custom",
            "name": "hunyuan-embedding",
            "dimensions": 1024,
            "batch_size": 8  # 调整为8，已测试可行
        },
        "doubao": {
            "description": "豆包嵌入模型",
            "provider": "custom",
            "name": "doubao-embedding-large-text-240915",
            "dimensions": 1024,
            "batch_size":256 
        },
        "baichuan": {
            "description": "百川嵌入模型",
            "provider": "custom",
            "name": "Baichuan-Text-Embedding",
            "dimensions": 1024,
            "batch_size": 16  # 错误信息显示最多16
        },
        "qwen": {
            "description": "通义千问嵌入模型",
            "provider": "custom",
            "name": "text-embedding-v3",
            "dimensions": 1024,
            "batch_size": 10  # 错误信息显示最多10
        },
        "baidu": {
            "description": "百度嵌入模型",
            "provider": "custom",
            "name": "Embedding-V1",
            "dimensions": 1024,
            "batch_size": 8  # 保守估计
        }
    }
    
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
        provider = self.AVAILABLE_MODELS[model_name]["provider"]
        provider_config = self.PROVIDER_CONFIGS[provider]
        
        # 设置API密钥
        self.api_key = api_key or provider_config["api_key"]
        if not self.api_key:
            raise ValueError(f"未设置API密钥，请在初始化时提供或确保环境变量已正确配置")
        
        # 设置API端点
        self.api_base = api_base or provider_config["base_url"]
        if not self.api_base:
            raise ValueError(f"未设置API端点，请在初始化时提供或确保环境变量已正确配置")
            
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
        # 为兼容性起见，使用模型的真实名称，而不是键
        self.model_api_name = self.AVAILABLE_MODELS[model_name]["name"]
        self.dimensions = self.AVAILABLE_MODELS[model_name]["dimensions"]
        self.batch_size = self.AVAILABLE_MODELS[model_name]["batch_size"]
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
        # 不再需要单独处理hunyuan模型，每个模型都使用其配置的batch_size
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        print(f"处理 {len(texts)} 条文本，分为 {total_batches} 批 (每批最多 {self.batch_size} 条)")
        
        # 按照模型支持的批量大小分批处理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            print(f"处理批次 {batch_num}/{total_batches} ({len(batch_texts)} 条文本)...")
            
            # 打印请求信息 - 诊断用
            print(f"[诊断] 请求模型: {self.model_api_name}")
            print(f"[诊断] 请求API端点: {self.api_base}")
            print(f"[诊断] 请求批次大小: {len(batch_texts)}")
            if len(batch_texts) > 0:
                print(f"[诊断] 第一条文本示例: {batch_texts[0][:100]}...")
            
            # 重试机制
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = self.client.embeddings.create(
                        model=self.model_api_name,
                        input=batch_texts
                    )
                    
                    # 打印响应信息 - 诊断用
                    print(f"[诊断] 响应类型: {type(response)}")
                    print(f"[诊断] 响应属性: {dir(response)}")
                    print(f"[诊断] 响应对象内容: {response}")
                    
                    if hasattr(response, 'data'):
                        print(f"[诊断] response.data类型: {type(response.data)}")
                        print(f"[诊断] response.data长度: {len(response.data) if response.data else 'None'}")
                        if response.data and len(response.data) > 0:
                            print(f"[诊断] 第一个数据项类型: {type(response.data[0])}")
                            print(f"[诊断] 第一个数据项属性: {dir(response.data[0])}")
                            if hasattr(response.data[0], 'embedding'):
                                print(f"[诊断] 第一个embedding类型: {type(response.data[0].embedding)}")
                                print(f"[诊断] 第一个embedding长度: {len(response.data[0].embedding) if response.data[0].embedding else 'None'}")
                            else:
                                print("[诊断] 第一个数据项没有embedding属性")
                    else:
                        print("[诊断] 响应对象没有data属性")
                    
                    # 更严格地检查response及其属性
                    if response and hasattr(response, 'data') and response.data:
                        # 确保每个data对象都有embedding属性
                        batch_embeddings = []
                        for idx, data in enumerate(response.data):
                            if hasattr(data, 'embedding') and data.embedding is not None:
                                batch_embeddings.append(data.embedding)
                            else:
                                # 如果单个结果缺少embedding，用零向量替代
                                print(f"警告: 数据项 {idx} 缺少embedding属性或为None，使用零向量替代")
                                print(f"[诊断] 数据项 {idx} 的内容: {data}")
                                print(f"[诊断] 数据项 {idx} 的属性: {dir(data)}")
                                batch_embeddings.append(np.zeros(self.dimensions))
                        
                        # 如果存在空的embedding，用零向量填充
                        if len(batch_embeddings) < len(batch_texts):
                            print(f"警告: 模型 {self.model_name} 返回的embedding数量少于请求数量")
                            print(f"[诊断] 返回数量: {len(batch_embeddings)}, 请求数量: {len(batch_texts)}")
                            for _ in range(len(batch_texts) - len(batch_embeddings)):
                                batch_embeddings.append(np.zeros(self.dimensions))
                                
                        all_embeddings.extend(batch_embeddings)
                        print(f"批次 {batch_num}/{total_batches} 处理完成")
                        break  # 成功处理，跳出重试循环
                    else:
                        print(f"警告: 模型 {self.model_name} 返回了空响应")
                        # 用零向量填充
                        all_embeddings.extend([np.zeros(self.dimensions) for _ in batch_texts])
                        break  # 即使是空响应也算完成，跳出重试循环
                except Exception as e:
                    retry_count += 1
                    print(f"批次 {batch_num} 处理出错 (尝试 {retry_count}/{max_retries}): {str(e)}")
                    print(f"[诊断] 错误类型: {type(e)}")
                    print(f"[诊断] 错误详情: {str(e)}")
                    import traceback
                    print(f"[诊断] 错误堆栈: {traceback.format_exc()}")
                    
                    if retry_count < max_retries:
                        time.sleep(2)  # 等待2秒后重试
                    else:
                        print(f"批量获取embedding时出错，达到最大重试次数: {str(e)}")
                        # 用零向量填充
                        all_embeddings.extend([np.zeros(self.dimensions) for _ in batch_texts])
                
        return np.array(all_embeddings)
    
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