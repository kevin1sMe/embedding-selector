import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 提供商配置
PROVIDER_CONFIGS = {
    "openai": {
        "base_url": os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
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
    "text-embedding-granite-embedding-278m-multilingual": {
        "description": "Granite多语言嵌入模型（本地）",
        "provider": "local",
        "name": "text-embedding-granite-embedding-278m-multilingual",
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
        "batch_size": 10  # 调整为8，已测试可行
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

def get_model_info(model_name):
    """
    获取指定模型的配置信息
    
    Args:
        model_name: 模型名称
    
    Returns:
        模型配置信息字典
    
    Raises:
        ValueError: 当模型名称不存在时抛出
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"不支持的模型: {model_name}。可用模型: {list(AVAILABLE_MODELS.keys())}")
    
    return AVAILABLE_MODELS[model_name]

def get_provider_config(provider_name):
    """
    获取指定提供商的配置信息
    
    Args:
        provider_name: 提供商名称
    
    Returns:
        提供商配置信息字典
    
    Raises:
        ValueError: 当提供商名称不存在时抛出
    """
    if provider_name not in PROVIDER_CONFIGS:
        raise ValueError(f"不支持的提供商: {provider_name}。可用提供商: {list(PROVIDER_CONFIGS.keys())}")
    
    return PROVIDER_CONFIGS[provider_name]

def list_available_models(provider=None):
    """
    列出所有可用的模型
    
    Args:
        provider: 可选，如果提供则只列出指定提供商的模型
    
    Returns:
        模型信息的列表，每个元素为 (model_name, description, dimensions, provider)
    """
    if provider:
        models = [(name, info["description"], info["dimensions"], info["provider"]) 
                 for name, info in AVAILABLE_MODELS.items()
                 if info["provider"] == provider]
    else:
        models = [(name, info["description"], info["dimensions"], info["provider"]) 
                 for name, info in AVAILABLE_MODELS.items()]
    
    return sorted(models) 