import os
from dotenv import load_dotenv
from pathlib import Path

# 加载环境变量 - 确保从项目根目录加载.env文件
project_root = Path(__file__).parent.parent.parent
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path=dotenv_path)

# 提供商配置
PROVIDER_CONFIGS = {
    "openai": {
        "base_url": os.getenv("DEFAULT_API_BASE") or "https://api.openai.com/v1",
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

# 打印配置进行调试
print(PROVIDER_CONFIGS)

# 预设可用的模型列表
AVAILABLE_MODELS = {
    # OpenAI官方模型（多语言支持）
    "text-embedding-ada-002": {
        "description": "OpenAI第二代Ada嵌入模型",
        "provider": "openai",
        "name": "text-embedding-ada-002",
        "dimensions": 1536,
        "max_tokens": 8191,
        "batch_size": 2048,
        "languages": "多语言"
    },
    "text-embedding-3-small": {
        "description": "OpenAI第三代小型嵌入模型",
        "provider": "openai",
        "name": "text-embedding-3-small",
        "dimensions": 1536,
        "max_tokens": 8191,
        "batch_size": 2048,
        "languages": "多语言"
    },
    "text-embedding-3-large": {
        "description": "OpenAI第三代大型嵌入模型",
        "provider": "openai",
        "name": "text-embedding-3-large",
        "dimensions": 3072,
        "max_tokens": 8191,
        "batch_size": 2048,
        "languages": "多语言"
    },
    
    # LM Studio本地模型
    "text-embedding-gte-large-zh": {
        "description": "GTE大型中文嵌入模型（本地）",
        "provider": "local",
        "name": "text-embedding-gte-large-zh",
        "dimensions": 1024,
        "max_tokens": 512,
        "batch_size": 512,
        "languages": "中文"
    },
    "text-embedding-bge-large-zh-v1.5": {
        "description": "百度开源的中英双语大型嵌入模型（本地）",
        "provider": "local",
        "name": "text-embedding-bge-large-zh-v1.5",
        "dimensions": 1024,
        "max_tokens": 512,
        "batch_size": 512,
        "languages": "中文、英文"
    },
    "text-embedding-m3e-base": {
        "description": "M3E基础嵌入模型（本地）",
        "provider": "local",
        "name": "text-embedding-m3e-base",
        "dimensions": 768,
        "max_tokens": 512,
        "batch_size": 512,
        "languages": "中文、英文"
    },
    "text-embedding-granite-embedding-278m-multilingual": {
        "description": "Granite多语言嵌入模型（本地）",
        "provider": "local",
        "name": "text-embedding-granite-embedding-278m-multilingual",
        "dimensions": 768,
        "max_tokens": 512,
        "batch_size": 512,
        "languages": "多语言（英文、德文、西班牙文、法文、日文、葡萄牙文、阿拉伯文、捷克文、意大利文、韩文、荷兰文、中文等）"
    },
    # "jina-embeddings-v2-base-zh" : {
    #     "description": "Jina开源的中英双语嵌入模型",
    #     "provider": "local",
    #     "name": "jina-embeddings-v2-base-zh",
    #     "dimensions": 768,
    #     "max_tokens": 512,
    #     "batch_size": 512,
    #     "languages": "中文、英文"
    # },
    "text-embedding-multilingual-e5-large-instruct" : {
        "description": "E5大型多语言嵌入模型",
        "provider": "local",
        "name": "text-embedding-multilingual-e5-large-instruct",
        "dimensions": 1024,
        "max_tokens": 512,
        "batch_size": 512,
        "languages": "多语言"
    },
    
    # 其他可能通过自定义API端点使用的模型
    "m3e-large": {
        "description": "Moka开源的中英双语嵌入模型",
        "provider": "custom",
        "name": "m3e-large",
        "dimensions": 1024,
        "max_tokens": 512,
        "batch_size": 512,
        "languages": "中文、英文"
    },

    # 新增模型
    "text-embedding-3-large": {
        "description": "OpenAI第三代大型嵌入模型",
        "provider": "openai",
        "name": "text-embedding-3-large",
        "dimensions": 3072,
        "max_tokens": 8191,
        "batch_size": 2048,
        "languages": "多语言"
    },
    "hunyuan": {
        "description": "腾讯混元嵌入模型",
        "provider": "custom",
        "name": "hunyuan-embedding",
        "dimensions": 1024,
        "max_tokens": 1024,
        "batch_size": 10,  # 联系客服说只支持10，官网写200...
        "languages": "中文、英文"
    },
    "doubao": {
        "description": "豆包嵌入模型",
        "provider": "custom",
        "name": "doubao-embedding-large-text-240915",
        "dimensions": 1024,
        "max_tokens": 4096,
        "batch_size": 256,
        "languages": "中文、英文"
    },
    "baichuan": {
        "description": "百川嵌入模型",
        "provider": "custom",
        "name": "Baichuan-Text-Embedding",
        "dimensions": 1024,
        "max_tokens": 512,
        "batch_size": 16,  # 错误信息显示最多16
        "languages": "中文、英文"
    },
    "qwen": {
        "description": "通义千问嵌入模型",
        "provider": "custom",
        "name": "text-embedding-v3",
        "dimensions": 1024,
        "max_tokens": 8192, # 单字符串时支持8192，多字符串时支持2048
        "batch_size": 10,  # 错误信息显示最多10
        "languages": "中文、英文"
    },
    "baidu": {
        "description": "百度嵌入模型",
        "provider": "custom",
        "name": "Embedding-V1",
        "dimensions": 1024,
        "max_tokens": 384, # 384token且不超过1000字节
        "batch_size": 16,  # 保守估计
        "languages": "中文、英文"
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