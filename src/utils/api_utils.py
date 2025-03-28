import os
import time
import numpy as np
import openai
from typing import List, Dict, Any, Optional, Union

def create_openai_client(api_key: str, base_url: str) -> openai.OpenAI:
    """
    创建OpenAI客户端
    
    Args:
        api_key: API密钥
        base_url: API端点URL
        
    Returns:
        OpenAI客户端实例
    """
    return openai.OpenAI(
        api_key=api_key,
        base_url=base_url
    )

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3
):
    """
    使用指数退避策略的重试装饰器
    
    Args:
        func: 要重试的函数
        initial_delay: 初始延迟秒数
        exponential_base: 指数基数
        jitter: 是否添加随机抖动
        max_retries: 最大重试次数
        
    Returns:
        装饰后的函数
    """
    def wrapper(*args, **kwargs):
        # 初始化变量
        num_retries = 0
        delay = initial_delay
        
        # 开始重试循环
        while True:
            try:
                return func(*args, **kwargs)
            except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
                # 如果已经达到最大重试次数，则抛出异常
                if num_retries >= max_retries:
                    raise e
                
                # 指数延迟增加
                delay *= exponential_base * (1 + jitter * 0.1 * np.random.random())
                
                # 打印重试信息
                print(f"API错误: {str(e)}. 重试中... ({num_retries+1}/{max_retries})")
                time.sleep(delay)
                
                # 增加重试计数
                num_retries += 1
    
    return wrapper

def batched_api_call(
    client: openai.OpenAI,
    model_name: str,
    items: List[str],
    batch_size: int,
    call_func_name: str = "embeddings.create",
    **kwargs
) -> List[Any]:
    """
    批量处理API调用
    
    Args:
        client: OpenAI客户端
        model_name: 模型名称
        items: 要处理的项目列表
        batch_size: 每批的大小
        call_func_name: API调用函数名称
        **kwargs: 其他关键字参数
        
    Returns:
        处理结果列表
    """
    all_results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    print(f"处理 {len(items)} 条项目，分为 {total_batches} 批 (每批最多 {batch_size} 条)")
    
    # 按照批量大小分批处理
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"处理批次 {batch_num}/{total_batches} ({len(batch)} 条项目)...")
        
        # 动态获取API调用函数
        call_func = client
        for part in call_func_name.split('.'):
            call_func = getattr(call_func, part)
        
        # 重试机制
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                kwargs["model"] = model_name
                kwargs["input"] = batch
                
                response = call_func(**kwargs)
                
                # 获取结果并添加到列表
                if call_func_name == "embeddings.create":
                    batch_results = [item.embedding for item in response.data]
                else:
                    batch_results = response
                
                all_results.extend(batch_results)
                break  # 成功，跳出重试循环
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"在{max_retries}次尝试后仍然失败: {str(e)}")
                    raise e
                
                # 指数退避
                wait_time = (2 ** retry_count) + np.random.random()
                print(f"API调用失败: {str(e)}. {wait_time:.1f}秒后重试... ({retry_count}/{max_retries})")
                time.sleep(wait_time)
    
    return all_results 