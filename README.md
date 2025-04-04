# Embedding选择器

这个项目提供了一个灵活的Embedding模型选择器，可以轻松切换不同的嵌入模型和API端点，用于评估各种模型在中文和中英文混合场景下的性能表现。

## 功能特点

- 支持切换不同的embedding模型（OpenAI和兼容的第三方模型）
- 支持自定义API端点
- 内置中文和中英文混合的测试用例
- 提供模型性能评估工具和指标
- 简单易用的API设计

## 项目结构

```
embedding-selector/
├── .env                  # 环境变量配置文件
├── pyproject.toml        # 项目配置和依赖管理
├── README.md             # 项目说明文档
├── run_test.py           # 快速测试脚本
└── src/
    ├── embedding_selector.py  # 主要的模型选择器类
    ├── test_data.py           # 测试数据（commit messages和查询）
    ├── test.py                # 模型评估脚本
    └── example.py             # 使用示例
```

## 快速开始

### 使用Nix环境

如果您使用Nix作为包管理器，可以使用项目中的`shell.nix`文件创建一个隔离的开发环境，它会自动解决所有系统依赖问题：

```bash
# 使用nix-shell启动一个带有所有依赖的shell环境
nix-shell

# 在nix-shell环境中，激活Python虚拟环境
source .venv/bin/activate

# 运行测试脚本
python src/test.py
```

您也可以使用单行命令运行：

```bash
# 一步完成：启动nix-shell并运行Python脚本
nix-shell --run 'source .venv/bin/activate && python src/test.py'
```

这种方法特别适合在不同的Linux发行版上避免系统库依赖问题，如`libstdc++`或`libz`等。

### 安装依赖

推荐使用[uv](https://github.com/astral-sh/uv)进行依赖管理，它比pip更快、更可靠：

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows

# 安装依赖
uv pip install -e .
```

如果您没有安装uv，也可以使用pip：

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows

# 安装依赖
pip install -e .
```

### 配置环境变量

编辑`.env`文件，设置您的API密钥和端点：

```
# 本地API端点（LM Studio）
DEFAULT_API_BASE=http://127.0.0.1:1234/v1

# OpenAI API密钥（使用本地模型时可以是任意值）
OPENAI_API_KEY=dummy-key
```

### 运行测试

确保LM Studio已经启动并加载了嵌入模型后，执行：

```bash
python run_test.py
```

这将测试所有配置的模型，并生成一个比较结果的表格。

### 运行示例

```bash
python -m src.example
```

## 使用方法

### 基本用法

```python
from src.embedding_selector import EmbeddingSelector

# 使用默认模型（text-embedding-gte-large-zh）
selector = EmbeddingSelector()

# 获取文本的embedding
text = "这是一个测试文本"
embedding = selector.get_embedding(text)

# 批量获取embedding
texts = ["第一个文本", "第二个文本", "第三个文本"]
embeddings = selector.get_batch_embeddings(texts)

# 计算两个embedding之间的相似度
similarity = selector.calculate_similarity(embeddings[0], embeddings[1])
print(f"相似度: {similarity}")
```

### 切换模型

```python
# 切换到另一个模型
selector.set_model("text-embedding-bge-large-zh-v1.5")

# 切换到第三个模型
selector.set_model("text-embedding-m3e-base")
```

### 使用自定义API端点

```python
# 设置自定义API端点
selector.set_api_endpoint("http://127.0.0.1:8000/v1")
```

### 相似度搜索

```python
# 查询文本
query = "如何优化性能"
query_embedding = selector.get_embedding(query)

# 候选文本列表
candidates = [
    "优化代码执行性能",
    "修复登录页面的bug",
    "添加新的用户界面"
]

# 获取候选文本的embedding
candidate_embeddings = selector.get_batch_embeddings(candidates)

# 找到最相似的候选文本
similar_texts = selector.find_most_similar(
    query_embedding, 
    candidate_embeddings, 
    candidates
)

# 打印结果
for i, (text, score) in enumerate(similar_texts):
    print(f"{i+1}. {text} (相似度: {score:.4f})")
```

## 支持的模型

以下是预设的可用模型列表：

- **LM Studio本地模型**:
  - `text-embedding-gte-large-zh`: GTE大型中文嵌入模型
  - `text-embedding-bge-large-zh-v1.5`: 百度开源的中英双语大型嵌入模型
  - `text-embedding-m3e-base`: M3E基础嵌入模型

- **OpenAI官方模型**:
  - `text-embedding-ada-002`: OpenAI第二代Ada嵌入模型
  - `text-embedding-3-small`: OpenAI第三代小型嵌入模型
  - `text-embedding-3-large`: OpenAI第三代大型嵌入模型

您可以根据需要在`EmbeddingSelector`类中添加更多模型。

## 评估指标

模型评估脚本使用以下指标来比较不同模型的性能：

- Precision@1: 查询结果中第一个结果的准确率
- Precision@3: 查询结果中前三个结果的平均准确率
- Precision@5: 查询结果中前五个结果的平均准确率
- 处理时间: 处理所有查询所需的时间

## 注意事项

- 请确保LM Studio已经启动并加载了相应的嵌入模型
- 如果使用OpenAI模型，请确保您的API密钥有足够的配额
- 评估多个模型可能会消耗较多的API配额

## 许可证

MIT 