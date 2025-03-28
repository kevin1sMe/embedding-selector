"""
测试数据集：中文和中英文混合的commit messages
"""

# 各种风格的commit messages作为测试数据
COMMIT_MESSAGES = [
    # 纯中文commit messages
    "修复首页加载速度慢的问题",
    "优化用户登录流程",
    "新增数据导出功能",
    "修复了用户反馈的崩溃问题",
    "更新文档说明",
    "重构了代码结构，提高了可维护性",
    "删除了废弃的API调用",
    "添加单元测试用例",
    "修改了配置文件中的默认设置",
    "解决了在iOS设备上的兼容性问题",
    
    # 中英文混合的commit messages
    "fix: 修复了登录页面的bug",
    "feat: 添加了新的payment接口",
    "docs: 更新API文档",
    "refactor: 重构用户认证模块",
    "test: 增加了对checkout流程的测试",
    "style: 调整了UI组件的样式",
    "perf: 优化了数据库查询性能",
    "chore: 更新了package依赖",
    "fix(ui): modal组件关闭按钮失效问题",
    "feat(api): 新增用户数据同步endpoint",
    
    # 技术专业术语混合的commit messages
    "修复Redis连接池泄露问题",
    "优化React组件的渲染性能",
    "新增Elasticsearch索引管理功能",
    "重构JWT认证逻辑，提高安全性",
    "解决了Docker容器内存占用过高的问题",
    "添加GraphQL查询缓存机制",
    "更新了Webpack配置，提高构建速度",
    "修复了多线程并发导致的数据不一致问题",
    "添加了对WebSocket连接的心跳检测",
    "优化了MongoDB聚合查询的执行效率",
    
    # 团队协作相关的commit messages
    "根据Code Review反馈修改代码",
    "合并develop分支的最新更改",
    "准备v2.0.0版本发布",
    "修复QA团队报告的regression问题",
    "实现了产品经理提出的新需求",
    "临时提交，WIP：用户管理模块",
    "协同后端API调整相应的前端代码",
    "根据UI设计稿更新组件样式",
    "添加了新功能的feature flag",
    "解决合并冲突，保留双方更改",
]

# 用于测试的查询语句
TEST_QUERIES = [
    # 功能相关查询
    "如何修复bug",
    "添加新功能",
    "更新文档",
    "优化性能",
    "重构代码",
    
    # 技术相关查询
    "关于React组件的提交",
    "数据库优化",
    "API开发",
    "UI界面调整",
    "Docker相关问题",
    
    # 过程相关查询
    "代码审查后的修改",
    "版本发布准备",
    "修复测试中发现的问题",
    "合并分支",
    "解决冲突"
] 