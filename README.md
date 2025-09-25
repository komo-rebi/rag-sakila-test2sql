# RAG Text2SQL 评估系统

基于Sakila数据集的Text2SQL系统评估框架，支持四大评估指标和细分召回率分析。

## 🎯 项目特色

- **四大评估指标**：执行准确率、语法正确率、语义相似度、检索召回率
- **细分召回率分析**：表级、列级、关键列、权重召回率四维分析
- **智能回退机制**：自动从完整版回退到简化版，确保系统可用性
- **可视化报告**：HTML报告 + JSON结果 + 控制台输出

## 📁 项目结构

```
rag-text2sql-homework/
├── src/                          # 核心源代码
│   ├── run_evaluation.py         # 主评估程序
│   ├── evaluation_metrics.py     # 指标计算引擎
│   ├── text2sql_system_simple.py # 简化版Text2SQL系统
│   ├── text2sql_system.py        # 完整版Text2SQL系统
│   ├── convert_sakila_dataset.py # 数据集转换工具
│   └── 05-text2sql-rag-v3-agent.py # 原始RAG实现参考
├── config/                       # 配置文件
│   └── salila_config_real_mock.yaml # 模拟模式配置
├── data/                         # 数据文件
│   ├── db_description.yaml       # 数据库字段描述
│   ├── ddl_statements.yaml       # DDL语句
│   ├── q2sql_pairs.json          # 问答对
│   └── sakila_real.json          # 转换后的标准数据集
├── results/                      # 评估结果
│   ├── evaluation_report_*.html  # HTML可视化报告
│   └── evaluation_results_*.json # JSON结果文件
├── docs/                         # 文档
│   └── README.md                 # 详细文档
├── requirements.txt              # Python依赖
└── README.md                     # 本文件
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 设置OpenAI API密钥
export OPENAI_API_KEY="your-api-key"
```

### 2. 运行评估
```bash
# 进入源代码目录
cd src

# 运行评估（模拟模式）
python run_evaluation.py --config ../config/salila_config_real_mock.yaml
```

### 3. 查看结果
- 控制台输出：实时评估进度和结果摘要
- HTML报告：`../results/evaluation_report_*.html`
- JSON结果：`../results/evaluation_results_*.json`

## 📊 评估结果

### 最新评估结果
- **综合得分**：98.40% ⭐⭐⭐⭐⭐
- **语法正确率**：100.00%
- **语义相似度**：97.44%
- **检索召回率**：97.22%

### 细分召回率分析
- **表级召回率**：97.22%
- **列级召回率**：72.22%
- **关键列召回率**：72.22%
- **权重召回率**：97.22%

## 🔧 配置说明

### 当前配置（模拟模式）
- **Text2SQL系统**：简化版（无需Milvus向量数据库）
- **运行模式**：模拟模式（无需MySQL数据库连接）
- **评估指标**：语法正确率 + 语义相似度 + 检索召回率

### 配置文件
- `config/salila_config_real_mock.yaml`：模拟模式配置（推荐）
- 可根据需要修改权重和参数

## 📈 技术架构

### Text2SQL系统
- **简化版**：基于文本匹配，无需向量数据库
- **完整版**：基于Milvus向量数据库的RAG检索

### 评估指标
1. **执行准确率**：通过实际执行SQL比较结果
2. **语法正确率**：SQL语法验证和结构检查
3. **语义相似度**：嵌入相似度 + 结构相似度
4. **检索召回率**：四维细分分析

## 📚 数据集

### Sakila数据集
- **规模**：16个表，36条查询
- **查询类型**：SELECT(9) + INSERT(9) + UPDATE(9) + DELETE(9)
- **业务域**：电影租赁管理系统

### 知识库结构
- **DDL语句**：完整的表结构定义
- **字段描述**：详细的业务含义说明
- **问答示例**：36条高质量的问题-SQL对

## 🛠️ 开发说明

### 核心模块
- `run_evaluation.py`：主评估程序，协调各模块
- `evaluation_metrics.py`：指标计算引擎，实现四大指标
- `text2sql_system_simple.py`：简化版Text2SQL系统
- `text2sql_system.py`：完整版Text2SQL系统

### 扩展开发
- 添加新的评估指标
- 支持新的数据集
- 优化Text2SQL系统性能

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

---

*RAG Text2SQL评估系统 - 让Text2SQL评估更精确、更全面、更智能！*