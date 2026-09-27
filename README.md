# 小说人物对话智能体（基于RAG + LangGraph）

## 项目简介

本项目是一个基于 **RAG（检索增强生成）+ LangGraph** 的小说人物对话智能体。加载百万字小说TXT文本，构建小说专属知识库，采用**关键词检索 + 向量检索**混合检索方案，结合**分数归一加权融合、时间线过滤、rerank重排**，基于 **LangGraph** 实现多轮迭代检索智能体。支持与小说内人物进行角色扮演对话，基于原著设定回答，不编造小说中不存在的剧情。

## 核心功能

| 功能模块 | 说明 |
|---------|------|
| 知识库构建 | 加载百万字小说TXT，中文友好切分，Qwen3-Embedding向量化，Chroma持久化存储，支持**断点续建** |
| 混合检索 | BM25关键词检索 + 向量语义检索，**分数归一+加权融合**，时间线过滤，rerank精排（可开关） |
| 多轮检索智能体 | 基于 **LangGraph + CRAG思想**，文档不相关时自动重写查询再检索，最多3轮 |
| 角色扮演对话 | 支持多角色（唐三/小舞/大师等），第一人称回答，基于原文不编造 |
| 上下文长度限制 | 截断送入LLM的原文片段总字符数，防止token溢出 |
| RAG效果评估 | 输入测试问答对，计算召回率/命中率/MRR，输出评估报告 |
| Web前端 | Streamlit交互页面，对话界面，展示引用原文片段+chunk_id |

## 整体架构

```
用户提问
  |
  v
[route_query] -- LLM判断：是否需要查知识库？
  |
  +-- 需要 --> [retrieve] 混合检索(BM25+向量+融合+过滤+rerank)
  |                |
  |          [grade_documents] LLM评估文档相关性
  |                |
  |                +-- 相关/已达3轮 --> [generate] 角色扮演 --> END
  |                +-- 不相关/<3轮 --> [rewrite_query] --> [retrieve]
  |
  +-- 不需要 --> [generate] 直接生成 --> END
```

## 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| 语言 | Python | 3.12 |
| LLM框架 | LangGraph + LangChain生态 | 1.2+ |
| 向量数据库 | Chroma | 1.5+ |
| 本地LLM | Ollama + Qwen2.5:7b | 4-bit量化 |
| Embedding | Qwen3-Embedding-0.6B | 1024维 |
| Rerank | BAAI/bge-reranker-v2-m3 | CrossEncoder |
| 后端 | FastAPI + Uvicorn | 0.141+ |
| 前端 | Streamlit | 1.64+ |
| 关键词检索 | rank_bm25 + jieba | BM25Okapi |

## 项目结构

```
novel_agent/
  ├── novel_docs/              # 小说TXT文本
  │   └── 《斗罗大陆》作者：唐家三少.txt
  ├── chroma_db/               # Chroma向量库持久化
  │   ├── chroma.sqlite3
  │   └── build_progress.json # 断点续建进度文件
  ├── src/
  │   ├── config.py            # 统一配置（路径/模型/检索参数/时间线阶段）
  │   ├── build_knowledge_base.py  # 知识库构建（断点续建）
  │   ├── retrieval_tools.py  # 混合检索（分数归一+融合+时间线过滤+rerank开关）
  │   ├── agent_graph.py      # LangGraph智能体（CRAG多轮检索+上下文限制）
  │   ├── main_api.py         # FastAPI后端
  │   ├── streamlit_app.py    # Streamlit前端
  │   ├── rag_evaluate.py     # RAG评估脚本
  │   ├── utils.py            # 通用工具（日志/异常/上下文截断）
  │   └── model_cache/        # HuggingFace模型离线缓存
  ├── logs/                    # 日志目录
  ├── venv/                    # Python虚拟环境
  ├── README.md
  ├── .gitignore
  └── requirements.txt
```

## 数据说明（重要）

- `novel_docs/`（小说原文）因版权原因**不随仓库分发**，克隆后请自行准备小说 TXT 放入该目录，再运行 `src\build_knowledge_base.py` 构建知识库
- `chroma_db/`、`chroma.db/` 为向量库持久化数据（约百 MB 级），由构建脚本自动生成，不入库
- 首次构建耗时数分钟，支持断点续建（进度写入 `build_progress.json`）

## 环境部署

### 1. 前置条件

- Python 3.12+
- NVIDIA GPU（推荐8GB+显存）
- [Ollama](https://ollama.com/) 已安装

### 2. 安装步骤

```bash
# 克隆项目
git clone <repo_url>
cd novel_agent

# 创建虚拟环境
python -m venv venv

# Windows激活
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 拉取Ollama模型
ollama pull qwen2.5:7b

# 启动Ollama服务
ollama serve
```

### 3. 预下载模型（离线模式）

项目默认离线运行，需提前下载模型到缓存目录：

```bash
# 下载Embedding模型到 src/model_cache/
# 下载Rerank模型到 ~/.cache/huggingface/hub/
# 可用 huggingface-cli 或手动下载
```

## 项目启动

按以下顺序分步启动：

```bash
# 步骤1：构建知识库（首次运行，需数分钟）
venv\Scripts\python.exe src\build_knowledge_base.py

# 步骤2：启动FastAPI后端
venv\Scripts\python.exe src\main_api.py

# 步骤3：启动Streamlit前端（新开终端）
venv\Scripts\python.exe -m streamlit run src\streamlit_app.py

# 步骤4（可选）：运行RAG效果评估
venv\Scripts\python.exe src\rag_evaluate.py
```

## 演示效果

1. 打开浏览器访问 Streamlit（默认 http://localhost:8501）
2. 侧边栏选择角色（如"唐三"）
3. 输入问题（如"你的武魂是什么？"）
4. 智能体自动：分析问题→检索原文→角色扮演回答
5. 回答下方展示引用的原文片段和chunk_id

## 项目局限性

1. **时间线过滤精度有限**：按chunk_id范围划分剧情阶段，精确度依赖chunk切分质量
2. **本地LLM推理速度**：qwen2.5:7b在RTX4060上单次回答约3-5秒
3. **显存限制**：8GB显存需控制batch_size，大模型需更高端GPU
4. **角色扮演一致性**：长对话中角色设定可能出现漂移
5. **无增量更新**：新增小说内容需重新构建整个知识库

## 后续优化方向

1. **更精准的时间线标注**：用LLM自动标注每个chunk的剧情阶段
2. **增量索引**：支持追加新内容时不重建整个向量库
3. **更大多型LLM**：升级到14B/32B模型提升回答质量
4. **多模态扩展**：支持小说插图、有声书等多模态检索
5. **用户反馈闭环**：用户标记不满意回答，反向优化检索权重
6. **部署优化**：Docker容器化部署，支持API网关和负载均衡
