"""
RAG问答脚本 rag_query.py（阶段2升级版）
=============================
用途：加载已构建的Chroma向量库，对接Ollama qwen2.5:7b，实现RAG问答
存放路径：E:\\novel_agent\\src\\rag_query.py
阶段：2（接入混合检索 hybrid_retriever，支持纯向量检索对比模式）

新版导入说明（消除langchain-community弃用警告）：
- from langchain_chroma import Chroma   （替代 langchain_community.vectorstores.Chroma）
- from langchain_ollama import ChatOllama（Ollama官方集成包）
- from langchain_core.documents import Document（用于把hybrid_retriever返回的dict转成Document）

问答流程（阶段2）：
用户提问 → 混合检索(BM25+向量+RRF+关键词过滤+rerank) Top-K原文片段 → 拼接提示词 → LLM生成回答
"""
import os
# 离线模式必须在import前设置，避免HuggingFace网络超时卡死
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # 新版独立集成包，避免langchain-community弃用警告
from langchain_ollama import ChatOllama
from langchain_core.documents import Document  # 阶段2新增：用于doc_dict转Document
# 阶段2新增：导入混合检索器（同目录模块，运行时需把src加入sys.path或从src目录运行）
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_retriever import get_retriever as get_hybrid_retriever

# ==================== 配置区 ====================
CHROMA_PATH = "E:/novel_agent/chroma_db"          # 已构建的向量库路径
MODEL_CACHE = "E:/novel_agent/src/model_cache"    # HuggingFace模型缓存
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"     # Embedding模型（须与构建时一致）
DEVICE = "cuda"
LLM_MODEL = "qwen2.5:7b"                           # Ollama本地Qwen模型
LLM_TEMP = 0.3                                     # 低温度，减少编造，回答更确定
TOP_K = 5                                          # 检索Top-K个相关片段

# 全局单例（避免每次问答重复加载模型）
_EMBEDDINGS = None
_VECTORDB = None
_LLM = None


def get_embeddings():
    """加载Embedding模型（与构建向量库时一致，向量维度必须匹配）"""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        print(f"[Embedding] 加载 {EMBEDDING_MODEL} ...")
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            cache_folder=MODEL_CACHE,
            model_kwargs={"device": DEVICE},
        )
    return _EMBEDDINGS


def load_vectordb():
    """加载已持久化的Chroma向量库"""
    global _VECTORDB
    if _VECTORDB is None:
        emb = get_embeddings()
        _VECTORDB = Chroma(persist_directory=CHROMA_PATH, embedding_function=emb)
        print(f"[向量库] 已加载，共 {_VECTORDB._collection.count()} 条记录")
    return _VECTORDB


def get_llm():
    """初始化Ollama本地Qwen模型"""
    global _LLM
    if _LLM is None:
        print(f"[LLM] 连接 Ollama {LLM_MODEL} ...")
        _LLM = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMP)
    return _LLM


def retrieve(question: str, k: int = TOP_K, use_hybrid: bool = True):
    """检索原文片段
    阶段2改动：新增use_hybrid参数
    - use_hybrid=True（默认）：调用混合检索器(BM25+向量+RRF+关键词过滤+rerank)
    - use_hybrid=False：用阶段1的纯向量检索（用于对比效果）
    返回langchain Document对象列表，保持与build_prompt兼容
    """
    if use_hybrid:
        # 阶段2：混合检索
        retriever = get_hybrid_retriever()
        results = retriever.retrieve(question, top_k=k)
        # hybrid_retriever返回的是(doc_dict, score)列表，转成Document保持兼容
        docs = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d, _ in results
        ]
    else:
        # 阶段1：纯向量检索（保留对比能力）
        vectordb = load_vectordb()
        docs = vectordb.similarity_search(question, k=k)
    return docs


def build_prompt(question: str, docs) -> str:
    """构建RAG提示词：把检索到的原文片段作为上下文，约束模型基于原文回答"""
    context = "\n\n".join([f"[片段{i + 1}] {d.page_content}" for i, d in enumerate(docs)])
    prompt = f"""你是一个小说知识库问答助手。请根据以下提供的小说原文片段回答用户问题。

要求：
1. 只依据下方原文片段回答，不要编造小说中不存在的设定或剧情
2. 如果原文片段中没有足够信息，请直接说明"原文中未提及相关内容"
3. 回答简洁准确，可适当引用原文

【小说原文片段】
{context}

【用户问题】
{question}

【回答】"""
    return prompt


def ask(question: str, use_hybrid: bool = True):
    """单次问答：检索 + 生成，返回(answer, docs)便于外部调用测试
    阶段2改动：新增use_hybrid参数，默认True用混合检索
    """
    # 1. 检索原文片段（阶段2默认混合检索）
    mode = "混合检索(BM25+向量+RRF+过滤+rerank)" if use_hybrid else "纯向量检索"
    print(f"\n[检索模式] {mode}")
    docs = retrieve(question, TOP_K, use_hybrid=use_hybrid)
    print(f"[检索] 命中 {len(docs)} 个片段：")
    for i, d in enumerate(docs):
        print(f"  片段{i + 1}(chunk_id={d.metadata.get('chunk_id')}): {d.page_content[:50]}...")

    # 2. 构建提示词并调用LLM生成
    prompt = build_prompt(question, docs)
    llm = get_llm()
    print(f"[生成] {LLM_MODEL} 推理中...")
    resp = llm.invoke(prompt)
    return resp.content, docs


def main():
    """交互式问答循环"""
    print("=" * 60)
    print("小说知识库问答 - 阶段2（混合检索）")
    print("=" * 60)
    # 预加载模型和向量库（混合检索器内部也会加载，这里预热避免首次问答慢）
    load_vectordb()
    get_llm()
    print(f"\nLLM: {LLM_MODEL} | 检索Top-K: {TOP_K} | 模式: 混合检索")
    print("输入问题开始问答，输入 exit 退出")
    print("特殊命令: /vec 用纯向量检索对比, /hyb 用混合检索(默认)\n")

    use_hybrid = True  # 默认混合检索
    while True:
        q = input("提问> ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit", "退出"):
            print("再见！")
            break
        # 切换检索模式命令
        if q == "/vec":
            use_hybrid = False
            print("[模式切换] 已切换到纯向量检索（阶段1模式）")
            continue
        if q == "/hyb":
            use_hybrid = True
            print("[模式切换] 已切换到混合检索（阶段2模式）")
            continue
        answer, _ = ask(q, use_hybrid=use_hybrid)
        print(f"\n回答：{answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()