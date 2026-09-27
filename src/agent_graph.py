"""
LangGraph 智能体模块 agent_graph.py
===================================
用途：构建多轮迭代检索智能体（CRAG思想），实现角色扮演对话
存放路径：E:/novel_agent/src/agent_graph.py

架构图：
  用户提问
    |
  [route_query] -- LLM判断：是否需要查知识库？
    |
    +-- 需要(剧情/设定/人物问题) --> [retrieve]混合检索
    |                                    |
    |                              [grade_documents]LLM评估文档相关性
    |                                    |
    |                                    +-- 相关/已达上限 --> [generate]角色扮演 --> END
    |                                    +-- 不相关/且<3轮 --> [rewrite_query]重写 --> [retrieve]
    |
    +-- 不需要(闲聊) --> [generate]直接生成 --> END

核心特性：
1. CRAG多轮检索：文档不相关时LLM重写查询再检索，最多3轮避免死循环
2. 上下文长度限制：truncate_context截断原文片段，防止token溢出
3. MemorySaver+thread_id：跨轮对话记忆
4. 角色扮演：generate节点system prompt设定角色，第一人称回答
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
from typing import TypedDict
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from config import (
    LLM_MODEL, LLM_TEMP, TOP_K_FINAL,
    MAX_RETRIEVAL_ROUNDS, DEFAULT_ROLE, HISTORY_WINDOW,
    MAX_CONTEXT_CHARS, ROLE_PROFILES, CHARACTER_ALIASES, EARLY_EVENT_KEYWORDS,
)
from utils import (
    safe_run, truncate_context,
    LLMInvokeError, RetrievalError,
    logger,
)

# 导入检索工具模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval_tools import get_retriever

# 全局LLM单例
_LLM = None


def get_llm():
    """获取全局LLM单例，Ollama未启动时友好报错"""
    global _LLM
    if _LLM is None:
        logger.info(f"[LLM] 连接 Ollama {LLM_MODEL} ...")
        try:
            _LLM = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMP)
            _LLM.invoke("测试")
            logger.info(f"[LLM] 连接成功，模型 {LLM_MODEL} 可用")
        except Exception as e:
            err_msg = str(e).lower()
            if "connection" in err_msg or "refused" in err_msg:
                raise LLMInvokeError(
                    f"Ollama服务未启动或无法连接，请先运行：ollama serve",
                    cause=e,
                    context={"model": LLM_MODEL}
                )
            if "not found" in err_msg or "model" in err_msg:
                raise LLMInvokeError(
                    f"模型 {LLM_MODEL} 不存在，请先拉取：ollama pull {LLM_MODEL}",
                    cause=e,
                    context={"model": LLM_MODEL}
                )
            raise LLMInvokeError(
                f"LLM初始化失败: {LLM_MODEL}",
                cause=e,
                context={"model": LLM_MODEL}
            )
    return _LLM


# ==================== State定义 ====================
class AgentState(TypedDict):
    """智能体状态：LangGraph各节点间传递的数据"""
    question: str            # 用户原始问题
    query: str              # 当前检索查询（可能被rewrite重写）
    documents: list         # 检索到的原文片段
    messages: list           # 对话历史（跨轮累积）
    retrieval_count: int    # 已检索次数
    need_retrieval: bool     # 路由标志
    docs_relevant: bool      # 文档相关性标志
    generation: str         # 最终回答
    role: str                # 角色设定


# ==================== 节点定义 ====================

@safe_run(log_msg="问题路由失败", reraise=True)
def route_query_node(state: AgentState):
    """节点1：判断用户问题是否需要检索知识库"""
    question = state["question"]
    llm = get_llm()

    prompt = f"""判断以下用户问题是否需要查询小说知识库原文来回答。

需要查询的情况：
- 询问小说剧情、人物设定、武魂/能力、时间线、人物关系等小说相关内容
- 询问"某某是谁""某某做了什么"等需要原文依据的问题

不需要查询的情况：
- 打招呼、闲聊、情绪表达
- 纯粹的角色扮演对话（如"你好""你今天怎么样"）

用户问题：{question}

只回答"需要"或"不需要"，不要解释。"""

    resp = llm.invoke(prompt)
    answer = resp.content.strip()
    need = "需要" in answer and "不需要" not in answer
    logger.info(f"  [路由] 问题需要检索知识库：{need}")

    return {
        "need_retrieval": need,
        "query": question,
        "retrieval_count": 0,
        "docs_relevant": False,
    }


def normalize_query(query: str, role: str) -> str:
    """查询归一化：人称替换为角色名 + 角色别名替换为正式名，提升关系型问题实体召回"""
    q = query.replace("你", role).replace("我", role)
    for alias, formal in CHARACTER_ALIASES.items():
        q = q.replace(alias, formal)
    return q


@safe_run(log_msg="文档检索失败", reraise=True)
def retrieve_node(state: AgentState):
    """节点2：调用混合检索器检索原文片段
    检索后用truncate_context限制上下文长度，防止token溢出
    """
    query = state["query"]
    query = normalize_query(query, state.get("role", DEFAULT_ROLE))
    # 早期回忆类问题：追加早期场景锚定词，保证多轮重写后仍检索早期剧情
    # 例：小舞问"你和唐三怎么认识的" -> 重写后可能丢失"怎么认识"，追加锚定词兜底
    question = state.get("question", query)
    if any(kw in question for kw in EARLY_EVENT_KEYWORDS) and "诺丁学院" not in query:
        query = f"{query} 诺丁学院 初识"
        logger.info(f"  [检索] 早期回忆锚定：{query}")
    count = state.get("retrieval_count", 0) + 1

    logger.info(f"  [检索] 第{count}轮，查询：{query}")
    retriever = get_retriever()
    results = retriever.retrieve(query, top_k=TOP_K_FINAL)

    # 转成Document对象
    docs = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d, _ in results
    ]

    # 上下文长度限制：截断防止token溢出
    docs, total_chars = truncate_context(docs, MAX_CONTEXT_CHARS)
    logger.info(f"  [检索] 命中{len(docs)}个片段，总字符数{total_chars}")
    for i, d in enumerate(docs):
        logger.info(f"    片段{i+1}(chunk_id={d.metadata.get('chunk_id')}): {d.page_content[:60]}...")

    return {"documents": docs, "retrieval_count": count}


@safe_run(log_msg="文档相关性评估失败", reraise=True)
def grade_documents_node(state: AgentState):
    """节点3：评估检索文档是否与用户问题相关（CRAG核心思想）"""
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        logger.info("  [评估] 无检索文档，标记为不相关")
        return {"docs_relevant": False}

    llm = get_llm()
    docs_text = "\n\n".join([
        f"[片段{i+1}] {d.page_content[:200]}"
        for i, d in enumerate(documents)
    ])

    prompt = f"""评估以下检索到的小说原文片段是否包含回答用户问题所需的信息。

判断标准：
1. 只有当片段中包含能直接回答用户问题所问内容的特定信息（具体的人物关系、事件经过、设定细节）时，才算"相关"
2. 如果片段只是出现了问题中的人名或关键词，但内容与问题所问的无关（例如问"A和B什么关系"，片段只讲B和其他人的事），算"不相关"
3. 多个片段中只要有一个片段真正回答了问题，就算"相关"
4. 请先在心里自问"这些片段能否真正回答用户的问题"，再作判断；不要因为片段中出现人名就轻易判"相关"，也不要因为有无关内容就判"不相关"

用户问题：{question}

【原文片段】
{docs_text}

只回答"相关"或"不相关"，不要解释。"""

    resp = llm.invoke(prompt)
    answer = resp.content.strip()
    relevant = "相关" in answer and "不相关" not in answer
    logger.info(f"  [评估] 文档相关性：{'相关' if relevant else '不相关'}")
    return {"docs_relevant": relevant}


@safe_run(log_msg="查询重写失败", reraise=True)
def rewrite_query_node(state: AgentState):
    """节点4：重写检索查询，使其更精准匹配小说原文"""
    question = state["question"]
    old_query = state["query"]
    documents = state.get("documents", [])
    count = state.get("retrieval_count", 0)

    llm = get_llm()
    docs_summary = "\n".join([d.page_content[:100] for d in documents[:3]])

    role = state.get("role", DEFAULT_ROLE)

    prompt = f"""之前的检索结果与用户问题不相关，请重写检索查询。

用户原始问题：{question}
当前扮演角色：{role}
之前的查询：{old_query}
检索到但不相关的片段：
{docs_summary}

重要：问题中的"你""我"指代当前扮演角色{role}。重写时保留角色身份，例如扮演大师时，"你和比比东什么关系"应重写为"大师 比比东 关系"，而不是"唐三 比比东 关系"。

请分析为什么不相关，并重写查询使其更精准地匹配小说原文。
要求：
1. 提取问题中的核心实体（人名、地名、武魂名等），用小说中的正式称呼
2. 保留角色身份，去掉疑问词，只保留关键词
3. 例如"三哥的武魂"重写为"唐三 武魂"

只输出重写后的查询，不要解释。"""

    resp = llm.invoke(prompt)
    new_query = resp.content.strip()
    logger.info(f"  [重写] 第{count}轮不相关，新查询：{new_query}")
    return {"query": new_query}


@safe_run(log_msg="回答生成失败", reraise=True)
def generate_node(state: AgentState):
    """节点5：生成角色扮演回答
    基于检索文档+角色设定+对话历史生成回答
    上下文已在retrieve节点截断，这里不再重复限制
    """
    question = state["question"]
    documents = state.get("documents", [])
    messages = state.get("messages", [])
    role = state.get("role", DEFAULT_ROLE)

    llm = get_llm()

    if documents:
        context = "\n\n".join([
            f"[原文片段{i+1}] (chunk_id={d.metadata.get('chunk_id', '?')}) {d.page_content}"
            for i, d in enumerate(documents)
        ])
        context_note = "请基于以下小说原文片段回答，不要编造小说中不存在的设定。"
    else:
        context = "（无需检索，基于角色理解回答）"
        context_note = "请基于你对角色的理解回答，保持角色设定。"

    # 对话历史（最近5轮=10条消息）
    history = ""
    if messages:
        recent = messages[-HISTORY_WINDOW:]
        history = "\n\n【对话历史】\n"
        for i, msg in enumerate(recent):
            speaker = "用户" if i % 2 == 0 else role
            history += f"{speaker}: {msg}\n"

    # 角色档案：保证角色一致性（无检索时也约束设定）
    role_profile = ROLE_PROFILES.get(role, "")
    if role_profile:
        role_section = f"\n【你的角色档案】\n{role_profile}\n"
    else:
        role_section = ""

    prompt = f"""你正在扮演小说《斗罗大陆》中的{role}。请用{role}的口吻和第一人称回答用户问题。
{role_section}
要求：
1. 严格依据【你的角色档案】中的设定扮演，不得违背档案中的出身、经历和性格；若原文片段与角色档案冲突，以角色档案为准
2. 保持{role}的角色设定和性格特点
3. {context_note}
4. 回答情节类问题（认识经过/事件经过/如何发生）时，只陈述检索片段中直接对应的情节；片段中未直接描述的事件不要补充细节，宁可回答"我不太记得这件事了"
5. 如果原文中没有相关信息，用角色口吻说"我不太记得这件事了"
6. 回答自然流畅，符合{role}的说话风格

{context_note}

【小说原文片段】
{context}
{history}

【用户问题】
{question}

【{role}的回答】"""

    resp = llm.invoke(prompt)
    generation = resp.content.strip()

    # 更新对话历史
    new_messages = list(messages) + [question, generation]
    return {"generation": generation, "messages": new_messages}


# ==================== 路由函数 ====================

def route_after_query(state: AgentState):
    """路由1：route_query之后，根据need_retrieval决定走向"""
    if state.get("need_retrieval", True):
        return "retrieve"
    return "generate"


def decide_to_generate(state: AgentState):
    """路由2：grade_documents之后，决定生成还是重写查询
    CRAG核心路由：相关则生成，不相关且<3轮则重写，不相关且>=3轮强制生成
    """
    docs_relevant = state.get("docs_relevant", False)
    count = state.get("retrieval_count", 0)

    if docs_relevant:
        logger.info(f"  [路由] 文档相关，开始生成")
        return "generate"

    if count >= MAX_RETRIEVAL_ROUNDS:
        logger.info(f"  [路由] 已达{MAX_RETRIEVAL_ROUNDS}轮上限，强制生成")
        return "generate"

    logger.info(f"  [路由] 文档不相关(第{count}轮)，重写查询再检索")
    return "rewrite_query"


# ==================== 构建图 ====================

def build_graph():
    """构建LangGraph智能体图"""
    graph = StateGraph(AgentState)

    graph.add_node("route_query", route_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "route_query")
    graph.add_conditional_edges(
        "route_query",
        route_after_query,
        {"retrieve": "retrieve", "generate": "generate"},
    )
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"rewrite_query": "rewrite_query", "generate": "generate"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# 全局图单例
_GRAPH = None


def get_graph():
    """获取全局LangGraph图单例"""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


# ==================== 对话入口 ====================
def chat(question: str, thread_id: str = "1", role: str = DEFAULT_ROLE):
    """智能体对话入口（供main_api调用）

    Args:
        question: 用户问题
        thread_id: 会话ID（同ID保持对话记忆）
        role: 角色名

    Returns:
        tuple: (回答字符串, 引用片段列表citations)
        citations: [{"chunk_id": int, "content": str}, ...]
    """
    try:
        graph = get_graph()
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "question": question,
            "role": role,
            "messages": [],
            "documents": [],
        }
        final_state = graph.invoke(initial_state, config=config)
        answer = final_state.get("generation", "（无回答）")
        documents = final_state.get("documents", [])
        # 提取引用信息供前端展示
        citations = [
            {"chunk_id": d.metadata.get("chunk_id", "?"), "content": d.page_content[:200]}
            for d in documents
        ]
        return answer, citations
    except (LLMInvokeError, RetrievalError) as e:
        logger.error(f"对话失败: {e}")
        return f"抱歉，出错了：{e.message}\n排查建议：检查Ollama服务是否启动、知识库是否已构建", []
    except Exception as e:
        logger.error(f"对话未知异常: {type(e).__name__}: {e}")
        return f"抱歉，发生未知错误：{type(e).__name__}: {e}", []


# ==================== 测试入口 ====================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("LangGraph 智能体测试 - 工程化版")
    logger.info("=" * 60)

    test_cases = [
        ("1", "唐三", "你好，你是谁？"),
        ("1", "唐三", "你的武魂是什么？"),
        ("1", "唐三", "小舞是谁？"),
        ("2", "小舞", "你好，介绍下你自己"),
    ]

    for tid, role, q in test_cases:
        logger.info(f"\n{'='*60}")
        logger.info(f"[会话{tid}] 角色={role} 问题={q}")
        logger.info("=" * 60)
        answer = chat(q, thread_id=tid, role=role)
        logger.info(f"回答: {answer}")
