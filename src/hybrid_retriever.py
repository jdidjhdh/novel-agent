"""
混合检索脚本 hybrid_retriever.py
=============================
用途：阶段2核心模块 → 阶段5工程化改造
- 关键词检索(BM25) + 向量检索 + RRF融合 + 关键词过滤 + rerank重排序
存放路径：E:/novel_agent/src/hybrid_retriever.py
阶段：2（混合检索+结果过滤+重排优化）→ 阶段5（工程化与优化）

设计思路（通俗讲解）：
1. 为什么需要混合检索？
   纯向量检索对"概念词"召回不准（比如查"唐三的武魂"，向量相似度可能召回到
   "风笑天的疾风双头狼"片段，因为两者都是武魂描述，语义相近）。
   BM25是基于词频的关键词检索，能精确匹配"唐三""武魂"这些关键词，召回更准。
   两路检索结果用RRF融合，取长补短。

2. 为什么需要关键词过滤（替代复杂的时间线过滤）？
   真正的"时间线过滤"需要预先标注每个chunk属于哪个剧情阶段（诺丁学院/史莱克/海神岛等），
   工程成本高。务实方案：从用户问题中抽取核心实体（人名+武魂名等），
   过滤掉不含这些关键词的片段。这样"查唐三武魂"就不会召回到不含"唐三"的片段。
   用soft filter：过滤后若结果太少(<3条)就回退到原结果，避免误伤。

3. 为什么需要rerank重排？
   召回阶段（BM25+向量）追求高召回率，会返回Top-20个片段，里面混有噪声。
   rerank用CrossEncoder对(query, doc)对做精排，能识别query和doc的细粒度相关性，
   把最相关的片段排到前面，最终只取Top-5给LLM。
   CrossEncoder比双塔向量检索更准，因为query和doc在attention层有交互。

阶段5工程化改造：
- 配置全部从 config.py 导入（CHROMA_PATH/MODEL_CACHE/模型名/TopK等）
- 异常处理用 safe_run/handle_oom 装饰器
- 5个核心方法加异常保护：BM25检索/向量检索/RRF融合/rerank/retrieve主流程
- CUDA OOM 自动清理缓存 + 友好报错

新版导入（消除langchain-community弃用警告）：
- from langchain_chroma import Chroma
- from langchain_huggingface import HuggingFaceEmbeddings

RRF (Reciprocal Rank Fusion) 公式：
   score(d) = Σ 1/(k + rank_i(d))，k通常取60
   其中rank_i(d)是文档d在第i路检索结果中的排名（从1开始）
   简单说：一个文档如果在两路检索中都靠前，它的RRF分数就高

面试口述版：
  "混合检索解决纯向量检索对概念词召回不准的问题。我用BM25做关键词检索，
  用Chroma向量库做语义检索，两路结果用RRF算法融合（公式1/(k+rank)），
  融合后用关键词过滤剔除不含核心实体的片段，最后用bge-reranker-v2-m3
  跨编码器对(query,doc)对精排，把Top-20召回压缩到Top-5给LLM。
  BM25负责精确匹配，向量检索负责语义召回，rerank负责细粒度相关性判别。
  阶段5我把5个核心方法都用safe_run装饰器做了异常保护，CUDA OOM自动清理缓存。"
"""
import os
# 离线模式：config.py已setdefault，这里再设一次兼容独立import
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import jieba  # 中文分词
from rank_bm25 import BM25Okapi  # BM25关键词检索算法
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # 新版独立集成包
from sentence_transformers import CrossEncoder  # rerank跨编码器

# 阶段5：从 config.py 统一导入配置
from config import (
    CHROMA_PATH, MODEL_CACHE,
    EMBEDDING_MODEL, RERANKER_MODEL, DEVICE,
    TOP_K_VECTOR, TOP_K_BM25, TOP_K_FINAL, RRF_K, MIN_FILTER_KEEP,
)
# 阶段5：从 exceptions.py 导入异常处理工具
from exceptions import (
    safe_run, handle_oom,
    RetrievalError, ModelLoadError, VectorDBError, GPUMemoryError,
    logger,
)

# 全局单例
_RETRIEVER = None


def _jieba_tokenize(text: str):
    """jieba中文分词，过滤空白字符和常见虚词
    BM25依赖词袋模型，必须先分词；过滤无意义虚词避免噪声
    """
    stop_words = set("，。！？；、的了在是和与我要你他她它")
    return [w for w in jieba.cut(text) if w.strip() and w not in stop_words]


class HybridRetriever:
    """混合检索器：BM25+向量+RRF+关键词过滤+rerank

    阶段5异常处理：每个核心方法加safe_run装饰器
    - 失败时抛RetrievalError（带function+args上下文）
    - rerank/embedding相关方法handle_oom自动处理CUDA OOM
    """

    def __init__(self):
        """初始化：加载embedding模型+向量库+BM25索引+rerank模型
        异常处理：任一加载失败抛对应自定义异常
        """
        logger.info("[HybridRetriever] 初始化中...")
        try:
            # 1. 加载embedding模型（向量检索用）
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                cache_folder=MODEL_CACHE,
                model_kwargs={"device": DEVICE},
            )
        except Exception as e:
            raise ModelLoadError(
                f"Embedding模型加载失败: {EMBEDDING_MODEL}",
                cause=e,
                context={"model": EMBEDDING_MODEL, "device": DEVICE}
            )

        try:
            # 2. 加载Chroma向量库
            self.vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
            count = self.vectordb._collection.count()
            logger.info(f"[HybridRetriever] 向量库加载完成，共 {count} 条记录")
        except Exception as e:
            raise VectorDBError(
                f"向量库读取失败: {CHROMA_PATH}",
                cause=e,
                context={"path": CHROMA_PATH}
            )

        try:
            # 3. 从Chroma取出全部chunk文本和metadata，用于构建BM25索引
            # 注意：7737个片段×500字符约4MB，内存无压力
            all_data = self.vectordb._collection.get(include=["documents", "metadatas"])
            self.all_chunks = [
                {"page_content": d, "metadata": m}
                for d, m in zip(all_data["documents"], all_data["metadatas"])
            ]
            # 4. 构建BM25索引（一次性，后续查询复用）
            logger.info("[HybridRetriever] 构建BM25索引...")
            tokenized_corpus = [_jieba_tokenize(c["page_content"]) for c in self.all_chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"[HybridRetriever] BM25索引构建完成，语料 {len(tokenized_corpus)} 篇")
        except Exception as e:
            raise RetrievalError(
                "BM25索引构建失败",
                cause=e,
                context={"path": CHROMA_PATH}
            )

        try:
            # 5. 加载rerank跨编码器模型
            logger.info(f"[HybridRetriever] 加载 rerank 模型 {RERANKER_MODEL} ...")
            self.reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE, max_length=512)
            logger.info("[HybridRetriever] 初始化完成")
        except Exception as e:
            raise ModelLoadError(
                f"rerank模型加载失败: {RERANKER_MODEL}",
                cause=e,
                context={"model": RERANKER_MODEL, "device": DEVICE}
            )

    # ---------- BM25关键词检索 ----------
    @safe_run(log_msg="BM25检索失败", reraise=True)
    def _bm25_search(self, query: str, k: int):
        """BM25检索：返回Top-K个(chunk, score)"""
        tokens = _jieba_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        # 按分数降序取Top-K的索引
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.all_chunks[i], float(scores[i])) for i in top_idx]

    # ---------- 向量检索 ----------
    @safe_run(log_msg="向量检索失败", reraise=True)
    def _vector_search(self, query: str, k: int):
        """向量检索：返回Top-K个(chunk, score)
        Chroma返回的是L2距离，越小越相似；为了和BM25分数方向一致（越大越相似），
        这里把距离取负数转成相似度
        """
        results = self.vectordb.similarity_search_with_score(query, k=k)
        return [
            ({"page_content": d.page_content, "metadata": d.metadata}, float(-s))
            for d, s in results
        ]

    # ---------- RRF融合 ----------
    @staticmethod
    def _rrf_fuse(vec_results, bm25_results, k: int = RRF_K):
        """RRF融合两路检索结果
        公式: score(d) = Σ 1/(k + rank_i(d))，rank从1开始
        用chunk_id做去重key（同一片段可能被两路都召回）
        """
        scores = {}
        docs = {}
        # 向量检索结果排名
        for rank, (doc, _) in enumerate(vec_results):
            cid = doc["metadata"].get("chunk_id")
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
            docs[cid] = doc
        # BM25结果排名
        for rank, (doc, _) in enumerate(bm25_results):
            cid = doc["metadata"].get("chunk_id")
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
            docs[cid] = doc
        # 按RRF总分降序
        sorted_cids = sorted(scores.keys(), key=lambda x: -scores[x])
        return [(docs[c], scores[c]) for c in sorted_cids]

    # ---------- 关键词抽取与过滤 ----------
    @staticmethod
    def _extract_keywords(query: str):
        """从用户问题抽取核心关键词（人名/实体等）
        简单策略：jieba分词后保留长度>=2的实词
        更精准可以用人名词典，但通用方案先用jieba
        """
        # 过滤疑问词等无意义词
        noise_words = set("什么怎么为什么如何哪里哪个")
        words = jieba.cut(query)
        keywords = [w for w in words if len(w) >= 2 and w not in noise_words]
        return keywords

    @staticmethod
    def _keyword_filter(docs, keywords, min_keep=MIN_FILTER_KEEP):
        """关键词过滤：只保留包含至少一个关键词的片段（soft filter）
        如果过滤后结果太少(<min_keep)，回退到原结果避免误伤
        """
        if not keywords or not docs:
            return docs
        filtered = [(doc, score) for doc, score in docs
                    if any(kw in doc["page_content"] for kw in keywords)]
        if len(filtered) < min_keep:
            # 过滤太严，回退原结果
            return docs
        return filtered

    # ---------- rerank重排 ----------
    @handle_oom
    @safe_run(log_msg="rerank重排失败", reraise=True)
    def _rerank(self, query: str, docs, top_k: int):
        """用CrossEncoder对(query, doc)对精排
        CrossEncoder把query和doc拼接送入transformer，输出相关性分数
        比双塔向量检索更准，因为query和doc在attention层有交互

        异常处理：
        - CUDA OOM → handle_oom 自动清理缓存 + 抛GPUMemoryError
        - 其他异常 → RetrievalError
        """
        if not docs:
            return []
        pairs = [(query, doc["page_content"]) for doc, _ in docs]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        # 按rerank分数降序取Top-K
        ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
        return [(doc_dict, float(rerank_score)) for (doc_dict, _), rerank_score in ranked[:top_k]]

    # ---------- 主检索入口 ----------
    @safe_run(log_msg="混合检索主流程失败", reraise=True)
    def retrieve(self, query: str, top_k: int = TOP_K_FINAL):
        """混合检索主流程：BM25+向量→RRF融合→关键词过滤→rerank重排
        返回[(doc_dict, score), ...]，doc_dict含page_content和metadata
        """
        # 步骤1：双路召回（粗排）
        vec_results = self._vector_search(query, TOP_K_VECTOR)
        bm25_results = self._bm25_search(query, TOP_K_BM25)

        # 步骤2：RRF融合
        fused = self._rrf_fuse(vec_results, bm25_results)

        # 步骤3：关键词过滤（剔除不含核心实体的片段）
        keywords = self._extract_keywords(query)
        if keywords:
            filtered = self._keyword_filter(fused, keywords)
        else:
            filtered = fused

        # 步骤4：rerank精排
        reranked = self._rerank(query, filtered, top_k)
        return reranked


def get_retriever():
    """获取全局HybridRetriever单例（避免重复加载模型）
    性能优化：单例复用，避免每次调用重新加载embedding+rerank模型（共2.2GB显存）
    """
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = HybridRetriever()
    return _RETRIEVER


# ==================== 测试入口 ====================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("混合检索模块测试 - 阶段5工程化版")
    logger.info("=" * 60)

    retriever = get_retriever()

    # 测试问题：阶段1已知的痛点问题
    test_queries = [
        "唐三的武魂是什么",
        "小舞是谁",
        "史莱克学院在哪里",
    ]

    for q in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"查询: {q}")
        logger.info("="*60)
        try:
            results = retriever.retrieve(q, top_k=TOP_K_FINAL)
            for i, (doc, score) in enumerate(results):
                cid = doc["metadata"].get("chunk_id")
                logger.info(f"  Top{i+1} (chunk_id={cid}, rerank_score={score:.4f}):")
                logger.info(f"    {doc['page_content'][:100]}...")
        except RetrievalError as e:
            logger.error(f"检索失败: {e}")