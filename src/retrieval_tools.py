"""
检索工具模块 retrieval_tools.py
================================
用途：实现混合检索（BM25关键词+向量检索）、分数归一加权融合、时间线过滤、rerank重排
存放路径：E:/novel_agent/src/retrieval_tools.py

检索流程：
  1. BM25关键词检索（精确匹配人名/实体词）
  2. 向量语义检索（语义召回相关片段）
  3. 分数归一+加权融合（替代RRF，权重可配置）
  4. 时间线过滤（按剧情阶段过滤不属于当前时间的片段）
  5. rerank精排（可开关，关闭时跳过减少推理开销）

设计要点：
- 分数归一：BM25和向量分数各自min-max归一化到[0,1]，再加权融合
- 时间线过滤：按chunk_id范围判断剧情阶段，soft filter不足回退
- rerank开关：RERANK_ENABLED=False时直接返回融合结果，省1GB显存
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import jieba
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

from config import (
    CHROMA_PATH, MODEL_CACHE,
    EMBEDDING_MODEL, RERANKER_MODEL, DEVICE,
    TOP_K_VECTOR, TOP_K_BM25, TOP_K_FINAL,
    MIN_FILTER_KEEP, TIMELINE_PHASES, EARLY_EVENT_KEYWORDS,
    WEIGHT_VECTOR, WEIGHT_BM25, RERANK_ENABLED,
)
from utils import (
    safe_run, handle_oom,
    RetrievalError, ModelLoadError, VectorDBError, GPUMemoryError,
    logger,
)

# 全局单例
_RETRIEVER = None


def _jieba_tokenize(text: str):
    """jieba中文分词，过滤空白字符和常见虚词"""
    stop_words = set("，。！？；、的了在是和与我要你他她它")
    return [w for w in jieba.cut(text) if w.strip() and w not in stop_words]


class HybridRetriever:
    """混合检索器：BM25+向量+分数归一加权融合+时间线过滤+rerank(可开关)"""

    def __init__(self):
        """初始化：加载embedding+向量库+BM25索引+rerank(若开启)"""
        logger.info("[HybridRetriever] 初始化中...")

        # 1. 加载embedding模型
        try:
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

        # 2. 加载Chroma向量库
        try:
            self.vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
            count = self.vectordb._collection.count()
            logger.info(f"[HybridRetriever] 向量库加载完成，共 {count} 条记录")
        except Exception as e:
            raise VectorDBError(
                f"向量库读取失败: {CHROMA_PATH}",
                cause=e,
                context={"path": CHROMA_PATH}
            )

        # 3. 构建BM25索引
        try:
            all_data = self.vectordb._collection.get(include=["documents", "metadatas"])
            self.all_chunks = [
                {"page_content": d, "metadata": m}
                for d, m in zip(all_data["documents"], all_data["metadatas"])
            ]
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

        # 4. 加载rerank模型（若开启）
        self.reranker = None
        if RERANK_ENABLED:
            try:
                logger.info(f"[HybridRetriever] 加载 rerank 模型 {RERANKER_MODEL} ...")
                self.reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE, max_length=512)
                logger.info("[HybridRetriever] rerank模型加载完成")
            except Exception as e:
                raise ModelLoadError(
                    f"rerank模型加载失败: {RERANKER_MODEL}",
                    cause=e,
                    context={"model": RERANKER_MODEL, "device": DEVICE}
                )
        else:
            logger.info("[HybridRetriever] rerank已关闭(RERANK_ENABLED=False)，跳过精排节省显存")

        logger.info("[HybridRetriever] 初始化完成")

    # ---------- BM25关键词检索 ----------
    @safe_run(log_msg="BM25检索失败", reraise=True)
    def _bm25_search(self, query: str, k: int):
        """BM25检索：返回Top-K个(chunk, score)"""
        tokens = _jieba_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.all_chunks[i], float(scores[i])) for i in top_idx]

    # ---------- 向量检索 ----------
    @safe_run(log_msg="向量检索失败", reraise=True)
    def _vector_search(self, query: str, k: int):
        """向量检索：返回Top-K个(chunk, score)
        Chroma返回L2距离(越小越相似)，取负转成相似度(越大越好)
        """
        results = self.vectordb.similarity_search_with_score(query, k=k)
        return [
            ({"page_content": d.page_content, "metadata": d.metadata}, float(-s))
            for d, s in results
        ]

    # ---------- 分数归一+加权融合 ----------
    @staticmethod
    def _normalize_and_fuse(vec_results, bm25_results,
                            w_vec=WEIGHT_VECTOR, w_bm25=WEIGHT_BM25):
        """分数归一+加权融合（替代RRF）

        1. 两路分数各自min-max归一化到[0,1]
        2. 加权融合：final = w_vec * vec_norm + w_bm25 * bm25_norm
        3. 用chunk_id去重合并
        """
        # 向量分数归一化
        vec_scores = [s for _, s in vec_results]
        vec_min, vec_max = min(vec_scores), max(vec_scores)
        vec_range = vec_max - vec_min if vec_max > vec_min else 1.0

        # BM25分数归一化
        bm25_scores = [s for _, s in bm25_results]
        bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0

        # 加权融合
        scores = {}
        docs = {}
        for doc, score in vec_results:
            cid = doc["metadata"].get("chunk_id")
            norm = (score - vec_min) / vec_range
            scores[cid] = scores.get(cid, 0) + w_vec * norm
            docs[cid] = doc
        for doc, score in bm25_results:
            cid = doc["metadata"].get("chunk_id")
            norm = (score - bm25_min) / bm25_range
            scores[cid] = scores.get(cid, 0) + w_bm25 * norm
            docs[cid] = doc

        sorted_cids = sorted(scores.keys(), key=lambda x: -scores[x])
        return [(docs[c], scores[c]) for c in sorted_cids]

    # ---------- 时间线过滤 ----------
    @staticmethod
    def _timeline_filter(docs, query, min_keep=MIN_FILTER_KEEP):
        """时间线过滤：按剧情阶段过滤不属于当前剧情时间的片段

        1. 从用户问题中识别剧情阶段关键词（如"史莱克""海神岛"等）
        2. 找到对应chunk_id范围
        3. 过滤掉不在该范围的片段
        4. soft filter：过滤后不足min_keep则回退原结果
        """
        if not docs:
            return docs

        # 识别用户问题中提到的剧情阶段
        matched_phases = []
        for phase_name, (start, end) in TIMELINE_PHASES.items():
            # 阶段名关键词出现在用户问题中
            if phase_name in query:
                matched_phases.append((start, end))
            # 部分匹配：如"史莱克学院"匹配"史莱克"
            elif len(phase_name) >= 3 and phase_name[:2] in query:
                matched_phases.append((start, end))

        if not matched_phases:
            # 早期回忆类问题（怎么认识/初见/小时候等）：默认限定最早剧情阶段
            # 这类问题通常询问早期事件，限定诺丁学院阶段可过滤后期剧情片段的时间错位
            if any(kw in query for kw in EARLY_EVENT_KEYWORDS):
                first_phase = next(iter(TIMELINE_PHASES.values()))
                matched_phases.append(first_phase)
                logger.info(f"[时间线过滤] 早期回忆类问题，默认限定最早阶段：诺丁学院{first_phase}")
            else:
                # 未提到任何剧情阶段，不做过滤
                return docs

        # 过滤：chunk_id在任一匹配阶段范围内
        filtered = []
        for doc, score in docs:
            cid = doc["metadata"].get("chunk_id", -1)
            if cid >= 0 and any(start <= cid <= end for start, end in matched_phases):
                filtered.append((doc, score))

        if len(filtered) < min_keep:
            logger.info(f"[时间线过滤] 过滤后仅{len(filtered)}条<{min_keep}，回退原结果")
            return docs

        logger.info(f"[时间线过滤] 过滤后保留{len(filtered)}/{len(docs)}条（匹配阶段：{[TIMELINE_PHASES.keys() for _ in [1]]}）")
        return filtered

    # ---------- rerank重排 ----------
    @handle_oom
    @safe_run(log_msg="rerank重排失败", reraise=True)
    def _rerank(self, query: str, docs, top_k: int):
        """用CrossEncoder对(query, doc)对精排
        rerank开关关闭时(RERANK_ENABLED=False)直接返回融合结果
        """
        if not docs:
            return []
        if self.reranker is None:
            # rerank关闭，直接取Top-K
            return [(doc, score) for doc, score in docs[:top_k]]

        pairs = [(query, doc["page_content"]) for doc, _ in docs]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
        return [(doc_dict, float(rerank_score)) for (doc_dict, _), rerank_score in ranked[:top_k]]

    # ---------- 主检索入口 ----------
    @safe_run(log_msg="混合检索主流程失败", reraise=True)
    def retrieve(self, query: str, top_k: int = TOP_K_FINAL):
        """混合检索主流程：BM25+向量→分数归一融合→时间线过滤→rerank重排
        返回[(doc_dict, score), ...]，doc_dict含page_content和metadata
        """
        # 步骤1：双路召回
        vec_results = self._vector_search(query, TOP_K_VECTOR)
        bm25_results = self._bm25_search(query, TOP_K_BM25)

        # 步骤2：分数归一+加权融合
        fused = self._normalize_and_fuse(vec_results, bm25_results)
        logger.info(f"[检索] 融合后候选数: {len(fused)}")

        # 步骤3：时间线过滤
        filtered = self._timeline_filter(fused, query)

        # 步骤4：rerank精排（可开关）
        reranked = self._rerank(query, filtered, top_k)
        return reranked


def get_retriever():
    """获取全局HybridRetriever单例（避免重复加载模型）"""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = HybridRetriever()
    return _RETRIEVER


# ==================== 测试入口 ====================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("混合检索模块测试 - 工程化版")
    logger.info("=" * 60)

    retriever = get_retriever()
    test_queries = [
        "唐三的武魂是什么",
        "小舞是谁",
        "史莱克学院在哪里",
    ]

    for q in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"查询: {q}")
        logger.info("=" * 60)
        try:
            results = retriever.retrieve(q, top_k=TOP_K_FINAL)
            for i, (doc, score) in enumerate(results):
                cid = doc["metadata"].get("chunk_id")
                logger.info(f"  Top{i+1} (chunk_id={cid}, score={score:.4f}):")
                logger.info(f"    {doc['page_content'][:100]}...")
        except RetrievalError as e:
            logger.error(f"检索失败: {e}")
