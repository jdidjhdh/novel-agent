"""
知识库构建脚本 build_knowledge_base.py
======================================
用途：加载百万字小说TXT，切分文本，用Qwen3-Embedding向量化，存入Chroma持久化向量库
存放路径：E:/novel_agent/src/build_knowledge_base.py

核心功能：
1. 多编码加载：中文小说常见GBK编码，自动尝试 utf-8/gbk/gb18030/big5
2. 中文友好切分：RecursiveCharacterTextSplitter，分隔符按段落>换行>句号优先级
3. 分批向量化：百万字切分后约7700+片段，分批写入控制显存（RTX4060 8G）
4. 断点续建：记录已构建chunk_id到进度文件，中断后重启跳过已构建部分
5. 持久化复用：向量库已存在且非空时直接加载，不重复向量化

断点续建原理：
  每批写入后，把当前进度（已处理的chunk_id范围）写入 build_progress.json
  重启时读取该文件，从断点处继续，避免从头重建（百万字向量化需数分钟）
"""

import os
# 离线模式
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import time
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    NOVEL_PATH, CHROMA_PATH, MODEL_CACHE, BUILD_PROGRESS_FILE,
    EMBEDDING_MODEL, DEVICE,
    CHUNK_SIZE, CHUNK_OVERLAP, BATCH_SIZE,
)
from utils import (
    safe_run, handle_oom,
    TextLoadError, ModelLoadError, VectorDBError,
    logger,
)


# ==================== 全局单例 ====================
_EMBEDDINGS = None


@safe_run(log_msg="文本加载失败", reraise=True)
def load_text(path: str) -> str:
    """加载小说文本，自动尝试多种编码（GBK优先中文小说常见编码）"""
    if not os.path.exists(path):
        logger.error(f"文件不存在: {path}")
        raise TextLoadError(
            f"小说文件不存在: {path}",
            context={"path": path, "exists": False}
        )

    encodings = ["utf-8", "gbk", "gb18030", "big5"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            logger.info(f"[文本加载] 使用 {enc} 编码成功，共 {len(text)} 字符")
            return text
        except UnicodeDecodeError:
            continue
    raise TextLoadError(
        f"无法用 {encodings} 解码文件：{path}",
        context={"path": path, "tried_encodings": encodings}
    )


def split_text(text: str):
    """切分文本：递归字符切分，分隔符按中文语义友好排序"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.create_documents([text])
    source_name = Path(NOVEL_PATH).name
    for i, chunk in enumerate(chunks):
        chunk.metadata = {"chunk_id": i, "source": source_name}
    logger.info(f"[文本切分] 共切分 {len(chunks)} 个片段，chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    return chunks


def get_embeddings():
    """初始化Qwen3-Embedding模型，全局单例复用"""
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS

    logger.info(f"[Embedding] 加载 {EMBEDDING_MODEL} 到 {DEVICE} ...")
    t0 = time.time()
    try:
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            cache_folder=MODEL_CACHE,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"batch_size": BATCH_SIZE},
        )
        logger.info(f"[Embedding] 加载完成，耗时 {time.time() - t0:.1f}s")
    except Exception as e:
        raise ModelLoadError(
            f"Embedding模型加载失败: {EMBEDDING_MODEL}",
            cause=e,
            context={"model": EMBEDDING_MODEL, "device": DEVICE, "cache": MODEL_CACHE}
        )
    return _EMBEDDINGS


# ==================== 断点续建工具 ====================
def _load_progress():
    """读取断点续建进度文件，返回已构建的chunk_id上限"""
    if not os.path.exists(BUILD_PROGRESS_FILE):
        return 0
    try:
        with open(BUILD_PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("completed_up_to", 0)
    except Exception as e:
        logger.warning(f"读取进度文件失败，从头开始: {e}")
        return 0


def _save_progress(completed_up_to: int):
    """保存断点续建进度（已构建到第几个chunk）"""
    try:
        os.makedirs(os.path.dirname(BUILD_PROGRESS_FILE), exist_ok=True)
        with open(BUILD_PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump({"completed_up_to": completed_up_to}, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"保存进度文件失败: {e}")


@handle_oom
@safe_run(log_msg="向量库构建失败", reraise=True)
def build_or_load(chunks, embeddings):
    """构建向量库（分批向量化，支持断点续建）或已存在且非空时直接加载

    断点续建逻辑：
    1. 向量库已存在且非空 + 进度文件显示全部完成 -> 直接加载
    2. 向量库已存在但进度未完成 -> 从断点继续追加写入
    3. 向量库不存在 -> 从头构建
    """
    total = len(chunks)

    # 场景1：向量库已存在，检查是否已完成
    if os.path.exists(CHROMA_PATH):
        try:
            existing = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            count = existing._collection.count()
            if count >= total:
                logger.info(f"[向量库] 已存在且含 {count} 条记录（全部完成），直接加载")
                _save_progress(total)  # 确保进度文件一致
                return existing
            logger.info(f"[向量库] 已存在 {count}/{total} 条，尝试断点续建")
        except Exception as e:
            raise VectorDBError(
                f"向量库读取失败: {CHROMA_PATH}",
                cause=e,
                context={"path": CHROMA_PATH}
            )

    # 读取断点续建进度
    completed_up_to = _load_progress()
    if completed_up_to > 0:
        logger.info(f"[断点续建] 从第 {completed_up_to} 个chunk继续（已完成 {completed_up_to}/{total}）")

    # 场景2/3：从断点处继续构建
    logger.info(f"[向量库] 开始构建，分批大小 {BATCH_SIZE}")
    vectordb = None
    t_start = time.time()

    try:
        # 如果已有部分数据，先加载已有向量库
        if completed_up_to > 0 and os.path.exists(CHROMA_PATH):
            vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        elif completed_up_to > 0:
            # 进度文件有记录但向量库目录被删了，从头开始
            logger.warning("[断点续建] 进度文件有记录但向量库不存在，从头开始")
            completed_up_to = 0

        # 分批向量化
        start_idx = completed_up_to
        for i in range(start_idx, total, BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            if vectordb is None:
                vectordb = Chroma.from_documents(batch, embeddings, persist_directory=CHROMA_PATH)
            else:
                vectordb.add_documents(batch)

            done = min(i + BATCH_SIZE, total)
            # 更新进度文件（断点续建关键）
            _save_progress(done)
            logger.info(f"  进度: {done}/{total} ({done * 100 // total}%) 用时 {time.time() - t_start:.0f}s")

        logger.info(f"[向量库] 构建完成，共 {vectordb._collection.count()} 条记录，总用时 {time.time() - t_start:.0f}s")
        # 构建完成后确保进度文件标记为全部完成
        _save_progress(total)
    except Exception as e:
        raise VectorDBError(
            f"向量库写入失败: {CHROMA_PATH}",
            cause=e,
            context={"path": CHROMA_PATH, "batch_index": i, "total": total}
        )
    return vectordb


def main():
    """主入口：构建/加载知识库并做相似度检索验证"""
    logger.info("=" * 60)
    logger.info("小说知识库构建脚本 - 工程化版（支持断点续建）")
    logger.info("=" * 60)

    try:
        text = load_text(NOVEL_PATH)
        chunks = split_text(text)
        embeddings = get_embeddings()
        vectordb = build_or_load(chunks, embeddings)

        # 验证：相似度检索测试
        logger.info("[验证] 相似度检索测试：查询 '唐三的武魂'")
        results = vectordb.similarity_search("唐三的武魂", k=3)
        for i, doc in enumerate(results):
            logger.info(f"  结果{i + 1} (chunk_id={doc.metadata.get('chunk_id')}): {doc.page_content[:80]}...")

        logger.info(f"构建完成！向量库路径: {CHROMA_PATH}")
    except (TextLoadError, ModelLoadError, VectorDBError) as e:
        logger.error(f"构建失败: {e}")
        logger.error(f"排查建议: 检查文件路径/模型缓存/Ollama服务/显存占用")
        raise


if __name__ == "__main__":
    main()
