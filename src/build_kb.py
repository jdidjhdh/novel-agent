"""
知识库构建脚本 build_kb.py
=============================
用途：加载百万字小说TXT，切分文本，用Qwen3-Embedding向量化，存入Chroma持久化向量库
存放路径：E:/novel_agent/src/build_kb.py
阶段：1（基础RAG脚本开发）→ 阶段5（工程化与优化改造）

设计要点：
1. 离线模式：在config.py中统一设置 HF_HUB_OFFLINE（import config即生效）
2. 多编码加载：中文小说常见GBK编码，自动尝试 utf-8/gbk/gb18030
3. 中文友好切分：RecursiveCharacterTextSplitter，分隔符按段落>换行>句号优先级
4. 分批向量化：百万字切分后约4000+片段，分批写入控制显存（RTX4060 8G）
5. 持久化复用：向量库已存在且非空时直接加载，不重复向量化
6. 性能优化（阶段5新增）：embedding全局单例缓存复用，多次build_kb不重复加载模型

阶段5工程化改造：
- 配置全部从 config.py 导入，不再硬编码
- 异常处理用 safe_run/handle_oom 装饰器
- embedding 模型加全局单例 _EMBEDDINGS，避免重复加载
- 文件存在性校验，缺失时抛 TextLoadError 友好提示

新版导入（消除langchain-community弃用警告）：
- from langchain_chroma import Chroma
"""
import os
# 离线模式：在 import config 时已经设好（config.py 首行 setdefault）
# 这里再次显式设置是为了兼容：有些场景 build_kb 被独立 import 而非 from config
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import time
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # 新版独立集成包，避免langchain-community弃用警告

# 阶段5：从 config.py 统一导入配置
from config import (
    NOVEL_PATH, CHROMA_PATH, MODEL_CACHE,
    EMBEDDING_MODEL, DEVICE,
    CHUNK_SIZE, CHUNK_OVERLAP, BATCH_SIZE,
)
# 阶段5：从 exceptions.py 导入异常处理工具
from exceptions import (
    safe_run, handle_oom,
    TextLoadError, ModelLoadError, VectorDBError, GPUMemoryError,
    logger,
)


# ==================== 全局单例（性能优化） ====================
# 全局embedding单例：避免多次build_kb重复加载模型（每次加载约30s+1.2GB显存）
_EMBEDDINGS = None


@safe_run(log_msg="文本加载失败", reraise=True)
def load_text(path: str) -> str:
    """加载小说文本，自动尝试多种编码（GBK优先中文小说常见编码）

    异常处理：
    - 文件不存在 → TextLoadError（带路径上下文）
    - 所有编码都失败 → TextLoadError（带尝试过的编码列表）
    - IO错误（如权限拒绝）→ 被safe_run捕获转TextLoadError
    """
    # 1. 文件存在性校验（提前失败，避免尝试编码再报错难定位）
    if not os.path.exists(path):
        logger.error(f"文件不存在: {path}")
        raise TextLoadError(
            f"小说文件不存在: {path}",
            context={"path": path, "exists": False}
        )

    # 2. 中文小说常见编码优先级：utf-8 > gbk > gb18030 > big5
    encodings = ["utf-8", "gbk", "gb18030", "big5"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            logger.info(f"[文本加载] 使用 {enc} 编码成功，共 {len(text)} 字符")
            return text
        except UnicodeDecodeError:
            # 当前编码解码失败，尝试下一个
            continue
    # 所有编码都失败
    raise TextLoadError(
        f"无法用 {encodings} 解码文件：{path}",
        context={"path": path, "tried_encodings": encodings}
    )


def split_text(text: str):
    """切分文本：递归字符切分，分隔符按中文语义友好排序

    参数（CHUNK_SIZE/OVERLAP）从config.py统一管理，支持环境变量覆盖
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 分隔符优先级：空行段落 > 换行 > 中文句号 > 问号 > 叹号 > 分号 > 逗号 > 空格
        # 这样切分会尽量在句号、段落边界断开，保持语义完整
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.create_documents([text])
    # 给每个片段加metadata（序号+来源），方便后续追溯原文位置
    source_name = Path(NOVEL_PATH).name
    for i, chunk in enumerate(chunks):
        chunk.metadata = {"chunk_id": i, "source": source_name}
    logger.info(f"[文本切分] 共切分 {len(chunks)} 个片段，chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    return chunks


def get_embeddings():
    """初始化Qwen3-Embedding模型，本地GPU加载（性能优化：全局单例复用）

    阶段5性能优化：
    - 全局单例 _EMBEDDINGS：多次build_kb调用只加载一次模型
    - 节省每次30s加载时间 + 避免显存碎片化
    - handle_oom装饰器：CUDA OOM时自动清理缓存+友好报错
    """
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        # 已加载过，直接复用单例
        return _EMBEDDINGS

    logger.info(f"[Embedding] 加载 {EMBEDDING_MODEL} 到 {DEVICE} ...")
    t0 = time.time()
    try:
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            cache_folder=MODEL_CACHE,
            model_kwargs={"device": DEVICE},
            # encode_kwargs 传给 sentence-transformers 的 encode 方法
            # 控制编码批大小避免显存溢出
            encode_kwargs={"batch_size": BATCH_SIZE},
        )
        logger.info(f"[Embedding] 加载完成，耗时 {time.time() - t0:.1f}s")
    except RuntimeError as e:
        # CUDA OOM或其他RuntimeError，handle_oom会处理
        raise
    except Exception as e:
        raise ModelLoadError(
            f"Embedding模型加载失败: {EMBEDDING_MODEL}",
            cause=e,
            context={"model": EMBEDDING_MODEL, "device": DEVICE, "cache": MODEL_CACHE}
        )
    return _EMBEDDINGS


@handle_oom
@safe_run(log_msg="向量库构建失败", reraise=True)
def build_or_load(chunks, embeddings):
    """构建向量库（分批向量化）或已存在且非空时直接加载

    阶段5异常处理：
    - Chroma读取失败 → VectorDBError
    - 写入失败（如磁盘满/权限）→ VectorDBError
    - CUDA OOM → handle_oom 装饰器先处理（自动清理缓存+抛GPUMemoryError）

    性能优化：
    - 已存在且非空时跳过重复构建（百万字向量化需数分钟）
    - 分批写入，控制单批显存
    """
    # 检查向量库是否已存在且非空，是则跳过重复构建
    if os.path.exists(CHROMA_PATH):
        try:
            existing = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            count = existing._collection.count()
            if count > 0:
                logger.info(f"[向量库] 已存在且含 {count} 条记录，跳过构建，直接加载")
                return existing
            logger.info("[向量库] 目录存在但为空，重新构建")
        except Exception as e:
            raise VectorDBError(
                f"向量库读取失败: {CHROMA_PATH}",
                cause=e,
                context={"path": CHROMA_PATH}
            )

    logger.info(f"[向量库] 开始构建，分批大小 {BATCH_SIZE}")
    vectordb = None
    total = len(chunks)
    t_start = time.time()
    try:
        for i in range(0, total, BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            if vectordb is None:
                # 第一批：用 from_documents 创建向量库并写入
                vectordb = Chroma.from_documents(batch, embeddings, persist_directory=CHROMA_PATH)
            else:
                # 后续批次：追加写入（新版chromadb指定persist_directory后自动落盘）
                vectordb.add_documents(batch)
            done = min(i + BATCH_SIZE, total)
            # 进度显示，flush=True 立即输出，避免长时间无反馈
            # 用 \r 同行刷新更友好（但日志里换行方便追溯，这里换行）
            logger.info(f"  进度: {done}/{total} ({done * 100 // total}%) 用时 {time.time() - t_start:.0f}s")
        logger.info(f"[向量库] 构建完成，共 {vectordb._collection.count()} 条记录，总用时 {time.time() - t_start:.0f}s")
    except Exception as e:
        # 写入失败（磁盘满/Chroma内部错误）
        raise VectorDBError(
            f"向量库写入失败: {CHROMA_PATH}",
            cause=e,
            context={"path": CHROMA_PATH, "batch_index": i, "total": total}
        )
    return vectordb


def main():
    """主入口：构建/加载知识库并做相似度检索验证"""
    logger.info("=" * 60)
    logger.info("小说知识库构建脚本 - 阶段5工程化版")
    logger.info("=" * 60)

    try:
        # 1. 文本加载
        text = load_text(NOVEL_PATH)

        # 2. 文本切分
        chunks = split_text(text)

        # 3. 初始化Embedding模型（性能优化：全局单例复用）
        embeddings = get_embeddings()

        # 4. 构建/加载向量库（分批向量化，支持持久化复用）
        vectordb = build_or_load(chunks, embeddings)

        # 5. 验证：相似度检索测试
        logger.info("[验证] 相似度检索测试：查询 '唐三的武魂'")
        results = vectordb.similarity_search("唐三的武魂", k=3)
        for i, doc in enumerate(results):
            logger.info(f"  结果{i + 1} (chunk_id={doc.metadata.get('chunk_id')}): {doc.page_content[:80]}...")

        logger.info(f"构建完成！向量库路径: {CHROMA_PATH}")
    except (TextLoadError, ModelLoadError, VectorDBError, GPUMemoryError) as e:
        # 已是自定义异常，友好提示
        logger.error(f"构建失败: {e}")
        logger.error(f"排查建议: 检查文件路径/模型缓存/Ollama服务/显存占用")
        raise


if __name__ == "__main__":
    main()