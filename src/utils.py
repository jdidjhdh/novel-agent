"""
通用工具模块 utils.py
=============================
用途：统一日志、自定义异常体系、异常装饰器、上下文长度限制
存放路径：E:/novel_agent/src/utils.py

合并说明：本文件整合了原 exceptions.py 的全部功能，并新增 truncate_context 上下文长度限制工具。
其他模块统一 from utils import ... 即可，不再需要单独 import exceptions。

异常分层：
  RagError (基类)
    +-- TextLoadError       文本读取失败（编码/不存在/IO）
    +-- ModelLoadError      模型加载失败（HF下载/加载/超时）
    +-- GPUMemoryError      显存溢出（CUDA OOM）
    +-- VectorDBError       向量库异常（Chroma读写/版本不兼容）
    +-- RetrievalError      检索异常（BM25/向量/rerank）
    +-- LLMInvokeError      LLM调用异常（Ollama未启动/超时）
"""

import functools
import logging
import logging.handlers
import os
import traceback
from pathlib import Path

# ==================== 日志配置 ====================
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "novel_agent.log"

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=_FORMAT,
    handlers=[
        logging.handlers.RotatingFileHandler(
            _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("novel_agent")


# ==================== 自定义异常分层 ====================
class RagError(Exception):
    """项目所有自定义异常的基类，携带 cause + context 便于排查"""
    def __init__(self, message: str, cause: Exception = None, context: dict = None):
        super().__init__(message)
        self.message = message
        self.cause = cause
        self.context = context or {}
        log_msg = f"{message}"
        if context:
            log_msg += f" | context={context}"
        if cause:
            log_msg += f" | cause={type(cause).__name__}: {cause}"
        logger.error(log_msg)
        logger.debug(traceback.format_exc())

    def __str__(self):
        return self.message


class TextLoadError(RagError):
    """文本加载异常：编码失败/文件不存在/读取IO错误"""


class ModelLoadError(RagError):
    """模型加载异常：HuggingFace模型下载/加载失败/超时"""


class GPUMemoryError(RagError):
    """显存溢出：embedding/rerank向量化时CUDA OOM"""


class VectorDBError(RagError):
    """向量库异常：Chroma读取失败/写入失败/版本不兼容"""


class RetrievalError(RagError):
    """检索异常：BM25/向量/rerank过程中出错"""


class LLMInvokeError(RagError):
    """LLM调用异常：Ollama未启动/超时/响应异常"""


# ==================== safe_run 装饰器 ====================
def safe_run(default=None, log_msg="操作失败", reraise=True):
    """函数装饰器：自动捕获异常，转成RagError，记录日志

    根据函数名自动推断异常类型：
        load_text/load   -> TextLoadError
        embed/rerank     -> ModelLoadError 或 GPUMemoryError
        vectordb/chroma  -> VectorDBError
        retrieve/search  -> RetrievalError
        llm/invoke/chat  -> LLMInvokeError
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RagError:
                raise
            except Exception as e:
                func_name = func.__name__
                context = {"function": func_name, "args_len": len(args)}
                logger.error(f"{log_msg} @ {func_name}: {type(e).__name__}: {e}")
                logger.debug(traceback.format_exc())
                if reraise:
                    err_msg = str(e).lower()
                    if "load_text" in func_name or ("load" in func_name and "model" not in func_name):
                        raise TextLoadError(log_msg, cause=e, context=context)
                    if "embed" in func_name or "rerank" in func_name:
                        if "cuda" in err_msg and ("out of memory" in err_msg or "oom" in err_msg):
                            raise GPUMemoryError(log_msg, cause=e, context=context)
                        raise ModelLoadError(log_msg, cause=e, context=context)
                    if "vectordb" in func_name or "chroma" in func_name.lower():
                        raise VectorDBError(log_msg, cause=e, context=context)
                    if "retrieve" in func_name or "search" in func_name:
                        raise RetrievalError(log_msg, cause=e, context=context)
                    if "llm" in func_name or "invoke" in func_name or "chat" in func_name:
                        raise LLMInvokeError(log_msg, cause=e, context=context)
                    raise RagError(log_msg, cause=e, context=context)
                return default
        return wrapper
    return decorator


# ==================== handle_oom 装饰器 ====================
def handle_oom(func):
    """专门捕获CUDA OOM的装饰器：清理缓存后抛GPUMemoryError"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            msg = str(e).lower()
            if "cuda" in msg and ("out of memory" in msg or "oom" in msg):
                try:
                    import torch
                    torch.cuda.empty_cache()
                    logger.warning("GPU显存溢出，已调用torch.cuda.empty_cache()清理缓存")
                except ImportError:
                    pass
                raise GPUMemoryError(
                    "GPU显存溢出，请减小BATCH_SIZE或换更小模型",
                    cause=e,
                    context={"function": func.__name__}
                )
            raise
    return wrapper


# ==================== 上下文长度限制工具 ====================
def truncate_context(documents, max_chars: int = 6000):
    """限制送入LLM的上下文总字符数，防止token溢出

    从前到后累加每个片段的字符数，超过max_chars就截断。
    保留排序靠前的片段（已经是rerank精排后的顺序）。

    Args:
        documents: Document列表或dict列表（含page_content）
        max_chars: 上下文最大字符数

    Returns:
        (截断后的文档列表, 实际总字符数)
    """
    result = []
    total = 0
    for doc in documents:
        content = doc.page_content if hasattr(doc, "page_content") else doc.get("page_content", "")
        if total + len(content) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                if hasattr(doc, "page_content"):
                    from langchain_core.documents import Document
                    truncated = Document(page_content=content[:remaining], metadata=doc.metadata)
                    result.append(truncated)
                else:
                    doc_copy = dict(doc)
                    doc_copy["page_content"] = content[:remaining]
                    result.append(doc_copy)
            logger.info(f"[上下文截断] 总字符数达 {total}，已截断")
            break
        result.append(doc)
        total += len(content)

    logger.info(f"[上下文限制] 保留 {len(result)} 个片段，总字符数 {total}/{max_chars}")
    return result, total
