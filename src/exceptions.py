"""
异常处理模块 exceptions.py
=============================
用途：阶段5——定义项目自定义异常 + 通用异常处理工具 + 统一日志
存放路径：E:/novel_agent/src/exceptions.py
阶段：5（工程化与优化）

设计原则：
1. 自定义异常分层：RagError基类 → 各具体异常子类
2. 友好错误信息：携带原异常cause + 上下文context，便于排查
3. 通用工具：safe_run装饰器自动捕获+日志，避免业务代码满屏try/except
4. handle_oom装饰器专门处理CUDA显存溢出，自动清理缓存
5. 统一日志：RotatingFileHandler写文件 + StreamHandler输出控制台

异常分层：
  RagError (基类)
    ├── TextLoadError       文本读取失败（编码/不存在/IO）
    ├── ModelLoadError      模型加载失败（HF下载/加载/超时）
    ├── GPUMemoryError      显存溢出（CUDA OOM）
    ├── VectorDBError       向量库异常（Chroma读写/版本不兼容）
    ├── RetrievalError      检索异常（BM25/向量/rerank）
    └── LLMInvokeError      LLM调用异常（Ollama未启动/超时）

面试口述版：
  "我设计了分层异常体系：基类RagError统一捕获，子类按场景细分
  （TextLoadError文本读取、ModelLoadError模型加载、GPUMemoryError显存溢出、
  VectorDBError向量库、RetrievalError检索、LLMInvokeError LLM调用）。
  safe_run装饰器根据函数名自动把底层异常映射成对应自定义异常，
  避免业务代码里写满try/except。handle_oom专门捕获CUDA OOM，
  自动清理缓存后抛GPUMemoryError，提醒用户减小batch_size。"
"""

import functools
import logging
import logging.handlers
import os
import traceback
from pathlib import Path

# ==================== 日志配置 ====================
# 日志目录：项目根/logs（首次import时创建）
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "novel_agent.log"

# 日志格式：时间 | 级别 | 模块 | 消息
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=_FORMAT,
    handlers=[
        # RotatingFileHandler：单文件5MB，最多保留3个历史
        logging.handlers.RotatingFileHandler(
            _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("novel_agent")


# ==================== 自定义异常分层 ====================
class RagError(Exception):
    """项目所有自定义异常的基类
    所有子类异常都带 message + cause(原异常) + context(场景信息)
    构造时自动写入日志，方便排查
    """
    def __init__(self, message: str, cause: Exception = None, context: dict = None):
        super().__init__(message)
        self.message = message
        self.cause = cause
        self.context = context or {}
        # 记录到日志
        log_msg = f"{message}"
        if context:
            log_msg += f" | context={context}"
        if cause:
            log_msg += f" | cause={type(cause).__name__}: {cause}"
        logger.error(log_msg)
        logger.debug(traceback.format_exc())

    def __str__(self):
        return self.message


# ---- 文本相关 ----
class TextLoadError(RagError):
    """文本加载异常：编码失败/文件不存在/读取IO错误"""


# ---- 模型相关 ----
class ModelLoadError(RagError):
    """模型加载异常：HuggingFace模型下载/加载失败/超时"""


class GPUMemoryError(RagError):
    """显存溢出：embedding/rerank向量化时CUDA OOM"""


# ---- 向量库相关 ----
class VectorDBError(RagError):
    """向量库异常：Chroma读取失败/写入失败/版本不兼容"""


# ---- 检索相关 ----
class RetrievalError(RagError):
    """检索异常：BM25/向量/rerank过程中出错"""


# ---- LLM相关 ----
class LLMInvokeError(RagError):
    """LLM调用异常：Ollama未启动/超时/响应异常"""


# ==================== 通用工具：safe_run装饰器 ====================
def safe_run(default=None, log_msg="操作失败", reraise=True):
    """函数装饰器：自动捕获异常，转成RagError，记录日志

    Args:
        default: 异常时返回的默认值（None表示抛出）
        log_msg: 失败时的日志前缀（描述操作用途）
        reraise: True=包装成RagError重新抛出；False=返回default不抛

    根据函数名自动推断异常类型：
        load_text/load        → TextLoadError
        embed/rerank          → ModelLoadError 或 GPUMemoryError（含CUDA关键字）
        vectordb/chroma       → VectorDBError
        retrieve/search       → RetrievalError
        llm/invoke/chat       → LLMInvokeError
        其他                  → RagError

    使用方式：
        @safe_run(default="", log_msg="检索失败", reraise=False)
        def retrieve(...): ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RagError:
                # 已是自定义异常，直接传播（避免重复包装）
                raise
            except Exception as e:
                func_name = func.__name__
                context = {"function": func_name, "args_len": len(args)}
                logger.error(f"{log_msg} @ {func_name}: {type(e).__name__}: {e}")
                logger.debug(traceback.format_exc())
                if reraise:
                    # 根据函数名推断异常类型
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


# ==================== 专门处理CUDA OOM的装饰器 ====================
def handle_oom(func):
    """专门捕获CUDA OOM的装饰器：清理缓存后抛GPUMemoryError
    用途：包裹向量化/rerank函数，自动处理显存溢出
    流程：
        1. 捕获RuntimeError
        2. 判断是否CUDA OOM（消息含"cuda"和"out of memory"）
        3. 是→torch.cuda.empty_cache()清理缓存 + 抛GPUMemoryError
        4. 否→原样抛出
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            msg = str(e).lower()
            if "cuda" in msg and ("out of memory" in msg or "oom" in msg):
                # 尝试清理GPU缓存，避免后续调用继续失败
                try:
                    import torch
                    torch.cuda.empty_cache()
                    logger.warning("GPU显存溢出，已调用torch.cuda.empty_cache()清理缓存")
                except ImportError:
                    pass
                raise GPUMemoryError(
                    "GPU显存溢出，请减小BATCH_SIZE或换更小模型（当前DEVICE=cuda）",
                    cause=e,
                    context={"function": func.__name__}
                )
            raise
    return wrapper