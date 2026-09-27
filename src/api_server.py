"""
FastAPI 后端服务 api_server.py
=============================
用途：阶段4 → 阶段5工程化改造——封装RAG智能体能力，对外提供REST API接口
存放路径：E:/novel_agent/src/api_server.py
阶段：4（前后端封装）→ 阶段5（工程化与优化）

接口列表：
  GET  /api/status  — 查询知识库状态（记录数、Ollama模型列表、显存占用）
  POST /api/chat    — 角色扮演对话（question/role/thread_id → answer）
  POST /api/upload  — 上传小说TXT文本
  POST /api/build   — 触发知识库构建（分批向量化）

启动方式：
  E:/novel_agent/venv/Scripts/python.exe E:/novel_agent/src/api_server.py
  或 uvicorn api_server:app --host 0.0.0.0 --port 8000

阶段5工程化改造：
- 配置全部从 config.py 导入（路径/端口/Ollama URL）
- 自定义异常分类响应：TextLoadError 400 / ModelLoadError 503 / GPUMemoryError 507 / VectorDBError 500
- /api/status 新增 GPU显存占用查询（nvidia-smi）
- 上传接口加文件大小+类型校验

面试口述版：
  "阶段5我把FastAPI接口按异常类型返回不同HTTP状态码：
  TextLoadError→400(客户端错误，文件问题)
  ModelLoadError→503(服务不可用，模型未就绪)
  GPUMemoryError→507(资源不足，显存溢出)
  VectorDBError→500(服务端错误，向量库异常)
  /api/status增加GPU显存查询，前端可显示当前显存占用。"
"""
import os
import sys

# 离线模式：config.py已setdefault，这里再设一次兼容独立import
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 把src目录加入path，方便import同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import requests as http_requests

# 阶段5：从 config.py 统一导入配置
from config import (
    NOVEL_DIR, CHROMA_PATH, OLLAMA_URL, HOST, PORT,
    OLLAMA_CHECK_TIMEOUT,
)
# 阶段5：从 exceptions.py 导入异常类型（用于按异常分类返回HTTP状态码）
from exceptions import (
    TextLoadError, ModelLoadError, GPUMemoryError, VectorDBError,
    RetrievalError, LLMInvokeError,
    logger,
)

# ==================== FastAPI应用 ====================
app = FastAPI(
    title="小说RAG智能体API",
    description="百万字小说知识库 + 角色扮演对话智能体",
    version="1.1.0",
)

# CORS：允许Streamlit前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 请求/响应模型 ====================
class ChatRequest(BaseModel):
    """对话请求"""
    question: str          # 用户问题
    role: str = "唐三"     # 角色设定
    thread_id: str = "1"   # 会话ID（同ID保持对话记忆）


class ChatResponse(BaseModel):
    """对话响应"""
    answer: str            # 回答内容
    role: str              # 角色名
    retrieval_count: int = 0  # 检索轮次


class BuildRequest(BaseModel):
    """知识库构建请求"""
    filepath: str = None   # 小说文件路径（None用默认路径）


# ==================== 异常到HTTP状态码映射 ====================
# 阶段5：按异常类型返回不同HTTP状态码，前端可针对性处理
def exception_to_status(exc: Exception) -> int:
    """自定义异常→HTTP状态码映射
    - TextLoadError     → 400 客户端错误（文件问题）
    - ModelLoadError    → 503 服务不可用（模型未就绪）
    - GPUMemoryError    → 507 资源不足（显存溢出）
    - VectorDBError     → 500 服务端错误
    - RetrievalError    → 500 服务端错误
    - LLMInvokeError    → 503 服务不可用（Ollama未启动）
    - 其他              → 500
    """
    if isinstance(exc, TextLoadError):
        return 400
    if isinstance(exc, ModelLoadError):
        return 503
    if isinstance(exc, GPUMemoryError):
        return 507
    if isinstance(exc, LLMInvokeError):
        return 503
    if isinstance(exc, VectorDBError):
        return 500
    if isinstance(exc, RetrievalError):
        return 500
    return 500


# ==================== 接口定义 ====================

@app.get("/api/status")
def get_status():
    """查询知识库和Ollama状态
    阶段5新增：GPU显存占用查询（nvidia-smi）
    返回：kb_exists, record_count, ollama_ok, models, gpu_memory
    """
    # 1. 检查知识库
    kb_exists = os.path.exists(CHROMA_PATH)
    record_count = 0
    if kb_exists:
        try:
            import sqlite3
            db_path = os.path.join(CHROMA_PATH, "chroma.sqlite3")
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            record_count = cursor.fetchone()[0]
            conn.close()
        except Exception as e:
            logger.warning(f"读取向量库记录数失败: {e}")

    # 2. 检查Ollama服务
    ollama_ok = False
    models = []
    try:
        resp = http_requests.get(f"{OLLAMA_URL}/api/tags", timeout=OLLAMA_CHECK_TIMEOUT)
        if resp.status_code == 200:
            ollama_ok = True
            models = [m["name"] for m in resp.json().get("models", [])]
    except Exception as e:
        logger.warning(f"Ollama服务不可达: {e}")

    # 3. 查询GPU显存占用（阶段5新增）
    gpu_memory = None
    try:
        import subprocess
        # nvidia-smi查询显存使用，输出格式：used,total
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(", ")
            gpu_memory = {
                "used_mb": int(parts[0]),
                "total_mb": int(parts[1]),
                "percent": round(int(parts[0]) / int(parts[1]) * 100, 1)
            }
    except Exception:
        # 没有NVIDIA GPU或nvidia-smi不可用，忽略
        pass

    return {
        "kb_exists": kb_exists,
        "record_count": record_count,
        "ollama_ok": ollama_ok,
        "models": models,
        "gpu_memory": gpu_memory,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """角色扮演对话接口
    对接LangGraph智能体（agent_graph.chat），支持多轮检索+对话记忆+角色扮演
    注意：首次调用需加载模型（约30s），后续调用快

    阶段5异常处理：捕获自定义异常，按类型返回不同HTTP状态码
    """
    try:
        from agent_graph import chat as agent_chat
        answer = agent_chat(req.question, thread_id=req.thread_id, role=req.role)
        return ChatResponse(answer=answer, role=req.role)
    except (TextLoadError, ModelLoadError, GPUMemoryError, VectorDBError,
            RetrievalError, LLMInvokeError) as e:
        # 已知异常，按类型返回HTTP状态码
        status = exception_to_status(e)
        raise HTTPException(status_code=status, detail=f"{type(e).__name__}: {e.message}")
    except Exception as e:
        logger.error(f"/api/chat 未知异常: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"智能体调用失败: {str(e)}")


@app.post("/api/upload")
async def upload_novel(file: UploadFile = File(...)):
    """上传小说TXT文本
    保存到 NOVEL_DIR 目录，返回文件路径供 /api/build 使用

    阶段5异常处理：
    - 文件名非.txt → 400
    - 文件过大(>200MB) → 413
    - IO失败 → 500
    """
    # 1. 文件类型校验
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="仅支持.txt文件")

    # 2. 读取内容（限制200MB，防止超大文件OOM）
    content = await file.read()
    max_size = 200 * 1024 * 1024  # 200MB
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail=f"文件过大(>{max_size//1024//1024}MB)，请控制小说长度")

    # 3. 保存
    try:
        os.makedirs(NOVEL_DIR, exist_ok=True)
        save_path = os.path.join(NOVEL_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(content)
        logger.info(f"[上传] 文件已保存: {save_path} ({len(content)//1024}KB)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    return {
        "filename": file.filename,
        "size": len(content),
        "path": save_path,
        "message": f"文件已保存到 {save_path}",
    }


@app.post("/api/build")
def build_knowledge_base(req: BuildRequest = None):
    """触发知识库构建
    调用build_kb的函数：加载文本→切分→向量化→存入Chroma
    支持自定义文件路径，默认用build_kb中的NOVEL_PATH

    阶段5异常处理：捕获自定义异常，按类型返回HTTP状态码
    - TextLoadError → 400（文件路径错误/编码失败）
    - GPUMemoryError → 507（显存溢出，建议减小BATCH_SIZE）
    - ModelLoadError → 503（模型加载失败）
    - VectorDBError → 500（向量库读写失败）
    """
    try:
        import build_kb

        # 确定文件路径
        filepath = req.filepath if req and req.filepath else build_kb.NOVEL_PATH

        # 分步构建（便于错误定位）
        # 步骤1：加载文本
        text = build_kb.load_text(filepath)

        # 步骤2：切分
        chunks = build_kb.split_text(text)

        # 步骤3：向量化并存入Chroma（分批，支持已存在时跳过）
        embeddings = build_kb.get_embeddings()
        vectordb = build_kb.build_or_load(chunks, embeddings)

        count = vectordb._collection.count()
        logger.info(f"[构建] 知识库构建完成，共{count}条记录")
        return {
            "status": "success",
            "record_count": count,
            "message": f"知识库构建完成，共{count}条记录",
        }
    except (TextLoadError, ModelLoadError, GPUMemoryError, VectorDBError) as e:
        status = exception_to_status(e)
        # GPUMemoryError给特别详细的修复建议
        detail = f"{type(e).__name__}: {e.message}"
        if isinstance(e, GPUMemoryError):
            detail += "。建议：减小config.py中的BATCH_SIZE（如改为16或8），或换更小的embedding模型"
        raise HTTPException(status_code=status, detail=detail)
    except Exception as e:
        logger.error(f"/api/build 未知异常: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"知识库构建失败: {str(e)}")


@app.get("/")
def root():
    """根路径：API信息"""
    return {
        "name": "小说RAG智能体API",
        "version": "1.1.0",
        "docs": "/docs",
        "endpoints": {
            "GET /api/status": "查询知识库状态+GPU显存",
            "POST /api/chat": "角色扮演对话",
            "POST /api/upload": "上传小说TXT",
            "POST /api/build": "构建知识库",
        },
    }


# ==================== 启动 ====================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("小说RAG智能体 FastAPI 后端服务 - 阶段5工程化版")
    logger.info(f"地址: http://{HOST}:{PORT}")
    logger.info(f"API文档: http://{HOST}:{PORT}/docs")
    logger.info("=" * 60)
    uvicorn.run(app, host=HOST, port=PORT)