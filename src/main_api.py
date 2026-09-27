"""
FastAPI 后端服务 main_api.py
=============================
用途：封装RAG智能体能力，对外提供REST API接口
存放路径：E:/novel_agent/src/main_api.py

接口列表：
  GET  /api/status  — 查询知识库状态（记录数、Ollama模型、GPU显存）
  POST /api/chat    — 角色扮演对话（question/role/thread_id -> answer）
  POST /api/upload  — 上传小说TXT文本
  POST /api/build   — 触发知识库构建（断点续建）

启动方式：
  E:/novel_agent/venv/Scripts/python.exe E:/novel_agent/src/main_api.py
  或 uvicorn main_api:app --host 0.0.0.0 --port 8000

异常分类响应：
  TextLoadError  -> 400 客户端错误
  ModelLoadError -> 503 服务不可用
  GPUMemoryError -> 507 资源不足
  VectorDBError  -> 500 服务端错误
  LLMInvokeError -> 503 Ollama未启动
"""

import os
import sys

# 离线模式
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import requests as http_requests

from config import (
    NOVEL_DIR, CHROMA_PATH, OLLAMA_URL, HOST, PORT,
    OLLAMA_CHECK_TIMEOUT,
)
from utils import (
    TextLoadError, ModelLoadError, GPUMemoryError, VectorDBError,
    RetrievalError, LLMInvokeError,
    logger,
)

# ==================== FastAPI应用 ====================
app = FastAPI(
    title="小说RAG智能体API",
    description="百万字小说知识库 + 角色扮演对话智能体",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 请求/响应模型 ====================
class ChatRequest(BaseModel):
    question: str
    role: str = "唐三"
    thread_id: str = "1"


class ChatResponse(BaseModel):
    answer: str
    role: str
    retrieval_count: int = 0
    citations: list = []  # 引用的原文片段[{chunk_id, content}]


class BuildRequest(BaseModel):
    filepath: str = None


# ==================== 异常到HTTP状态码映射 ====================
def exception_to_status(exc: Exception) -> int:
    """自定义异常 -> HTTP状态码"""
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
    """查询知识库和Ollama状态，含GPU显存占用"""
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

    ollama_ok = False
    models = []
    try:
        resp = http_requests.get(f"{OLLAMA_URL}/api/tags", timeout=OLLAMA_CHECK_TIMEOUT)
        if resp.status_code == 200:
            ollama_ok = True
            models = [m["name"] for m in resp.json().get("models", [])]
    except Exception as e:
        logger.warning(f"Ollama服务不可达: {e}")

    gpu_memory = None
    try:
        import subprocess
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
    """角色扮演对话接口，对接LangGraph智能体"""
    try:
        from agent_graph import chat as agent_chat
        answer, citations = agent_chat(req.question, thread_id=req.thread_id, role=req.role)
        return ChatResponse(answer=answer, role=req.role, citations=citations)
    except (TextLoadError, ModelLoadError, GPUMemoryError, VectorDBError,
            RetrievalError, LLMInvokeError) as e:
        status = exception_to_status(e)
        raise HTTPException(status_code=status, detail=f"{type(e).__name__}: {e.message}")
    except Exception as e:
        logger.error(f"/api/chat 未知异常: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"智能体调用失败: {str(e)}")


@app.post("/api/upload")
async def upload_novel(file: UploadFile = File(...)):
    """上传小说TXT文本"""
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="仅支持.txt文件")

    content = await file.read()
    max_size = 200 * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail=f"文件过大(>{max_size//1024//1024}MB)")

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
    """触发知识库构建（支持断点续建）"""
    try:
        import build_knowledge_base as bkb

        filepath = req.filepath if req and req.filepath else bkb.NOVEL_PATH
        text = bkb.load_text(filepath)
        chunks = bkb.split_text(text)
        embeddings = bkb.get_embeddings()
        vectordb = bkb.build_or_load(chunks, embeddings)

        count = vectordb._collection.count()
        logger.info(f"[构建] 知识库构建完成，共{count}条记录")
        return {
            "status": "success",
            "record_count": count,
            "message": f"知识库构建完成，共{count}条记录",
        }
    except (TextLoadError, ModelLoadError, GPUMemoryError, VectorDBError) as e:
        status = exception_to_status(e)
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
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "GET /api/status": "查询知识库状态+GPU显存",
            "POST /api/chat": "角色扮演对话",
            "POST /api/upload": "上传小说TXT",
            "POST /api/build": "构建知识库（断点续建）",
        },
    }


# ==================== 启动 ====================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("小说RAG智能体 FastAPI 后端服务 - 工程化版")
    logger.info(f"地址: http://{HOST}:{PORT}")
    logger.info(f"API文档: http://{HOST}:{PORT}/docs")
    logger.info("=" * 60)
    uvicorn.run(app, host=HOST, port=PORT)
