"""
Streamlit Web前端 app.py
=============================
用途：阶段4 → 阶段5工程化改造——Web前端页面
- 角色选择 + 多轮对话 + 消息历史
- 知识库管理：状态查看 + TXT上传 + 一键构建
- 对话记忆：同thread_id保持上下文

存放路径：E:/novel_agent/src/app.py
阶段：4（前后端封装）→ 阶段5（工程化与优化）

启动方式（先启动FastAPI后端，再启动Streamlit）：
  终端1：E:/novel_agent/venv/Scripts/python.exe E:/novel_agent/src/api_server.py
  终端2：E:/novel_agent/venv/Scripts/python.exe -m streamlit run E:/novel_agent/src/app.py

阶段5工程化改造：
- 配置全部从 config.py 导入（API_URL/超时/角色列表）
- 上传错误按HTTP状态码分类显示（400文件问题/413过大/507显存溢出）
- 知识库状态新增GPU显存占用显示
- 多阶段进度提示（分析问题→检索原文→生成回答）
"""
import streamlit as st
import requests
import os

# 阶段5：从 config.py 统一导入配置
from config import (
    FRONT_API_URL as API_URL,
    FRONT_TIMEOUT_CHAT as TIMEOUT_CHAT,
    FRONT_TIMEOUT_BUILD as TIMEOUT_BUILD,
    ROLES,
)
from exceptions import logger

# ==================== Session State 初始化 ====================
if "messages" not in st.session_state:
    st.session_state.messages = []      # 对话历史 [{role, content, role_name}]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "1"    # 会话ID
if "selected_role" not in st.session_state:
    st.session_state.selected_role = ROLES[0]


def check_api():
    """检查后端API是否在线"""
    try:
        resp = requests.get(f"{API_URL}/api/status", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def call_chat(question, role, thread_id):
    """调用后端对话接口
    阶段5异常处理：按HTTP状态码返回友好错误
    """
    try:
        resp = requests.post(
            f"{API_URL}/api/chat",
            json={"question": question, "role": role, "thread_id": thread_id},
            timeout=TIMEOUT_CHAT,
        )
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", "未知错误")
            except Exception:
                detail = resp.text
            # 按状态码分类提示
            if resp.status_code == 503:
                st.error(f"❌ 服务未就绪：{detail}\n请检查Ollama是否启动")
            elif resp.status_code == 507:
                st.error(f"❌ 显存溢出：{detail}\n请减小config.py中的BATCH_SIZE")
            elif resp.status_code == 400:
                st.error(f"❌ 请求错误：{detail}")
            else:
                st.error(f"❌ API错误({resp.status_code}): {detail}")
            return None
        return resp.json()
    except requests.Timeout:
        st.error("❌ 请求超时，首次调用需加载模型约30s，请稍后重试")
        return None
    except Exception as e:
        st.error(f"❌ 网络错误: {e}")
        return None


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="小说角色扮演智能体",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("## 📚 小说智能体")
    st.caption("百万字小说RAG + 角色扮演")

    # API状态检查
    api_ok = check_api()
    if api_ok:
        st.success("✅ 后端服务在线", icon="✅")
    else:
        st.error("❌ 后端服务未启动", icon="❌")
        st.info("请先运行：`python api_server.py`")

    st.divider()

    # 角色选择
    st.markdown("### 🎭 角色选择")
    selected = st.selectbox("选择对话角色", ROLES, index=ROLES.index(st.session_state.selected_role))
    st.session_state.selected_role = selected

    # 会话ID
    st.session_state.thread_id = st.text_input(
        "会话ID", value=st.session_state.thread_id,
        help="相同ID保持对话记忆，换ID开始新对话"
    )

    # 清空对话
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # 知识库管理
    st.markdown("### 📖 知识库管理")

    # 状态查看
    if st.button("查看知识库状态", use_container_width=True):
        if api_ok:
            with st.spinner("查询中..."):
                info = requests.get(f"{API_URL}/api/status", timeout=5).json()
                col1, col2 = st.columns(2)
                col1.metric("知识库", "✅ 存在" if info["kb_exists"] else "❌ 不存在")
                col2.metric("记录数", info.get("record_count", 0))
                st.metric("Ollama", "✅ 在线" if info["ollama_ok"] else "❌ 离线")
                # 阶段5新增：GPU显存占用显示
                gpu = info.get("gpu_memory")
                if gpu:
                    st.metric(
                        "GPU显存",
                        f"{gpu['used_mb']}MB / {gpu['total_mb']}MB",
                        f"{gpu['percent']}%"
                    )
                if info.get("models"):
                    st.text("可用模型：")
                    for m in info["models"]:
                        st.text(f"  • {m}")
        else:
            st.warning("后端未启动")

    # 上传小说
    st.markdown("---")
    uploaded = st.file_uploader("📤 上传小说TXT", type=["txt"], label_visibility="collapsed")

    if uploaded is not None:
        st.info(f"已选择: {uploaded.name} ({uploaded.size // 1024}KB)")
        if st.button("🔧 构建知识库", use_container_width=True, type="primary"):
            if not api_ok:
                st.error("后端未启动，无法构建")
            else:
                # 步骤1：上传文件
                with st.spinner("上传文件中..."):
                    try:
                        files = {"file": (uploaded.name, uploaded.getvalue(), "text/plain")}
                        upload_resp = requests.post(f"{API_URL}/api/upload", files=files, timeout=30)
                        if upload_resp.status_code != 200:
                            st.error(f"上传失败: {upload_resp.json().get('detail', upload_resp.text)}")
                            st.stop()
                        file_path = upload_resp.json()["path"]
                    except Exception as e:
                        st.error(f"上传异常: {e}")
                        st.stop()

                # 步骤2：构建知识库
                with st.spinner("构建知识库中（百万字分批向量化，预计数分钟）...", show_time=True):
                    try:
                        build_resp = requests.post(
                            f"{API_URL}/api/build",
                            json={"filepath": file_path},
                            timeout=TIMEOUT_BUILD,
                        )
                        if build_resp.status_code == 200:
                            result = build_resp.json()
                            st.success(f"✅ {result['message']}")
                            st.balloons()
                        else:
                            try:
                                detail = build_resp.json().get("detail", "未知错误")
                            except Exception:
                                detail = build_resp.text
                            # 按状态码分类
                            if build_resp.status_code == 507:
                                st.error(f"❌ 显存溢出: {detail}")
                                st.info("💡 建议：减小config.py中的BATCH_SIZE（如改为16或8）")
                            elif build_resp.status_code == 400:
                                st.error(f"❌ 文件问题: {detail}")
                            elif build_resp.status_code == 503:
                                st.error(f"❌ 模型未就绪: {detail}")
                            else:
                                st.error(f"❌ 构建失败({build_resp.status_code}): {detail}")
                    except requests.Timeout:
                        st.error("❌ 构建超时（百万字向量化需要几分钟，可适当延长TIMEOUT_BUILD）")
                    except Exception as e:
                        st.error(f"请求异常: {e}")


# ==================== 主区域：对话界面 ====================
st.markdown(f"# 🎭 {st.session_state.selected_role}的对话空间")

# 显示对话历史
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant":
            st.markdown(f"**[{msg['role_name']}]**")
        st.markdown(msg["content"])

# 输入框
question = st.chat_input(f"向{st.session_state.selected_role}提问...")

if question:
    if not api_ok:
        st.error("后端服务未启动，请先运行 api_server.py")
        st.stop()

    # 显示用户消息
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question, "role_name": "用户"})

    # 调用API获取回答（多阶段进度提示）
    with st.chat_message("assistant", avatar="🤖"):
        role_name = st.session_state.selected_role
        st.markdown(f"**[{role_name}]**")
        with st.status("正在思考...", expanded=True) as status:
            # 阶段5：多阶段进度提示（用户偏好）
            st.write("🧠 分析问题类型...")
            st.write("📚 检索小说原文...")
            st.write("🎭 角色扮演生成回答...")
            result = call_chat(question, role_name, st.session_state.thread_id)
            if result:
                status.update(label="回答就绪", state="complete")
                st.markdown(result["answer"])
                # 保存到历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "role_name": role_name,
                })
            else:
                status.update(label="出错了", state="error")