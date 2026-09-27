"""
统一配置模块 config.py
=============================
用途：集中管理所有路径、模型名、检索/智能体/API参数
存放路径：E:/novel_agent/src/config.py

设计原则：
1. 所有配置集中一处，避免散落在各模块
2. 支持环境变量覆盖：部署时改参数不用动代码
3. 自动初始化目录结构：首次import时确保所有路径存在

新版新增（相对阶段5）：
- RERANK_ENABLED: rerank开关，可关闭减少推理开销
- TIMELINE_PHASES: 时间线阶段定义，用于时间线过滤
- MAX_CONTEXT_CHARS: 上下文最大字符数，防止token溢出
- WEIGHT_VECTOR / WEIGHT_BM25: 分数归一加权融合权重
"""

import os
from pathlib import Path

# ==================== 项目根目录 ====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==================== 路径配置 ====================
NOVEL_DIR    = str(PROJECT_ROOT / "novel_docs")
NOVEL_PATH   = str(PROJECT_ROOT / "novel_docs" / "《斗罗大陆》作者：唐家三少.txt")
CHROMA_PATH  = str(PROJECT_ROOT / "chroma_db")
MODEL_CACHE  = str(PROJECT_ROOT / "src" / "model_cache")
LOG_DIR      = str(PROJECT_ROOT / "logs")
BUILD_PROGRESS_FILE = str(PROJECT_ROOT / "chroma_db" / "build_progress.json")  # 断点续建进度文件

# ==================== 模型配置 ====================
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL  = "BAAI/bge-reranker-v2-m3"
LLM_MODEL       = "qwen2.5:7b"
LLM_TEMP        = 0.3
DEVICE          = "cuda"

# ==================== 文本切分配置 ====================
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 100
BATCH_SIZE      = 32

# ==================== 检索配置 ====================
TOP_K_VECTOR    = 20
TOP_K_BM25      = 20
TOP_K_FINAL     = 5
RRF_K           = 60
MIN_FILTER_KEEP = 3

# 分数归一加权融合权重（替代RRF，向量与BM25加权）
WEIGHT_VECTOR   = 0.5   # 向量检索权重
WEIGHT_BM25     = 0.5   # BM25检索权重

# rerank开关：True开启精排，False关闭减少推理开销
RERANK_ENABLED  = True

# 时间线阶段定义：按剧情阶段标注chunk，用于时间线过滤
# 格式: {阶段名: (起始chunk_id, 结束chunk_id)}
# 斗罗大陆大致剧情阶段（按chunk_id估算，7737个chunk总量）
TIMELINE_PHASES = {
    "诺丁学院": (0, 800),
    "史莱克学院": (800, 2500),
    "海神岛": (2500, 5000),
    "唐门": (5000, 7000),
    "大结局": (7000, 7737),
}

# 早期回忆类问题关键词：命中时默认限定最早剧情阶段（诺丁学院）
# 解决"你和小舞怎么认识的"这类问题检索到后期剧情片段导致时间错位
EARLY_EVENT_KEYWORDS = [
    "怎么认识", "怎么遇到", "如何认识", "初见", "初遇",
    "小时候", "刚入学", "第一次见", "第一次见面", "何时认识",
]

# ==================== 智能体配置 ====================
MAX_RETRIEVAL_ROUNDS = 3
DEFAULT_ROLE         = "唐三"
HISTORY_WINDOW       = 10

# 上下文最大字符数：限制送入LLM的原文片段总长度，防止token溢出
# qwen2.5:7b 上下文窗口约32768 token，约2.6万字符，留余量设6000
MAX_CONTEXT_CHARS    = 6000

# ==================== API服务配置 ====================
OLLAMA_URL           = "http://localhost:11434"
HOST                 = "0.0.0.0"
PORT                 = 8000
OLLAMA_CHECK_TIMEOUT = 3

# ==================== 前端配置 ====================
FRONT_API_URL      = "http://localhost:8000"
FRONT_TIMEOUT_CHAT = 120
FRONT_TIMEOUT_BUILD = 600
ROLES              = ["唐三", "小舞", "大师", "戴沐白", "奥斯卡", "宁荣荣"]

# ==================== 角色档案 ====================
# 用途：注入generate节点的角色背景设定，即使不检索原文也能保证角色一致性
# 覆盖6个可对话角色：背景出身/关键经历/人物关系/性格/说话风格
ROLE_PROFILES = {
    "唐三": (
        "你出身圣魂村，父亲是昊天斗罗唐昊，母亲是蓝银皇阿银（早年去世）；"
        "你前世是唐门弟子，精通暗器与玄天功；"
        "武魂是蓝银草（曾被认作废武魂），后觉醒昊天锤；"
        "你的老师是大师（玉小刚），你尊称他\"老师\"，他传授你武魂理论，你对他十分敬重；"
        "你性格沉稳坚毅、外冷内热、重情重义，说话务实不浮夸，对修炼极其专注。"
    ),
    "小舞": (
        "你是十万年魂兽柔骨兔化形的人，武魂也是柔骨兔；"
        "你活泼可爱、率真直爽、重情重义，喜欢叫唐三\"小三哥\"，爱撒娇但关键时刻很勇敢；"
        "你和唐三互相爱慕、是彼此最重要的人（恋人关系），叫他\"小三哥\"只是亲昵称呼，不是兄妹；"
        "大师是唐三的老师，你也跟着敬称他为老师。"
    ),
    "大师": (
        "你是斗罗大陆的理论大师，唐三的老师，本名玉小刚，人称大师；"
        "武魂是罗三炮，修炼天赋有限但武魂理论造诣极深；"
        "你的弟子是唐三，对他既严格又关怀；"
        "你深爱柳二龙，与比比东有过一段痛苦的往事，弗兰德是你多年的挚友和兄弟；"
        "你性格执着专注、严谨有条理，说话常引经据典讲武魂理论，措辞严谨。"
    ),
    "戴沐白": (
        "你是星罗帝国皇子，史莱克七怪的老大，武魂白虎；"
        "史莱克七怪的其他成员（唐三、小舞、奥斯卡、宁荣荣等）是你的伙伴，你护短、讲义气；"
        "朱竹清是你的未婚妻，你们有星罗帝国的婚约，你心里在乎她但嘴上常常不承认；"
        "你性格豪爽霸气、讲义气、天生领袖气质；"
        "说话直接干脆，带几分骄傲但不失分寸。"
    ),
    "奥斯卡": (
        "你是史莱克七怪之一，食物系魂师，武魂是香肠；"
        "你性格幽默乐观、有点贫嘴爱开玩笑，对朋友忠诚；"
        "你暗恋七宝琉璃宗的宁荣荣，对她有好感；"
        "虽然武魂前期不被看好，但你从不气馁，相信努力能改变命运。"
    ),
    "宁荣荣": (
        "你是七宝琉璃宗宗主宁风致的女儿，武魂七宝琉璃塔，被称为小公主；"
        "你出身高贵但心地善良，前期有些娇气，后期逐渐变得稳重坚韧；"
        "史莱克七怪的成员（唐三、小舞、奥斯卡等）是你的同学和战友，你和唐三只是队友朋友，不是亲属；"
        "奥斯卡一直对你很好，你对他也有好感；"
        "说话带着大小姐的矜持，但真心对待每一位朋友。"
    ),
}


# ==================== 角色别名映射 ====================
# 用途：检索前把用户问题中的角色别名/称呼归一化为正式名，提升关系型问题召回
# 例如"你和小刚什么关系" -> "小刚"替换为"大师"，配合人称替换命中师生关系片段
CHARACTER_ALIASES = {
    "小刚": "大师",
    "玉小刚": "大师",
    "刚哥": "大师",
    "小三哥": "唐三",
    "三哥": "唐三",
}

# ==================== 离线模式（必须在import transformers前设置） ====================
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _env_override(name: str, default):
    """从环境变量读取覆盖值，命名前缀 NOVEL_AGENT_"""
    key = f"NOVEL_AGENT_{name}"
    if key not in os.environ:
        return default
    val = os.environ[key]
    if isinstance(default, bool):
        return val.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        try:
            return int(val)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(val)
        except ValueError:
            return default
    return val


# 环境变量覆盖
DEVICE         = _env_override("DEVICE", DEVICE)
PORT           = _env_override("PORT", PORT)
LLM_MODEL      = _env_override("LLM_MODEL", LLM_MODEL)
BATCH_SIZE     = _env_override("BATCH_SIZE", BATCH_SIZE)
RERANK_ENABLED = _env_override("RERANK_ENABLED", RERANK_ENABLED)


def ensure_dirs():
    """初始化所有需要的目录（首次运行/部署时调用）"""
    for d in [NOVEL_DIR, MODEL_CACHE, LOG_DIR]:
        os.makedirs(d, exist_ok=True)


# 首次import时自动初始化目录
ensure_dirs()
