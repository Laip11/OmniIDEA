"""
Agent 构建模块 - 创建各种角色的 Agent
DeepResearch 框架 - workspace 文件系统

Skills 机制遵循 LangChain Deep Agents 的 Agent Skills 规范：
- 每个 skill 是 Skills/<name>/SKILL.md，使用 YAML frontmatter 定义 name 和 description
- 通过 create_deep_agent 的 skills 参数原生注入，Agent 自动进行 progressive disclosure
- 同时保留 nodes 层面的 load_available_skills() 作为补充上下文注入
"""

from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

from prompts import (
    build_system_prompt,
    PLANNING_AGENT_PROMPT,
    SURVEY_AGENT_PROMPT,
    IDEA_AGENT_PROMPT,
    ASSESSMENT_RUBRIC_AGENT_PROMPT,
    REPORT_AGENT_PROMPT
)
from tools import (
    set_shared_store,
    SURVEY_TOOLS,
    AGENT_BASE_TOOLS,
    ALL_TOOLS,
)
from storage import create_workspace_store

# ==========================================
# 基础设施配置 (Store & Backend)
# ==========================================

# 创建共享的 workspace 存储（支持分层目录结构）
shared_store = create_workspace_store()

# 将 shared_store 注入到 tools 模块，使 write_file tool 能同步更新内存状态
set_shared_store(shared_store)

# 后端路由（deepagents>=0.6 新写法）：直接实例化 BackendProtocol 而非 factory。
# StateBackend 运行时经 get_config() 读写 state，StoreBackend 经 get_store() 获取 store，
# 实例本身不绑定 runtime、无状态，可被所有 Agent 安全共享。
# StoreBackend 必须显式传 namespace（0.7.0 起移除 legacy 的 assistant_id 探测）：
# 本项目所有 Agent 共享同一 workspace 且调用时未传 assistant_id，
# 固定 ("filesystem",) 与 legacy 探测的实际解析结果完全一致。
workspace_backend = CompositeBackend(
    default=StateBackend(),
    routes={"/workspace/": StoreBackend(namespace=lambda _runtime: ("filesystem",))},
)

# ==========================================
# LLM 配置 - 支持不同温度设置
# ==========================================

import os as _os

_DEFAULT_MODEL = "openai:gpt-4o-mini"

_model = _os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL)
if not str(_model).startswith("openai:"):
    _model = f"openai:{_model}"

_env_key = _os.environ.get("OPENAI_API_KEY", "").strip()
if not _env_key:
    raise ValueError(
        "OPENAI_API_KEY is required. Copy .env.example to .env and set your API key."
    )

# Support OPENAI_BASE_URL or OPENAI_API_BASE for proxies and compatible gateways.
def _openai_base_url() -> str:
    return (
        _os.environ.get("OPENAI_BASE_URL")
        or _os.environ.get("OPENAI_API_BASE")
        or ""
    ).strip().rstrip("/")

_base = _openai_base_url() or "https://api.openai.com/v1"
_key = _env_key

LLM_CONFIG = {
    "model": _model,
    "api_key": _key,
    "base_url": _base,
}

# 从环境变量读取 max_tokens 配置，默认 16384
MAX_OUTPUT_TOKENS = int(_os.environ.get("MAX_NEW_TOKENS", "16384"))

# 标准温度模型（用于 Planning、Assessment、Report）
model_standard = init_chat_model(
    model=LLM_CONFIG["model"],
    temperature=0.7,
    api_key=LLM_CONFIG["api_key"],
    base_url=LLM_CONFIG["base_url"],
    max_tokens=MAX_OUTPUT_TOKENS,
)

# 高温度模型（用于 Idea Agent - 促进创造性生成）
model_creative = init_chat_model(
    model=LLM_CONFIG["model"],
    temperature=0.95,
    api_key=LLM_CONFIG["api_key"],
    base_url=LLM_CONFIG["base_url"],
    max_tokens=MAX_OUTPUT_TOKENS,
)

# 低温度模型（用于 Survey Agent - 确保准确性）
model_precise = init_chat_model(
    model=LLM_CONFIG["model"],
    temperature=0.3,
    api_key=LLM_CONFIG["api_key"],
    base_url=LLM_CONFIG["base_url"],
    max_tokens=MAX_OUTPUT_TOKENS,
)


def build_agent(name: str, role_prompt: str, tools=None, model=None,
                skills=None):
    """
    构建不同角色的 Agent，支持通过 skills 参数原生注入 Skills。

    遵循 LangChain Deep Agents 的 Agent Skills 规范：
    - skills 路径列表传入 create_deep_agent，Agent 启动时读取 SKILL.md frontmatter
    - Agent 收到 prompt 时通过 progressive disclosure 按需加载匹配的 skill 完整内容
    - 无需在 nodes 中手动拼接 skill 到 user message

    Args:
        name: Agent 名称
        role_prompt: 角色提示词
        tools: 可用工具列表
        model: 使用的 LLM 模型
        skills: Skills 源路径列表，如 ["/Skills/"]，
                为 None 时自动从 shared_store 获取
    """
    if tools is None:
        tools = []
    if model is None:
        model = model_standard

    # 自动获取 skills 源路径（如果未显式指定）
    if skills is None:
        skills = shared_store.get_skills_source_paths()

    system_prompt = build_system_prompt(name, role_prompt)

    return create_deep_agent(
        model=model,
        store=shared_store,
        backend=workspace_backend,
        tools=tools,
        system_prompt=system_prompt,
        skills=skills,
        checkpointer=MemorySaver()
    )

# ==========================================
# 初始化所有 Deep Agents
# ==========================================

# Survey Agent (调研员) - 低温度确保准确性
survey_agent = build_agent(
    "Survey Agent",
    SURVEY_AGENT_PROMPT,
    tools=SURVEY_TOOLS,
    model=model_precise
)

# Idea Agent (创新员) - 高温度促进创造性 + 文件读写
idea_agent = build_agent(
    "Idea Innovation Agent",
    IDEA_AGENT_PROMPT,
    tools=AGENT_BASE_TOOLS,
    model=model_creative
)

# 注意：Assessment Agent 不再拥有全局实例——它在 run_assessment_node 中按需构建，
# 以便将当前 rubric.md 内容注入系统提示的 {rubric_content} 占位符。

# Assessment Rubric Agent (评估标准设计专家) - 轻量角色，仅负责生成/写回 rubric
# 与 Assessment Agent 分离：评估员的系统提示包含完整评估流程（含 MANDATORY 写 feedback），
# 若复用它会驱动 Agent 在 rubric 步骤里把整轮评估都做完。
assessment_rubric_agent = build_agent(
    "Assessment Rubric Agent",
    ASSESSMENT_RUBRIC_AGENT_PROMPT,
    tools=AGENT_BASE_TOOLS,
    model=model_standard
)

# Report Agent (报告生成员) - 标准温度 + 文件读写
report_agent = build_agent(
    "Report Generation Agent",
    REPORT_AGENT_PROMPT,
    tools=AGENT_BASE_TOOLS,
    model=model_standard
)

# Planning Agent (智能规划员) - 标准温度 + 全部工具（核心调度者需要全局能力）
planning_agent = build_agent(
    "Planning Agent",
    PLANNING_AGENT_PROMPT,
    tools=ALL_TOOLS,
    model=model_standard
)

