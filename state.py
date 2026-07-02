import operator
from typing import TypedDict, Annotated, Literal, List, Optional, Dict
from langchain_core.messages import BaseMessage

# ==========================================
# 研究阶段枚举
# ==========================================
ResearchPhase = Literal[
    "initialization",  # 初始化阶段
    "survey",          # 调研阶段
    "idea",            # 创意阶段
    "assessment",      # 评估阶段
    "report"           # 报告阶段
]

# ==========================================
# 多维度上下文/记忆
# ==========================================
class MultiDimensionalContext(TypedDict):
    """多维度上下文管理（领域识别）"""
    primary_domain: str
    domain_keywords: Dict[str, List[str]]

# ==========================================
# 研究计划结构
# ==========================================
class ResearchPlan(TypedDict):
    """研究计划 - 由 Planning Agent 在初始化阶段生成"""
    primary_domain: str
    domain_keywords: Dict[str, List[str]]
    max_iterations: int

# ==========================================
# Planning Agent 决策结果
# ==========================================
class PlanningDecision(TypedDict):
    """Planning Agent 的决策结果"""
    user_intent: Literal["continue", "iterate", "finish", "upload", "guidance", "auto"]
    next_action: Literal[
        "Survey",
        "Idea",
        "Assessment",
        "Report",
        "Finish",
        "Upload",
        "HumanReview",
    ]
    reasoning: str
    feedback_summary: Optional[str]
    focus_areas: Optional[str]

# ==========================================
# 研究工作流全局状态
# ==========================================
class ResearchState(TypedDict):
    """研究工作流的全局状态 - 支持 workspace 分层文件系统"""

    # ========== 基础信息 ==========
    topic: str

    # ========== 研究计划 (初始化阶段生成) ==========
    research_plan: Optional[ResearchPlan]
    plan_confirmed: bool

    # ========== 多维度上下文/记忆 ==========
    multi_context: Optional[MultiDimensionalContext]

    # ========== 流程控制 ==========
    current_phase: Optional[ResearchPhase]
    completed_phase: Optional[ResearchPhase]
    next_agent: Literal[
        "Survey", "Idea", "Assessment",
        "Report", "HumanReview", "FINISH"
    ]

    # ========== 迭代控制 ==========
    iteration_count: int
    iteration_history: Annotated[List[dict], operator.add]

    # ========== Survey 阶段数据 ==========
    search_directions: Optional[List[str]]

    # ========== 消息与反馈 ==========
    messages: Annotated[List[BaseMessage], operator.add]
    human_feedback: Optional[str]
    focus_areas: Optional[str]

    # ========== Human-in-the-Loop ==========
    pending_human_action: Optional[Literal[
        "plan_review",
        "survey_review",
        "idea_review",
        "assessment_review",
        "report_review"
    ]]

    # ========== 自动模式 ==========
    # 当用户表示不想参与时，启用自动模式，后续流程不再询问用户
    auto_mode: bool

    # ========== Idea↔Assessment 迭代控制 ==========
    # 专门追踪 Idea→Assessment→Idea 闭环的迭代次数（独立于全局 iteration_count）
    idea_assessment_iterations: int

    # ========== 文件与资源 ==========
    uploaded_files: Optional[List[str]]

    # ========== 决策记录 ==========
    planning_decision: Optional[PlanningDecision]

    # ========== 当前活跃的 Skill ==========
    active_skill: Optional[str]

    # ========== Memory 版本追踪 ==========
    memory_version: int

    # ========== 终止条件 ==========
    termination_reason: Optional[Literal[
        "max_iterations",
        "user_terminated",
    ]]
