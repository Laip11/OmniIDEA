"""
工作流图定义 - DeepResearch 框架核心工作流
流程：Planner(调度+初始化) → 研究执行(Survey→Idea→Assessment→Report) → 人类评审 → 总结输出
支持评估反馈驱动的迭代闭环：Assessment → Idea（基于结构化评估反馈定向优化想法）
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import ResearchState
from nodes import (
    run_survey_node,
    run_idea_node,
    run_assessment_node,
    run_report_node,
    human_review_node,
    planning_node
)

# Checkpointer 用于支持 interrupt
checkpointer = MemorySaver()

def router(state: ResearchState):
    """
    主路由逻辑 - 根据 next_agent 决定下一个节点
    """
    next_agent = state.get("next_agent")
    if next_agent == "FINISH":
        return END
    return next_agent

def build_workflow() -> StateGraph:
    """
    构建并返回工作流图

    工作流结构：
    ┌─────────────┐
    │   Planner   │ ◄──────────────────────────────────┐
    │ (含初始化)  │                                    │
    └──────┬──────┘                                    │
           │ (条件路由)                                 │
           ▼                                           │
    ┌──────────────────────────────────────────────┐   │
    │ Survey → Idea → Assessment → Report → END   │   │
    └──────────────────────────────────────────────┘   │
           │         ↑                             │   │
           │         │  评估反馈驱动迭代            │   │
           │         │  (Assessment→Idea)           │   │
           │         │  基于结构化评估反馈定向优化   │   │
           │         └─────────────────────────────┘   │
           ▼                                           │
    ┌─────────────┐                                    │
    │ HumanReview │ ───────────────────────────────────┘
    └─────────────┘
    """
    workflow = StateGraph(ResearchState)

    # ========== 添加节点 ==========
    workflow.add_node("Planner", planning_node)
    workflow.add_node("Survey", run_survey_node)
    workflow.add_node("Idea", run_idea_node)
    workflow.add_node("Assessment", run_assessment_node)
    workflow.add_node("Report", run_report_node)
    workflow.add_node("HumanReview", human_review_node)

    # ========== 设置入口 ==========
    workflow.set_entry_point("Planner")

    # ========== 添加条件边 ==========
    workflow.add_conditional_edges(
        "Planner",
        router,
        {
            "Survey": "Survey",
            "Idea": "Idea",
            "Assessment": "Assessment",
            "Report": "Report",
            "HumanReview": "HumanReview",
            END: END
        }
    )

    # ========== 添加固定边 ==========
    workflow.add_edge("Survey", "Planner")
    workflow.add_edge("Idea", "Planner")
    workflow.add_edge("Assessment", "Planner")
    workflow.add_edge("Report", "Planner")
    workflow.add_edge("HumanReview", "Planner")

    return workflow

def compile_app():
    """编译并返回可执行的应用"""
    workflow = build_workflow()
    return workflow.compile(checkpointer=checkpointer)

# 预编译的应用实例
app = compile_app()
