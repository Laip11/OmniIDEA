import os
import sys
import json
from dotenv import load_dotenv

DOTENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(DOTENV_PATH, override=True)

from langgraph.types import Command
from graph import app, checkpointer
from agents import shared_store
from storage import OUTPUTS_FINAL_REPORT, PLANNING_GLOBAL_CONTEXT


def print_welcome_banner(query: str):
    """Print session welcome banner."""
    print("\n" + "=" * 60)
    print("OmniIDEA — Multi-Agent Research Workflow")
    print("=" * 60)
    print(f"Research topic: {query}")
    print("-" * 60)
    print("Pipeline: Survey → Idea → Assessment → Report")
    print("Tip: ask to redo a phase (e.g. 'redo survey', 'regenerate ideas')")
    print("-" * 60)
    print("Provide feedback in natural language at each checkpoint.")
    print("Exit anytime with: quit / exit / q")
    print("=" * 60)


def parse_cli_args(argv: list[str]) -> tuple[str, str]:
    """Parse CLI args, supporting optional --thread SESSION_ID."""
    thread_id = "main_thread"
    query_parts: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--thread":
            if i + 1 >= len(argv):
                print("Error: --thread requires a session id.")
                sys.exit(1)
            thread_id = argv[i + 1]
            i += 2
        else:
            query_parts.append(argv[i])
            i += 1
    return " ".join(query_parts).strip(), thread_id


def run_with_human_in_loop(query: str, thread_id: str = "main_thread"):
    """
    运行带有 Human-in-the-Loop 的研究工作流

    Args:
        query: 用户的研究问题/需求
        thread_id: 线程 ID，用于状态持久化
    """
    initial_state = {
        "topic": query,
        "iteration_count": 0,
        "messages": [],
        "human_feedback": None,
        "focus_areas": None,
        "pending_human_action": None,
        "uploaded_files": [],
        "research_plan": None,
        "plan_confirmed": False,
        "current_phase": None,
        "completed_phase": None,
        "iteration_history": [],
        "termination_reason": None,
        "active_skill": None,
        "auto_mode": False,  # 自动模式：用户表示不想参与时启用
        "idea_assessment_iterations": 0,  # Idea↔Assessment 闭环迭代计数
    }

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}

    print_welcome_banner(query)
    print("🚀 系统启动...")

    # 打印可用 Skills
    available_skills = shared_store.list_skills()
    if available_skills:
        print(f"   🎯 已加载 Skills: {', '.join(available_skills)}")

    # 记录初始 topic 的 memory 版本
    memory_version = shared_store.record_memory_version(
        user_input=query,
        input_type="topic",
        current_phase="initialization",
        iteration_count=0
    )
    initial_state["memory_version"] = memory_version

    # 检查是否有之前中断的状态
    current_state = app.get_state(config)

    if current_state.values:
        print("\n📂 检测到之前的运行状态，正在恢复...")
        if current_state.next:
            print(f"   当前等待节点: {current_state.next}")

        values = current_state.values
        if values.get("research_plan"):
            print(f"   研究计划: 已生成")
            print(f"   计划确认: {'是' if values.get('plan_confirmed') else '否'}")
        print(f"   迭代次数: {values.get('iteration_count', 0)}")
        print(f"   当前阶段: {values.get('current_phase', '未开始')}")
        if values.get("auto_mode"):
            print(f"   🤖 自动模式: 已启用")

    input_state = initial_state if not current_state.values else None

    while True:
        try:
            for event in app.stream(input_state, config=config):
                for node_name, node_output in event.items():
                    if node_name == "__interrupt__":
                        pass

            state = app.get_state(config)

            if state.next:
                if "HumanReview" in state.next:
                    user_input = input("\n👤 您的输入: ").strip()

                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("\n👋 用户退出，流程终止。")
                        save_current_progress(state.values)
                        break

                    input_state = Command(resume=user_input)
                    continue
                else:
                    input_state = None
                    continue
            else:
                print("\n" + "=" * 60)
                print("✅ 研究流程已完成！")
                print("=" * 60)

                if state.values:
                    termination_reason = state.values.get("termination_reason")
                    if termination_reason:
                        reason_map = {
                            "max_iterations": "达到最大迭代次数",
                            "user_terminated": "用户主动终止",
                        }
                        print(f"📋 终止原因: {reason_map.get(termination_reason, termination_reason)}")
                break

        except KeyboardInterrupt:
            print("\n\n⚠️ 检测到中断信号，正在保存进度...")
            state = app.get_state(config)
            if state.values:
                save_current_progress(state.values)
            print("💾 进度已保存，下次运行时将自动恢复。")
            break
        except Exception as error:
            print(f"\n❌ 发生错误: {error}")
            import traceback
            traceback.print_exc()
            break

    print_final_results()


def save_current_progress(values: dict):
    """保存当前进度到 Planning/global_context.json"""
    try:
        progress = {
            "topic": values.get("topic"),
            "iteration_count": values.get("iteration_count", 0),
            "current_phase": values.get("current_phase"),
            "completed_phase": values.get("completed_phase"),
            "plan_confirmed": values.get("plan_confirmed", False),
        }
        shared_store.write_file(
            PLANNING_GLOBAL_CONTEXT,
            json.dumps(progress, ensure_ascii=False, indent=2)
        )
        print("💾 进度快照已保存")
    except Exception as error:
        print(f"⚠️ 保存进度失败: {error}")


def print_final_results():
    """打印最终结果"""
    print("\n" + "=" * 60)
    print("📊 研究成果汇总")
    print("=" * 60)

    try:
        all_files = shared_store.list_files()

        print(f"\n📁 Workspace 文件 ({len(all_files)} 个):")
        for file_path in sorted(all_files):
            print(f"   • {file_path}")

        workspace_dir = shared_store.get_workspace_dir()
        print(f"\n💾 Workspace 目录: {workspace_dir}")

        # 尝试读取最终报告
        final_report = shared_store.read_file(OUTPUTS_FINAL_REPORT)
        if final_report:
            print("\n" + "=" * 60)
            print("📝 最终报告预览 (前 500 字符):")
            print("=" * 60)
            preview = final_report[:500] + "..." if len(final_report) > 500 else final_report
            print(preview)
            print("=" * 60)
            print(f"📄 完整报告请查看: {os.path.join(workspace_dir, OUTPUTS_FINAL_REPORT)}")
        else:
            print("\n⚠️ 未找到最终报告文件")

    except Exception as error:
        print(f"⚠️ 读取结果时出错: {error}")

def main():
    """CLI entry point."""
    query, thread_id = parse_cli_args(sys.argv[1:])

    if not query:
        print("OmniIDEA — Multi-Agent Research Workflow")
        print("Enter your research question:")
        query = input("> ").strip()

    if not query:
        print("\nUsage:")
        print('  python main.py "Your research question"')
        print('  python main.py --thread SESSION_ID "Your research question"')
        sys.exit(1)

    run_with_human_in_loop(query, thread_id)


if __name__ == "__main__":
    main()
