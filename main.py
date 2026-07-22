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
import cli


def print_welcome_banner(query: str):
    """Print session welcome banner."""
    cli.banner(
        "OmniIDEA — Multi-Agent Research Workflow",
        [
            ("Topic", query),
            ("Pipeline", "Survey → Idea → Assessment → Report"),
            ("Checkpoints", "natural-language feedback, e.g. 'redo survey', 'regenerate ideas'"),
            ("Exit", "quit / exit / q anytime"),
        ],
    )


def parse_cli_args(argv: list[str]) -> tuple[str, str]:
    """Parse CLI args, supporting optional --thread SESSION_ID."""
    thread_id = "main_thread"
    query_parts: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--thread":
            if i + 1 >= len(argv):
                cli.error("--thread requires a session id")
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
        "rubric_dimensions": None,  # Assessment rubric 动态维度源数据（首轮生成后填充）
        "rubric_research_domain": None,  # Rubric Agent 修正后的领域描述
    }

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}

    print_welcome_banner(query)

    # 打印可用 Skills
    available_skills = shared_store.list_skills()
    if available_skills:
        cli.info(f"Skills loaded: {', '.join(available_skills)}")

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
        cli.section("Previous session detected — resuming")
        if current_state.next:
            cli.kv("Waiting node", str(current_state.next))

        values = current_state.values
        if values.get("research_plan"):
            cli.kv("Research plan", "generated")
            cli.kv("Plan confirmed", "yes" if values.get("plan_confirmed") else "no")
        cli.kv("Iterations", str(values.get("iteration_count", 0)))
        cli.kv("Current phase", str(values.get("current_phase") or "not started"))
        if values.get("auto_mode"):
            cli.kv("Auto mode", "enabled")

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
                    user_input = input("\nYour input > ").strip()

                    if user_input.lower() in ['quit', 'exit', 'q']:
                        cli.note("User quit — workflow terminated")
                        save_current_progress(state.values)
                        break

                    input_state = Command(resume=user_input)
                    continue
                else:
                    input_state = None
                    continue
            else:
                cli.stage("Research workflow completed")

                if state.values:
                    termination_reason = state.values.get("termination_reason")
                    if termination_reason:
                        reason_map = {
                            "max_iterations": "max iterations reached",
                            "user_terminated": "terminated by user",
                        }
                        cli.kv("Termination", reason_map.get(termination_reason, termination_reason))
                break

        except KeyboardInterrupt:
            cli.warn("Interrupted — saving progress...")
            state = app.get_state(config)
            if state.values:
                save_current_progress(state.values)
            cli.info("Progress saved; it will resume automatically on next run")
            break
        except Exception as error:
            cli.error(f"Workflow error: {error}")
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
        cli.debug("Progress snapshot saved")
    except Exception as error:
        cli.warn(f"Failed to save progress: {error}")


def print_final_results():
    """打印最终结果"""
    cli.section("Research output summary")

    try:
        all_files = shared_store.list_files()

        cli.info(f"Workspace files ({len(all_files)}):")
        for file_path in sorted(all_files):
            print(f"    - {file_path}")

        workspace_dir = shared_store.get_workspace_dir()
        cli.kv("Workspace dir", workspace_dir)

        # 尝试读取最终报告
        final_report = shared_store.read_file(OUTPUTS_FINAL_REPORT)
        if final_report:
            cli.section("Final report preview (first 500 chars)")
            preview = final_report[:500] + "..." if len(final_report) > 500 else final_report
            print(preview)
            cli.rule("=")
            cli.ok(f"Full report: {os.path.join(workspace_dir, OUTPUTS_FINAL_REPORT)}")
        else:
            cli.warn("Final report not found")

    except Exception as error:
        cli.warn(f"Failed to read results: {error}")

def main():
    """CLI entry point."""
    query, thread_id = parse_cli_args(sys.argv[1:])

    if not query:
        print("OmniIDEA — Multi-Agent Research Workflow")
        query = input("Enter your research question > ").strip()

    if not query:
        print("\nUsage:")
        print('  python main.py "Your research question"')
        print('  python main.py --thread SESSION_ID "Your research question"')
        sys.exit(1)

    run_with_human_in_loop(query, thread_id)


if __name__ == "__main__":
    main()
