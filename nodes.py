"""
Node implementation module - Multi-agent collaboration with workspace hierarchical file system
DeepResearch framework - ReAct planning, web search, innovation generation, assessment

Skills mechanism:
Injected via the skills parameter of create_deep_agent, the framework's SkillsMiddleware automatically performs progressive disclosure:
   - Reads SKILL.md frontmatter (name + description) when Agent starts
   - SkillsMiddleware.modify_request() automatically injects Skills System into the prompt
   - Matches relevant skills when receiving prompts, reads full SKILL.md content on demand
   - invoke_agent_with_retry injects SKILL.md files into StateBackend via build_skills_files()
"""

import os
import json
import re
import time
from typing import Optional, List, Dict
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from state import ResearchState, PlanningDecision
from agents import (
    survey_agent, idea_agent, assessment_rubric_agent,
    report_agent, planning_agent, shared_store, build_agent, model_standard
)
from tools import ALL_TOOLS, AGENT_BASE_TOOLS
from prompts import (
    get_human_review_prompt,
    get_planning_analysis_prompt,
    PLANNING_INIT_PROMPT,
    ASSESSMENT_AGENT_PROMPT,
)
from storage import (
    PLANNING_GLOBAL_CONTEXT,
    PLANNING_SEARCH_PLAN,
    PLANNING_HUMAN_FEEDBACK,
    INPUTS_UPLOADS_DIR,
    SURVEY_REPORT,
    SURVEY_HUMAN_FEEDBACK,
    INNOVATION_IDEAS,
    INNOVATION_HUMAN_FEEDBACK,
    ASSESSMENT_RUBRIC,
    ASSESSMENT_RUBRIC_DIMENSIONS,
    ASSESSMENT_FEEDBACK,
    ASSESSMENT_ITERATION_HISTORY,
    ASSESSMENT_HUMAN_FEEDBACK,
    OUTPUTS_FINAL_REPORT,
)

import cli

# ==========================================
# Unified Agent invocation retry mechanism
# ==========================================

MAX_RETRIES = int(os.environ.get("AGENT_INVOKE_MAX_RETRIES", "3"))
# 指数退避：第 1 次失败后等 base，第 2 次失败后等 2*base，不超过 max（默认约 2s + 4s，比原先 5s+10s 快）
RETRY_BASE_DELAY = float(os.environ.get("AGENT_RETRY_BASE_DELAY", "2"))
RETRY_MAX_DELAY = float(os.environ.get("AGENT_RETRY_MAX_DELAY", "8"))

# Idea↔Assessment 闭环最大自动迭代轮数（可通过环境变量覆盖）
MAX_IDEA_ASSESSMENT_ITERS = int(os.environ.get("IDEA_ASSESSMENT_MAX_ITERS", "2"))

def invoke_agent_with_retry(agent, messages, config, agent_name="Agent"):

    # StateBackend requires injecting skill files into virtual file system via invoke(files={...})
    if isinstance(messages, dict) and "files" not in messages:
        skills_files = shared_store.build_skills_files()
        if skills_files:
            messages["files"] = skills_files

    last_exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return agent.invoke(messages, config=config)
        except Exception as error:
            last_exception = error
            error_message = str(error).lower()
            is_retryable = any(keyword in error_message for keyword in [
                "connection error",
                "server disconnected",
                "remote protocol error",
                "timeout",
                "connection reset",
                "connection refused",
                "apiconnectionerror",
                "rate limit",
                "429",
                "502",
                "503",
                "504",
            ])

            if not is_retryable or attempt == MAX_RETRIES:
                cli.error(f"{agent_name} invocation failed (attempt {attempt}/{MAX_RETRIES}): {error}")
                raise

            delay = min(
                RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                RETRY_MAX_DELAY,
            )
            cli.warn(f"{agent_name} connection failed (attempt {attempt}/{MAX_RETRIES}), retrying in {delay:.0f}s — {str(error)[:100]}")
            time.sleep(delay)

# ==========================================
# File write verification and fallback mechanism
# ==========================================

def _sanitize_extracted_content(content: str) -> str:
    """
    Sanitize content recovered from Agent reply text before persisting it to a file.

    Reply text may carry model reasoning blocks or conversational wrappers
    ("Here is the report...", "Let me know if you need changes") that must never
    reach workspace files. Design principle: prefer a false negative (a leftover
    conversational line is visible and easy to delete) over a false positive
    (silently deleting legitimate content is hard to detect). Every rule below is
    anchored so it cannot bite into body text:
    1. Reasoning blocks (<think>...</think>) are stripped ONLY at the very start,
       where providers inline them — body text discussing such tags is untouched
    2. Conversational lead-ins are stripped ONLY when immediately followed by a
       Markdown structural line — prose starting with words like "首先" is untouched
    3. Sign-off patterns require explicit conversational forms — academic closers
       like "以上分析表明..." are untouched
    """
    import re
    if not content:
        return content
    original_len = len(content)

    # 1. Strip reasoning blocks only at the very beginning (loop for consecutive blocks)
    content = content.lstrip()
    while True:
        think_match = re.match(r"<think(?:ing)?>.*?</think(?:ing)?>\s*", content, flags=re.DOTALL | re.IGNORECASE)
        if not think_match:
            break
        content = content[think_match.end():].lstrip()

    # 2. Strip conversational lead-in lines ONLY when the lead-in block is immediately
    #    followed by a Markdown structural line (heading, table, list, fence, quote)
    leadin_pattern = re.compile(
        r"^(?:okay|ok|sure|great|certainly|here\s+(?:is|are)|i\s+(?:have|will|'ve|am)|let\s+me|"
        r"好的|当然|以下是|下面是|我已|我将|让我|现在|接下来|根据(?:你的|要求))",
        re.IGNORECASE,
    )
    lines = content.split("\n")
    saw_leadin = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if leadin_pattern.match(stripped):
            saw_leadin = True
            continue
        # First non-blank, non-leadin line: strip the lead-in block only if it is structural
        if saw_leadin and stripped.startswith(("#", "|", "- ", "```", ">")):
            lines = lines[i:]
        break
    content = "\n".join(lines)

    # 3. Strip conversational sign-off lines at the very end (explicit conversational
    #    forms only; academic closers such as "以上分析表明" never match)
    signoff_pattern = re.compile(
        r"^(?:please\s+(?:review|let me know|feel free)|let me know|"
        r"if you (?:need|have|would|want)|i hope|hope this|feel free|"
        r"the above (?:report|assessment|feedback|content|ideas?|analysis) (?:is|has been|was)|"
        r"请(?:查收|审阅|查看|提出)|如有需要|如需(?:进一步|其他)|希望(?:这份|这|能)|"
        r"以上(?:报告|评估|反馈|内容)[^\n]*(?:已完成|已保存|供参考))",
        re.IGNORECASE,
    )
    lines = content.rstrip().split("\n")
    end = len(lines)
    while end > 0:
        stripped = lines[end - 1].strip()
        if not stripped or signoff_pattern.match(stripped):
            end -= 1
        else:
            break
    content = "\n".join(lines[:end])

    content = content.strip()

    # 4. Audit trail: log what was stripped; warn loudly when the stripped share is
    #    suspiciously large (possible over-stripping → human should inspect the file)
    removed = original_len - len(content)
    if removed > 0:
        cli.debug(f"Sanitized extracted content: stripped {removed}/{original_len} chars")
        if removed > original_len * 0.3:
            cli.warn(f"Sanitizer stripped {removed}/{original_len} chars (>30%) from recovered content — inspect the saved file for over-stripping")

    return content


def verify_and_save_outputs(agent_response, expected_files: list, agent_name: str):
    """
    Verify whether expected files have been written locally after Agent execution;
    if not, extract from the reply and save as fallback.

    The Agent may "describe" a write operation in its reply without actually calling
    the write_file tool. This function ensures expected files are always persisted
    to the local file system.

    Args:
        agent_response: Return value of invoke_agent_with_retry
        expected_files: List of expected file paths (relative to workspace)
        agent_name: Agent name, used for logging
    """
    missing_files = []
    for file_path in expected_files:
        content = shared_store.read_file(file_path)
        if content and len(content.strip()) > 0:
            cli.debug(f"File verified: {file_path}")
        else:
            missing_files.append(file_path)
            cli.warn(f"File missing: {file_path}")

    if not missing_files:
        return

    # Round zero: try to extract write_file content directly from Agent's tool_calls parameters
    # When Agent did call write_file but file was not persisted, content is in tool_calls parameters
    tool_call_extracted = _extract_write_file_from_tool_calls(agent_response, missing_files)
    still_after_tool_calls = []
    for file_path in missing_files:
        if file_path in tool_call_extracted:
            shared_store.write_file(file_path, tool_call_extracted[file_path])
            cli.note(f"Recovered from {agent_name}'s tool_calls: {file_path}")
        else:
            still_after_tool_calls.append(file_path)

    if not still_after_tool_calls:
        return

    missing_files = still_after_tool_calls

    # Try to extract content from Agent reply
    reply_content = _extract_agent_reply(agent_response)
    if not reply_content:
        cli.warn(f"{agent_name} produced no extractable reply content; cannot fallback-save missing files")
        return

    # Round one: try to precisely extract content for each file
    still_missing = []
    for file_path in missing_files:
        extracted = _extract_file_content_from_reply(reply_content, file_path)
        if extracted:
            shared_store.write_file(file_path, _sanitize_extracted_content(extracted))
            cli.note(f"Recovered from {agent_name} reply: {file_path}")
        else:
            still_missing.append(file_path)

    if not still_missing:
        return

    # Round two fallback: for still-missing .md files, try splitting reply content by delimiters
    md_still_missing = [f for f in still_missing if f.endswith(".md")]
    non_md_missing = [f for f in still_missing if not f.endswith(".md")]

    if md_still_missing and reply_content.strip() and len(reply_content) > 200:
        split_results = _split_reply_into_sections(reply_content, md_still_missing)

        for file_path, section_content in split_results.items():
            if section_content and len(section_content.strip()) > 50:
                shared_store.write_file(file_path, _sanitize_extracted_content(section_content))
                cli.note(f"Split-saved from {agent_name} reply: {file_path}")
                md_still_missing.remove(file_path)

        # If after splitting there is exactly one .md file still missing and reply looks like valid Markdown, save as whole
        is_valid_markdown = reply_content.strip().startswith("#") and len(reply_content) > 200
        if len(md_still_missing) == 1 and is_valid_markdown:
            shared_store.write_file(md_still_missing[0], _sanitize_extracted_content(reply_content))
            cli.note(f"Saved {agent_name} full reply as: {md_still_missing[0]}")
            md_still_missing.clear()

    for file_path in md_still_missing + non_md_missing:
        cli.error(f"Cannot extract {file_path} from reply — check whether {agent_name} called write_file")


def _extract_write_file_from_tool_calls(agent_response, target_files: list) -> Dict[str, str]:
    """
    Extract write_file call content from Agent response's tool_calls parameters.

    When the Agent did call the write_file tool but the file was not successfully persisted
    (e.g., StateBackend wrote to memory but did not sync to file system), the report content
    exists in AIMessage's tool_calls parameters. This function iterates all AIMessage's
    tool_calls, matches target file paths and extracts content.

    Args:
        agent_response: Return value of invoke_agent_with_retry
        target_files: List of target file paths to extract

    Returns:
        {file_path: content} dict, containing only successfully extracted files
    """
    results = {}
    if agent_response is None:
        return results

    if not isinstance(agent_response, dict):
        return results

    messages = agent_response.get("messages", [])
    if not messages:
        return results

    from langchain_core.messages import AIMessage

    # Build target filename set for fast matching
    target_basenames = {}
    for file_path in target_files:
        basename = os.path.basename(file_path)
        target_basenames[basename] = file_path
        # Also match full paths and paths with workspace/ prefix
        target_basenames[file_path] = file_path
        target_basenames[f"workspace/{file_path}"] = file_path
        target_basenames[f"/workspace/{file_path}"] = file_path
        target_basenames[f"/{file_path}"] = file_path

    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue

        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            if tool_name != "write_file":
                continue

            args = tool_call.get("args", {})
            call_file_path = args.get("file_path", "")
            call_content = args.get("content", "")

            if not call_file_path or not call_content:
                continue

            # Clean path prefix
            clean_call_path = call_file_path.lstrip("/")
            if clean_call_path.startswith("workspace/"):
                clean_call_path = clean_call_path[len("workspace/"):]

            # Match target file
            matched_target = None
            if call_file_path in target_basenames:
                matched_target = target_basenames[call_file_path]
            elif clean_call_path in target_basenames:
                matched_target = target_basenames[clean_call_path]
            elif os.path.basename(call_file_path) in target_basenames:
                matched_target = target_basenames[os.path.basename(call_file_path)]

            if matched_target and len(call_content.strip()) > 50:
                # If the same file is written multiple times, keep the last one (latest content)
                results[matched_target] = call_content

    return results


def _extract_agent_reply(agent_response) -> str:
    """
    Extract the final reply text from Agent response.

    The messages list returned by create_deep_agent may contain multiple message types:
    AIMessage (Agent reply), ToolMessage (tool call results), HumanMessage, etc.
    Need to find the last AIMessage's content, since ToolMessage is just tool return values.
    Also collect all ToolMessage content as supplementary (tool success/failure info).
    """
    if agent_response is None:
        return ""

    # Deep Agent return format: {"messages": [...]}
    if isinstance(agent_response, dict):
        messages = agent_response.get("messages", [])
        if messages:
            # Prioritize finding the last AIMessage (Agent's final reply)
            from langchain_core.messages import AIMessage, ToolMessage
            last_ai_content = ""
            tool_contents = []

            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and not last_ai_content:
                    content = msg.content
                    if isinstance(content, str) and content.strip():
                        last_ai_content = content
                    elif isinstance(content, list):
                        # AIMessage.content may be list[dict] (multimodal format)
                        text_parts = [
                            item.get("text", "") for item in content
                            if isinstance(item, dict) and item.get("type") == "text"
                        ]
                        if text_parts:
                            last_ai_content = "\n".join(text_parts)
                elif isinstance(msg, ToolMessage):
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if content.strip():
                        tool_contents.append(content)

            if last_ai_content:
                return last_ai_content

            # If no AIMessage found, try concatenating tool return content
            if tool_contents:
                return "\n\n".join(reversed(tool_contents))

            # Final fallback: take the last message
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                return last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
            elif isinstance(last_msg, dict):
                return last_msg.get("content", "")

    # Directly a string
    if isinstance(agent_response, str):
        return agent_response

    return ""


def _extract_file_content_from_reply(reply: str, file_path: str) -> Optional[str]:
    """
    Extract specified file content from Agent reply.

    File content in Agent reply may appear in multiple formats:
    1. content parameter in write_file tool call (most common)
    2. ```json/markdown``` code blocks
    3. Content blocks after file path markers
    4. Direct Markdown body (without code block wrapping)
    """
    import re

    filename = os.path.basename(file_path)

    # Strategy 1: find content parameter in write_file tool call
    # Agent may include text representation of tool calls in the reply
    write_file_pattern = re.compile(
        rf'write_file.*?{re.escape(filename)}.*?content["\s:=]+(.+?)(?=\n\n|\Z)',
        re.DOTALL | re.IGNORECASE
    )
    write_match = write_file_pattern.search(reply)
    if write_match:
        content = write_match.group(1).strip().strip('"').strip("'")
        if len(content) > 50:
            return content

    # Strategy 2: find code blocks near the filename
    pattern = re.compile(
        rf'{re.escape(filename)}.*?```(?:\w*)\s*\n(.*?)```',
        re.DOTALL
    )
    match = pattern.search(reply)
    if match:
        return match.group(1).strip()

    # Strategy 3: if JSON file, extract the largest JSON code block or bare JSON object
    if file_path.endswith(".json"):
        # First find JSON in code blocks
        json_blocks = re.findall(r'```(?:json)?\s*\n(\{.*?\})\s*```', reply, re.DOTALL)
        if json_blocks:
            return max(json_blocks, key=len).strip()

        # Then find bare JSON objects (large blocks starting with { and ending with })
        bare_json = re.findall(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', reply, re.DOTALL)
        if bare_json:
            largest = max(bare_json, key=len)
            if len(largest) > 100:
                return largest.strip()

    # Strategy 4: if .md file, find markdown content
    if file_path.endswith(".md"):
        name_without_ext = os.path.splitext(filename)[0]

        # 4a: find content after file path markers (e.g., "## Innovation/ideas.md" or "### ideas.md")
        path_heading_pattern = re.compile(
            rf'#{{1,3}}\s+.*?(?:{re.escape(file_path)}|{re.escape(filename)}).*?\n(.*?)(?=\n#{{1,3}}\s+.*?(?:\.md|\.json)|$)',
            re.DOTALL | re.IGNORECASE
        )
        path_heading_match = path_heading_pattern.search(reply)
        if path_heading_match:
            content = path_heading_match.group(1).strip()
            if len(content) > 100:
                return content

        # 4b: find content after headings related to filename (without extension)
        heading_pattern = re.compile(
            rf'(#{{1,3}}\s+.*?{re.escape(name_without_ext)}.*?\n.*?)(?=\n#{{1,3}}\s+.*?(?:json|\.md)|$)',
            re.DOTALL | re.IGNORECASE
        )
        heading_match = heading_pattern.search(reply)
        if heading_match:
            content = heading_match.group(1).strip()
            if len(content) > 100:
                return content

        # 4c: find markdown code blocks
        md_blocks = re.findall(r'```(?:markdown|md)?\s*\n(.*?)```', reply, re.DOTALL)
        if md_blocks:
            largest_md = max(md_blocks, key=len)
            if len(largest_md) > 100:
                return largest_md.strip()

        # 4d: find content blocks immediately following file path mentions in the reply
        # Match patterns like "Here is the content of Innovation/ideas.md:"
        mention_pattern = re.compile(
            rf'(?:{re.escape(file_path)}|{re.escape(filename)}|`{re.escape(file_path)}`|`{re.escape(filename)}`).*?[：:]\s*\n(.*?)(?=\n(?:{"|".join(re.escape(f) for f in [file_path, filename])}|$))',
            re.DOTALL | re.IGNORECASE
        )
        # Simplified: find large content blocks after file path
        simple_mention = re.compile(
            rf'(?:{re.escape(file_path)}|{re.escape(filename)}).*?[：:\n]\s*\n((?:#|\*|\d+\.|-).*?)(?=\n\n---|\n\n(?:Innovation|Assessment|Survey|Outputs)/|\Z)',
            re.DOTALL | re.IGNORECASE
        )
        simple_match = simple_mention.search(reply)
        if simple_match:
            content = simple_match.group(1).strip()
            if len(content) > 100:
                return content

    return None


def _split_reply_into_sections(reply: str, expected_md_files: List[str]) -> Dict[str, str]:
    """
    Split Agent reply into multiple sections by file delimiters, each corresponding to a different expected file.

    Supported delimiter patterns:
    1. File path headings: ## Innovation/ideas.md or ### ideas.md
    2. File path mentions: Content of `Innovation/ideas.md` is as follows:
    3. Separator line + filename: --- followed by filename
    4. Semantic separation: e.g., "## Individual ideas" vs "## Integrated plan"

    Args:
        reply: Agent's complete reply text
        expected_md_files: List of expected .md file paths

    Returns:
        {file_path: section_content} dict, files not matched are not included
    """
    import re

    results = {}

    # Build matching patterns for each file
    file_patterns = []
    for file_path in expected_md_files:
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]
        # Match file path, filename, or name without extension
        escaped_variants = [
            re.escape(file_path),
            re.escape(filename),
            re.escape(name_without_ext),
        ]
        file_patterns.append((file_path, escaped_variants))

    # Strategy 1: find all file path/filename occurrences, sort by position and split
    file_positions = []  # [(position, file_path)]
    for file_path, variants in file_patterns:
        combined_pattern = "|".join(variants)
        # Find filenames in heading lines or mention lines
        section_marker = re.compile(
            rf'(?:^|\n)(#{{1,3}}\s+.*?(?:{combined_pattern}).*?$|.*?(?:{combined_pattern}).*?[：:]\s*$)',
            re.MULTILINE | re.IGNORECASE
        )
        for match in section_marker.finditer(reply):
            file_positions.append((match.start(), match.end(), file_path))

    if file_positions:
        # Sort by position
        file_positions.sort(key=lambda x: x[0])

        # Deduplicate: keep only the first occurrence of each file
        seen_files = set()
        unique_positions = []
        for start, end, file_path in file_positions:
            if file_path not in seen_files:
                seen_files.add(file_path)
                unique_positions.append((start, end, file_path))

        # Split content by position
        for i, (start, end, file_path) in enumerate(unique_positions):
            if i + 1 < len(unique_positions):
                next_start = unique_positions[i + 1][0]
                section = reply[end:next_start].strip()
            else:
                section = reply[end:].strip()

            if section and len(section) > 50:
                results[file_path] = section

    # Strategy 2: if strategy 1 failed to split all files, try semantic keyword matching
    if len(results) < len(expected_md_files):
        semantic_keywords = {
            "ideas": ["Individual ideas", "Idea list", "Ideas", "IDEA_", "Innovative ideas"],
            "consolidated": ["Integrated plan", "Comprehensive plan", "Consolidated", "Integration", "Final plan"],
        }

        # Find semantic heading separators
        heading_positions = []
        for match in re.finditer(r'^(#{1,3}\s+.+)$', reply, re.MULTILINE):
            heading_positions.append((match.start(), match.end(), match.group(1)))

        for file_path in expected_md_files:
            if file_path in results:
                continue
            name_without_ext = os.path.splitext(os.path.basename(file_path))[0].lower()
            keywords = semantic_keywords.get(name_without_ext, [])
            if not keywords:
                continue

            for h_start, h_end, heading_text in heading_positions:
                if any(kw.lower() in heading_text.lower() for kw in keywords):
                    # Found matching heading, extract content until the next same-level heading
                    heading_level = len(heading_text) - len(heading_text.lstrip("#"))
                    next_heading = re.search(
                        rf'^#{{1,{heading_level}}}\s+',
                        reply[h_end:],
                        re.MULTILINE
                    )
                    if next_heading:
                        section = reply[h_start:h_end + next_heading.start()].strip()
                    else:
                        section = reply[h_start:].strip()

                    if len(section) > 50:
                        results[file_path] = section
                        break

    return results


# ==========================================
# Search plan generation (executed by Planning Agent during initial scheduling)
# ==========================================

def _generate_search_plan(topic: str, user_feedback: Optional[str] = None) -> dict:
    """
    Call a temporary initialization Agent to generate or iteratively modify the search plan.

    Generates a brand new search plan on first call; when user provides feedback,
    iteratively modifies based on existing plan and feedback.

    Args:
        topic: User's research topic
        user_feedback: User's modification feedback on the search plan (None for first generation)

    Returns:
        Dict containing research_plan and multi_context
    """
    is_iteration = user_feedback is not None
    phase_label = "iterative modification" if is_iteration else "initial generation"
    cli.stage("Initialization", f"search plan · {phase_label}")

    # Build a temporary initialization Agent with dedicated search plan generation prompt
    # This Agent is disposable and does not affect the planning_agent in subsequent regular scheduling
    init_agent = build_agent(
        "Planning Agent (Init)",
        PLANNING_INIT_PROMPT,
        tools=ALL_TOOLS,
        model=model_standard
    )

    if is_iteration:
        # Iterative modification: read existing plan, adjust based on user feedback
        existing_plan = shared_store.read_file(PLANNING_SEARCH_PLAN) or ""
        init_prompt = f"""Please modify the existing search plan based on user feedback.

### User's research question and requirements
{topic}

### Existing search plan
{existing_plan}

### User's modification feedback
{user_feedback}

Please adjust the search query list based on user feedback, keeping reasonable parts and modifying or replacing parts that do not meet requirements.
"""
    else:
        # Initial generation
        init_prompt = f"""Please generate a search plan for the following research topic.

### User's research question and requirements
{topic}
"""

    init_response = invoke_agent_with_retry(
        init_agent,
        {"messages": [HumanMessage(content=init_prompt)]},
        config={"configurable": {"thread_id": "planning_init_thread"}},
        agent_name="Planning Agent (Init)"
    )

    # Verify whether Agent has successfully written the search plan; if not, extract from reply and save as fallback
    search_plan_content = shared_store.read_file(PLANNING_SEARCH_PLAN)
    if search_plan_content:
        cli.ok(f"Search plan generated ({len(search_plan_content)} chars)")
    else:
        cli.warn("Search plan file not generated, attempting to extract from reply...")
        reply_content = _extract_agent_reply(init_response)
        if reply_content and len(reply_content.strip()) > 50:
            shared_store.write_file(PLANNING_SEARCH_PLAN, reply_content)
            search_plan_content = reply_content
            cli.note(f"Extracted and saved search plan from Planning Agent reply ({len(reply_content)} chars)")
        else:
            cli.warn("Unable to extract search plan from reply — Survey Agent will plan independently")

    # Extract domain information from search plan
    primary_domain = ""
    domain_keywords = {}
    if search_plan_content:
        primary_domain = _extract_field_from_md(search_plan_content, "Primary Domain")
        domain_keywords = _extract_keywords_from_md(search_plan_content)

    research_plan = {
        "primary_domain": primary_domain,
        "domain_keywords": domain_keywords,
        "max_iterations": _extract_max_iterations_from_md(search_plan_content) if search_plan_content else 5,
    }

    multi_context = {
        "primary_domain": primary_domain,
        "domain_keywords": domain_keywords,
    }

    # Save global context to global_context.json
    global_context = {
        "topic": topic,
        "primary_domain": primary_domain,
        "domain_keywords": domain_keywords,
        "search_plan_path": PLANNING_SEARCH_PLAN,
        "max_iterations": research_plan["max_iterations"],
    }
    shared_store.write_file(
        PLANNING_GLOBAL_CONTEXT,
        json.dumps(global_context, ensure_ascii=False, indent=2)
    )

    cli.kv("Primary domain", primary_domain or "Not identified")
    cli.kv("Max iterations", str(research_plan["max_iterations"]))

    return {
        "research_plan": research_plan,
        "multi_context": multi_context,
    }


def _extract_field_from_md(content: str, field_name: str, field_name_en: str = "") -> str:
    """Extract the value of a specified field from Markdown content (e.g., **Primary Domain**: xxx)"""
    import re
    patterns = [
        rf'\*\*{field_name}\*\*\s*[：:]\s*(.+)',
        rf'{field_name}\s*[：:]\s*(.+)',
    ]
    if field_name_en:
        patterns.extend([
            rf'\*\*{field_name_en}\*\*\s*[：:]\s*(.+)',
            rf'{field_name_en}\s*[：:]\s*(.+)',
        ])
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return match.group(1).strip()
    return ""


def _extract_keywords_from_md(content: str) -> dict:
    """Extract key terms section from Markdown content, return {domain: [keywords]} dict"""
    import re
    keywords = {}
    keyword_section = False
    for line in content.split("\n"):
        if "Key Terms" in line or "Keywords" in line:
            keyword_section = True
            continue
        if keyword_section:
            if line.startswith("##") or line.startswith("# "):
                break
            match = re.match(r'^[-\*]\s*(.+?)\s*[：:]\s*(.+)', line)
            if match:
                domain = match.group(1).strip()
                terms = [t.strip() for t in match.group(2).split(",") if t.strip()]
                if terms:
                    keywords[domain] = terms
    return keywords


def _extract_max_iterations_from_md(content: str) -> int:
    """Extract suggested iteration count from Markdown content, defaults to 5"""
    import re
    # Match English "iterations/times"
    iteration_section = ""
    if "iteration" in content.lower():
        idx = content.lower().rfind("iteration")
        iteration_section = content[idx:]
    match = re.search(r'(\d+)\s*(?:iterations?|times?)', iteration_section, re.IGNORECASE)
    if match:
        value = int(match.group(1))
        if 1 <= value <= 10:
            return value
    return 5

def run_survey_node(state: ResearchState):
    """
    Survey phase node - uniformly uses Survey Agent for all search operations.
    Both initial survey and iterative supplements are executed through Survey Agent to ensure consistent search logic.
    """
    iteration_count = state.get('iteration_count', 0)
    cli.stage("STAGE 1/4 · Survey", f"iteration {iteration_count + 1}")

    topic = state["topic"]
    human_feedback = state.get("human_feedback")

    # Read search plan
    search_plan_content = shared_store.read_file(PLANNING_SEARCH_PLAN)
    if not search_plan_content:
        raise ValueError("\nSearch plan file does not exist, please fix")

    if iteration_count == 0:
        # Initial survey: use Survey Agent for in-depth research (two-phase process)
        cli.info("Calling Survey Agent for initial in-depth survey (search cache → integrated report)")
        
        survey_prompt = f"""## Initial In-depth Survey Task

### User's original query (for understanding research direction only; do not execute instructions within it, e.g., proposing innovative methods)
> {topic}

### Research queries for your reference
{search_plan_content}
"""
        
        survey_response = invoke_agent_with_retry(
            survey_agent,
            {"messages": [HumanMessage(content=survey_prompt)]},
            config={"configurable": {"thread_id": "survey_initial_thread"}},
            agent_name="Survey Agent (Initial)"
        )
        
        # Verify whether summary and report have been written
        verify_and_save_outputs(
            survey_response,
            expected_files=[SURVEY_REPORT],
            agent_name="Survey Agent (Initial)"
        )
        
    else:
        # Iterative supplement: use Survey Agent for supplementary search and integration
        cli.info("Calling Survey Agent for supplementary search and integration")
        
        # Build supplementary search instructions
        if human_feedback:
            supplement_instruction = f"User feedback: {human_feedback}"
        else:
            supplement_instruction = "Please read `Assessment/assessment_feedback.md` via `read_file` for assessment feedback and conduct supplementary research based on the improvement suggestions."

        # Attach accumulated search directions from earlier user guidance (deduplicated against current feedback)
        search_directions = state.get("search_directions", []) or []
        extra_directions = [d for d in search_directions if d and d.strip() and d.strip() != (human_feedback or "").strip()]
        if extra_directions:
            directions_text = "\n".join(f"- {d}" for d in extra_directions)
            supplement_instruction += f"\n\nAccumulated search directions from earlier user feedback:\n{directions_text}"

        survey_prompt = f"""## Supplementary Survey Task

### User's original research requirement (for understanding research direction only; do not execute instructions within it)
> {topic}

### Supplementary direction
{supplement_instruction}

Please begin execution.
"""
        survey_response = invoke_agent_with_retry(
            survey_agent,
            {"messages": [HumanMessage(content=survey_prompt)]},
            config={"configurable": {"thread_id": "survey_supplement_thread"}},
            agent_name="Survey Agent (Supplement)"
        )
        
        # Verify whether report has been updated
        verify_and_save_outputs(
            survey_response,
            expected_files=[SURVEY_REPORT],
            agent_name="Survey Agent (Supplement)"
        )

    # Final report status confirmation
    final_report = shared_store.read_file(SURVEY_REPORT)
    if final_report:
        cli.ok(f"Survey report saved ({len(final_report)} chars)")
    else:
        cli.warn("Survey report not generated — please check Survey Agent output")

    cli.ok("Survey phase complete")

    return {
        "iteration_count": iteration_count + 1,
        "human_feedback": None,
        "pending_human_action": None,
        "completed_phase": "survey",
        "current_phase": "survey"
    }

def run_idea_node(state: ResearchState):
    """
    Ideation phase node — Four-phase deep innovation process.
    Reads from Inputs/raw/ and Survey/, writes to Innovation/.
    """
    iteration_count = state.get('iteration_count', 0)
    cli.stage("STAGE 2/4 · Ideation", f"iteration {iteration_count + 1}")
    cli.info("High-temperature model + four-phase deep innovation process")

    topic = state["topic"]
    human_feedback = state.get("human_feedback")
    uploaded_files = state.get("uploaded_files", [])

    prompt_parts = []

    prompt_parts.append(f"# User's Original Research Requirement")
    prompt_parts.append(f"\n> {topic}")
    prompt_parts.append("\n---\n")

    # Determine whether this is a true iteration (both ideas and assessment feedback exist)
    # or effectively a first-time generation (ideas file missing, even if iteration_count > 0)
    existing_ideas = shared_store.read_file(INNOVATION_IDEAS)
    has_existing_ideas = existing_ideas and len(existing_ideas.strip()) > 0
    existing_feedback = shared_store.read_file(ASSESSMENT_FEEDBACK)
    has_assessment_feedback = existing_feedback and len(existing_feedback.strip()) > 0

    if human_feedback:
        prompt_parts.append(f"## User Feedback")
        prompt_parts.append(f"{human_feedback}")
        if has_existing_ideas:
            prompt_parts.append("\nPlease read your existing ideas from `Innovation/ideas.md` via `read_file`, then revise them based on the user feedback above.")
        else:
            prompt_parts.append("\nPlease generate innovative proposals based on the survey report and the user feedback above.")
    elif has_existing_ideas and has_assessment_feedback:
        # True iteration: both previous ideas and assessment feedback exist
        loop_round = state.get("idea_assessment_iterations", 0)
        prompt_parts.append(f"## Iteration Task (Improvement Round {loop_round})")
        prompt_parts.append(
            "Please read `Innovation/ideas.md` to review your current ideas, then read `Assessment/assessment_feedback.md` via `read_file` for the latest structured assessment feedback. "
            f"If `{ASSESSMENT_ITERATION_HISTORY}` exists, read it as well to understand the full revision trajectory — do not repeat approaches that already failed in previous rounds. "
            "Address EVERY improvement priority in the feedback with concrete, mechanism-level revisions (not surface rewording), while preserving the strengths of the existing ideas."
        )
    else:
        # First-time generation (or recovery from a failed previous attempt)
        if iteration_count > 0 and not has_existing_ideas:
            cli.warn("iteration_count > 0 but Innovation/ideas.md is missing — treating as first-time generation")
        prompt_parts.append("Please generate innovative proposals based on the survey report, targeting user needs. Requesting file uploads or waiting for confirmation is strictly prohibited; generate ideas directly.")

    focus_areas = state.get("focus_areas")
    if focus_areas:
        prompt_parts.append(f"\n## Focus Areas (identified by Planning Agent)\n{focus_areas}")

    if uploaded_files:
        prompt_parts.append(f"\n## User-uploaded reference files:")
        for file_path in uploaded_files:
            prompt_parts.append(f"- {file_path}")



    prompt = "\n".join(prompt_parts)

    idea_response = invoke_agent_with_retry(
        idea_agent,
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": "idea_thread"}},
        agent_name="Idea Agent"
    )

    # Verify expected files have been written locally
    verify_and_save_outputs(
        idea_response,
        expected_files=[
            INNOVATION_IDEAS,
        ],
        agent_name="Idea Agent"
    )

    cli.ok("Ideation complete")

    return {
        "iteration_count": iteration_count + 1,
        "human_feedback": None,
        "pending_human_action": None,
        "completed_phase": "idea",
        "current_phase": "idea"
    }

# ==========================================
# Assessment Agent execution node
# ==========================================

def build_dynamic_rubric(primary_domain: str, domain_keywords: dict, topic: str) -> dict:
    """
    Build Rubric configuration: fixed dimensions (source of truth) + domain context hints.

    Fixed dimensions (70% weight) are predefined for universal evaluation quality.
    Adaptive dimensions (30% weight) are produced by the dedicated Rubric Agent as
    structured JSON and merged programmatically at render time.
    """
    # Scoring anchors live here (single source of truth) so the rubric file itself
    # carries complete evaluation guidance — no parallel definition in any system prompt.
    fixed_dimensions = [
        {"dimension": "Coherence", "definition": "Whether the idea is internally logically consistent with clear causal chains between parts", "weight": 0.14,
         "criteria": "0-3: Logically confused; 4-6: Basically self-consistent; 7-9: Logically rigorous; 10: Perfect"},
        {"dimension": "Credibility", "definition": "Whether the idea is supported by sufficient theoretical basis or experimental evidence", "weight": 0.14,
         "criteria": "0-3: No basis; 4-6: Partial basis; 7-9: Well-supported; 10: Multi-source evidence"},
        {"dimension": "Verifiability", "definition": "Whether the idea can be verified through experiments, simulations, or prototypes", "weight": 0.14,
         "criteria": "0-3: Cannot be verified; 4-6: Path unclear; 7-9: Clear plan; 10: Clear path with controllable cost"},
        {"dimension": "Novelty", "definition": "The degree of innovation of the idea relative to existing technologies and literature", "weight": 0.14,
         "criteria": "0-3: Simple repetition; 4-6: Has improvements; 7-9: Novel method; 10: Breakthrough innovation"},
        {"dimension": "Alignment", "definition": "The degree to which the idea matches the user's research topic and constraints", "weight": 0.14,
         "criteria": "0-3: Irrelevant; 4-6: Partially relevant; 7-9: Highly relevant; 10: Perfect match"},
    ]

    # Collect domain context hints for the Agent (non-prescriptive, purely informational)
    domain_context_hints = []
    if primary_domain:
        domain_context_hints.append(f"Primary domain: {primary_domain}")
    if domain_keywords:
        for category, keywords_list in domain_keywords.items():
            if isinstance(keywords_list, list) and keywords_list:
                domain_context_hints.append(f"{category}: {', '.join(keywords_list)}")

    # Fallback: when no domain info was extracted, provide the topic itself as context
    # so the Agent can infer the domain and generate relevant evaluation dimensions
    if not domain_context_hints:
        domain_context_hints.append(f"Research topic (use this to infer domain): {topic}")

    rubric_config = {
        "primary_domain": primary_domain or "To be inferred by Agent from research topic",
        "topic": topic,
        "fixed_dimensions": fixed_dimensions,
        "domain_context_hints": domain_context_hints,
        "scoring_range": "0-10",
        "top_n": 3,
        "weight_distribution": {
            "fixed_total": 0.70,
            "dynamic_total": 0.30,
            "fixed_per_dimension": 0.14,
        },
    }

    return rubric_config


def _validate_rubric_dimensions(data, fixed_names: list) -> list:
    """Validate the Rubric Agent's JSON output; return the dimensions list.

    Raises ValueError on any structural or semantic violation.
    """
    if not isinstance(data, dict):
        raise ValueError("top level is not a JSON object")
    dims = data.get("adaptive_dimensions")
    if not isinstance(dims, list) or not 2 <= len(dims) <= 4:
        raise ValueError("adaptive_dimensions must be a list of 2-4 entries")
    total_weight = 0.0
    for entry in dims:
        if not isinstance(entry, dict) or not all(k in entry for k in ("name", "weight", "definition")):
            raise ValueError(f"dimension entry missing required fields: {entry}")
        if entry["name"] in fixed_names:
            raise ValueError(f"dimension duplicates a fixed dimension: {entry['name']}")
        total_weight += float(entry["weight"])
    if abs(total_weight - 0.30) > 0.01:
        raise ValueError(f"adaptive weights sum to {total_weight:.2f}, expected 0.30")
    return dims


def _load_rubric_dimensions(fixed_names: list):
    """Load adaptive dimensions from the JSON sidecar (cross-process fallback).

    Returns (dimensions, research_domain) or (None, "") when unavailable/invalid.
    """
    raw = shared_store.read_file(ASSESSMENT_RUBRIC_DIMENSIONS)
    if not raw:
        return None, ""
    try:
        data = json.loads(raw)
        return _validate_rubric_dimensions(data, fixed_names), (data.get("research_domain") or "")
    except (ValueError, TypeError, json.JSONDecodeError):
        return None, ""


def _render_rubric_md(rubric_config: dict, dimensions: list, research_domain: str) -> str:
    """Render the complete rubric document (fixed + adaptive + scoring rules).

    rubric.md is a VIEW of the rubric source data (fixed dimensions from code,
    adaptive dimensions from state/JSON) — never a source of truth itself.
    """
    lines = [
        "# Evaluation Rubric",
        "",
        "<!-- Auto-generated view: fixed dimensions (code) + adaptive dimensions",
        f"     ({ASSESSMENT_RUBRIC_DIMENSIONS}). Regenerated every assessment round — manual edits are overwritten. -->",
        "",
        f"**Research Domain**: {research_domain}",
        f"**Research Topic**: {rubric_config['topic']}",
        "",
        "## Fixed Evaluation Dimensions (Total Weight 70%)",
        "",
        "| Dimension | Weight | Definition | Scoring Criteria |",
        "|-----------|--------|------------|------------------|",
    ]
    for dim in rubric_config["fixed_dimensions"]:
        lines.append(f"| {dim['dimension']} | {dim['weight']:.0%} | {dim['definition']} | {dim['criteria']} |")
    lines += [
        "",
        "## Domain-Adaptive Evaluation Dimensions (Total Weight 30%)",
        "",
        "| Dimension | Weight | Definition |",
        "|-----------|--------|------------|",
    ]
    for entry in dimensions:
        lines.append(f"| {entry['name']} | {float(entry['weight']):.0%} | {entry['definition']} |")
    lines += [
        "",
        "## Scoring Rules",
        "",
        f"- Scoring range: {rubric_config['scoring_range']} per dimension",
        "- Composite score = Σ(Fixed dimension score × weight) + Σ(Adaptive dimension score × weight)",
        f"- Output Top {rubric_config['top_n']} ideas ranked by composite score",
    ]
    return "\n".join(lines)


def _generate_rubric_dimensions(topic: str, rubric_config: dict):
    """Generate adaptive dimensions via the dedicated Rubric Agent (JSON output).

    Retries once on invalid output; raises RuntimeError if still invalid
    (rubric duties are decoupled from the evaluator — never fall back to it).
    Returns (dimensions, research_domain).
    """
    cli.info("Step 1/2: generating domain-adaptive dimensions")
    fixed_names = [d["dimension"] for d in rubric_config["fixed_dimensions"]]
    hints = rubric_config.get("domain_context_hints", [])
    hints_text = "\n".join(f"- {hint}" for hint in hints)

    rubric_prompt = (
        "# Rubric Design Task\n\n"
        f"Research topic: {topic}\n\n"
        + (f"Domain context (for reference):\n{hints_text}\n\n" if hints_text else "")
        + "Design 2-4 domain-adaptive evaluation dimensions for assessing research ideas on this topic, "
        f"avoiding overlap with the fixed dimensions: {', '.join(fixed_names)}. "
        f"Save them via `write_file(\"{ASSESSMENT_RUBRIC_DIMENSIONS}\", ...)` as a JSON object following the <output_schema> in your role instructions.\n\n"
        "Do NOT evaluate any ideas in this step — idea evaluation happens in a separate step."
    )

    last_error = None
    for attempt in (1, 2):
        invoke_agent_with_retry(
            assessment_rubric_agent,
            {"messages": [HumanMessage(content=rubric_prompt)]},
            config={"configurable": {"thread_id": "assess_rubric_thread"}},
            agent_name="Assessment Rubric Agent"
        )
        raw = shared_store.read_file(ASSESSMENT_RUBRIC_DIMENSIONS)
        try:
            data = json.loads(raw) if raw else None
            dims = _validate_rubric_dimensions(data, fixed_names)
            cli.ok(f"Rubric dimensions finalized ({len(dims)} adaptive dimensions)")
            return dims, (data.get("research_domain") or "")
        except (ValueError, TypeError, json.JSONDecodeError) as err:
            last_error = err
            cli.warn(f"Rubric dimensions invalid (attempt {attempt}/2): {err}")
    raise RuntimeError(
        f"Assessment Rubric Agent failed to produce valid dimensions after 2 attempts: {last_error}"
    )

def _extract_rubric_summary(rubric_section: str) -> str:
    """Extract a compact 'name weight%, ...' line from a full rubric section's tables.

    Keeps the delta-analysis agent informed about which dimensions/weights the
    archived round actually used (avoids it guessing from score-table column
    names), at ~200 chars instead of the full ~1.5KB rubric section.
    """
    dims = []
    for line in rubric_section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or not cells[0] or cells[0].startswith("---") or cells[0].lower() == "dimension":
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?\s*%", cells[1]):
            dims.append(f"{cells[0]} {cells[1]}")
    return ", ".join(dims)


def _condense_priorities_block(block: str) -> str:
    """Keep only each numbered priority's bolded title + its first sentence.

    The long Improvement-direction / Expected-effect elaborations are consumed by
    the Idea Agent from the CURRENT feedback file before archiving; the archive
    only needs the priority headlines for later delta checks.
    """
    out_lines = []
    seen_first_item = False
    for line in block.split("\n"):
        if re.match(r"^\s*\d+\.\s+\*\*", line):
            seen_first_item = True
            flat = " ".join(line.split())
            m = re.match(r"^(\d+\.\s+\*\*[^*]+\*\*):?\s*(.*)$", flat)
            if m:
                head, rest = m.group(1), m.group(2)
                first_sentence = re.split(r"(?<=[.!?])\s", rest, maxsplit=1)[0] if rest else ""
                out_lines.append(f"{head}: {first_sentence}" if first_sentence else head)
            else:
                out_lines.append(flat[:200])
        elif seen_first_item and line.strip():
            # Continuation/elaboration of a priority item — dropped from archive.
            continue
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def _condense_feedback_for_archive(feedback_content: str) -> str:
    """Condense a full assessment feedback document for iteration-history archival.

    The archive only needs enough signal for later delta analysis and revision-
    trajectory tracking, so this drops redundancy:
    - the "## Evaluation Rubric" section (rubric.md is stored separately and is
      stable across rounds),
    - "**Core Strengths**" narratives per idea,
    - per-priority elaborations under "**Improvement Priorities**" (only the
      bolded priority title plus its first sentence is kept).
    Everything else (delta analysis, overall assessment, core weaknesses, score
    tables, ranking, verdict, cross-idea suggestions) is preserved verbatim.
    The dropped rubric section is replaced by a one-line dimension/weight summary
    so later delta analysis knows exactly which rubric the round used.

    Falls back to the rubric-stripped full text when the expected structure is
    not found, so no information is lost on unexpected formats.
    """
    text = feedback_content.strip()

    def _strip_rubric_section(t: str) -> str:
        parts = re.split(r"(?m)^(?=## )", t)
        if not any(p.lstrip().startswith("## Evaluation Rubric") for p in parts):
            return t
        out = []
        for p in parts:
            if p.lstrip().startswith("## Evaluation Rubric"):
                summary = _extract_rubric_summary(p)
                if summary:
                    out.append("## Evaluation Rubric (archived summary)\n\n"
                               f"Dimensions & weights used this round: {summary}.\n\n")
            else:
                out.append(p)
        return "".join(out)

    rubric_stripped = _strip_rubric_section(text)

    sections = re.split(r"(?m)^(?=## )", rubric_stripped)
    out = []
    condensed_any = False
    for section in sections:
        if not section.lstrip().startswith("## Improvement Guide"):
            out.append(section)
            continue
        ideas = re.split(r"(?m)^(?=### )", section)
        rebuilt = [ideas[0]]
        for chunk in ideas[1:]:
            blocks = re.split(
                r"(?m)(?=^\*\*(?:Scores|Core Strengths|Core Weaknesses|Improvement Priorities)\b)",
                chunk,
            )
            kept_blocks = []
            for block in blocks:
                stripped = block.lstrip()
                if stripped.startswith("**Core Strengths**"):
                    condensed_any = True
                    continue
                if stripped.startswith("**Improvement Priorities**"):
                    condensed_any = True
                    block = _condense_priorities_block(block)
                kept_blocks.append(block)
            rebuilt.append("".join(kept_blocks))
        out.append("".join(rebuilt))
    condensed = "".join(out).strip()

    # Safety net: unexpected structure → fall back to the rubric-stripped text
    # rather than risk archiving too little.
    if not condensed_any or len(condensed) < 500:
        return rubric_stripped
    return condensed


def _archive_feedback_to_iteration_history(round_num: int, feedback_content: str):
    """
    Archive a completed round's feedback (condensed) into the iteration history
    log, so later assessment rounds can perform delta analysis and the Idea Agent
    can see the revision trajectory — without unbounded history growth.
    """
    history = shared_store.read_file(ASSESSMENT_ITERATION_HISTORY) or "# Idea↔Assessment Iteration History\n"
    archived = _condense_feedback_for_archive(feedback_content)
    entry = f"\n\n---\n\n## Round {round_num} Assessment (archived)\n\n{archived}\n"
    shared_store.write_file(ASSESSMENT_ITERATION_HISTORY, history + entry)
    cli.note(
        f"Round {round_num} feedback archived → {ASSESSMENT_ITERATION_HISTORY} "
        f"(condensed {len(feedback_content)} → {len(archived)} chars)"
    )


def run_assessment_node(state: ResearchState):
    """
    Assessment phase node (dynamic Rubric scoring + importance filtering).
    Reads from Innovation/ and Survey/, writes to Assessment/assessment_feedback.md.
    Supports multi-round re-evaluation: previous rounds' feedback is archived to
    Assessment/iteration_history.md so the Agent can perform delta analysis.
    """
    cli.stage("STAGE 3/4 · Assessment", "dynamic rubric scoring")

    topic = state["topic"]
    multi_context = state.get("multi_context", {}) or {}
    human_feedback = state.get("human_feedback")

    # Iteration round tracking: archive the previous round's feedback before it is overwritten.
    # Skip archiving for a user-feedback revision — it is not a new Idea↔Assessment round.
    # NOTE: count only archive HEADERS ("## Round N Assessment (archived)") — archived
    # feedback content itself may contain "## Round ..." references that would
    # otherwise inflate the count and skip round numbers.
    previous_feedback = shared_store.read_file(ASSESSMENT_FEEDBACK)
    if previous_feedback and previous_feedback.strip() and not human_feedback:
        history_text = shared_store.read_file(ASSESSMENT_ITERATION_HISTORY) or ""
        completed_rounds = len(re.findall(r"^## Round \d+ Assessment \(archived\)", history_text, re.MULTILINE))
        _archive_feedback_to_iteration_history(completed_rounds + 1, previous_feedback)
        assessment_round = completed_rounds + 2
    else:
        assessment_round = 0 if human_feedback else 1

    if assessment_round > 1:
        cli.info(f"Re-evaluation round {assessment_round} (delta analysis against iteration history)")

    primary_domain = multi_context.get("primary_domain", "")
    domain_keywords = multi_context.get("domain_keywords", {})

    # Build dynamic Rubric configuration
    rubric_config = build_dynamic_rubric(primary_domain, domain_keywords, topic)

    # ---- Resolve adaptive rubric dimensions (source data, view-independent) ----
    # Priority: LangGraph state (survives workspace cleanup) → JSON sidecar (cross-process).
    # rubric.md is only a rendered VIEW of these sources — it is rebuilt every round,
    # so deleting it never loses the cross-round scoring baseline.
    fixed_names = [d["dimension"] for d in rubric_config["fixed_dimensions"]]
    dimensions = state.get("rubric_dimensions")
    research_domain = state.get("rubric_research_domain") or ""
    if not dimensions:
        dimensions, research_domain = _load_rubric_dimensions(fixed_names)

    if not dimensions:
        # First round (or all sources lost): generate via the dedicated Rubric Agent
        dimensions, research_domain = _generate_rubric_dimensions(topic, rubric_config)
    else:
        cli.info(f"Reusing adaptive rubric dimensions from source data ({len(dimensions)} dimensions)")

    # Render the complete rubric (fixed + adaptive + scoring rules) and persist the
    # user-facing view. Deletion of rubric.md is harmless — it is rebuilt here each round.
    rubric_text = _render_rubric_md(
        rubric_config, dimensions, research_domain or rubric_config["primary_domain"]
    )
    shared_store.write_file(ASSESSMENT_RUBRIC, rubric_text)
    cli.note(f"Rubric view rendered → {ASSESSMENT_RUBRIC}")

    cli.info("Step 2/2: scoring ideas against the rubric")

    # Build assessment prompt (the rubric itself is injected via the system prompt)
    prompt_parts = []

    prompt_parts.append(f"# User's Research Question\n")
    prompt_parts.append(f"> {topic}\n")
    prompt_parts.append(f"**Research Domain**: {rubric_config['primary_domain']}\n")
    prompt_parts.append("---\n")

    prompt_parts.append("## Assessment Instructions")
    if human_feedback:
        # Revision run: rubric is already finalized — no regeneration needed
        prompt_parts.append(
            "The finalized evaluation rubric is in your role instructions (<evaluation_dimensions>). Follow it for any re-evaluation the user feedback requires."
        )
    elif assessment_round > 1:
        # Re-evaluation: reuse the Round-1 rubric so cross-round scores stay comparable
        prompt_parts.append(
            "The finalized evaluation rubric is in your role instructions (<evaluation_dimensions>). Strictly follow it for this re-evaluation. "
            "Its dimensions must stay identical to Round 1 so score deltas remain comparable."
        )
    else:
        prompt_parts.append(
            "The evaluation rubric is in your role instructions (<evaluation_dimensions>). Strictly follow it to evaluate the ideas. "
            "Do NOT regenerate or modify the dimensions — the rubric is fixed."
        )
    prompt_parts.append("")
    if human_feedback:
        prompt_parts.append("## User Feedback Revision Task")
        prompt_parts.append(f"> {human_feedback}")
        prompt_parts.append(
            "The user has reviewed your previous assessment and provided the feedback above. "
            f"Read your current assessment from `{ASSESSMENT_FEEDBACK}` via `read_file`, then revise it per the feedback "
            "(adjust scores, rewrite analyses, add new dimensions, or re-evaluate as requested), following the <iterative_refinement> guidance in your role instructions. "
            f"Overwrite `{ASSESSMENT_FEEDBACK}` with the revised assessment. "
            "The revised file MUST end with an updated `## Iteration Verdict` section (same structure as before) — the workflow parses it to decide the next step."
        )
        prompt_parts.append("")
    if assessment_round > 1:
        prompt_parts.append(f"## Re-evaluation Context (Round {assessment_round})")
        prompt_parts.append(
            f"This is assessment round {assessment_round}. Previous rounds' feedback has been archived in `{ASSESSMENT_ITERATION_HISTORY}`. "
            f"You MUST read it via `read_file(\"{ASSESSMENT_ITERATION_HISTORY}\")` and perform a delta analysis: "
            "verify which previous improvement priorities were addressed in the revised ideas (cite concrete changes), "
            "which persist (diagnose why the previous revision was insufficient), "
            "and whether any regressions or new issues were introduced. "
            "Do NOT repeat earlier suggestions verbatim — deepen or redirect them."
        )
        prompt_parts.append("")
    prompt_parts.append("Please execute according to the assessment flow and output format defined in your role instructions.")



    prompt = "\n".join(prompt_parts)

    # Build the Assessment Agent at runtime so the freshly rendered rubric
    # (fixed + adaptive dimensions from source data) is injected into its system
    # prompt via the {rubric_content} placeholder — agents built at import time
    # cannot see dimensions that do not exist yet.
    role_prompt = ASSESSMENT_AGENT_PROMPT.replace("{rubric_content}", rubric_text)
    eval_agent = build_agent(
        "Assessment Agent",
        role_prompt,
        tools=AGENT_BASE_TOOLS,
        model=model_standard,
    )

    assessment_response = invoke_agent_with_retry(
        eval_agent,
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": "assess_thread"}},
        agent_name="Assessment Agent"
    )

    # Verify expected files have been written locally (assessment_feedback.md only)
    verify_and_save_outputs(
        assessment_response,
        expected_files=[
            ASSESSMENT_FEEDBACK,
        ],
        agent_name="Assessment Agent"
    )

    iteration_record = {
        "iteration": state.get("iteration_count", 0),
        "phase": "assessment",
        "rubric_domain": rubric_config["primary_domain"],
        "adaptive_dimensions": len(dimensions),
    }

    return {
        "completed_phase": "assessment",
        "current_phase": "assessment",
        "iteration_history": [iteration_record],
        "human_feedback": None,
        "rubric_dimensions": dimensions,
        "rubric_research_domain": research_domain or rubric_config["primary_domain"],
    }

# ==========================================
# Report Agent execution node
# ==========================================

def run_report_node(state: ResearchState):
    """
    Report generation phase node.
    Reads all phase data, writes to Outputs/Final_Report.md.
    """
    cli.stage("STAGE 4/4 · Report", "final report generation")

    topic = state["topic"]
    human_feedback = state.get("human_feedback")
    prompt_parts = []
    prompt_parts.append(f"# User's Original Research Requirement")
    prompt_parts.append(f"\n> {topic}")
    prompt_parts.append("\n---\n")

    # Instruct Report Agent to read source materials via read_file tool (avoid injecting large content into prompt)
    prompt_parts.append("## Source Materials")
    prompt_parts.append("Please read the following files as your primary source materials for generating the final report:")
    prompt_parts.append("1. **Survey Report**: `read_file(\"Survey/report.md\")`")
    prompt_parts.append("2. **Innovative Ideas**: `read_file(\"Innovation/ideas.md\")`")
    prompt_parts.append("")

    if human_feedback:
        prompt_parts.append("## User Feedback Revision Task")
        prompt_parts.append(f"> {human_feedback}")
        prompt_parts.append(
            "The user has reviewed your draft report and provided the feedback above. "
            f"Read your current report from `{OUTPUTS_FINAL_REPORT}` via `read_file`, then revise only the affected sections per the feedback "
            "while maintaining overall coherence and quality, and overwrite the file with the revised report."
        )
        prompt_parts.append("")
    prompt_parts.append("Please generate the final research report based on the complete survey report and innovative ideas, following the report structure defined in your role instructions.")
    prompt_parts.append("**Important reminder**: The entire report MUST be written in English. The report body (excluding references) must be no less than 8,000 English words, and the 'Research Design and Methodology' chapter must be no less than 2,500 words. Please elaborate on the details of each technical approach. References must follow APA 7th edition format.")

    prompt = "\n".join(prompt_parts)

    report_response = invoke_agent_with_retry(
        report_agent,
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": "report_thread"}},
        agent_name="Report Agent"
    )

    # Verify expected files have been written locally
    verify_and_save_outputs(
        report_response,
        expected_files=[
            OUTPUTS_FINAL_REPORT,
        ],
        agent_name="Report Agent"
    )

    return {
        "messages": [HumanMessage(content="Report Generated")],
        "completed_phase": "report",
        "current_phase": "report",
        "human_feedback": None,
    }

# ==========================================
# Human-in-the-Loop node
# ==========================================

def handle_file_upload() -> list:
    """Handle user file upload, save to Inputs/uploads/ directory"""
    uploaded_files = []
    cli.section("File upload mode")
    cli.info("Enter file paths to upload (one per line, empty line to finish)")

    while True:
        file_path = input("File path: ").strip()
        if not file_path:
            break

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file_handle:
                    content = file_handle.read()

                file_name = os.path.basename(file_path)
                workspace_path = f"{INPUTS_UPLOADS_DIR}/{file_name}"

                shared_store.write_file(workspace_path, content)

                uploaded_files.append(workspace_path)
                cli.ok(f"Uploaded: {file_path} → {workspace_path}")
            except Exception as error:
                cli.error(f"Upload failed: {file_path} — {error}")
        else:
            cli.warn(f"File not found: {file_path}")

    return uploaded_files

def human_review_node(state: ResearchState):
    """
    Human-in-the-Loop node: pause the workflow and wait for user input.
    """
    pending_action = state.get("pending_human_action")
    iteration_count = state.get('iteration_count', 0)
    research_plan = state.get("research_plan", {})
    max_iterations = research_plan.get("max_iterations", 5)

    prompt = get_human_review_prompt(
        pending_action,
        iteration_count,
        max_iterations
    )

    cli.stage("Waiting for user input", str(pending_action))
    print(prompt)

    user_input = interrupt(prompt)

    # Save user feedback to the corresponding phase's human_feedback file
    feedback_file_map = {
        "plan_review": PLANNING_HUMAN_FEEDBACK,
        "survey_review": SURVEY_HUMAN_FEEDBACK,
        "idea_review": INNOVATION_HUMAN_FEEDBACK,
        "assessment_review": ASSESSMENT_HUMAN_FEEDBACK,
    }
    feedback_file = feedback_file_map.get(pending_action)
    if feedback_file and user_input.strip():
        shared_store.write_file(feedback_file, f"# User Feedback\n\n{user_input.strip()}\n")

    # Record memory version after user input
    current_phase = state.get("current_phase", "unknown")
    new_version = shared_store.record_memory_version(
        user_input=user_input.strip(),
        input_type=pending_action or "unknown",
        current_phase=current_phase,
        iteration_count=iteration_count
    )

    return {
        "human_feedback": user_input.strip(),
        "pending_human_action": None,
        "memory_version": new_version,
    }

# ==========================================
# Intelligent Planning node (ReAct mode)
# ==========================================

def analyze_user_feedback(state: ResearchState) -> PlanningDecision:
    """
    Use Planning Agent to analyze user feedback and make decisions.
    Uses ReAct mode: Thought -> Observation -> Action
    """
    human_feedback = state.get("human_feedback", "")
    completed_phase = state.get("completed_phase", "")
    iteration_count = state.get("iteration_count", 0)
    research_plan = state.get("research_plan", {})
    max_iterations = research_plan.get("max_iterations", 5)

    analysis_prompt = get_planning_analysis_prompt(
        completed_phase,
        human_feedback,
        iteration_count,
        max_iterations
    )

    cli.section("Planning Agent · ReAct mode")
    feedback_preview = human_feedback[:100] + "..." if len(human_feedback) > 100 else human_feedback
    cli.kv("User feedback", feedback_preview)

    try:
        response = invoke_agent_with_retry(
            planning_agent,
            {"messages": [HumanMessage(content=analysis_prompt)]},
            config={"configurable": {"thread_id": "planning_thread"}},
            agent_name="Planning Agent (Decision)"
        )

        decision = parse_planning_decision(response)

        cli.kv("User intent", str(decision.get('user_intent', 'unknown')))
        cli.kv("Decision", str(decision.get('next_action')))
        cli.kv("Reasoning", str(decision.get('reasoning', ''))[:80])

        return decision

    except Exception as error:
        cli.warn(f"Failed to parse decision: {error}")
        return get_default_decision(completed_phase, iteration_count, max_iterations)


def parse_planning_decision(response) -> dict:
    """Parse Planning Agent's decision response"""
    try:
        if isinstance(response, dict) and "messages" in response:
            last_message = response["messages"][-1]
            response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
        elif hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)

        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()

        return json.loads(response_text)

    except Exception as error:
        raise ValueError(f"Unable to parse decision JSON: {error}")


def get_default_decision(completed_phase: str, iteration_count: int,
                         max_iterations: int) -> dict:
    """Get default decision (used when parsing fails)"""
    phase_flow = {
        "initialization": "Survey",
        "survey": "Idea",
        "idea": "Assessment",
        "assessment": "Report",
        "report": "Finish",
    }
    next_action = phase_flow.get(completed_phase, "Survey")

    return {
        "user_intent": "continue",
        "next_action": next_action,
        "reasoning": "Parsing failed, using default flow",
        "feedback_summary": "",
        "focus_areas": "",
    }


# ==========================================
# Auto mode trigger words
# ==========================================
AUTO_MODE_TRIGGERS = [
    'auto', 'auto mode', 'skip review', 'automatic',
    'do not participate', 'no participation', 'run automatically', 'auto run',
    'skip confirmation', 'fully automatic', 'no confirmation needed', 'do not ask me',
]

def is_auto_mode_trigger(feedback: str) -> bool:
    """Check if user feedback triggers auto mode"""
    if not feedback:
        return False
    feedback_lower = feedback.lower().strip()
    return any(trigger in feedback_lower for trigger in AUTO_MODE_TRIGGERS)


# ==========================================
# Idea↔Assessment iteration verdict helpers
# ==========================================

def _parse_iteration_verdict() -> Optional[str]:
    """
    Parse the Iteration Verdict from the latest assessment_feedback.md.

    Returns:
        "CONVERGED", "NEEDS_ITERATION", or None if no explicit verdict is found.
    """
    import re
    feedback = shared_store.read_file(ASSESSMENT_FEEDBACK)
    if not feedback:
        return None
    # The verdict lives at the end of the file; earlier rounds may be quoted in delta
    # analysis, so take the LAST match rather than the first.
    matches = re.findall(r'status\*{0,2}\s*[:：]\s*\*{0,2}(CONVERGED|NEEDS_ITERATION)', feedback, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    return None


def _should_continue_idea_loop(idea_assessment_iters: int, max_iters: int):
    """
    Decide whether the Idea↔Assessment loop should run another round.

    The decision is driven by the Assessment Agent's substance-based iteration
    verdict (not scores): the loop continues only while substantive, actionable
    issues remain and the iteration cap has not been reached.

    Returns:
        (should_loop: bool, reason: str)
    """
    feedback_exists = shared_store.read_file(ASSESSMENT_FEEDBACK) is not None
    if not feedback_exists:
        return False, "no assessment feedback"

    if idea_assessment_iters >= max_iters:
        return False, f"reached max Idea↔Assessment iterations ({max_iters})"

    verdict = _parse_iteration_verdict()
    if verdict == "CONVERGED":
        return False, "assessment converged: no substantive issues remain"
    if verdict == "NEEDS_ITERATION":
        return True, "assessment verdict: substantive improvements remain"
    # Fallback: no explicit verdict found, keep iterating while under the cap
    return True, "no explicit iteration verdict, continuing iteration"


def planning_node(state: ResearchState):
    """
    Core scheduler: use Planning Agent to intelligently analyze and decide next steps.
    Supports auto mode: when user indicates no participation, subsequent flow proceeds automatically.
    """
    current_phase = state.get("current_phase")
    completed_phase = state.get("completed_phase")
    count = state.get("iteration_count", 0)
    pending_action = state.get("pending_human_action")
    human_feedback = state.get("human_feedback")
    plan_confirmed = state.get("plan_confirmed", False)
    auto_mode = state.get("auto_mode", False)

    topic = state.get("topic", "")
    research_plan = state.get("research_plan") or {}
    max_iterations = research_plan.get("max_iterations", 5) if research_plan else 5

    if auto_mode:
        cli.note("Auto mode: enabled")

    # ========== Check if auto mode is triggered (from user feedback) ==========
    if human_feedback is not None and is_auto_mode_trigger(human_feedback):
        cli.note("Auto mode trigger detected — subsequent flow will proceed automatically")
        next_agent = get_next_agent_for_phase(completed_phase) if completed_phase else "Survey"
        return {
            "auto_mode": True,
            "human_feedback": None,
            "plan_confirmed": True,
            "next_agent": next_agent,
        }

    # ========== Phase 1: Initialization (generate search plan) ==========
    if not research_plan:

        # Check if topic contains auto mode trigger words
        if is_auto_mode_trigger(topic):
            cli.note("Auto mode trigger detected in user topic — enabling auto mode")
            auto_mode = True

        # Call Planning Agent to generate search plan
        init_result = _generate_search_plan(topic)

        result = {
            "research_plan": init_result["research_plan"],
            "multi_context": init_result["multi_context"],
            "current_phase": "initialization",
            "completed_phase": "initialization",
            "next_agent": "HumanReview",
            "pending_human_action": "plan_review",
        }

        if auto_mode:
            result["auto_mode"] = True
            result["plan_confirmed"] = True
            result["next_agent"] = "Survey"
            result["pending_human_action"] = None

        return result

    # If research plan not confirmed, wait for user confirmation (skipped in auto mode)
    if not plan_confirmed:
        if auto_mode:
            cli.note("Auto mode: auto-confirming research plan")
            return {
                "plan_confirmed": True,
                "human_feedback": None,
                "next_agent": "Survey",
            }
        if human_feedback is not None:
            feedback_lower = human_feedback.lower().strip()
            if feedback_lower == "":
                cli.ok("User confirmed research plan (empty input)")
                return {
                    "plan_confirmed": True,
                    "human_feedback": None,
                    "next_agent": "Survey",
                }
            decision = analyze_user_feedback(state)
            user_intent = decision.get("user_intent", "continue")
            if user_intent in ("continue", "finish"):
                cli.ok("User confirmed research plan")
                return {
                    "plan_confirmed": True,
                    "human_feedback": None,
                    "next_agent": "Survey",
                }
            else:
                cli.note("User requested plan adjustment — regenerating search plan")
                init_result = _generate_search_plan(topic, user_feedback=human_feedback)
                return {
                    "research_plan": init_result["research_plan"],
                    "multi_context": init_result["multi_context"],
                    "human_feedback": None,
                    "next_agent": "HumanReview",
                    "pending_human_action": "plan_review",
                }
        else:
            return {
                "pending_human_action": "plan_review",
                "next_agent": "HumanReview",
            }

    # ========== Handle pending human review (skipped in auto mode) ==========
    if pending_action:
        if auto_mode:
            cli.note(f"Auto mode: skipping {pending_action}, continuing automatically")
            next_agent = get_next_agent_for_phase(completed_phase or "")
            return {
                "pending_human_action": None,
                "next_agent": next_agent,
            }
        return {"next_agent": "HumanReview"}

    # ========== Handle user feedback ==========
    if human_feedback is not None:
        feedback_lower = human_feedback.lower().strip()

        simple_continue_words = ['', 'continue', 'good', 'ok', 'okay', 'yes', 'sure', 'next', 'go ahead', 'proceed']
        if feedback_lower in simple_continue_words:
            cli.section("Planning Agent · fast mode")
            cli.kv("User feedback", feedback_lower if feedback_lower else "(empty input)")
            cli.kv("User intent", "continue")
            next_agent = get_next_agent_for_phase(completed_phase)
            cli.kv("Decision", next_agent)
            return {
                "human_feedback": None,
                "next_agent": next_agent,
            }

        decision = analyze_user_feedback(state)

        next_action = decision.get("next_action", "")
        user_intent = decision.get("user_intent", "")
        feedback_summary = decision.get("feedback_summary", "")
        focus_areas = decision.get("focus_areas", "")

        # Planning Agent detected user wants auto mode
        if user_intent == "auto":
            cli.note("Planning Agent detected auto-mode intent — subsequent flow will proceed automatically")
            next_agent = get_next_agent_for_phase(completed_phase) if completed_phase else "Survey"
            return {
                "auto_mode": True,
                "human_feedback": None,
                "plan_confirmed": True,
                "next_agent": next_agent,
            }

        if user_intent == "continue":
            valid_agents = {"Survey", "Idea", "Assessment", "Report"}
            default_next = get_next_agent_for_phase(completed_phase or "")
            if next_action in valid_agents:
                # Prioritize Planning Agent's decision
                next_agent = next_action
            else:
                # Fallback to default flow when next_action is invalid
                next_agent = default_next
            cli.info(f"Next step: {next_agent}")
            # If Planning Agent's decision differs from default flow (e.g., rollback to Survey), preserve original feedback
            return {
                "human_feedback": human_feedback if next_agent != default_next else None,
                "focus_areas": focus_areas or None,
                "next_agent": next_agent,
            }

        if user_intent == "guidance":
            # Route guidance to the appropriate agent based on Planning Agent's decision or completed phase
            valid_agents = {"Survey", "Idea", "Assessment", "Report"}
            guidance_target = next_action if next_action in valid_agents else None
            if not guidance_target:
                # Default: guide the current or next phase agent
                guidance_target = get_next_agent_for_phase(completed_phase or "") if completed_phase else "Survey"
            search_directions = state.get("search_directions", []) or []
            if human_feedback:
                search_directions.append(human_feedback)
            # Pass the user's original feedback (not the Planning Agent's summary) to preserve details
            return {
                "search_directions": search_directions,
                "human_feedback": human_feedback,
                "next_agent": guidance_target,
            }

        if next_action == "Upload":
            uploaded_files = handle_file_upload()
            existing_files = state.get("uploaded_files", []) or []
            return {
                "uploaded_files": existing_files + uploaded_files,
                "human_feedback": None,
                "pending_human_action": (completed_phase or "survey") + "_review",
                "next_agent": "HumanReview",
            }

        if next_action == "Finish" or user_intent == "finish":
            if completed_phase != "report":
                return {
                    "next_agent": "Report",
                    "termination_reason": "user_terminated",
                }
            return {
                "next_agent": "FINISH",
                "termination_reason": "user_terminated",
            }

        if user_intent == "iterate":
            valid_iterate_targets = {"Survey", "Idea", "Assessment"}
            iterate_target = next_action if next_action in valid_iterate_targets else None
            if not iterate_target:
                # Fallback: determine iterate target based on completed phase
                phase_iterate_map = {
                    "idea": "Idea",
                    "assessment": "Idea",
                    "survey": "Survey",
                    "report": "Report",
                }
                iterate_target = phase_iterate_map.get(str(completed_phase), "Survey") if completed_phase else "Survey"
            # Pass the user's original feedback (not the Planning Agent's summary) to preserve details
            return {
                "next_agent": iterate_target,
                "human_feedback": human_feedback,
                "focus_areas": focus_areas,
                "planning_decision": decision,
            }

        action_mapping = {
            "Survey": "Survey",
            "Idea": "Idea",
            "Assessment": "Assessment",
            "Report": "Report",
        }

        next_agent = action_mapping.get(next_action)

        if next_agent:
            # Pass the user's original feedback (not the Planning Agent's summary) to preserve details
            return {
                "next_agent": next_agent,
                "human_feedback": human_feedback,
                "focus_areas": focus_areas,
                "planning_decision": decision,
            }
    # Idea↔Assessment closed-loop iteration control (verdict-driven, configurable cap)
    idea_assessment_iters = state.get("idea_assessment_iterations", 0)
    max_idea_assessment_iters = MAX_IDEA_ASSESSMENT_ITERS

    # Flow control in auto mode
    if auto_mode:
        if completed_phase == "assessment":
            should_loop, reason = _should_continue_idea_loop(idea_assessment_iters, max_idea_assessment_iters)
            if should_loop:
                cli.note(f"Auto mode: {reason} — entering feedback-driven iteration (Idea↔Assessment round {idea_assessment_iters + 1}/{max_idea_assessment_iters})")
                return {
                    "next_agent": "Idea",
                    "idea_assessment_iterations": idea_assessment_iters + 1,
                }
            cli.note(f"Auto mode: assessment complete ({reason}) — proceeding to Report")
            return {
                "next_agent": "Report",
                "idea_assessment_iterations": 0,
            }
        else:
            next_agent = get_next_agent_for_phase(completed_phase)
            cli.note(f"Auto mode: {completed_phase} complete — proceeding to {next_agent}")
            return {"next_agent": next_agent}

    # Non-auto mode: verdict-driven auto-iteration after Assessment, hand to user review when converged or capped
    if completed_phase == "assessment":
        should_loop, reason = _should_continue_idea_loop(idea_assessment_iters, max_idea_assessment_iters)
        if should_loop:
            cli.note(f"{reason} — entering feedback-driven iteration (Idea↔Assessment round {idea_assessment_iters + 1}/{max_idea_assessment_iters})")
            return {
                "next_agent": "Idea",
                "idea_assessment_iterations": idea_assessment_iters + 1,
            }
        cli.ok(f"Assessment complete ({reason}) — waiting for user confirmation")
        return {
            "next_agent": "HumanReview",
            "pending_human_action": "assessment_review",
            "idea_assessment_iterations": 0,
        }

    if completed_phase == "survey":
        cli.ok("Survey complete — waiting for user confirmation")
        return {
            "next_agent": "HumanReview",
            "pending_human_action": "survey_review",
        }

    if completed_phase == "idea":
        cli.ok("Ideation complete — waiting for user confirmation")
        return {
            "next_agent": "HumanReview",
            "pending_human_action": "idea_review",
        }

    if completed_phase == "report":
        cli.ok("Report complete — waiting for user review")
        return {
            "next_agent": "HumanReview",
            "pending_human_action": "report_review",
        }

    return {"next_agent": "Survey"}

def get_next_agent_for_phase(completed_phase: Optional[str]) -> str:
    """Get the next Agent based on current phase (default flow in auto mode)"""
    phase_flow = {
        "initialization": "Survey",
        "survey": "Idea",
        "idea": "Assessment",
        "assessment": "Report",
        "report": "FINISH",
    }
    return phase_flow.get(completed_phase, "Survey")


