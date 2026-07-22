import os
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tavily import TavilyClient
from typing import Annotated, List, Literal, Optional
from langchain_core.tools import InjectedToolArg, tool

import cli

_tavily_api_key = os.getenv("TAVILY_API_KEY")
if not _tavily_api_key:
    cli.warn("TAVILY_API_KEY not set — Tavily search disabled (set it in .env to enable)")
    tavily_client = None
else:
    tavily_client = TavilyClient(api_key=_tavily_api_key)

# ==========================================
# 网络搜索（只返回摘要，不抓取完整内容）
# ==========================================

def _execute_tavily_query(query: str, max_results: int = 25, topic: str = "general") -> str:
    """执行单次 Tavily 搜索并格式化结果（单发/批量工具共用的私有实现）。"""
    search_results = tavily_client.search(
        query,
        max_results=max_results,
        topic=topic,
        search_depth='advanced'
    )

    result_texts = []
    for i, result in enumerate(search_results.get("results", []), 1):
        url = result.get("url", "")
        title = result.get("title", "Untitled")
        snippet = result.get("content", "")[:500]  # 只取摘要前500字符

        result_text = f"""### [{i}] {title}
**URL:** {url}
**Snippet:** {snippet}
"""
        result_texts.append(result_text)

    return f"""🔍 Found {len(result_texts)} result(s) for '{query}':

{chr(10).join(result_texts)}"""


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 25,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """Search the web for information on a given query.

    Returns search results with titles, URLs, and snippets. Use web_fetch to get full content of specific URLs.
    When you have multiple INDEPENDENT queries, prefer batch_web_search (concurrent, faster).

    Args:
        query: Search query to execute
        max_results: Maximum number of results to return (default: 10)
        topic: Topic filter - 'general', 'news', or 'finance' (default: 'general')

    Returns:
        Formatted search results with titles, URLs, and snippets (not full content)
    """
    response = _execute_tavily_query(query, max_results=max_results, topic=topic)
    return response + "\n\n💡 **Tip:** Use `web_fetch(url)` to get full content of specific URLs that look most relevant."


_BATCH_MAX_QUERIES = 6


@tool(parse_docstring=True)
def batch_web_search(
    queries: List[str],
    max_results_per_query: Annotated[int, InjectedToolArg] = 8,
) -> str:
    """Run multiple INDEPENDENT web searches concurrently in a single call.

    Prefer this over repeated `tavily_search` calls whenever you have several queries
    that do NOT depend on each other's results (e.g., different angles of the same
    research round): one call fires all queries in parallel and returns every result
    set together, saving substantial wall-clock time. Use single `tavily_search` only
    when the next query depends on a previous result.

    Args:
        queries: List of search queries to run concurrently (max 6 per call; extra queries are dropped)
        max_results_per_query: Maximum number of results per query (default: 8)

    Returns:
        Search result sections grouped per query (titles, URLs, snippets, not full content)
    """
    if tavily_client is None:
        return "Error: Tavily search is disabled (TAVILY_API_KEY not set)."

    # 去重保序 + 防风暴截断
    unique_queries = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))
    if not unique_queries:
        return "Error: no valid queries provided."
    dropped = max(0, len(unique_queries) - _BATCH_MAX_QUERIES)
    run_queries = unique_queries[:_BATCH_MAX_QUERIES]

    def _safe_run(q: str):
        try:
            return q, _execute_tavily_query(q, max_results=max_results_per_query)
        except Exception as exc:  # 错误隔离：单查询失败不影响整批
            return q, f"⚠️ search failed: {exc}"

    sections = {}
    with ThreadPoolExecutor(max_workers=min(len(run_queries), _BATCH_MAX_QUERIES)) as pool:
        futures = [pool.submit(_safe_run, q) for q in run_queries]
        for fut in as_completed(futures):
            q, text = fut.result()
            sections[q] = text

    header = f"🔍 Batch search: {len(run_queries)} queries executed concurrently"
    if dropped:
        header += f" (dropped {dropped} extra quer{'y' if dropped == 1 else 'ies'} — max {_BATCH_MAX_QUERIES} per call)"
    parts = [f'## Query: "{q}"\n\n{sections[q]}' for q in run_queries]  # 按输入顺序输出
    return (
        header + "\n\n" + "\n\n".join(parts)
        + "\n\n💡 **Tip:** Use `web_fetch(url)` to get full content of specific URLs that look most relevant."
    )


# pdf_extract_text 的返回值会作为 ToolMessage 进入对话；与 web_fetch 类似需限制长度，避免大 PDF 撑爆上下文。
# 完整抽取结果仍会写入 workspace/Inputs/raw/pdf_cache/<basename>.txt。
_MAX_PDF_TOOL_RETURN_CHARS = 10000


@tool(parse_docstring=True)
def pdf_extract_text(
    file_path: str,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    extract_tables: bool = False,
) -> str:
    """Extract text content from a PDF file.

    Reads a PDF file and extracts text from all or specified pages.
    Supports both regular text extraction and table extraction.
    The file_path can be an absolute path or a path relative to the workspace
    (e.g., 'Inputs/uploads/paper.pdf').

    Args:
        file_path: Path to the PDF file. Can be absolute or relative to workspace.
        start_page: First page to extract (1-indexed, inclusive). Defaults to first page.
        end_page: Last page to extract (1-indexed, inclusive). Defaults to last page.
        extract_tables: Whether to also extract tables as structured text.

    Returns:
        Extracted text content with page markers, and optionally tables.
    """
    cli.tool("pdf_extract", file_path)

    if not os.path.isabs(file_path):
        workspace_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
        file_path = os.path.join(workspace_dir, file_path)

    if not os.path.isfile(file_path):
        return f"❌ File not found: {file_path}"

    if not file_path.lower().endswith(".pdf"):
        return f"❌ Not a PDF file: {file_path}"

    extracted_text = _extract_with_pdfplumber(file_path, start_page, end_page, extract_tables)
    if extracted_text is None:
        extracted_text = _extract_with_pypdf(file_path, start_page, end_page)

    if extracted_text is None:
        return f"❌ Failed to extract text from '{file_path}'. Please install pypdf or pdfplumber: pip install pypdf pdfplumber"

    _cache_extracted_text(file_path, extracted_text)

    # 工具返回值会进入对话历史；不截断时整本 PDF 易导致上下文窗口爆炸（read_file 同理）。
    # 完整文本已写入 Inputs/raw/pdf_cache/<name>.txt，可按页分段再调 pdf_extract_text 或 read_file。
    total_len = len(extracted_text)
    if total_len > _MAX_PDF_TOOL_RETURN_CHARS:
        cache_rel = f"Inputs/raw/pdf_cache/{os.path.splitext(os.path.basename(file_path))[0]}.txt"
        head = extracted_text[:_MAX_PDF_TOOL_RETURN_CHARS]
        return (
            head
            + f"\n\n... [Truncated for LLM context: showing first {_MAX_PDF_TOOL_RETURN_CHARS:,} of {total_len:,} characters. "
            f"Full text is saved at `{cache_rel}`. Use `pdf_extract_text(..., start_page=, end_page=)` for other page ranges, "
            f"or `read_file(\"{cache_rel}\")` only for small files — large files may still overflow context.]"
        )
    return extracted_text

def _cache_extracted_text(source_pdf_path: str, extracted_text: str) -> None:
    """Save extracted text to pdf_cache directory for reuse by other agents."""
    try:
        workspace_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
        cache_dir = os.path.join(workspace_dir, "Inputs", "raw", "pdf_cache")
        os.makedirs(cache_dir, exist_ok=True)

        pdf_basename = os.path.splitext(os.path.basename(source_pdf_path))[0]
        cache_file_path = os.path.join(cache_dir, f"{pdf_basename}.txt")

        with open(cache_file_path, "w", encoding="utf-8") as cache_file:
            cache_file.write(extracted_text)

        cli.debug(f"PDF cache saved: Inputs/raw/pdf_cache/{pdf_basename}.txt")
    except Exception as error:
        cli.warn(f"Cache write failed: {error}")

def _extract_with_pdfplumber(
    file_path: str,
    start_page: Optional[int],
    end_page: Optional[int],
    extract_tables: bool,
) -> Optional[str]:
    """Extract text using pdfplumber (preferred, better layout preservation)."""
    try:
        import pdfplumber
    except ImportError:
        return None

    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            first_page = max(1, start_page or 1)
            last_page = min(total_pages, end_page or total_pages)

            result_parts = [
                f"📄 **{os.path.basename(file_path)}** — {total_pages} pages total, "
                f"extracting pages {first_page}-{last_page}\n"
            ]

            for page_num in range(first_page - 1, last_page):
                page = pdf.pages[page_num]
                result_parts.append(f"\n--- Page {page_num + 1} ---\n")

                page_text = page.extract_text()
                if page_text:
                    result_parts.append(page_text)
                else:
                    result_parts.append("(No text content on this page)")

                if extract_tables:
                    tables = page.extract_tables()
                    for table_index, table in enumerate(tables):
                        if not table:
                            continue
                        result_parts.append(f"\n📊 Table {table_index + 1} on Page {page_num + 1}:\n")
                        for row in table:
                            cleaned_row = [cell if cell else "" for cell in row]
                            result_parts.append(" | ".join(cleaned_row))

            return "\n".join(result_parts)

    except Exception as error:
        return f"❌ pdfplumber extraction error: {error}"

def _extract_with_pypdf(
    file_path: str,
    start_page: Optional[int],
    end_page: Optional[int],
) -> Optional[str]:
    """Extract text using pypdf (fallback)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None

    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        first_page = max(1, start_page or 1)
        last_page = min(total_pages, end_page or total_pages)

        result_parts = [
            f"📄 **{os.path.basename(file_path)}** — {total_pages} pages total, "
            f"extracting pages {first_page}-{last_page}\n"
        ]

        for page_num in range(first_page - 1, last_page):
            page = reader.pages[page_num]
            result_parts.append(f"\n--- Page {page_num + 1} ---\n")

            page_text = page.extract_text()
            if page_text:
                result_parts.append(page_text)
            else:
                result_parts.append("(No text content on this page)")

        metadata = reader.metadata
        if metadata:
            meta_parts = []
            if metadata.title:
                meta_parts.append(f"**Title:** {metadata.title}")
            if metadata.author:
                meta_parts.append(f"**Author:** {metadata.author}")
            if metadata.subject:
                meta_parts.append(f"**Subject:** {metadata.subject}")
            if meta_parts:
                result_parts.insert(1, "\n".join(meta_parts) + "\n")

        return "\n".join(result_parts)

    except Exception as error:
        return f"❌ pypdf extraction error: {error}"

# ==========================================
# Workspace 文件操作工具
# ==========================================# workspace 根目录（与 storage.py 中的 create_workspace_store 保持一致）
_WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
# 共享的 WorkspaceStore 引用，由 agents.py 初始化后通过 set_shared_store() 注入。
# 这样 write_file tool 写文件时可以同步更新 shared_store 的内存状态，
# 避免 nodes.py 通过 shared_store.read_file() 读不到 Agent 通过 tool 写入的内容。
_shared_store = None

def set_shared_store(store) -> None:
    """注入共享的 WorkspaceStore 实例，供 workspace 工具同步内存状态。

    必须在 agents.py 创建 shared_store 之后、构建 Agent 之前调用。
    """
    global _shared_store
    _shared_store = store

def _resolve_workspace_path(relative_path: str) -> str:
    """将相对路径解析为 workspace 内的绝对路径，并做安全校验。"""
    clean_path = relative_path.lstrip("/").lstrip("\\")
    if clean_path.startswith("workspace/") or clean_path.startswith("workspace\\"):
        clean_path = clean_path[len("workspace/"):]
    absolute_path = os.path.normpath(os.path.join(_WORKSPACE_DIR, clean_path))
    if not absolute_path.startswith(os.path.normpath(_WORKSPACE_DIR)):
        raise ValueError(f"Path traversal detected: '{relative_path}' resolves outside workspace")
    return absolute_path


@tool(parse_docstring=True)
def read_file(file_path: str) -> str:
    """Read the content of a file from the workspace.

    Reads a text file from the workspace directory and returns its content.
    Use this to access research plans, survey results, ideas, assessments,
    and any other files produced by other agents.

    Common paths:
    - Planning/search_plan.md — search plan
    - Planning/global_context.json — global context
    - Survey/report.md — survey summary report
    - Innovation/ideas.md — generated ideas
    - Assessment/assessment_feedback.md — assessment feedback
    - Assessment/rubric.md — evaluation rubric
    - Assessment/iteration_history.md — iteration history
    - Outputs/Final_Report.md — final report

    Args:
        file_path: Relative path within the workspace (e.g., 'Planning/search_plan.md')

    Returns:
        The file content as a string, or an error message if the file does not exist.
    """
    # 清理路径：去除前导斜杠和 workspace/ 前缀，确保路径一致性
    clean_file_path = file_path.lstrip("/").lstrip("\\")
    if clean_file_path.startswith("workspace/") or clean_file_path.startswith("workspace\\"):
        clean_file_path = clean_file_path[len("workspace/"):]
    try:
        # 优先从文件系统读取（确保拿到最新内容，无论是 tool 还是 shared_store 写入的）
        absolute_path = _resolve_workspace_path(clean_file_path)
        if not os.path.isfile(absolute_path):
            # 文件系统没有，再尝试从 shared_store 内存读取
            if _shared_store is not None:
                memory_content = _shared_store.read_file(clean_file_path)
                if memory_content is not None:
                    size_kb = len(memory_content.encode("utf-8")) / 1024
                    cli.tool("read_file", f"{clean_file_path} ({size_kb:.1f} KB, cached)")
                    return memory_content

            available = _list_directory_contents(os.path.dirname(clean_file_path))
            hint = f"\n📂 Available files in '{os.path.dirname(clean_file_path) or 'root'}':\n{available}" if available else ""
            return f"❌ File not found: {clean_file_path}{hint}"

        with open(absolute_path, "r", encoding="utf-8") as file_handle:
            content = file_handle.read()

        size_kb = len(content.encode("utf-8")) / 1024
        cli.tool("read_file", f"{clean_file_path} ({size_kb:.1f} KB)")
        return content

    except ValueError as error:
        return f"❌ Security error: {error}"
    except Exception as error:
        return f"❌ Failed to read '{file_path}': {error}"


@tool(parse_docstring=True)
def write_file(file_path: str, content: str) -> str:
    """Write content to a file in the workspace.

    Creates or overwrites a file in the workspace directory.
    Parent directories are created automatically if they don't exist.
    Use this to save research results, ideas, assessments, reports, etc.

    Common output paths:
    - Planning/search_plan.md — search plan
    - Survey/report.md — survey summary report
    - Innovation/ideas.md — generated ideas
    - Assessment/assessment_feedback.md — assessment feedback
    - Assessment/rubric.md — evaluation rubric
    - Assessment/rubric_dimensions.json — rubric source data
    - Outputs/Final_Report.md — final report

    Args:
        file_path: Relative path within the workspace (e.g., 'Survey/report.md')
        content: The text content to write to the file

    Returns:
        A success message with the file path and size, or an error message.
    """
    # 清理路径：去除前导斜杠和 workspace/ 前缀，确保路径一致性
    clean_file_path = file_path.lstrip("/").lstrip("\\")
    if clean_file_path.startswith("workspace/") or clean_file_path.startswith("workspace\\"):
        clean_file_path = clean_file_path[len("workspace/"):]
    try:
        # 优先通过 shared_store 写入（同时更新内存 + 文件系统）
        if _shared_store is not None:
            _shared_store.write_file(clean_file_path, content)
        else:
            # 降级：直接写文件系统（shared_store 未注入时）
            absolute_path = _resolve_workspace_path(clean_file_path)
            os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
            with open(absolute_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(content)

        size_kb = len(content.encode("utf-8")) / 1024
        cli.tool("write_file", f"{clean_file_path} ({size_kb:.1f} KB)")
        return f"✅ Successfully wrote {size_kb:.1f} KB to '{clean_file_path}'"

    except ValueError as error:
        return f"❌ Security error: {error}"
    except Exception as error:
        return f"❌ Failed to write '{clean_file_path}': {error}"


@tool(parse_docstring=True)
def append_file(file_path: str, content: str) -> str:
    """Append content to an existing file in the workspace, or create it if it doesn't exist.

    Unlike write_file which overwrites, this tool appends content to the end of the file.
    Useful for incrementally building up research notes, survey summaries, etc.

    Primary use case:
    - Survey/report.md — append supplementary findings during iterative research

    Args:
        file_path: Relative path within the workspace (e.g., 'Survey/report.md')
        content: The text content to append to the file

    Returns:
        A success message with the file path and appended size, or an error message.
    """
    # 清理路径
    clean_file_path = file_path.lstrip("/").lstrip("\\")
    if clean_file_path.startswith("workspace/") or clean_file_path.startswith("workspace\\"):
        clean_file_path = clean_file_path[len("workspace/"):]
    try:
        absolute_path = _resolve_workspace_path(clean_file_path)
        os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

        # 读取已有内容
        existing_content = ""
        if os.path.isfile(absolute_path):
            with open(absolute_path, "r", encoding="utf-8") as file_handle:
                existing_content = file_handle.read()

        # 追加新内容（确保有换行分隔）
        separator = "\n\n" if existing_content and not existing_content.endswith("\n\n") else ""
        new_content = existing_content + separator + content

        # 写入文件系统
        with open(absolute_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(new_content)

        # 同步到 shared_store 内存
        if _shared_store is not None:
            _shared_store.write_file(clean_file_path, new_content)

        appended_kb = len(content.encode("utf-8")) / 1024
        total_kb = len(new_content.encode("utf-8")) / 1024
        cli.tool("append_file", f"{clean_file_path} (+{appended_kb:.1f} KB, total {total_kb:.1f} KB)")
        return f"✅ Successfully appended {appended_kb:.1f} KB to '{clean_file_path}' (total: {total_kb:.1f} KB)"

    except ValueError as error:
        return f"❌ Security error: {error}"
    except Exception as error:
        return f"❌ Failed to append to '{clean_file_path}': {error}"


def _list_directory_contents(directory: str) -> str:
    """列出 workspace 指定目录下的文件，用于提示 Agent。"""
    target_dir = os.path.join(_WORKSPACE_DIR, directory) if directory else _WORKSPACE_DIR
    if not os.path.exists(target_dir):
        return ""
    entries = []
    for root, dirs, files in os.walk(target_dir):
        for filename in files:
            if filename.endswith(".meta.json"):
                continue
            file_path = os.path.join(root, filename)
            relative = os.path.relpath(file_path, _WORKSPACE_DIR)
            entries.append(f"  - {relative}")
    return "\n".join(entries[:30])  # 最多显示 30 个


@tool(parse_docstring=True)
def list_files(directory: str = "") -> str:
    """List files in a workspace directory.

    Lists all files under the specified directory in the workspace.
    Use this to discover what files are available, check if other agents
    have produced their outputs, or find uploaded files.

    Top-level directories:
    - Inputs/uploads/ — user-uploaded files
    - Inputs/raw/ — collected literature cache
    - Planning/ — research plans and context
    - Survey/ — survey results
    - Innovation/ — generated ideas
    - Assessment/ — assessment results
    - Outputs/ — final reports
    - ../Skills/ — agent skill definitions (outside workspace)

    Args:
        directory: Relative directory path within workspace (e.g., 'Inputs/uploads').
                   Use empty string '' to list all files in the workspace root.

    Returns:
        A formatted list of files found, or a message if the directory is empty.
    """
    cli.tool("list_files", directory or "(root)")
    try:
        target_dir = os.path.join(_WORKSPACE_DIR, directory) if directory else _WORKSPACE_DIR
        if not os.path.exists(target_dir):
            return f"📂 Directory '{directory}' does not exist in workspace."

        file_entries = []
        for root, _dirs, files in os.walk(target_dir):
            for filename in sorted(files):
                if filename.endswith(".meta.json"):
                    continue
                file_path = os.path.join(root, filename)
                relative = os.path.relpath(file_path, _WORKSPACE_DIR)
                file_size = os.path.getsize(file_path)
                if file_size >= 1024 * 1024:
                    size_str = f"{file_size / (1024 * 1024):.1f} MB"
                elif file_size >= 1024:
                    size_str = f"{file_size / 1024:.1f} KB"
                else:
                    size_str = f"{file_size} B"
                file_entries.append(f"  - {relative} ({size_str})")

        if not file_entries:
            return f"📂 Directory '{directory or 'workspace'}' is empty."

        header = f"📂 Files in '{directory or 'workspace'}' ({len(file_entries)} files):\n"
        return header + "\n".join(file_entries)

    except Exception as error:
        return f"❌ Failed to list '{directory}': {error}"


# ==========================================
# 网页全文抓取工具
# ==========================================

_MAX_CONTENT_LENGTH = 10000  # 最大抓取内容长度（字符）
_JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
_JINA_READER_BASE_URL = "https://r.jina.ai/"

@tool(parse_docstring=True)
def web_fetch(url: str) -> str:
    """Fetch and extract the main text content from a webpage.

    Uses Jina Reader API to intelligently extract and convert webpage content
    into clean, LLM-friendly Markdown text. Handles JavaScript-rendered pages,
    automatically filters ads/navigation noise, and preserves document structure.

    Supports standard HTTP/HTTPS URLs. For PDF files, use pdf_extract_text instead.

    Args:
        url: The full URL of the webpage to fetch (e.g., 'https://example.com/article')

    Returns:
        The extracted main text content of the webpage in Markdown format, or an error message.
    """
    cli.tool("web_fetch", url)

    if not url.startswith(("http://", "https://")):
        return f"❌ Invalid URL: '{url}'. Must start with http:// or https://"

    # 常见 PDF 直链（含 ? 后缀的也排除）
    _url_lower = url.split("?", 1)[0].lower()
    if _url_lower.endswith(".pdf") or "/article_pdf/" in url.lower():
        return f"❌ This URL points to a PDF. Download to `Inputs/uploads/` and use `pdf_extract_text`, or use an HTML landing page URL."

    jina_url = f"{_JINA_READER_BASE_URL}{url}"

    try:
        headers = {
            "Authorization": f"Bearer {_JINA_API_KEY}",
            "X-Retain-Images": "none",
            "X-Retain-Links": "gpt-oss",
        }

        # (connect_timeout, read_timeout)：避免长时间卡在建立连接阶段
        t0 = time.perf_counter()
        response = requests.get(jina_url, headers=headers, timeout=(15, 60))
        response.raise_for_status()

        extracted_text = response.text.strip()
        elapsed = time.perf_counter() - t0

        if not extracted_text or len(extracted_text) < 50:
            cli.warn(f"Jina returned little content ({elapsed:.1f}s): {url}")
            return f"⚠️ Could not extract meaningful content from: {url}"

        # 截断过长内容
        if len(extracted_text) > _MAX_CONTENT_LENGTH:
            extracted_text = extracted_text[:_MAX_CONTENT_LENGTH] + \
                f"\n\n... [Content truncated at {_MAX_CONTENT_LENGTH} characters]"

        cli.ok(f"Fetched {len(extracted_text)} chars in {elapsed:.1f}s")
        return f"🌐 **Content from:** {url}\n\n{extracted_text}"

    except requests.exceptions.Timeout:
        return f"❌ Timeout fetching '{url}' (60s limit exceeded)"
    except requests.exceptions.ConnectionError:
        return f"❌ Connection error fetching '{url}'"
    except requests.exceptions.HTTPError as error:
        return f"❌ HTTP error fetching '{url}': {error.response.status_code}"
    except Exception as error:
        return f"❌ Failed to fetch '{url}': {error}"

# ==========================================
# 工具导出列表
# ==========================================

# Workspace 文件操作工具（所有 Agent 共用）
WORKSPACE_TOOLS = [
    read_file,
    write_file,
    list_files,
]

# PDF 处理工具
PDF_TOOLS = [
    pdf_extract_text,
]

# Survey Agent 使用的工具（文件操作 + 追加写入 + 搜索 + 网页抓取 + PDF 提取）
SURVEY_TOOLS = [
    *WORKSPACE_TOOLS,
    append_file,      # 追加写入（用于逐条缓存调研摘要）
    tavily_search,    # 网络搜索（单发，用于有依赖的追问）
    batch_web_search, # 批量并发搜索（多个独立查询一次调用，省时）
    web_fetch,
    pdf_extract_text,
]

# 非搜索类 Agent 使用的工具（文件操作）
AGENT_BASE_TOOLS = [
    *WORKSPACE_TOOLS,
]

# 所有可用工具
ALL_TOOLS = [
    *WORKSPACE_TOOLS,
    tavily_search,
    web_fetch,
    pdf_extract_text,
]
