"""
CLI 输出统一模块 —— 分级、对齐、带时间戳的终端格式化。

输出规范：
    [HH:MM:SS] INFO   一般信息
    [HH:MM:SS] OK     成功/完成
    [HH:MM:SS] WARN   警告（可恢复的异常）
    [HH:MM:SS] ERROR  错误
    [HH:MM:SS] TOOL   工具调用（文件读写、搜索、抓取等）
    [HH:MM:SS] NOTE   流程关键节点提示
    [HH:MM:SS] DEBUG  底层细节（默认隐藏，OMNIIDEA_VERBOSE=1 开启）

阶段大标题用 stage()，小节用 section()，键值对用 kv()。
颜色在管道输出或 NO_COLOR 环境变量存在时自动禁用。
"""

import os
import sys
from datetime import datetime

_LINE_WIDTH = 64

_VERBOSE = os.environ.get("OMNIIDEA_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


_COLOR = _supports_color()

_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}

_LEVEL_COLOR = {
    "INFO": "blue",
    "OK": "green",
    "WARN": "yellow",
    "ERROR": "red",
    "TOOL": "cyan",
    "NOTE": "magenta",
    "DEBUG": "dim",
}


def _c(color: str, text: str) -> str:
    if not _COLOR:
        return text
    return f"{_CODES[color]}{text}{_CODES['reset']}"


def _emit(level: str, msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    if _COLOR:
        ts_s = _c("dim", ts)
        tag = _c(_LEVEL_COLOR.get(level, "reset"), f"{level:<5}")
    else:
        ts_s, tag = ts, f"{level:<5}"
    print(f"[{ts_s}] {tag} {msg}", flush=True)


# ---------------------------------------------
# 分级输出
# ---------------------------------------------

def info(msg: str) -> None:
    _emit("INFO", msg)


def ok(msg: str) -> None:
    _emit("OK", msg)


def warn(msg: str) -> None:
    _emit("WARN", msg)


def error(msg: str) -> None:
    _emit("ERROR", msg)


def note(msg: str) -> None:
    _emit("NOTE", msg)


def debug(msg: str) -> None:
    """底层细节，仅 OMNIIDEA_VERBOSE=1 时显示。"""
    if _VERBOSE:
        _emit("DEBUG", msg)


def tool(action: str, detail: str = "") -> None:
    """工具调用输出：TOOL  read_file  Survey/report.md (61.6 KB)"""
    msg = action if not detail else f"{action}  {detail}"
    _emit("TOOL", msg)


# ---------------------------------------------
# 结构化输出
# ---------------------------------------------

def rule(char: str = "-") -> None:
    print(char * _LINE_WIDTH, flush=True)


def stage(title: str, subtitle: str = "") -> None:
    """阶段大标题，前后各一条分隔线。"""
    head = f"  {title}"
    if subtitle:
        head += f"  ·  {subtitle}"
    print(flush=True)
    if _COLOR:
        print(_c("bold", "=" * _LINE_WIDTH), flush=True)
        print(_c("bold", head), flush=True)
        print(_c("bold", "=" * _LINE_WIDTH), flush=True)
    else:
        print("=" * _LINE_WIDTH, flush=True)
        print(head, flush=True)
        print("=" * _LINE_WIDTH, flush=True)


def section(title: str) -> None:
    """小节标题。"""
    print(flush=True)
    if _COLOR:
        print(_c("bold", f"-- {title} " + "-" * max(2, _LINE_WIDTH - len(title) - 4)), flush=True)
    else:
        print(f"-- {title} " + "-" * max(2, _LINE_WIDTH - len(title) - 4), flush=True)


def kv(key: str, value: str, indent: int = 2) -> None:
    """对齐键值对：  key         : value"""
    pad = " " * indent
    if _COLOR:
        print(f"{pad}{_c('dim', f'{key:<14}')}: {value}", flush=True)
    else:
        print(f"{pad}{key:<14}: {value}", flush=True)


def banner(title: str, rows: list) -> None:
    """启动横幅：标题 + 键值行，框在分隔线内。"""
    print(flush=True)
    line = "=" * _LINE_WIDTH
    if _COLOR:
        print(_c("bold", line), flush=True)
        print(_c("bold", f"  {title}"), flush=True)
        print(_c("bold", line), flush=True)
    else:
        print(line, flush=True)
        print(f"  {title}", flush=True)
        print(line, flush=True)
    for key, value in rows:
        kv(key, value)
    rule("=")
