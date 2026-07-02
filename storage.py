"""
存储模块 - 支持 workspace 分层目录结构的存储实现

Workspace 目录结构：
├── Inputs/
│   ├── uploads/              # 用户上传的原始文件
│   └── raw/                  # 收集到的文献/内容缓存
├── Planning/
│   ├── task_breakdown.json
│   ├── global_context.json
│   └── human_feedback_plan.md
├── Survey/
│   ├── surveyed.md
│   ├── report.md
│   └── human_feedback_survey.md
├── Innovation/
│   ├── ideas.md
│   ├── consolidated.md
│   └── human_feedback_idea.md
├── Assessment/
│   ├── assessment_results.json
│   └── human_feedback_eval.md
└── Outputs/
    ├── draft.md
    └── Final_Report.md

Skills 目录（位于 workspace 外部，与 workspace 同级）：
Skills/
└── <skill-name>/
    └── SKILL.md           # 标准 Agent Skills 规范文件

Skills 机制遵循 LangChain Deep Agents 的 Agent Skills 规范：
- 每个 skill 是 Skills/ 下的一个子目录，包含 SKILL.md 和可选的附属文件
- SKILL.md 使用 YAML frontmatter 定义 name 和 description
- 采用 progressive disclosure：Agent 先读 description 判断相关性，匹配后才读完整内容
"""

import os
import re
import json
from typing import Optional, Any, Dict, List
from langgraph.store.memory import InMemoryStore


# ==========================================
# Workspace 路径常量
# ==========================================

WORKSPACE_DIRS = [
    "Inputs/uploads",
    "Inputs/raw",
    "Planning",
    "Survey",
    "Innovation",
    "Assessment",
    "Outputs",
]

# 标准文件路径常量（供各模块引用）
PLANNING_GLOBAL_CONTEXT = "Planning/global_context.json"
PLANNING_SEARCH_PLAN = "Planning/search_plan.md"
PLANNING_HUMAN_FEEDBACK = "Planning/human_feedback_plan.md"

INPUTS_UPLOADS_DIR = "Inputs/uploads"

SURVEY_REPORT = "Survey/report.md"
SURVEY_HUMAN_FEEDBACK = "Survey/human_feedback_survey.md"

INNOVATION_IDEAS = "Innovation/ideas.md"
INNOVATION_HUMAN_FEEDBACK = "Innovation/human_feedback_idea.md"

ASSESSMENT_RUBRIC = "Assessment/rubric.md"
ASSESSMENT_RESULTS = "Assessment/assessment_results.json"
ASSESSMENT_FEEDBACK = "Assessment/assessment_feedback.md"
ASSESSMENT_HUMAN_FEEDBACK = "Assessment/human_feedback_eval.md"

OUTPUTS_FINAL_REPORT = "Outputs/Final_Report.md"


class WorkspaceStore(InMemoryStore):
    """
    基于 workspace 分层目录结构的存储类。

    继承自 InMemoryStore，在内存操作的同时同步到本地 workspace 文件系统。
    namespace 直接映射到 workspace 子目录，key 为文件名。
    """

    def __init__(self, workspace_dir: str = "./workspace", skills_dir: str = None):
        super().__init__()
        self.workspace_dir = os.path.abspath(workspace_dir)
        if skills_dir is None:
            # Skills 目录默认在 workspace 的同级目录下
            self.skills_dir = os.path.join(os.path.dirname(self.workspace_dir), "Skills")
        else:
            self.skills_dir = os.path.abspath(skills_dir)
        self._ensure_workspace_structure()
        self._load_from_workspace()

    def _ensure_workspace_structure(self):
        """确保 workspace 目录结构和 Skills 目录存在"""
        for sub_dir in WORKSPACE_DIRS:
            full_path = os.path.join(self.workspace_dir, sub_dir)
            os.makedirs(full_path, exist_ok=True)
        os.makedirs(self.skills_dir, exist_ok=True)
        print(f"   📁 Workspace 目录已就绪: {self.workspace_dir}")
        print(f"   📁 Skills 目录已就绪: {self.skills_dir}")

    def _resolve_path(self, namespace: tuple, key: str) -> str:
        """
        将 namespace + key 解析为 workspace 内的绝对文件路径。

        支持两种调用方式：
        1. namespace=("Planning",), key="task_breakdown.json"
           -> workspace/Planning/task_breakdown.json
        2. namespace=("Inputs", "raw"), key="collected_papers.md"
           -> workspace/Inputs/raw/collected_papers.md
        """
        clean_key = key.lstrip("/")
        if clean_key.startswith("workspace/"):
            clean_key = clean_key[len("workspace/"):]

        namespace_path = os.path.join(self.workspace_dir, *namespace) if namespace else self.workspace_dir

        if "/" in clean_key:
            key_dir = os.path.dirname(clean_key)
            namespace_path = os.path.join(namespace_path, key_dir)
            clean_key = os.path.basename(clean_key)

        os.makedirs(namespace_path, exist_ok=True)
        return os.path.join(namespace_path, clean_key)

    def _load_from_workspace(self):
        """从 workspace 目录加载已有文件到内存"""
        if not os.path.exists(self.workspace_dir):
            return

        loaded_count = 0
        for root, _dirs, files in os.walk(self.workspace_dir):
            for filename in files:
                if filename.endswith(".meta.json"):
                    continue

                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, self.workspace_dir)

                try:
                    with open(file_path, "r", encoding="utf-8") as file_handle:
                        content = file_handle.read()

                    parts = rel_path.split(os.sep)
                    if len(parts) > 1:
                        namespace = tuple(parts[:-1])
                        key = parts[-1]
                    else:
                        namespace = ("Outputs",)
                        key = parts[0]

                    super().put(namespace, key, {"content": content})
                    loaded_count += 1
                except Exception as error:
                    print(f"   ⚠️ 加载文件失败 {file_path}: {error}")

        if loaded_count > 0:
            print(f"   📂 已加载 {loaded_count} 个文件到内存")

    def put(self, namespace: tuple, key: str, value: Any, index: Optional[bool] = None) -> None:
        """存储数据，同时同步到 workspace 文件"""
        super().put(namespace, key, value)
        self._sync_to_workspace(namespace, key, value)

    def _convert_to_string(self, content: Any) -> str:
        """将内容转换为字符串格式"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            if all(isinstance(item, str) for item in content):
                return "\n".join(content)
            return json.dumps(content, ensure_ascii=False, indent=2)
        elif isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False, indent=2)
        return str(content)

    def _sync_to_workspace(self, namespace: tuple, key: str, value: Any):
        """同步数据到 workspace 文件"""
        try:
            local_path = self._resolve_path(namespace, key)

            if isinstance(value, dict):
                content = value.get("content", value)
            else:
                content = value

            content_str = self._convert_to_string(content)

            with open(local_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(content_str)

            print(f"   💾 已写入: {os.path.relpath(local_path, self.workspace_dir)}")
        except Exception as error:
            print(f"   ⚠️ 写入失败: {error}")

    def delete(self, namespace: tuple, key: str) -> None:
        """删除数据，同时删除 workspace 文件"""
        super().delete(namespace, key)
        try:
            local_path = self._resolve_path(namespace, key)
            if os.path.exists(local_path):
                os.remove(local_path)
                print(f"   🗑️ 已删除: {os.path.relpath(local_path, self.workspace_dir)}")
        except Exception as error:
            print(f"   ⚠️ 删除失败: {error}")

    def read_file(self, relative_path: str) -> Optional[str]:
        """
        便捷方法：直接通过相对路径读取 workspace 文件内容。

        Args:
            relative_path: 相对于 workspace 根目录的路径，如 "Planning/task_breakdown.json"

        Returns:
            文件内容字符串，文件不存在则返回 None
        """
        full_path = os.path.join(self.workspace_dir, relative_path)
        if not os.path.exists(full_path):
            return None
        try:
            with open(full_path, "r", encoding="utf-8") as file_handle:
                return file_handle.read()
        except Exception:
            return None

    def write_file(self, relative_path: str, content: str):
        """
        便捷方法：直接通过相对路径写入 workspace 文件。

        同时更新内存 store。

        Args:
            relative_path: 相对于 workspace 根目录的路径
            content: 文件内容
        """
        # 清理路径：去除前导斜杠和 workspace/ 前缀
        clean_path = relative_path.replace("\\", "/").lstrip("/")
        if clean_path.startswith("workspace/"):
            clean_path = clean_path[len("workspace/"):]
        parts = clean_path.split("/")
        if len(parts) > 1:
            namespace = tuple(parts[:-1])
            key = parts[-1]
        else:
            namespace = ("Outputs",)
            key = parts[0]

        self.put(namespace, key, {"content": content})

    def list_files(self, directory: str = "") -> list:
        """列出 workspace 指定目录下的所有文件"""
        target_dir = os.path.join(self.workspace_dir, directory) if directory else self.workspace_dir
        if not os.path.exists(target_dir):
            return []

        result = []
        for root, _dirs, files in os.walk(target_dir):
            for filename in files:
                if not filename.endswith(".meta.json"):
                    file_path = os.path.join(root, filename)
                    result.append(os.path.relpath(file_path, self.workspace_dir))
        return result

    def get_workspace_dir(self) -> str:
        """返回 workspace 根目录的绝对路径"""
        return self.workspace_dir

    # ==========================================
    # History / Memory 版本管理
    # ==========================================

    def get_current_version(self) -> int:
        """
        读取 History 目录下最大版本号。

        通过扫描 History/ 目录中 version_N.json 文件名来确定当前最大版本号。

        Returns:
            当前最大版本号，如果没有版本文件则返回 0
        """
        history_dir = os.path.join(self.workspace_dir, "History")
        if not os.path.exists(history_dir):
            return 0

        max_version = 0
        for filename in os.listdir(history_dir):
            if filename.startswith("version_") and filename.endswith(".json"):
                try:
                    version_num = int(filename[len("version_"):-len(".json")])
                    max_version = max(max_version, version_num)
                except ValueError:
                    continue
        return max_version

    def record_memory_version(self, user_input: str, input_type: str,
                              current_phase: str, iteration_count: int) -> int:
        """
        记录一个新的 memory 版本：保存用户输入和当前 workspace 所有文件快照。

        Args:
            user_input: 用户的原始输入文本
            input_type: 输入类型，如 "topic", "plan_review", "survey_review",
                        "idea_review", "assessment_review", "report_review"
            current_phase: 当前研究阶段
            iteration_count: 当前迭代次数

        Returns:
            新创建的版本号
        """
        from datetime import datetime

        new_version = self.get_current_version() + 1

        # 遍历 workspace 中除 History/ 外的所有文件，生成快照
        workspace_snapshot = {}
        for root, _dirs, files in os.walk(self.workspace_dir):
            # 跳过 History 目录本身
            rel_root = os.path.relpath(root, self.workspace_dir)
            if rel_root == "History" or rel_root.startswith("History" + os.sep):
                continue

            for filename in files:
                if filename.endswith(".meta.json"):
                    continue
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, self.workspace_dir)
                try:
                    with open(file_path, "r", encoding="utf-8") as fh:
                        workspace_snapshot[rel_path] = fh.read()
                except (UnicodeDecodeError, IOError):
                    # 跳过无法读取的二进制文件
                    workspace_snapshot[rel_path] = "<binary file, skipped>"

        version_record = {
            "version": new_version,
            "timestamp": datetime.now().isoformat(),
            "input_type": input_type,
            "user_input": user_input,
            "current_phase": current_phase,
            "iteration_count": iteration_count,
            "workspace_snapshot": workspace_snapshot,
        }

        # 写入版本文件（直接写文件，不经过 put 以避免触发 _load_from_workspace 的内存膨胀）
        history_dir = os.path.join(self.workspace_dir, "History")
        os.makedirs(history_dir, exist_ok=True)
        version_file = os.path.join(history_dir, f"version_{new_version}.json")
        with open(version_file, "w", encoding="utf-8") as fh:
            json.dump(version_record, fh, ensure_ascii=False, indent=2)

        print(f"   📸 Memory version {new_version} recorded (input_type={input_type}, phase={current_phase})")
        return new_version

    def list_versions(self) -> List[Dict[str, Any]]:
        """
        列出所有版本的摘要信息。

        Returns:
            版本摘要列表，每个元素包含 version、timestamp、input_type、user_input_preview
        """
        history_dir = os.path.join(self.workspace_dir, "History")
        if not os.path.exists(history_dir):
            return []

        versions = []
        for filename in sorted(os.listdir(history_dir)):
            if not (filename.startswith("version_") and filename.endswith(".json")):
                continue

            file_path = os.path.join(history_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    record = json.load(fh)
                user_input = record.get("user_input", "")
                preview = user_input[:100] + "..." if len(user_input) > 100 else user_input
                versions.append({
                    "version": record.get("version"),
                    "timestamp": record.get("timestamp"),
                    "input_type": record.get("input_type"),
                    "current_phase": record.get("current_phase"),
                    "iteration_count": record.get("iteration_count"),
                    "user_input_preview": preview,
                })
            except (json.JSONDecodeError, IOError):
                continue

        return versions

    # ==========================================
    # Skills 管理 - 遵循 Agent Skills 规范
    # ==========================================

    def _parse_skill_frontmatter(self, content: str) -> Dict[str, Any]:
        """
        解析 SKILL.md 的 YAML frontmatter。

        支持标准格式：
        ---
        name: skill-name
        description: Use this skill for ...
        license: MIT
        compatibility: ...
        metadata:
          author: ...
          version: "1.0"
        allowed-tools: fetch_url
        ---

        Returns:
            解析后的 frontmatter 字典，至少包含 name 和 description
        """
        frontmatter = {}
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if not frontmatter_match:
            return frontmatter

        frontmatter_text = frontmatter_match.group(1)

        current_key = None
        current_indent = 0

        for line in frontmatter_text.split('\n'):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            # 检测缩进级别
            leading_spaces = len(line) - len(line.lstrip())

            # 顶层 key: value
            if leading_spaces == 0 and ':' in stripped:
                key, _, value = stripped.partition(':')
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if value:
                    frontmatter[key] = value
                else:
                    # 可能是嵌套对象的开始
                    frontmatter[key] = {}
                    current_key = key
                    current_indent = leading_spaces
            elif current_key and leading_spaces > current_indent and ':' in stripped:
                # 嵌套 key: value
                sub_key, _, sub_value = stripped.partition(':')
                sub_key = sub_key.strip()
                sub_value = sub_value.strip().strip('"').strip("'")
                if isinstance(frontmatter.get(current_key), dict):
                    frontmatter[current_key][sub_key] = sub_value

        return frontmatter

    def _get_skill_body(self, content: str) -> str:
        """
        提取 SKILL.md 中 frontmatter 之后的正文内容。

        Returns:
            去除 frontmatter 后的 Markdown 正文
        """
        frontmatter_match = re.match(r'^---\s*\n.*?\n---\s*\n', content, re.DOTALL)
        if frontmatter_match:
            return content[frontmatter_match.end():]
        return content

    def list_skills(self) -> List[str]:
        """
        列出 Skills/ 目录下所有可用的 skill 名称。

        每个 skill 是 Skills/ 下的一个子目录，必须包含 SKILL.md 文件。

        Returns:
            skill 名称列表（即子目录名）
        """
        if not os.path.exists(self.skills_dir):
            return []

        skill_names = []
        for entry in sorted(os.listdir(self.skills_dir)):
            entry_path = os.path.join(self.skills_dir, entry)
            skill_md_path = os.path.join(entry_path, "SKILL.md")
            if os.path.isdir(entry_path) and os.path.isfile(skill_md_path):
                skill_names.append(entry)

        return skill_names

    def load_skill(self, skill_name: str) -> Optional[str]:
        """
        加载指定 skill 的完整 SKILL.md 内容。

        Args:
            skill_name: skill 名称（Skills/ 下的子目录名）

        Returns:
            SKILL.md 的完整内容，skill 不存在则返回 None
        """
        skill_md_path = os.path.join(self.skills_dir, skill_name, "SKILL.md")
        if not os.path.isfile(skill_md_path):
            return None

        try:
            with open(skill_md_path, "r", encoding="utf-8") as file_handle:
                return file_handle.read()
        except Exception:
            return None

    def get_skill_descriptor(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定 skill 的描述信息（仅 frontmatter），用于 progressive disclosure。

        Agent 先通过 description 判断是否需要使用该 skill，
        匹配后再通过 load_skill() 读取完整内容。

        Args:
            skill_name: skill 名称

        Returns:
            包含 name、description 等 frontmatter 字段的字典，
            skill 不存在则返回 None
        """
        content = self.load_skill(skill_name)
        if content is None:
            return None

        frontmatter = self._parse_skill_frontmatter(content)
        if not frontmatter.get("name"):
            frontmatter["name"] = skill_name
        if not frontmatter.get("description"):
            frontmatter["description"] = f"Skill: {skill_name}"

        return frontmatter

    def get_all_skill_descriptors(self) -> List[Dict[str, Any]]:
        """
        获取所有可用 skill 的描述信息列表。

        用于在 Agent 系统提示中注入 Skills System 部分，
        让 Agent 按需决定是否使用某个 skill（progressive disclosure）。

        Returns:
            所有 skill 的 frontmatter 描述列表
        """
        descriptors = []
        for skill_name in self.list_skills():
            descriptor = self.get_skill_descriptor(skill_name)
            if descriptor:
                descriptor["_skill_dir"] = skill_name
                descriptors.append(descriptor)
        return descriptors

    def list_skill_files(self, skill_name: str) -> List[str]:
        """
        列出指定 skill 目录下的所有文件（相对于 workspace 根目录）。

        Args:
            skill_name: skill 名称

        Returns:
            文件路径列表（相对于 workspace 根目录）
        """
        skill_dir = os.path.join(self.skills_dir, skill_name)
        if not os.path.isdir(skill_dir):
            return []

        result = []
        for root, _dirs, files in os.walk(skill_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, os.path.dirname(self.skills_dir))
                result.append(rel_path)
        return result

    def get_skills_source_paths(self) -> List[str]:
        """
        获取 skills 源路径列表，用于传递给 create_deep_agent 的 skills 参数。

        遵循 Deep Agents SDK 规范：路径使用正斜杠，相对于 backend root。

        Returns:
            skills 源路径列表，如 ["/Skills/"]
        """
        if not os.path.exists(self.skills_dir) or not self.list_skills():
            return []
        return ["/Skills/"]

    def get_skills_dir(self) -> str:
        """返回 Skills 根目录的绝对路径"""
        return self.skills_dir

    def build_skills_files(self) -> Dict[str, Any]:
        """
        读取所有 SKILL.md 文件并用 create_file_data() 格式化，
        用于在 StateBackend 下通过 invoke(files={...}) 注入虚拟文件系统。

        StateBackend 不从磁盘读取文件，需要通过 invoke 的 files 参数
        将 skill 文件内容以 create_file_data() 格式注入到 state 中。

        Returns:
            {"/Skills/<name>/SKILL.md": create_file_data(content), ...} 格式的字典，
            如果没有可用 skill 则返回空字典
        """
        from deepagents.backends.utils import create_file_data

        skills_files = {}
        for skill_name in self.list_skills():
            content = self.load_skill(skill_name)
            if content:
                virtual_path = f"/Skills/{skill_name}/SKILL.md"
                skills_files[virtual_path] = create_file_data(content)

        if skills_files:
            print(f"   📦 已构建 {len(skills_files)} 个 skill 文件用于 StateBackend 注入")

        return skills_files


# ==========================================
# 工厂函数
# ==========================================

def create_workspace_store(workspace_dir: str = None, skills_dir: str = None) -> WorkspaceStore:
    """
    创建并返回 WorkspaceStore 实例。

    Args:
        workspace_dir: workspace 根目录路径，默认为当前模块所在目录下的 workspace/
        skills_dir: Skills 根目录路径，默认为当前模块所在目录下的 Skills/

    Returns:
        初始化完成的 WorkspaceStore 实例
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    if workspace_dir is None:
        workspace_dir = os.path.join(module_dir, "workspace")
    if skills_dir is None:
        skills_dir = os.path.join(module_dir, "Skills")
    return WorkspaceStore(workspace_dir=workspace_dir, skills_dir=skills_dir)