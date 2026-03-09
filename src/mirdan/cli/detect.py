"""Project type, framework, and IDE auto-detection."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DetectedProject:
    """Result of project auto-detection."""

    project_type: str = "unknown"  # python, node, rust, go, java, unknown
    project_name: str = ""
    primary_language: str = ""
    frameworks: list[str] = field(default_factory=list)
    framework_versions: dict[str, str] = field(default_factory=dict)
    detected_ides: list[str] = field(default_factory=list)
    manifest_path: str = ""
    sub_path: str = ""  # relative path from workspace root (empty for single-project)


@dataclass
class DetectedWorkspace:
    """Result of workspace/monorepo detection."""

    root_dir: str = ""
    workspace_type: str = "auto"  # auto, uv, npm, cargo, vscode
    projects: list[DetectedProject] = field(default_factory=list)
    all_languages: list[str] = field(default_factory=list)
    all_frameworks: list[str] = field(default_factory=list)
    detected_ides: list[str] = field(default_factory=list)


def detect_project(directory: Path | None = None) -> DetectedProject:
    """Detect the project type, frameworks, and IDE from the given directory.

    Args:
        directory: Project root directory. Defaults to cwd.

    Returns:
        DetectedProject with all discovered metadata.
    """
    if directory is None:
        directory = Path.cwd()

    result = DetectedProject()

    # Detect project type from manifests (order: most specific first)
    _detect_from_pyproject(directory, result)
    if result.project_type == "unknown":
        _detect_from_package_json(directory, result)
    if result.project_type == "unknown":
        _detect_from_cargo(directory, result)
    if result.project_type == "unknown":
        _detect_from_go_mod(directory, result)
    if result.project_type == "unknown":
        _detect_from_pom(directory, result)

    # Detect IDEs from directory presence
    _detect_ides(directory, result)

    return result


def detect_framework_version(framework: str, directory: Path | None = None) -> str | None:
    """Detect the version of a specific framework from project manifests.

    Args:
        framework: Framework name to look up (e.g., "react", "fastapi").
        directory: Project root directory. Defaults to cwd.

    Returns:
        Version string if found, None otherwise.
    """
    if directory is None:
        directory = Path.cwd()

    # Check pyproject.toml
    pyproject = directory / "pyproject.toml"
    if pyproject.exists():
        version = _get_python_dep_version(pyproject, framework)
        if version:
            return version

    # Check package.json
    pkg_json = directory / "package.json"
    if pkg_json.exists():
        version = _get_node_dep_version(pkg_json, framework)
        if version:
            return version

    return None


def _detect_from_pyproject(directory: Path, result: DetectedProject) -> None:
    """Detect Python project from pyproject.toml."""
    pyproject = directory / "pyproject.toml"
    if not pyproject.exists():
        return

    result.project_type = "python"
    result.primary_language = "python"
    result.manifest_path = str(pyproject)

    try:
        content = pyproject.read_text()
    except OSError:
        return

    # Extract project name
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("name") and "=" in stripped:
            name = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            result.project_name = name
            break

    # Detect Python frameworks from dependencies
    python_frameworks = {
        "fastapi": "fastapi",
        "django": "django",
        "flask": "flask",
        "sqlalchemy": "sqlalchemy",
        "pydantic": "pydantic",
        "celery": "celery",
        "pytest": "pytest",
    }
    deps_text = content.lower()
    for dep, framework in python_frameworks.items():
        if dep in deps_text:
            result.frameworks.append(framework)
            version = _get_python_dep_version(pyproject, dep)
            if version:
                result.framework_versions[framework] = version


def _detect_from_package_json(directory: Path, result: DetectedProject) -> None:
    """Detect Node.js project from package.json."""
    pkg_json = directory / "package.json"
    if not pkg_json.exists():
        return

    result.project_type = "node"
    result.manifest_path = str(pkg_json)

    try:
        data = json.loads(pkg_json.read_text())
    except (OSError, json.JSONDecodeError):
        return

    result.project_name = data.get("name", "")

    # Detect primary language (TypeScript if tsconfig present)
    if (directory / "tsconfig.json").exists():
        result.primary_language = "typescript"
    else:
        result.primary_language = "javascript"

    # Detect Node frameworks
    all_deps: dict[str, str] = {}
    all_deps.update(data.get("dependencies", {}))
    all_deps.update(data.get("devDependencies", {}))

    node_frameworks = {
        "react": "react",
        "next": "next.js",
        "vue": "vue",
        "nuxt": "nuxt",
        "svelte": "svelte",
        "@nestjs/core": "nestjs",
        "express": "express",
        "tailwindcss": "tailwind",
    }
    for dep, framework in node_frameworks.items():
        if dep in all_deps:
            result.frameworks.append(framework)
            version = all_deps[dep].lstrip("^~>=<")
            if version:
                result.framework_versions[framework] = version


def _detect_from_cargo(directory: Path, result: DetectedProject) -> None:
    """Detect Rust project from Cargo.toml."""
    cargo = directory / "Cargo.toml"
    if not cargo.exists():
        return

    result.project_type = "rust"
    result.primary_language = "rust"
    result.manifest_path = str(cargo)

    try:
        content = cargo.read_text()
    except OSError:
        return

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("name") and "=" in stripped:
            name = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            result.project_name = name
            break

    rust_frameworks = ["actix-web", "axum", "rocket", "tokio", "serde"]
    for fw in rust_frameworks:
        if fw in content:
            result.frameworks.append(fw)


def _detect_from_go_mod(directory: Path, result: DetectedProject) -> None:
    """Detect Go project from go.mod."""
    go_mod = directory / "go.mod"
    if not go_mod.exists():
        return

    result.project_type = "go"
    result.primary_language = "go"
    result.manifest_path = str(go_mod)

    try:
        content = go_mod.read_text()
    except OSError:
        return

    lines = content.splitlines()
    if lines:
        first = lines[0].strip()
        if first.startswith("module "):
            result.project_name = first.split(" ", 1)[1].strip()

    go_frameworks = {"gin-gonic/gin": "gin", "labstack/echo": "echo"}
    for dep, framework in go_frameworks.items():
        if dep in content:
            result.frameworks.append(framework)


def _detect_from_pom(directory: Path, result: DetectedProject) -> None:
    """Detect Java project from pom.xml or build.gradle."""
    if (directory / "pom.xml").exists():
        result.project_type = "java"
        result.primary_language = "java"
        result.manifest_path = str(directory / "pom.xml")
    elif (directory / "build.gradle").exists() or (directory / "build.gradle.kts").exists():
        result.project_type = "java"
        result.primary_language = "java"
        gradle = "build.gradle.kts" if (directory / "build.gradle.kts").exists() else "build.gradle"
        result.manifest_path = str(directory / gradle)


def _detect_ides(directory: Path, result: DetectedProject) -> None:
    """Detect IDEs from directory presence and environment variables."""
    ide_dirs = {
        ".cursor": "cursor",
        ".claude": "claude-code",
        ".vscode": "vscode",
        ".idea": "intellij",
    }
    for dirname, ide in ide_dirs.items():
        if (directory / dirname).is_dir():
            result.detected_ides.append(ide)

    # Also check environment variables for IDE detection
    if "cursor" not in result.detected_ides and os.environ.get("CURSOR_TRACE_ID"):
        result.detected_ides.append("cursor")
    if "claude-code" not in result.detected_ides and os.environ.get("CLAUDE_CODE_RUNNING"):
        result.detected_ides.append("claude-code")


def _get_python_dep_version(pyproject: Path, dep_name: str) -> str | None:
    """Extract a dependency version from pyproject.toml.

    Handles both simple 'dep>=version' and PEP 621 formats.
    """
    try:
        content = pyproject.read_text()
    except OSError:
        return None

    # Look for dependency lines like: "fastapi>=0.100.0"
    for line in content.splitlines():
        stripped = line.strip().strip('"').strip("'").strip(",")
        if dep_name in stripped.lower():
            # Extract version specifier
            for sep in [">=", "==", "~=", "<=", ">"]:
                if sep in stripped:
                    version = stripped.split(sep, 1)[1].strip().rstrip('"').rstrip("'").rstrip(",")
                    # Take only the first version number (before any comma)
                    version = version.split(",")[0].strip()
                    if version and version[0].isdigit():
                        return version
    return None


def _get_node_dep_version(pkg_json: Path, dep_name: str) -> str | None:
    """Extract a dependency version from package.json."""
    try:
        data = json.loads(pkg_json.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    # Check both dependencies and devDependencies
    for section in ["dependencies", "devDependencies"]:
        deps = data.get(section, {})
        if dep_name in deps:
            version: str = str(deps[dep_name]).lstrip("^~>=<")
            return version

    return None


# ---------------------------------------------------------------------------
# Workspace / Monorepo Detection
# ---------------------------------------------------------------------------


def detect_workspace(directory: Path | None = None) -> DetectedWorkspace | None:
    """Detect a workspace/monorepo containing multiple sub-projects.

    Checks for explicit workspace manifests first (uv, npm, cargo, vscode),
    then falls back to scanning immediate subdirectories for projects.

    Returns None if fewer than 2 sub-projects are found.

    Args:
        directory: Workspace root directory. Defaults to cwd.

    Returns:
        DetectedWorkspace if a workspace is found, None otherwise.
    """
    if directory is None:
        directory = Path.cwd()

    workspace = DetectedWorkspace(root_dir=str(directory))

    # 1. Check root manifests for workspace declarations
    found = _detect_workspace_from_pyproject(directory, workspace)
    if not found:
        found = _detect_workspace_from_package_json(directory, workspace)
    if not found:
        found = _detect_workspace_from_cargo(directory, workspace)
    if not found:
        found = _detect_workspace_from_vscode(directory, workspace)

    # 2. Fallback: scan immediate subdirectories
    if not found:
        _scan_subdirectories(directory, workspace)
        if workspace.projects:
            workspace.workspace_type = "auto"

    # 3. Require at least 2 sub-projects
    if len(workspace.projects) < 2:
        return None

    # 4. Set sub_path on each project and aggregate languages/frameworks
    seen_langs: list[str] = []
    seen_frameworks: list[str] = []
    for proj in workspace.projects:
        if not proj.sub_path:
            proj.sub_path = str(Path(proj.manifest_path).parent.relative_to(directory))
            if proj.sub_path == ".":
                proj.sub_path = ""
        if proj.primary_language and proj.primary_language not in seen_langs:
            seen_langs.append(proj.primary_language)
        for fw in proj.frameworks:
            if fw not in seen_frameworks:
                seen_frameworks.append(fw)

    workspace.all_languages = seen_langs
    workspace.all_frameworks = seen_frameworks

    # 5. IDE detection on root
    ide_result = DetectedProject()
    _detect_ides(directory, ide_result)
    workspace.detected_ides = ide_result.detected_ides

    return workspace


def resolve_file_to_project(workspace: DetectedWorkspace, file_path: str) -> DetectedProject | None:
    """Resolve a file path to its containing sub-project.

    Uses longest prefix match against sub-project paths.

    Args:
        workspace: The detected workspace.
        file_path: File path (relative to workspace root).

    Returns:
        The matching DetectedProject, or None if no match.
    """
    best_match: DetectedProject | None = None
    best_len = 0
    for proj in workspace.projects:
        prefix = proj.sub_path
        if not prefix:
            continue
        # Ensure prefix ends with / for correct matching
        normalized = prefix if prefix.endswith("/") else prefix + "/"
        if file_path.startswith(normalized) and len(prefix) > best_len:
            best_match = proj
            best_len = len(prefix)
    return best_match


def _detect_workspace_from_pyproject(directory: Path, workspace: DetectedWorkspace) -> bool:
    """Check pyproject.toml for [tool.uv.workspace] members."""
    pyproject = directory / "pyproject.toml"
    if not pyproject.exists():
        return False

    try:
        content = pyproject.read_text()
    except OSError:
        return False

    # Look for [tool.uv.workspace] section with members
    in_uv_workspace = False
    members: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "[tool.uv.workspace]":
            in_uv_workspace = True
            continue
        if in_uv_workspace:
            if stripped.startswith("[") and stripped.endswith("]") and "=" not in stripped:
                break  # new section
            if stripped.startswith("members") and "=" in stripped:
                # Parse members = ["api", "web", "shared"]
                value = stripped.split("=", 1)[1].strip()
                members.extend(_parse_toml_string_array(value, content, stripped))

    if not members:
        return False

    workspace.workspace_type = "uv"
    for member_pattern in members:
        # Handle glob patterns like "packages/*"
        member_dirs = _resolve_glob_members(directory, member_pattern)
        for member_dir in member_dirs:
            proj = detect_project(member_dir)
            if proj.project_type != "unknown":
                proj.sub_path = str(member_dir.relative_to(directory))
                workspace.projects.append(proj)

    return len(workspace.projects) > 0


def _detect_workspace_from_package_json(directory: Path, workspace: DetectedWorkspace) -> bool:
    """Check package.json for 'workspaces' key."""
    pkg_json = directory / "package.json"
    if not pkg_json.exists():
        return False

    try:
        data = json.loads(pkg_json.read_text())
    except (OSError, json.JSONDecodeError):
        return False

    workspaces = data.get("workspaces")
    if not workspaces:
        return False

    # workspaces can be a list or {"packages": [...]}
    if isinstance(workspaces, dict):
        workspaces = workspaces.get("packages", [])
    if not isinstance(workspaces, list):
        return False

    workspace.workspace_type = "npm"
    for pattern in workspaces:
        member_dirs = _resolve_glob_members(directory, pattern)
        for member_dir in member_dirs:
            proj = detect_project(member_dir)
            if proj.project_type != "unknown":
                proj.sub_path = str(member_dir.relative_to(directory))
                workspace.projects.append(proj)

    return len(workspace.projects) > 0


def _detect_workspace_from_cargo(directory: Path, workspace: DetectedWorkspace) -> bool:
    """Check Cargo.toml for [workspace] members."""
    cargo = directory / "Cargo.toml"
    if not cargo.exists():
        return False

    try:
        content = cargo.read_text()
    except OSError:
        return False

    in_workspace = False
    members: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "[workspace]":
            in_workspace = True
            continue
        if in_workspace:
            if stripped.startswith("[") and stripped.endswith("]") and "=" not in stripped:
                break
            if stripped.startswith("members") and "=" in stripped:
                value = stripped.split("=", 1)[1].strip()
                members.extend(_parse_toml_string_array(value, content, stripped))

    if not members:
        return False

    workspace.workspace_type = "cargo"
    for member_pattern in members:
        member_dirs = _resolve_glob_members(directory, member_pattern)
        for member_dir in member_dirs:
            proj = detect_project(member_dir)
            if proj.project_type != "unknown":
                proj.sub_path = str(member_dir.relative_to(directory))
                workspace.projects.append(proj)

    return len(workspace.projects) > 0


def _detect_workspace_from_vscode(directory: Path, workspace: DetectedWorkspace) -> bool:
    """Check for .code-workspace file with folders."""
    for item in directory.iterdir():
        if item.suffix == ".code-workspace" and item.is_file():
            try:
                data = json.loads(item.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            folders = data.get("folders", [])
            if not isinstance(folders, list):
                continue
            workspace.workspace_type = "vscode"
            for folder in folders:
                folder_path = folder.get("path", "") if isinstance(folder, dict) else str(folder)
                if not folder_path:
                    continue
                member_dir = (directory / folder_path).resolve()
                if member_dir.is_dir():
                    proj = detect_project(member_dir)
                    if proj.project_type != "unknown":
                        proj.sub_path = str(member_dir.relative_to(directory))
                        workspace.projects.append(proj)
            if workspace.projects:
                return True
    return False


def _scan_subdirectories(directory: Path, workspace: DetectedWorkspace) -> None:
    """Scan immediate subdirectories for projects (auto-detection fallback)."""
    try:
        children = sorted(directory.iterdir())
    except OSError:
        return

    for child in children:
        if not child.is_dir():
            continue
        # Skip hidden directories and common non-project dirs
        if child.name.startswith(".") or child.name in {
            "node_modules",
            "__pycache__",
            "venv",
            ".venv",
            "dist",
            "build",
            "target",
        }:
            continue
        proj = detect_project(child)
        if proj.project_type != "unknown":
            proj.sub_path = child.name
            workspace.projects.append(proj)


def _parse_toml_string_array(value: str, full_content: str, current_line: str) -> list[str]:
    """Parse a TOML string array like '["api", "web"]' from a value string.

    Handles single-line arrays and basic multi-line arrays.
    """
    items: list[str] = []
    # Single-line: ["api", "web"]
    if "[" in value and "]" in value:
        inner = value[value.index("[") + 1 : value.index("]")]
        for part in inner.split(","):
            cleaned = part.strip().strip('"').strip("'")
            if cleaned:
                items.append(cleaned)
    elif "[" in value:
        # Multi-line array: collect until we find ]
        lines = full_content.splitlines()
        collecting = False
        for line in lines:
            if current_line in line:
                collecting = True
                # Parse items on the same line as the opening bracket
                if "[" in line:
                    after_bracket = line[line.index("[") + 1 :]
                    for part in after_bracket.split(","):
                        cleaned = part.strip().strip('"').strip("'").rstrip(",")
                        if cleaned and cleaned != "]":
                            items.append(cleaned)
                continue
            if collecting:
                stripped = line.strip()
                if "]" in stripped:
                    before_bracket = stripped[: stripped.index("]")]
                    for part in before_bracket.split(","):
                        cleaned = part.strip().strip('"').strip("'").rstrip(",")
                        if cleaned:
                            items.append(cleaned)
                    break
                cleaned = stripped.strip('"').strip("'").rstrip(",")
                if cleaned:
                    items.append(cleaned)
    return items


def _resolve_glob_members(directory: Path, pattern: str) -> list[Path]:
    """Resolve a glob pattern to actual directories.

    Handles both literal paths and glob patterns like 'packages/*'.
    """
    if "*" in pattern:
        # Use glob matching
        return sorted(p for p in directory.glob(pattern) if p.is_dir())
    else:
        # Literal path
        member_dir = directory / pattern
        if member_dir.is_dir():
            return [member_dir]
        return []
