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
