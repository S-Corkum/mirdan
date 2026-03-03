"""``mirdan init`` — project initialization wizard."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from mirdan.cli.detect import DetectedProject, detect_project
from mirdan.config import MirdanConfig, ProjectConfig, QualityConfig


def run_init(args: list[str]) -> None:
    """Run the project initialization wizard.

    Args:
        args: CLI arguments after ``init``.
    """
    # Parse flags
    install_hooks = False
    force_cursor = False
    force_claude_code = False
    directory = Path.cwd()
    remaining: list[str] = []

    for arg in args:
        if arg == "--hooks":
            install_hooks = True
        elif arg == "--cursor":
            force_cursor = True
        elif arg == "--claude-code":
            force_claude_code = True
        elif arg in ("--help", "-h"):
            _print_init_help()
            sys.exit(0)
        else:
            remaining.append(arg)

    if remaining:
        directory = Path(remaining[0])

    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    print(f"Initializing mirdan in {directory.resolve()}")
    print()

    # Step 1: Detect project
    detected = detect_project(directory)

    # Override IDE detection if explicit flags given
    if force_cursor and "cursor" not in detected.detected_ides:
        detected.detected_ides.append("cursor")
    if force_claude_code and "claude-code" not in detected.detected_ides:
        detected.detected_ides.append("claude-code")

    _print_detection(detected)

    # Determine platform profile
    platform = _determine_platform(detected, force_cursor, force_claude_code)

    # Step 2: Generate config (platform-aware defaults)
    config = _build_config(detected, platform)
    config_dir = directory / ".mirdan"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"

    if config_path.exists():
        print(f"Config already exists at {config_path}")
        response = input("Overwrite? [y/N] ").strip().lower()
        if response not in ("y", "yes"):
            print("Skipping config generation.")
        else:
            _write_config(config, config_path, platform)
    else:
        _write_config(config, config_path, platform)

    # Step 3: Create rules directory with template
    rules_dir = config_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    _create_rules_template(rules_dir, detected)

    # Step 4: IDE integrations (platform-specific)
    if platform == "cursor" or "cursor" in detected.detected_ides:
        _setup_cursor(directory, detected)
    if platform == "claude-code" or "claude-code" in detected.detected_ides:
        _setup_claude_code(directory, detected)

    # Step 5: Hook installation (auto-install for explicit platform profiles)
    if install_hooks or force_cursor or force_claude_code:
        _setup_hooks(directory, detected)

    print()
    print("Done! mirdan is configured for this project.")
    if platform != "generic":
        print(f"  Platform profile: {platform}")
    print()
    print("Next steps:")
    print("  1. Review .mirdan/config.yaml")
    print("  2. Customize rules in .mirdan/rules/")
    print("  3. Start the server: mirdan serve")
    if install_hooks or force_cursor or force_claude_code:
        print("  4. Hooks installed — edits will be auto-validated")


def _determine_platform(
    detected: DetectedProject,
    force_cursor: bool,
    force_claude_code: bool,
) -> str:
    """Determine the platform profile to use.

    Returns:
        "cursor", "claude-code", or "generic"
    """
    if force_cursor:
        return "cursor"
    if force_claude_code:
        return "claude-code"
    # Auto-detect from IDEs
    if "cursor" in detected.detected_ides and "claude-code" not in detected.detected_ides:
        return "cursor"
    if "claude-code" in detected.detected_ides and "cursor" not in detected.detected_ides:
        return "claude-code"
    return "generic"


def _print_detection(detected: DetectedProject) -> None:
    """Print what was detected."""
    print(f"  Project type:  {detected.project_type}")
    if detected.project_name:
        print(f"  Project name:  {detected.project_name}")
    if detected.primary_language:
        print(f"  Language:      {detected.primary_language}")
    if detected.frameworks:
        fw_list = ", ".join(detected.frameworks)
        print(f"  Frameworks:    {fw_list}")
    if detected.framework_versions:
        for fw, ver in detected.framework_versions.items():
            print(f"    {fw}: {ver}")
    if detected.detected_ides:
        ides = ", ".join(detected.detected_ides)
        print(f"  IDEs detected: {ides}")
    print()


def _build_config(detected: DetectedProject, platform: str = "generic") -> MirdanConfig:
    """Build a MirdanConfig from detected project info.

    Args:
        detected: Auto-detected project metadata.
        platform: Platform profile ("cursor", "claude-code", "generic").
    """
    project = ProjectConfig(
        name=detected.project_name,
        type=detected.project_type,
        primary_language=detected.primary_language,
        frameworks=detected.frameworks,
    )
    quality = QualityConfig(
        security="strict",
        architecture="moderate",
        documentation="moderate",
        testing="strict",
    )
    return MirdanConfig(project=project, quality=quality)


def _write_config(config: MirdanConfig, path: Path, platform: str = "generic") -> None:
    """Write config to YAML file with platform-aware defaults."""
    data = config.model_dump(exclude_defaults=False)
    # Simplify: only write project and quality sections
    output: dict[str, Any] = {
        "version": data["version"],
        "project": data["project"],
        "quality": data["quality"],
    }

    # Platform-specific defaults
    if platform == "cursor":
        output["platform"] = {
            "name": "cursor",
            "context_level": "none",  # Cursor gathers its own context
            "tool_budget_aware": True,  # 4-tool mode for 40-tool limit
        }
    elif platform == "claude-code":
        output["platform"] = {
            "name": "claude-code",
            "context_level": "auto",  # Claude Code benefits from context gathering
            "tool_budget_aware": False,
        }

    with path.open("w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
    print(f"  Created {path}")


def _create_rules_template(rules_dir: Path, detected: DetectedProject) -> None:
    """Create an example custom rules YAML file."""
    template_path = rules_dir / "example.yaml"
    if template_path.exists():
        return

    template: dict[str, list[dict[str, str]]] = {
        "rules": [
            {
                "id": "CUSTOM001",
                "rule": "no-print-statements",
                "pattern": r"\bprint\s*\(",
                "severity": "warning",
                "message": "print() detected — use logging instead",
                "suggestion": "Replace print() with logging.info() or logging.debug()",
            },
        ],
    }

    with template_path.open("w") as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)
    print(f"  Created {template_path}")


def _setup_cursor(directory: Path, detected: DetectedProject) -> None:
    """Generate .cursor/rules/ .mdc files."""
    from mirdan.integrations.cursor import generate_cursor_rules

    rules_dir = directory / ".cursor" / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    generated = generate_cursor_rules(rules_dir, detected)
    for path in generated:
        print(f"  Created {path}")


def _setup_claude_code(directory: Path, detected: DetectedProject) -> None:
    """Generate Claude Code configuration files, skills, agents, and MCP config."""
    from mirdan.integrations.claude_code import (
        generate_agents,
        generate_claude_code_config,
        generate_mcp_json,
        generate_skills,
    )

    # MCP server registration
    mcp_path = generate_mcp_json(directory)
    print(f"  Created {mcp_path}")

    # Hooks + rules
    generated = generate_claude_code_config(directory, detected)
    for path in generated:
        print(f"  Created {path}")

    # Skills
    skill_paths = generate_skills(directory, detected)
    for path in skill_paths:
        print(f"  Created {path}")

    # Agents
    agent_paths = generate_agents(directory, detected)
    for path in agent_paths:
        print(f"  Created {path}")

    print()
    print("  Claude Code configured! mirdan will validate edits automatically.")
    if skill_paths:
        print("  Skills installed: /mirdan:code, /mirdan:debug, /mirdan:review")


def _setup_hooks(directory: Path, detected: DetectedProject) -> None:
    """Install hook scripts for detected IDEs.

    Copies hook scripts to appropriate locations based on detected IDEs.
    Falls back to pre-commit config if no IDE-specific hooks apply.
    """
    hooks_installed = False
    hooks_pkg_dir = _get_hooks_dir()

    # Cursor hooks
    if "cursor" in detected.detected_ides:
        hooks_installed |= _install_cursor_hooks(directory, hooks_pkg_dir)

    # Claude Code hooks
    if "claude-code" in detected.detected_ides:
        hooks_installed |= _install_claude_code_hooks(directory, hooks_pkg_dir)

    # Git pre-commit hook
    git_dir = directory / ".git"
    if git_dir.is_dir():
        hooks_installed |= _install_git_pre_commit(directory, hooks_pkg_dir)

    # Pre-commit config as fallback
    if not hooks_installed:
        _install_pre_commit_config(directory, hooks_pkg_dir)


def _get_hooks_dir() -> Path:
    """Get the path to the bundled hooks directory."""
    # hooks/ is at the package root, sibling to src/
    pkg_dir = Path(__file__).resolve().parent.parent.parent.parent
    hooks_dir = pkg_dir / "hooks"
    if hooks_dir.is_dir():
        return hooks_dir
    # Fallback: look relative to the installed package
    return Path(__file__).resolve().parent.parent.parent / "hooks"


def _install_cursor_hooks(directory: Path, hooks_pkg_dir: Path) -> bool:
    """Install Cursor-specific hooks."""
    src = hooks_pkg_dir / "cursor" / "post-edit.sh"
    if not src.exists():
        return False

    dest_dir = directory / ".cursor" / "hooks"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "post-edit.sh"
    shutil.copy2(src, dest)
    dest.chmod(0o755)
    print(f"  Installed {dest}")
    return True


def _install_claude_code_hooks(directory: Path, hooks_pkg_dir: Path) -> bool:
    """Install Claude Code hooks."""
    src = hooks_pkg_dir / "claude-code" / "hooks.json"
    if not src.exists():
        return False

    dest_dir = directory / ".claude"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "hooks.json"

    if dest.exists():
        print(f"  {dest} already exists, skipping")
        return True

    shutil.copy2(src, dest)
    print(f"  Installed {dest}")
    return True


def _install_git_pre_commit(directory: Path, hooks_pkg_dir: Path) -> bool:
    """Install git pre-commit hook."""
    src = hooks_pkg_dir / "cursor" / "pre-commit.sh"
    if not src.exists():
        return False

    dest = directory / ".git" / "hooks" / "pre-commit"
    if dest.exists():
        print(f"  {dest} already exists, skipping")
        return True

    shutil.copy2(src, dest)
    dest.chmod(0o755)
    print(f"  Installed {dest}")
    return True


def _install_pre_commit_config(directory: Path, hooks_pkg_dir: Path) -> bool:
    """Install .pre-commit-config.yaml."""
    src = hooks_pkg_dir / "pre-commit-config.yaml"
    if not src.exists():
        return False

    dest = directory / ".pre-commit-config.yaml"
    if dest.exists():
        print(f"  {dest} already exists, skipping")
        return True

    shutil.copy2(src, dest)
    print(f"  Installed {dest}")
    return True


def _print_init_help() -> None:
    """Print usage help for the init command."""
    print("Usage: mirdan init [directory] [options]")
    print()
    print("Options:")
    print("  --hooks        Install hook scripts for auto-validation")
    print("  --cursor       Use Cursor platform profile")
    print("  --claude-code  Use Claude Code platform profile")
    print("  -h, --help     Show this help")
