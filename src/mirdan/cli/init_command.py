"""``mirdan init`` — project initialization wizard."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from mirdan.cli.detect import DetectedProject, DetectedWorkspace, detect_project, detect_workspace
from mirdan.config import (
    MirdanConfig,
    PlatformProfile,
    ProjectConfig,
    QualityConfig,
    SubProjectConfig,
    WorkspaceConfig,
)


def run_init(args: list[str]) -> None:
    """Run the project initialization wizard.

    Args:
        args: CLI arguments after ``init``.
    """
    # Parse flags
    install_hooks = False
    force_cursor = False
    force_claude_code = False
    force_all = False
    upgrade = False
    learn = False
    quality_profile = ""
    directory = Path.cwd()
    remaining: list[str] = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--hooks":
            install_hooks = True
        elif arg == "--cursor":
            force_cursor = True
        elif arg == "--claude-code":
            force_claude_code = True
        elif arg == "--all":
            force_all = True
        elif arg == "--upgrade":
            upgrade = True
        elif arg == "--learn":
            learn = True
        elif arg.startswith("--quality-profile"):
            if "=" in arg:
                quality_profile = arg.split("=", 1)[1]
            elif i + 1 < len(args):
                i += 1
                quality_profile = args[i]
        elif arg in ("--help", "-h"):
            _print_init_help()
            sys.exit(0)
        else:
            remaining.append(arg)
        i += 1

    # --all implies both platforms
    if force_all:
        force_cursor = True
        force_claude_code = True

    if remaining:
        directory = Path(remaining[0])

    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    if upgrade:
        _run_upgrade(directory)
        return

    print(f"Initializing mirdan in {directory.resolve()}")
    print()

    # Step 1: Detect workspace first, then fall back to single project
    workspace = detect_workspace(directory)
    is_workspace = workspace is not None

    if is_workspace and workspace is not None:
        _print_workspace_detection(workspace)
        detected = _workspace_as_detected(workspace)
        languages: list[str] | None = workspace.all_languages
    else:
        detected = detect_project(directory)
        languages = None

    # Override IDE detection if explicit flags given
    if force_cursor and "cursor" not in detected.detected_ides:
        detected.detected_ides.append("cursor")
    if force_claude_code and "claude-code" not in detected.detected_ides:
        detected.detected_ides.append("claude-code")

    _print_detection(detected)

    # Determine platform profile
    platform = _determine_platform(detected, force_cursor, force_claude_code)

    # Step 2: Generate config (platform-aware defaults)
    if is_workspace and workspace is not None:
        config = _build_workspace_config(workspace, platform, quality_profile=quality_profile)
    else:
        config = _build_config(detected, platform, quality_profile=quality_profile)
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

    # Step 3b: Learn conventions from codebase
    if learn:
        _learn_conventions(directory, rules_dir, detected, workspace=workspace)

    # Step 4: IDE integrations (platform-specific)
    if platform == "cursor" or "cursor" in detected.detected_ides:
        _setup_cursor(directory, detected, languages=languages)
    if platform == "claude-code" or "claude-code" in detected.detected_ides:
        _setup_claude_code(directory, detected, languages=languages)

    # Step 5: Generate root AGENTS.md (cross-platform standard)
    _setup_agents_md(directory, detected, platform, languages=languages)

    # Step 6: pre-commit configuration
    _setup_precommit(directory, detected)

    # Step 7: Hook installation (auto-install for explicit platform profiles)
    if install_hooks or force_cursor or force_claude_code:
        _setup_hooks(directory, detected)

    print()
    if is_workspace and workspace is not None:
        n_projects = len(workspace.projects)
        print(f"Done! mirdan is configured for this workspace ({n_projects} projects).")
    else:
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


def _learn_conventions(
    directory: Path,
    rules_dir: Path,
    detected: DetectedProject,
    workspace: DetectedWorkspace | None = None,
) -> None:
    """Scan the project and generate custom rules from conventions.

    In workspace mode, iterates each sub-project and scans independently,
    saving per-project conventions under a ``projects`` key.

    Args:
        directory: Project root directory.
        rules_dir: Rules directory to write to.
        detected: Auto-detected project metadata.
        workspace: Optional detected workspace for multi-project scanning.
    """
    from mirdan.core.convention_extractor import ConventionExtractor
    from mirdan.core.rule_generator import RuleGenerator

    if workspace is not None:
        # Workspace mode: scan each sub-project independently
        print("  Scanning workspace projects for conventions...")
        all_project_conventions: dict[str, Any] = {"projects": {}}
        total_rules = 0

        for proj in workspace.projects:
            proj_dir = directory / proj.sub_path if proj.sub_path else directory
            if not proj_dir.is_dir():
                continue

            proj_name = proj.project_name or proj.sub_path or "root"
            print(f"    Scanning {proj_name}...")
            extractor = ConventionExtractor()
            language = proj.primary_language or "auto"
            result = extractor.scan(proj_dir, language=language)

            if not result.conventions:
                print(f"    No conventions found in {proj_name}")
                continue

            print(f"    Found {len(result.conventions)} convention(s) in {proj_name}")
            generator = RuleGenerator()
            rules = generator.generate_from_conventions(result.conventions)

            if rules:
                all_project_conventions["projects"][proj_name] = {
                    "path": proj.sub_path,
                    "language": proj.primary_language,
                    "rules": [r.model_dump() if hasattr(r, "model_dump") else r for r in rules],
                }
                total_rules += len(rules)

        if total_rules > 0:
            output_path = rules_dir / "conventions.yaml"
            import yaml

            with output_path.open("w") as f:
                yaml.dump(all_project_conventions, f, default_flow_style=False, sort_keys=False)
            print(f"  Created {output_path} ({total_rules} rules across workspace)")
        else:
            print("  No high-confidence conventions found in workspace.")
        return

    # Single-project mode
    print("  Scanning codebase for conventions...")
    extractor = ConventionExtractor()
    language = detected.primary_language or "auto"
    result = extractor.scan(directory, language=language)

    if not result.conventions:
        print("  No conventions discovered.")
        return

    print(f"  Found {len(result.conventions)} convention(s)")

    generator = RuleGenerator()
    rules = generator.generate_from_conventions(result.conventions)

    if not rules:
        print("  No high-confidence conventions to generate rules from.")
        return

    output_path = rules_dir / "conventions.yaml"
    generator.generate_and_write(result.conventions, output_path)
    print(f"  Created {output_path} ({len(rules)} rules)")


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


def _build_config(
    detected: DetectedProject,
    platform: str = "generic",
    quality_profile: str = "",
) -> MirdanConfig:
    """Build a MirdanConfig from detected project info.

    Args:
        detected: Auto-detected project metadata.
        platform: Platform profile ("cursor", "claude-code", "generic").
        quality_profile: Named quality profile (e.g. "enterprise").
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

    # Build platform profile
    if platform == "cursor":
        plat = PlatformProfile(
            name="cursor",
            context_level="none",
            tool_budget_aware=True,
        )
    elif platform == "claude-code":
        plat = PlatformProfile(
            name="claude-code",
            context_level="auto",
            tool_budget_aware=False,
        )
    else:
        plat = PlatformProfile()

    config = MirdanConfig(project=project, quality=quality, platform=plat)

    if quality_profile:
        config.quality_profile = quality_profile

    return config


def _workspace_as_detected(workspace: DetectedWorkspace) -> DetectedProject:
    """Create a synthetic DetectedProject from a workspace.

    Summarises the workspace as a single project entry for functions
    that expect a ``DetectedProject``.

    Args:
        workspace: The detected workspace.

    Returns:
        A synthetic DetectedProject with ``project_type="workspace"``,
        ``primary_language`` set to the most common language, and
        ``frameworks`` set to the union of all sub-project frameworks.
    """
    from collections import Counter

    # Most common language across sub-projects
    lang_counts: Counter[str] = Counter()
    for proj in workspace.projects:
        if proj.primary_language:
            lang_counts[proj.primary_language] += 1
    primary_language = lang_counts.most_common(1)[0][0] if lang_counts else ""

    # Union of all frameworks
    seen: set[str] = set()
    all_frameworks: list[str] = []
    for proj in workspace.projects:
        for fw in proj.frameworks:
            if fw not in seen:
                seen.add(fw)
                all_frameworks.append(fw)

    return DetectedProject(
        project_type="workspace",
        project_name="",
        primary_language=primary_language,
        frameworks=all_frameworks,
        detected_ides=list(workspace.detected_ides),
    )


def _build_workspace_config(
    workspace: DetectedWorkspace,
    platform: str = "generic",
    quality_profile: str = "",
) -> MirdanConfig:
    """Build a MirdanConfig for a workspace with multiple sub-projects.

    Args:
        workspace: The detected workspace.
        platform: Platform profile ("cursor", "claude-code", "generic").
        quality_profile: Named quality profile (e.g. "enterprise").
    """
    # Build synthetic detected for the top-level project config
    detected = _workspace_as_detected(workspace)
    project = ProjectConfig(
        name=detected.project_name,
        type="workspace",
        primary_language=detected.primary_language,
        frameworks=detected.frameworks,
    )
    quality = QualityConfig(
        security="strict",
        architecture="moderate",
        documentation="moderate",
        testing="strict",
    )

    # Build platform profile
    if platform == "cursor":
        plat = PlatformProfile(
            name="cursor",
            context_level="none",
            tool_budget_aware=True,
        )
    elif platform == "claude-code":
        plat = PlatformProfile(
            name="claude-code",
            context_level="auto",
            tool_budget_aware=False,
        )
    else:
        plat = PlatformProfile()

    # Build sub-project configs
    sub_projects = [
        SubProjectConfig(
            path=proj.sub_path,
            name=proj.project_name,
            primary_language=proj.primary_language,
            frameworks=list(proj.frameworks),
        )
        for proj in workspace.projects
    ]

    ws_config = WorkspaceConfig(
        enabled=True,
        workspace_type=workspace.workspace_type,
        projects=sub_projects,
    )

    config = MirdanConfig(
        version="2.0",
        project=project,
        quality=quality,
        platform=plat,
        workspace=ws_config,
    )

    if quality_profile:
        config.quality_profile = quality_profile

    return config


def _print_workspace_detection(workspace: DetectedWorkspace) -> None:
    """Pretty-print detected workspace projects.

    Args:
        workspace: The detected workspace.
    """
    print(f"  Workspace detected ({workspace.workspace_type})")
    print(f"  Sub-projects: {len(workspace.projects)}")
    for proj in workspace.projects:
        lang_str = f" [{proj.primary_language}]" if proj.primary_language else ""
        fw_str = f" ({', '.join(proj.frameworks)})" if proj.frameworks else ""
        name = proj.project_name or proj.sub_path or "(root)"
        print(f"    - {name}{lang_str}{fw_str}")
    if workspace.all_languages:
        print(f"  Languages: {', '.join(workspace.all_languages)}")
    if workspace.all_frameworks:
        print(f"  Frameworks: {', '.join(workspace.all_frameworks)}")
    print()


def _write_config(config: MirdanConfig, path: Path, platform: str = "generic") -> None:
    """Write config to YAML file with platform-aware defaults."""
    data = config.model_dump(exclude_defaults=False)
    # Simplify: only write project and quality sections
    output: dict[str, Any] = {
        "version": data["version"],
        "project": data["project"],
        "quality": data["quality"],
    }

    # Write platform profile from config
    plat_data = config.platform.model_dump()
    if plat_data["name"] != "generic":
        output["platform"] = plat_data

    # Write quality profile if non-default
    if config.quality_profile and config.quality_profile != "default":
        output["quality_profile"] = config.quality_profile

    # Write workspace section when workspace is enabled
    if config.is_workspace:
        output["workspace"] = config.workspace.model_dump()

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


def _setup_cursor(
    directory: Path,
    detected: DetectedProject,
    languages: list[str] | None = None,
    *,
    force_regenerate: bool = False,
) -> None:
    """Generate Cursor configuration: hooks, rules, agents, and MCP config.

    Args:
        directory: Project root directory.
        detected: Auto-detected project metadata.
        languages: Optional list of languages for multi-language rule generation.
        force_regenerate: If True, overwrite existing commands, subagents, and
            skills with latest mirdan content (used by --upgrade).
    """
    from mirdan.core.quality_standards import QualityStandards
    from mirdan.integrations.cursor import (
        generate_cursor_agents,
        generate_cursor_commands,
        generate_cursor_environment,
        generate_cursor_hook_scripts,
        generate_cursor_hooks,
        generate_cursor_mcp_json,
        generate_cursor_rules,
        generate_cursor_sandbox,
        generate_cursor_skills,
        generate_cursor_subagents,
    )

    # Try to use dynamic generation with QualityStandards
    try:
        standards = QualityStandards()
    except Exception:
        standards = None

    cursor_dir = directory / ".cursor"

    # Generate hooks.json (only if not exists — respect user customizations)
    hooks_path = generate_cursor_hooks(cursor_dir)
    if hooks_path:
        print(f"  Created {hooks_path}")
    else:
        # hooks.json exists, but ensure hook scripts are present
        script_paths = generate_cursor_hook_scripts(cursor_dir)
        for path in script_paths:
            print(f"  Created {path}")

    # Generate .cursor/rules/*.mdc
    rules_dir = cursor_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    generated = generate_cursor_rules(rules_dir, detected, standards=standards, languages=languages)
    for path in generated:
        print(f"  Created {path}")

    # Generate AGENTS.md and BUGBOT.md
    agent_paths = generate_cursor_agents(cursor_dir, detected, standards=standards)
    for path in agent_paths:
        print(f"  Created {path}")

    # Generate .cursor/commands/*.md
    command_paths = generate_cursor_commands(cursor_dir, force=force_regenerate)
    for path in command_paths:
        print(f"  Created {path}")

    # Generate .cursor/agents/*.md (subagent definitions)
    subagent_paths = generate_cursor_subagents(cursor_dir, force=force_regenerate)
    for path in subagent_paths:
        print(f"  Created {path}")

    # Generate .cursor/skills/*/SKILL.md (skill definitions)
    skill_paths = generate_cursor_skills(cursor_dir, force=force_regenerate)
    for path in skill_paths:
        print(f"  Created {path}")

    # Generate .cursor/environment.json (cloud agent environment)
    env_path = generate_cursor_environment(cursor_dir, detected)
    if env_path:
        print(f"  Created {env_path}")

    # Generate .cursor/sandbox.json (secure defaults)
    sandbox_path = generate_cursor_sandbox(cursor_dir, detected)
    if sandbox_path:
        print(f"  Created {sandbox_path}")

    # Generate .cursor/mcp.json
    mcp_path = generate_cursor_mcp_json(cursor_dir)
    print(f"  Created {mcp_path}")


def _setup_claude_code(
    directory: Path,
    detected: DetectedProject,
    languages: list[str] | None = None,
) -> None:
    """Generate Claude Code configuration files, skills, agents, and MCP config.

    Args:
        directory: Project root directory.
        detected: Auto-detected project metadata.
        languages: Optional list of languages for multi-language rule generation.
    """
    from mirdan.integrations.claude_code import (
        generate_agents,
        generate_claude_code_config,
        generate_mcp_json,
        generate_skills,
    )
    from mirdan.integrations.self_managing import SelfManagingIntegration

    # MCP server registration
    mcp_path = generate_mcp_json(directory)
    print(f"  Created {mcp_path}")

    # Hooks + rules
    generated = generate_claude_code_config(directory, detected, languages=languages)
    for path in generated:
        print(f"  Created {path}")

    # Self-managing workflow rule
    self_managing = SelfManagingIntegration()
    workflow_path = self_managing.write_workflow_rule(directory, detected, languages=languages)
    print(f"  Created {workflow_path}")

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
    print("  Workflow rule installed — no CLAUDE.md mirdan instructions needed.")
    if skill_paths:
        print("  Skills installed: /code, /debug, /review, /plan, /quality, /scan, /gate")


def _setup_agents_md(
    directory: Path,
    detected: DetectedProject,
    platform: str = "generic",
    languages: list[str] | None = None,
) -> None:
    """Generate root AGENTS.md (cross-platform standard).

    Args:
        directory: Project root directory.
        detected: Auto-detected project metadata.
        platform: Platform profile name.
        languages: Optional list of languages for multi-language sections.
    """
    from mirdan.core.quality_standards import QualityStandards
    from mirdan.integrations.agents_md import generate_root_agents_md

    try:
        standards = QualityStandards()
    except Exception:
        standards = None

    agents_path = generate_root_agents_md(
        directory,
        detected,
        standards=standards,
        platform=platform,
        languages=languages,
    )
    print(f"  Created {agents_path}")


def _run_upgrade(directory: Path) -> None:
    """Upgrade an existing mirdan installation.

    Detects existing config version, merges new fields, and regenerates
    integration files (hooks, rules, AGENTS.md, workflow).

    If a workspace is detected but the existing config is not workspace-aware,
    the config is upgraded to workspace mode.

    Args:
        directory: Project root directory.
    """
    print(f"Upgrading mirdan in {directory.resolve()}")
    print()

    config_path = directory / ".mirdan" / "config.yaml"
    if not config_path.exists():
        print("No existing mirdan config found. Run 'mirdan init' first.")
        sys.exit(1)

    # Load existing config (auto-merges with defaults for new fields)
    config = MirdanConfig.load(config_path)

    # Detect workspace
    workspace = detect_workspace(directory)
    is_workspace = workspace is not None

    if is_workspace and workspace is not None:
        _print_workspace_detection(workspace)
        detected = _workspace_as_detected(workspace)
        languages: list[str] | None = workspace.all_languages

        # Upgrade to workspace config if not already workspace-aware
        if not config.is_workspace:
            print("  Upgrading config to workspace mode...")
            platform_name = config.platform.name
            config = _build_workspace_config(workspace, platform_name)
    else:
        detected = detect_project(directory)
        languages = None

    _print_detection(detected)

    # Re-save config (adds new fields with defaults)
    _write_config(config, config_path, platform="generic")

    # Determine platform
    platform = _determine_platform(detected, False, False)

    # Regenerate integration files
    if platform == "claude-code" or "claude-code" in detected.detected_ides:
        _setup_claude_code(directory, detected, languages=languages)

    if platform == "cursor" or "cursor" in detected.detected_ides:
        _setup_cursor(directory, detected, languages=languages, force_regenerate=True)

    # Always regenerate AGENTS.md
    _setup_agents_md(directory, detected, platform, languages=languages)

    print()
    print("Upgrade complete!")
    print("  - Config updated with new defaults")
    print("  - Integration files regenerated")
    print("  - AGENTS.md updated")
    if is_workspace and workspace is not None:
        print(f"  - Workspace mode: {len(workspace.projects)} sub-projects")


def _setup_precommit(directory: Path, detected: DetectedProject) -> None:
    """Generate pre-commit configuration."""
    from mirdan.integrations.github_ci import generate_precommit_config

    precommit_path = generate_precommit_config(directory)
    if precommit_path:
        print(f"  Created {precommit_path}")


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
    print("  --hooks                  Install hook scripts for auto-validation")
    print("  --cursor                 Use Cursor platform profile")
    print("  --claude-code            Use Claude Code platform profile")
    print("  --all                    Set up both Cursor and Claude Code")
    print("  --quality-profile NAME   Set quality profile (e.g. enterprise, startup)")
    print("  --learn                  Scan codebase and generate custom rules from conventions")
    print("  --upgrade                Upgrade existing config (merge new fields, regenerate)")
    print("  -h, --help               Show this help")
