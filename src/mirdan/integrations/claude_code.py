"""Generate Claude Code configuration files for mirdan integration."""

from __future__ import annotations

import json
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

from mirdan.cli.detect import DetectedProject


def generate_claude_code_config(
    project_dir: Path,
    detected: DetectedProject,
) -> list[Path]:
    """Generate Claude Code configuration files.

    Creates:
    - ``.claude/hooks.json`` — PreToolUse, PostToolUse, Stop hooks
    - ``.claude/rules/*.md`` — mirdan quality rule files

    Hooks.json is only created if it doesn't already exist (respects user
    customizations). Rule files are always overwritten as generated config.

    Args:
        project_dir: The project root directory.
        detected: Detected project metadata.

    Returns:
        List of generated/updated file paths.
    """
    generated: list[Path] = []

    # Generate hooks.json (only if not exists — respect user customizations)
    hooks_path = _generate_hooks(project_dir)
    if hooks_path:
        generated.append(hooks_path)

    # Generate rule files (always overwrite — these are generated config)
    rule_paths = _generate_rules(project_dir, detected)
    generated.extend(rule_paths)

    return generated


def generate_skills(project_dir: Path, detected: DetectedProject) -> list[Path]:
    """Generate .claude/skills/ SKILL.md files from templates.

    Creates skill directories with SKILL.md files for /code, /debug, /review.

    Args:
        project_dir: The project root directory.
        detected: Detected project metadata.

    Returns:
        List of generated skill file paths.
    """
    generated: list[Path] = []
    templates_pkg = _get_templates_package()
    if templates_pkg is None:
        return generated

    for skill_name in ("code", "debug", "review", "plan", "quality"):
        skill_dir = project_dir / ".claude" / "skills" / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        try:
            template_dir = templates_pkg / "skills" / skill_name
            template_file = template_dir / "SKILL.md"
            content = template_file.read_text()
            dest = skill_dir / "SKILL.md"
            dest.write_text(content)
            generated.append(dest)
        except (FileNotFoundError, TypeError, AttributeError):
            continue

    return generated


def generate_agents(project_dir: Path, detected: DetectedProject) -> list[Path]:
    """Generate .claude/agents/ markdown files from templates.

    Creates specialized agent definitions:
    - quality-gate: Full quality validation
    - security-audit: Security-focused audit
    - test-quality: Test code quality validation
    - convention-check: Project convention compliance

    Args:
        project_dir: The project root directory.
        detected: Detected project metadata.

    Returns:
        List of generated agent file paths.
    """
    generated: list[Path] = []
    templates_pkg = _get_templates_package()
    if templates_pkg is None:
        return generated

    agents_dir = project_dir / ".claude" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    agent_names = (
        "quality-gate",
        "security-audit",
        "test-quality",
        "convention-check",
        "architecture-reviewer",
    )
    for agent_name in agent_names:
        try:
            template_file = templates_pkg / "agents" / f"{agent_name}.md"
            content = template_file.read_text()
            dest = agents_dir / f"{agent_name}.md"
            dest.write_text(content)
            generated.append(dest)
        except (FileNotFoundError, TypeError, AttributeError):
            continue

    return generated


def generate_rules(project_dir: Path, detected: DetectedProject) -> list[Path]:
    """Generate .claude/rules/ enforcement files.

    Public wrapper around _generate_rules for external use.

    Args:
        project_dir: The project root directory.
        detected: Detected project metadata.

    Returns:
        List of generated rule file paths.
    """
    return _generate_rules(project_dir, detected)


def generate_mcp_json(project_dir: Path) -> Path:
    """Generate .mcp.json with mirdan MCP server configuration.

    Detects the mirdan installation method and writes the appropriate config.

    Args:
        project_dir: The project root directory.

    Returns:
        Path to the generated .mcp.json file.
    """
    mcp_json_path = project_dir / ".mcp.json"

    # Detect installation method
    command, args = detect_mirdan_command()

    config = {
        "mcpServers": {
            "mirdan": {
                "type": "stdio",
                "command": command,
                "args": args,
            }
        }
    }

    # If .mcp.json already exists, merge rather than overwrite
    if mcp_json_path.exists():
        try:
            existing = json.loads(mcp_json_path.read_text())
            if "mcpServers" in existing:
                existing["mcpServers"]["mirdan"] = config["mcpServers"]["mirdan"]
                config = existing
        except (json.JSONDecodeError, KeyError):
            pass

    with mcp_json_path.open("w") as f:
        json.dump(config, f, indent=2)

    return mcp_json_path


def export_plugin(output_dir: Path) -> Path:
    """Export mirdan as a Claude Code plugin.

    Creates the complete plugin directory structure with manifest, MCP config,
    skills, agents, and README.

    Args:
        output_dir: Directory to export the plugin to.

    Returns:
        Path to the exported plugin directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    templates_pkg = _get_templates_package()
    if templates_pkg is None:
        msg = "Could not load mirdan templates package"
        raise RuntimeError(msg)

    # Copy plugin manifest
    plugin_dir = output_dir / ".claude-plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)

    plugin_json = {
        "name": "mirdan",
        "description": (
            "AI Code Quality Runtime — automatic quality enforcement"
            " with AI-specific slop detection"
        ),
        "version": _get_version(),
        "author": "mirdan",
    }
    with (plugin_dir / "plugin.json").open("w") as f:
        json.dump(plugin_json, f, indent=2)

    # Copy MCP config
    command, args = detect_mirdan_command()
    mcp_config = {
        "mcpServers": {
            "mirdan": {
                "type": "stdio",
                "command": command,
                "args": args,
            }
        }
    }
    with (output_dir / ".mcp.json").open("w") as f:
        json.dump(mcp_config, f, indent=2)

    # Copy skills
    for skill_name in ("code", "debug", "review", "plan", "quality"):
        skill_dir = output_dir / "skills" / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        try:
            template_file = templates_pkg / "skills" / skill_name / "SKILL.md"
            content = template_file.read_text()
            (skill_dir / "SKILL.md").write_text(content)
        except (FileNotFoundError, TypeError, AttributeError):
            continue

    # Copy agents
    agents_dir = output_dir / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    for agent_name in (
        "quality-gate",
        "security-audit",
        "test-quality",
        "convention-check",
        "architecture-reviewer",
    ):
        try:
            template_file = templates_pkg / "agents" / f"{agent_name}.md"
            content = template_file.read_text()
            (agents_dir / f"{agent_name}.md").write_text(content)
        except (FileNotFoundError, TypeError, AttributeError):
            continue

    # Write README
    try:
        readme_pkg = files("mirdan").joinpath("..") / "plugin" / "README.md"
        readme_content = readme_pkg.read_text()
    except (FileNotFoundError, TypeError, AttributeError):
        readme_content = (
            "# mirdan Claude Code Plugin\n\n"
            "AI Code Quality Runtime for Claude Code.\n\n"
            "## Installation\n\n"
            "```bash\n"
            "claude --plugin-dir ./mirdan-plugin\n"
            "```\n"
        )
    (output_dir / "README.md").write_text(readme_content)

    return output_dir


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_hooks(project_dir: Path) -> Path | None:
    """Generate .claude/hooks.json using HookTemplateGenerator.

    Delegates to the dedicated hook template generator for full
    lifecycle coverage. Only creates hooks.json if it doesn't exist
    (respects user customizations).

    Returns:
        Path to hooks.json if created, None if already exists.
    """
    claude_dir = project_dir / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)

    hooks_path = claude_dir / "hooks.json"
    if hooks_path.exists():
        return None

    from mirdan.integrations.hook_templates import (
        HookStringency,
        HookTemplateGenerator,
    )

    command, _args = detect_mirdan_command()
    mirdan_cmd = command if not _args else f"{command} {' '.join(_args)}"

    generator = HookTemplateGenerator(mirdan_command=mirdan_cmd)
    config = generator.generate_claude_code_hooks(
        stringency=HookStringency.COMPREHENSIVE,
    )

    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    with hooks_path.open("w") as f:
        json.dump(config, f, indent=2)
    return hooks_path


def _generate_rules(project_dir: Path, detected: DetectedProject) -> list[Path]:
    """Generate .claude/rules/*.md files from templates.

    Args:
        project_dir: The project root directory.
        detected: Detected project metadata.

    Returns:
        List of generated rule file paths.
    """
    rules_dir = project_dir / ".claude" / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    templates = _load_rule_templates()

    # Always generate quality and security rules
    for template_name in ("mirdan-quality.md", "mirdan-security.md"):
        if template_name in templates:
            path = rules_dir / template_name
            path.write_text(templates[template_name])
            generated.append(path)

    # Language-specific rules
    lang = detected.primary_language
    if lang == "python" and "mirdan-python.md" in templates:
        path = rules_dir / "mirdan-python.md"
        path.write_text(templates["mirdan-python.md"])
        generated.append(path)

    if lang in ("typescript", "javascript") and "mirdan-typescript.md" in templates:
        path = rules_dir / "mirdan-typescript.md"
        path.write_text(templates["mirdan-typescript.md"])
        generated.append(path)

    return generated


def _load_rule_templates() -> dict[str, str]:
    """Load .md rule templates from the Claude Code templates directory."""
    templates: dict[str, str] = {}
    templates_pkg = _get_templates_package()
    if templates_pkg is None:
        return templates

    try:
        for item in templates_pkg.iterdir():
            if item.name.endswith(".md"):
                templates[item.name] = item.read_text()
    except (TypeError, AttributeError):
        pass

    return templates


def _get_templates_package() -> Traversable | None:
    """Get the templates package traversable."""
    try:
        return files("mirdan.integrations.templates.claude_code")
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        return None


def detect_mirdan_command() -> tuple[str, list[str]]:
    """Detect how mirdan is installed and return (command, args)."""
    import shutil as _shutil

    # Check if uvx is available (preferred)
    if _shutil.which("uvx"):
        return "uvx", ["mirdan"]

    # Check if mirdan is directly on PATH
    if _shutil.which("mirdan"):
        return "mirdan", []

    # Fallback to python -m
    return "python", ["-m", "mirdan"]


def _get_version() -> str:
    """Get the mirdan version string."""
    try:
        from mirdan import __version__
        return __version__
    except ImportError:
        return "0.1.0"


# ---------------------------------------------------------------------------
# Platform Adapter
# ---------------------------------------------------------------------------


class ClaudeCodeAdapter:
    """Platform adapter for Claude Code integration.

    Delegates to existing public functions, providing an alternative
    entry point via the PlatformAdapter interface.
    """

    def __init__(
        self,
        project_dir: Path,
        detected: DetectedProject,
        standards: object | None = None,
    ) -> None:
        self.project_dir = project_dir
        self.detected = detected
        self.standards = standards

    def generate_hooks(self) -> list[Path]:
        """Generate Claude Code hooks and rules."""
        return generate_claude_code_config(self.project_dir, self.detected)

    def generate_rules(self) -> list[Path]:
        """Generate .claude/rules/ enforcement files."""
        return generate_rules(self.project_dir, self.detected)

    def generate_agents(self) -> list[Path]:
        """Generate .claude/agents/ markdown files."""
        return generate_agents(self.project_dir, self.detected)

    def generate_skills(self) -> list[Path]:
        """Generate .claude/skills/ SKILL.md files."""
        return generate_skills(self.project_dir, self.detected)

    def generate_mcp_config(self) -> Path | None:
        """Generate .mcp.json with mirdan MCP server configuration."""
        return generate_mcp_json(self.project_dir)

    def generate_all(self) -> list[Path]:
        """Call all generators, return all created paths."""
        paths: list[Path] = []
        paths.extend(self.generate_hooks())
        paths.extend(self.generate_rules())
        paths.extend(self.generate_agents())
        paths.extend(self.generate_skills())
        mcp = self.generate_mcp_config()
        if mcp:
            paths.append(mcp)
        return paths
