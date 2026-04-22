"""Generate Claude Code configuration files for mirdan integration."""

from __future__ import annotations

import json
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

from mirdan.cli.detect import DetectedProject


def generate_claude_code_config(
    project_dir: Path,
    detected: DetectedProject,
    languages: list[str] | None = None,
    upgrade: bool = False,
) -> list[Path]:
    """Generate Claude Code configuration files.

    Creates:
    - ``.claude/settings.json`` — hook definitions under the ``"hooks"`` key
      (Claude Code only reads hooks from settings files; a standalone
      ``.claude/hooks.json`` is never loaded).
    - ``.claude/hooks/validate-file.sh`` — stdin-reading helper for PostToolUse
    - ``.claude/rules/*.md`` — mirdan quality rule files

    When ``upgrade=False`` (default), an existing ``hooks`` key in
    ``settings.json`` is preserved. When ``upgrade=True``, the existing
    ``settings.json`` is backed up to ``settings.json.bak`` and the ``hooks``
    key is regenerated — this is how ``mirdan init --upgrade`` picks up new
    hook features (e.g., sidecar triage). Other keys in ``settings.json``
    (permissions, mcpServers, etc.) are always preserved.

    Args:
        project_dir: The project root directory.
        detected: Detected project metadata.
        languages: Optional list of languages to generate rules for (workspace mode).
        upgrade: If True, regenerate the hooks block with backup of any
            existing ``settings.json``.

    Returns:
        List of generated/updated file paths.
    """
    generated: list[Path] = []

    hooks_path = _generate_hooks(project_dir, upgrade=upgrade)
    if hooks_path:
        generated.append(hooks_path)

    # Generate rule files (always overwrite — these are generated config)
    rule_paths = _generate_rules(project_dir, detected, languages=languages)
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

    for skill_name in ("code", "debug", "review", "plan", "plan-review", "quality", "scan", "gate"):
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
        "plan-reviewer",
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
        "categories": ["code-quality", "security", "ai-safety"],
        "skills": ["code", "debug", "review", "plan", "plan-review", "quality", "scan", "gate"],
        "agents": [
            "quality-gate",
            "security-audit",
            "test-quality",
            "convention-check",
            "architecture-reviewer",
            "plan-reviewer",
        ],
        "hooks": True,
        "mcpServers": ["mirdan"],
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
    for skill_name in ("code", "debug", "review", "plan", "plan-review", "quality", "scan", "gate"):
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
        "plan-reviewer",
    ):
        try:
            template_file = templates_pkg / "agents" / f"{agent_name}.md"
            content = template_file.read_text()
            (agents_dir / f"{agent_name}.md").write_text(content)
        except (FileNotFoundError, TypeError, AttributeError):
            continue

    # Copy rules
    rules_dir = output_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    for rule_name, content in _load_rule_templates().items():
        (rules_dir / rule_name).write_text(content)

    # Generate hooks.json
    from mirdan.integrations.hook_templates import HookTemplateGenerator

    hook_gen = HookTemplateGenerator()
    hooks_data = hook_gen.generate()
    with (output_dir / "hooks.json").open("w") as f:
        json.dump(hooks_data, f, indent=2)

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


def _generate_hooks(project_dir: Path, upgrade: bool = False) -> Path | None:
    """Merge Claude Code hook definitions into ``.claude/settings.json``.

    Claude Code loads hooks only from its settings files (global, project,
    or local ``settings.json``) — a standalone ``.claude/hooks.json`` is
    never read. This function uses ``HookTemplateGenerator`` to build the
    hook config, then merges it into ``.claude/settings.json`` under the
    ``"hooks"`` key while preserving any other existing keys.

    Reads the project's mirdan config to detect whether LLM features are
    enabled; when they are, the generator produces LLM-aware hooks
    (sidecar-curl on UserPromptSubmit, ``mirdan check --smart`` on Stop).

    Also emits ``.claude/hooks/validate-file.sh`` — a stdin-reading helper
    referenced by the PostToolUse hook. Claude Code does not perform shell
    variable substitution in hook commands, so the file_path must be
    extracted from the stdin JSON payload by a script.

    If a legacy ``.claude/hooks.json`` exists (from mirdan < 2.0.5), it is
    renamed to ``hooks.json.deprecated`` so operators can see the old
    content but Claude Code will ignore it (as it always did).

    Args:
        project_dir: The project root directory.
        upgrade: If True and ``settings.json`` already has a ``"hooks"``
            key, back up the whole file to ``settings.json.bak`` before
            regenerating. Default False preserves user customizations.

    Returns:
        Path to ``settings.json`` if written, None if its ``hooks`` key was
        preserved.
    """
    from mirdan.config import MirdanConfig
    from mirdan.integrations.hook_templates import (
        HookStringency,
        HookTemplateGenerator,
    )

    claude_dir = project_dir / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    settings_path = claude_dir / "settings.json"

    # Detect LLM enablement from the project's config so hooks match runtime
    cfg, _ = MirdanConfig.find_config_with_path(project_dir)
    llm_enabled = cfg.llm.enabled

    command, _args = detect_mirdan_command()
    mirdan_cmd = command if not _args else f"{command} {' '.join(_args)}"

    # Pass the absolute script path so PostToolUse doesn't break when the
    # shell cwd moves into a subdir (e.g. a monorepo submodule).
    hook_script_path = str((claude_dir / "hooks" / "validate-file.sh").resolve())
    generator = HookTemplateGenerator(
        mirdan_command=mirdan_cmd,
        hook_script_path=hook_script_path,
    )
    generated = generator.generate_claude_code_hooks(
        stringency=HookStringency.COMPREHENSIVE,
        llm_enabled=llm_enabled,
    )
    new_hooks = generated["hooks"]

    # Load existing settings.json; preserve any non-hook keys
    settings = _load_settings_json(settings_path)

    existing_hooks = settings.get("hooks")
    wrote_settings = False
    if existing_hooks is None or upgrade:
        if existing_hooks is not None and upgrade:
            backup = settings_path.with_suffix(".json.bak")
            backup.write_text(json.dumps(settings, indent=2))
        settings["hooks"] = new_hooks
        settings_path.write_text(json.dumps(settings, indent=2))
        wrote_settings = True

    # Migrate legacy hooks.json regardless — it was never read by Claude Code
    _migrate_legacy_hooks_json(claude_dir)

    # Emit the stdin-reading PostToolUse helper script — referenced by hooks
    _write_validate_file_script(claude_dir, mirdan_cmd)
    return settings_path if wrote_settings else None


def _load_settings_json(settings_path: Path) -> dict[str, Any]:
    """Load ``settings.json`` as a dict, tolerating missing or invalid files.

    If the file is present but unparseable, it is renamed to
    ``settings.json.corrupt.bak`` so we can write a fresh one without
    losing the prior contents.
    """
    if not settings_path.exists():
        return {}
    try:
        data = json.loads(settings_path.read_text())
    except json.JSONDecodeError:
        settings_path.replace(settings_path.with_suffix(".json.corrupt.bak"))
        return {}
    return data if isinstance(data, dict) else {}


def _migrate_legacy_hooks_json(claude_dir: Path) -> None:
    """Rename a legacy ``.claude/hooks.json`` to ``hooks.json.deprecated``.

    Prior mirdan versions (<=2.0.4) wrote hook definitions to
    ``.claude/hooks.json``, a path Claude Code never loads. Rename the
    file so operators understand it is defunct without silently deleting
    their prior content.
    """
    legacy = claude_dir / "hooks.json"
    if not legacy.exists():
        return
    deprecated = claude_dir / "hooks.json.deprecated"
    if deprecated.exists():
        return
    legacy.rename(deprecated)


def _write_validate_file_script(claude_dir: Path, mirdan_cmd: str) -> None:
    """Write .claude/hooks/validate-file.sh — stdin-reading validator helper.

    Claude Code sends hook input as JSON on stdin (not as substituted shell
    variables). This script extracts ``tool_input.file_path`` from that JSON
    and runs ``mirdan validate --quick`` on the file.
    """
    hooks_dir = claude_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    script_path = hooks_dir / "validate-file.sh"
    extract_path = (
        "import json,sys; "
        "d=json.load(sys.stdin); "
        "print(d.get('tool_input',{}).get('file_path',''))"
    )
    script_path.write_text(
        f"""#!/bin/bash
# Claude Code PostToolUse hook: validate the edited file via mirdan.
# Reads Claude Code's hook JSON from stdin and extracts the file path.
set -e
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "{extract_path}")
if [ -z "$FILE_PATH" ]; then
  exit 0
fi
[ -f "$FILE_PATH" ] || exit 0
{mirdan_cmd} validate --quick --scope essential --file "$FILE_PATH" --format micro 2>&1
"""
    )
    script_path.chmod(0o755)


def _generate_rules(
    project_dir: Path,
    detected: DetectedProject,
    languages: list[str] | None = None,
) -> list[Path]:
    """Generate .claude/rules/*.md files from templates.

    Args:
        project_dir: The project root directory.
        detected: Detected project metadata.
        languages: Optional list of languages to generate rules for (workspace mode).

    Returns:
        List of generated rule file paths.
    """
    rules_dir = project_dir / ".claude" / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    templates = _load_rule_templates()

    # Always generate quality, security, and workflow rules
    for template_name in (
        "mirdan-quality.md",
        "mirdan-security.md",
        "mirdan-workflow.md",
    ):
        if template_name in templates:
            path = rules_dir / template_name
            path.write_text(templates[template_name])
            generated.append(path)

    # Language-specific rules — iterate all languages in workspace mode
    langs = languages or ([detected.primary_language] if detected.primary_language else [])
    for lang in langs:
        if lang == "python" and "mirdan-python.md" in templates:
            path = rules_dir / "mirdan-python.md"
            if path not in generated:
                path.write_text(templates["mirdan-python.md"])
                generated.append(path)

        if lang in ("typescript", "javascript") and "mirdan-typescript.md" in templates:
            path = rules_dir / "mirdan-typescript.md"
            if path not in generated:
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
    from mirdan import __version__

    return __version__


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
