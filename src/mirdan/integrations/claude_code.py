"""Generate Claude Code configuration files for mirdan integration."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from mirdan.cli.detect import DetectedProject


def generate_claude_code_config(
    project_dir: Path,
    detected: DetectedProject,
) -> list[Path]:
    """Generate Claude Code configuration files.

    Creates:
    - ``.claude/hooks.json`` — PostToolUse (quick validate) + PreCommit (full validate)
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


def _generate_hooks(project_dir: Path) -> Path | None:
    """Generate .claude/hooks.json if not present.

    Returns:
        Path to hooks.json if created, None if already exists.
    """
    claude_dir = project_dir / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)

    hooks_path = claude_dir / "hooks.json"
    if hooks_path.exists():
        return None

    hooks_config = {
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "Write|Edit",
                    "hooks": [
                        {
                            "type": "command",
                            "command": (
                                "mirdan validate"
                                " --quick"
                                " --file $TOOL_INPUT_FILE_PATH"
                                " --format text"
                            ),
                        }
                    ],
                }
            ],
            "PreCommit": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "mirdan validate --diff --format text",
                        }
                    ],
                }
            ],
        }
    }

    with hooks_path.open("w") as f:
        json.dump(hooks_config, f, indent=2)

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
    templates = _load_templates()

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


def _load_templates() -> dict[str, str]:
    """Load .md templates from the Claude Code templates directory."""
    templates: dict[str, str] = {}
    try:
        templates_pkg = files("mirdan.integrations.templates.claude_code")
        for item in templates_pkg.iterdir():
            if item.name.endswith(".md"):
                templates[item.name] = item.read_text()
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass
    return templates
