"""Generate Cursor IDE .mdc rule files for mirdan integration."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from mirdan.cli.detect import DetectedProject


def generate_cursor_rules(
    rules_dir: Path,
    detected: DetectedProject,
) -> list[Path]:
    """Generate .cursor/rules/*.mdc files from templates.

    Args:
        rules_dir: The .cursor/rules/ directory to write into.
        detected: Detected project metadata.

    Returns:
        List of generated file paths.
    """
    generated: list[Path] = []
    templates = _load_templates()

    # Always generate the always-on rule
    if "mirdan-always.mdc" in templates:
        path = rules_dir / "mirdan-always.mdc"
        path.write_text(templates["mirdan-always.mdc"])
        generated.append(path)

    # Language-specific rules
    lang = detected.primary_language
    if lang == "python" and "mirdan-python.mdc" in templates:
        path = rules_dir / "mirdan-python.mdc"
        path.write_text(templates["mirdan-python.mdc"])
        generated.append(path)

    if lang in ("typescript", "javascript") and "mirdan-typescript.mdc" in templates:
        path = rules_dir / "mirdan-typescript.mdc"
        path.write_text(templates["mirdan-typescript.mdc"])
        generated.append(path)

    # Security rule (always)
    if "mirdan-security.mdc" in templates:
        path = rules_dir / "mirdan-security.mdc"
        path.write_text(templates["mirdan-security.mdc"])
        generated.append(path)

    # Planning rule (always)
    if "mirdan-planning.mdc" in templates:
        path = rules_dir / "mirdan-planning.mdc"
        path.write_text(templates["mirdan-planning.mdc"])
        generated.append(path)

    return generated


def _load_templates() -> dict[str, str]:
    """Load .mdc templates from the package templates directory."""
    templates: dict[str, str] = {}
    try:
        templates_pkg = files("mirdan.integrations.templates")
        for item in templates_pkg.iterdir():
            if item.name.endswith(".mdc"):
                templates[item.name] = item.read_text()
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass
    return templates
