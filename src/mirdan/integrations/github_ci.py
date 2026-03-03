"""Generate GitHub CI/CD integration files for mirdan."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def generate_github_action(project_dir: Path) -> Path | None:
    """Generate .github/workflows/mirdan.yml from template.

    Args:
        project_dir: The project root directory.

    Returns:
        Path to the generated workflow file, or None if template not found.
    """
    workflows_dir = project_dir / ".github" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)

    dest = workflows_dir / "mirdan.yml"
    if dest.exists():
        return None  # Don't overwrite existing workflows

    content = _load_action_template()
    if content is None:
        return None

    dest.write_text(content)
    return dest


def generate_precommit_config(project_dir: Path) -> Path | None:
    """Generate .pre-commit-config.yaml from template.

    Args:
        project_dir: The project root directory.

    Returns:
        Path to the generated config, or None if already exists.
    """
    dest = project_dir / ".pre-commit-config.yaml"
    if dest.exists():
        return None  # Don't overwrite existing config

    content = _load_precommit_template()
    if content is None:
        return None

    dest.write_text(content)
    return dest


def _load_action_template() -> str | None:
    """Load the GitHub Action template."""
    try:
        templates_pkg = files("mirdan.integrations.templates")
        template = templates_pkg / "github-action.yml"
        return template.read_text()
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        return None


def _load_precommit_template() -> str | None:
    """Load the pre-commit config template."""
    try:
        templates_pkg = files("mirdan.integrations.templates")
        template = templates_pkg / "pre-commit-config.yml"
        return template.read_text()
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        return None
