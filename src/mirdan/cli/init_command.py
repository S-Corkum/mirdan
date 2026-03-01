"""``mirdan init`` — project initialization wizard."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from mirdan.cli.detect import DetectedProject, detect_project
from mirdan.config import MirdanConfig, ProjectConfig, QualityConfig


def run_init(args: list[str]) -> None:
    """Run the project initialization wizard.

    Args:
        args: CLI arguments after ``init``.
    """
    directory = Path(args[0]) if args else Path.cwd()
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    print(f"Initializing mirdan in {directory.resolve()}")
    print()

    # Step 1: Detect project
    detected = detect_project(directory)
    _print_detection(detected)

    # Step 2: Generate config
    config = _build_config(detected)
    config_dir = directory / ".mirdan"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"

    if config_path.exists():
        print(f"Config already exists at {config_path}")
        response = input("Overwrite? [y/N] ").strip().lower()
        if response not in ("y", "yes"):
            print("Skipping config generation.")
        else:
            _write_config(config, config_path)
    else:
        _write_config(config, config_path)

    # Step 3: Create rules directory with template
    rules_dir = config_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    _create_rules_template(rules_dir, detected)

    # Step 4: IDE integrations
    if "cursor" in detected.detected_ides:
        _setup_cursor(directory, detected)
    if "claude-code" in detected.detected_ides:
        _print_claude_code_guidance()

    print()
    print("Done! mirdan is configured for this project.")
    print()
    print("Next steps:")
    print("  1. Review .mirdan/config.yaml")
    print("  2. Customize rules in .mirdan/rules/")
    print("  3. Start the server: mirdan serve")


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


def _build_config(detected: DetectedProject) -> MirdanConfig:
    """Build a MirdanConfig from detected project info."""
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


def _write_config(config: MirdanConfig, path: Path) -> None:
    """Write config to YAML file."""
    data = config.model_dump(exclude_defaults=False)
    # Simplify: only write project and quality sections
    output = {
        "version": data["version"],
        "project": data["project"],
        "quality": data["quality"],
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


def _print_claude_code_guidance() -> None:
    """Print guidance for Claude Code plugin setup."""
    print()
    print("  Claude Code detected! To install the mirdan plugin:")
    print("    npm install -g mirdan-claude-code")
    print("  Or add to your .claude/settings.json manually.")
