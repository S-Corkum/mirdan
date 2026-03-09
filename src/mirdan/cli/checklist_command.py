"""``mirdan checklist`` — verification checklist CLI command."""

from __future__ import annotations

import json
import sys
from typing import Any

from mirdan.config import MirdanConfig
from mirdan.core.prompt_composer import PromptComposer
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import Intent, TaskType


def run_checklist(args: list[str]) -> None:
    """Display a verification checklist for a task type.

    Args:
        args: CLI arguments after ``checklist``.
    """
    parsed = _parse_args(args)

    if parsed.get("error"):
        print(f"Error: {parsed['error']}", file=sys.stderr)
        _print_checklist_help()
        sys.exit(2)

    task_type_str = parsed.get("task_type")
    if not task_type_str:
        print("Error: --task-type is required", file=sys.stderr)
        _print_checklist_help()
        sys.exit(2)

    try:
        task = TaskType(task_type_str)
    except ValueError:
        print(f"Error: unknown task type: {task_type_str}", file=sys.stderr)
        print(f"  Valid types: {', '.join(t.value for t in TaskType)}")
        sys.exit(2)

    touches_security = parsed.get("security", False)

    config = MirdanConfig.find_config()
    standards = QualityStandards(config=config.quality)
    composer = PromptComposer(standards, config=config.enhancement)

    intent = Intent(
        original_prompt="",
        task_type=task,
        touches_security=touches_security,
    )
    steps = composer.generate_verification_steps(intent)

    output_format = parsed.get("format", "text")
    if output_format == "json":
        print(json.dumps({"task_type": task.value, "checklist": steps}, indent=2))
    else:
        print(f"Checklist for {task.value}:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")


def _parse_args(args: list[str]) -> dict[str, Any]:
    """Parse checklist subcommand arguments."""
    parsed: dict[str, Any] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--task-type" and i + 1 < len(args):
            parsed["task_type"] = args[i + 1]
            i += 2
        elif arg == "--security":
            parsed["security"] = True
            i += 1
        elif arg == "--format" and i + 1 < len(args):
            fmt = args[i + 1]
            if fmt not in ("json", "text"):
                parsed["error"] = f"invalid format: {fmt}"
                return parsed
            parsed["format"] = fmt
            i += 2
        elif arg in ("--help", "-h"):
            _print_checklist_help()
            sys.exit(0)
        else:
            parsed["error"] = f"unknown argument: {arg}"
            return parsed
    return parsed


def _print_checklist_help() -> None:
    """Print usage help for the checklist command."""
    print("Usage: mirdan checklist [options]")
    print()
    print("Options:")
    print(
        "  --task-type TYPE     Task type"
        " (required: generation|refactor|debug|review|test|planning)"
    )
    print("  --security           Include security-related checks")
    print("  --format FORMAT      Output format (text|json)")
    print("  -h, --help           Show this help")
