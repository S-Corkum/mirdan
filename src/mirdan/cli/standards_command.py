"""``mirdan standards`` — quality standards lookup CLI command."""

from __future__ import annotations

import json
import sys

import yaml

from mirdan.config import MirdanConfig
from mirdan.core.quality_standards import QualityStandards


def run_standards(args: list[str]) -> None:
    """Display quality standards for a language/framework.

    Args:
        args: CLI arguments after ``standards``.
    """
    parsed = _parse_args(args)

    if parsed.get("error"):
        print(f"Error: {parsed['error']}", file=sys.stderr)
        _print_standards_help()
        sys.exit(2)

    language = parsed.get("language")
    if not language:
        print("Error: --language is required", file=sys.stderr)
        _print_standards_help()
        sys.exit(2)
        return

    config = MirdanConfig.find_config()
    standards = QualityStandards(config=config.quality)

    result = standards.get_all_standards(
        language=language,
        framework=parsed.get("framework", ""),
        category=parsed.get("category", "all"),
    )

    output_format = parsed.get("format", "yaml")
    if output_format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(yaml.dump(result, default_flow_style=False, sort_keys=False))


def _parse_args(args: list[str]) -> dict:
    """Parse standards subcommand arguments."""
    parsed: dict = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--language" and i + 1 < len(args):
            parsed["language"] = args[i + 1]
            i += 2
        elif arg == "--framework" and i + 1 < len(args):
            parsed["framework"] = args[i + 1]
            i += 2
        elif arg == "--category" and i + 1 < len(args):
            parsed["category"] = args[i + 1]
            i += 2
        elif arg == "--format" and i + 1 < len(args):
            fmt = args[i + 1]
            if fmt not in ("json", "yaml"):
                parsed["error"] = f"invalid format: {fmt}"
                return parsed
            parsed["format"] = fmt
            i += 2
        elif arg in ("--help", "-h"):
            _print_standards_help()
            sys.exit(0)
        else:
            parsed["error"] = f"unknown argument: {arg}"
            return parsed
    return parsed


def _print_standards_help() -> None:
    """Print usage help for the standards command."""
    print("Usage: mirdan standards [options]")
    print()
    print("Options:")
    print("  --language LANG      Language (required: python, typescript, etc.)")
    print("  --framework FRAME    Framework filter (fastapi, react, etc.)")
    print("  --category CAT       Category filter (security|architecture|style|all)")
    print("  --format FORMAT      Output format (yaml|json)")
    print("  -h, --help           Show this help")
