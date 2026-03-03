"""``mirdan plugin export`` — export mirdan as a Claude Code plugin."""

from __future__ import annotations

import sys
from pathlib import Path


def run_plugin(args: list[str]) -> None:
    """Route plugin subcommands.

    Args:
        args: CLI arguments after ``plugin``.
    """
    if not args or args[0] in ("--help", "-h"):
        _print_help()
        return

    if args[0] == "export":
        _export(args[1:])
    else:
        print(f"Unknown plugin subcommand: {args[0]}")
        _print_help()
        sys.exit(1)


def _export(args: list[str]) -> None:
    """Export mirdan as a Claude Code plugin."""
    output_dir = Path("./mirdan-plugin")

    for i, arg in enumerate(args):
        if arg == "--output-dir" and i + 1 < len(args):
            output_dir = Path(args[i + 1])
        elif arg in ("--help", "-h"):
            _print_export_help()
            return

    from mirdan.integrations.claude_code import export_plugin

    print(f"Exporting mirdan plugin to {output_dir.resolve()}")
    result = export_plugin(output_dir)
    print(f"Plugin exported to {result}")
    print()
    print(f"Install with: claude --plugin-dir {result}")


def _print_help() -> None:
    """Print plugin command help."""
    print("Usage: mirdan plugin <subcommand>")
    print()
    print("Subcommands:")
    print("  export     Export mirdan as a Claude Code plugin")
    print()
    print("Options:")
    print("  -h, --help  Show this help")


def _print_export_help() -> None:
    """Print plugin export help."""
    print("Usage: mirdan plugin export [options]")
    print()
    print("Options:")
    print("  --output-dir PATH  Output directory (default: ./mirdan-plugin)")
    print("  -h, --help         Show this help")
