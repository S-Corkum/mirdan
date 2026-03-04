"""``mirdan plugin export`` — export mirdan as a plugin for Claude Code or Cursor."""

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
    """Export mirdan as a plugin."""
    output_dir = Path("./mirdan-plugin")
    target = "claude-code"  # default

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--output-dir" and i + 1 < len(args):
            output_dir = Path(args[i + 1])
            i += 2
        elif arg == "--cursor":
            target = "cursor"
            i += 1
        elif arg == "--all":
            target = "all"
            i += 1
        elif arg in ("--help", "-h"):
            _print_export_help()
            return
        else:
            i += 1

    if target in ("claude-code", "all"):
        cc_dir = output_dir if target == "claude-code" else output_dir / "claude-code"
        _export_claude_code(cc_dir)

    if target in ("cursor", "all"):
        cur_dir = output_dir if target == "cursor" else output_dir / "cursor"
        _export_cursor(cur_dir)


def _export_claude_code(output_dir: Path) -> None:
    from mirdan.integrations.claude_code import export_plugin

    print(f"Exporting Claude Code plugin to {output_dir.resolve()}")
    result = export_plugin(output_dir)
    print(f"  Plugin exported to {result}")


def _export_cursor(output_dir: Path) -> None:
    from mirdan.integrations.cursor_plugin import CursorPluginExporter

    print(f"Exporting Cursor plugin to {output_dir.resolve()}")
    exporter = CursorPluginExporter()
    result = exporter.export(output_dir)
    print(f"  Plugin exported to {result}")


def _print_help() -> None:
    """Print plugin command help."""
    print("Usage: mirdan plugin <subcommand>")
    print()
    print("Subcommands:")
    print("  export     Export mirdan as a plugin")
    print()
    print("Options:")
    print("  -h, --help  Show this help")


def _print_export_help() -> None:
    """Print plugin export help."""
    print("Usage: mirdan plugin export [options]")
    print()
    print("Options:")
    print("  --output-dir PATH  Output directory (default: ./mirdan-plugin)")
    print("  --cursor           Export as Cursor plugin (default: Claude Code)")
    print("  --all              Export for both Claude Code and Cursor")
    print("  -h, --help         Show this help")
