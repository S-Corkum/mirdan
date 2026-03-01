"""Mirdan CLI - entry point with subcommand routing.

Routes:
- ``mirdan`` (bare) → starts MCP server (backward compatible)
- ``mirdan serve`` → starts MCP server (explicit)
- ``mirdan init`` → project initialization wizard
"""

from __future__ import annotations

import sys


def main() -> None:
    """Route CLI subcommands or default to serve."""
    args = sys.argv[1:]

    if not args or args[0] == "serve":
        _serve()
    elif args[0] == "init":
        _init(args[1:])
    elif args[0] in ("--help", "-h"):
        _print_help()
    elif args[0] in ("--version", "-V"):
        _print_version()
    else:
        print(f"Unknown command: {args[0]}")
        _print_help()
        sys.exit(1)


def _serve() -> None:
    """Start the MCP server."""
    from mirdan.server import main as server_main

    server_main()


def _init(args: list[str]) -> None:
    """Run the project initialization wizard."""
    from mirdan.cli.init_command import run_init

    run_init(args)


def _print_help() -> None:
    """Print CLI usage help."""
    from mirdan import __version__

    print(f"mirdan v{__version__} - AI Code Quality Orchestrator")
    print()
    print("Usage: mirdan [command]")
    print()
    print("Commands:")
    print("  (none)     Start the MCP server (default)")
    print("  serve      Start the MCP server")
    print("  init       Initialize mirdan for the current project")
    print()
    print("Options:")
    print("  -h, --help     Show this help")
    print("  -V, --version  Show version")


def _print_version() -> None:
    """Print version."""
    from mirdan import __version__

    print(f"mirdan {__version__}")
