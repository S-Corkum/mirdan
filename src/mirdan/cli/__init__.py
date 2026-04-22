"""Mirdan CLI - entry point with subcommand routing.

Routes:
- ``mirdan`` (bare) → starts MCP server (backward compatible)
- ``mirdan serve`` → starts MCP server (explicit)
- ``mirdan init`` → project initialization wizard
- ``mirdan validate`` → code quality validation
- ``mirdan standards`` → quality standards lookup
- ``mirdan checklist`` → verification checklist
- ``mirdan scan`` → codebase convention extraction
- ``mirdan plugin`` → plugin management (export)
- ``mirdan report`` → project quality report
- ``mirdan fix`` → auto-fix code quality violations
- ``mirdan gate`` → quality gate for changed files
- ``mirdan profile`` → manage quality profiles
- ``mirdan export`` → export results (SARIF, badge, JSON)
- ``mirdan triage`` → classify a task via local LLM
- ``mirdan check`` → run lint + typecheck + test checks
"""

from __future__ import annotations

import sys

from mirdan.config import ConfigError


def main() -> None:
    """Route CLI subcommands or default to serve.

    Wraps dispatch in a ``ConfigError`` guard so malformed config files
    or failed Pydantic validation surface as a clean single-line error
    rather than a raw Python traceback.
    """
    args = sys.argv[1:]

    try:
        if not args or args[0] == "serve":
            _serve()
        elif args[0] == "init":
            _init(args[1:])
        elif args[0] == "validate":
            _validate(args[1:])
        elif args[0] == "standards":
            _standards(args[1:])
        elif args[0] == "checklist":
            _checklist(args[1:])
        elif args[0] == "scan":
            _scan(args[1:])
        elif args[0] == "plugin":
            _plugin(args[1:])
        elif args[0] == "report":
            _report(args[1:])
        elif args[0] == "fix":
            _fix(args[1:])
        elif args[0] == "gate":
            _gate(args[1:])
        elif args[0] == "profile":
            _profile(args[1:])
        elif args[0] == "export":
            _export(args[1:])
        elif args[0] == "llm":
            _llm(args[1:])
        elif args[0] == "fine-tune":
            _finetune(args[1:])
        elif args[0] == "triage":
            _triage(args[1:])
        elif args[0] == "check":
            _check(args[1:])
        elif args[0] in ("--help", "-h"):
            _print_help()
        elif args[0] in ("--version", "-V"):
            _print_version()
        else:
            print(f"Unknown command: {args[0]}")
            _print_help()
            sys.exit(1)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _serve() -> None:
    """Start the MCP server."""
    from mirdan.server import main as server_main

    server_main()


def _init(args: list[str]) -> None:
    """Run the project initialization wizard."""
    from mirdan.cli.init_command import run_init

    run_init(args)


def _validate(args: list[str]) -> None:
    """Run code quality validation."""
    from mirdan.cli.validate_command import run_validate

    run_validate(args)


def _standards(args: list[str]) -> None:
    """Display quality standards."""
    from mirdan.cli.standards_command import run_standards

    run_standards(args)


def _checklist(args: list[str]) -> None:
    """Display verification checklist."""
    from mirdan.cli.checklist_command import run_checklist

    run_checklist(args)


def _scan(args: list[str]) -> None:
    """Scan codebase for conventions."""
    from mirdan.cli.scan_command import run_scan

    run_scan(args)


def _plugin(args: list[str]) -> None:
    """Manage mirdan plugin."""
    from mirdan.cli.plugin_command import run_plugin

    run_plugin(args)


def _report(args: list[str]) -> None:
    """Generate project quality report."""
    from mirdan.cli.report_command import run_report

    run_report(args)


def _fix(args: list[str]) -> None:
    """Auto-fix code quality violations."""
    from mirdan.cli.fix_command import run_fix

    run_fix(args)


def _gate(args: list[str]) -> None:
    """Run quality gate on changed files."""
    from mirdan.cli.gate_command import run_gate

    run_gate(args)


def _profile(args: list[str]) -> None:
    """Manage quality profiles."""
    from mirdan.cli.profile_command import run_profile

    run_profile(args)


def _export(args: list[str]) -> None:
    """Export validation results."""
    from mirdan.cli.export_command import run_export

    run_export(args)


def _llm(args: list[str]) -> None:
    """Local LLM management."""
    if not args:
        print("Usage: mirdan llm <setup|status|warmup|metrics>")
        sys.exit(1)

    if args[0] == "setup":
        from mirdan.cli.llm_setup_command import run_llm_setup

        run_llm_setup(args[1:])
    elif args[0] == "status":
        from mirdan.cli.llm_command import run_llm_status

        run_llm_status(args[1:])
    elif args[0] == "warmup":
        from mirdan.cli.llm_command import run_llm_warmup

        run_llm_warmup(args[1:])
    elif args[0] == "metrics":
        from mirdan.cli.llm_command import run_llm_metrics

        run_llm_metrics(args[1:])
    else:
        print(f"Unknown llm subcommand: {args[0]}")
        print("Usage: mirdan llm <setup|status|warmup|metrics>")
        sys.exit(1)


def _finetune(args: list[str]) -> None:
    """Fine-tuning data management."""
    from mirdan.cli.finetune_command import run_finetune

    run_finetune(args)


def _triage(args: list[str]) -> None:
    """Classify a coding task via local LLM."""
    from mirdan.cli.triage_command import run_triage

    run_triage(args)


def _check(args: list[str]) -> None:
    """Run lint, typecheck, and test checks."""
    from mirdan.cli.check_command import run_check

    run_check(args)


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
    print("  validate   Validate code quality")
    print("  standards  Show quality standards for a language")
    print("  checklist  Show verification checklist for a task type")
    print("  scan       Scan codebase to discover conventions")
    print("  plugin     Plugin management (export)")
    print("  report     Generate project quality report")
    print("  fix        Auto-fix code quality violations")
    print("  gate       Quality gate — validate all changed files")
    print("  profile    Manage quality profiles (list, suggest, apply)")
    print("  export     Export results (sarif, badge, json)")
    print("  llm        Local LLM management (setup, status, warmup, metrics)")
    print("  fine-tune  Fine-tuning data management (status, export)")
    print("  triage     Classify a task via local LLM")
    print("  check      Run lint + typecheck + test checks")
    print()
    print("Options:")
    print("  -h, --help     Show this help")
    print("  -V, --version  Show version")


def _print_version() -> None:
    """Print version."""
    from mirdan import __version__

    print(f"mirdan {__version__}")
