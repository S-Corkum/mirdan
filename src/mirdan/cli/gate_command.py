"""``mirdan gate`` — quality gate for Stop hooks and CI.

Validates all changed files and returns exit code 0 (pass) or 1 (fail).
Designed for use as a Claude Code Stop hook command.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from mirdan.config import MirdanConfig
from mirdan.core.code_validator import CodeValidator
from mirdan.core.quality_standards import QualityStandards


def run_gate(args: list[str]) -> None:
    """Run the quality gate command.

    Gets changed files from git, validates each, and returns
    exit code 0 (pass) or 1 (fail).

    Args:
        args: CLI arguments after ``gate``.
    """
    parsed = _parse_gate_args(args)

    if parsed.get("help"):
        _print_gate_help()
        sys.exit(0)

    output_format = parsed.get("format", "text")

    # Get changed files
    changed_files = _get_changed_files()
    if not changed_files:
        _output_pass(0, 0.0, output_format, [])
        sys.exit(0)

    config = MirdanConfig.find_config()
    standards = QualityStandards(config=config.quality)
    validator = CodeValidator(
        standards, config=config.quality, thresholds=config.thresholds,
    )

    # Validate each file
    total_errors = 0
    total_score = 0.0
    files_with_errors = 0
    file_results: list[dict[str, Any]] = []

    for file_path in changed_files:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            continue

        try:
            code = path.read_text()
            if not code.strip():
                continue
        except (OSError, UnicodeDecodeError):
            continue

        result = validator.validate(
            code=code, language="auto", check_security=True,
        )

        error_count = sum(1 for v in result.violations if v.severity == "error")
        total_errors += error_count
        total_score += result.score
        if error_count > 0:
            files_with_errors += 1

        file_results.append({
            "file": file_path,
            "score": result.score,
            "passed": result.passed,
            "errors": error_count,
        })

    files_validated = len(file_results)
    avg_score = total_score / files_validated if files_validated else 1.0

    if total_errors == 0:
        _output_pass(files_validated, avg_score, output_format, file_results)
        sys.exit(0)
    else:
        _output_fail(
            files_validated, total_errors, files_with_errors,
            avg_score, output_format, file_results,
        )
        sys.exit(1)


def _get_changed_files() -> list[str]:
    """Get list of files changed relative to HEAD.

    Checks both staged and unstaged changes.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]

        # Also include staged files not yet committed
        staged = subprocess.run(
            ["git", "diff", "--name-only", "--staged"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
        )
        staged_files = [f.strip() for f in staged.stdout.strip().split("\n") if f.strip()]

        # Deduplicate while preserving order
        seen: set[str] = set()
        combined: list[str] = []
        for f in files + staged_files:
            if f not in seen:
                seen.add(f)
                combined.append(f)
        return combined
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def _output_pass(
    files: int,
    avg_score: float,
    output_format: str,
    file_results: list[dict[str, Any]],
) -> None:
    """Output pass result."""
    if output_format == "json":
        print(json.dumps({
            "status": "PASS",
            "files": files,
            "avg_score": round(avg_score, 2),
            "file_results": file_results,
        }))
    else:
        print(f"PASS: {files} files, score {avg_score:.2f}")


def _output_fail(
    files: int,
    total_errors: int,
    files_with_errors: int,
    avg_score: float,
    output_format: str,
    file_results: list[dict[str, Any]],
) -> None:
    """Output fail result."""
    if output_format == "json":
        print(json.dumps({
            "status": "FAIL",
            "files": files,
            "total_errors": total_errors,
            "files_with_errors": files_with_errors,
            "avg_score": round(avg_score, 2),
            "file_results": file_results,
        }))
    else:
        print(f"FAIL: {total_errors} errors in {files_with_errors} files")


def _parse_gate_args(args: list[str]) -> dict[str, Any]:
    """Parse gate subcommand arguments."""
    parsed: dict[str, Any] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--format" and i + 1 < len(args):
            parsed["format"] = args[i + 1]
            i += 2
        elif arg in ("--help", "-h"):
            parsed["help"] = True
            i += 1
        else:
            i += 1
    return parsed


def _print_gate_help() -> None:
    """Print usage help for the gate command."""
    print("Usage: mirdan gate [options]")
    print()
    print("Quality gate — validates all changed files.")
    print("Returns exit code 0 (pass) or 1 (fail).")
    print()
    print("Options:")
    print("  --format FORMAT    Output format (text|json)")
    print("  -h, --help         Show this help")
    print()
    print("Exit codes: 0=pass, 1=fail")
