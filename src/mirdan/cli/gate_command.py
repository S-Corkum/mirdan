"""``mirdan gate`` — quality gate for Stop hooks and CI.

Validates all changed files and returns exit code 0 (pass) or 1 (fail).
Designed for use as a Claude Code Stop hook command.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
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
        standards,
        config=config.quality,
        thresholds=config.thresholds,
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
            code=code,
            language="auto",
            check_security=True,
            file_path=file_path,
        )

        error_count = sum(1 for v in result.violations if v.severity == "error")
        warning_count = sum(1 for v in result.violations if v.severity == "warning")
        total_errors += error_count
        total_score += result.score
        if error_count > 0:
            files_with_errors += 1

        entry: dict[str, Any] = {
            "file": file_path,
            "score": result.score,
            "passed": result.passed,
            "errors": error_count,
            "warnings": warning_count,
        }

        # Attach sub-project info when in workspace mode
        if config.is_workspace:
            sub = config.resolve_project_for_path(file_path)
            if sub is not None:
                entry["project"] = sub.path
                entry["project_language"] = sub.primary_language

        file_results.append(entry)

    files_validated = len(file_results)
    avg_score = total_score / files_validated if files_validated else 1.0

    # Dependency vulnerability check
    include_deps = "--include-dependencies" in args or "--include-deps" in args
    if include_deps or config.dependencies.scan_on_gate:
        blocking = _scan_dependencies_for_gate(config)
        if blocking:
            print(
                f"FAIL: {len(blocking)} dependency vulnerabilities at or above "
                f"'{config.dependencies.fail_on_severity}' severity"
            )
            sys.exit(1)

    if total_errors == 0:
        _output_pass(files_validated, avg_score, output_format, file_results)
        sys.exit(0)
    else:
        _output_fail(
            files_validated,
            total_errors,
            files_with_errors,
            avg_score,
            output_format,
            file_results,
            is_workspace=config.is_workspace,
        )
        sys.exit(1)


def _scan_dependencies_for_gate(config: MirdanConfig) -> list[Any]:
    """Scan dependencies for vulnerabilities, workspace-aware.

    In workspace mode, iterates each sub-project directory and merges
    all vulnerability findings. In single-project mode, scans cwd.

    Returns:
        List of blocking vulnerability findings.
    """
    import asyncio

    from mirdan.core.manifest_parser import ManifestParser
    from mirdan.core.vuln_scanner import VulnScanner

    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    fail_threshold = sev_order.get(config.dependencies.fail_on_severity, 1)

    if config.is_workspace:
        # Collect unique project directories from workspace config
        seen_dirs: set[Path] = set()
        project_dirs: list[Path] = []
        for sub in config.workspace.projects:
            sub_dir = Path.cwd() / sub.path
            if sub_dir not in seen_dirs:
                seen_dirs.add(sub_dir)
                project_dirs.append(sub_dir)
    else:
        project_dirs = [Path.cwd()]

    all_blocking: list[Any] = []
    for project_dir in project_dirs:
        manifest_parser = ManifestParser(project_dir=project_dir)
        packages = manifest_parser.parse()
        if not packages:
            continue
        vuln_scanner = VulnScanner(
            cache_dir=project_dir / ".mirdan" / "cache",
            ttl=config.dependencies.osv_cache_ttl,
        )
        findings = asyncio.run(vuln_scanner.scan(packages))
        blocking = [f for f in findings if sev_order.get(f.severity, 4) <= fail_threshold]
        all_blocking.extend(blocking)

    return all_blocking


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
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "files": files,
                    "avg_score": round(avg_score, 2),
                    "file_results": file_results,
                }
            )
        )
    else:
        print(f"PASS: {files} files, score {avg_score:.2f}")


def _output_fail(
    files: int,
    total_errors: int,
    files_with_errors: int,
    avg_score: float,
    output_format: str,
    file_results: list[dict[str, Any]],
    *,
    is_workspace: bool = False,
) -> None:
    """Output fail result."""
    if output_format == "json":
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "files": files,
                    "total_errors": total_errors,
                    "files_with_errors": files_with_errors,
                    "avg_score": round(avg_score, 2),
                    "file_results": file_results,
                }
            )
        )
    elif is_workspace:
        print(f"FAIL: {total_errors} errors in {files_with_errors} files")
        print()
        _print_workspace_grouped_results(file_results)
    else:
        print(f"FAIL: {total_errors} errors in {files_with_errors} files")


def _print_workspace_grouped_results(file_results: list[dict[str, Any]]) -> None:
    """Print file results grouped by sub-project for workspace mode."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in file_results:
        project_key = entry.get("project", "(root)")
        grouped[project_key].append(entry)

    for project_path in sorted(grouped):
        entries = grouped[project_path]
        lang = ""
        for e in entries:
            if e.get("project_language"):
                lang = e["project_language"]
                break
        header = f"  [{project_path}]"
        if lang:
            header += f" ({lang})"
        print(header)
        for e in entries:
            errors = e.get("errors", 0)
            warnings = e.get("warnings", 0)
            print(f"    {e['score']:.2f} {e['file']} ({errors}E {warnings}W)")
        print()


def _parse_gate_args(args: list[str]) -> dict[str, Any]:
    """Parse gate subcommand arguments."""
    parsed: dict[str, Any] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--format" and i + 1 < len(args):
            parsed["format"] = args[i + 1]
            i += 2
        elif arg in ("--include-dependencies", "--include-deps"):
            parsed["include_deps"] = True
            i += 1
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
    print("  --format FORMAT              Output format (text|json)")
    print("  --include-dependencies       Also scan for vulnerable dependencies")
    print("  --include-deps               Alias for --include-dependencies")
    print("  -h, --help                   Show this help")
    print()
    print("Exit codes: 0=pass, 1=fail")
