"""``mirdan report`` — generate quality summary for a project."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from mirdan.config import MirdanConfig
from mirdan.core.code_validator import CodeValidator
from mirdan.core.quality_standards import QualityStandards

logger = logging.getLogger(__name__)

# File extensions to scan by language
_LANG_EXTENSIONS: dict[str, list[str]] = {
    "python": [".py"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx"],
    "rust": [".rs"],
    "go": [".go"],
}


def run_report(args: list[str]) -> None:
    """Run the quality report command.

    Args:
        args: CLI arguments after ``report``.
    """
    parsed = _parse_report_args(args)

    if parsed.get("help"):
        _print_report_help()
        sys.exit(0)

    # Session report mode: generate summary from session tracker
    if parsed.get("session"):
        _run_session_report(parsed)
        return

    # Compact state mode: output compaction state for preservation
    if parsed.get("compact_state"):
        _run_compact_state_report(parsed)
        return

    directory = Path(parsed.get("directory", ".")).resolve()
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(2)

    output_format = parsed.get("format", "text")
    language_filter = parsed.get("language")

    config = MirdanConfig.find_config(directory)
    standards = QualityStandards(config=config.quality)
    validator = CodeValidator(standards, config=config.quality, thresholds=config.thresholds)

    # Discover source files
    source_files = _discover_source_files(directory, language_filter)

    if not source_files:
        print("No source files found to analyze.", file=sys.stderr)
        sys.exit(0)

    # Validate each file
    file_results: list[dict[str, Any]] = []
    total_violations = {"error": 0, "warning": 0, "info": 0}
    total_score = 0.0
    passed_count = 0

    for file_path in source_files:
        try:
            code = file_path.read_text()
            if not code.strip():
                continue
            result = validator.validate(code=code, language="auto", check_security=True)
            rel_path = str(file_path.relative_to(directory))
            file_results.append(
                {
                    "file": rel_path,
                    "score": result.score,
                    "passed": result.passed,
                    "language": result.language_detected,
                    "violations": len(result.violations),
                    "errors": sum(1 for v in result.violations if v.severity == "error"),
                    "warnings": sum(1 for v in result.violations if v.severity == "warning"),
                }
            )
            total_score += result.score
            if result.passed:
                passed_count += 1
            for v in result.violations:
                total_violations[v.severity] = total_violations.get(v.severity, 0) + 1
        except Exception:
            logger.debug("Skipping file due to analysis error", exc_info=True)
            continue

    if not file_results:
        print("No files could be analyzed.", file=sys.stderr)
        sys.exit(0)

    avg_score = total_score / len(file_results)
    pass_rate = passed_count / len(file_results)

    report = {
        "directory": str(directory),
        "files_analyzed": len(file_results),
        "avg_score": round(avg_score, 3),
        "pass_rate": round(pass_rate, 3),
        "total_violations": total_violations,
        "files": sorted(file_results, key=lambda f: f["score"]),
    }

    if output_format == "json":
        print(json.dumps(report, indent=2))
    elif output_format == "markdown":
        _output_markdown(report)
    else:
        _output_text_report(report)


def _discover_source_files(
    directory: Path,
    language_filter: str | None = None,
) -> list[Path]:
    """Discover source files to analyze."""
    files: list[Path] = []

    if language_filter:
        extensions = _LANG_EXTENSIONS.get(language_filter, [])
    else:
        extensions = [ext for exts in _LANG_EXTENSIONS.values() for ext in exts]

    for ext in extensions:
        for path in directory.rglob(f"*{ext}"):
            # Skip hidden dirs, venvs, node_modules, __pycache__
            parts = path.parts
            _skip = {"node_modules", "__pycache__", "venv", ".venv", "dist", "build"}
            if any(p.startswith(".") or p in _skip for p in parts):
                continue
            files.append(path)

    return sorted(files)


def _output_text_report(report: dict[str, Any]) -> None:
    """Print human-readable text report."""
    print(f"mirdan Quality Report: {report['directory']}")
    print(f"{'=' * 60}")
    print(f"Files analyzed: {report['files_analyzed']}")
    print(f"Average score:  {report['avg_score']:.3f}")
    print(f"Pass rate:      {report['pass_rate']:.1%}")
    print(f"Total errors:   {report['total_violations'].get('error', 0)}")
    print(f"Total warnings: {report['total_violations'].get('warning', 0)}")
    print()

    # Bottom 10 files by score
    worst = [f for f in report["files"] if not f["passed"]]
    if worst:
        print("Files with violations:")
        for f in worst[:10]:
            print(f"  {f['score']:.2f} {f['file']} ({f['errors']}E {f['warnings']}W)")
        print()

    # Top AI-specific violations summary
    print("Pass/Fail:")
    status = "PASS" if report["pass_rate"] >= 0.8 else "NEEDS WORK"
    print(f"  Overall: {status}")


def _output_markdown(report: dict[str, Any]) -> None:
    """Print markdown-formatted report."""
    avg = report["avg_score"]
    pr = report["pass_rate"]
    te = report["total_violations"].get("error", 0)
    tw = report["total_violations"].get("warning", 0)

    print("# mirdan Quality Report\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Files analyzed | {report['files_analyzed']} |")
    print(f"| Average score | {avg:.3f} |")
    print(f"| Pass rate | {pr:.1%} |")
    print(f"| Total errors | {te} |")
    print(f"| Total warnings | {tw} |")
    print()

    worst = [f for f in report["files"] if not f["passed"]]
    if worst:
        print("## Files Needing Attention\n")
        print("| Score | File | Errors | Warnings |")
        print("|-------|------|--------|----------|")
        for f in worst[:15]:
            print(f"| {f['score']:.2f} | `{f['file']}` | {f['errors']} | {f['warnings']} |")


def _parse_report_args(args: list[str]) -> dict[str, Any]:
    """Parse report subcommand arguments."""
    parsed: dict[str, Any] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--format" and i + 1 < len(args):
            parsed["format"] = args[i + 1]
            i += 2
        elif arg == "--language" and i + 1 < len(args):
            parsed["language"] = args[i + 1]
            i += 2
        elif arg == "--session":
            parsed["session"] = True
            i += 1
        elif arg == "--compact-state":
            parsed["compact_state"] = True
            i += 1
        elif arg in ("--help", "-h"):
            parsed["help"] = True
            i += 1
        else:
            parsed["directory"] = arg
            i += 1
    return parsed


def _run_session_report(parsed: dict[str, Any]) -> None:
    """Generate a session quality report.

    Uses the same validation infrastructure to scan changed files
    and produce a session-oriented summary.
    """
    import subprocess as _sp

    output_format = parsed.get("format", "text")

    # Get changed files from git
    try:
        result = _sp.run(
            ["git", "diff", "--name-only", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
        )
        changed_files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    except (FileNotFoundError, _sp.TimeoutExpired):
        changed_files = []

    if not changed_files:
        session_data = {
            "validation_count": 0,
            "avg_score": 1.0,
            "files_validated": 0,
            "unresolved_errors": 0,
        }
    else:
        config = MirdanConfig.find_config()
        standards = QualityStandards(config=config.quality)
        validator = CodeValidator(
            standards,
            config=config.quality,
            thresholds=config.thresholds,
        )

        file_results: list[dict[str, Any]] = []
        total_score = 0.0
        total_errors = 0

        for file_str in changed_files:
            file_path = Path(file_str)
            if not file_path.exists() or not file_path.is_file():
                continue
            try:
                code = file_path.read_text()
                if not code.strip():
                    continue
                res = validator.validate(code=code, language="auto", check_security=True)
                errors = sum(1 for v in res.violations if v.severity == "error")
                total_errors += errors
                total_score += res.score
                file_results.append(
                    {
                        "file": file_str,
                        "score": res.score,
                        "passed": res.passed,
                        "errors": errors,
                    }
                )
            except Exception:
                logger.debug("Skipping file in session report", exc_info=True)
                continue

        files_validated = len(file_results)
        session_data = {
            "validation_count": files_validated,
            "avg_score": total_score / files_validated if files_validated else 1.0,
            "files_validated": files_validated,
            "unresolved_errors": total_errors,
        }

    if output_format == "json":
        print(json.dumps(session_data, indent=2))
    else:
        from mirdan.integrations.self_managing import SelfManagingIntegration

        integration = SelfManagingIntegration()
        print(
            integration.generate_session_summary(
                session_data,
                file_results=file_results if changed_files else None,
            )
        )


def _run_compact_state_report(parsed: dict[str, Any]) -> None:
    """Output current compaction state for context preservation."""
    from mirdan.integrations.self_managing import SelfManagingIntegration
    from mirdan.models import CompactState

    output_format = parsed.get("format", "text")

    # Create a minimal compact state from current context
    state = CompactState()

    if output_format == "json":
        print(json.dumps(state.to_dict(), indent=2))
    else:
        integration = SelfManagingIntegration()
        print(integration.generate_compaction_state(state))


def _print_report_help() -> None:
    """Print usage help for the report command."""
    print("Usage: mirdan report [directory] [options]")
    print()
    print("Generate a quality summary for a project.")
    print()
    print("Options:")
    print("  --format FORMAT     Output format (text|json|markdown)")
    print("  --language LANG     Filter by language (python|typescript|javascript|rust|go)")
    print("  --session           Generate session quality report (validates changed files)")
    print("  --compact-state     Output compaction state for context preservation")
    print("  -h, --help          Show this help")
