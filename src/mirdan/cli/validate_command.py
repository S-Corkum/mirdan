"""``mirdan validate`` — code quality validation CLI command."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from mirdan.config import MirdanConfig
from mirdan.core.code_validator import CodeValidator
from mirdan.core.diff_parser import parse_unified_diff
from mirdan.core.linter_orchestrator import create_linter_runner, merge_linter_violations
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import ValidationResult


def run_validate(args: list[str]) -> None:
    """Run code quality validation from the CLI.

    Supports three input modes:
    - ``--file path``: validate a file on disk
    - ``--stdin``: read code from stdin
    - ``--diff``: read unified diff from stdin

    Args:
        args: CLI arguments after ``validate``.
    """
    parsed = _parse_args(args)

    if parsed.get("error"):
        print(f"Error: {parsed['error']}", file=sys.stderr)
        _print_validate_help()
        sys.exit(2)

    config = MirdanConfig.find_config()
    standards = QualityStandards(config=config.quality)
    validator = CodeValidator(standards, config=config.quality, thresholds=config.thresholds)

    output_format = parsed.get("format", "text")
    language = parsed.get("language", "auto")
    check_security = parsed.get("security", True)

    run_lint = parsed.get("lint", False)
    quick_mode = parsed.get("quick", False)

    try:
        if parsed.get("staged"):
            diff_text = _get_staged_diff()
            if not diff_text.strip():
                result = ValidationResult(
                    passed=True,
                    score=1.0,
                    language_detected=language if language != "auto" else "unknown",
                    standards_checked=["security"],
                )
            elif quick_mode:
                parsed_diff = parse_unified_diff(diff_text)
                added_code = parsed_diff.get_added_code()
                if not added_code.strip():
                    result = ValidationResult(
                        passed=True,
                        score=1.0,
                        language_detected=language if language != "auto" else "unknown",
                        standards_checked=["security"],
                    )
                else:
                    result = validator.validate_quick(code=added_code, language=language)
            else:
                result = _validate_diff(validator, diff_text, language, check_security)
        elif parsed.get("diff"):
            diff_text = sys.stdin.read()
            if quick_mode:
                parsed_diff = parse_unified_diff(diff_text)
                added_code = parsed_diff.get_added_code()
                if not added_code.strip():
                    result = ValidationResult(
                        passed=True,
                        score=1.0,
                        language_detected=language if language != "auto" else "unknown",
                        standards_checked=["security"],
                    )
                else:
                    result = validator.validate_quick(code=added_code, language=language)
            else:
                result = _validate_diff(validator, diff_text, language, check_security)
        elif parsed.get("stdin"):
            code = sys.stdin.read()
            if quick_mode:
                result = validator.validate_quick(code=code, language=language)
            else:
                result = validator.validate(
                    code=code,
                    language=language,
                    check_security=check_security,
                )
        elif parsed.get("file"):
            file_path = Path(parsed["file"])
            if not file_path.exists():
                print(f"Error: file not found: {file_path}", file=sys.stderr)
                sys.exit(2)
            code = file_path.read_text()
            if quick_mode:
                result = validator.validate_quick(code=code, language=language)
            else:
                result = validator.validate(
                    code=code,
                    language=language,
                    check_security=check_security,
                )
                # Run external linters if requested
                if run_lint:
                    result = _run_linters(result, file_path, language, config)
        else:
            print("Error: one of --file, --stdin, or --diff is required", file=sys.stderr)
            _print_validate_help()
            sys.exit(2)
            return  # unreachable, but helps type-checker
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
        return

    _output_result(result, output_format, parsed.get("file"))
    sys.exit(0 if result.passed else 1)


def _get_staged_diff() -> str:
    """Get the diff of staged (git add) changes."""
    try:
        result = subprocess.run(
            ["git", "diff", "--staged", "--unified=0"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _validate_diff(
    validator: CodeValidator,
    diff_text: str,
    language: str,
    check_security: bool,
) -> ValidationResult:
    """Validate only added lines from a unified diff."""
    parsed_diff = parse_unified_diff(diff_text)
    added_code = parsed_diff.get_added_code()

    if not added_code.strip():
        return ValidationResult(
            passed=True,
            score=1.0,
            language_detected=language if language != "auto" else "unknown",
            standards_checked=["style", "security"],
        )

    return validator.validate(
        code=added_code,
        language=language,
        check_security=check_security,
        check_architecture=False,
        check_style=True,
    )


def _run_linters(
    result: ValidationResult,
    file_path: Path,
    language: str,
    config: MirdanConfig,
) -> ValidationResult:
    """Run external linters and merge violations into the result."""
    runner = create_linter_runner(config)

    # Use the detected language from validation if auto
    lang = result.language_detected if language == "auto" else language

    linter_violations = asyncio.run(runner.run(file_path, lang))

    if linter_violations:
        return merge_linter_violations(result, linter_violations, config.thresholds)

    return result


def _output_result(
    result: ValidationResult,
    output_format: str,
    file_path: str | None = None,
) -> None:
    """Format and print validation result."""
    if output_format == "json":
        output = result.to_dict(severity_threshold="info")
        if file_path:
            output["file"] = file_path
        print(json.dumps(output, indent=2))

    elif output_format == "github":
        for v in result.violations:
            level = "error" if v.severity == "error" else "warning"
            file_part = f"file={file_path}," if file_path else ""
            line_part = f"line={v.line}," if v.line else ""
            print(f"::{level} {file_part}{line_part}col={v.column or 1}::{v.id}: {v.message}")

    elif output_format == "micro":
        _output_micro(result)

    else:  # text
        _output_text(result, file_path)


def _output_micro(result: ValidationResult) -> None:
    """Ultra-minimal single-line output for hooks."""
    if result.passed:
        print(f"PASS {result.score:.2f}")
        return

    error_count = sum(1 for v in result.violations if v.severity == "error")
    warn_count = sum(1 for v in result.violations if v.severity == "warning")
    violation_parts = []
    for v in result.violations:
        if v.severity in ("error", "warning"):
            loc = f":L{v.line}" if v.line else ""
            violation_parts.append(f"{v.id}{loc}")
    violations_str = " ".join(violation_parts[:10])  # Cap at 10 for brevity
    print(f"FAIL:{error_count}E {warn_count}W {violations_str}")


def _output_text(result: ValidationResult, file_path: str | None = None) -> None:
    """Human-readable text output."""
    status = "PASS" if result.passed else "FAIL"
    file_label = f" ({file_path})" if file_path else ""
    print(f"{status}{file_label} - score: {result.score:.2f}")

    if not result.violations:
        return

    for v in result.violations:
        loc = f":{v.line}" if v.line else ""
        prefix = f"  {v.severity.upper():7s} {v.id}"
        print(f"{prefix}{loc} {v.message}")
        if v.suggestion:
            print(f"          -> {v.suggestion}")


def _parse_args(args: list[str]) -> dict:
    """Parse validate subcommand arguments."""
    parsed: dict = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--file" and i + 1 < len(args):
            parsed["file"] = args[i + 1]
            i += 2
        elif arg == "--stdin":
            parsed["stdin"] = True
            i += 1
        elif arg == "--diff":
            parsed["diff"] = True
            i += 1
        elif arg == "--language" and i + 1 < len(args):
            parsed["language"] = args[i + 1]
            i += 2
        elif arg == "--security":
            parsed["security"] = True
            i += 1
        elif arg == "--no-security":
            parsed["security"] = False
            i += 1
        elif arg == "--lint":
            parsed["lint"] = True
            i += 1
        elif arg == "--no-lint":
            parsed["lint"] = False
            i += 1
        elif arg == "--quick":
            parsed["quick"] = True
            i += 1
        elif arg == "--staged":
            parsed["staged"] = True
            i += 1
        elif arg == "--format" and i + 1 < len(args):
            fmt = args[i + 1]
            if fmt not in ("json", "text", "github", "micro"):
                parsed["error"] = f"invalid format: {fmt}"
                return parsed
            parsed["format"] = fmt
            i += 2
        elif arg in ("--help", "-h"):
            _print_validate_help()
            sys.exit(0)
        else:
            parsed["error"] = f"unknown argument: {arg}"
            return parsed
    return parsed


def _print_validate_help() -> None:
    """Print usage help for the validate command."""
    print("Usage: mirdan validate [options]")
    print()
    print("Input (one required):")
    print("  --file PATH       Validate a file")
    print("  --stdin            Read code from stdin")
    print("  --diff             Read unified diff from stdin")
    print("  --staged           Validate git staged changes")
    print()
    print("Options:")
    print("  --language LANG    Language (python|typescript|javascript|rust|go|auto)")
    print("  --security         Enable security checks (default)")
    print("  --no-security      Disable security checks")
    print("  --lint             Run external linters (ruff, eslint, mypy)")
    print("  --no-lint          Skip external linters (default)")
    print("  --quick            Fast security-only validation (for hooks)")
    print("  --format FORMAT    Output format (text|json|github|micro)")
    print("  -h, --help         Show this help")
    print()
    print("Exit codes: 0=pass, 1=fail, 2=error")
