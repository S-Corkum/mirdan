"""``mirdan check`` — run lint + typecheck + test with optional LLM analysis and fix."""

from __future__ import annotations

import asyncio
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

from mirdan.config import CheckRunnerConfig, MirdanConfig


def run_check(args: list[str]) -> None:
    """Run checks via sidecar or locally, print structured summary.

    Usage:
        mirdan check --smart [files...]
        mirdan check --smart --fix [files...]

    Args:
        args: CLI arguments after ``check``.
    """
    if "--help" in args or "-h" in args:
        _print_check_help()
        sys.exit(0)

    fix_mode = "--fix" in args
    files = [a for a in args if not a.startswith("-")]

    if fix_mode and not files:
        print("Error: --fix requires specific file paths", file=sys.stderr)
        sys.exit(1)

    # Try sidecar first (report-only mode — fix mode runs locally)
    if not fix_mode:
        sidecar_result = _try_sidecar(files)
        if sidecar_result is not None:
            print(json.dumps(sidecar_result, indent=2))
            _write_to_session_bridge(sidecar_result)
            return

    # Fallback: run tools locally
    config = MirdanConfig.find_config()
    checks_config = _resolve_checks(config)

    test_files = _filter_test_files(files)
    typecheck_files = _typecheck_target(checks_config.typecheck_command, files)
    lint_result = _run_subprocess(checks_config.lint_command, files)
    typecheck_result = _run_subprocess(checks_config.typecheck_command, typecheck_files)
    test_result = _run_subprocess(
        checks_config.test_command, test_files, timeout=checks_config.test_timeout
    )

    # Auto-fix lint if configured
    auto_fixed: list[str] = []
    if checks_config.auto_fix_lint and lint_result["returncode"] != 0:
        fix_result = _run_subprocess(checks_config.lint_command + " --fix", files)
        if fix_result["returncode"] == 0:
            auto_fixed.append("lint auto-fixed")
        # Re-run lint to see what remains
        lint_result = _run_subprocess(checks_config.lint_command, files)

    all_pass = (
        lint_result["returncode"] == 0
        and typecheck_result["returncode"] == 0
        and test_result["returncode"] == 0
    )

    # Fix mode: apply LLM-generated fixes for remaining issues
    fix_report: dict[str, Any] | None = None
    if fix_mode and not all_pass and config.llm.enabled:
        fix_report = asyncio.run(_run_llm_fix_loop(files, lint_result, typecheck_result, config))
        if fix_report:
            # Re-run checks to verify
            lint_result = _run_subprocess(checks_config.lint_command, files)
            typecheck_result = _run_subprocess(checks_config.typecheck_command, files)
            test_result = _run_subprocess(
                checks_config.test_command, [], timeout=checks_config.test_timeout
            )
            all_pass = (
                lint_result["returncode"] == 0
                and typecheck_result["returncode"] == 0
                and test_result["returncode"] == 0
            )

    results_list = [lint_result, typecheck_result, test_result]
    code_quality_pass = all(r.get("classification") != "code_quality" for r in results_list)
    infra_ok = all(r.get("classification") != "infrastructure" for r in results_list)
    infra_failures = [
        name
        for name, r in (
            ("lint", lint_result),
            ("typecheck", typecheck_result),
            ("test", test_result),
        )
        if r.get("classification") == "infrastructure"
    ]

    result: dict[str, Any] = {
        "lint": lint_result,
        "typecheck": typecheck_result,
        "test": test_result,
        "all_pass": all_pass,
        "code_quality_pass": code_quality_pass,
        "infra_ok": infra_ok,
        "infra_failures": infra_failures,
        "auto_fixed": auto_fixed,
        "summary": "all checks pass" if all_pass else "some checks failed",
    }
    if fix_report:
        result["llm_fixes"] = fix_report

    print(json.dumps(result, indent=2))
    _write_to_session_bridge(result)


async def _run_llm_fix_loop(
    files: list[str],
    lint_result: dict[str, Any],
    typecheck_result: dict[str, Any],
    config: MirdanConfig,
) -> dict[str, Any] | None:
    """Run LLM fix generation for failing files.

    Args:
        files: File paths to fix.
        lint_result: Lint subprocess result dict.
        typecheck_result: Typecheck subprocess result dict.
        config: Mirdan config with LLM settings.

    Returns:
        Fix report dict, or None if no fixes were possible.
    """
    from mirdan.core.llm_fixer import LLMFixer
    from mirdan.llm.manager import LLMManager

    manager = LLMManager.create_if_enabled(config.llm)
    if not manager:
        return None

    fixer = LLMFixer(llm_manager=manager, config=config.llm)

    # Build per-file violations from tool output
    # The LLM will see the raw error messages and the file content
    all_reports: list[dict[str, Any]] = []

    for file_path in files:
        if not Path(file_path).exists():
            continue

        # Extract violations mentioning this file from lint/typecheck output
        file_name = Path(file_path).name
        lint_output = lint_result.get("stdout", "") + "\n" + lint_result.get("stderr", "")
        tc_output = typecheck_result.get("stdout", "") + "\n" + typecheck_result.get("stderr", "")

        violations: list[dict[str, Any]] = [
            {"id": "LINT", "message": line.strip(), "tool": "lint"}
            for line in lint_output.splitlines()
            if file_name in line or file_path in line
        ] + [
            {"id": "TYPE", "message": line.strip(), "tool": "typecheck"}
            for line in tc_output.splitlines()
            if file_name in line or file_path in line
        ]

        if not violations:
            continue

        report = await fixer.fix_file(file_path, violations)
        if report.applied:
            all_reports.append(report.to_dict())

    if not all_reports:
        return None

    return {
        "files_fixed": len(all_reports),
        "details": all_reports,
    }


def _resolve_checks(config: MirdanConfig) -> CheckRunnerConfig:
    """Resolve the effective ``CheckRunnerConfig`` for this invocation.

    Legacy configs written under 2.0.x have ``project.primary_language`` set
    from detection but no ``llm.checks`` block persisted — so
    ``config.llm.checks`` falls back to the Pydantic defaults (Python
    tooling). When the loaded ``checks`` is exactly those defaults AND the
    project's primary language has its own known default set, swap in the
    language-appropriate commands.

    Known ambiguity: if a user has explicitly set ``ruff check`` / ``mypy``
    / ``pytest`` as their checks in a non-Python project (unlikely but
    possible), this helper will overwrite them. They can force their choice
    back by changing any one of the three commands.
    """
    is_python_default = (
        config.llm.checks.lint_command == "ruff check"
        and config.llm.checks.typecheck_command == "mypy"
        and config.llm.checks.test_command == "pytest -x --tb=short"
    )
    lang = config.project.primary_language.lower()
    if is_python_default and lang and lang != "python":
        from mirdan.core.check_defaults import defaults_for_language

        fallback = defaults_for_language(lang)
        if fallback is not None:
            return fallback
    return config.llm.checks


def _filter_test_files(files: list[str]) -> list[str]:
    """Return the subset of ``files`` that look like test files.

    A path is treated as a test file when any path segment starts with
    ``test_``, ends with ``_test`` before the extension, or is named
    ``tests``. When no input files are given, returns an empty list
    (caller runs the default test-command target).
    """
    if not files:
        return []

    def _is_test(path: str) -> bool:
        parts = Path(path).parts
        for part in parts:
            if part == "tests" or part.startswith("test_"):
                return True
        stem = Path(path).stem
        return stem.endswith("_test")

    return [f for f in files if _is_test(f)]


def _typecheck_target(command: str, files: list[str]) -> list[str]:
    """Pick arguments for the typecheck subprocess.

    Some typecheckers (``mypy``) error when invoked with no target and no
    config-level target. If the user supplied files, use those; otherwise
    fall back to the current directory so the command has *something* to
    check. Commands that already include an explicit target in the string
    (e.g. ``mypy src/``) are left untouched.
    """
    if files:
        return files

    tokens = shlex.split(command)
    if not tokens:
        return []

    base = Path(tokens[0]).name
    if base != "mypy":
        return []

    has_target = any(
        tok in {"src", "src/", "."} or tok.startswith(("src/", "./", "/"))
        for tok in tokens[1:]
    )
    if has_target:
        return []

    # `mypy` with no target fails; `.` is a safe default that mypy can scope
    # via project pyproject/mypy.ini settings.
    return ["."]


def _run_subprocess(command: str, files: list[str], timeout: int = 60) -> dict[str, Any]:
    """Run a command with optional file arguments.

    Uses shlex.split to safely tokenize the command string and passes
    arguments as a list to subprocess.run WITHOUT shell=True, preventing
    shell injection via malicious config values or filenames.

    Args:
        command: Command string (will be split with shlex).
        files: File paths to append as separate arguments.
        timeout: Timeout in seconds for the subprocess.

    Returns:
        Dict with command, returncode, stdout, stderr.
    """
    args = shlex.split(command) + (files if files else [])
    display_cmd = " ".join(args)

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": display_cmd,
            "returncode": result.returncode,
            "stdout": result.stdout[:5000],  # Cap output size
            "stderr": result.stderr[:2000],
            "classification": "ok" if result.returncode == 0 else "code_quality",
        }
    except subprocess.TimeoutExpired:
        return {
            "command": display_cmd,
            "returncode": -1,
            "stdout": "",
            "stderr": f"timeout after {timeout}s",
            "classification": "infrastructure",
        }
    except FileNotFoundError:
        return {
            "command": display_cmd,
            "returncode": -1,
            "stdout": "",
            "stderr": f"command not found: {args[0] if args else command}",
            "classification": "infrastructure",
        }


def _try_sidecar(files: list[str]) -> dict[str, Any] | None:
    """Try to POST to the sidecar's /check endpoint.

    Returns:
        Response dict, or None if sidecar is not running.
    """
    port_file = Path(".mirdan/sidecar.port")
    if not port_file.exists():
        return None

    try:
        port = int(port_file.read_text().strip())
        resp = httpx.post(
            f"http://127.0.0.1:{port}/check",
            json={"files": files},
            timeout=60.0,
        )
        if resp.status_code == 200:
            result: dict[str, Any] = resp.json()
            return result
    except (httpx.ConnectError, httpx.TimeoutException, ValueError, OSError):
        pass

    return None


def _write_to_session_bridge(result: dict[str, Any]) -> None:
    """Write check result to session bridge."""
    try:
        from mirdan.coordination.session_bridge import get_session_id, write_check_result

        session_id = get_session_id()
        write_check_result(session_id, result)
    except Exception:
        pass


def _print_check_help() -> None:
    print("mirdan check — run lint, typecheck, and test checks")
    print()
    print("Usage:")
    print("  mirdan check --smart [files...]")
    print("  mirdan check --smart --fix [files...]")
    print()
    print("Options:")
    print("  --smart    Enable LLM-enhanced analysis (when available)")
    print("  --fix      Apply LLM-generated fixes for failing checks")
    print("  -h, --help Show this help")
