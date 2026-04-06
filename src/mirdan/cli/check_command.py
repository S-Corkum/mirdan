"""``mirdan check`` — run lint + typecheck + test with optional LLM analysis."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

from mirdan.config import MirdanConfig


def run_check(args: list[str]) -> None:
    """Run checks via sidecar or locally, print structured summary.

    Usage:
        mirdan check --smart [files...]

    Args:
        args: CLI arguments after ``check``.
    """
    if "--help" in args or "-h" in args:
        _print_check_help()
        sys.exit(0)

    smart = "--smart" in args
    files = [a for a in args if not a.startswith("-")]

    # Try sidecar first
    result = _try_sidecar(files)
    if result is not None:
        print(json.dumps(result, indent=2))
        _write_to_session_bridge(result)
        return

    # Fallback: run tools locally
    config = MirdanConfig.find_config()
    checks_config = config.llm.checks

    lint_result = _run_subprocess(checks_config.lint_command, files)
    typecheck_result = _run_subprocess(checks_config.typecheck_command, files)
    test_result = _run_subprocess(checks_config.test_command, [])

    # Auto-fix lint if configured
    auto_fixed: list[str] = []
    if checks_config.auto_fix_lint and lint_result["returncode"] != 0:
        fix_result = _run_subprocess(checks_config.lint_command + " --fix", files)
        if fix_result["returncode"] == 0:
            auto_fixed.append("lint auto-fixed")

    all_pass = (
        lint_result["returncode"] == 0
        and typecheck_result["returncode"] == 0
        and test_result["returncode"] == 0
    )

    result = {
        "lint": lint_result,
        "typecheck": typecheck_result,
        "test": test_result,
        "all_pass": all_pass,
        "auto_fixed": auto_fixed,
        "summary": "all checks pass" if all_pass else "some checks failed",
    }

    print(json.dumps(result, indent=2))
    _write_to_session_bridge(result)


def _run_subprocess(command: str, files: list[str]) -> dict[str, Any]:
    """Run a shell command with optional file arguments.

    Args:
        command: Shell command string.
        files: File paths to append.

    Returns:
        Dict with command, returncode, stdout, stderr.
    """
    full_cmd = command
    if files:
        full_cmd = f"{command} {' '.join(files)}"

    try:
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return {
            "command": full_cmd,
            "returncode": result.returncode,
            "stdout": result.stdout[:5000],  # Cap output size
            "stderr": result.stderr[:2000],
        }
    except subprocess.TimeoutExpired:
        return {
            "command": full_cmd,
            "returncode": -1,
            "stdout": "",
            "stderr": "timeout after 60s",
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
        from mirdan.llm.session_bridge import get_session_id, write_check_result

        session_id = get_session_id()
        write_check_result(session_id, result)
    except Exception:
        pass


def _print_check_help() -> None:
    print("mirdan check — run lint, typecheck, and test checks")
    print()
    print("Usage:")
    print("  mirdan check --smart [files...]")
    print()
    print("Options:")
    print("  --smart    Enable LLM-enhanced analysis (when available)")
    print("  -h, --help Show this help")
