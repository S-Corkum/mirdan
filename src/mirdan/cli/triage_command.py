"""``mirdan triage`` — classify a task via local LLM or sidecar."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx


def run_triage(args: list[str]) -> None:
    """Read prompt from stdin, classify via sidecar or fallback, print result.

    Usage:
        echo '{"prompt":"fix unused import"}' | mirdan triage --stdin

    Args:
        args: CLI arguments after ``triage``.
    """
    if "--help" in args or "-h" in args:
        _print_triage_help()
        sys.exit(0)

    if "--stdin" not in args:
        print("Error: --stdin flag is required", file=sys.stderr)
        print("Usage: echo '{\"prompt\":\"...\"}' | mirdan triage --stdin")
        sys.exit(1)

    # Read prompt from stdin
    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        # Treat raw text as the prompt
        data = {"prompt": raw.strip()}

    prompt = data.get("prompt", "")
    if not prompt:
        print(json.dumps({"error": "no prompt provided"}))
        sys.exit(1)

    # Try sidecar first (fast path — reuses warm model)
    result = _try_sidecar(data)
    if result is not None:
        print(json.dumps(result))
        _write_to_session_bridge(result)
        return

    # Fallback: return a default (triage engine wired in Phase 3)
    fallback = {
        "classification": "paid_required",
        "confidence": 0.0,
        "reasoning": "no sidecar or triage engine available",
    }
    print(json.dumps(fallback))
    _write_to_session_bridge(fallback)


def _try_sidecar(data: dict[str, Any]) -> dict[str, Any] | None:
    """Try to POST to the sidecar's /triage endpoint.

    Returns:
        Response dict, or None if sidecar is not running.
    """
    port_file = Path(".mirdan/sidecar.port")
    if not port_file.exists():
        return None

    try:
        port = int(port_file.read_text().strip())
        resp = httpx.post(
            f"http://127.0.0.1:{port}/triage",
            json=data,
            timeout=10.0,
        )
        if resp.status_code == 200:
            result: dict[str, Any] = resp.json()
            return result
    except (httpx.ConnectError, httpx.TimeoutException, ValueError, OSError):
        pass

    return None


def _write_to_session_bridge(result: dict[str, Any]) -> None:
    """Write triage result to session bridge for MCP dedup."""
    try:
        from mirdan.coordination.session_bridge import get_session_id, write_triage

        session_id = get_session_id()
        write_triage(session_id, result)
    except Exception:
        pass  # Non-critical — don't fail the CLI command


def _print_triage_help() -> None:
    print("mirdan triage — classify a coding task")
    print()
    print("Usage:")
    print("  echo '{\"prompt\":\"...\"}' | mirdan triage --stdin")
    print()
    print("Options:")
    print("  --stdin    Read prompt from stdin (required)")
    print("  -h, --help Show this help")
