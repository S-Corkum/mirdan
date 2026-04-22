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
        print('Usage: echo \'{"prompt":"..."}\' | mirdan triage --stdin')
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
        _augment_with_human_label(result)
        print(json.dumps(result))
        _write_to_session_bridge(result)
        return

    # Fallback: spin up a short-lived LLMManager in-process. Slow first time
    # (cold-load ~30-60s on Intel), but avoids the useless "no sidecar" stub.
    result = _try_local_triage(prompt)
    if result is not None:
        _augment_with_human_label(result)
        print(json.dumps(result))
        _write_to_session_bridge(result)
        return

    # Last resort: rules-only stub (LLM not configured or failed to load)
    fallback = {
        "classification": "paid_required",
        "confidence": 0.0,
        "reasoning": "no sidecar and local LLM unavailable — proceed normally",
    }
    _augment_with_human_label(fallback)
    print(json.dumps(fallback))
    _write_to_session_bridge(fallback)


def _augment_with_human_label(payload: dict[str, Any]) -> None:
    """Attach a plain-English ``meaning`` field if the classification is known."""
    try:
        from mirdan.llm.sidecar import _augment_with_human_label as _impl

        _impl(payload)
    except Exception:  # labelling is best-effort — never fail the CLI for it
        pass


def _try_local_triage(prompt: str) -> dict[str, Any] | None:
    """Run triage in-process via LLMManager when sidecar is unreachable.

    Returns a response dict, or None if the local LLM subsystem is disabled
    or unavailable. Cold-loads the model on first call — expect 30-60s
    latency on a CPU-only Intel host.
    """
    import asyncio

    async def _run() -> dict[str, Any] | None:
        from mirdan.config import MirdanConfig
        from mirdan.core.triage import TriageEngine
        from mirdan.llm.manager import LLMManager

        cfg = MirdanConfig.find_config()
        mgr = LLMManager.create_if_enabled(cfg.llm)
        if mgr is None:
            return None
        import contextlib

        try:
            await mgr.startup()
            engine = TriageEngine(llm_manager=mgr, config=cfg.llm)
            result = await engine.classify(prompt)
            if result is None:
                return None
            return dict(result.to_dict())
        finally:
            with contextlib.suppress(Exception):
                await mgr.shutdown()

    try:
        return asyncio.run(_run())
    except Exception:  # fall through to the stub on any failure
        return None


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
    print('  echo \'{"prompt":"..."}\' | mirdan triage --stdin')
    print()
    print("Options:")
    print("  --stdin    Read prompt from stdin (required)")
    print("  -h, --help Show this help")
