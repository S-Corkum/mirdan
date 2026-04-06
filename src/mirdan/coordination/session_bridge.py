"""File-based session coordination between hook scripts and MCP server."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Base directory for session data.
_SESSIONS_DIR = ".mirdan/sessions"


def get_session_id() -> str:
    """Get session ID from environment or generate a fallback UUID.

    Checks CLAUDE_SESSION_ID, then CURSOR_SESSION_ID, then generates
    a random UUID for standalone usage.

    Returns:
        Session ID string.
    """
    return (
        os.environ.get("CLAUDE_SESSION_ID")
        or os.environ.get("CURSOR_SESSION_ID")
        or str(uuid.uuid4())
    )


def _session_dir(session_id: str) -> Path:
    """Get the directory for a session's data files."""
    return Path(_SESSIONS_DIR) / session_id


def write_triage(session_id: str, result: dict[str, Any]) -> None:
    """Write triage result for a session.

    Args:
        session_id: Session identifier.
        result: Triage result dict to persist.
    """
    session_dir = _session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    data = {**result, "_timestamp": time.time()}
    triage_file = session_dir / "triage.json"
    triage_file.write_text(json.dumps(data, indent=2))
    logger.debug("Wrote triage for session %s", session_id)


def read_triage(session_id: str, max_age_seconds: float = 300.0) -> dict[str, Any] | None:
    """Read cached triage result if fresh enough.

    Args:
        session_id: Session identifier.
        max_age_seconds: Maximum age in seconds before result is stale.

    Returns:
        Triage result dict, or None if missing or stale.
    """
    triage_file = _session_dir(session_id) / "triage.json"
    if not triage_file.exists():
        return None

    try:
        data: dict[str, Any] = json.loads(triage_file.read_text())
        timestamp = data.get("_timestamp", 0)
        if time.time() - timestamp > max_age_seconds:
            logger.debug("Triage for session %s is stale", session_id)
            return None
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Failed to read triage for session %s: %s", session_id, exc)
        return None


def write_check_result(session_id: str, result: dict[str, Any]) -> None:
    """Write check runner result for a session.

    Args:
        session_id: Session identifier.
        result: Check result dict to persist.
    """
    session_dir = _session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    data = {**result, "_timestamp": time.time()}
    check_file = session_dir / "check.json"
    check_file.write_text(json.dumps(data, indent=2))
    logger.debug("Wrote check result for session %s", session_id)


def read_check_result(session_id: str) -> dict[str, Any] | None:
    """Read cached check result.

    Args:
        session_id: Session identifier.

    Returns:
        Check result dict, or None if missing.
    """
    check_file = _session_dir(session_id) / "check.json"
    if not check_file.exists():
        return None

    try:
        result: dict[str, Any] = json.loads(check_file.read_text())
        return result
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Failed to read check result for session %s: %s", session_id, exc)
        return None


def cleanup_stale(max_age_hours: float = 24.0) -> int:
    """Delete session directories older than max_age_hours.

    Args:
        max_age_hours: Maximum age in hours before a session dir is deleted.

    Returns:
        Number of session directories removed.
    """
    sessions_dir = Path(_SESSIONS_DIR)
    if not sessions_dir.is_dir():
        return 0

    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0

    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue
        try:
            # Use the newest file's mtime as the session's last activity
            newest = max(
                (f.stat().st_mtime for f in session_dir.iterdir() if f.is_file()),
                default=0,
            )
            if newest < cutoff:
                shutil.rmtree(session_dir)
                removed += 1
        except OSError:
            continue

    if removed:
        logger.info("Cleaned up %d stale session(s)", removed)
    return removed
