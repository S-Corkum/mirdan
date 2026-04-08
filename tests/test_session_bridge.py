"""Tests for session bridge file-based coordination."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

from mirdan.coordination import session_bridge


class TestGetSessionId:
    """Tests for get_session_id()."""

    def test_returns_claude_session_id(self) -> None:
        with patch.dict(os.environ, {"CLAUDE_SESSION_ID": "abc-123"}):
            assert session_bridge.get_session_id() == "abc-123"

    def test_returns_cursor_session_id(self) -> None:
        with patch.dict(os.environ, {"CURSOR_SESSION_ID": "cur-456"}, clear=False):
            env = os.environ.copy()
            env.pop("CLAUDE_SESSION_ID", None)
            with patch.dict(os.environ, env, clear=True):
                assert session_bridge.get_session_id() == "cur-456"

    def test_returns_uuid_fallback(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            sid = session_bridge.get_session_id()
            assert len(sid) == 36  # UUID format


class TestWriteReadTriage:
    """Tests for triage write/read roundtrip."""

    def test_write_and_read_roundtrip(self, tmp_path: Path) -> None:
        with patch.object(session_bridge, "_SESSIONS_DIR", str(tmp_path / "sessions")):
            data = {"classification": "local_only", "confidence": 0.9, "reasoning": "simple"}
            session_bridge.write_triage("test-session", data)

            result = session_bridge.read_triage("test-session")

            assert result is not None
            assert result["classification"] == "local_only"
            assert result["confidence"] == 0.9
            assert "_timestamp" in result

    def test_read_returns_none_when_missing(self, tmp_path: Path) -> None:
        with patch.object(session_bridge, "_SESSIONS_DIR", str(tmp_path / "sessions")):
            assert session_bridge.read_triage("nonexistent") is None

    def test_read_returns_none_when_stale(self, tmp_path: Path) -> None:
        with patch.object(session_bridge, "_SESSIONS_DIR", str(tmp_path / "sessions")):
            data = {"classification": "local_only", "confidence": 0.9, "reasoning": "old"}
            session_bridge.write_triage("stale-session", data)

            # Backdate the timestamp
            triage_file = tmp_path / "sessions" / "stale-session" / "triage.json"
            content = json.loads(triage_file.read_text())
            content["_timestamp"] = time.time() - 600  # 10 minutes ago
            triage_file.write_text(json.dumps(content))

            result = session_bridge.read_triage("stale-session", max_age_seconds=300)
            assert result is None

    def test_read_respects_max_age(self, tmp_path: Path) -> None:
        with patch.object(session_bridge, "_SESSIONS_DIR", str(tmp_path / "sessions")):
            data = {"classification": "local_only", "confidence": 0.9, "reasoning": "fresh"}
            session_bridge.write_triage("fresh-session", data)

            # With a very long max_age, it should still be fresh
            result = session_bridge.read_triage("fresh-session", max_age_seconds=3600)
            assert result is not None


class TestWriteReadCheckResult:
    """Tests for check result write/read roundtrip."""

    def test_write_and_read_roundtrip(self, tmp_path: Path) -> None:
        with patch.object(session_bridge, "_SESSIONS_DIR", str(tmp_path / "sessions")):
            data = {"all_pass": True, "summary": "all clean"}
            session_bridge.write_check_result("test-session", data)

            result = session_bridge.read_check_result("test-session")

            assert result is not None
            assert result["all_pass"] is True
            assert result["summary"] == "all clean"

    def test_read_returns_none_when_missing(self, tmp_path: Path) -> None:
        with patch.object(session_bridge, "_SESSIONS_DIR", str(tmp_path / "sessions")):
            assert session_bridge.read_check_result("nonexistent") is None


class TestCleanupStale:
    """Tests for cleanup_stale()."""

    def test_removes_old_sessions(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / "sessions"
        with patch.object(session_bridge, "_SESSIONS_DIR", str(sessions_dir)):
            # Create an old session
            old_dir = sessions_dir / "old-session"
            old_dir.mkdir(parents=True)
            old_file = old_dir / "triage.json"
            old_file.write_text("{}")
            # Backdate the file
            old_time = time.time() - (48 * 3600)  # 48 hours ago
            os.utime(old_file, (old_time, old_time))

            removed = session_bridge.cleanup_stale(max_age_hours=24)
            assert removed == 1
            assert not old_dir.exists()

    def test_keeps_recent_sessions(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / "sessions"
        with patch.object(session_bridge, "_SESSIONS_DIR", str(sessions_dir)):
            # Create a recent session
            recent_dir = sessions_dir / "recent-session"
            recent_dir.mkdir(parents=True)
            (recent_dir / "triage.json").write_text("{}")

            removed = session_bridge.cleanup_stale(max_age_hours=24)
            assert removed == 0
            assert recent_dir.exists()

    def test_handles_empty_sessions_dir(self, tmp_path: Path) -> None:
        with patch.object(session_bridge, "_SESSIONS_DIR", str(tmp_path / "nonexistent")):
            assert session_bridge.cleanup_stale() == 0
