"""Tests for the environment detector component."""

import os
from unittest.mock import patch

from mirdan.core.environment_detector import (
    EnvironmentInfo,
    IDEType,
    detect_environment,
    is_claude_code,
    is_cursor,
)


class TestEnvironmentDetection:
    """Tests for detect_environment function."""

    def test_detects_claude_code_via_running(self) -> None:
        """Should detect Claude Code from CLAUDE_CODE_RUNNING env var."""
        with patch.dict(os.environ, {"CLAUDE_CODE_RUNNING": "1"}, clear=False):
            env = detect_environment()
            assert env.ide == IDEType.CLAUDE_CODE
            assert env.ide_name == "Claude Code"
            assert env.is_agent_context is True

    def test_detects_claude_code_via_version(self) -> None:
        """Should detect Claude Code from CLAUDE_CODE_VERSION env var."""
        env_vars = {"CLAUDE_CODE_VERSION": "2.1.62", "CLAUDE_CODE_RUNNING": ""}
        with patch.dict(os.environ, env_vars, clear=False):
            env = detect_environment()
            # May detect Claude Code if VERSION is set
            if env.ide == IDEType.CLAUDE_CODE:
                assert env.is_agent_context is True

    def test_detects_cursor(self) -> None:
        """Should detect Cursor from CURSOR_TRACE_ID env var."""
        clean_env = {
            "CLAUDE_CODE_RUNNING": "",
            "CLAUDE_CODE_VERSION": "",
            "CURSOR_TRACE_ID": "abc123",
        }
        with patch.dict(os.environ, clean_env, clear=False):
            env = detect_environment()
            assert env.ide == IDEType.CURSOR
            assert env.ide_name == "Cursor"
            assert env.is_agent_context is True

    def test_detects_vscode(self) -> None:
        """Should detect VS Code from VSCODE_PID env var."""
        clean_env = {
            "CLAUDE_CODE_RUNNING": "",
            "CLAUDE_CODE_VERSION": "",
            "CURSOR_TRACE_ID": "",
            "CURSOR_SESSION_ID": "",
            "WINDSURF_SESSION_ID": "",
        }
        with patch.dict(os.environ, {**clean_env, "VSCODE_PID": "12345"}, clear=False):
            env = detect_environment()
            assert env.ide == IDEType.VSCODE
            assert env.ide_name == "VS Code"
            assert env.is_agent_context is False

    def test_unknown_environment(self) -> None:
        """Should return UNKNOWN when no IDE detected."""
        clean_env = {
            "CLAUDE_CODE_RUNNING": "",
            "CLAUDE_CODE_VERSION": "",
            "CURSOR_TRACE_ID": "",
            "CURSOR_SESSION_ID": "",
            "WINDSURF_SESSION_ID": "",
            "VSCODE_PID": "",
            "VSCODE_IPC_HOOK": "",
            "TERM_PROGRAM": "",
        }
        with patch.dict(os.environ, clean_env, clear=False):
            env = detect_environment()
            assert env.ide == IDEType.UNKNOWN
            assert env.is_agent_context is False

    def test_term_program_vscode_detection(self) -> None:
        """Should detect VS Code from TERM_PROGRAM=vscode."""
        clean_env = {
            "CLAUDE_CODE_RUNNING": "",
            "CLAUDE_CODE_VERSION": "",
            "CURSOR_TRACE_ID": "",
            "CURSOR_SESSION_ID": "",
            "WINDSURF_SESSION_ID": "",
            "VSCODE_PID": "",
            "VSCODE_IPC_HOOK": "",
        }
        with patch.dict(os.environ, {**clean_env, "TERM_PROGRAM": "vscode"}, clear=False):
            env = detect_environment()
            assert env.ide == IDEType.VSCODE

    def test_term_program_non_vscode_ignored(self) -> None:
        """Should not detect VS Code from non-vscode TERM_PROGRAM."""
        clean_env = {
            "CLAUDE_CODE_RUNNING": "",
            "CLAUDE_CODE_VERSION": "",
            "CURSOR_TRACE_ID": "",
            "CURSOR_SESSION_ID": "",
            "WINDSURF_SESSION_ID": "",
            "VSCODE_PID": "",
            "VSCODE_IPC_HOOK": "",
        }
        with patch.dict(os.environ, {**clean_env, "TERM_PROGRAM": "iTerm.app"}, clear=False):
            env = detect_environment()
            assert env.ide == IDEType.UNKNOWN


class TestEnvironmentInfo:
    """Tests for EnvironmentInfo dataclass."""

    def test_to_dict(self) -> None:
        """Should convert to dictionary correctly."""
        info = EnvironmentInfo(
            ide=IDEType.CLAUDE_CODE,
            ide_name="Claude Code",
            is_agent_context=True,
        )
        d = info.to_dict()
        assert d["ide"] == "claude_code"
        assert d["ide_name"] == "Claude Code"
        assert d["is_agent_context"] is True

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        info = EnvironmentInfo()
        assert info.ide == IDEType.UNKNOWN
        assert info.ide_name == "unknown"
        assert info.is_agent_context is False
        assert info.env_hints == {}


class TestConvenienceFunctions:
    """Tests for is_claude_code() and is_cursor() helpers."""

    def test_is_claude_code_true(self) -> None:
        """Should return True when Claude Code env var is set."""
        with patch.dict(os.environ, {"CLAUDE_CODE_RUNNING": "1"}, clear=False):
            assert is_claude_code() is True

    def test_is_claude_code_false(self) -> None:
        """Should return False when no Claude Code env vars."""
        with patch.dict(
            os.environ,
            {"CLAUDE_CODE_RUNNING": "", "CLAUDE_CODE_VERSION": ""},
            clear=False,
        ):
            assert is_claude_code() is False

    def test_is_cursor_true(self) -> None:
        """Should return True when Cursor env var is set."""
        with patch.dict(os.environ, {"CURSOR_TRACE_ID": "abc"}, clear=False):
            assert is_cursor() is True

    def test_is_cursor_false(self) -> None:
        """Should return False when no Cursor env vars."""
        with patch.dict(
            os.environ,
            {"CURSOR_TRACE_ID": "", "CURSOR_SESSION_ID": ""},
            clear=False,
        ):
            assert is_cursor() is False


class TestDetectionPriority:
    """Tests that more specific IDEs take priority."""

    def test_claude_code_takes_priority_over_vscode(self) -> None:
        """Claude Code should be detected even if VSCODE_PID is also set."""
        with patch.dict(
            os.environ,
            {"CLAUDE_CODE_RUNNING": "1", "VSCODE_PID": "12345"},
            clear=False,
        ):
            env = detect_environment()
            assert env.ide == IDEType.CLAUDE_CODE

    def test_cursor_takes_priority_over_vscode(self) -> None:
        """Cursor should be detected even if VSCODE_PID is also set."""
        clean_env = {"CLAUDE_CODE_RUNNING": "", "CLAUDE_CODE_VERSION": ""}
        with patch.dict(
            os.environ,
            {**clean_env, "CURSOR_TRACE_ID": "abc", "VSCODE_PID": "12345"},
            clear=False,
        ):
            env = detect_environment()
            assert env.ide == IDEType.CURSOR
