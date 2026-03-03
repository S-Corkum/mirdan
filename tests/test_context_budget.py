"""Tests for context budget awareness features."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from mirdan.core.environment_detector import EnvironmentInfo, _detect_context_budget, detect_environment
from mirdan.core.output_formatter import OutputFormatter
from mirdan.core.session_manager import SessionManager
from mirdan.models import CompactState, Intent, TaskType


class TestContextBudgetDetection:
    """Tests for detecting context budget from environment."""

    def test_no_budget_by_default(self) -> None:
        """Should return None when no budget env vars are set."""
        with patch.dict(os.environ, {}, clear=True):
            budget = _detect_context_budget()
            assert budget is None

    def test_mirdan_context_budget_env(self) -> None:
        """Should detect MIRDAN_CONTEXT_BUDGET env var."""
        with patch.dict(os.environ, {"MIRDAN_CONTEXT_BUDGET": "50000"}):
            budget = _detect_context_budget()
            assert budget == 50000

    def test_claude_context_remaining_env(self) -> None:
        """Should detect CLAUDE_CONTEXT_REMAINING env var."""
        with patch.dict(os.environ, {"CLAUDE_CONTEXT_REMAINING": "100000"}):
            budget = _detect_context_budget()
            assert budget == 100000

    def test_context_budget_env(self) -> None:
        """Should detect CONTEXT_BUDGET env var."""
        with patch.dict(os.environ, {"CONTEXT_BUDGET": "75000"}):
            budget = _detect_context_budget()
            assert budget == 75000

    def test_invalid_budget_ignored(self) -> None:
        """Should return None for non-numeric values."""
        with patch.dict(os.environ, {"MIRDAN_CONTEXT_BUDGET": "not_a_number"}):
            budget = _detect_context_budget()
            assert budget is None

    def test_zero_budget_ignored(self) -> None:
        """Should return None for zero budget."""
        with patch.dict(os.environ, {"MIRDAN_CONTEXT_BUDGET": "0"}):
            budget = _detect_context_budget()
            assert budget is None

    def test_negative_budget_ignored(self) -> None:
        """Should return None for negative budget."""
        with patch.dict(os.environ, {"MIRDAN_CONTEXT_BUDGET": "-1000"}):
            budget = _detect_context_budget()
            assert budget is None

    def test_priority_order(self) -> None:
        """MIRDAN_CONTEXT_BUDGET should take priority."""
        with patch.dict(os.environ, {
            "MIRDAN_CONTEXT_BUDGET": "10000",
            "CLAUDE_CONTEXT_REMAINING": "20000",
        }):
            budget = _detect_context_budget()
            assert budget == 10000


class TestEnvironmentInfoBudget:
    """Tests for budget in EnvironmentInfo."""

    def test_budget_in_to_dict(self) -> None:
        """Budget should appear in to_dict when set."""
        info = EnvironmentInfo(context_budget=50000)
        d = info.to_dict()
        assert d["context_budget"] == 50000

    def test_no_budget_in_to_dict(self) -> None:
        """Budget should not appear in to_dict when None."""
        info = EnvironmentInfo()
        d = info.to_dict()
        assert "context_budget" not in d

    def test_detect_environment_includes_budget(self) -> None:
        """detect_environment() should include budget when available."""
        with patch.dict(os.environ, {
            "CLAUDE_CODE_RUNNING": "1",
            "MIRDAN_CONTEXT_BUDGET": "80000",
        }):
            env_info = detect_environment()
            assert env_info.context_budget == 80000


class TestOutputFormatterCompaction:
    """Tests for compaction formatting in OutputFormatter."""

    @pytest.fixture
    def formatter(self) -> OutputFormatter:
        """Create an OutputFormatter instance."""
        return OutputFormatter()

    def test_format_for_compaction(self, formatter: OutputFormatter) -> None:
        """Should produce minimal state for compaction."""
        data = {
            "session_id": "abc123",
            "task_type": "generation",
            "language": "python",
            "touches_security": True,
            "score": 0.85,
            "violations_count": {"error": 2, "warning": 1},
            "frameworks": ["fastapi"],
        }
        result = formatter.format_for_compaction(data)
        assert "mirdan_compact_state" in result
        state = result["mirdan_compact_state"]
        assert state["session_id"] == "abc123"
        assert state["language"] == "python"
        assert state["touches_security"] is True
        assert state["last_score"] == 0.85
        assert state["open_violations"] == 2

    def test_format_quality_context_basic(self, formatter: OutputFormatter) -> None:
        """Should produce quality context for injection."""
        session_data = {
            "task_type": "generation",
            "language": "python",
            "touches_security": False,
            "frameworks": ["fastapi"],
        }
        result = formatter.format_quality_context(session_data)
        assert "mirdan_quality_context" in result
        ctx = result["mirdan_quality_context"]
        assert ctx["language"] == "python"
        assert ctx["frameworks"] == ["fastapi"]

    def test_format_quality_context_with_validation(self, formatter: OutputFormatter) -> None:
        """Should include validation data when provided."""
        session_data = {"task_type": "generation", "language": "python"}
        validation_data = {"passed": True, "score": 0.95, "summary": "All checks passed"}
        result = formatter.format_quality_context(session_data, validation_data)
        ctx = result["mirdan_quality_context"]
        assert "last_validation" in ctx
        assert ctx["last_validation"]["score"] == 0.95

    def test_format_quality_context_tight_budget(self, formatter: OutputFormatter) -> None:
        """With tight budget, should compress validation data."""
        session_data = {"task_type": "generation", "language": "python"}
        validation_data = {"passed": True, "score": 0.95, "summary": "All checks passed"}
        result = formatter.format_quality_context(session_data, validation_data, budget=300)
        ctx = result["mirdan_quality_context"]
        assert "last_score" in ctx
        assert "last_validation" not in ctx


class TestSessionManagerSerialize:
    """Tests for session serialization/restoration."""

    @pytest.fixture
    def manager(self) -> SessionManager:
        """Create a SessionManager instance."""
        return SessionManager()

    def test_serialize_session(self, manager: SessionManager) -> None:
        """Should serialize a session to dict."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=["fastapi"],
            touches_security=True,
        )
        session = manager.create_from_intent(intent)
        serialized = manager.serialize(session.session_id)

        assert serialized["session_id"] == session.session_id
        assert serialized["task_type"] == "generation"
        assert serialized["detected_language"] == "python"
        assert serialized["touches_security"] is True
        assert "fastapi" in serialized["frameworks"]

    def test_serialize_missing_session(self, manager: SessionManager) -> None:
        """Should return empty dict for missing session."""
        result = manager.serialize("nonexistent")
        assert result == {}

    def test_restore_session(self, manager: SessionManager) -> None:
        """Should restore a session from serialized data."""
        data = {
            "session_id": "restored123",
            "task_type": "debug",
            "detected_language": "typescript",
            "frameworks": ["react"],
            "touches_security": False,
        }
        session = manager.restore(data)
        assert session is not None
        assert session.session_id == "restored123"
        assert session.task_type == TaskType.DEBUG
        assert session.detected_language == "typescript"
        assert session.touches_security is False

    def test_restore_empty_data(self, manager: SessionManager) -> None:
        """Should return None for empty data."""
        result = manager.restore({})
        assert result is None

    def test_restore_no_session_id(self, manager: SessionManager) -> None:
        """Should return None when session_id is missing."""
        result = manager.restore({"task_type": "debug"})
        assert result is None

    def test_restore_invalid_task_type(self, manager: SessionManager) -> None:
        """Should handle invalid task type gracefully."""
        data = {
            "session_id": "test123",
            "task_type": "invalid_type",
        }
        session = manager.restore(data)
        assert session is not None
        assert session.task_type == TaskType.UNKNOWN

    def test_roundtrip(self, manager: SessionManager) -> None:
        """Serialize then restore should preserve essential fields."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.REFACTOR,
            primary_language="go",
            frameworks=["gin"],
            touches_security=True,
        )
        original = manager.create_from_intent(intent)
        serialized = manager.serialize(original.session_id)
        restored = manager.restore(serialized)

        assert restored is not None
        assert restored.session_id == original.session_id
        assert restored.task_type == original.task_type
        assert restored.detected_language == original.detected_language
        assert restored.touches_security == original.touches_security

    def test_restored_session_is_accessible(self, manager: SessionManager) -> None:
        """Restored sessions should be accessible via get()."""
        data = {
            "session_id": "get_test",
            "task_type": "generation",
            "detected_language": "python",
        }
        manager.restore(data)
        session = manager.get("get_test")
        assert session is not None
        assert session.detected_language == "python"
