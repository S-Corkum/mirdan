"""Tests for the self-managing integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from mirdan.cli.detect import DetectedProject
from mirdan.integrations.self_managing import SelfManagingIntegration
from mirdan.models import CompactState


@pytest.fixture
def detected() -> DetectedProject:
    """Create a detected project fixture."""
    return DetectedProject(
        project_name="test-project",
        project_type="application",
        primary_language="python",
        frameworks=["fastapi"],
        framework_versions={"fastapi": "0.100.0"},
        detected_ides=["claude-code"],
    )


@pytest.fixture
def integration() -> SelfManagingIntegration:
    """Create a SelfManagingIntegration instance."""
    return SelfManagingIntegration()


class TestWorkflowRuleGeneration:
    """Tests for workflow rule generation."""

    def test_generates_non_empty(
        self,
        integration: SelfManagingIntegration,
        detected: DetectedProject,
    ) -> None:
        """Should generate non-empty content."""
        content = integration.generate_workflow_rule(detected)
        assert isinstance(content, str)
        assert len(content) > 0

    def test_has_quality_sandwich(
        self,
        integration: SelfManagingIntegration,
        detected: DetectedProject,
    ) -> None:
        """Should include the quality sandwich workflow."""
        content = integration.generate_workflow_rule(detected)
        assert "enhance_prompt" in content
        assert "validate_code_quality" in content

    def test_has_tool_table(
        self,
        integration: SelfManagingIntegration,
        detected: DetectedProject,
    ) -> None:
        """Should include the available tools table."""
        content = integration.generate_workflow_rule(detected)
        assert "enhance_prompt" in content
        assert "validate_quick" in content

    def test_has_language(
        self,
        integration: SelfManagingIntegration,
        detected: DetectedProject,
    ) -> None:
        """Should include the detected language."""
        content = integration.generate_workflow_rule(detected)
        assert "python" in content.lower()

    def test_has_frameworks(
        self,
        integration: SelfManagingIntegration,
        detected: DetectedProject,
    ) -> None:
        """Should include detected frameworks."""
        content = integration.generate_workflow_rule(detected)
        assert "fastapi" in content.lower()

    def test_mentions_auto_fix(
        self,
        integration: SelfManagingIntegration,
        detected: DetectedProject,
    ) -> None:
        """Should mention the auto-fix capability."""
        content = integration.generate_workflow_rule(detected)
        assert "fix" in content.lower()


class TestQualityContext:
    """Tests for quality context generation."""

    def test_basic_context(self, integration: SelfManagingIntegration) -> None:
        """Should generate basic quality context."""
        content = integration.generate_quality_context()
        assert "mirdan" in content
        assert "enhance_prompt" in content
        assert "validate_code_quality" in content

    def test_with_session_data(self, integration: SelfManagingIntegration) -> None:
        """Should include session data when provided."""
        session_data = {
            "language": "python",
            "touches_security": True,
            "frameworks": ["fastapi"],
        }
        content = integration.generate_quality_context(session_data)
        assert "python" in content.lower()
        assert "Security" in content or "security" in content
        assert "fastapi" in content.lower()

    def test_without_session_data(self, integration: SelfManagingIntegration) -> None:
        """Should work without session data."""
        content = integration.generate_quality_context()
        assert "Active Session" not in content


class TestCompactionState:
    """Tests for compaction state serialization."""

    def test_generate_compaction_state(self, integration: SelfManagingIntegration) -> None:
        """Should generate compact state string."""
        state = CompactState(
            session_id="abc123",
            task_type="generation",
            language="python",
            touches_security=True,
            last_score=0.85,
            open_violations=2,
            frameworks=["fastapi"],
        )
        content = integration.generate_compaction_state(state)
        assert "abc123" in content
        assert "generation" in content
        assert "python" in content
        assert "sensitive" in content
        assert "0.85" in content
        assert "2" in content
        assert "fastapi" in content

    def test_restore_from_compaction(self, integration: SelfManagingIntegration) -> None:
        """Should restore state from compacted text."""
        state = CompactState(
            session_id="abc123",
            task_type="generation",
            language="python",
            touches_security=True,
            last_score=0.85,
            open_violations=2,
            frameworks=["fastapi"],
        )
        text = integration.generate_compaction_state(state)
        restored = integration.restore_from_compaction(text)

        assert restored["session_id"] == "abc123"
        assert restored["task_type"] == "generation"
        assert restored["language"] == "python"
        assert restored["touches_security"] is True
        assert restored["last_score"] == 0.85
        assert restored["open_violations"] == 2
        assert "fastapi" in restored["frameworks"]

    def test_restore_empty_text(self, integration: SelfManagingIntegration) -> None:
        """Should handle empty text gracefully."""
        restored = integration.restore_from_compaction("")
        assert restored == {}

    def test_restore_partial_state(self, integration: SelfManagingIntegration) -> None:
        """Should handle partial state text."""
        state = CompactState(session_id="abc123", language="python")
        text = integration.generate_compaction_state(state)
        restored = integration.restore_from_compaction(text)
        assert restored["session_id"] == "abc123"
        assert restored["language"] == "python"
        # Fields not present should not be in dict
        assert "last_score" not in restored

    def test_roundtrip_all_fields(self, integration: SelfManagingIntegration) -> None:
        """Full roundtrip should preserve all fields."""
        state = CompactState(
            session_id="xyz789",
            task_type="debug",
            language="typescript",
            touches_security=False,
            last_score=0.92,
            open_violations=0,
            frameworks=["react", "nextjs"],
        )
        text = integration.generate_compaction_state(state)
        restored = integration.restore_from_compaction(text)

        assert restored["session_id"] == "xyz789"
        assert restored["task_type"] == "debug"
        assert restored["language"] == "typescript"


class TestWriteWorkflowRule:
    """Tests for writing workflow rule to disk."""

    def test_writes_file(
        self,
        tmp_path: Path,
        integration: SelfManagingIntegration,
        detected: DetectedProject,
    ) -> None:
        """Should write mirdan-workflow.md to .claude/rules/."""
        result = integration.write_workflow_rule(tmp_path, detected)
        assert result == tmp_path / ".claude" / "rules" / "mirdan-workflow.md"
        assert result.exists()

    def test_content_is_valid(
        self,
        tmp_path: Path,
        integration: SelfManagingIntegration,
        detected: DetectedProject,
    ) -> None:
        """Written content should match generate_workflow_rule()."""
        path = integration.write_workflow_rule(tmp_path, detected)
        written = path.read_text()
        generated = integration.generate_workflow_rule(detected)
        assert written == generated

    def test_creates_directory_structure(
        self,
        tmp_path: Path,
        integration: SelfManagingIntegration,
        detected: DetectedProject,
    ) -> None:
        """Should create .claude/rules/ directory."""
        integration.write_workflow_rule(tmp_path, detected)
        assert (tmp_path / ".claude" / "rules").is_dir()


class TestCompactStateModel:
    """Tests for the CompactState dataclass."""

    def test_to_dict(self) -> None:
        """Should serialize to dict."""
        state = CompactState(session_id="abc", language="python")
        d = state.to_dict()
        assert d["session_id"] == "abc"
        assert d["language"] == "python"

    def test_from_dict(self) -> None:
        """Should deserialize from dict."""
        data = {"session_id": "abc", "language": "python", "touches_security": True}
        state = CompactState.from_dict(data)
        assert state.session_id == "abc"
        assert state.language == "python"
        assert state.touches_security is True

    def test_from_dict_defaults(self) -> None:
        """Should use defaults for missing fields."""
        state = CompactState.from_dict({})
        assert state.session_id == ""
        assert state.language == ""
        assert state.touches_security is False
        assert state.last_score is None
        assert state.open_violations == 0
        assert state.frameworks == []

    def test_roundtrip(self) -> None:
        """to_dict/from_dict should roundtrip."""
        original = CompactState(
            session_id="abc",
            task_type="generation",
            language="python",
            touches_security=True,
            last_score=0.9,
            open_violations=3,
            frameworks=["django"],
        )
        d = original.to_dict()
        restored = CompactState.from_dict(d)
        assert restored.session_id == original.session_id
        assert restored.task_type == original.task_type
        assert restored.language == original.language
        assert restored.touches_security == original.touches_security
        assert restored.last_score == original.last_score
        assert restored.open_violations == original.open_violations
        assert restored.frameworks == original.frameworks
