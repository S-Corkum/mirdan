"""Tests for TriageEngine classification."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from mirdan.core.triage import TriageEngine
from mirdan.llm.prompts.triage import TRIAGE_SCHEMA, build_triage_prompt
from mirdan.models import Intent, TaskClassification, TaskType, TriageResult


def _make_intent(
    task_type: TaskType = TaskType.GENERATION,
    touches_security: bool = False,
    ambiguity_score: float = 0.0,
) -> Intent:
    """Create a minimal Intent for testing."""
    return Intent(
        original_prompt="test",
        task_type=task_type,
        touches_security=touches_security,
        ambiguity_score=ambiguity_score,
    )


# ---------------------------------------------------------------------------
# Pre-filter tests (no LLM needed)
# ---------------------------------------------------------------------------


class TestTriagePrefilter:
    """Tests for rules-based pre-filter."""

    @pytest.mark.asyncio
    async def test_security_escalates_to_paid_required(self) -> None:
        engine = TriageEngine()
        intent = _make_intent(touches_security=True)

        result = await engine.classify("implement auth", intent)

        assert result is not None
        assert result.classification == TaskClassification.PAID_REQUIRED
        assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_high_ambiguity_escalates(self) -> None:
        engine = TriageEngine()
        intent = _make_intent(ambiguity_score=0.8)

        result = await engine.classify("fix the thing", intent)

        assert result is not None
        assert result.classification == TaskClassification.PAID_REQUIRED

    @pytest.mark.asyncio
    async def test_planning_escalates(self) -> None:
        engine = TriageEngine()
        intent = _make_intent(task_type=TaskType.PLANNING)

        result = await engine.classify("plan the migration", intent)

        assert result is not None
        assert result.classification == TaskClassification.PAID_REQUIRED

    @pytest.mark.asyncio
    async def test_normal_task_passes_prefilter(self) -> None:
        """Non-security, non-ambiguous, non-planning tasks should not be pre-filtered."""
        engine = TriageEngine()  # No LLM, so returns None after passing prefilter
        intent = _make_intent()

        result = await engine.classify("add docstrings", intent)

        assert result is None  # No LLM available, prefilter didn't catch it


# ---------------------------------------------------------------------------
# LLM classification tests
# ---------------------------------------------------------------------------


class TestTriageLLMClassification:
    """Tests for LLM-based classification."""

    @pytest.mark.asyncio
    async def test_classifies_local_only(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = {
            "classification": "local_only",
            "confidence": 0.95,
            "reasoning": "Simple import removal",
        }

        engine = TriageEngine(llm_manager=mock_llm)
        result = await engine.classify("fix unused import", _make_intent())

        assert result is not None
        assert result.classification == TaskClassification.LOCAL_ONLY
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_classifies_local_assist(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = {
            "classification": "local_assist",
            "confidence": 0.85,
            "reasoning": "Docstring addition",
        }

        engine = TriageEngine(llm_manager=mock_llm)
        result = await engine.classify("add docstrings", _make_intent())

        assert result is not None
        assert result.classification == TaskClassification.LOCAL_ASSIST

    @pytest.mark.asyncio
    async def test_classifies_paid_minimal(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = {
            "classification": "paid_minimal",
            "confidence": 0.80,
            "reasoning": "Standard CRUD endpoint",
        }

        engine = TriageEngine(llm_manager=mock_llm)
        result = await engine.classify("add GET /users", _make_intent())

        assert result is not None
        assert result.classification == TaskClassification.PAID_MINIMAL

    @pytest.mark.asyncio
    async def test_classifies_paid_required(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = {
            "classification": "paid_required",
            "confidence": 0.90,
            "reasoning": "Complex security feature",
        }

        engine = TriageEngine(llm_manager=mock_llm)
        result = await engine.classify("implement JWT auth", _make_intent())

        assert result is not None
        assert result.classification == TaskClassification.PAID_REQUIRED


# ---------------------------------------------------------------------------
# Low confidence escalation
# ---------------------------------------------------------------------------


class TestTriageLowConfidence:
    """Tests for low confidence escalation."""

    @pytest.mark.asyncio
    async def test_low_confidence_escalates(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = {
            "classification": "local_only",
            "confidence": 0.5,
            "reasoning": "Uncertain about scope",
        }

        engine = TriageEngine(llm_manager=mock_llm)
        result = await engine.classify("fix the bug", _make_intent())

        assert result is not None
        assert result.classification == TaskClassification.PAID_REQUIRED
        assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_confidence_at_threshold_passes(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = {
            "classification": "local_only",
            "confidence": 0.7,
            "reasoning": "Just above threshold",
        }

        engine = TriageEngine(llm_manager=mock_llm)
        result = await engine.classify("rename variable", _make_intent())

        assert result is not None
        assert result.classification == TaskClassification.LOCAL_ONLY


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestTriageGraceful:
    """Tests for graceful degradation."""

    @pytest.mark.asyncio
    async def test_returns_none_when_llm_unavailable(self) -> None:
        engine = TriageEngine(llm_manager=None)
        result = await engine.classify("test prompt", _make_intent())
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_triage_disabled(self) -> None:
        from mirdan.config import LLMConfig

        mock_llm = AsyncMock()
        config = LLMConfig(triage=False)
        engine = TriageEngine(llm_manager=mock_llm, config=config)

        result = await engine.classify("test", _make_intent())
        assert result is None
        mock_llm.generate_structured.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_none_when_llm_returns_none(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = None

        engine = TriageEngine(llm_manager=mock_llm)
        result = await engine.classify("test", _make_intent())
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_invalid_classification(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = {
            "classification": "invalid_category",
            "confidence": 0.9,
            "reasoning": "bad",
        }

        engine = TriageEngine(llm_manager=mock_llm)
        result = await engine.classify("test", _make_intent())
        assert result is None


# ---------------------------------------------------------------------------
# Ceremony override and token budget
# ---------------------------------------------------------------------------


class TestTriageMappings:
    """Tests for ceremony override and token budget mappings."""

    def test_ceremony_overrides(self) -> None:
        assert TriageEngine.get_ceremony_override(TaskClassification.LOCAL_ONLY) == "micro"
        assert TriageEngine.get_ceremony_override(TaskClassification.LOCAL_ASSIST) == "light"
        assert TriageEngine.get_ceremony_override(TaskClassification.PAID_MINIMAL) == "standard"
        assert TriageEngine.get_ceremony_override(TaskClassification.PAID_REQUIRED) is None

    def test_token_budgets(self) -> None:
        assert TriageEngine.get_token_budget(TaskClassification.LOCAL_ONLY) == 0
        assert TriageEngine.get_token_budget(TaskClassification.LOCAL_ASSIST) == 2000
        assert TriageEngine.get_token_budget(TaskClassification.PAID_MINIMAL) == 4000
        assert TriageEngine.get_token_budget(TaskClassification.PAID_REQUIRED) == 0


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestTriagePrompt:
    """Tests for triage prompt construction."""

    def test_build_triage_prompt_includes_system(self) -> None:
        prompt = build_triage_prompt("fix unused import")
        assert "task classifier" in prompt
        assert "local_only" in prompt
        assert "paid_required" in prompt

    def test_build_triage_prompt_includes_user_input(self) -> None:
        prompt = build_triage_prompt("add JWT auth")
        assert "add JWT auth" in prompt

    def test_build_triage_prompt_includes_few_shot(self) -> None:
        prompt = build_triage_prompt("test")
        assert "unused import" in prompt  # From few-shot examples
        assert "docstrings" in prompt

    def test_schema_has_required_fields(self) -> None:
        assert "classification" in TRIAGE_SCHEMA["required"]
        assert "confidence" in TRIAGE_SCHEMA["required"]
        assert "reasoning" in TRIAGE_SCHEMA["required"]
