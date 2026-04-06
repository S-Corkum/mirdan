"""Tests for PromptOptimizer (BRAIN model, FULL profile only)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirdan.config import LLMConfig
from mirdan.core.prompt_optimizer import PromptOptimizer
from mirdan.llm.prompts.optimization import (
    TARGET_MODEL_PROFILES,
    build_optimization_prompt,
    build_pruning_prompt,
)
from mirdan.models import LLMResponse, ModelRole


# ---------------------------------------------------------------------------
# optimize()
# ---------------------------------------------------------------------------


class TestPromptOptimizerOptimize:
    """Tests for PromptOptimizer.optimize()."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_llm(self) -> None:
        optimizer = PromptOptimizer(llm_manager=None)
        result = await optimizer.optimize("task", [], "", "", "sonnet")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self) -> None:
        config = LLMConfig(prompt_optimization=False)
        optimizer = PromptOptimizer(llm_manager=AsyncMock(), config=config)
        result = await optimizer.optimize("task", [], "", "", "sonnet")
        assert result is None

    @pytest.mark.asyncio
    async def test_fast_model_does_full_optimization(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.is_role_available = MagicMock(return_value=False)  # No BRAIN
        # FAST tier: pruning via generate_structured
        mock_llm.generate_structured.return_value = {
            "kept": [{"item": "context", "score": 0.8}],
            "pruned": [],
        }
        # FAST tier: LLM-powered prompt generation (not just assembly)
        mock_llm.generate.return_value = LLMResponse(
            content="Optimized by FAST model with thinking",
            model="gemma4-e4b", role=ModelRole.FAST,
            elapsed_ms=200.0, tokens_used=50,
        )

        optimizer = PromptOptimizer(llm_manager=mock_llm)
        result = await optimizer.optimize("task", ["context"], "", "", "sonnet")

        assert result is not None
        assert result.text == "Optimized by FAST model with thinking"
        assert "fast" in result.optimization_notes[1].lower()

    @pytest.mark.asyncio
    async def test_falls_back_to_assembly_when_llm_fails(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.is_role_available = MagicMock(return_value=False)  # No BRAIN
        mock_llm.generate_structured.return_value = {
            "kept": [{"item": "context", "score": 0.8}],
            "pruned": [],
        }
        mock_llm.generate.return_value = None  # LLM generation fails

        optimizer = PromptOptimizer(llm_manager=mock_llm)
        result = await optimizer.optimize("task", ["context"], "", "", "sonnet")

        assert result is not None  # Falls back to deterministic assembly
        assert "Task: task" in result.text

    @pytest.mark.asyncio
    async def test_returns_optimized_prompt(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.is_role_available = MagicMock(return_value=True)

        # Pruning returns kept items
        mock_llm.generate_structured.return_value = {
            "kept": [{"item": "important context", "score": 0.9, "reason": "relevant"}],
            "pruned": [{"item": "irrelevant", "score": 0.1, "reason": "not needed"}],
        }
        # Optimization returns prompt text
        mock_llm.generate.return_value = LLMResponse(
            content="Optimized prompt for Sonnet",
            model="gemma4-31b",
            role=ModelRole.BRAIN,
            elapsed_ms=500.0,
            tokens_used=100,
        )

        optimizer = PromptOptimizer(llm_manager=mock_llm)
        result = await optimizer.optimize(
            "add auth endpoint",
            ["important context", "irrelevant"],
            "use enhance_prompt",
            "security required",
            "sonnet",
        )

        assert result is not None
        assert result.text == "Optimized prompt for Sonnet"
        assert result.context_pruned == 1
        assert result.target_model == "sonnet"

    @pytest.mark.asyncio
    async def test_opus_targeting(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.is_role_available = MagicMock(return_value=True)
        mock_llm.generate_structured.return_value = {"kept": [], "pruned": []}
        mock_llm.generate.return_value = LLMResponse(
            content="Concise Opus prompt", model="m", role=ModelRole.BRAIN,
            elapsed_ms=0.0, tokens_used=0,
        )

        optimizer = PromptOptimizer(llm_manager=mock_llm)
        result = await optimizer.optimize("task", [], "", "", "opus")

        assert result is not None
        assert result.target_model == "opus"
        assert "Opus" in result.optimization_notes[0]

    @pytest.mark.asyncio
    async def test_haiku_targeting(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.is_role_available = MagicMock(return_value=True)
        mock_llm.generate_structured.return_value = {"kept": [], "pruned": []}
        mock_llm.generate.return_value = LLMResponse(
            content="Step-by-step Haiku prompt", model="m", role=ModelRole.BRAIN,
            elapsed_ms=0.0, tokens_used=0,
        )

        optimizer = PromptOptimizer(llm_manager=mock_llm)
        result = await optimizer.optimize("task", [], "", "", "haiku")

        assert result is not None
        assert result.target_model == "haiku"

    @pytest.mark.asyncio
    async def test_cursor_ide_aggressive_pruning(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.is_role_available = MagicMock(return_value=True)
        mock_llm.generate_structured.return_value = {
            "kept": [{"item": "essential", "score": 0.9, "reason": "needed"}],
            "pruned": [
                {"item": "a", "score": 0.2, "reason": "x"},
                {"item": "b", "score": 0.1, "reason": "x"},
            ],
        }
        mock_llm.generate.return_value = LLMResponse(
            content="Cursor-optimized", model="m", role=ModelRole.BRAIN,
            elapsed_ms=0.0, tokens_used=0,
        )

        optimizer = PromptOptimizer(llm_manager=mock_llm)
        result = await optimizer.optimize(
            "task", ["essential", "a", "b"], "", "", "sonnet", detected_ide="cursor"
        )

        assert result is not None
        assert any("aggressive" in note.lower() for note in result.optimization_notes)

    @pytest.mark.asyncio
    async def test_graceful_fallback_on_generate_failure(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.is_role_available = MagicMock(return_value=True)
        mock_llm.generate_structured.return_value = {"kept": [], "pruned": []}
        mock_llm.generate.return_value = None  # LLM generation fails

        optimizer = PromptOptimizer(llm_manager=mock_llm)
        result = await optimizer.optimize("task", [], "", "", "sonnet")

        # Falls back to deterministic assembly instead of returning None
        assert result is not None
        assert "Task: task" in result.text


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestOptimizationPrompts:
    """Tests for prompt template construction."""

    def test_target_model_profiles_exist(self) -> None:
        assert "opus" in TARGET_MODEL_PROFILES
        assert "sonnet" in TARGET_MODEL_PROFILES
        assert "haiku" in TARGET_MODEL_PROFILES

    def test_pruning_prompt_includes_cursor_instruction(self) -> None:
        prompt = build_pruning_prompt("task", ["ctx"], "sonnet", is_cursor=True)
        assert "Cursor" in prompt
        assert "aggressive" in prompt.lower()

    def test_pruning_prompt_no_cursor(self) -> None:
        prompt = build_pruning_prompt("task", ["ctx"], "sonnet", is_cursor=False)
        assert "Cursor" not in prompt

    def test_optimization_prompt_includes_model_name(self) -> None:
        prompt = build_optimization_prompt("task", "ctx", "tools", "reqs", "opus")
        assert "Opus" in prompt

    def test_optimization_prompt_haiku_step_by_step(self) -> None:
        prompt = build_optimization_prompt("task", "ctx", "tools", "reqs", "haiku")
        assert "step-by-step" in prompt.lower()
