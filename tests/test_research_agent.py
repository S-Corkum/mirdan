"""Tests for ResearchAgent agentic loop."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mirdan.config import LLMConfig
from mirdan.core.research_agent import ResearchAgent
from mirdan.models import (
    Intent,
    LLMResponse,
    MCPToolResult,
    ModelRole,
    TaskType,
    ToolRecommendation,
)


def _make_intent(prompt: str = "add auth endpoint") -> Intent:
    return Intent(original_prompt=prompt, task_type=TaskType.GENERATION)


def _make_tool_recs() -> list[ToolRecommendation]:
    return [
        ToolRecommendation(mcp="context7", action="get-library-docs", reason="get API docs"),
        ToolRecommendation(mcp="enyal", action="enyal_recall", reason="check conventions"),
    ]


# ---------------------------------------------------------------------------
# Core research loop
# ---------------------------------------------------------------------------


class TestResearchAgentLoop:
    """Tests for the agentic research loop."""

    @pytest.mark.asyncio
    async def test_selects_tools_and_synthesizes(self) -> None:
        mock_llm = AsyncMock()
        mock_llm._selector = MagicMock()
        mock_llm._selector.select.return_value = MagicMock()

        # First call: select context7, second: select enyal, third: null (done)
        mock_llm.generate_structured.side_effect = [
            {"tool": {"mcp": "context7", "name": "get-library-docs", "arguments": {"topic": "auth"}}},
            {"tool": {"mcp": "enyal", "name": "enyal_recall", "arguments": {"query": "auth"}}},
            {"tool": None},
        ]
        mock_llm.generate.return_value = LLMResponse(
            content="Authentication uses JWT with FastAPI dependency injection.",
            model="brain", role=ModelRole.BRAIN, elapsed_ms=500.0, tokens_used=50,
        )

        mock_registry = AsyncMock()
        mock_registry.call_tools_parallel.return_value = [
            MCPToolResult(mcp_name="test", tool_name="test", success=True, data="docs result")
        ]

        config = LLMConfig(research_agent=True)
        agent = ResearchAgent(llm_manager=mock_llm, registry=mock_registry, config=config)
        result = await agent.research(_make_intent(), _make_tool_recs())

        assert result is not None
        assert result.tool_calls_made == 2
        assert "JWT" in result.synthesis

    @pytest.mark.asyncio
    async def test_max_iteration_limit(self) -> None:
        mock_llm = AsyncMock()
        mock_llm._selector = MagicMock()
        mock_llm._selector.select.return_value = MagicMock()

        # Always selects a tool (never null) — should stop at MAX_ITERATIONS
        mock_llm.generate_structured.return_value = {
            "tool": {"mcp": "context7", "name": "get-docs", "arguments": {}}
        }
        mock_llm.generate.return_value = LLMResponse(
            content="synthesis", model="m", role=ModelRole.BRAIN, elapsed_ms=0.0, tokens_used=0
        )

        mock_registry = AsyncMock()
        mock_registry.call_tools_parallel.return_value = [
            MCPToolResult(mcp_name="test", tool_name="test", success=True, data="data")
        ]

        config = LLMConfig(research_agent=True)
        agent = ResearchAgent(llm_manager=mock_llm, registry=mock_registry, config=config)
        result = await agent.research(_make_intent(), _make_tool_recs())

        assert result is not None
        assert result.tool_calls_made == 5  # MAX_ITERATIONS

    @pytest.mark.asyncio
    async def test_tool_execution_failure_continues(self) -> None:
        mock_llm = AsyncMock()
        mock_llm._selector = MagicMock()
        mock_llm._selector.select.return_value = MagicMock()

        mock_llm.generate_structured.side_effect = [
            {"tool": {"mcp": "context7", "name": "fail-tool", "arguments": {}}},
            {"tool": {"mcp": "enyal", "name": "ok-tool", "arguments": {}}},
            {"tool": None},
        ]
        mock_llm.generate.return_value = LLMResponse(
            content="partial synthesis", model="m", role=ModelRole.BRAIN, elapsed_ms=0.0, tokens_used=0
        )

        mock_registry = AsyncMock()
        # First call fails, second succeeds
        mock_registry.call_tools_parallel.side_effect = [
            [MCPToolResult(mcp_name="ctx", tool_name="fail", success=False, error="timeout")],
            [MCPToolResult(mcp_name="enyal", tool_name="ok", success=True, data="result")],
        ]

        config = LLMConfig(research_agent=True)
        agent = ResearchAgent(llm_manager=mock_llm, registry=mock_registry, config=config)
        result = await agent.research(_make_intent(), _make_tool_recs())

        assert result is not None
        assert result.tool_calls_made == 1  # Only the successful one counted


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestResearchAgentGraceful:
    """Tests for graceful degradation."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_llm(self) -> None:
        agent = ResearchAgent(llm_manager=None)
        result = await agent.research(_make_intent(), _make_tool_recs())
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self) -> None:
        config = LLMConfig(research_agent=False)
        agent = ResearchAgent(llm_manager=AsyncMock(), config=config)
        result = await agent.research(_make_intent(), _make_tool_recs())
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_registry(self) -> None:
        config = LLMConfig(research_agent=True)
        agent = ResearchAgent(llm_manager=AsyncMock(), registry=None, config=config)
        result = await agent.research(_make_intent(), _make_tool_recs())
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_brain_unavailable(self) -> None:
        mock_llm = AsyncMock()
        mock_llm._selector = MagicMock()
        mock_llm._selector.select.return_value = None  # No BRAIN

        config = LLMConfig(research_agent=True)
        agent = ResearchAgent(llm_manager=mock_llm, registry=AsyncMock(), config=config)
        result = await agent.research(_make_intent(), _make_tool_recs())
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_results(self) -> None:
        mock_llm = AsyncMock()
        mock_llm._selector = MagicMock()
        mock_llm._selector.select.return_value = MagicMock()
        mock_llm.generate_structured.return_value = {"tool": None}  # Immediately done

        config = LLMConfig(research_agent=True)
        agent = ResearchAgent(llm_manager=mock_llm, registry=AsyncMock(), config=config)
        result = await agent.research(_make_intent(), _make_tool_recs())
        assert result is None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestResearchPrompts:
    """Tests for research prompt templates."""

    def test_tool_selection_prompt_includes_task(self) -> None:
        from mirdan.llm.prompts.research import build_tool_selection_prompt

        prompt = build_tool_selection_prompt(
            "add auth", [{"mcp": "ctx7", "name": "docs"}], []
        )
        assert "add auth" in prompt
        assert "ctx7" in prompt

    def test_synthesis_prompt_includes_results(self) -> None:
        from mirdan.llm.prompts.research import build_synthesis_prompt

        prompt = build_synthesis_prompt("task", [{"data": "found something"}])
        assert "found something" in prompt
