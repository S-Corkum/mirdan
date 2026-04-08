"""Tests for ResearchAgent agentic loop."""

from __future__ import annotations

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
        mock_llm.is_role_available = MagicMock(return_value=True)

        # First call: select context7, second: select enyal, third: null (done)
        mock_llm.generate_structured.side_effect = [
            {
                "tool": {
                    "mcp": "context7",
                    "name": "get-library-docs",
                    "arguments": {"topic": "auth"},
                }
            },
            {"tool": {"mcp": "enyal", "name": "enyal_recall", "arguments": {"query": "auth"}}},
            {"tool": None},
        ]
        mock_llm.generate.return_value = LLMResponse(
            content="Authentication uses JWT with FastAPI dependency injection.",
            model="brain",
            role=ModelRole.BRAIN,
            elapsed_ms=500.0,
            tokens_used=50,
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
        mock_llm.is_role_available = MagicMock(return_value=True)

        # Always selects a tool (never null) — should stop at MAX_ITERATIONS
        mock_llm.generate_structured.return_value = {
            "tool": {"mcp": "context7", "name": "query-docs", "arguments": {}}
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
        mock_llm.is_role_available = MagicMock(return_value=True)

        mock_llm.generate_structured.side_effect = [
            {"tool": {"mcp": "context7", "name": "query-docs", "arguments": {}}},
            {"tool": {"mcp": "enyal", "name": "enyal_recall", "arguments": {}}},
            {"tool": None},
        ]
        mock_llm.generate.return_value = LLMResponse(
            content="partial synthesis",
            model="m",
            role=ModelRole.BRAIN,
            elapsed_ms=0.0,
            tokens_used=0,
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
        mock_llm.is_role_available = MagicMock(return_value=False)  # No BRAIN

        config = LLMConfig(research_agent=True)
        agent = ResearchAgent(llm_manager=mock_llm, registry=AsyncMock(), config=config)
        result = await agent.research(_make_intent(), _make_tool_recs())
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_results(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.is_role_available = MagicMock(return_value=True)
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

        prompt = build_tool_selection_prompt("add auth", [{"mcp": "ctx7", "name": "docs"}], [])
        assert "add auth" in prompt
        assert "ctx7" in prompt

    def test_synthesis_prompt_includes_results(self) -> None:
        from mirdan.llm.prompts.research import build_synthesis_prompt

        prompt = build_synthesis_prompt("task", [{"data": "found something"}])
        assert "found something" in prompt


# ---------------------------------------------------------------------------
# Security: Read-only allowlist
# ---------------------------------------------------------------------------


class TestResearchAgentAllowlist:
    """Tests for the read-only tool allowlist."""

    @pytest.mark.asyncio
    async def test_blocks_enyal_remember(self) -> None:
        """enyal_remember (write) must be blocked."""
        mock_llm = AsyncMock()
        mock_llm.is_role_available = MagicMock(return_value=True)
        mock_llm.generate_structured.side_effect = [
            {
                "tool": {
                    "mcp": "enyal",
                    "name": "enyal_remember",
                    "arguments": {"content": "malicious"},
                }
            },
            {"tool": None},
        ]
        mock_llm.generate.return_value = LLMResponse(
            content="synthesis", model="m", role=ModelRole.BRAIN, elapsed_ms=0, tokens_used=0
        )

        mock_registry = AsyncMock()
        config = LLMConfig(research_agent=True)
        agent = ResearchAgent(llm_manager=mock_llm, registry=mock_registry, config=config)
        await agent.research(_make_intent(), _make_tool_recs())

        # Should not have called the registry at all — tool was blocked
        mock_registry.call_tools_parallel.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_blocks_enyal_forget(self) -> None:
        """enyal_forget (write) must be blocked."""
        mock_llm = AsyncMock()
        mock_llm.is_role_available = MagicMock(return_value=True)
        mock_llm.generate_structured.side_effect = [
            {"tool": {"mcp": "enyal", "name": "enyal_forget", "arguments": {"entry_id": "x"}}},
            {"tool": None},
        ]

        mock_registry = AsyncMock()
        config = LLMConfig(research_agent=True)
        agent = ResearchAgent(llm_manager=mock_llm, registry=mock_registry, config=config)
        await agent.research(_make_intent(), _make_tool_recs())

        mock_registry.call_tools_parallel.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_blocks_unknown_mcp(self) -> None:
        """Unknown MCP names must be blocked entirely."""
        mock_llm = AsyncMock()
        mock_llm.is_role_available = MagicMock(return_value=True)
        mock_llm.generate_structured.side_effect = [
            {"tool": {"mcp": "malicious_server", "name": "steal_data", "arguments": {}}},
            {"tool": None},
        ]

        mock_registry = AsyncMock()
        config = LLMConfig(research_agent=True)
        agent = ResearchAgent(llm_manager=mock_llm, registry=mock_registry, config=config)
        await agent.research(_make_intent(), _make_tool_recs())

        mock_registry.call_tools_parallel.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_allows_enyal_recall(self) -> None:
        """enyal_recall (read) must be allowed."""
        mock_llm = AsyncMock()
        mock_llm.is_role_available = MagicMock(return_value=True)
        mock_llm.generate_structured.side_effect = [
            {"tool": {"mcp": "enyal", "name": "enyal_recall", "arguments": {"query": "test"}}},
            {"tool": None},
        ]
        mock_llm.generate.return_value = LLMResponse(
            content="synthesis", model="m", role=ModelRole.BRAIN, elapsed_ms=0, tokens_used=0
        )

        mock_registry = AsyncMock()
        mock_registry.call_tools_parallel.return_value = [
            MCPToolResult(mcp_name="enyal", tool_name="enyal_recall", success=True, data="found")
        ]

        config = LLMConfig(research_agent=True)
        agent = ResearchAgent(llm_manager=mock_llm, registry=mock_registry, config=config)
        await agent.research(_make_intent(), _make_tool_recs())

        mock_registry.call_tools_parallel.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_allows_context7_read(self) -> None:
        """context7/get-library-docs (read) must be allowed."""
        mock_llm = AsyncMock()
        mock_llm.is_role_available = MagicMock(return_value=True)
        mock_llm.generate_structured.side_effect = [
            {"tool": {"mcp": "context7", "name": "get-library-docs", "arguments": {}}},
            {"tool": None},
        ]
        mock_llm.generate.return_value = LLMResponse(
            content="synthesis", model="m", role=ModelRole.BRAIN, elapsed_ms=0, tokens_used=0
        )

        mock_registry = AsyncMock()
        mock_registry.call_tools_parallel.return_value = [
            MCPToolResult(
                mcp_name="context7", tool_name="get-library-docs", success=True, data="docs"
            )
        ]

        config = LLMConfig(research_agent=True)
        agent = ResearchAgent(llm_manager=mock_llm, registry=mock_registry, config=config)
        await agent.research(_make_intent(), _make_tool_recs())

        mock_registry.call_tools_parallel.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_blocks_github_write(self) -> None:
        """github/create_pull_request (write) must be blocked."""
        mock_llm = AsyncMock()
        mock_llm.is_role_available = MagicMock(return_value=True)
        mock_llm.generate_structured.side_effect = [
            {"tool": {"mcp": "github", "name": "create_pull_request", "arguments": {}}},
            {"tool": None},
        ]

        mock_registry = AsyncMock()
        config = LLMConfig(research_agent=True)
        agent = ResearchAgent(llm_manager=mock_llm, registry=mock_registry, config=config)
        await agent.research(_make_intent(), _make_tool_recs())

        mock_registry.call_tools_parallel.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_prompt_excludes_non_allowlisted_mcps(self) -> None:
        """Tool descriptions should not include MCPs not in RESEARCH_SAFE_TOOLS."""

        recs = [
            ToolRecommendation(mcp="context7", action="get-library-docs", reason="docs"),
            ToolRecommendation(mcp="unknown_mcp", action="dangerous_tool", reason="bad"),
            ToolRecommendation(mcp="enyal", action="enyal_recall", reason="memory"),
        ]

        mock_llm = AsyncMock()
        mock_llm.is_role_available = MagicMock(return_value=True)
        mock_llm.generate_structured.return_value = {"tool": None}

        config = LLMConfig(research_agent=True)
        agent = ResearchAgent(llm_manager=mock_llm, registry=AsyncMock(), config=config)
        await agent.research(_make_intent(), recs)

        # Check the prompt that was sent to generate_structured
        call_args = mock_llm.generate_structured.call_args
        prompt = call_args[0][1]  # Second positional arg is the prompt
        assert "unknown_mcp" not in prompt
        assert "dangerous_tool" not in prompt
