"""Tests for the active orchestrator module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mirdan.core.active_orchestrator import ToolExecutor
from mirdan.models import MCPToolResult, ToolRecommendation


def _mock_registry(configured_mcps: set[str] | None = None) -> MagicMock:
    """Create a mock MCPClientRegistry."""
    registry = MagicMock()
    configured = configured_mcps or set()

    registry.is_configured.side_effect = lambda name: name in configured
    registry.get_capabilities.return_value = None
    registry.call_tools_parallel = AsyncMock(return_value=[])

    return registry


class TestRecommendationToCall:
    """Tests for converting recommendations to tool calls."""

    def test_explicit_tool_name_in_params(self) -> None:
        """Should use explicit tool_name from params."""
        registry = _mock_registry({"context7"})
        orchestrator = ToolExecutor(registry)

        rec = ToolRecommendation(
            mcp="context7",
            action="fetch docs",
            params={"tool_name": "query-docs", "query": "routing"},
        )
        call = orchestrator._recommendation_to_call(rec)
        assert call is not None
        assert call.tool_name == "query-docs"
        assert call.arguments["query"] == "routing"

    def test_infer_tool_from_action_keyword(self) -> None:
        """Should infer tool name from action keywords."""
        registry = _mock_registry({"enyal"})
        orchestrator = ToolExecutor(registry)

        rec = ToolRecommendation(
            mcp="enyal",
            action="Recall project conventions and past decisions",
        )
        call = orchestrator._recommendation_to_call(rec)
        assert call is not None
        assert call.tool_name == "enyal_recall"

    def test_infer_context7_documentation(self) -> None:
        """Should map documentation keyword to resolve-library-id."""
        registry = _mock_registry({"context7"})
        orchestrator = ToolExecutor(registry)

        rec = ToolRecommendation(
            mcp="context7",
            action="Fetch documentation for React",
        )
        call = orchestrator._recommendation_to_call(rec)
        assert call is not None
        assert call.tool_name == "resolve-library-id"

    def test_infer_github_commits(self) -> None:
        """Should map commits keyword to list_commits."""
        registry = _mock_registry({"github"})
        orchestrator = ToolExecutor(registry)

        rec = ToolRecommendation(
            mcp="github",
            action="Check recent commits for related changes",
        )
        call = orchestrator._recommendation_to_call(rec)
        assert call is not None
        assert call.tool_name == "list_commits"

    def test_unknown_action_returns_none(self) -> None:
        """Should return None when tool cannot be inferred."""
        registry = _mock_registry({"unknown-mcp"})
        orchestrator = ToolExecutor(registry)

        rec = ToolRecommendation(
            mcp="unknown-mcp",
            action="do something obscure",
        )
        call = orchestrator._recommendation_to_call(rec)
        assert call is None

    def test_fallback_to_first_discovered_tool(self) -> None:
        """Should use first discovered tool as fallback."""
        registry = _mock_registry({"custom"})

        # Mock discovered capabilities with a tool
        from mirdan.models import MCPCapabilities, MCPToolInfo

        capabilities = MCPCapabilities(
            tools=[MCPToolInfo(name="custom_tool", description="A custom tool")]
        )
        registry.get_capabilities.return_value = capabilities

        orchestrator = ToolExecutor(registry)
        rec = ToolRecommendation(
            mcp="custom",
            action="do something unusual",
        )
        call = orchestrator._recommendation_to_call(rec)
        assert call is not None
        assert call.tool_name == "custom_tool"


class TestInvokeRecommendations:
    """Tests for invoking recommendations."""

    @pytest.mark.asyncio
    async def test_invokes_configured_mcps(self) -> None:
        """Should invoke tools for configured MCPs."""
        registry = _mock_registry({"enyal"})
        registry.call_tools_parallel = AsyncMock(
            return_value=[
                MCPToolResult(
                    mcp_name="enyal",
                    tool_name="enyal_recall",
                    success=True,
                    data="conventions data",
                )
            ]
        )

        orchestrator = ToolExecutor(registry)
        recs = [
            ToolRecommendation(
                mcp="enyal",
                action="Recall conventions",
                params={"query": "conventions"},
            )
        ]

        results = await orchestrator.invoke_recommendations(recs)
        assert len(results) == 1
        assert results[0].success is True
        registry.call_tools_parallel.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_unconfigured_mcps(self) -> None:
        """Should skip recommendations for unconfigured MCPs."""
        registry = _mock_registry(set())  # Nothing configured
        orchestrator = ToolExecutor(registry)

        recs = [
            ToolRecommendation(mcp="context7", action="Fetch documentation"),
            ToolRecommendation(mcp="enyal", action="Recall conventions"),
        ]

        results = await orchestrator.invoke_recommendations(recs)
        assert results == []
        registry.call_tools_parallel.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_configured_and_unconfigured(self) -> None:
        """Should only invoke configured MCPs."""
        registry = _mock_registry({"enyal"})
        registry.call_tools_parallel = AsyncMock(
            return_value=[
                MCPToolResult(
                    mcp_name="enyal",
                    tool_name="enyal_recall",
                    success=True,
                )
            ]
        )

        orchestrator = ToolExecutor(registry)
        recs = [
            ToolRecommendation(mcp="context7", action="Fetch documentation"),
            ToolRecommendation(mcp="enyal", action="Recall conventions"),
        ]

        results = await orchestrator.invoke_recommendations(recs)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_empty_recommendations(self) -> None:
        """Should handle empty recommendation list."""
        registry = _mock_registry()
        orchestrator = ToolExecutor(registry)

        results = await orchestrator.invoke_recommendations([])
        assert results == []

    @pytest.mark.asyncio
    async def test_passes_timeout(self) -> None:
        """Should pass timeout to call_tools_parallel."""
        registry = _mock_registry({"enyal"})
        registry.call_tools_parallel = AsyncMock(return_value=[])

        orchestrator = ToolExecutor(registry)
        recs = [
            ToolRecommendation(mcp="enyal", action="Recall conventions"),
        ]

        await orchestrator.invoke_recommendations(recs, timeout=5.0)
        registry.call_tools_parallel.assert_called_once()
        _, kwargs = registry.call_tools_parallel.call_args
        assert kwargs["timeout"] == 5.0


class TestGetInvocableCount:
    """Tests for counting invocable recommendations."""

    def test_all_configured(self) -> None:
        """Should count all when all MCPs configured."""
        registry = _mock_registry({"enyal", "context7"})
        orchestrator = ToolExecutor(registry)

        recs = [
            ToolRecommendation(mcp="enyal", action="Recall conventions"),
            ToolRecommendation(mcp="context7", action="Fetch documentation"),
        ]

        assert orchestrator.get_invocable_count(recs) == 2

    def test_none_configured(self) -> None:
        """Should return 0 when no MCPs configured."""
        registry = _mock_registry(set())
        orchestrator = ToolExecutor(registry)

        recs = [
            ToolRecommendation(mcp="enyal", action="Recall conventions"),
        ]

        assert orchestrator.get_invocable_count(recs) == 0

    def test_partially_configured(self) -> None:
        """Should count only configured and inferable recommendations."""
        registry = _mock_registry({"enyal"})
        orchestrator = ToolExecutor(registry)

        recs = [
            ToolRecommendation(mcp="enyal", action="Recall conventions"),
            ToolRecommendation(mcp="context7", action="Fetch documentation"),
        ]

        assert orchestrator.get_invocable_count(recs) == 1

    def test_empty_list(self) -> None:
        """Should return 0 for empty list."""
        registry = _mock_registry()
        orchestrator = ToolExecutor(registry)
        assert orchestrator.get_invocable_count([]) == 0
