"""Research agent — BRAIN model autonomously gathers context via MCPs."""

from __future__ import annotations

import logging
from typing import Any

from mirdan.config import LLMConfig
from mirdan.llm.prompts.research import (
    MAX_ITERATIONS,
    MAX_TOKEN_BUDGET,
    SYNTHESIS_SAMPLING,
    TOOL_SELECTION_SAMPLING,
    TOOL_SELECTION_SCHEMA,
    build_synthesis_prompt,
    build_tool_selection_prompt,
)
from mirdan.models import (
    Intent,
    MCPToolCall,
    MCPToolResult,
    ModelRole,
    ResearchResult,
    ToolRecommendation,
)

logger = logging.getLogger(__name__)

# Read-only tools safe for autonomous research agent use.
# The research agent gathers context — it must NEVER modify state.
# Keyed by MCP name → frozenset of allowed tool names.
RESEARCH_SAFE_TOOLS: dict[str, frozenset[str]] = {
    "context7": frozenset({
        "resolve-library-id", "query-docs", "get-library-docs",
    }),
    "enyal": frozenset({
        "enyal_recall", "enyal_recall_by_scope", "enyal_get",
        "enyal_traverse", "enyal_impact", "enyal_edges",
        "enyal_history", "enyal_stats", "enyal_health",
        "enyal_review", "enyal_analytics",
    }),
    "sequential-thinking": frozenset({"sequentialthinking"}),
    "github": frozenset({
        "get_me", "list_issues", "search_issues", "issue_read",
        "list_pull_requests", "search_pull_requests", "pull_request_read",
        "search_code", "get_file_contents",
        "list_branches", "list_commits", "get_commit",
        "list_releases", "get_latest_release", "get_release_by_tag",
    }),
}


class ResearchAgent:
    """Autonomously gathers context by calling MCPs in an agentic loop.

    Uses the BRAIN model (31B) to select which MCP tool to call next,
    executes it via MCPClientRegistry, and synthesizes results. FULL
    profile only, experimental, off by default.
    """

    def __init__(
        self,
        llm_manager: Any = None,
        registry: Any = None,
        config: LLMConfig | None = None,
    ) -> None:
        self._llm = llm_manager
        self._registry = registry  # MCPClientRegistry
        self._config = config or LLMConfig()

    async def research(
        self,
        intent: Intent,
        tool_recommendations: list[ToolRecommendation],
    ) -> ResearchResult | None:
        """Run the agentic research loop.

        Args:
            intent: Analyzed task intent.
            tool_recommendations: Available tools from ToolAdvisor.

        Returns:
            ResearchResult with synthesis, or None if BRAIN unavailable.
        """
        if not self._llm or not self._config.research_agent:
            return None

        if not self._registry:
            return None

        # Check BRAIN availability
        if not await self._is_brain_available():
            return None

        # Build tool descriptions — only include MCPs with safe read-only tools
        tool_descriptions = [
            {"mcp": r.mcp, "name": r.action, "description": r.reason}
            for r in tool_recommendations
            if r.mcp in RESEARCH_SAFE_TOOLS
        ]

        results: list[dict[str, Any]] = []
        total_tokens = 0

        # Agentic loop
        for iteration in range(MAX_ITERATIONS):
            if total_tokens >= MAX_TOKEN_BUDGET:
                logger.info("Research agent token budget exhausted at iteration %d", iteration)
                break

            # Select next tool
            tool_call = await self._select_tool(
                intent.original_prompt, tool_descriptions, results
            )
            if tool_call is None:
                logger.info("Research agent completed after %d iterations", iteration)
                break

            # Execute tool
            tool_result = await self._execute_tool(tool_call)
            if tool_result is not None:
                results.append(tool_result)
                total_tokens += tool_result.get("tokens", 0)

        if not results:
            return None

        # Synthesize results
        synthesis = await self._synthesize(intent.original_prompt, results)

        return ResearchResult(
            synthesis=synthesis or "",
            sources=[
                {"mcp": r.get("mcp", ""), "tool": r.get("tool", ""), "summary": r.get("summary", "")}
                for r in results
            ],
            tool_calls_made=len(results),
            tokens_used=total_tokens,
        )

    async def _is_brain_available(self) -> bool:
        """Check if BRAIN model is selectable."""
        try:
            result = self._llm._selector.select(
                ModelRole.BRAIN, 30000, architecture="arm64"
            )
            return result is not None
        except Exception:
            return False

    async def _select_tool(
        self,
        task_description: str,
        tool_descriptions: list[dict[str, str]],
        previous_results: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Ask BRAIN to select the next tool to call.

        Args:
            task_description: Developer's task.
            tool_descriptions: Available tools.
            previous_results: Results so far.

        Returns:
            Tool call dict with mcp/name/arguments, or None if done.
        """
        prompt = build_tool_selection_prompt(
            task_description, tool_descriptions, previous_results
        )

        try:
            result = await self._llm.generate_structured(
                ModelRole.BRAIN, prompt, TOOL_SELECTION_SCHEMA, **TOOL_SELECTION_SAMPLING
            )
            if not result:
                return None

            tool = result.get("tool")
            if tool is None:
                return None  # Research complete

            # Validate tool call has required fields
            if not tool.get("mcp") or not tool.get("name"):
                return None

            selected: dict[str, Any] = tool
            return selected
        except Exception:
            logger.debug("Tool selection failed", exc_info=True)
            return None

    async def _execute_tool(self, tool_call: dict[str, Any]) -> dict[str, Any] | None:
        """Execute a single MCP tool call after allowlist validation.

        Args:
            tool_call: Dict with mcp, name, arguments.

        Returns:
            Result dict with mcp, tool, data, summary; or None on failure.
        """
        mcp_name = tool_call.get("mcp", "")
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        # SECURITY: Enforce read-only allowlist — the LLM must not modify state
        safe_tools = RESEARCH_SAFE_TOOLS.get(mcp_name)
        if safe_tools is None or tool_name not in safe_tools:
            logger.warning(
                "Research agent blocked disallowed tool: %s/%s", mcp_name, tool_name
            )
            return None

        call = MCPToolCall(mcp_name=mcp_name, tool_name=tool_name, arguments=arguments)

        try:
            results = await self._registry.call_tools_parallel([call])
            if results and results[0].success:
                data = results[0].data
                summary = str(data)[:500] if data else ""
                return {
                    "mcp": mcp_name,
                    "tool": tool_name,
                    "data": data,
                    "summary": summary,
                    "tokens": len(summary) // 4,  # Rough token estimate
                }
            elif results:
                logger.debug(
                    "Tool %s/%s failed: %s", mcp_name, tool_name, results[0].error
                )
        except Exception:
            logger.debug("Tool execution failed: %s/%s", mcp_name, tool_name, exc_info=True)

        return None

    async def _synthesize(
        self,
        task_description: str,
        results: list[dict[str, Any]],
    ) -> str | None:
        """Synthesize research results into a concise summary.

        Args:
            task_description: Developer's task.
            results: All tool results to synthesize.

        Returns:
            Synthesis text, or None on failure.
        """
        prompt = build_synthesis_prompt(task_description, results)

        try:
            response = await self._llm.generate(
                ModelRole.BRAIN, prompt, **SYNTHESIS_SAMPLING
            )
            if response and response.content:
                text: str = response.content.strip()
                return text
        except Exception:
            logger.debug("Synthesis failed", exc_info=True)

        return None
