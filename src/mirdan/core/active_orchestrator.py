"""Active MCP Orchestrator — translates recommendations into actual tool calls.

When `auto_invoke=True`, this module converts ToolRecommendations into
MCPToolCalls and executes them via the MCPClientRegistry. This transforms
mirdan from a passive advisor into an active orchestration layer.
"""

from __future__ import annotations

import logging
from typing import Any

from mirdan.core.client_registry import MCPClientRegistry
from mirdan.models import MCPToolCall, MCPToolResult, ToolRecommendation

logger = logging.getLogger(__name__)

# Maps MCP names to their typical tool-name patterns for common actions
_ACTION_TO_TOOL: dict[str, dict[str, str]] = {
    "context7": {
        "documentation": "resolve-library-id",
        "fetch": "resolve-library-id",
        "query": "get-library-docs",
    },
    "enyal": {
        "recall": "enyal_recall",
        "remember": "enyal_remember",
        "conventions": "enyal_recall",
    },
    "github": {
        "commits": "list_commits",
        "issues": "list_issues",
        "pull_requests": "list_pull_requests",
        "pr": "pull_request_read",
    },
    "filesystem": {
        "search": "search_files",
        "read": "read_file",
    },
}


class ActiveOrchestrator:
    """Executes MCP tool recommendations via the client registry.

    Translates high-level ToolRecommendation objects into concrete
    MCPToolCall objects and executes them through the registry.
    """

    def __init__(self, registry: MCPClientRegistry) -> None:
        """Initialize with a client registry.

        Args:
            registry: MCPClientRegistry with configured MCP clients.
        """
        self._registry = registry

    async def invoke_recommendations(
        self,
        recommendations: list[ToolRecommendation],
        timeout: float | None = None,
    ) -> list[MCPToolResult]:
        """Convert recommendations to tool calls and execute them.

        Only invokes recommendations for MCPs that are actually configured
        in the client registry. Unconfigured MCPs are skipped with a
        warning-level log.

        Args:
            recommendations: Tool recommendations from MCPOrchestrator.
            timeout: Optional timeout override for execution.

        Returns:
            List of MCPToolResult for executed calls.
        """
        calls: list[MCPToolCall] = []
        skipped: list[str] = []

        for rec in recommendations:
            if not self._registry.is_configured(rec.mcp):
                skipped.append(rec.mcp)
                continue

            call = self._recommendation_to_call(rec)
            if call:
                calls.append(call)

        if skipped:
            logger.info(
                "Skipped %d recommendation(s) for unconfigured MCPs: %s",
                len(skipped),
                ", ".join(sorted(set(skipped))),
            )

        if not calls:
            logger.debug("No executable tool calls from recommendations")
            return []

        logger.info("Invoking %d tool call(s) from recommendations", len(calls))
        return await self._registry.call_tools_parallel(calls, timeout=timeout)

    def _recommendation_to_call(
        self,
        rec: ToolRecommendation,
    ) -> MCPToolCall | None:
        """Convert a single ToolRecommendation to an MCPToolCall.

        Uses the recommendation's params if present, otherwise attempts
        to infer the tool name from the action description and MCP name.

        Args:
            rec: A ToolRecommendation.

        Returns:
            MCPToolCall if conversion succeeds, None otherwise.
        """
        tool_name = self._infer_tool_name(rec)
        if not tool_name:
            logger.debug(
                "Could not infer tool name for MCP '%s' action '%s'",
                rec.mcp,
                rec.action,
            )
            return None

        arguments: dict[str, Any] = dict(rec.params) if rec.params else {}

        return MCPToolCall(
            mcp_name=rec.mcp,
            tool_name=tool_name,
            arguments=arguments,
        )

    def _infer_tool_name(self, rec: ToolRecommendation) -> str | None:
        """Infer the tool name from a recommendation.

        Checks the params dict for an explicit tool_name, then falls back
        to action keyword matching against _ACTION_TO_TOOL.

        Args:
            rec: A ToolRecommendation.

        Returns:
            Tool name string, or None if cannot be inferred.
        """
        # Explicit tool_name in params takes priority
        if "tool_name" in rec.params:
            return str(rec.params["tool_name"])

        # Try keyword matching from the action description
        mcp_tools = _ACTION_TO_TOOL.get(rec.mcp, {})
        action_lower = rec.action.lower()

        for keyword, tool_name in mcp_tools.items():
            if keyword in action_lower:
                return tool_name

        # If capabilities are discovered, use the first tool as fallback
        capabilities = self._registry.get_capabilities(rec.mcp)
        if capabilities and capabilities.tools:
            logger.debug(
                "Using first discovered tool '%s' for MCP '%s'",
                capabilities.tools[0].name,
                rec.mcp,
            )
            return capabilities.tools[0].name

        return None

    def get_invocable_count(
        self,
        recommendations: list[ToolRecommendation],
    ) -> int:
        """Count how many recommendations can actually be invoked.

        Args:
            recommendations: Tool recommendations to check.

        Returns:
            Number of recommendations with configured MCPs and inferable tools.
        """
        count = 0
        for rec in recommendations:
            if self._registry.is_configured(rec.mcp) and self._recommendation_to_call(rec):
                count += 1
        return count
