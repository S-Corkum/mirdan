"""MCP Orchestrator - Determines which MCP tools should be used."""

from typing import Any

from mirdan.config import OrchestrationConfig
from mirdan.models import Intent, TaskType, ToolRecommendation


class MCPOrchestrator:
    """Determines which MCP tools should be used for a given intent."""

    def __init__(self, config: OrchestrationConfig | None = None):
        """Initialize with optional orchestration configuration.

        Args:
            config: Orchestration config for MCP preferences
        """
        self._config = config

    # Known MCPs and their capabilities
    KNOWN_MCPS: dict[str, dict[str, Any]] = {
        "context7": {
            "capabilities": ["documentation", "framework_docs", "api_reference"],
        },
        "filesystem": {
            "capabilities": ["file_read", "file_search", "codebase_analysis"],
        },
        "desktop-commander": {
            "capabilities": ["file_read", "file_write", "command_execution"],
        },
        "github": {
            "capabilities": ["repository", "issues", "pull_requests", "commits"],
        },
        "enyal": {
            "capabilities": ["project_context", "decisions", "conventions"],
        },
    }

    def suggest_tools(
        self,
        intent: Intent,
        available_mcps: list[str] | None = None,
    ) -> list[ToolRecommendation]:
        """Suggest which MCP tools should be used for a given intent."""
        # Dispatch to planning-specific for PLANNING tasks
        if intent.task_type == TaskType.PLANNING:
            return self.suggest_tools_for_planning(intent, available_mcps)

        recommendations: list[ToolRecommendation] = []

        # If no MCPs specified, assume common ones are available
        if available_mcps is None:
            available_mcps = list(self.KNOWN_MCPS.keys())

        # Documentation needs
        if intent.uses_external_framework and "context7" in available_mcps:
            frameworks_str = ", ".join(intent.frameworks) if intent.frameworks else "the framework"
            recommendations.append(
                ToolRecommendation(
                    mcp="context7",
                    action=f"Fetch documentation for {frameworks_str}",
                    priority="high",
                    params={"libraries": intent.frameworks},
                    reason="Get current API documentation to avoid hallucinated methods",
                )
            )

        # Project context from memory
        if "enyal" in available_mcps:
            recommendations.append(
                ToolRecommendation(
                    mcp="enyal",
                    action="Recall project conventions and past decisions",
                    priority="high",
                    params={"query": "conventions decisions patterns"},
                    reason="Apply consistent patterns from project history",
                )
            )

        # Codebase analysis needs
        if intent.task_type in [TaskType.GENERATION, TaskType.REFACTOR]:
            if "filesystem" in available_mcps:
                recommendations.append(
                    ToolRecommendation(
                        mcp="filesystem",
                        action="Search for similar patterns in the codebase",
                        priority="high",
                        params={"task_type": intent.task_type.value},
                        reason="Find existing patterns to maintain consistency",
                    )
                )
            elif "desktop-commander" in available_mcps:
                recommendations.append(
                    ToolRecommendation(
                        mcp="desktop-commander",
                        action="Read relevant source files for context",
                        priority="high",
                        reason="Understand existing code structure",
                    )
                )

        # GitHub context
        if "github" in available_mcps:
            if intent.task_type == TaskType.DEBUG:
                recommendations.append(
                    ToolRecommendation(
                        mcp="github",
                        action="Check recent commits for related changes",
                        priority="medium",
                        reason="Understand recent changes that might be relevant",
                    )
                )
            if intent.task_type == TaskType.REVIEW:
                recommendations.append(
                    ToolRecommendation(
                        mcp="github",
                        action="Get PR details and diff for review context",
                        priority="high",
                        reason="Understand full scope of changes being reviewed",
                    )
                )

        # Security scanning recommendations
        if intent.touches_security:
            recommendations.append(
                ToolRecommendation(
                    mcp="security-scanner",
                    action="Scan generated code for vulnerabilities",
                    priority="high",
                    reason="Validate security posture of security-related code",
                )
            )

        # Sort by preference before returning
        return self._sort_by_preference(recommendations)

    def _sort_by_preference(
        self, recommendations: list[ToolRecommendation]
    ) -> list[ToolRecommendation]:
        """Sort recommendations by prefer_mcps configuration.

        MCPs listed in prefer_mcps appear first, in order.
        Other MCPs appear after, sorted alphabetically for stability.

        Args:
            recommendations: List of tool recommendations

        Returns:
            Sorted list with preferred MCPs first
        """
        if not self._config or not self._config.prefer_mcps:
            return recommendations

        prefer_mcps = self._config.prefer_mcps

        def sort_key(rec: ToolRecommendation) -> tuple[int, str]:
            try:
                idx = prefer_mcps.index(rec.mcp)
            except ValueError:
                idx = len(prefer_mcps)  # Non-preferred go after preferred
            return (idx, rec.mcp)  # Secondary sort by name for stability

        return sorted(recommendations, key=sort_key)

    def suggest_tools_for_planning(
        self,
        intent: Intent,
        available_mcps: list[str] | None = None,
    ) -> list[ToolRecommendation]:
        """Suggest tools specifically for PLANNING tasks.

        Planning requires more aggressive tool usage to verify all facts
        BEFORE writing plan steps.
        """
        recommendations: list[ToolRecommendation] = []

        if available_mcps is None:
            available_mcps = list(self.KNOWN_MCPS.keys())

        # MANDATORY: enyal for conventions FIRST (critical priority)
        if "enyal" in available_mcps:
            recommendations.append(
                ToolRecommendation(
                    mcp="enyal",
                    action="Recall ALL project conventions, patterns, and past decisions",
                    priority="critical",
                    params={"query": "conventions patterns decisions architecture"},
                    reason="Plans MUST follow project conventions - verify BEFORE planning",
                )
            )

        # MANDATORY: Filesystem for structure verification
        if "filesystem" in available_mcps:
            recommendations.append(
                ToolRecommendation(
                    mcp="filesystem",
                    action="Glob project structure and Read ALL files to be modified",
                    priority="critical",
                    reason="You CANNOT plan changes to files you haven't Read",
                )
            )
        elif "desktop-commander" in available_mcps:
            recommendations.append(
                ToolRecommendation(
                    mcp="desktop-commander",
                    action="Read ALL files that will be modified",
                    priority="critical",
                    reason="You CANNOT plan changes to files you haven't Read",
                )
            )

        # MANDATORY: context7 for any framework APIs
        if intent.uses_external_framework and "context7" in available_mcps:
            frameworks_str = (
                ", ".join(intent.frameworks) if intent.frameworks else "detected frameworks"
            )
            recommendations.append(
                ToolRecommendation(
                    mcp="context7",
                    action=f"Query documentation for ALL APIs from {frameworks_str}",
                    priority="critical",
                    params={"libraries": intent.frameworks},
                    reason="You CANNOT reference APIs without verification",
                )
            )

        # HIGH: GitHub for recent context
        if "github" in available_mcps:
            recommendations.append(
                ToolRecommendation(
                    mcp="github",
                    action="Check recent commits and open PRs for context",
                    priority="high",
                    reason="Recent changes may affect plan",
                )
            )

        return self._sort_by_preference(recommendations)

    def get_available_mcp_info(self) -> dict[str, dict[str, Any]]:
        """Return information about known MCPs."""
        return self.KNOWN_MCPS.copy()
