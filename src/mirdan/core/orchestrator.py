"""MCP Orchestrator - Determines which MCP tools should be used."""

from typing import Any

from mirdan.models import Intent, TaskType, ToolRecommendation


class MCPOrchestrator:
    """Determines which MCP tools should be used for a given intent."""

    # Known MCPs and their capabilities
    KNOWN_MCPS: dict[str, dict[str, Any]] = {
        "context7": {
            "capabilities": ["documentation", "framework_docs", "api_reference"],
            "actions": ["fetch_docs", "search_docs"],
        },
        "filesystem": {
            "capabilities": ["file_read", "file_search", "codebase_analysis"],
            "actions": ["read_file", "search_files", "list_directory"],
        },
        "desktop-commander": {
            "capabilities": ["file_read", "file_write", "command_execution"],
            "actions": ["read_file", "write_file", "execute_command"],
        },
        "github": {
            "capabilities": ["repository", "issues", "pull_requests", "commits"],
            "actions": ["get_repo", "list_issues", "get_pr", "search_code"],
        },
        "memory": {
            "capabilities": ["conversation_history", "user_preferences"],
            "actions": ["search_memory", "get_recent"],
        },
        "enyal": {
            "capabilities": ["project_context", "decisions", "conventions"],
            "actions": ["recall", "remember"],
        },
    }

    def suggest_tools(
        self,
        intent: Intent,
        available_mcps: list[str] | None = None,
    ) -> list[ToolRecommendation]:
        """Suggest which MCP tools should be used for a given intent."""
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

        return recommendations

    def get_available_mcp_info(self) -> dict[str, dict[str, Any]]:
        """Return information about known MCPs."""
        return self.KNOWN_MCPS.copy()
