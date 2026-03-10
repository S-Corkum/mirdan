"""MCP Orchestrator - Determines which MCP tools should be used."""

from typing import Any

from mirdan.config import OrchestrationConfig
from mirdan.models import Intent, SessionContext, TaskType, ToolRecommendation


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
            "capabilities": [
                "project_context",
                "decisions",
                "conventions",
                "knowledge_graph",
                "impact_analysis",
                "knowledge_maintenance",
            ],
        },
        "sequential-thinking": {
            "capabilities": ["deep_analysis", "structured_reasoning", "planning", "debugging"],
        },
    }

    def suggest_tools(
        self,
        intent: Intent,
        available_mcps: list[str] | None = None,
        session: SessionContext | None = None,
    ) -> list[ToolRecommendation]:
        """Suggest which MCP tools should be used for a given intent."""
        # Dispatch to planning-specific for PLANNING tasks
        if intent.task_type == TaskType.PLANNING:
            return self.suggest_tools_for_planning(intent, available_mcps, session)

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

        # Project context from memory — routing depends on session state
        if "enyal" in available_mcps:
            if session and session.validation_count > 0 and session.unresolved_errors > 0:
                # Re-call with persistent failures: target recall to resolution patterns
                recommendations.append(
                    ToolRecommendation(
                        mcp="enyal",
                        action=(
                            "Recall patterns for resolving validation"
                            " failures and similar past fixes"
                        ),
                        priority="high",
                        params={"query": "fix bug error violation resolution patterns"},
                        reason="Targeted recall for persistent quality failures across sessions",
                    )
                )
            elif not (session and session.validation_count > 0 and session.unresolved_errors == 0):
                # First call or no session: standard convention recall
                recommendations.append(
                    ToolRecommendation(
                        mcp="enyal",
                        action="Recall project conventions and past decisions",
                        priority="high",
                        params={"query": "conventions decisions patterns"},
                        reason="Apply consistent patterns from project history",
                    )
                )
            # else: prior validation passed — skip generic recall (already effective)

        # Post-task knowledge persistence (first call only)
        if "enyal" in available_mcps and not (
            session and session.validation_count > 0
        ):
            recommendations.append(
                ToolRecommendation(
                    mcp="enyal",
                    action=(
                        "After completing the task, store decisions and patterns "
                        "via enyal_remember with appropriate tags and scope"
                    ),
                    priority="medium",
                    params={"tool": "enyal_remember", "when": "after_completion"},
                    reason="Persist insights for future sessions",
                )
            )

        # Enyal graph operations for architecture-aware tasks
        if "enyal" in available_mcps and intent.task_type == TaskType.REFACTOR:
            recommendations.append(
                ToolRecommendation(
                    mcp="enyal",
                    action=(
                        "Traverse knowledge graph around the refactored area "
                        "to understand related conventions and dependencies"
                    ),
                    priority="high",
                    params={"tool": "enyal_traverse", "max_depth": 2},
                    reason="Refactoring may break assumptions stored in the knowledge graph",
                )
            )
            recommendations.append(
                ToolRecommendation(
                    mcp="enyal",
                    action=(
                        "Check impact of changing existing patterns — "
                        "find entries that depend on current conventions"
                    ),
                    priority="high",
                    params={"tool": "enyal_impact"},
                    reason="Understand what depends on patterns being refactored",
                )
            )

        # Deep analysis for complex tasks
        if "sequential-thinking" in available_mcps:
            if intent.task_type == TaskType.DEBUG:
                recommendations.append(
                    ToolRecommendation(
                        mcp="sequential-thinking",
                        action=(
                            "Form structured hypotheses about root cause, "
                            "then plan systematic verification for each"
                        ),
                        priority="high",
                        reason="Structured reasoning prevents unfocused investigation",
                    )
                )
            elif intent.task_type == TaskType.REFACTOR:
                recommendations.append(
                    ToolRecommendation(
                        mcp="sequential-thinking",
                        action=(
                            "Analyze refactoring scope: all callers, dependencies, "
                            "and potential regressions before making changes"
                        ),
                        priority="medium",
                        reason=(
                            "Refactoring has cascading effects that benefit"
                            " from upfront analysis"
                        ),
                    )
                )
            elif intent.touches_security:
                recommendations.append(
                    ToolRecommendation(
                        mcp="sequential-thinking",
                        action=(
                            "Analyze security implications: threat model, "
                            "attack surface, and defense-in-depth strategy"
                        ),
                        priority="high",
                        reason="Security-sensitive code requires thorough threat analysis",
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

        # Security scanning recommendations (only when explicitly available)
        if intent.touches_security and "security-scanner" in available_mcps:
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
        session: SessionContext | None = None,
    ) -> list[ToolRecommendation]:
        """Suggest tools specifically for PLANNING tasks.

        Planning requires more aggressive tool usage to verify all facts
        BEFORE writing plan steps.
        """
        recommendations: list[ToolRecommendation] = []

        if available_mcps is None:
            available_mcps = list(self.KNOWN_MCPS.keys())

        # MANDATORY: Deep analysis FIRST
        if "sequential-thinking" in available_mcps:
            recommendations.append(
                ToolRecommendation(
                    mcp="sequential-thinking",
                    action=(
                        "Analyze the task deeply before writing plan steps. "
                        "Think through: scope, phases, dependencies, completeness, "
                        "and risks. Start with totalThoughts: 8, adjust to 10-15 "
                        "for architectural tasks"
                    ),
                    priority="critical",
                    reason=(
                        "Plans require structured analysis of scope"
                        " and dependencies BEFORE writing steps"
                    ),
                )
            )

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

            # Traverse knowledge graph for full architecture context
            recommendations.append(
                ToolRecommendation(
                    mcp="enyal",
                    action=(
                        "Traverse knowledge graph around the planned area "
                        "to discover related decisions and dependencies"
                    ),
                    priority="high",
                    params={"tool": "enyal_traverse", "max_depth": 2},
                    reason=(
                        "Plans must account for existing architecture"
                        " decisions and their dependencies"
                    ),
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
