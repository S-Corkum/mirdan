"""MCP Orchestrator - Determines which MCP tools should be used."""

from typing import Any

from mirdan.config import OrchestrationConfig
from mirdan.models import Intent, SessionContext, TaskType, ToolRecommendation


class ToolAdvisor:
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

        # If no MCPs specified, assume common ones are available
        if available_mcps is None:
            available_mcps = list(self.KNOWN_MCPS.keys())

        recommendations: list[ToolRecommendation] = []
        recommendations.extend(self._recommend_context7(intent, available_mcps))
        recommendations.extend(self._recommend_enyal(intent, available_mcps, session))
        recommendations.extend(self._recommend_sequential_thinking(intent, available_mcps))
        recommendations.extend(self._recommend_filesystem(intent, available_mcps))
        recommendations.extend(self._recommend_github(intent, available_mcps))
        recommendations.extend(self._recommend_security_scanner(intent, available_mcps))

        # Sort by preference before returning
        return self._sort_by_preference(recommendations)

    def _recommend_context7(
        self,
        intent: Intent,
        available_mcps: list[str],
    ) -> list[ToolRecommendation]:
        """Recommend context7 tools for documentation needs."""
        if not (intent.uses_external_framework and "context7" in available_mcps):
            return []

        frameworks_str = ", ".join(intent.frameworks) if intent.frameworks else "the framework"
        return [
            ToolRecommendation(
                mcp="context7",
                action=f"Fetch documentation for {frameworks_str}",
                priority="high",
                params={"libraries": intent.frameworks},
                reason="Get current API documentation to avoid hallucinated methods",
            )
        ]

    def _recommend_enyal(
        self,
        intent: Intent,
        available_mcps: list[str],
        session: SessionContext | None = None,
    ) -> list[ToolRecommendation]:
        """Recommend enyal tools for project context, persistence, and graph operations."""
        if "enyal" not in available_mcps:
            return []

        recommendations: list[ToolRecommendation] = []

        # Project context from memory — routing depends on session state
        if session and session.validation_count > 0 and session.unresolved_errors > 0:
            # Re-call with persistent failures: target recall to resolution patterns
            recommendations.append(
                ToolRecommendation(
                    mcp="enyal",
                    action=(
                        "Recall patterns for resolving validation failures and similar past fixes"
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
        if not (session and session.validation_count > 0):
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

        # Graph operations for architecture-aware tasks
        if intent.task_type == TaskType.REFACTOR:
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

        return recommendations

    def _recommend_sequential_thinking(
        self,
        intent: Intent,
        available_mcps: list[str],
    ) -> list[ToolRecommendation]:
        """Recommend sequential-thinking tools for deep analysis.

        Sequential-thinking is recommended for ALL task types to encourage
        structured reasoning before implementation. Priority varies by task
        complexity: security-sensitive and debugging tasks get "high",
        generation/refactor/review/test get "medium".
        """
        if "sequential-thinking" not in available_mcps:
            return []

        # Security-sensitive tasks always get high-priority thinking
        if intent.touches_security:
            return [
                ToolRecommendation(
                    mcp="sequential-thinking",
                    action=(
                        "Analyze security implications: threat model, "
                        "attack surface, and defense-in-depth strategy"
                    ),
                    priority="high",
                    reason="Security-sensitive code requires thorough threat analysis",
                )
            ]

        if intent.task_type == TaskType.DEBUG:
            return [
                ToolRecommendation(
                    mcp="sequential-thinking",
                    action=(
                        "Form structured hypotheses about root cause, "
                        "then plan systematic verification for each"
                    ),
                    priority="high",
                    reason="Structured reasoning prevents unfocused investigation",
                )
            ]

        if intent.task_type == TaskType.REFACTOR:
            return [
                ToolRecommendation(
                    mcp="sequential-thinking",
                    action=(
                        "Analyze refactoring scope: all callers, dependencies, "
                        "and potential regressions before making changes"
                    ),
                    priority="medium",
                    reason="Refactoring has cascading effects that benefit from upfront analysis",
                )
            ]

        if intent.task_type == TaskType.GENERATION:
            return [
                ToolRecommendation(
                    mcp="sequential-thinking",
                    action=(
                        "Plan the implementation approach: break down the task, "
                        "identify components, dependencies, and edge cases "
                        "before writing code"
                    ),
                    priority="medium",
                    reason="Upfront planning produces better-structured code with fewer iterations",
                )
            ]

        if intent.task_type == TaskType.REVIEW:
            return [
                ToolRecommendation(
                    mcp="sequential-thinking",
                    action=(
                        "Structure the review: identify areas of concern, "
                        "check for patterns across changes, and assess "
                        "overall design coherence before commenting"
                    ),
                    priority="medium",
                    reason="Structured review catches systemic issues, not just line-level problems",
                )
            ]

        if intent.task_type == TaskType.TEST:
            return [
                ToolRecommendation(
                    mcp="sequential-thinking",
                    action=(
                        "Analyze what needs testing: identify edge cases, "
                        "boundary conditions, error paths, and interaction "
                        "points before writing test code"
                    ),
                    priority="medium",
                    reason="Thinking through test scenarios first produces better coverage",
                )
            ]

        return []

    def _recommend_filesystem(
        self,
        intent: Intent,
        available_mcps: list[str],
    ) -> list[ToolRecommendation]:
        """Recommend filesystem tools for codebase analysis, with desktop-commander fallback."""
        if intent.task_type not in [TaskType.GENERATION, TaskType.REFACTOR]:
            return []

        if "filesystem" in available_mcps:
            return [
                ToolRecommendation(
                    mcp="filesystem",
                    action="Search for similar patterns in the codebase",
                    priority="high",
                    params={"task_type": intent.task_type.value},
                    reason="Find existing patterns to maintain consistency",
                )
            ]
        elif "desktop-commander" in available_mcps:
            return [
                ToolRecommendation(
                    mcp="desktop-commander",
                    action="Read relevant source files for context",
                    priority="high",
                    reason="Understand existing code structure",
                )
            ]

        return []

    def _recommend_github(
        self,
        intent: Intent,
        available_mcps: list[str],
    ) -> list[ToolRecommendation]:
        """Recommend GitHub tools for repository context."""
        if "github" not in available_mcps:
            return []

        recommendations: list[ToolRecommendation] = []

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

        return recommendations

    def _recommend_security_scanner(
        self,
        intent: Intent,
        available_mcps: list[str],
    ) -> list[ToolRecommendation]:
        """Recommend security scanner when security is relevant."""
        if not (intent.touches_security and "security-scanner" in available_mcps):
            return []

        return [
            ToolRecommendation(
                mcp="security-scanner",
                action="Scan generated code for vulnerabilities",
                priority="high",
                reason="Validate security posture of security-related code",
            )
        ]

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
