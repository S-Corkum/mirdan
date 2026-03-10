"""Tests for the MCP Orchestrator module."""

import pytest

from mirdan.config import OrchestrationConfig
from mirdan.core.orchestrator import MCPOrchestrator
from mirdan.models import Intent, SessionContext, TaskType, ToolRecommendation


@pytest.fixture
def orchestrator() -> MCPOrchestrator:
    """Create an MCPOrchestrator instance."""
    return MCPOrchestrator()


class TestToolSuggestions:
    """Tests for suggest_tools method."""

    def test_suggest_tools_returns_list(self, orchestrator: MCPOrchestrator) -> None:
        """Should return a list of ToolRecommendation."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        result = orchestrator.suggest_tools(intent)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, ToolRecommendation)

    def test_enyal_always_recommended(self, orchestrator: MCPOrchestrator) -> None:
        """Should always recommend enyal when available."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        result = orchestrator.suggest_tools(intent, available_mcps=["enyal"])
        mcp_names = [r.mcp for r in result]
        assert "enyal" in mcp_names

    def test_default_available_mcps_when_none_provided(self, orchestrator: MCPOrchestrator) -> None:
        """Should use KNOWN_MCPS keys when available_mcps is None."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        result = orchestrator.suggest_tools(intent, available_mcps=None)
        # Should recommend multiple MCPs from defaults
        assert len(result) > 1

    def test_filters_by_available_mcps(self, orchestrator: MCPOrchestrator) -> None:
        """Should only recommend MCPs from available_mcps list."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            uses_external_framework=True,
            frameworks=["react"],
        )
        # Only include enyal, not context7
        result = orchestrator.suggest_tools(intent, available_mcps=["enyal"])
        mcp_names = [r.mcp for r in result]
        assert "context7" not in mcp_names
        assert "enyal" in mcp_names


class TestFrameworkDocumentation:
    """Tests for framework documentation recommendations."""

    def test_context7_recommended_for_external_framework(
        self, orchestrator: MCPOrchestrator
    ) -> None:
        """Should recommend context7 when uses_external_framework=True."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            uses_external_framework=True,
            frameworks=["react"],
        )
        result = orchestrator.suggest_tools(intent, available_mcps=["context7"])
        mcp_names = [r.mcp for r in result]
        assert "context7" in mcp_names

    def test_context7_includes_framework_names(self, orchestrator: MCPOrchestrator) -> None:
        """Should include framework names in context7 action text."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            uses_external_framework=True,
            frameworks=["react", "next.js"],
        )
        result = orchestrator.suggest_tools(intent, available_mcps=["context7"])
        context7_rec = next((r for r in result if r.mcp == "context7"), None)
        assert context7_rec is not None
        assert "react" in context7_rec.action.lower() or "next.js" in context7_rec.action.lower()

    def test_no_context7_when_not_available(self, orchestrator: MCPOrchestrator) -> None:
        """Should not recommend context7 when not in available_mcps."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            uses_external_framework=True,
            frameworks=["react"],
        )
        result = orchestrator.suggest_tools(intent, available_mcps=["enyal", "filesystem"])
        mcp_names = [r.mcp for r in result]
        assert "context7" not in mcp_names


class TestTaskTypeRecommendations:
    """Tests for task-type specific recommendations."""

    def test_filesystem_for_generation(self, orchestrator: MCPOrchestrator) -> None:
        """Should recommend filesystem for GENERATION tasks."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        result = orchestrator.suggest_tools(intent, available_mcps=["filesystem"])
        mcp_names = [r.mcp for r in result]
        assert "filesystem" in mcp_names

    def test_filesystem_for_refactor(self, orchestrator: MCPOrchestrator) -> None:
        """Should recommend filesystem for REFACTOR tasks."""
        intent = Intent(original_prompt="test", task_type=TaskType.REFACTOR)
        result = orchestrator.suggest_tools(intent, available_mcps=["filesystem"])
        mcp_names = [r.mcp for r in result]
        assert "filesystem" in mcp_names

    def test_desktop_commander_fallback(self, orchestrator: MCPOrchestrator) -> None:
        """Should fall back to desktop-commander when filesystem unavailable."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        # Only desktop-commander available, not filesystem
        result = orchestrator.suggest_tools(intent, available_mcps=["desktop-commander", "enyal"])
        mcp_names = [r.mcp for r in result]
        assert "desktop-commander" in mcp_names
        assert "filesystem" not in mcp_names

    def test_github_for_debug(self, orchestrator: MCPOrchestrator) -> None:
        """Should recommend github for DEBUG tasks."""
        intent = Intent(original_prompt="test", task_type=TaskType.DEBUG)
        result = orchestrator.suggest_tools(intent, available_mcps=["github"])
        mcp_names = [r.mcp for r in result]
        assert "github" in mcp_names

    def test_github_for_review(self, orchestrator: MCPOrchestrator) -> None:
        """Should recommend github for REVIEW tasks."""
        intent = Intent(original_prompt="test", task_type=TaskType.REVIEW)
        result = orchestrator.suggest_tools(intent, available_mcps=["github"])
        mcp_names = [r.mcp for r in result]
        assert "github" in mcp_names

    def test_security_scanner_when_explicitly_available(
        self, orchestrator: MCPOrchestrator
    ) -> None:
        """Should recommend security-scanner when explicitly in available_mcps."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            touches_security=True,
        )
        result = orchestrator.suggest_tools(intent, available_mcps=["security-scanner"])
        mcp_names = [r.mcp for r in result]
        assert "security-scanner" in mcp_names

    def test_security_scanner_excluded_by_default(self, orchestrator: MCPOrchestrator) -> None:
        """Should NOT recommend security-scanner when available_mcps is None."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            touches_security=True,
        )
        result = orchestrator.suggest_tools(intent, available_mcps=None)
        mcp_names = [r.mcp for r in result]
        assert "security-scanner" not in mcp_names

    def test_security_scanner_excluded_when_not_in_list(
        self, orchestrator: MCPOrchestrator
    ) -> None:
        """Should NOT recommend security-scanner when not in available_mcps."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            touches_security=True,
        )
        result = orchestrator.suggest_tools(intent, available_mcps=["github"])
        mcp_names = [r.mcp for r in result]
        assert "security-scanner" not in mcp_names


class TestMCPPreferences:
    """Tests for MCP preference sorting."""

    def test_sort_by_preference_orders_correctly(self) -> None:
        """Should order recommendations by prefer_mcps configuration."""
        config = OrchestrationConfig(prefer_mcps=["enyal", "context7"])
        orchestrator = MCPOrchestrator(config=config)
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            uses_external_framework=True,
            frameworks=["react"],
        )
        result = orchestrator.suggest_tools(
            intent, available_mcps=["context7", "enyal", "filesystem"]
        )
        mcp_names = [r.mcp for r in result]
        # enyal should come before context7 based on prefer_mcps order
        if "enyal" in mcp_names and "context7" in mcp_names:
            assert mcp_names.index("enyal") < mcp_names.index("context7")

    def test_sort_by_preference_no_config_returns_original(
        self, orchestrator: MCPOrchestrator
    ) -> None:
        """Should return original order when no config."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        result = orchestrator.suggest_tools(intent)
        # Without config, should still return valid recommendations
        assert isinstance(result, list)

    def test_non_preferred_sorted_alphabetically(self) -> None:
        """Should sort non-preferred MCPs alphabetically for stability."""
        config = OrchestrationConfig(prefer_mcps=["enyal"])
        orchestrator = MCPOrchestrator(config=config)
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            uses_external_framework=True,
            frameworks=["react"],
        )
        result = orchestrator.suggest_tools(
            intent, available_mcps=["filesystem", "context7", "enyal"]
        )
        # Get non-enyal MCPs
        non_preferred = [r.mcp for r in result if r.mcp != "enyal"]
        # They should be sorted alphabetically
        assert non_preferred == sorted(non_preferred)


class TestSequentialThinkingRecommendations:
    """Tests for sequential-thinking routing logic."""

    def test_sequential_thinking_for_debug(self, orchestrator: MCPOrchestrator) -> None:
        """Should recommend sequential-thinking for DEBUG tasks."""
        intent = Intent(original_prompt="fix the crash", task_type=TaskType.DEBUG)
        result = orchestrator.suggest_tools(
            intent, available_mcps=["sequential-thinking"]
        )
        mcp_names = [r.mcp for r in result]
        assert "sequential-thinking" in mcp_names
        rec = next(r for r in result if r.mcp == "sequential-thinking")
        assert rec.priority == "high"

    def test_sequential_thinking_for_refactor(self, orchestrator: MCPOrchestrator) -> None:
        """Should recommend sequential-thinking for REFACTOR tasks."""
        intent = Intent(original_prompt="refactor auth module", task_type=TaskType.REFACTOR)
        result = orchestrator.suggest_tools(
            intent, available_mcps=["sequential-thinking"]
        )
        mcp_names = [r.mcp for r in result]
        assert "sequential-thinking" in mcp_names
        rec = next(r for r in result if r.mcp == "sequential-thinking")
        assert rec.priority == "medium"

    def test_sequential_thinking_for_security_sensitive(
        self, orchestrator: MCPOrchestrator
    ) -> None:
        """Should recommend sequential-thinking for security-sensitive GENERATION tasks."""
        intent = Intent(
            original_prompt="add auth endpoint",
            task_type=TaskType.GENERATION,
            touches_security=True,
        )
        result = orchestrator.suggest_tools(
            intent, available_mcps=["sequential-thinking"]
        )
        mcp_names = [r.mcp for r in result]
        assert "sequential-thinking" in mcp_names
        rec = next(r for r in result if r.mcp == "sequential-thinking")
        assert rec.priority == "high"

    def test_no_sequential_thinking_for_simple_generation(
        self, orchestrator: MCPOrchestrator
    ) -> None:
        """Should NOT recommend sequential-thinking for simple GENERATION tasks."""
        intent = Intent(original_prompt="add a button", task_type=TaskType.GENERATION)
        result = orchestrator.suggest_tools(
            intent, available_mcps=["sequential-thinking"]
        )
        mcp_names = [r.mcp for r in result]
        assert "sequential-thinking" not in mcp_names

    def test_no_sequential_thinking_when_unavailable(
        self, orchestrator: MCPOrchestrator
    ) -> None:
        """Should NOT recommend sequential-thinking when not in available_mcps."""
        intent = Intent(original_prompt="fix the crash", task_type=TaskType.DEBUG)
        result = orchestrator.suggest_tools(intent, available_mcps=["enyal", "github"])
        mcp_names = [r.mcp for r in result]
        assert "sequential-thinking" not in mcp_names

    def test_sequential_thinking_for_planning(self, orchestrator: MCPOrchestrator) -> None:
        """Should recommend sequential-thinking with critical priority for PLANNING tasks."""
        intent = Intent(original_prompt="plan a migration", task_type=TaskType.PLANNING)
        result = orchestrator.suggest_tools(
            intent, available_mcps=["sequential-thinking", "enyal", "filesystem"]
        )
        mcp_names = [r.mcp for r in result]
        assert "sequential-thinking" in mcp_names
        rec = next(r for r in result if r.mcp == "sequential-thinking")
        assert rec.priority == "critical"


class TestEnyalGraphRecommendations:
    """Tests for enyal graph operation recommendations."""

    def test_enyal_traverse_for_refactor(self, orchestrator: MCPOrchestrator) -> None:
        """Should recommend enyal_traverse for REFACTOR tasks."""
        intent = Intent(original_prompt="refactor auth module", task_type=TaskType.REFACTOR)
        result = orchestrator.suggest_tools(intent, available_mcps=["enyal"])
        traverse_recs = [
            r for r in result if r.mcp == "enyal" and "traverse" in r.params.get("tool", "")
        ]
        assert len(traverse_recs) == 1
        assert traverse_recs[0].priority == "high"

    def test_enyal_impact_for_refactor(self, orchestrator: MCPOrchestrator) -> None:
        """Should recommend enyal_impact for REFACTOR tasks."""
        intent = Intent(original_prompt="refactor auth module", task_type=TaskType.REFACTOR)
        result = orchestrator.suggest_tools(intent, available_mcps=["enyal"])
        impact_recs = [
            r for r in result if r.mcp == "enyal" and "enyal_impact" in r.params.get("tool", "")
        ]
        assert len(impact_recs) == 1
        assert impact_recs[0].priority == "high"

    def test_no_enyal_graph_for_generation(self, orchestrator: MCPOrchestrator) -> None:
        """Should NOT recommend traverse/impact for simple GENERATION tasks."""
        intent = Intent(original_prompt="add a button", task_type=TaskType.GENERATION)
        result = orchestrator.suggest_tools(intent, available_mcps=["enyal"])
        graph_recs = [
            r
            for r in result
            if r.mcp == "enyal" and r.params.get("tool") in ("enyal_traverse", "enyal_impact")
        ]
        assert graph_recs == []

    def test_enyal_remember_on_first_call(self, orchestrator: MCPOrchestrator) -> None:
        """Should recommend enyal_remember on first call (no session)."""
        intent = Intent(original_prompt="add feature", task_type=TaskType.GENERATION)
        result = orchestrator.suggest_tools(intent, available_mcps=["enyal"], session=None)
        remember_recs = [
            r for r in result if r.mcp == "enyal" and "enyal_remember" in r.params.get("tool", "")
        ]
        assert len(remember_recs) == 1
        assert remember_recs[0].priority == "medium"

    def test_no_enyal_remember_after_validation(self, orchestrator: MCPOrchestrator) -> None:
        """Should NOT recommend enyal_remember after validation has run."""
        intent = Intent(original_prompt="fix issue", task_type=TaskType.DEBUG)
        session = SessionContext(session_id="test", validation_count=1, unresolved_errors=0)
        result = orchestrator.suggest_tools(intent, available_mcps=["enyal"], session=session)
        remember_recs = [
            r for r in result if r.mcp == "enyal" and "enyal_remember" in r.params.get("tool", "")
        ]
        assert remember_recs == []

    def test_enyal_traverse_in_planning(self, orchestrator: MCPOrchestrator) -> None:
        """Should recommend enyal_traverse for PLANNING tasks."""
        intent = Intent(original_prompt="plan a migration", task_type=TaskType.PLANNING)
        result = orchestrator.suggest_tools(
            intent, available_mcps=["sequential-thinking", "enyal", "filesystem"]
        )
        traverse_recs = [
            r for r in result if r.mcp == "enyal" and "traverse" in r.params.get("tool", "")
        ]
        assert len(traverse_recs) == 1
        assert traverse_recs[0].priority == "high"

    def test_no_enyal_graph_when_unavailable(self, orchestrator: MCPOrchestrator) -> None:
        """Should NOT recommend enyal graph ops when enyal not available."""
        intent = Intent(original_prompt="refactor module", task_type=TaskType.REFACTOR)
        result = orchestrator.suggest_tools(intent, available_mcps=["filesystem"])
        enyal_recs = [r for r in result if r.mcp == "enyal"]
        assert enyal_recs == []


class TestAvailableMCPInfo:
    """Tests for get_available_mcp_info method."""

    def test_returns_copy_not_original(self, orchestrator: MCPOrchestrator) -> None:
        """Should return a copy, not the original dict."""
        info1 = orchestrator.get_available_mcp_info()
        info1["test_key"] = {"description": "test_value"}
        info2 = orchestrator.get_available_mcp_info()
        assert "test_key" not in info2

    def test_contains_all_known_mcps(self, orchestrator: MCPOrchestrator) -> None:
        """Should contain all 6 known MCPs."""
        info = orchestrator.get_available_mcp_info()
        expected_mcps = [
            "context7", "filesystem", "desktop-commander",
            "github", "enyal", "sequential-thinking",
        ]
        for mcp in expected_mcps:
            assert mcp in info
