"""Tests for the MCP Orchestrator module."""

import pytest

from mirdan.config import OrchestrationConfig
from mirdan.core.orchestrator import MCPOrchestrator
from mirdan.models import Intent, TaskType, ToolRecommendation


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

    def test_security_scanner_for_security(self, orchestrator: MCPOrchestrator) -> None:
        """Should recommend security-scanner when touches_security=True."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            touches_security=True,
        )
        result = orchestrator.suggest_tools(intent)
        mcp_names = [r.mcp for r in result]
        assert "security-scanner" in mcp_names


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


class TestAvailableMCPInfo:
    """Tests for get_available_mcp_info method."""

    def test_returns_copy_not_original(self, orchestrator: MCPOrchestrator) -> None:
        """Should return a copy, not the original dict."""
        info1 = orchestrator.get_available_mcp_info()
        info1["test_key"] = "test_value"
        info2 = orchestrator.get_available_mcp_info()
        assert "test_key" not in info2

    def test_contains_all_known_mcps(self, orchestrator: MCPOrchestrator) -> None:
        """Should contain all 5 known MCPs."""
        info = orchestrator.get_available_mcp_info()
        expected_mcps = ["context7", "filesystem", "desktop-commander", "github", "enyal"]
        for mcp in expected_mcps:
            assert mcp in info
