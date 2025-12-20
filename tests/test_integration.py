"""Integration tests for mirdan."""

from unittest.mock import AsyncMock, patch

import pytest

from mirdan.config import MCPClientConfig, MirdanConfig, OrchestrationConfig
from mirdan.core.context_aggregator import ContextAggregator
from mirdan.core.gatherers.base import GathererResult
from mirdan.core.intent_analyzer import IntentAnalyzer
from mirdan.core.orchestrator import MCPOrchestrator
from mirdan.core.prompt_composer import PromptComposer
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import ContextBundle


class TestEnhancePromptIntegration:
    """Integration tests for enhance_prompt flow."""

    @pytest.mark.asyncio
    async def test_enhance_prompt_without_mcp_config(self) -> None:
        """Should work with empty MCP configuration (backward compatible)."""
        # Use empty config - no MCPs configured
        config = MirdanConfig()

        # Initialize all components
        intent_analyzer = IntentAnalyzer()
        quality_standards = QualityStandards()
        prompt_composer = PromptComposer(quality_standards)
        mcp_orchestrator = MCPOrchestrator()
        context_aggregator = ContextAggregator(config)

        # Simulate the enhance_prompt flow
        prompt = "create a FastAPI endpoint for user registration"

        # Analyze intent
        intent = intent_analyzer.analyze(prompt)
        assert intent.task_type.value == "generation"
        assert intent.primary_language == "python"
        assert "fastapi" in intent.frameworks

        # Get tool recommendations
        tool_recommendations = mcp_orchestrator.suggest_tools(intent)
        assert len(tool_recommendations) > 0

        # Gather context (should return empty since no MCPs configured)
        context = await context_aggregator.gather_all(intent)
        assert isinstance(context, ContextBundle)
        # Empty config means empty context
        assert len(context.documentation_hints) == 0

        # Compose enhanced prompt
        enhanced = prompt_composer.compose(intent, context, tool_recommendations)

        # Verify result structure
        result = enhanced.to_dict()
        assert "enhanced_prompt" in result
        assert "task_type" in result
        assert result["task_type"] == "generation"
        assert result["language"] == "python"
        assert "quality_requirements" in result
        assert "verification_steps" in result
        assert "tool_recommendations" in result

        # Cleanup
        await context_aggregator.close()

    @pytest.mark.asyncio
    async def test_enhance_prompt_with_mock_mcps(self) -> None:
        """Should populate context when MCPs are available."""
        # Create config with MCPs
        config = MirdanConfig()
        config.orchestration = OrchestrationConfig(
            mcp_clients={
                "context7": MCPClientConfig(type="http", url="https://example.com"),
            }
        )

        # Initialize components
        intent_analyzer = IntentAnalyzer()
        quality_standards = QualityStandards()
        prompt_composer = PromptComposer(quality_standards)
        mcp_orchestrator = MCPOrchestrator()
        context_aggregator = ContextAggregator(config)

        prompt = "create a FastAPI endpoint using Pydantic models"
        intent = intent_analyzer.analyze(prompt)

        # Mock the gatherers to return data
        mock_context = ContextBundle(
            documentation_hints=["[fastapi] FastAPI docs here..."],
            existing_patterns=["def create_endpoint(): ..."],
            tech_stack={"fastapi": "0.100.0"},
        )

        with patch.object(
            context_aggregator,
            "gather_all",
            new_callable=AsyncMock,
            return_value=mock_context,
        ):
            context = await context_aggregator.gather_all(intent)

        assert len(context.documentation_hints) > 0
        assert len(context.existing_patterns) > 0
        assert "fastapi" in context.tech_stack

        # Compose with populated context
        tool_recommendations = mcp_orchestrator.suggest_tools(intent)
        enhanced = prompt_composer.compose(intent, context, tool_recommendations)

        result = enhanced.to_dict()
        assert "enhanced_prompt" in result
        # The enhanced prompt should contain more context now
        assert len(result["enhanced_prompt"]) > len(prompt)

        await context_aggregator.close()

    @pytest.mark.asyncio
    async def test_full_flow_with_security_task(self) -> None:
        """Should detect and handle security-related tasks appropriately."""
        config = MirdanConfig()
        intent_analyzer = IntentAnalyzer()
        quality_standards = QualityStandards()
        prompt_composer = PromptComposer(quality_standards)
        mcp_orchestrator = MCPOrchestrator()
        context_aggregator = ContextAggregator(config)

        prompt = "implement JWT authentication with password hashing"

        intent = intent_analyzer.analyze(prompt)

        # Should detect as security-related
        assert intent.touches_security is True

        tool_recommendations = mcp_orchestrator.suggest_tools(intent)
        context = await context_aggregator.gather_all(intent)
        enhanced = prompt_composer.compose(intent, context, tool_recommendations)

        result = enhanced.to_dict()
        assert result["touches_security"] is True

        # Verification steps should include security-related checks
        security_keywords = ["password", "sensitive", "sanitiz", "credential"]
        assert any(
            any(keyword in step.lower() for keyword in security_keywords)
            for step in result["verification_steps"]
        )

        await context_aggregator.close()

    @pytest.mark.asyncio
    async def test_graceful_degradation(self) -> None:
        """Should gracefully degrade when gatherers fail."""
        config = MirdanConfig()
        config.orchestration = OrchestrationConfig(
            mcp_clients={
                "context7": MCPClientConfig(type="http", url="https://example.com"),
            }
        )

        intent_analyzer = IntentAnalyzer()
        quality_standards = QualityStandards()
        prompt_composer = PromptComposer(quality_standards)
        mcp_orchestrator = MCPOrchestrator()
        context_aggregator = ContextAggregator(config)

        prompt = "create a React component"
        intent = intent_analyzer.analyze(prompt)

        # Create mock gatherers that fail
        for gatherer in context_aggregator._gatherers:
            gatherer.is_available = AsyncMock(return_value=True)
            gatherer.gather = AsyncMock(
                return_value=GathererResult(
                    success=False,
                    context=ContextBundle(),
                    error="Connection failed",
                )
            )

        # Should still complete successfully with empty context
        context = await context_aggregator.gather_all(intent)
        assert isinstance(context, ContextBundle)

        # Should still be able to compose a prompt
        tool_recommendations = mcp_orchestrator.suggest_tools(intent)
        enhanced = prompt_composer.compose(intent, context, tool_recommendations)

        result = enhanced.to_dict()
        assert "enhanced_prompt" in result
        assert result["task_type"] is not None

        await context_aggregator.close()


class TestConfigurationLoading:
    """Tests for configuration loading."""

    def test_empty_config_works(self) -> None:
        """Should work with default empty configuration."""
        config = MirdanConfig()

        assert config.orchestration.mcp_clients == {}
        assert config.orchestration.gather_timeout == 10.0
        assert config.orchestration.gatherer_timeout == 3.0

    def test_config_with_clients(self) -> None:
        """Should properly parse MCP client configurations."""
        config = MirdanConfig()
        config.orchestration = OrchestrationConfig(
            mcp_clients={
                "context7": MCPClientConfig(
                    type="http",
                    url="https://context7.com/mcp",
                    timeout=30.0,
                ),
                "enyal": MCPClientConfig(
                    type="stdio",
                    command="uvx",
                    args=["enyal", "serve"],
                    env={"LOG_LEVEL": "DEBUG"},
                ),
            }
        )

        assert "context7" in config.orchestration.mcp_clients
        assert "enyal" in config.orchestration.mcp_clients

        context7 = config.orchestration.mcp_clients["context7"]
        assert context7.type == "http"
        assert context7.url == "https://context7.com/mcp"

        enyal = config.orchestration.mcp_clients["enyal"]
        assert enyal.type == "stdio"
        assert enyal.command == "uvx"
        assert enyal.args == ["enyal", "serve"]
        assert enyal.env == {"LOG_LEVEL": "DEBUG"}
