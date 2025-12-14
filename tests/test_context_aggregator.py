"""Tests for ContextAggregator."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirdan.config import MCPClientConfig, MirdanConfig, OrchestrationConfig
from mirdan.core.context_aggregator import ContextAggregator
from mirdan.core.gatherers.base import GathererResult
from mirdan.models import ContextBundle, Intent, TaskType


@pytest.fixture
def empty_config() -> MirdanConfig:
    """Create a config with no MCP clients."""
    return MirdanConfig()


@pytest.fixture
def full_config() -> MirdanConfig:
    """Create a config with all MCP clients."""
    config = MirdanConfig()
    config.orchestration = OrchestrationConfig(
        mcp_clients={
            "context7": MCPClientConfig(type="http", url="https://example.com"),
            "filesystem": MCPClientConfig(type="stdio", command="npx", args=["test"]),
            "enyal": MCPClientConfig(type="stdio", command="uvx", args=["enyal"]),
            "github": MCPClientConfig(type="stdio", command="npx", args=["github"]),
        },
        gather_timeout=10.0,
        gatherer_timeout=3.0,
    )
    return config


@pytest.fixture
def generation_intent() -> Intent:
    """Create a generation intent."""
    return Intent(
        original_prompt="create a FastAPI endpoint",
        task_type=TaskType.GENERATION,
        primary_language="python",
        frameworks=["fastapi"],
    )


class TestContextAggregator:
    """Tests for ContextAggregator."""

    def test_creates_all_gatherers(self, full_config: MirdanConfig) -> None:
        """Should create all four gatherer types."""
        aggregator = ContextAggregator(full_config)

        assert len(aggregator._gatherers) == 4

    @pytest.mark.asyncio
    async def test_gather_all_with_no_available_gatherers(
        self, empty_config: MirdanConfig, generation_intent: Intent
    ) -> None:
        """Should return empty context when no gatherers available."""
        aggregator = ContextAggregator(empty_config)

        result = await aggregator.gather_all(generation_intent)

        assert isinstance(result, ContextBundle)
        assert len(result.existing_patterns) == 0
        assert len(result.relevant_files) == 0
        assert len(result.documentation_hints) == 0

    @pytest.mark.asyncio
    async def test_gather_all_runs_available_gatherers(
        self, full_config: MirdanConfig, generation_intent: Intent
    ) -> None:
        """Should run only available gatherers."""
        aggregator = ContextAggregator(full_config)

        # Create mock gatherers
        mock_gatherer1 = MagicMock()
        mock_gatherer1.name = "MockGatherer1"
        mock_gatherer1.is_available = AsyncMock(return_value=True)
        mock_gatherer1.gather = AsyncMock(
            return_value=GathererResult(
                success=True,
                context=ContextBundle(existing_patterns=["pattern1"]),
            )
        )

        mock_gatherer2 = MagicMock()
        mock_gatherer2.name = "MockGatherer2"
        mock_gatherer2.is_available = AsyncMock(return_value=False)
        mock_gatherer2.gather = AsyncMock()

        aggregator._gatherers = [mock_gatherer1, mock_gatherer2]

        result = await aggregator.gather_all(generation_intent)

        # Only gatherer1 should have been called
        mock_gatherer1.gather.assert_called_once()
        mock_gatherer2.gather.assert_not_called()
        assert "pattern1" in result.existing_patterns

    @pytest.mark.asyncio
    async def test_gather_all_merges_results(
        self, full_config: MirdanConfig, generation_intent: Intent
    ) -> None:
        """Should merge results from all gatherers."""
        aggregator = ContextAggregator(full_config)

        # Create mock gatherers with different results
        mock_gatherer1 = MagicMock()
        mock_gatherer1.name = "MockGatherer1"
        mock_gatherer1.is_available = AsyncMock(return_value=True)
        mock_gatherer1.gather = AsyncMock(
            return_value=GathererResult(
                success=True,
                context=ContextBundle(
                    existing_patterns=["pattern1"],
                    relevant_files=["file1.py"],
                ),
            )
        )

        mock_gatherer2 = MagicMock()
        mock_gatherer2.name = "MockGatherer2"
        mock_gatherer2.is_available = AsyncMock(return_value=True)
        mock_gatherer2.gather = AsyncMock(
            return_value=GathererResult(
                success=True,
                context=ContextBundle(
                    existing_patterns=["pattern2"],
                    documentation_hints=["hint1"],
                ),
            )
        )

        aggregator._gatherers = [mock_gatherer1, mock_gatherer2]

        result = await aggregator.gather_all(generation_intent)

        assert "pattern1" in result.existing_patterns
        assert "pattern2" in result.existing_patterns
        assert "file1.py" in result.relevant_files
        assert "hint1" in result.documentation_hints

    @pytest.mark.asyncio
    async def test_gather_all_handles_gatherer_failure(
        self, full_config: MirdanConfig, generation_intent: Intent
    ) -> None:
        """Should continue when individual gatherer fails."""
        aggregator = ContextAggregator(full_config)

        # Create mock gatherers - one fails, one succeeds
        mock_gatherer1 = MagicMock()
        mock_gatherer1.name = "FailingGatherer"
        mock_gatherer1.is_available = AsyncMock(return_value=True)
        mock_gatherer1.gather = AsyncMock(
            return_value=GathererResult(
                success=False,
                context=ContextBundle(),
                error="Failed to connect",
            )
        )

        mock_gatherer2 = MagicMock()
        mock_gatherer2.name = "SucceedingGatherer"
        mock_gatherer2.is_available = AsyncMock(return_value=True)
        mock_gatherer2.gather = AsyncMock(
            return_value=GathererResult(
                success=True,
                context=ContextBundle(existing_patterns=["pattern1"]),
            )
        )

        aggregator._gatherers = [mock_gatherer1, mock_gatherer2]

        result = await aggregator.gather_all(generation_intent)

        # Should still have data from successful gatherer
        assert "pattern1" in result.existing_patterns

    @pytest.mark.asyncio
    async def test_gather_all_handles_gatherer_exception(
        self, full_config: MirdanConfig, generation_intent: Intent
    ) -> None:
        """Should handle exceptions from gatherers."""
        aggregator = ContextAggregator(full_config)

        mock_gatherer1 = MagicMock()
        mock_gatherer1.name = "ExceptionGatherer"
        mock_gatherer1.is_available = AsyncMock(return_value=True)
        mock_gatherer1.gather = AsyncMock(side_effect=Exception("Unexpected error"))

        mock_gatherer2 = MagicMock()
        mock_gatherer2.name = "SucceedingGatherer"
        mock_gatherer2.is_available = AsyncMock(return_value=True)
        mock_gatherer2.gather = AsyncMock(
            return_value=GathererResult(
                success=True,
                context=ContextBundle(existing_patterns=["pattern1"]),
            )
        )

        aggregator._gatherers = [mock_gatherer1, mock_gatherer2]

        result = await aggregator.gather_all(generation_intent)

        # Should still have data from successful gatherer
        assert "pattern1" in result.existing_patterns

    @pytest.mark.asyncio
    async def test_gather_all_respects_context_level(
        self, full_config: MirdanConfig, generation_intent: Intent
    ) -> None:
        """Should pass context level to gatherers."""
        aggregator = ContextAggregator(full_config)

        mock_gatherer = MagicMock()
        mock_gatherer.name = "MockGatherer"
        mock_gatherer.is_available = AsyncMock(return_value=True)
        mock_gatherer.gather = AsyncMock(
            return_value=GathererResult(success=True, context=ContextBundle())
        )

        aggregator._gatherers = [mock_gatherer]

        await aggregator.gather_all(generation_intent, context_level="comprehensive")

        mock_gatherer.gather.assert_called_once_with(generation_intent, "comprehensive")

    @pytest.mark.asyncio
    async def test_gather_all_handles_timeout(
        self, full_config: MirdanConfig, generation_intent: Intent
    ) -> None:
        """Should handle global timeout gracefully."""
        # Set a very short timeout
        full_config.orchestration.gather_timeout = 0.01
        aggregator = ContextAggregator(full_config)

        # Create a gatherer that takes too long
        async def slow_gather(*args, **kwargs):
            await asyncio.sleep(10)  # Much longer than timeout
            return GathererResult(success=True, context=ContextBundle())

        mock_gatherer = MagicMock()
        mock_gatherer.name = "SlowGatherer"
        mock_gatherer.is_available = AsyncMock(return_value=True)
        mock_gatherer.gather = slow_gather

        aggregator._gatherers = [mock_gatherer]

        # Should complete without hanging, returning empty context
        result = await aggregator.gather_all(generation_intent)

        assert isinstance(result, ContextBundle)

    @pytest.mark.asyncio
    async def test_close_closes_registry(self, full_config: MirdanConfig) -> None:
        """Should close the registry on close."""
        aggregator = ContextAggregator(full_config)

        with patch.object(aggregator._registry, "close_all", new_callable=AsyncMock) as mock_close:
            await aggregator.close()
            mock_close.assert_called_once()

    def test_merge_results_deduplicates(self, full_config: MirdanConfig) -> None:
        """Should deduplicate merged results."""
        aggregator = ContextAggregator(full_config)

        results = [
            GathererResult(
                success=True,
                context=ContextBundle(
                    existing_patterns=["pattern1", "pattern2"],
                    relevant_files=["file1.py"],
                ),
            ),
            GathererResult(
                success=True,
                context=ContextBundle(
                    existing_patterns=["pattern2", "pattern3"],  # pattern2 is duplicate
                    relevant_files=["file1.py", "file2.py"],  # file1.py is duplicate
                ),
            ),
        ]

        merged = aggregator._merge_results(results)

        assert len(merged.existing_patterns) == 3
        assert "pattern1" in merged.existing_patterns
        assert "pattern2" in merged.existing_patterns
        assert "pattern3" in merged.existing_patterns

        assert len(merged.relevant_files) == 2
        assert "file1.py" in merged.relevant_files
        assert "file2.py" in merged.relevant_files

    def test_merge_results_skips_failed(self, full_config: MirdanConfig) -> None:
        """Should skip failed gatherer results."""
        aggregator = ContextAggregator(full_config)

        results = [
            GathererResult(
                success=False,
                context=ContextBundle(existing_patterns=["should_not_appear"]),
                error="Failed",
            ),
            GathererResult(
                success=True,
                context=ContextBundle(existing_patterns=["should_appear"]),
            ),
        ]

        merged = aggregator._merge_results(results)

        assert "should_not_appear" not in merged.existing_patterns
        assert "should_appear" in merged.existing_patterns
