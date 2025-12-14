"""Tests for context gatherers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirdan.config import MCPClientConfig, MirdanConfig, OrchestrationConfig
from mirdan.core.client_registry import MCPClientRegistry
from mirdan.core.gatherers import (
    Context7Gatherer,
    EnyalGatherer,
    FilesystemGatherer,
    GitHubGatherer,
)
from mirdan.models import Intent, TaskType


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
        }
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


@pytest.fixture
def debug_intent() -> Intent:
    """Create a debug intent."""
    return Intent(
        original_prompt="fix the login bug",
        task_type=TaskType.DEBUG,
        primary_language="python",
    )


@pytest.fixture
def no_framework_intent() -> Intent:
    """Create an intent with no frameworks."""
    return Intent(
        original_prompt="do something",
        task_type=TaskType.UNKNOWN,
    )


class TestContext7Gatherer:
    """Tests for Context7Gatherer."""

    @pytest.mark.asyncio
    async def test_is_available_returns_false_when_not_configured(
        self, empty_config: MirdanConfig
    ) -> None:
        """Should return False when context7 is not configured."""
        registry = MCPClientRegistry(empty_config)
        gatherer = Context7Gatherer(registry)
        assert await gatherer.is_available() is False

    @pytest.mark.asyncio
    async def test_is_available_returns_true_when_configured(
        self, full_config: MirdanConfig
    ) -> None:
        """Should return True when context7 is configured."""
        registry = MCPClientRegistry(full_config)
        gatherer = Context7Gatherer(registry)
        assert await gatherer.is_available() is True

    @pytest.mark.asyncio
    async def test_gather_with_no_frameworks_returns_empty(
        self, full_config: MirdanConfig, no_framework_intent: Intent
    ) -> None:
        """Should return empty context when no frameworks detected."""
        registry = MCPClientRegistry(full_config)
        gatherer = Context7Gatherer(registry)

        result = await gatherer.gather(no_framework_intent)

        assert result.success is True
        assert len(result.context.documentation_hints) == 0
        assert result.metadata.get("reason") == "no_frameworks"

    @pytest.mark.asyncio
    async def test_gather_fetches_docs_for_frameworks(
        self, full_config: MirdanConfig, generation_intent: Intent
    ) -> None:
        """Should fetch docs for each framework in intent."""
        registry = MCPClientRegistry(full_config)
        gatherer = Context7Gatherer(registry)

        # Mock the client
        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Context7-compatible library ID: /pallets/flask\nSome docs here"
        mock_result.content = [mock_content]
        mock_client.call_tool = AsyncMock(return_value=mock_result)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(registry, "get_client", return_value=mock_client):
            result = await gatherer.gather(generation_intent)

        assert result.success is True
        assert mock_client.call_tool.called

    @pytest.mark.asyncio
    async def test_gather_handles_unavailable_mcp(
        self, empty_config: MirdanConfig, generation_intent: Intent
    ) -> None:
        """Should handle when MCP is not available."""
        registry = MCPClientRegistry(empty_config)
        gatherer = Context7Gatherer(registry)

        result = await gatherer.gather(generation_intent)

        assert result.success is False
        assert "not available" in result.error

    def test_name_property(self, empty_config: MirdanConfig) -> None:
        """Should return correct name."""
        registry = MCPClientRegistry(empty_config)
        gatherer = Context7Gatherer(registry)
        assert gatherer.name == "Context7Gatherer"

    def test_required_mcp_property(self, empty_config: MirdanConfig) -> None:
        """Should return correct required MCP."""
        registry = MCPClientRegistry(empty_config)
        gatherer = Context7Gatherer(registry)
        assert gatherer.required_mcp == "context7"


class TestFilesystemGatherer:
    """Tests for FilesystemGatherer."""

    @pytest.mark.asyncio
    async def test_is_available_returns_false_when_not_configured(
        self, empty_config: MirdanConfig
    ) -> None:
        """Should return False when filesystem is not configured."""
        registry = MCPClientRegistry(empty_config)
        gatherer = FilesystemGatherer(registry)
        assert await gatherer.is_available() is False

    @pytest.mark.asyncio
    async def test_is_available_returns_true_when_configured(
        self, full_config: MirdanConfig
    ) -> None:
        """Should return True when filesystem is configured."""
        registry = MCPClientRegistry(full_config)
        gatherer = FilesystemGatherer(registry)
        assert await gatherer.is_available() is True

    @pytest.mark.asyncio
    async def test_gather_searches_by_language(
        self, full_config: MirdanConfig, generation_intent: Intent
    ) -> None:
        """Should search for files matching language patterns."""
        registry = MCPClientRegistry(full_config)
        gatherer = FilesystemGatherer(registry)

        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "src/main.py\nsrc/utils.py"
        mock_result.content = [mock_content]
        mock_client.call_tool = AsyncMock(return_value=mock_result)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(registry, "get_client", return_value=mock_client):
            result = await gatherer.gather(generation_intent)

        assert result.success is True
        assert result.metadata.get("language") == "python"

    def test_name_property(self, empty_config: MirdanConfig) -> None:
        """Should return correct name."""
        registry = MCPClientRegistry(empty_config)
        gatherer = FilesystemGatherer(registry)
        assert gatherer.name == "FilesystemGatherer"

    def test_required_mcp_property(self, empty_config: MirdanConfig) -> None:
        """Should return correct required MCP."""
        registry = MCPClientRegistry(empty_config)
        gatherer = FilesystemGatherer(registry)
        assert gatherer.required_mcp == "filesystem"


class TestEnyalGatherer:
    """Tests for EnyalGatherer."""

    @pytest.mark.asyncio
    async def test_is_available_returns_false_when_not_configured(
        self, empty_config: MirdanConfig
    ) -> None:
        """Should return False when enyal is not configured."""
        registry = MCPClientRegistry(empty_config)
        gatherer = EnyalGatherer(registry)
        assert await gatherer.is_available() is False

    @pytest.mark.asyncio
    async def test_is_available_returns_true_when_configured(
        self, full_config: MirdanConfig
    ) -> None:
        """Should return True when enyal is configured."""
        registry = MCPClientRegistry(full_config)
        gatherer = EnyalGatherer(registry)
        assert await gatherer.is_available() is True

    @pytest.mark.asyncio
    async def test_gather_builds_query_from_intent(
        self, full_config: MirdanConfig, generation_intent: Intent
    ) -> None:
        """Should build appropriate query from intent."""
        registry = MCPClientRegistry(full_config)
        gatherer = EnyalGatherer(registry)

        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_content = MagicMock()
        mock_content.text = '{"results": []}'
        mock_result.content = [mock_content]
        mock_client.call_tool = AsyncMock(return_value=mock_result)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(registry, "get_client", return_value=mock_client):
            result = await gatherer.gather(generation_intent)

        assert result.success is True
        # Check query was built with language
        assert "python" in result.metadata.get("query", "")

    def test_build_query_includes_language(
        self, empty_config: MirdanConfig, generation_intent: Intent
    ) -> None:
        """Should include language in query."""
        registry = MCPClientRegistry(empty_config)
        gatherer = EnyalGatherer(registry)

        query = gatherer._build_query(generation_intent)

        assert "python" in query

    def test_build_query_includes_frameworks(
        self, empty_config: MirdanConfig, generation_intent: Intent
    ) -> None:
        """Should include frameworks in query."""
        registry = MCPClientRegistry(empty_config)
        gatherer = EnyalGatherer(registry)

        query = gatherer._build_query(generation_intent)

        assert "fastapi" in query

    def test_name_property(self, empty_config: MirdanConfig) -> None:
        """Should return correct name."""
        registry = MCPClientRegistry(empty_config)
        gatherer = EnyalGatherer(registry)
        assert gatherer.name == "EnyalGatherer"

    def test_required_mcp_property(self, empty_config: MirdanConfig) -> None:
        """Should return correct required MCP."""
        registry = MCPClientRegistry(empty_config)
        gatherer = EnyalGatherer(registry)
        assert gatherer.required_mcp == "enyal"


class TestGitHubGatherer:
    """Tests for GitHubGatherer."""

    @pytest.mark.asyncio
    async def test_is_available_returns_false_when_not_configured(
        self, empty_config: MirdanConfig
    ) -> None:
        """Should return False when github is not configured."""
        registry = MCPClientRegistry(empty_config)
        gatherer = GitHubGatherer(registry)
        assert await gatherer.is_available() is False

    @pytest.mark.asyncio
    async def test_is_available_returns_true_when_configured(
        self, full_config: MirdanConfig
    ) -> None:
        """Should return True when github is configured."""
        registry = MCPClientRegistry(full_config)
        gatherer = GitHubGatherer(registry)
        assert await gatherer.is_available() is True

    @pytest.mark.asyncio
    async def test_gather_without_repo_context_returns_empty(
        self, full_config: MirdanConfig, generation_intent: Intent
    ) -> None:
        """Should return empty when no repo context is set."""
        registry = MCPClientRegistry(full_config)
        gatherer = GitHubGatherer(registry)

        result = await gatherer.gather(generation_intent)

        assert result.success is True
        assert result.metadata.get("reason") == "no_repo_context"

    @pytest.mark.asyncio
    async def test_gather_with_repo_context(
        self, full_config: MirdanConfig, generation_intent: Intent
    ) -> None:
        """Should gather context when repo is set."""
        registry = MCPClientRegistry(full_config)
        gatherer = GitHubGatherer(registry)
        gatherer.set_repo_context("owner", "repo")

        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_content = MagicMock()
        mock_content.text = '{"dependencies": {"fastapi": "^0.100.0"}}'
        mock_result.content = [mock_content]
        mock_client.call_tool = AsyncMock(return_value=mock_result)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(registry, "get_client", return_value=mock_client):
            result = await gatherer.gather(generation_intent)

        assert result.success is True
        assert result.metadata.get("owner") == "owner"
        assert result.metadata.get("repo") == "repo"

    def test_set_repo_context(self, empty_config: MirdanConfig) -> None:
        """Should set repo context correctly."""
        registry = MCPClientRegistry(empty_config)
        gatherer = GitHubGatherer(registry)

        gatherer.set_repo_context("myorg", "myrepo")

        assert gatherer._owner == "myorg"
        assert gatherer._repo == "myrepo"

    def test_name_property(self, empty_config: MirdanConfig) -> None:
        """Should return correct name."""
        registry = MCPClientRegistry(empty_config)
        gatherer = GitHubGatherer(registry)
        assert gatherer.name == "GitHubGatherer"

    def test_required_mcp_property(self, empty_config: MirdanConfig) -> None:
        """Should return correct required MCP."""
        registry = MCPClientRegistry(empty_config)
        gatherer = GitHubGatherer(registry)
        assert gatherer.required_mcp == "github"
