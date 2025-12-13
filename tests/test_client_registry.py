"""Tests for MCPClientRegistry."""

import pytest
from unittest.mock import MagicMock, patch

from mirdan.config import MCPClientConfig, MirdanConfig, OrchestrationConfig
from mirdan.core.client_registry import MCPClientRegistry


@pytest.fixture
def empty_config() -> MirdanConfig:
    """Create a config with no MCP clients."""
    return MirdanConfig()


@pytest.fixture
def stdio_config() -> MirdanConfig:
    """Create a config with a stdio MCP client."""
    config = MirdanConfig()
    config.orchestration = OrchestrationConfig(
        mcp_clients={
            "enyal": MCPClientConfig(
                type="stdio",
                command="uvx",
                args=["enyal", "serve"],
            )
        }
    )
    return config


@pytest.fixture
def http_config() -> MirdanConfig:
    """Create a config with an http MCP client."""
    config = MirdanConfig()
    config.orchestration = OrchestrationConfig(
        mcp_clients={
            "context7": MCPClientConfig(
                type="http",
                url="https://context7.com/mcp",
            )
        }
    )
    return config


class TestMCPClientRegistry:
    """Tests for MCPClientRegistry."""

    def test_is_configured_returns_false_for_unconfigured_mcp(
        self, empty_config: MirdanConfig
    ) -> None:
        """Unconfigured MCPs should return False."""
        registry = MCPClientRegistry(empty_config)
        assert registry.is_configured("context7") is False
        assert registry.is_configured("enyal") is False
        assert registry.is_configured("nonexistent") is False

    def test_is_configured_returns_true_for_configured_mcp(
        self, stdio_config: MirdanConfig
    ) -> None:
        """Configured MCPs should return True."""
        registry = MCPClientRegistry(stdio_config)
        assert registry.is_configured("enyal") is True
        assert registry.is_configured("context7") is False

    @pytest.mark.asyncio
    async def test_get_client_returns_none_for_unconfigured(
        self, empty_config: MirdanConfig
    ) -> None:
        """get_client should return None for unconfigured MCPs."""
        registry = MCPClientRegistry(empty_config)
        client = await registry.get_client("context7")
        assert client is None

    @pytest.mark.asyncio
    async def test_get_client_creates_stdio_client(
        self, stdio_config: MirdanConfig
    ) -> None:
        """Should create StdioTransport client for type='stdio'."""
        registry = MCPClientRegistry(stdio_config)

        with patch(
            "mirdan.core.client_registry.StdioTransport"
        ) as mock_transport, patch(
            "mirdan.core.client_registry.Client"
        ) as mock_client:
            mock_transport_instance = MagicMock()
            mock_transport.return_value = mock_transport_instance

            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            client = await registry.get_client("enyal")

            assert client is mock_client_instance
            mock_transport.assert_called_once_with(
                command="uvx",
                args=["enyal", "serve"],
                env=None,
                cwd=None,
            )
            mock_client.assert_called_once_with(mock_transport_instance)

    @pytest.mark.asyncio
    async def test_get_client_creates_http_client(
        self, http_config: MirdanConfig
    ) -> None:
        """Should create StreamableHttpTransport client for type='http'."""
        registry = MCPClientRegistry(http_config)

        with patch(
            "mirdan.core.client_registry.StreamableHttpTransport"
        ) as mock_transport, patch(
            "mirdan.core.client_registry.Client"
        ) as mock_client:
            mock_transport_instance = MagicMock()
            mock_transport.return_value = mock_transport_instance

            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            client = await registry.get_client("context7")

            assert client is mock_client_instance
            mock_transport.assert_called_once_with(url="https://context7.com/mcp")
            mock_client.assert_called_once_with(mock_transport_instance)

    @pytest.mark.asyncio
    async def test_get_client_caches_clients(
        self, stdio_config: MirdanConfig
    ) -> None:
        """Same client should be returned on subsequent calls."""
        registry = MCPClientRegistry(stdio_config)

        with patch(
            "mirdan.core.client_registry.StdioTransport"
        ) as mock_transport, patch(
            "mirdan.core.client_registry.Client"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            client1 = await registry.get_client("enyal")
            client2 = await registry.get_client("enyal")

            assert client1 is client2
            # Should only create one client
            assert mock_client.call_count == 1

    @pytest.mark.asyncio
    async def test_close_all_clears_clients(
        self, stdio_config: MirdanConfig
    ) -> None:
        """close_all should clear all cached clients."""
        registry = MCPClientRegistry(stdio_config)

        with patch(
            "mirdan.core.client_registry.StdioTransport"
        ), patch("mirdan.core.client_registry.Client") as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            await registry.get_client("enyal")
            assert len(registry._clients) == 1

            await registry.close_all()
            assert len(registry._clients) == 0

    def test_create_client_raises_for_invalid_type(
        self, empty_config: MirdanConfig
    ) -> None:
        """Should raise ValueError for invalid transport type."""
        registry = MCPClientRegistry(empty_config)
        invalid_config = MCPClientConfig(type="invalid")

        with pytest.raises(ValueError, match="invalid type"):
            registry._create_client("test", invalid_config)

    def test_create_client_raises_for_stdio_without_command(
        self, empty_config: MirdanConfig
    ) -> None:
        """Should raise ValueError for stdio without command."""
        registry = MCPClientRegistry(empty_config)
        invalid_config = MCPClientConfig(type="stdio")

        with pytest.raises(ValueError, match="no command specified"):
            registry._create_client("test", invalid_config)

    def test_create_client_raises_for_http_without_url(
        self, empty_config: MirdanConfig
    ) -> None:
        """Should raise ValueError for http without url."""
        registry = MCPClientRegistry(empty_config)
        invalid_config = MCPClientConfig(type="http")

        with pytest.raises(ValueError, match="no url specified"):
            registry._create_client("test", invalid_config)
