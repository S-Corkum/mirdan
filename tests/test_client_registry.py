"""Tests for MCPClientRegistry."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestCapabilityDiscovery:
    """Tests for MCP capability discovery."""

    @pytest.fixture
    def mock_capabilities_response(self) -> dict:
        """Create mock responses for capability discovery."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.inputSchema = {"type": "object"}

        mock_resource = MagicMock()
        mock_resource.uri = "test://resource"
        mock_resource.name = "test_resource"
        mock_resource.description = "A test resource"
        mock_resource.mimeType = "text/plain"

        mock_prompt = MagicMock()
        mock_prompt.name = "test_prompt"
        mock_prompt.description = "A test prompt"

        return {
            "tools": [mock_tool],
            "resources": [mock_resource],
            "templates": [],
            "prompts": [mock_prompt],
        }

    @pytest.fixture
    def stdio_config(self) -> MirdanConfig:
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

    @pytest.mark.asyncio
    async def test_discover_capabilities_returns_capabilities(
        self, stdio_config: MirdanConfig, mock_capabilities_response: dict
    ) -> None:
        """Should discover and return capabilities."""
        registry = MCPClientRegistry(stdio_config)

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_capabilities_response["tools"])
        mock_client.list_resources = AsyncMock(return_value=mock_capabilities_response["resources"])
        mock_client.list_resource_templates = AsyncMock(return_value=mock_capabilities_response["templates"])
        mock_client.list_prompts = AsyncMock(return_value=mock_capabilities_response["prompts"])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(registry, "_create_client", return_value=mock_client):
            capabilities = await registry.discover_capabilities("enyal")

        assert capabilities is not None
        assert len(capabilities.tools) == 1
        assert capabilities.tools[0].name == "test_tool"
        assert capabilities.has_tool("test_tool") is True
        assert capabilities.has_tool("nonexistent") is False

    @pytest.mark.asyncio
    async def test_discover_capabilities_handles_failure(
        self, stdio_config: MirdanConfig
    ) -> None:
        """Should handle discovery failures gracefully."""
        registry = MCPClientRegistry(stdio_config)

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(side_effect=Exception("Connection failed"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(registry, "_create_client", return_value=mock_client):
            capabilities = await registry.discover_capabilities("enyal")

        assert capabilities is None
        assert registry.get_discovery_error("enyal") == "Connection failed"

    @pytest.mark.asyncio
    async def test_get_client_triggers_discovery(
        self, stdio_config: MirdanConfig, mock_capabilities_response: dict
    ) -> None:
        """get_client should trigger capability discovery."""
        registry = MCPClientRegistry(stdio_config)

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_capabilities_response["tools"])
        mock_client.list_resources = AsyncMock(return_value=mock_capabilities_response["resources"])
        mock_client.list_resource_templates = AsyncMock(return_value=mock_capabilities_response["templates"])
        mock_client.list_prompts = AsyncMock(return_value=mock_capabilities_response["prompts"])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "mirdan.core.client_registry.StdioTransport"
        ), patch(
            "mirdan.core.client_registry.Client", return_value=mock_client
        ):
            client = await registry.get_client("enyal")

        assert client is not None
        capabilities = registry.get_capabilities("enyal")
        assert capabilities is not None
        assert len(capabilities.tools) == 1

    @pytest.mark.asyncio
    async def test_get_capabilities_returns_cached(
        self, stdio_config: MirdanConfig, mock_capabilities_response: dict
    ) -> None:
        """get_capabilities should return cached capabilities."""
        registry = MCPClientRegistry(stdio_config)

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_capabilities_response["tools"])
        mock_client.list_resources = AsyncMock(return_value=[])
        mock_client.list_resource_templates = AsyncMock(return_value=[])
        mock_client.list_prompts = AsyncMock(return_value=[])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(registry, "_create_client", return_value=mock_client):
            await registry.get_client("enyal")

        # Second call should return cached
        capabilities = registry.get_capabilities("enyal")
        assert capabilities is not None
        # list_tools should only be called once (during discovery)
        assert mock_client.list_tools.call_count == 1

    def test_has_tool_returns_true_when_unknown(
        self, stdio_config: MirdanConfig
    ) -> None:
        """has_tool should return True when capabilities unknown (permissive)."""
        registry = MCPClientRegistry(stdio_config)

        # No discovery done yet
        result = registry.has_tool("enyal", "any_tool")

        assert result is True  # Permissive default

    @pytest.mark.asyncio
    async def test_has_tool_returns_false_when_tool_missing(
        self, stdio_config: MirdanConfig, mock_capabilities_response: dict
    ) -> None:
        """has_tool should return False when tool confirmed missing."""
        registry = MCPClientRegistry(stdio_config)

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_capabilities_response["tools"])
        mock_client.list_resources = AsyncMock(return_value=[])
        mock_client.list_resource_templates = AsyncMock(return_value=[])
        mock_client.list_prompts = AsyncMock(return_value=[])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(registry, "_create_client", return_value=mock_client):
            await registry.discover_capabilities("enyal")

        assert registry.has_tool("enyal", "test_tool") is True
        assert registry.has_tool("enyal", "nonexistent_tool") is False

    @pytest.mark.asyncio
    async def test_close_all_clears_capabilities(
        self, stdio_config: MirdanConfig, mock_capabilities_response: dict
    ) -> None:
        """close_all should clear capabilities along with clients."""
        registry = MCPClientRegistry(stdio_config)

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_capabilities_response["tools"])
        mock_client.list_resources = AsyncMock(return_value=[])
        mock_client.list_resource_templates = AsyncMock(return_value=[])
        mock_client.list_prompts = AsyncMock(return_value=[])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(registry, "_create_client", return_value=mock_client):
            await registry.get_client("enyal")

        assert registry.get_capabilities("enyal") is not None

        await registry.close_all()

        assert registry.get_capabilities("enyal") is None
        assert len(registry._clients) == 0
        assert len(registry._capabilities) == 0

    @pytest.mark.asyncio
    async def test_discover_capabilities_force_refresh(
        self, stdio_config: MirdanConfig
    ) -> None:
        """force=True should re-discover even if cached."""
        registry = MCPClientRegistry(stdio_config)

        # First discovery
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool_v1"
        mock_tool1.description = None
        mock_tool1.inputSchema = None

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[mock_tool1])
        mock_client.list_resources = AsyncMock(return_value=[])
        mock_client.list_resource_templates = AsyncMock(return_value=[])
        mock_client.list_prompts = AsyncMock(return_value=[])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(registry, "_create_client", return_value=mock_client):
            await registry.discover_capabilities("enyal")

        assert registry.has_tool("enyal", "tool_v1") is True

        # Update mock for second discovery
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool_v2"
        mock_tool2.description = None
        mock_tool2.inputSchema = None
        mock_client.list_tools = AsyncMock(return_value=[mock_tool2])

        with patch.object(registry, "_create_client", return_value=mock_client):
            await registry.discover_capabilities("enyal", force=True)

        assert registry.has_tool("enyal", "tool_v2") is True
        assert registry.has_tool("enyal", "tool_v1") is False
