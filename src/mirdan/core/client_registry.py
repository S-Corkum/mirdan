"""MCP Client Registry - Manages connections to external MCP servers."""

import logging
import os

from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport

from mirdan.config import MCPClientConfig, MirdanConfig

logger = logging.getLogger(__name__)


class MCPClientRegistry:
    """Manages FastMCP Client connections to external MCP servers."""

    def __init__(self, config: MirdanConfig) -> None:
        """Initialize registry with configuration.

        Args:
            config: Mirdan configuration containing MCP client definitions
        """
        self._config = config
        self._clients: dict[str, Client] = {}

    def is_configured(self, mcp_name: str) -> bool:
        """Check if an MCP is configured.

        Args:
            mcp_name: Name of the MCP (e.g., 'context7', 'enyal')

        Returns:
            True if the MCP is configured, False otherwise
        """
        return mcp_name in self._config.orchestration.mcp_clients

    async def get_client(self, mcp_name: str) -> Client | None:
        """Get a client for the named MCP, creating if needed.

        Args:
            mcp_name: Name of the MCP to connect to

        Returns:
            FastMCP Client instance, or None if not configured
        """
        if not self.is_configured(mcp_name):
            logger.debug("MCP '%s' is not configured", mcp_name)
            return None

        # Return cached client if available
        if mcp_name in self._clients:
            return self._clients[mcp_name]

        # Create new client
        client_config = self._config.orchestration.mcp_clients[mcp_name]
        try:
            client = self._create_client(mcp_name, client_config)
            self._clients[mcp_name] = client
            logger.debug("Created client for MCP '%s'", mcp_name)
            return client
        except Exception as e:
            logger.error("Failed to create client for MCP '%s': %s", mcp_name, e)
            return None

    def _create_client(self, name: str, client_config: MCPClientConfig) -> Client:
        """Create a FastMCP Client based on configuration.

        Args:
            name: Name identifier for the client
            client_config: Configuration for the client connection

        Returns:
            Configured FastMCP Client instance

        Raises:
            ValueError: If transport type is invalid
        """
        if client_config.type == "stdio":
            if not client_config.command:
                raise ValueError(f"MCP '{name}' has type 'stdio' but no command specified")

            # Expand environment variables in cwd
            cwd = None
            if client_config.cwd:
                cwd = os.path.expandvars(client_config.cwd)

            # Expand environment variables in env values
            env = {k: os.path.expandvars(v) for k, v in client_config.env.items()}

            transport = StdioTransport(
                command=client_config.command,
                args=client_config.args,
                env=env if env else None,
                cwd=cwd,
            )
            return Client(transport)

        elif client_config.type == "http":
            if not client_config.url:
                raise ValueError(f"MCP '{name}' has type 'http' but no url specified")

            # Expand environment variables in url
            url = os.path.expandvars(client_config.url)

            transport = StreamableHttpTransport(url=url)
            return Client(transport)

        else:
            raise ValueError(
                f"MCP '{name}' has invalid type '{client_config.type}'. "
                "Must be 'stdio' or 'http'"
            )

    async def close_all(self) -> None:
        """Close all client connections.

        Should be called during server shutdown.
        """
        for name, client in self._clients.items():
            try:
                # FastMCP Client doesn't have a close method directly on Client
                # but we clear our references
                logger.debug("Closing client for MCP '%s'", name)
            except Exception as e:
                logger.error("Error closing client for MCP '%s': %s", name, e)

        self._clients.clear()
        logger.debug("All MCP clients closed")
