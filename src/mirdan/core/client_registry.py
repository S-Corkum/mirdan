"""MCP Client Registry - Manages connections to external MCP servers."""

import asyncio
import logging
import os
import time
from datetime import UTC
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport

from mirdan.config import MCPClientConfig, MirdanConfig
from mirdan.models import (
    MCPCapabilities,
    MCPPromptInfo,
    MCPResourceInfo,
    MCPResourceTemplateInfo,
    MCPToolCall,
    MCPToolInfo,
    MCPToolResult,
)

logger = logging.getLogger(__name__)


class MCPClientRegistry:
    """Manages FastMCP Client connections to external MCP servers."""

    def __init__(self, config: MirdanConfig) -> None:
        """Initialize registry with configuration.

        Args:
            config: Mirdan configuration containing MCP client definitions
        """
        self._config = config
        self._clients: dict[str, Client[Any]] = {}
        self._capabilities: dict[str, MCPCapabilities] = {}
        self._discovery_errors: dict[str, str] = {}

    def is_configured(self, mcp_name: str) -> bool:
        """Check if an MCP is configured.

        Args:
            mcp_name: Name of the MCP (e.g., 'context7', 'enyal')

        Returns:
            True if the MCP is configured, False otherwise
        """
        return mcp_name in self._config.orchestration.mcp_clients

    async def get_client(self, mcp_name: str) -> Client[Any] | None:
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

            # Discover capabilities (best effort - doesn't block client return)
            if mcp_name not in self._capabilities:
                capabilities = await self._discover_capabilities(mcp_name, client)
                if capabilities:
                    self._capabilities[mcp_name] = capabilities

            self._clients[mcp_name] = client
            logger.debug("Created client for MCP '%s'", mcp_name)
            return client
        except Exception as e:
            logger.error("Failed to create client for MCP '%s': %s", mcp_name, e)
            return None

    def get_capabilities(self, mcp_name: str) -> MCPCapabilities | None:
        """Get discovered capabilities for an MCP.

        Args:
            mcp_name: Name of the MCP

        Returns:
            MCPCapabilities if discovered, None if not yet discovered or failed
        """
        return self._capabilities.get(mcp_name)

    def has_tool(self, mcp_name: str, tool_name: str) -> bool:
        """Check if an MCP has a specific tool.

        If capabilities haven't been discovered yet, returns True (permissive)
        to allow the tool call to proceed and fail gracefully if needed.

        Args:
            mcp_name: Name of the MCP
            tool_name: Name of the tool to check

        Returns:
            True if tool exists or capabilities unknown, False if tool confirmed missing
        """
        capabilities = self._capabilities.get(mcp_name)
        if capabilities is None:
            # Capabilities not discovered - be permissive, let it try
            logger.debug(
                "Capabilities for MCP '%s' not discovered, assuming tool '%s' exists",
                mcp_name,
                tool_name,
            )
            return True
        return capabilities.has_tool(tool_name)

    async def discover_capabilities(
        self,
        mcp_name: str,
        force: bool = False,
    ) -> MCPCapabilities | None:
        """Explicitly discover or refresh capabilities for an MCP.

        Args:
            mcp_name: Name of the MCP to discover
            force: If True, re-discover even if already cached

        Returns:
            MCPCapabilities if discovery succeeds, None otherwise
        """
        if not self.is_configured(mcp_name):
            logger.debug("Cannot discover capabilities - MCP '%s' not configured", mcp_name)
            return None

        if not force and mcp_name in self._capabilities:
            return self._capabilities[mcp_name]

        # Get or create client
        client = await self.get_client(mcp_name)
        if client is None:
            return None

        # Force re-discovery
        if force:
            capabilities = await self._discover_capabilities(mcp_name, client)
            if capabilities:
                self._capabilities[mcp_name] = capabilities
            return capabilities

        # Return cached (was populated by get_client)
        return self._capabilities.get(mcp_name)

    def get_discovery_error(self, mcp_name: str) -> str | None:
        """Get the last discovery error for an MCP.

        Args:
            mcp_name: Name of the MCP

        Returns:
            Error message if discovery failed, None otherwise
        """
        return self._discovery_errors.get(mcp_name)

    def _create_client(self, name: str, client_config: MCPClientConfig) -> Client[Any]:
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

            transport: StdioTransport | StreamableHttpTransport = StdioTransport(
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
                f"MCP '{name}' has invalid type '{client_config.type}'. Must be 'stdio' or 'http'"
            )

    async def _discover_capabilities(
        self,
        mcp_name: str,
        client: Client[Any],
    ) -> MCPCapabilities | None:
        """Discover capabilities of an MCP server.

        Args:
            mcp_name: Name of the MCP
            client: Client instance to use for discovery

        Returns:
            MCPCapabilities if discovery succeeds, None otherwise
        """
        from datetime import datetime

        try:
            async with client:
                tools_list = await client.list_tools()
                resources_list = await client.list_resources()
                templates_list = await client.list_resource_templates()
                prompts_list = await client.list_prompts()

                capabilities = MCPCapabilities(
                    tools=[
                        MCPToolInfo(
                            name=t.name,
                            description=t.description,
                            input_schema=t.inputSchema if hasattr(t, "inputSchema") else None,
                        )
                        for t in tools_list
                    ],
                    resources=[
                        MCPResourceInfo(
                            uri=str(r.uri),
                            name=r.name,
                            description=r.description,
                            mime_type=r.mimeType if hasattr(r, "mimeType") else None,
                        )
                        for r in resources_list
                    ],
                    resource_templates=[
                        MCPResourceTemplateInfo(
                            uri_template=t.uriTemplate,
                            name=t.name,
                            description=t.description,
                        )
                        for t in templates_list
                    ],
                    prompts=[
                        MCPPromptInfo(
                            name=p.name,
                            description=p.description,
                        )
                        for p in prompts_list
                    ],
                    discovered_at=datetime.now(UTC).isoformat(),
                )

                logger.info(
                    "Discovered capabilities for MCP '%s': %d tools, %d resources, %d prompts",
                    mcp_name,
                    len(capabilities.tools),
                    len(capabilities.resources),
                    len(capabilities.prompts),
                )

                # Clear any previous error
                self._discovery_errors.pop(mcp_name, None)

                return capabilities

        except Exception as e:
            logger.warning("Failed to discover capabilities for MCP '%s': %s", mcp_name, e)
            self._discovery_errors[mcp_name] = str(e)
            return None

    async def close_all(self) -> None:
        """Close all client connections.

        Should be called during server shutdown.
        """
        for name in self._clients:
            try:
                # FastMCP Client doesn't have a close method directly on Client
                # but we clear our references
                logger.debug("Closing client for MCP '%s'", name)
            except Exception as e:
                logger.error("Error closing client for MCP '%s': %s", name, e)

        self._clients.clear()
        self._capabilities.clear()
        self._discovery_errors.clear()
        logger.debug("All MCP clients and capabilities cleared")

    async def call_tools_parallel(
        self,
        calls: list[MCPToolCall],
        timeout: float | None = None,
    ) -> list[MCPToolResult]:
        """Execute multiple MCP tool calls concurrently.

        Args:
            calls: List of tool calls to execute
            timeout: Global timeout in seconds (defaults to gather_timeout from config)

        Returns:
            List of MCPToolResult in same order as input calls
        """
        if not calls:
            return []

        if timeout is None:
            timeout = self._config.orchestration.gather_timeout

        async def _execute_call(call: MCPToolCall) -> MCPToolResult:
            """Execute a single tool call and return result."""
            start_time = time.perf_counter()

            # Check if MCP is configured
            if not self.is_configured(call.mcp_name):
                return MCPToolResult(
                    mcp_name=call.mcp_name,
                    tool_name=call.tool_name,
                    success=False,
                    error=f"MCP '{call.mcp_name}' is not configured",
                    elapsed_ms=(time.perf_counter() - start_time) * 1000,
                )

            try:
                client = await self.get_client(call.mcp_name)
                if client is None:
                    return MCPToolResult(
                        mcp_name=call.mcp_name,
                        tool_name=call.tool_name,
                        success=False,
                        error=f"Failed to get client for MCP '{call.mcp_name}'",
                        elapsed_ms=(time.perf_counter() - start_time) * 1000,
                    )

                async with client:
                    result = await client.call_tool(call.tool_name, call.arguments)

                    # Parse result content - extract text from content list
                    data = None
                    if hasattr(result, "content") and result.content:
                        texts = [c.text for c in result.content if hasattr(c, "text")]
                        data = texts[0] if len(texts) == 1 else texts

                    return MCPToolResult(
                        mcp_name=call.mcp_name,
                        tool_name=call.tool_name,
                        success=True,
                        data=data,
                        elapsed_ms=(time.perf_counter() - start_time) * 1000,
                    )

            except Exception as e:
                logger.warning(
                    "Tool call failed for %s.%s: %s",
                    call.mcp_name,
                    call.tool_name,
                    e,
                )
                return MCPToolResult(
                    mcp_name=call.mcp_name,
                    tool_name=call.tool_name,
                    success=False,
                    error=str(e),
                    elapsed_ms=(time.perf_counter() - start_time) * 1000,
                )

        # Create tasks for all calls
        tasks = [_execute_call(call) for call in calls]

        try:
            # Execute all calls concurrently with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )

            # Convert any exceptions in results to error MCPToolResult
            final_results: list[MCPToolResult] = []
            for i, result in enumerate(results):
                if isinstance(result, BaseException):
                    final_results.append(
                        MCPToolResult(
                            mcp_name=calls[i].mcp_name,
                            tool_name=calls[i].tool_name,
                            success=False,
                            error=str(result),
                        )
                    )
                elif isinstance(result, MCPToolResult):
                    final_results.append(result)

            return final_results

        except TimeoutError:
            logger.warning("call_tools_parallel timed out after %s seconds", timeout)
            # Return timeout errors for all calls
            return [
                MCPToolResult(
                    mcp_name=call.mcp_name,
                    tool_name=call.tool_name,
                    success=False,
                    error=f"Global timeout after {timeout}s",
                )
                for call in calls
            ]
