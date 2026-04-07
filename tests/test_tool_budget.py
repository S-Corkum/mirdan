"""Tests for MCP tool budget filtering strategy."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

import mirdan.server as server_mod
from mirdan.config import MirdanConfig, PlatformProfile


async def _get_tool_names() -> set[str]:
    """Get registered tool names via the public API."""
    tools = await server_mod.mcp.list_tools()
    return {t.name for t in tools}


class TestToolPriority:
    """Tests for _TOOL_PRIORITY list."""

    def test_has_seven_entries(self) -> None:
        """_TOOL_PRIORITY should have exactly 7 entries."""
        assert len(server_mod._TOOL_PRIORITY) == 7

    def test_priority_order(self) -> None:
        """Tools should be in correct priority order."""
        expected = [
            "validate_code_quality",
            "validate_quick",
            "enhance_prompt",
            "get_quality_standards",
            "get_quality_trends",
            "scan_dependencies",
            "scan_conventions",
        ]
        assert expected == server_mod._TOOL_PRIORITY

    @pytest.mark.asyncio
    async def test_all_tools_registered(self) -> None:
        """All priority tools should be registered in the MCP server."""
        registered = await _get_tool_names()
        for tool in server_mod._TOOL_PRIORITY:
            assert tool in registered, f"{tool} not registered"


class TestToolBudgetFiltering:
    """Tests for MIRDAN_TOOL_BUDGET env var filtering via _lifespan()."""

    @pytest.fixture(autouse=True)
    async def _save_restore_tools(self) -> AsyncIterator[None]:
        """Save and restore the tool registry around each test."""
        tools = await server_mod.mcp.list_tools()
        saved = {t.name: t for t in tools}
        yield
        # Remove any tools that shouldn't be there
        current = await server_mod.mcp.list_tools()
        for t in current:
            if t.name not in saved:
                server_mod.mcp.local_provider.remove_tool(t.name)
        # Re-add any tools that were removed
        current_names = {t.name for t in await server_mod.mcp.list_tools()}
        for name, tool in saved.items():
            if name not in current_names:
                server_mod.mcp.local_provider.add_tool(tool)
        # Reset components
        server_mod._provider = None

    @pytest.mark.asyncio
    async def test_budget_two_keeps_top_two(self) -> None:
        """MIRDAN_TOOL_BUDGET=2 should keep only validate_code_quality and validate_quick."""
        with patch.dict(os.environ, {"MIRDAN_TOOL_BUDGET": "2"}):
            async with server_mod._lifespan(server_mod.mcp):
                remaining = await _get_tool_names()
                assert remaining == {"validate_code_quality", "validate_quick"}

    @pytest.mark.asyncio
    async def test_budget_zero_removes_all(self) -> None:
        """MIRDAN_TOOL_BUDGET=0 should remove all tools."""
        with patch.dict(os.environ, {"MIRDAN_TOOL_BUDGET": "0"}):
            async with server_mod._lifespan(server_mod.mcp):
                remaining = await _get_tool_names()
                assert len(remaining) == 0

    @pytest.mark.asyncio
    async def test_no_env_var_keeps_all(self) -> None:
        """Without MIRDAN_TOOL_BUDGET, all 7 tools should remain."""
        env = os.environ.copy()
        env.pop("MIRDAN_TOOL_BUDGET", None)
        with patch.dict(os.environ, env, clear=True):
            async with server_mod._lifespan(server_mod.mcp):
                remaining = await _get_tool_names()
                assert len(remaining) == 7

    @pytest.mark.asyncio
    async def test_invalid_env_var_keeps_all(self) -> None:
        """Invalid MIRDAN_TOOL_BUDGET should keep all tools (fallback)."""
        with patch.dict(os.environ, {"MIRDAN_TOOL_BUDGET": "not-a-number"}):
            async with server_mod._lifespan(server_mod.mcp):
                remaining = await _get_tool_names()
                assert len(remaining) == 7

    @pytest.mark.asyncio
    async def test_budget_three_keeps_top_three(self) -> None:
        """MIRDAN_TOOL_BUDGET=3 should keep top 3 priority tools."""
        with patch.dict(os.environ, {"MIRDAN_TOOL_BUDGET": "3"}):
            async with server_mod._lifespan(server_mod.mcp):
                remaining = await _get_tool_names()
                assert remaining == {
                    "validate_code_quality",
                    "validate_quick",
                    "enhance_prompt",
                }

    @pytest.mark.asyncio
    async def test_budget_five_keeps_all(self) -> None:
        """MIRDAN_TOOL_BUDGET=5 should keep top 5 tools."""
        with patch.dict(os.environ, {"MIRDAN_TOOL_BUDGET": "5"}):
            async with server_mod._lifespan(server_mod.mcp):
                remaining = await _get_tool_names()
                assert len(remaining) == 5

    @pytest.mark.asyncio
    async def test_empty_string_keeps_all(self) -> None:
        """Empty MIRDAN_TOOL_BUDGET should keep all tools."""
        with patch.dict(os.environ, {"MIRDAN_TOOL_BUDGET": ""}):
            async with server_mod._lifespan(server_mod.mcp):
                remaining = await _get_tool_names()
                assert len(remaining) == 7


class TestPlatformProfile:
    """Tests for PlatformProfile model."""

    def test_default_values(self) -> None:
        """PlatformProfile should have sensible defaults."""
        profile = PlatformProfile()
        assert profile.name == "generic"
        assert profile.context_level == "auto"
        assert profile.tool_budget_aware is False

    def test_custom_values(self) -> None:
        """PlatformProfile should accept custom values."""
        profile = PlatformProfile(
            name="cursor",
            context_level="minimal",
            tool_budget_aware=True,
        )
        assert profile.name == "cursor"
        assert profile.context_level == "minimal"
        assert profile.tool_budget_aware is True
