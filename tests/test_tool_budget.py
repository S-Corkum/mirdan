"""Tests for MCP tool budget filtering strategy."""

from __future__ import annotations

import os
from collections.abc import Iterator
from unittest.mock import patch

import pytest

import mirdan.server as server_mod
from mirdan.config import MirdanConfig, PlatformProfile


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

    def test_all_tools_registered(self) -> None:
        """All priority tools should be registered in the MCP server."""
        registered = set(server_mod.mcp._tool_manager._tools.keys())
        for tool in server_mod._TOOL_PRIORITY:
            assert tool in registered, f"{tool} not registered"


class TestToolBudgetFiltering:
    """Tests for MIRDAN_TOOL_BUDGET env var filtering via _lifespan()."""

    @pytest.fixture(autouse=True)
    def _save_restore_tools(self) -> Iterator[None]:
        """Save and restore the tool registry around each test."""
        original_tools = dict(server_mod.mcp._tool_manager._tools)
        yield
        server_mod.mcp._tool_manager._tools.clear()
        server_mod.mcp._tool_manager._tools.update(original_tools)
        # Also reset components
        server_mod._provider = None

    @pytest.mark.asyncio
    async def test_budget_two_keeps_top_two(self) -> None:
        """MIRDAN_TOOL_BUDGET=2 should keep only validate_code_quality and validate_quick."""
        with patch.dict(os.environ, {"MIRDAN_TOOL_BUDGET": "2"}):
            async with server_mod._lifespan(server_mod.mcp):
                remaining = set(server_mod.mcp._tool_manager._tools.keys())
                assert remaining == {"validate_code_quality", "validate_quick"}

    @pytest.mark.asyncio
    async def test_budget_zero_removes_all(self) -> None:
        """MIRDAN_TOOL_BUDGET=0 should remove all tools."""
        with patch.dict(os.environ, {"MIRDAN_TOOL_BUDGET": "0"}):
            async with server_mod._lifespan(server_mod.mcp):
                remaining = set(server_mod.mcp._tool_manager._tools.keys())
                assert len(remaining) == 0

    @pytest.mark.asyncio
    async def test_no_env_var_keeps_all(self) -> None:
        """Without MIRDAN_TOOL_BUDGET, all 6 tools should remain."""
        env = os.environ.copy()
        env.pop("MIRDAN_TOOL_BUDGET", None)
        with patch.dict(os.environ, env, clear=True):
            async with server_mod._lifespan(server_mod.mcp):
                remaining = set(server_mod.mcp._tool_manager._tools.keys())
                assert len(remaining) == 7

    @pytest.mark.asyncio
    async def test_invalid_env_var_keeps_all(self) -> None:
        """Invalid MIRDAN_TOOL_BUDGET should keep all tools (fallback)."""
        with patch.dict(os.environ, {"MIRDAN_TOOL_BUDGET": "not-a-number"}):
            async with server_mod._lifespan(server_mod.mcp):
                remaining = set(server_mod.mcp._tool_manager._tools.keys())
                assert len(remaining) == 7

    @pytest.mark.asyncio
    async def test_budget_three_keeps_top_three(self) -> None:
        """MIRDAN_TOOL_BUDGET=3 should keep top 3 priority tools."""
        with patch.dict(os.environ, {"MIRDAN_TOOL_BUDGET": "3"}):
            async with server_mod._lifespan(server_mod.mcp):
                remaining = set(server_mod.mcp._tool_manager._tools.keys())
                assert remaining == {
                    "validate_code_quality",
                    "validate_quick",
                    "enhance_prompt",
                }

    @pytest.mark.asyncio
    async def test_budget_five_keeps_all(self) -> None:
        """MIRDAN_TOOL_BUDGET=5 should keep all tools."""
        with patch.dict(os.environ, {"MIRDAN_TOOL_BUDGET": "5"}):
            async with server_mod._lifespan(server_mod.mcp):
                remaining = set(server_mod.mcp._tool_manager._tools.keys())
                assert len(remaining) == 5

    @pytest.mark.asyncio
    async def test_empty_string_keeps_all(self) -> None:
        """Empty MIRDAN_TOOL_BUDGET should keep all tools."""
        with patch.dict(os.environ, {"MIRDAN_TOOL_BUDGET": ""}):
            async with server_mod._lifespan(server_mod.mcp):
                remaining = set(server_mod.mcp._tool_manager._tools.keys())
                assert len(remaining) == 7


class TestPlatformProfile:
    """Tests for PlatformProfile model."""

    def test_defaults(self) -> None:
        """PlatformProfile should have correct defaults."""
        profile = PlatformProfile()
        assert profile.name == "generic"
        assert profile.context_level == "auto"
        assert profile.tool_budget_aware is False

    def test_cursor_profile(self) -> None:
        """Cursor platform profile should be configurable."""
        profile = PlatformProfile(
            name="cursor",
            context_level="none",
            tool_budget_aware=True,
        )
        assert profile.name == "cursor"
        assert profile.context_level == "none"
        assert profile.tool_budget_aware is True

    def test_platform_in_config(self) -> None:
        """MirdanConfig should have platform field."""
        config = MirdanConfig()
        assert isinstance(config.platform, PlatformProfile)
        assert config.platform.name == "generic"

    def test_model_dump(self) -> None:
        """PlatformProfile should serialize correctly."""
        profile = PlatformProfile(name="cursor", tool_budget_aware=True)
        data = profile.model_dump()
        assert data["name"] == "cursor"
        assert data["tool_budget_aware"] is True
