"""Tests for M5: Lean enhance_prompt with context_level='none'."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import mirdan.server as server_mod
from mirdan.models import ContextBundle

_enhance_prompt = server_mod.enhance_prompt.fn


@pytest.fixture(autouse=True)
def _reset_components() -> None:
    server_mod._components = None
    yield
    server_mod._components = None


class TestLeanEnhancePrompt:
    """Tests for context_level='none' mode."""

    async def test_skips_context_gathering(self) -> None:
        """context_level='none' should NOT call gather_all."""
        server_mod._get_components()
        with patch.object(
            server_mod._get_components().context_aggregator,
            "gather_all",
            new_callable=AsyncMock,
        ) as mock_gather:
            result = await _enhance_prompt("Create a Python function", context_level="none")
        mock_gather.assert_not_awaited()

    async def test_returns_quality_requirements(self) -> None:
        """Should still include quality_requirements even without context."""
        server_mod._get_components()
        with patch.object(
            server_mod._get_components().context_aggregator,
            "gather_all",
            new_callable=AsyncMock,
        ):
            result = await _enhance_prompt("Write Python code", context_level="none")
        assert "quality_requirements" in result
        assert isinstance(result["quality_requirements"], list)

    async def test_returns_verification_steps(self) -> None:
        """Should still include verification_steps."""
        server_mod._get_components()
        with patch.object(
            server_mod._get_components().context_aggregator,
            "gather_all",
            new_callable=AsyncMock,
        ):
            result = await _enhance_prompt("Write Python code", context_level="none")
        assert "verification_steps" in result

    async def test_returns_session_id(self) -> None:
        """Should still create a session and return session_id."""
        server_mod._get_components()
        with patch.object(
            server_mod._get_components().context_aggregator,
            "gather_all",
            new_callable=AsyncMock,
        ):
            result = await _enhance_prompt("Write Python code", context_level="none")
        assert "session_id" in result
        assert len(result["session_id"]) > 0

    async def test_existing_context_levels_still_work(self) -> None:
        """context_level='auto' should still call gather_all."""
        server_mod._get_components()
        mock_gather = AsyncMock(return_value=ContextBundle())
        with patch.object(
            server_mod._get_components().context_aggregator,
            "gather_all",
            mock_gather,
        ):
            await _enhance_prompt("Write Python code", context_level="auto")
        mock_gather.assert_awaited_once()

    async def test_minimal_context_level(self) -> None:
        """context_level='minimal' should still call gather_all."""
        server_mod._get_components()
        mock_gather = AsyncMock(return_value=ContextBundle())
        with patch.object(
            server_mod._get_components().context_aggregator,
            "gather_all",
            mock_gather,
        ):
            await _enhance_prompt("Write Python code", context_level="minimal")
        mock_gather.assert_awaited_once()
