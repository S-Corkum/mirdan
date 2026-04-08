"""Tests for LLMManager facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirdan.config import LLMConfig
from mirdan.llm.health import HealthMonitor
from mirdan.llm.manager import LLMManager
from mirdan.llm.registry import ModelSelector
from mirdan.models import (
    HardwareInfo,
    HardwareProfile,
    HealthState,
    LLMResponse,
    ModelInfo,
    ModelRole,
)

# ---------------------------------------------------------------------------
# create_if_enabled
# ---------------------------------------------------------------------------


class TestCreateIfEnabled:
    """Tests for LLMManager.create_if_enabled()."""

    def test_returns_none_when_disabled(self) -> None:
        config = LLMConfig(enabled=False)
        assert LLMManager.create_if_enabled(config) is None

    def test_returns_manager_when_enabled(self) -> None:
        config = LLMConfig(enabled=True)
        manager = LLMManager.create_if_enabled(config)
        assert isinstance(manager, LLMManager)


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------


class TestStartup:
    """Tests for LLMManager.startup()."""

    @pytest.mark.asyncio
    async def test_startup_with_unavailable_backend(self) -> None:
        config = LLMConfig(enabled=True, backend="ollama")
        manager = LLMManager(config)

        mock_backend = AsyncMock()
        mock_backend.is_available.return_value = False

        with (
            patch.object(manager, "_create_backend", return_value=mock_backend),
            patch("mirdan.llm.manager.HardwareDetector") as mock_hw,
        ):
            mock_hw.detect.return_value = HardwareInfo(
                architecture="x86_64",
                total_ram_mb=16384,
                available_ram_mb=8000,
                detected_profile=HardwareProfile.STANDARD,
            )
            await manager.startup()

        # Backend was checked but no warmup or sidecar started
        mock_backend.is_available.assert_awaited_once()
        assert manager._health is None

    @pytest.mark.asyncio
    async def test_startup_cleans_stale_port_file(self, tmp_path: Path) -> None:
        config = LLMConfig(enabled=True)
        manager = LLMManager(config)

        port_file = tmp_path / ".mirdan" / "sidecar.port"
        port_file.parent.mkdir(parents=True)
        port_file.write_text("12345")

        mock_backend = AsyncMock()
        mock_backend.is_available.return_value = False

        with (
            patch.object(manager, "_create_backend", return_value=mock_backend),
            patch("mirdan.llm.manager.HardwareDetector") as mock_hw,
            patch("mirdan.llm.manager.Path") as mock_path_cls,
        ):
            # Make the Path(".mirdan/sidecar.port") point to our tmp file
            mock_port_path = MagicMock()
            mock_port_path.exists.return_value = True
            mock_path_cls.return_value = mock_port_path
            mock_hw.detect.return_value = HardwareInfo(
                architecture="x86_64",
                total_ram_mb=16384,
                available_ram_mb=8000,
                detected_profile=HardwareProfile.STANDARD,
            )

            await manager.startup()

        mock_port_path.unlink.assert_called_once()


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


class TestGenerate:
    """Tests for LLMManager.generate()."""

    def _setup_manager(self) -> tuple[LLMManager, AsyncMock]:
        config = LLMConfig(enabled=True)
        manager = LLMManager(config)

        mock_backend = AsyncMock()
        manager._backend = mock_backend

        manager._hardware = HardwareInfo(
            architecture="arm64",
            total_ram_mb=16384,
            available_ram_mb=8000,
            detected_profile=HardwareProfile.STANDARD,
        )

        manager._health = HealthMonitor(hardware=manager._hardware)
        manager._health.transition(HealthState.AVAILABLE)

        model = ModelInfo(
            name="gemma4-e2b-q4",
            role=ModelRole.FAST,
            active_memory_mb=3500,
            quality_score=0.60,
            model_family="gemma4",
        )
        manager._selector = ModelSelector(installed_models=[model])

        return manager, mock_backend

    @pytest.mark.asyncio
    async def test_returns_none_when_not_available(self) -> None:
        config = LLMConfig(enabled=True)
        manager = LLMManager(config)
        # No health monitor set up
        result = await manager.generate(ModelRole.FAST, "test prompt")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_warming_up(self) -> None:
        manager, _ = self._setup_manager()
        assert manager._health is not None
        manager._health.transition(HealthState.WARMING_UP)

        result = await manager.generate(ModelRole.FAST, "test")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_response_when_available(self) -> None:
        manager, mock_backend = self._setup_manager()
        expected = LLMResponse(
            content="generated", model="gemma4-e2b-q4", role=ModelRole.FAST,
            elapsed_ms=50.0, tokens_used=10,
        )
        mock_backend.generate.return_value = expected

        with patch("mirdan.llm.manager.HardwareDetector.get_available_memory_mb", return_value=8000):
            result = await manager.generate(ModelRole.FAST, "test prompt")

        assert result is not None
        assert result.content == "generated"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_models_installed(self) -> None:
        manager, _ = self._setup_manager()
        # Clear all installed models — truly nothing available
        manager._selector.update_installed([])

        with patch("mirdan.llm.manager.HardwareDetector.get_available_memory_mb", return_value=8000):
            result = await manager.generate(ModelRole.FAST, "test")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_timeout(self) -> None:
        manager, mock_backend = self._setup_manager()

        async def slow_generate(*args: Any, **kwargs: Any) -> LLMResponse:
            import asyncio
            await asyncio.sleep(100)
            return LLMResponse(content="", model="m", role=ModelRole.FAST, elapsed_ms=0, tokens_used=0)

        mock_backend.generate.side_effect = slow_generate

        # Replace health monitor with a mock that returns a very short timeout
        mock_health = MagicMock()
        mock_health.state = HealthState.AVAILABLE
        mock_health.effective_timeout = 0.01
        manager._health = mock_health

        with patch("mirdan.llm.manager.HardwareDetector.get_available_memory_mb", return_value=50000):
            result = await manager.generate(ModelRole.FAST, "test")

        assert result is None


# ---------------------------------------------------------------------------
# generate_structured
# ---------------------------------------------------------------------------


class TestGenerateStructured:
    """Tests for LLMManager.generate_structured()."""

    def _setup_manager(self) -> tuple[LLMManager, AsyncMock]:
        config = LLMConfig(enabled=True)
        manager = LLMManager(config)

        mock_backend = AsyncMock()
        manager._backend = mock_backend

        manager._hardware = HardwareInfo(
            architecture="arm64",
            total_ram_mb=16384,
            available_ram_mb=8000,
            detected_profile=HardwareProfile.STANDARD,
        )

        manager._health = HealthMonitor(hardware=manager._hardware)
        manager._health.transition(HealthState.AVAILABLE)

        model = ModelInfo(
            name="gemma4-e2b-q4",
            role=ModelRole.FAST,
            active_memory_mb=3500,
            quality_score=0.60,
            model_family="gemma4",
        )
        manager._selector = ModelSelector(installed_models=[model])

        return manager, mock_backend

    @pytest.mark.asyncio
    async def test_returns_structured_response(self) -> None:
        manager, mock_backend = self._setup_manager()
        mock_backend.generate_structured.return_value = {"key": "value"}

        with patch("mirdan.llm.manager.HardwareDetector.get_available_memory_mb", return_value=8000):
            # Non-Ollama backend, so format param is used directly
            result = await manager.generate_structured(
                ModelRole.FAST, "prompt", schema={"type": "object"}
            )

        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_returns_none_when_not_available(self) -> None:
        config = LLMConfig(enabled=True)
        manager = LLMManager(config)

        result = await manager.generate_structured(
            ModelRole.FAST, "prompt", schema={}
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_gemma4_ollama_fallback(self) -> None:
        """Gemma 4 + Ollama + thinking=False → text fallback."""
        manager, mock_backend = self._setup_manager()
        mock_backend.generate.return_value = LLMResponse(
            content='{"result": "ok"}',
            model="gemma4-e2b-q4",
            role=ModelRole.FAST,
            elapsed_ms=10.0,
            tokens_used=5,
        )

        with (
            patch("mirdan.llm.manager.HardwareDetector.get_available_memory_mb", return_value=8000),
            patch.object(manager, "_is_ollama_backend", return_value=True),
        ):
            result = await manager.generate_structured(
                ModelRole.FAST, "prompt", schema={"type": "object"}, thinking=False
            )

        assert result == {"result": "ok"}
        # Should have called generate (text), not generate_structured
        mock_backend.generate.assert_awaited_once()
        mock_backend.generate_structured.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gemma4_with_thinking_uses_format(self) -> None:
        """Gemma 4 + thinking=True → format param works fine."""
        manager, mock_backend = self._setup_manager()
        mock_backend.generate_structured.return_value = {"result": "ok"}

        with (
            patch("mirdan.llm.manager.HardwareDetector.get_available_memory_mb", return_value=8000),
            patch.object(manager, "_is_ollama_backend", return_value=True),
        ):
            result = await manager.generate_structured(
                ModelRole.FAST, "prompt", schema={}, thinking=True
            )

        assert result == {"result": "ok"}
        mock_backend.generate_structured.assert_awaited_once()


# ---------------------------------------------------------------------------
# health & shutdown
# ---------------------------------------------------------------------------


class TestHealthAndShutdown:
    """Tests for health() and shutdown()."""

    @pytest.mark.asyncio
    async def test_health_without_monitor(self) -> None:
        manager = LLMManager(LLMConfig(enabled=True))
        health = await manager.health()
        assert health.state == HealthState.UNAVAILABLE

    @pytest.mark.asyncio
    async def test_health_with_monitor(self) -> None:
        manager = LLMManager(LLMConfig(enabled=True))
        manager._health = HealthMonitor()
        manager._health.transition(HealthState.AVAILABLE)

        health = await manager.health()
        assert health.state == HealthState.AVAILABLE

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up(self) -> None:
        manager = LLMManager(LLMConfig(enabled=True))
        manager._sidecar = AsyncMock()
        manager._health = HealthMonitor()
        manager._backend = AsyncMock()

        await manager.shutdown()

        manager._sidecar.stop.assert_awaited_once()
        manager._backend.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# parse_keep_alive
# ---------------------------------------------------------------------------


class TestParseKeepAlive:
    """Tests for _parse_keep_alive."""

    def test_minutes(self) -> None:
        config = LLMConfig(enabled=True, model_keep_alive="5m")
        manager = LLMManager(config)
        assert manager._parse_keep_alive() == 300

    def test_seconds(self) -> None:
        config = LLMConfig(enabled=True, model_keep_alive="30s")
        manager = LLMManager(config)
        assert manager._parse_keep_alive() == 30

    def test_raw_int(self) -> None:
        config = LLMConfig(enabled=True, model_keep_alive="600")
        manager = LLMManager(config)
        assert manager._parse_keep_alive() == 600

    def test_fallback(self) -> None:
        config = LLMConfig(enabled=True, model_keep_alive="invalid")
        manager = LLMManager(config)
        assert manager._parse_keep_alive() == 300
