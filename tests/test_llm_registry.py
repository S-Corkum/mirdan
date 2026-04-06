"""Tests for ModelRegistry and ModelSelector."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mirdan.llm.registry import KNOWN_MODELS, ModelRegistry, ModelSelector
from mirdan.models import ModelInfo, ModelRole


# ---------------------------------------------------------------------------
# ModelRegistry.discover()
# ---------------------------------------------------------------------------


class TestModelRegistryDiscover:
    """Tests for discovering installed models."""

    @pytest.mark.asyncio
    async def test_discovers_ollama_model(self) -> None:
        backend = AsyncMock()
        backend.list_models.return_value = [
            ModelInfo(name="gemma4:e2b", role=ModelRole.FAST, active_memory_mb=3500, quality_score=0.0)
        ]

        registry = ModelRegistry()
        installed = await registry.discover(backend=backend, gguf_dir="/nonexistent")

        assert len(installed) == 1
        assert installed[0].name == "gemma4-e2b-q4"
        assert installed[0].quality_score == 0.60
        assert installed[0].active_memory_mb == 3500

    @pytest.mark.asyncio
    async def test_discovers_gguf_model(self, tmp_path: Path) -> None:
        gguf_dir = tmp_path / "models"
        gguf_dir.mkdir()
        (gguf_dir / "gemma-4-E4B-it-Q3_K_M.gguf").write_bytes(b"fake")

        registry = ModelRegistry()
        installed = await registry.discover(backend=None, gguf_dir=str(gguf_dir))

        assert len(installed) == 1
        assert installed[0].name == "gemma4-e4b-q3"
        assert installed[0].quality_score == 0.67

    @pytest.mark.asyncio
    async def test_discovers_both_sources(self, tmp_path: Path) -> None:
        backend = AsyncMock()
        backend.list_models.return_value = [
            ModelInfo(name="gemma4:e2b", role=ModelRole.FAST, active_memory_mb=0, quality_score=0.0)
        ]

        gguf_dir = tmp_path / "models"
        gguf_dir.mkdir()
        (gguf_dir / "gemma-4-E4B-it-Q3_K_M.gguf").write_bytes(b"fake")

        registry = ModelRegistry()
        installed = await registry.discover(backend=backend, gguf_dir=str(gguf_dir))

        names = {m.name for m in installed}
        assert "gemma4-e2b-q4" in names
        assert "gemma4-e4b-q3" in names

    @pytest.mark.asyncio
    async def test_returns_empty_when_nothing_installed(self) -> None:
        backend = AsyncMock()
        backend.list_models.return_value = []

        registry = ModelRegistry()
        installed = await registry.discover(backend=backend, gguf_dir="/nonexistent")

        assert installed == []

    @pytest.mark.asyncio
    async def test_handles_backend_failure(self) -> None:
        backend = AsyncMock()
        backend.list_models.side_effect = Exception("connection refused")

        registry = ModelRegistry()
        installed = await registry.discover(backend=backend, gguf_dir="/nonexistent")

        assert installed == []

    @pytest.mark.asyncio
    async def test_ignores_unknown_ollama_tags(self) -> None:
        backend = AsyncMock()
        backend.list_models.return_value = [
            ModelInfo(name="llama3:8b", role=ModelRole.FAST, active_memory_mb=5000, quality_score=0.0)
        ]

        registry = ModelRegistry()
        installed = await registry.discover(backend=backend, gguf_dir="/nonexistent")

        assert installed == []

    @pytest.mark.asyncio
    async def test_ignores_unknown_gguf_files(self, tmp_path: Path) -> None:
        gguf_dir = tmp_path / "models"
        gguf_dir.mkdir()
        (gguf_dir / "random-model.gguf").write_bytes(b"fake")

        registry = ModelRegistry()
        installed = await registry.discover(backend=None, gguf_dir=str(gguf_dir))

        assert installed == []

    @pytest.mark.asyncio
    async def test_sets_correct_role(self) -> None:
        backend = AsyncMock()
        backend.list_models.return_value = [
            ModelInfo(name="gemma4:31b", role=ModelRole.FAST, active_memory_mb=0, quality_score=0.0),
            ModelInfo(name="gemma4:e2b", role=ModelRole.FAST, active_memory_mb=0, quality_score=0.0),
        ]

        registry = ModelRegistry()
        installed = await registry.discover(backend=backend, gguf_dir="/nonexistent")

        brain = [m for m in installed if m.role == ModelRole.BRAIN]
        fast = [m for m in installed if m.role == ModelRole.FAST]
        assert len(brain) == 1
        assert brain[0].name == "gemma4-31b-q4"
        assert len(fast) == 1


# ---------------------------------------------------------------------------
# ModelRegistry.detect_capabilities()
# ---------------------------------------------------------------------------


class TestModelRegistryCapabilities:
    """Tests for detect_capabilities()."""

    @pytest.mark.asyncio
    async def test_detects_gemma_capabilities(self) -> None:
        backend = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model_info": {"general.architecture": "gemma2"},
            "template": "{{ if .Tools }}tool_call{{ end }}",
        }
        mock_resp.raise_for_status = MagicMock()
        backend._client.post = AsyncMock(return_value=mock_resp)

        registry = ModelRegistry()
        caps = await registry.detect_capabilities(backend, "gemma4:e2b")

        assert caps["supports_structured_output"] is True
        assert caps["supports_thinking"] is True
        assert caps["supports_tools"] is True

    @pytest.mark.asyncio
    async def test_returns_defaults_on_failure(self) -> None:
        backend = AsyncMock()
        backend._client.post.side_effect = Exception("connection refused")

        registry = ModelRegistry()
        caps = await registry.detect_capabilities(backend, "test")

        assert caps["supports_tools"] is False
        assert caps["supports_structured_output"] is False
        assert caps["supports_thinking"] is False


# ---------------------------------------------------------------------------
# ModelSelector.select()
# ---------------------------------------------------------------------------


class TestModelSelector:
    """Tests for dynamic model selection."""

    def _make_installed(self) -> list[ModelInfo]:
        return [
            ModelInfo(name="gemma4-e4b-q3", role=ModelRole.FAST, active_memory_mb=4500, quality_score=0.67),
            ModelInfo(name="gemma4-e2b-q4", role=ModelRole.FAST, active_memory_mb=3500, quality_score=0.60),
            ModelInfo(name="gemma4-e2b-q3", role=ModelRole.FAST, active_memory_mb=3200, quality_score=0.58),
            ModelInfo(name="gemma4-31b-q4", role=ModelRole.BRAIN, active_memory_mb=17000, quality_score=0.85),
        ]

    def test_selects_highest_quality_that_fits(self) -> None:
        selector = ModelSelector(installed_models=self._make_installed())
        # 8000 MB available - 2048 buffer = 5952 budget → E4B Q3 (4500) fits
        result = selector.select(ModelRole.FAST, available_memory_mb=8000)

        assert result is not None
        assert result.name == "gemma4-e4b-q3"

    def test_falls_back_to_smaller_model(self) -> None:
        selector = ModelSelector(installed_models=self._make_installed())
        # 5500 MB available - 2048 buffer = 3452 budget → E4B Q3 (4500) doesn't fit, E2B Q4 (3500) doesn't fit, E2B Q3 (3200) fits
        result = selector.select(ModelRole.FAST, available_memory_mb=5500)

        assert result is not None
        assert result.name == "gemma4-e2b-q3"

    def test_returns_none_when_nothing_fits(self) -> None:
        selector = ModelSelector(installed_models=self._make_installed())
        # 4000 MB available - 2048 buffer = 1952 budget → nothing fits
        result = selector.select(ModelRole.FAST, available_memory_mb=4000)

        assert result is None

    def test_brain_requires_arm64(self) -> None:
        selector = ModelSelector(installed_models=self._make_installed())
        result = selector.select(ModelRole.BRAIN, available_memory_mb=30000, architecture="x86_64")

        assert result is None

    def test_brain_works_on_arm64(self) -> None:
        selector = ModelSelector(installed_models=self._make_installed())
        result = selector.select(ModelRole.BRAIN, available_memory_mb=30000, architecture="arm64")

        assert result is not None
        assert result.name == "gemma4-31b-q4"

    def test_brain_returns_none_when_not_enough_memory(self) -> None:
        selector = ModelSelector(installed_models=self._make_installed())
        result = selector.select(ModelRole.BRAIN, available_memory_mb=15000, architecture="arm64")

        assert result is None

    def test_returns_none_with_no_installed_models(self) -> None:
        selector = ModelSelector(installed_models=[])
        result = selector.select(ModelRole.FAST, available_memory_mb=10000)

        assert result is None

    def test_update_installed(self) -> None:
        selector = ModelSelector(installed_models=[])
        assert selector.select(ModelRole.FAST, available_memory_mb=10000) is None

        selector.update_installed(self._make_installed())
        assert selector.select(ModelRole.FAST, available_memory_mb=10000) is not None

    def test_e2b_q4_selected_at_6gb(self) -> None:
        selector = ModelSelector(installed_models=self._make_installed())
        # 6000 MB - 2048 buffer = 3952 → E4B (4500) too big, E2B Q4 (3500) fits
        result = selector.select(ModelRole.FAST, available_memory_mb=6000)

        assert result is not None
        assert result.name == "gemma4-e2b-q4"
