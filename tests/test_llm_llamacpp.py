"""Tests for LlamaCppBackend with fully mocked llama-cpp-python."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mirdan.models import HealthState, ModelRole

# ---------------------------------------------------------------------------
# Helpers — mock the Llama class so tests run without llama-cpp-python
# ---------------------------------------------------------------------------


def _mock_llama_result(text: str = "hello", completion_tokens: int = 5) -> dict[str, Any]:
    """Build a mock result from Llama.__call__ (text completion)."""
    return {
        "choices": [{"text": text}],
        "usage": {"completion_tokens": completion_tokens},
    }


def _mock_chat_result(content: str = "reply", completion_tokens: int = 3) -> dict[str, Any]:
    """Build a mock result from Llama.create_chat_completion."""
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"completion_tokens": completion_tokens},
    }


def _make_backend(
    model_path: str = "/fake/model.gguf",
    n_ctx: int = 4096,
    keep_alive_seconds: int = 300,
) -> Any:
    """Create a LlamaCppBackend with LLAMACPP_AVAILABLE mocked to True."""
    with patch("mirdan.llm.llamacpp.LLAMACPP_AVAILABLE", True):
        from mirdan.llm.llamacpp import LlamaCppBackend

        return LlamaCppBackend(
            model_path=model_path,
            n_ctx=n_ctx,
            keep_alive_seconds=keep_alive_seconds,
        )


def _make_backend_with_mock_llm(
    model_path: str = "/fake/model.gguf",
) -> tuple[Any, MagicMock]:
    """Create a backend with a pre-injected mock Llama instance (skips lazy load)."""
    backend = _make_backend(model_path=model_path)
    mock_llm = MagicMock()
    backend._llm = mock_llm
    backend._last_use = 0.0
    return backend, mock_llm


# ---------------------------------------------------------------------------
# Module-level availability
# ---------------------------------------------------------------------------


class TestModuleAvailability:
    """Tests for the module-level is_available() function."""

    def test_is_available_returns_false_without_package(self) -> None:
        with patch("mirdan.llm.llamacpp.LLAMACPP_AVAILABLE", False):
            from mirdan.llm.llamacpp import is_available

            assert is_available() is False

    def test_is_available_returns_true_with_package(self) -> None:
        with patch("mirdan.llm.llamacpp.LLAMACPP_AVAILABLE", True):
            from mirdan.llm.llamacpp import is_available

            assert is_available() is True


class TestLlamaCppInit:
    """Tests for LlamaCppBackend constructor."""

    def test_raises_without_llamacpp(self) -> None:
        with patch("mirdan.llm.llamacpp.LLAMACPP_AVAILABLE", False):
            from mirdan.llm.llamacpp import LlamaCppBackend

            with pytest.raises(RuntimeError, match="llama-cpp-python is not installed"):
                LlamaCppBackend(model_path="/fake.gguf")

    def test_stores_config(self) -> None:
        backend = _make_backend(model_path="/my/model.gguf", n_ctx=2048)
        assert backend._model_path == "/my/model.gguf"
        assert backend._n_ctx == 2048
        assert backend._n_threads >= 1
        assert backend._llm is None

    def test_default_threads(self) -> None:
        backend = _make_backend()
        import os

        expected = max(1, (os.cpu_count() or 4) // 2)
        assert backend._n_threads == expected


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


class TestLlamaCppGenerate:
    """Tests for LlamaCppBackend.generate()."""

    @pytest.mark.asyncio
    async def test_returns_content(self) -> None:
        backend, mock_llm = _make_backend_with_mock_llm()
        mock_llm.return_value = _mock_llama_result("generated text", 10)

        result = await backend.generate("prompt", "test-model")

        assert result.content == "generated text"
        assert result.model == "test-model"
        assert result.role == ModelRole.FAST
        assert result.tokens_used == 10
        assert result.elapsed_ms > 0

    @pytest.mark.asyncio
    async def test_respects_max_tokens(self) -> None:
        backend, mock_llm = _make_backend_with_mock_llm()
        mock_llm.return_value = _mock_llama_result()

        await backend.generate("prompt", "m", max_tokens=256)
        mock_llm.assert_called_once_with("prompt", max_tokens=256)

    @pytest.mark.asyncio
    async def test_default_max_tokens(self) -> None:
        backend, mock_llm = _make_backend_with_mock_llm()
        mock_llm.return_value = _mock_llama_result()

        await backend.generate("prompt", "m")
        mock_llm.assert_called_once_with("prompt", max_tokens=512)

    @pytest.mark.asyncio
    async def test_returns_empty_on_runtime_error(self) -> None:
        backend, mock_llm = _make_backend_with_mock_llm()
        mock_llm.side_effect = RuntimeError("model crashed")

        result = await backend.generate("prompt", "m")

        assert result.content == ""
        assert result.tokens_used == 0

    @pytest.mark.asyncio
    async def test_handles_empty_choices(self) -> None:
        backend, mock_llm = _make_backend_with_mock_llm()
        mock_llm.return_value = {"choices": [], "usage": {}}

        result = await backend.generate("prompt", "m")
        assert result.content == ""


# ---------------------------------------------------------------------------
# generate_structured()
# ---------------------------------------------------------------------------


class TestLlamaCppGenerateStructured:
    """Tests for LlamaCppBackend.generate_structured()."""

    @pytest.mark.asyncio
    async def test_returns_parsed_json(self) -> None:
        backend, mock_llm = _make_backend_with_mock_llm()
        structured = {"classification": "local_only", "confidence": 0.9}
        mock_llm.create_chat_completion.return_value = _mock_chat_result(
            json.dumps(structured)
        )

        result = await backend.generate_structured("prompt", "m", schema={"type": "object"})

        assert result == structured

    @pytest.mark.asyncio
    async def test_falls_back_on_invalid_json(self) -> None:
        backend, mock_llm = _make_backend_with_mock_llm()

        # First call (structured) returns invalid JSON
        mock_llm.create_chat_completion.return_value = _mock_chat_result("not json {")
        # Fallback generate call returns valid JSON
        valid_json = '{"fallback": true}'
        mock_llm.return_value = _mock_llama_result(valid_json, 5)

        result = await backend.generate_structured("prompt", "m", schema={})

        assert result == {"fallback": True}

    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_total_failure(self) -> None:
        backend, mock_llm = _make_backend_with_mock_llm()

        # Structured returns invalid JSON
        mock_llm.create_chat_completion.return_value = _mock_chat_result("not json")
        # Fallback also returns invalid JSON
        mock_llm.return_value = _mock_llama_result("still not json", 0)

        result = await backend.generate_structured("prompt", "m", schema={})
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_runtime_error(self) -> None:
        backend, mock_llm = _make_backend_with_mock_llm()
        mock_llm.create_chat_completion.side_effect = RuntimeError("crashed")

        result = await backend.generate_structured("prompt", "m", schema={})
        assert result == {}


# ---------------------------------------------------------------------------
# chat()
# ---------------------------------------------------------------------------


class TestLlamaCppChat:
    """Tests for LlamaCppBackend.chat()."""

    @pytest.mark.asyncio
    async def test_returns_message_content(self) -> None:
        backend, mock_llm = _make_backend_with_mock_llm()
        mock_llm.create_chat_completion.return_value = _mock_chat_result("hello!", 7)

        messages = [{"role": "user", "content": "hi"}]
        result = await backend.chat(messages, "m")

        assert result.content == "hello!"
        assert result.tokens_used == 7

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self) -> None:
        backend, mock_llm = _make_backend_with_mock_llm()
        mock_llm.create_chat_completion.side_effect = OSError("disk error")

        result = await backend.chat([{"role": "user", "content": "hi"}], "m")

        assert result.content == ""
        assert result.tokens_used == 0


# ---------------------------------------------------------------------------
# chat_with_tools()
# ---------------------------------------------------------------------------


class TestLlamaCppChatWithTools:
    """Tests for LlamaCppBackend.chat_with_tools()."""

    @pytest.mark.asyncio
    async def test_injects_tool_descriptions(self) -> None:
        backend, mock_llm = _make_backend_with_mock_llm()
        mock_llm.create_chat_completion.return_value = _mock_chat_result("tool result")

        tools = [{"name": "search", "description": "search code"}]
        messages = [{"role": "user", "content": "find it"}]
        result = await backend.chat_with_tools(messages, "m", tools)

        assert result.content == "tool result"
        # Verify system message was prepended
        call_args = mock_llm.create_chat_completion.call_args
        sent_messages = call_args[1]["messages"]
        assert sent_messages[0]["role"] == "system"
        assert "search" in sent_messages[0]["content"]
        assert sent_messages[1]["content"] == "find it"


# ---------------------------------------------------------------------------
# is_available()
# ---------------------------------------------------------------------------


class TestLlamaCppIsAvailable:
    """Tests for LlamaCppBackend.is_available()."""

    @pytest.mark.asyncio
    async def test_true_when_package_and_file_exist(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"fake model data")

        backend = _make_backend(model_path=str(model_file))

        with patch("mirdan.llm.llamacpp.LLAMACPP_AVAILABLE", True):
            assert await backend.is_available() is True

    @pytest.mark.asyncio
    async def test_false_when_file_missing(self) -> None:
        backend = _make_backend(model_path="/nonexistent/model.gguf")

        with patch("mirdan.llm.llamacpp.LLAMACPP_AVAILABLE", True):
            assert await backend.is_available() is False

    @pytest.mark.asyncio
    async def test_false_when_package_missing(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"fake")

        backend = _make_backend(model_path=str(model_file))

        with patch("mirdan.llm.llamacpp.LLAMACPP_AVAILABLE", False):
            assert await backend.is_available() is False


# ---------------------------------------------------------------------------
# list_models()
# ---------------------------------------------------------------------------


class TestLlamaCppListModels:
    """Tests for LlamaCppBackend.list_models()."""

    @pytest.mark.asyncio
    async def test_returns_model_info(self, tmp_path: Path) -> None:
        model_file = tmp_path / "gemma-4-E4B-it-Q3_K_M.gguf"
        model_file.write_bytes(b"x" * (4000 * 1024 * 1024))  # ~4000 MB

        backend = _make_backend(model_path=str(model_file))

        with patch("mirdan.llm.llamacpp.LLAMACPP_AVAILABLE", True):
            models = await backend.list_models()

        assert len(models) == 1
        assert models[0].name == "gemma-4-E4B-it-Q3_K_M"
        assert models[0].role == ModelRole.FAST
        # active_memory = file_size_mb + 300 KV cache
        assert models[0].active_memory_mb == 4000 + 300

    @pytest.mark.asyncio
    async def test_returns_empty_when_unavailable(self) -> None:
        backend = _make_backend(model_path="/nonexistent.gguf")

        with patch("mirdan.llm.llamacpp.LLAMACPP_AVAILABLE", True):
            assert await backend.list_models() == []


# ---------------------------------------------------------------------------
# health()
# ---------------------------------------------------------------------------


class TestLlamaCppHealth:
    """Tests for LlamaCppBackend.health()."""

    @pytest.mark.asyncio
    async def test_available(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"x" * 1024 * 1024)

        backend = _make_backend(model_path=str(model_file))

        with patch("mirdan.llm.llamacpp.LLAMACPP_AVAILABLE", True):
            health = await backend.health()

        assert health.state == HealthState.AVAILABLE
        assert health.models_loaded == ["model"]

    @pytest.mark.asyncio
    async def test_unavailable(self) -> None:
        backend = _make_backend(model_path="/nonexistent.gguf")

        with patch("mirdan.llm.llamacpp.LLAMACPP_AVAILABLE", True):
            health = await backend.health()

        assert health.state == HealthState.UNAVAILABLE
        assert health.models_loaded == []


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestLlamaCppClose:
    """Tests for LlamaCppBackend.close()."""

    @pytest.mark.asyncio
    async def test_close_releases_model(self) -> None:
        backend, mock_llm = _make_backend_with_mock_llm()
        await backend.close()

        mock_llm.close.assert_called_once()
        assert backend._llm is None

    @pytest.mark.asyncio
    async def test_close_cancels_unload_task(self) -> None:
        backend, _ = _make_backend_with_mock_llm()
        mock_task = MagicMock()
        mock_task.done.return_value = False
        backend._unload_task = mock_task

        await backend.close()

        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_noop_when_no_model(self) -> None:
        backend = _make_backend()
        await backend.close()  # Should not raise


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------


class TestLlamaCppLazyLoading:
    """Tests for lazy model loading via _ensure_loaded()."""

    @pytest.mark.asyncio
    async def test_loads_model_on_first_generate(self) -> None:
        backend = _make_backend()
        mock_llama_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.return_value = _mock_llama_result("loaded!", 1)
        mock_llama_cls.return_value = mock_instance

        with patch("mirdan.llm.llamacpp.Llama", mock_llama_cls):
            result = await backend.generate("test", "m")

        assert result.content == "loaded!"
        mock_llama_cls.assert_called_once_with(
            model_path=backend._model_path,
            n_ctx=backend._n_ctx,
            n_threads=backend._n_threads,
            verbose=False,
        )

    @pytest.mark.asyncio
    async def test_reuses_loaded_model(self) -> None:
        backend, mock_llm = _make_backend_with_mock_llm()
        mock_llm.return_value = _mock_llama_result("ok", 1)

        await backend.generate("first", "m")
        await backend.generate("second", "m")

        # Model __call__ invoked twice, not reloaded
        assert mock_llm.call_count == 2
