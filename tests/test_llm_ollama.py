"""Tests for OllamaBackend with mocked HTTP responses."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from mirdan.llm.ollama import OllamaBackend
from mirdan.models import HealthState, ModelRole


def _make_response(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
) -> httpx.Response:
    """Build a mock httpx.Response."""
    content = json.dumps(json_data or {}).encode()
    return httpx.Response(status_code=status_code, content=content)


def _mock_transport(handler: Any) -> httpx.AsyncClient:
    """Create an AsyncClient with a MockTransport and base_url."""
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://localhost:11434",
    )


class TestOllamaGenerate:
    """Tests for OllamaBackend.generate()."""

    @pytest.mark.asyncio
    async def test_returns_content_from_response(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/generate"
            body = json.loads(request.content)
            assert body["model"] == "gemma4:e2b"
            assert body["prompt"] == "Hello"
            assert body["stream"] is False
            return _make_response(json_data={"response": "Hi there!", "eval_count": 12})

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        result = await backend.generate("Hello", "gemma4:e2b")

        assert result.content == "Hi there!"
        assert result.model == "gemma4:e2b"
        assert result.role == ModelRole.FAST
        assert result.tokens_used == 12
        assert result.elapsed_ms > 0

    @pytest.mark.asyncio
    async def test_passes_keep_alive(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["keep_alive"] == "10m"
            return _make_response(json_data={"response": "ok", "eval_count": 1})

        backend = OllamaBackend(keep_alive="10m")
        backend._client = _mock_transport(handler)
        await backend.generate("test", "m")

    @pytest.mark.asyncio
    async def test_passes_kwargs_to_api(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["temperature"] == 0.0
            return _make_response(json_data={"response": "ok", "eval_count": 1})

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        await backend.generate("test", "m", temperature=0.0)

    @pytest.mark.asyncio
    async def test_returns_empty_on_timeout(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out")

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        result = await backend.generate("test", "m")

        assert result.content == ""
        assert result.tokens_used == 0

    @pytest.mark.asyncio
    async def test_returns_empty_on_connect_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        result = await backend.generate("test", "m")

        assert result.content == ""
        assert result.tokens_used == 0

    @pytest.mark.asyncio
    async def test_returns_empty_on_http_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return _make_response(status_code=500)

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        result = await backend.generate("test", "m")

        assert result.content == ""


class TestOllamaGenerateStructured:
    """Tests for OllamaBackend.generate_structured()."""

    @pytest.mark.asyncio
    async def test_returns_parsed_json(self) -> None:
        structured = {"classification": "local_only", "confidence": 0.9}

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["format"] == {"type": "object"}
            return _make_response(json_data={"response": json.dumps(structured)})

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        result = await backend.generate_structured("test", "m", schema={"type": "object"})

        assert result == structured

    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_invalid_json(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return _make_response(json_data={"response": "not valid json {"})

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        result = await backend.generate_structured("test", "m", schema={})

        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_timeout(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out")

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        result = await backend.generate_structured("test", "m", schema={})

        assert result == {}


class TestOllamaChat:
    """Tests for OllamaBackend.chat()."""

    @pytest.mark.asyncio
    async def test_returns_message_content(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/chat"
            body = json.loads(request.content)
            assert body["messages"] == [{"role": "user", "content": "hi"}]
            assert body["stream"] is False
            return _make_response(json_data={"message": {"content": "hello!"}, "eval_count": 5})

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        result = await backend.chat([{"role": "user", "content": "hi"}], "m")

        assert result.content == "hello!"
        assert result.tokens_used == 5

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        result = await backend.chat([{"role": "user", "content": "hi"}], "m")

        assert result.content == ""


class TestOllamaChatWithTools:
    """Tests for OllamaBackend.chat_with_tools()."""

    @pytest.mark.asyncio
    async def test_passes_tools_to_chat(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["tools"] == [{"name": "search"}]
            return _make_response(json_data={"message": {"content": "used tool"}, "eval_count": 3})

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        result = await backend.chat_with_tools(
            [{"role": "user", "content": "find it"}], "m", tools=[{"name": "search"}]
        )

        assert result.content == "used tool"


class TestOllamaIsAvailable:
    """Tests for OllamaBackend.is_available()."""

    @pytest.mark.asyncio
    async def test_returns_true_on_200(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/tags"
            return _make_response(json_data={"models": []})

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        assert await backend.is_available() is True

    @pytest.mark.asyncio
    async def test_returns_false_on_timeout(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timeout")

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        assert await backend.is_available() is False

    @pytest.mark.asyncio
    async def test_returns_false_on_connect_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        assert await backend.is_available() is False


class TestOllamaListModels:
    """Tests for OllamaBackend.list_models()."""

    @pytest.mark.asyncio
    async def test_parses_models_list(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return _make_response(
                json_data={
                    "models": [
                        {"name": "gemma4:e2b", "size": 3_500_000_000},
                        {"name": "gemma4:e4b", "size": 5_000_000_000},
                    ]
                }
            )

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        models = await backend.list_models()

        assert len(models) == 2
        assert models[0].name == "gemma4:e2b"
        assert models[0].active_memory_mb == 3500
        assert models[0].role == ModelRole.FAST
        assert models[1].name == "gemma4:e4b"
        assert models[1].active_memory_mb == 5000

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        assert await backend.list_models() == []

    @pytest.mark.asyncio
    async def test_handles_missing_size(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return _make_response(json_data={"models": [{"name": "test-model"}]})

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        models = await backend.list_models()

        assert len(models) == 1
        assert models[0].active_memory_mb == 0


class TestOllamaHealth:
    """Tests for OllamaBackend.health()."""

    @pytest.mark.asyncio
    async def test_available_with_models(self) -> None:
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _make_response(
                json_data={"models": [{"name": "gemma4:e2b", "size": 3_500_000_000}]}
            )

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        health = await backend.health()

        assert health.state == HealthState.AVAILABLE
        assert health.models_loaded == ["gemma4:e2b"]

    @pytest.mark.asyncio
    async def test_unavailable_on_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        backend = OllamaBackend()
        backend._client = _mock_transport(handler)
        health = await backend.health()

        assert health.state == HealthState.UNAVAILABLE
        assert health.models_loaded == []


class TestOllamaClose:
    """Tests for OllamaBackend.close()."""

    @pytest.mark.asyncio
    async def test_close_calls_aclose(self) -> None:
        backend = OllamaBackend()
        backend._client = AsyncMock()
        await backend.close()
        backend._client.aclose.assert_awaited_once()


class TestOllamaInit:
    """Tests for OllamaBackend constructor."""

    def test_default_values(self) -> None:
        backend = OllamaBackend()
        assert backend._base_url == "http://localhost:11434"
        assert backend._timeout == 30.0
        assert backend._keep_alive == "5m"

    def test_custom_values(self) -> None:
        backend = OllamaBackend(
            base_url="http://127.0.0.1:9999/",
            timeout=10.0,
            keep_alive="10m",
        )
        assert backend._base_url == "http://127.0.0.1:9999"
        assert backend._timeout == 10.0
        assert backend._keep_alive == "10m"

    def test_strips_trailing_slash(self) -> None:
        backend = OllamaBackend(base_url="http://localhost:11434/")
        assert backend._base_url == "http://localhost:11434"

    def test_rejects_remote_url(self) -> None:
        with pytest.raises(ValueError, match="localhost"):
            OllamaBackend(base_url="http://evil.com:11434")

    def test_rejects_internal_ip(self) -> None:
        with pytest.raises(ValueError, match="localhost"):
            OllamaBackend(base_url="http://192.168.1.100:11434")

    def test_accepts_ipv6_loopback(self) -> None:
        backend = OllamaBackend(base_url="http://[::1]:11434")
        assert backend._base_url == "http://[::1]:11434"
