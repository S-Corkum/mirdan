"""Ollama HTTP backend for local LLM inference."""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from mirdan.models import HealthState, LLMHealth, LLMResponse, ModelInfo, ModelRole

logger = logging.getLogger(__name__)


class OllamaBackend:
    """Connects to a local Ollama daemon via REST API.

    Implements LocalLLMProtocol. All methods return graceful defaults
    on connection failure — callers never see exceptions from this class.
    """

    _ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        keep_alive: str = "5m",
    ) -> None:
        # Defense in depth: reject non-localhost URLs even if config validation is bypassed
        parsed = urlparse(base_url)
        hostname = (parsed.hostname or "").lower()
        if hostname not in self._ALLOWED_HOSTS:
            raise ValueError(
                f"OllamaBackend only connects to localhost (got: {hostname!r}). "
                "Remote Ollama endpoints are not supported."
            )
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._keep_alive = keep_alive
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

    async def generate(self, prompt: str, model: str, **kwargs: Any) -> LLMResponse:
        """Generate a text completion via /api/generate.

        Args:
            prompt: The prompt text.
            model: Ollama model tag (e.g. "gemma4:e2b").
            **kwargs: Extra fields passed to the Ollama API.

        Returns:
            LLMResponse with generated content, or empty on failure.
        """
        start = time.monotonic()
        try:
            resp = await self._client.post(
                "/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": self._keep_alive,
                    **kwargs,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.monotonic() - start) * 1000
            return LLMResponse(
                content=data.get("response", ""),
                model=model,
                role=ModelRole.FAST,
                elapsed_ms=elapsed,
                tokens_used=data.get("eval_count", 0),
            )
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
            logger.warning("Ollama generate failed: %s", exc)
            elapsed = (time.monotonic() - start) * 1000
            return LLMResponse(
                content="", model=model, role=ModelRole.FAST, elapsed_ms=elapsed, tokens_used=0
            )

    async def generate_structured(
        self, prompt: str, model: str, schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """Generate a structured JSON response via /api/generate with format param.

        Args:
            prompt: The prompt text.
            model: Ollama model tag.
            schema: JSON schema passed as the ``format`` parameter.
            **kwargs: Extra fields passed to the Ollama API.

        Returns:
            Parsed dict from the model's JSON output, or empty dict on failure.
        """
        start = time.monotonic()
        try:
            resp = await self._client.post(
                "/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": schema,
                    "keep_alive": self._keep_alive,
                    **kwargs,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            parsed: dict[str, Any] = json.loads(data.get("response", "{}"))
            return parsed
        except (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.HTTPStatusError,
            json.JSONDecodeError,
        ) as exc:
            logger.warning("Ollama generate_structured failed: %s", exc)
            return {}

    async def chat(self, messages: list[dict[str, Any]], model: str, **kwargs: Any) -> LLMResponse:
        """Multi-turn chat via /api/chat.

        Args:
            messages: Chat messages in OpenAI-style format.
            model: Ollama model tag.
            **kwargs: Extra fields passed to the Ollama API.

        Returns:
            LLMResponse with assistant reply, or empty on failure.
        """
        start = time.monotonic()
        try:
            resp = await self._client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": self._keep_alive,
                    **kwargs,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.monotonic() - start) * 1000
            content = data.get("message", {}).get("content", "")
            return LLMResponse(
                content=content,
                model=model,
                role=ModelRole.FAST,
                elapsed_ms=elapsed,
                tokens_used=data.get("eval_count", 0),
            )
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
            logger.warning("Ollama chat failed: %s", exc)
            elapsed = (time.monotonic() - start) * 1000
            return LLMResponse(
                content="", model=model, role=ModelRole.FAST, elapsed_ms=elapsed, tokens_used=0
            )

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat with tool/function calling via /api/chat with tools param.

        Args:
            messages: Chat messages.
            model: Ollama model tag.
            tools: Tool definitions in OpenAI-style format.
            **kwargs: Extra fields passed to the Ollama API.

        Returns:
            LLMResponse, potentially with tool calls in structured_data.
        """
        return await self.chat(messages, model, tools=tools, **kwargs)

    async def is_available(self) -> bool:
        """Check if Ollama daemon is reachable via /api/tags."""
        try:
            resp = await self._client.get("/api/tags", timeout=2.0)
            return resp.status_code == 200
        except (httpx.TimeoutException, httpx.ConnectError):
            return False

    async def list_models(self) -> list[ModelInfo]:
        """List models available in the local Ollama instance.

        Returns:
            List of ModelInfo from /api/tags, or empty list on failure.
        """
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [
                ModelInfo(
                    name=m["name"],
                    role=ModelRole.FAST,
                    active_memory_mb=int(m.get("size", 0) / 1_000_000),
                    quality_score=0.0,
                )
                for m in data.get("models", [])
            ]
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError):
            return []

    async def health(self) -> LLMHealth:
        """Get health status combining availability and model list.

        Returns:
            LLMHealth with AVAILABLE or UNAVAILABLE state.
        """
        available = await self.is_available()
        models = await self.list_models() if available else []
        state = HealthState.AVAILABLE if available else HealthState.UNAVAILABLE
        return LLMHealth(state=state, models_loaded=[m.name for m in models])

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
