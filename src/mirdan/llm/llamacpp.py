"""llama-cpp-python in-process backend for memory-optimal inference."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import anyio

from mirdan.models import HealthState, LLMHealth, LLMResponse, ModelInfo, ModelRole

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama  # type: ignore[import-not-found,unused-ignore]

    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    Llama = None  # type: ignore[assignment,misc]


def is_available() -> bool:
    """Check if llama-cpp-python is installed."""
    return LLAMACPP_AVAILABLE


class LlamaCppBackend:
    """Direct GGUF loading via llama-cpp-python. Zero daemon overhead.

    Saves ~200MB vs Ollama by running inference in mirdan's own process.
    Optimal backend for 16GB machines. All blocking C calls are wrapped
    in ``anyio.to_thread.run_sync()`` to avoid blocking the event loop.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        keep_alive_seconds: int = 300,
    ) -> None:
        if not LLAMACPP_AVAILABLE:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Install with: pip install llama-cpp-python"
            )
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_threads = n_threads or max(1, (os.cpu_count() or 4) // 2)
        self._keep_alive = keep_alive_seconds
        self._llm: Any = None  # Llama instance, typed as Any for optional dep
        self._last_use: float = 0.0
        self._unload_task: asyncio.Task[None] | None = None

    def _load_model_sync(self) -> Any:
        """Load GGUF model synchronously. Must be called via anyio.to_thread.run_sync()."""
        logger.info("Loading GGUF model: %s", self._model_path)
        return Llama(
            model_path=self._model_path,
            n_ctx=self._n_ctx,
            n_threads=self._n_threads,
            verbose=False,
        )

    async def _ensure_loaded(self) -> Any:
        """Lazy-load model on first use. Runs model load in thread to avoid blocking."""
        if self._llm is None:
            self._llm = await anyio.to_thread.run_sync(self._load_model_sync)
        self._last_use = time.monotonic()
        self._schedule_unload()
        return self._llm

    def _schedule_unload(self) -> None:
        """Schedule model unload after keep_alive period. Cancels previous timer."""
        if self._unload_task and not self._unload_task.done():
            self._unload_task.cancel()
        self._unload_task = asyncio.create_task(self._unload_after_idle())

    async def _unload_after_idle(self) -> None:
        """Wait for keep_alive then unload model to free RAM."""
        await asyncio.sleep(self._keep_alive)
        if self._llm and (time.monotonic() - self._last_use) >= self._keep_alive:
            logger.info("Unloading model after %ds idle", self._keep_alive)
            self._llm.close()
            self._llm = None

    async def generate(self, prompt: str, model: str, **kwargs: Any) -> LLMResponse:
        """Generate a text completion via llama.cpp.

        Args:
            prompt: The prompt text.
            model: Model name (for response metadata; actual model is the loaded GGUF).
            **kwargs: Supports ``max_tokens`` (default 512).

        Returns:
            LLMResponse with generated content, or empty on failure.
        """
        start = time.monotonic()
        try:
            llm = await self._ensure_loaded()
            max_tokens = kwargs.get("max_tokens", 512)
            result = await anyio.to_thread.run_sync(
                lambda: llm(prompt, max_tokens=max_tokens)
            )
            content = result["choices"][0]["text"] if result.get("choices") else ""
            elapsed = (time.monotonic() - start) * 1000
            tokens = result.get("usage", {}).get("completion_tokens", 0)
            return LLMResponse(
                content=content,
                model=model,
                role=ModelRole.FAST,
                elapsed_ms=elapsed,
                tokens_used=tokens,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning("LlamaCpp generate failed: %s", exc)
            elapsed = (time.monotonic() - start) * 1000
            return LLMResponse(
                content="", model=model, role=ModelRole.FAST, elapsed_ms=elapsed, tokens_used=0
            )

    async def generate_structured(
        self, prompt: str, model: str, schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """Generate structured JSON output using llama.cpp grammar constraints.

        Falls back to text generation with JSON parsing if grammar fails.

        Args:
            prompt: The prompt text.
            model: Model name.
            schema: JSON schema for constraining output.
            **kwargs: Supports ``max_tokens`` (default 512).

        Returns:
            Parsed dict, or empty dict on failure.
        """
        try:
            llm = await self._ensure_loaded()
            max_tokens = kwargs.get("max_tokens", 512)
            result = await anyio.to_thread.run_sync(
                lambda: llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object", "schema": schema},
                    max_tokens=max_tokens,
                )
            )
            content = (
                result["choices"][0]["message"]["content"]
                if result.get("choices")
                else "{}"
            )
            parsed: dict[str, Any] = json.loads(content)
            return parsed
        except json.JSONDecodeError:
            logger.warning(
                "LlamaCpp structured output was not valid JSON, trying text fallback"
            )
            response = await self.generate(
                prompt + "\nRespond with valid JSON only.", model, **kwargs
            )
            try:
                fallback: dict[str, Any] = json.loads(response.content)
                return fallback
            except json.JSONDecodeError:
                return {}
        except (RuntimeError, OSError) as exc:
            logger.warning("LlamaCpp generate_structured failed: %s", exc)
            return {}

    async def chat(
        self, messages: list[dict[str, Any]], model: str, **kwargs: Any
    ) -> LLMResponse:
        """Multi-turn chat via llama.cpp create_chat_completion.

        Args:
            messages: Chat messages in OpenAI-style format.
            model: Model name.
            **kwargs: Supports ``max_tokens`` (default 512).

        Returns:
            LLMResponse with assistant reply, or empty on failure.
        """
        start = time.monotonic()
        try:
            llm = await self._ensure_loaded()
            max_tokens = kwargs.get("max_tokens", 512)
            result = await anyio.to_thread.run_sync(
                lambda: llm.create_chat_completion(messages=messages, max_tokens=max_tokens)
            )
            content = (
                result["choices"][0]["message"]["content"]
                if result.get("choices")
                else ""
            )
            elapsed = (time.monotonic() - start) * 1000
            tokens = result.get("usage", {}).get("completion_tokens", 0)
            return LLMResponse(
                content=content,
                model=model,
                role=ModelRole.FAST,
                elapsed_ms=elapsed,
                tokens_used=tokens,
            )
        except (RuntimeError, OSError) as exc:
            logger.warning("LlamaCpp chat failed: %s", exc)
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
        """Chat with tool calling via prompt augmentation.

        Since llama.cpp doesn't natively support tool calling for all models,
        tool definitions are injected as a system message.

        Args:
            messages: Chat messages.
            model: Model name.
            tools: Tool definitions in OpenAI-style format.
            **kwargs: Extra options.

        Returns:
            LLMResponse with potential tool call in content.
        """
        tool_desc = json.dumps(tools, indent=2)
        system_msg = (
            f"Available tools:\n{tool_desc}\n\n"
            'To call a tool, respond with JSON: '
            '{"tool_call": {"name": "...", "arguments": {...}}}'
        )
        augmented = [{"role": "system", "content": system_msg}] + messages
        return await self.chat(augmented, model, **kwargs)

    async def is_available(self) -> bool:
        """Check if llama-cpp-python is installed and the model file exists."""
        return LLAMACPP_AVAILABLE and Path(self._model_path).exists()

    async def list_models(self) -> list[ModelInfo]:
        """List the single model this backend is configured with.

        Returns:
            Single-element list with the GGUF model info, or empty if unavailable.
        """
        if not await self.is_available():
            return []
        size_mb = Path(self._model_path).stat().st_size // (1024 * 1024)
        name = Path(self._model_path).stem
        return [
            ModelInfo(
                name=name,
                role=ModelRole.FAST,
                active_memory_mb=size_mb + 300,  # GGUF file + ~300MB KV cache
                quality_score=0.0,
            )
        ]

    async def health(self) -> LLMHealth:
        """Get health status based on package availability and model file.

        Returns:
            LLMHealth with AVAILABLE or UNAVAILABLE state.
        """
        available = await self.is_available()
        models = await self.list_models() if available else []
        state = HealthState.AVAILABLE if available else HealthState.UNAVAILABLE
        return LLMHealth(state=state, models_loaded=[m.name for m in models])

    async def close(self) -> None:
        """Release model memory and cancel the unload timer."""
        if self._unload_task and not self._unload_task.done():
            self._unload_task.cancel()
        if self._llm:
            self._llm.close()
            self._llm = None
            logger.info("LlamaCpp backend closed, model unloaded")
