"""Protocol definition for local LLM backends."""

from __future__ import annotations

from typing import Any, Protocol

from mirdan.models import (
    HealthState,
    LLMHealth,
    LLMResponse,
    ModelInfo,
    ModelRole,
)


class LocalLLMProtocol(Protocol):
    """Interface that all LLM backends must implement.

    Every backend (Ollama, llama-cpp-python, InMemoryBackend) implements this
    protocol. Consumers call through LLMManager, never directly.
    """

    async def generate(self, prompt: str, model: str, **kwargs: Any) -> LLMResponse:
        """Generate a text completion.

        Args:
            prompt: The prompt text.
            model: Model name or tag to use.
            **kwargs: Backend-specific options (temperature, etc.).

        Returns:
            LLMResponse with generated content.
        """
        ...

    async def generate_structured(
        self, prompt: str, model: str, schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """Generate a structured (JSON) response constrained by a schema.

        Args:
            prompt: The prompt text.
            model: Model name or tag.
            schema: JSON schema the output must conform to.
            **kwargs: Backend-specific options.

        Returns:
            Parsed dictionary matching the schema.
        """
        ...

    async def chat(
        self, messages: list[dict[str, Any]], model: str, **kwargs: Any
    ) -> LLMResponse:
        """Multi-turn chat completion.

        Args:
            messages: Chat messages in OpenAI-style format.
            model: Model name or tag.
            **kwargs: Backend-specific options.

        Returns:
            LLMResponse with assistant reply.
        """
        ...

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat with tool/function calling support.

        Args:
            messages: Chat messages.
            model: Model name or tag.
            tools: Tool definitions in OpenAI-style format.
            **kwargs: Backend-specific options.

        Returns:
            LLMResponse, potentially with tool calls in structured_data.
        """
        ...

    async def is_available(self) -> bool:
        """Check if this backend is reachable and ready."""
        ...

    async def list_models(self) -> list[ModelInfo]:
        """List models available through this backend."""
        ...

    async def health(self) -> LLMHealth:
        """Get detailed health status."""
        ...

    async def close(self) -> None:
        """Release resources held by this backend."""
        ...


class InMemoryBackend:
    """Test double implementing LocalLLMProtocol with canned responses.

    Queue responses via `.responses` and `.structured_responses` before
    calling generate/generate_structured. Pops from front on each call.
    Falls back to empty defaults when queues are exhausted.
    """

    def __init__(self) -> None:
        self.responses: list[LLMResponse] = []
        self.structured_responses: list[dict[str, Any]] = []
        self.available: bool = True
        self.models: list[ModelInfo] = []
        self._call_count: int = 0
        self.prompts_received: list[str] = []

    async def generate(self, prompt: str, model: str, **kwargs: Any) -> LLMResponse:
        """Return next canned response or an empty default."""
        self._call_count += 1
        self.prompts_received.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(
            content="", model=model, role=ModelRole.FAST, elapsed_ms=0.0, tokens_used=0
        )

    async def generate_structured(
        self, prompt: str, model: str, schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """Return next canned structured response or empty dict."""
        self._call_count += 1
        self.prompts_received.append(prompt)
        if self.structured_responses:
            return self.structured_responses.pop(0)
        return {}

    async def chat(
        self, messages: list[dict[str, Any]], model: str, **kwargs: Any
    ) -> LLMResponse:
        """Delegate to generate using the last message content."""
        return await self.generate(messages[-1].get("content", ""), model, **kwargs)

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Delegate to generate using the last message content."""
        return await self.generate(messages[-1].get("content", ""), model, **kwargs)

    async def is_available(self) -> bool:
        """Return the configured availability flag."""
        return self.available

    async def list_models(self) -> list[ModelInfo]:
        """Return the configured model list."""
        return self.models

    async def health(self) -> LLMHealth:
        """Return health based on availability flag."""
        state = HealthState.AVAILABLE if self.available else HealthState.UNAVAILABLE
        return LLMHealth(state=state, models_loaded=[m.name for m in self.models])

    async def close(self) -> None:
        """No-op for test double."""
