"""Tests for LocalLLMProtocol and InMemoryBackend."""

import pytest

from mirdan.llm import InMemoryBackend, LocalLLMProtocol
from mirdan.models import (
    HealthState,
    LLMResponse,
    ModelInfo,
    ModelRole,
)


class TestInMemoryBackendProtocolCompliance:
    """Verify InMemoryBackend satisfies LocalLLMProtocol."""

    def test_is_protocol_compatible(self) -> None:
        """InMemoryBackend must be usable wherever LocalLLMProtocol is expected."""
        backend: LocalLLMProtocol = InMemoryBackend()
        assert backend is not None


class TestInMemoryBackendGenerate:
    """Tests for InMemoryBackend.generate()."""

    @pytest.mark.asyncio
    async def test_returns_canned_response(self) -> None:
        backend = InMemoryBackend()
        canned = LLMResponse(
            content="hello world",
            model="test-model",
            role=ModelRole.FAST,
            elapsed_ms=10.0,
            tokens_used=5,
        )
        backend.responses.append(canned)

        result = await backend.generate("test prompt", "test-model")
        assert result.content == "hello world"
        assert result.model == "test-model"
        assert result.role == ModelRole.FAST
        assert result.tokens_used == 5

    @pytest.mark.asyncio
    async def test_returns_empty_default_when_queue_exhausted(self) -> None:
        backend = InMemoryBackend()
        result = await backend.generate("prompt", "model-x")
        assert result.content == ""
        assert result.model == "model-x"
        assert result.tokens_used == 0

    @pytest.mark.asyncio
    async def test_pops_responses_in_order(self) -> None:
        backend = InMemoryBackend()
        for i in range(3):
            backend.responses.append(
                LLMResponse(
                    content=f"response-{i}",
                    model="m",
                    role=ModelRole.FAST,
                    elapsed_ms=0.0,
                    tokens_used=0,
                )
            )

        for i in range(3):
            result = await backend.generate("p", "m")
            assert result.content == f"response-{i}"

    @pytest.mark.asyncio
    async def test_records_prompts(self) -> None:
        backend = InMemoryBackend()
        await backend.generate("first", "m")
        await backend.generate("second", "m")
        assert backend.prompts_received == ["first", "second"]

    @pytest.mark.asyncio
    async def test_increments_call_count(self) -> None:
        backend = InMemoryBackend()
        assert backend._call_count == 0
        await backend.generate("p", "m")
        await backend.generate("p", "m")
        assert backend._call_count == 2


class TestInMemoryBackendGenerateStructured:
    """Tests for InMemoryBackend.generate_structured()."""

    @pytest.mark.asyncio
    async def test_returns_canned_structured_response(self) -> None:
        backend = InMemoryBackend()
        backend.structured_responses.append({"classification": "local_only", "confidence": 0.9})

        result = await backend.generate_structured("prompt", "m", schema={})
        assert result == {"classification": "local_only", "confidence": 0.9}

    @pytest.mark.asyncio
    async def test_returns_empty_dict_when_exhausted(self) -> None:
        backend = InMemoryBackend()
        result = await backend.generate_structured("prompt", "m", schema={})
        assert result == {}

    @pytest.mark.asyncio
    async def test_records_prompt(self) -> None:
        backend = InMemoryBackend()
        await backend.generate_structured("structured prompt", "m", schema={})
        assert "structured prompt" in backend.prompts_received


class TestInMemoryBackendChat:
    """Tests for chat and chat_with_tools."""

    @pytest.mark.asyncio
    async def test_chat_delegates_to_generate(self) -> None:
        backend = InMemoryBackend()
        canned = LLMResponse(
            content="chat reply",
            model="m",
            role=ModelRole.FAST,
            elapsed_ms=0.0,
            tokens_used=3,
        )
        backend.responses.append(canned)

        messages = [{"role": "user", "content": "hello"}]
        result = await backend.chat(messages, "m")
        assert result.content == "chat reply"
        assert backend.prompts_received == ["hello"]

    @pytest.mark.asyncio
    async def test_chat_with_tools_delegates_to_generate(self) -> None:
        backend = InMemoryBackend()
        messages = [{"role": "user", "content": "use tool"}]
        tools = [{"name": "search", "description": "search code"}]
        result = await backend.chat_with_tools(messages, "m", tools)
        assert result.content == ""
        assert backend.prompts_received == ["use tool"]


class TestInMemoryBackendHealth:
    """Tests for availability and health reporting."""

    @pytest.mark.asyncio
    async def test_available_by_default(self) -> None:
        backend = InMemoryBackend()
        assert await backend.is_available() is True

    @pytest.mark.asyncio
    async def test_unavailable_when_set(self) -> None:
        backend = InMemoryBackend()
        backend.available = False
        assert await backend.is_available() is False

    @pytest.mark.asyncio
    async def test_health_available(self) -> None:
        backend = InMemoryBackend()
        backend.models = [
            ModelInfo(
                name="test-model",
                role=ModelRole.FAST,
                active_memory_mb=3500,
                quality_score=0.6,
            )
        ]
        health = await backend.health()
        assert health.state == HealthState.AVAILABLE
        assert health.models_loaded == ["test-model"]

    @pytest.mark.asyncio
    async def test_health_unavailable(self) -> None:
        backend = InMemoryBackend()
        backend.available = False
        health = await backend.health()
        assert health.state == HealthState.UNAVAILABLE

    @pytest.mark.asyncio
    async def test_list_models_empty_by_default(self) -> None:
        backend = InMemoryBackend()
        assert await backend.list_models() == []

    @pytest.mark.asyncio
    async def test_list_models_returns_configured(self) -> None:
        backend = InMemoryBackend()
        model = ModelInfo(
            name="gemma4-e2b",
            role=ModelRole.FAST,
            active_memory_mb=3500,
            quality_score=0.6,
        )
        backend.models = [model]
        result = await backend.list_models()
        assert len(result) == 1
        assert result[0].name == "gemma4-e2b"

    @pytest.mark.asyncio
    async def test_close_is_noop(self) -> None:
        backend = InMemoryBackend()
        await backend.close()  # Should not raise
