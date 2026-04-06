"""Local Intelligence Layer — local LLM integration for mirdan."""

from mirdan.llm.manager import LLMManager
from mirdan.llm.protocol import InMemoryBackend, LocalLLMProtocol

__all__ = ["LocalLLMProtocol", "InMemoryBackend", "LLMManager"]
