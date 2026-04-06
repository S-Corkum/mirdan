"""Model registry with quality/memory metadata and dynamic selection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mirdan.models import HardwareProfile, ModelInfo, ModelRole

logger = logging.getLogger(__name__)


# Known models with verified memory footprints and quality scores.
# gguf_file: filename to look for in gguf_dir (llama-cpp-python path).
# ollama_tag: tag for `ollama pull` / Ollama API (Ollama path).
KNOWN_MODELS: list[dict[str, Any]] = [
    {
        "name": "gemma4-e4b-q3",
        "active_memory_mb": 4500,
        "quality_score": 0.67,
        "gguf_file": "gemma-4-E4B-it-Q3_K_M.gguf",
        "role": "fast",
        "model_family": "gemma4",
    },
    {
        "name": "gemma4-e2b-q4",
        "active_memory_mb": 3500,
        "quality_score": 0.60,
        "ollama_tag": "gemma4:e2b",
        "role": "fast",
        "model_family": "gemma4",
    },
    {
        "name": "gemma4-e2b-q3",
        "active_memory_mb": 3200,
        "quality_score": 0.58,
        "gguf_file": "gemma-4-E2B-it-Q3_K_M.gguf",
        "role": "fast",
        "model_family": "gemma4",
    },
    {
        "name": "gemma4-31b-q4",
        "active_memory_mb": 17000,
        "quality_score": 0.85,
        "ollama_tag": "gemma4:31b",
        "role": "brain",
        "model_family": "gemma4",
    },
]

# Buffer reserved when selecting models to avoid memory pressure.
_MEMORY_BUFFER_MB = 2048


class ModelRegistry:
    """Registry of known local LLM models with discovery and metadata.

    Scans Ollama tags and a GGUF directory to find which known models
    are actually installed on this machine.
    """

    def __init__(self, known_models: list[dict[str, Any]] | None = None) -> None:
        self._known = known_models or KNOWN_MODELS

    async def discover(
        self,
        backend: Any | None = None,
        gguf_dir: str = "~/.mirdan/models",
    ) -> list[ModelInfo]:
        """Scan for installed models from Ollama and GGUF directory.

        Args:
            backend: An object implementing LocalLLMProtocol (for Ollama discovery).
            gguf_dir: Directory to scan for GGUF files.

        Returns:
            List of ModelInfo for all installed known models.
        """
        installed: list[ModelInfo] = []
        ollama_tags: set[str] = set()
        gguf_files: set[str] = set()

        # Discover Ollama models
        if backend is not None:
            try:
                models = await backend.list_models()
                ollama_tags = {m.name for m in models}
            except Exception:
                logger.warning("Failed to query Ollama for model list")

        # Discover GGUF files
        gguf_path = Path(gguf_dir).expanduser()
        if gguf_path.is_dir():
            gguf_files = {f.name for f in gguf_path.iterdir() if f.suffix == ".gguf"}

        # Match against known models
        for entry in self._known:
            role = ModelRole.BRAIN if entry["role"] == "brain" else ModelRole.FAST
            ollama_tag = entry.get("ollama_tag")
            gguf_file = entry.get("gguf_file")

            found = False
            if ollama_tag and ollama_tag in ollama_tags:
                found = True
            if gguf_file and gguf_file in gguf_files:
                found = True

            if found:
                installed.append(
                    ModelInfo(
                        name=entry["name"],
                        role=role,
                        active_memory_mb=entry["active_memory_mb"],
                        quality_score=entry["quality_score"],
                        model_family=entry.get("model_family", "unknown"),
                    )
                )

        logger.info(
            "Discovered %d installed models (Ollama tags: %d, GGUF files: %d)",
            len(installed),
            len(ollama_tags),
            len(gguf_files),
        )
        return installed

class ModelSelector:
    """Selects the best available model for a given role and memory budget.

    Iterates known models by quality score descending. Returns the first
    model that fits within available memory minus a safety buffer.
    """

    def __init__(
        self,
        installed_models: list[ModelInfo] | None = None,
        memory_buffer_mb: int = _MEMORY_BUFFER_MB,
    ) -> None:
        self._installed = installed_models or []
        self._memory_buffer = memory_buffer_mb

    def update_installed(self, models: list[ModelInfo]) -> None:
        """Update the list of installed models after discovery.

        Args:
            models: Fresh list of installed ModelInfo from registry.discover().
        """
        self._installed = models

    def select(
        self,
        role: ModelRole,
        available_memory_mb: int,
        architecture: str = "",
    ) -> ModelInfo | None:
        """Select the best model for the given role and memory constraints.

        Args:
            role: FAST or BRAIN.
            available_memory_mb: Currently available RAM in MB.
            architecture: CPU architecture (e.g. "arm64"). BRAIN runs on any arch but is faster with Metal on arm64.

        Returns:
            Best-fitting ModelInfo, or None if nothing fits.
        """
        if role == ModelRole.BRAIN and architecture != "arm64":
            logger.info("BRAIN on %s: slower without Metal, but supported", architecture)

        candidates = [m for m in self._installed if m.role == role]
        # Sort by quality descending — pick the best that fits
        candidates.sort(key=lambda m: m.quality_score, reverse=True)

        budget = available_memory_mb - self._memory_buffer
        for model in candidates:
            if model.active_memory_mb <= budget:
                logger.info(
                    "Selected %s (quality=%.2f, memory=%dMB, budget=%dMB)",
                    model.name,
                    model.quality_score,
                    model.active_memory_mb,
                    budget,
                )
                return model

        logger.info(
            "No %s model fits in %dMB (budget after buffer: %dMB)",
            role.value,
            available_memory_mb,
            budget,
        )
        return None
