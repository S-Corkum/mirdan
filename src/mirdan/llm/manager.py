"""LLM Manager facade — single entry point for all local LLM operations."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mirdan.core.check_runner import CheckRunner
    from mirdan.core.triage import TriageEngine
    from mirdan.llm.sidecar import Sidecar

import anyio

from mirdan.config import LLMConfig
from mirdan.llm.health import HardwareDetector, HealthMonitor
from mirdan.llm.registry import ModelRegistry, ModelSelector
from mirdan.models import (
    HardwareInfo,
    HealthState,
    LLMHealth,
    LLMResponse,
    ModelRole,
)

logger = logging.getLogger(__name__)


class LLMManager:
    """Facade for all local LLM operations.

    All mirdan code calls LLMManager — never backends directly.
    Returns None from generate/generate_structured when the LLM
    subsystem is not available, warming up, or disabled.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._backend: Any = None  # LocalLLMProtocol implementation
        self._registry = ModelRegistry()
        self._selector = ModelSelector()
        self._health: HealthMonitor | None = None
        self._hardware: HardwareInfo | None = None
        self._sidecar: Sidecar | None = None
        self._warmup_task: asyncio.Task[None] | None = None
        # Engines wired during startup (Phase 3 / Phase 4)
        self.triage_engine: TriageEngine | None = None
        self.check_runner: CheckRunner | None = None

    @classmethod
    def create_if_enabled(cls, config: LLMConfig) -> LLMManager | None:
        """Create an LLMManager if LLM features are enabled.

        Args:
            config: LLM configuration.

        Returns:
            LLMManager instance, or None if disabled.
        """
        if not config.enabled:
            return None
        return cls(config)

    async def startup(self) -> None:
        """Initialize all LLM components. Non-blocking — warmup runs in background.

        Sequence: cleanup stale port file → hardware detection → backend
        creation → model discovery → health warmup → sidecar start.
        """
        # 0. Configure SSL for corporate environments (Netskope, Zscaler, etc.)
        from mirdan.core.ssl_config import configure_ssl

        configure_ssl()

        # 1. Cleanup stale port file from previous session
        port_file = Path(".mirdan/sidecar.port")
        if port_file.exists():
            try:
                port_file.unlink()
            except OSError:
                pass

        # 1. Hardware detection
        self._hardware = await anyio.to_thread.run_sync(HardwareDetector.detect)
        logger.info(
            "Hardware: %s, %dMB RAM, profile=%s",
            self._hardware.architecture,
            self._hardware.total_ram_mb,
            self._hardware.detected_profile.value,
        )

        # 2. Create backend (auto-select)
        self._backend = self._create_backend()
        if self._backend is None:
            logger.warning("No LLM backend available — features disabled")
            return

        if not await self._backend.is_available():
            logger.warning("LLM backend unavailable — features disabled")
            return

        # 3. Discover installed models and update selector
        installed = await self._registry.discover(
            backend=self._backend,
            gguf_dir=self._config.gguf_dir,
        )
        self._selector.update_installed(installed)

        if not installed:
            logger.warning("No known models found — LLM features limited")

        # 4. Health monitor + background warmup
        # Use the first discovered model name for warmup (not a fake name)
        warmup_model = installed[0].name if installed else None
        self._health = HealthMonitor(hardware=self._hardware)
        await self._health.warmup_background(self._backend, model_name=warmup_model)
        if self._health._warmup_task:
            self._health._warmup_task.add_done_callback(self._on_warmup_done)

        # 5. Create triage engine (used by sidecar and enhance_prompt)
        if self._config.triage:
            from mirdan.core.triage import TriageEngine

            self.triage_engine = TriageEngine(llm_manager=self, config=self._config)

        # 6. Create check runner (used by sidecar and Stop hook)
        if self._config.check_runner:
            from mirdan.cli.check_command import _resolve_checks
            from mirdan.config import MirdanConfig
            from mirdan.core.check_runner import CheckRunner

            # Apply the same runtime language fallback the CLI uses so sidecar
            # and Stop-hook consumers get language-appropriate commands even
            # when the persisted config lacks an explicit llm.checks block.
            runner_cfg = _resolve_checks(MirdanConfig.find_config())
            self.check_runner = CheckRunner(
                llm_manager=self,
                config=self._config,
                checks_override=runner_cfg,
            )

        # 7. Start HTTP sidecar
        from mirdan.llm.sidecar import Sidecar

        self._sidecar = Sidecar(self)
        await self._sidecar.start()

    def _create_backend(self) -> Any:
        """Create the appropriate backend based on config and availability.

        Auto-selection: llama-cpp-python if installed AND GGUF file found,
        otherwise Ollama.

        Returns:
            A LocalLLMProtocol implementation, or None if neither is available.
        """
        backend_choice = self._config.backend

        if backend_choice in ("auto", "llamacpp"):
            try:
                from mirdan.llm.llamacpp import LlamaCppBackend, is_available

                if is_available():
                    # Find a GGUF file in the configured directory
                    gguf_path = Path(self._config.gguf_dir).expanduser()
                    if gguf_path.is_dir():
                        gguf_files = list(gguf_path.glob("*.gguf"))
                        if gguf_files:
                            # Pick the first one — ModelSelector handles quality
                            model_path = str(gguf_files[0])
                            logger.info("Using llama-cpp-python with %s", model_path)
                            return LlamaCppBackend(
                                model_path=model_path,
                                n_ctx=self._config.n_ctx,
                                n_threads=self._config.n_threads,
                                keep_alive_seconds=self._parse_keep_alive(),
                            )
            except Exception as exc:
                logger.warning("llama-cpp-python not usable: %s", exc)

            if backend_choice == "llamacpp":
                logger.warning("llamacpp backend requested but not available")
                return None

        if backend_choice in ("auto", "ollama"):
            try:
                from mirdan.llm.ollama import OllamaBackend

                logger.info("Using Ollama backend at %s", self._config.ollama_url)
                return OllamaBackend(
                    base_url=self._config.ollama_url,
                    timeout=120.0,  # Must accommodate cold model loading (30-60s on Intel)
                    keep_alive=self._config.model_keep_alive,
                )
            except Exception as exc:
                logger.warning("Ollama backend not usable: %s", exc)

        return None

    def _parse_keep_alive(self) -> int:
        """Parse keep_alive string (e.g. '5m') to seconds."""
        val = self._config.model_keep_alive
        if val.endswith("m"):
            return int(val[:-1]) * 60
        if val.endswith("s"):
            return int(val[:-1])
        try:
            return int(val)
        except ValueError:
            return 300

    def _on_warmup_done(self, task: asyncio.Task[None]) -> None:
        """Log warmup errors instead of silently swallowing them."""
        if not task.cancelled() and task.exception():
            logger.error("Model warmup failed: %s", task.exception())

    async def generate(self, role: ModelRole, prompt: str, **kwargs: Any) -> LLMResponse | None:
        """Generate a text completion using the local LLM.

        Args:
            role: FAST or BRAIN model role.
            prompt: The prompt text.
            **kwargs: Sampling params (temperature, top_p, etc.).

        Returns:
            LLMResponse, or None if LLM is not available.
        """
        if not self._health or self._health.state != HealthState.AVAILABLE:
            return None
        if not self._backend or not self._hardware:
            return None

        available_ram = HardwareDetector.get_available_memory_mb()
        model = self._selector.select(role, available_ram, architecture=self._hardware.architecture)
        # Fallback: if selector rejects due to low available RAM but models are
        # installed (e.g., Ollama already loaded the model into RAM, which makes
        # available RAM appear low), use the best installed model for this role.
        if not model:
            candidates = [m for m in self._selector._installed if m.role == role]
            if candidates:
                model = max(candidates, key=lambda m: m.quality_score)
                logger.debug("Selector rejected all models; using installed %s", model.name)
        if not model:
            return None

        try:
            timeout = self._health.effective_timeout
            return await asyncio.wait_for(
                self._backend.generate(prompt, model.name, **kwargs),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning("LLM generate timed out after %.1fs", self._health.effective_timeout)
            return None

    async def generate_structured(
        self, role: ModelRole, prompt: str, schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any] | None:
        """Generate structured JSON output. Handles Gemma 4 think=false bug.

        Args:
            role: FAST or BRAIN model role.
            prompt: The prompt text.
            schema: JSON schema for structured output.
            **kwargs: Sampling params. ``thinking=True`` enables thinking mode.

        Returns:
            Parsed dict, or None if LLM is not available.
        """
        if not self._health or self._health.state != HealthState.AVAILABLE:
            return None
        if not self._backend or not self._hardware:
            return None

        available_ram = HardwareDetector.get_available_memory_mb()
        model = self._selector.select(role, available_ram, architecture=self._hardware.architecture)
        if not model:
            candidates = [m for m in self._selector._installed if m.role == role]
            if candidates:
                model = max(candidates, key=lambda m: m.quality_score)
        if not model:
            return None

        # Gemma 4 bug workaround: Ollama format param silently breaks with think=false.
        # Use prompt-based JSON fallback for Gemma 4 + Ollama + thinking disabled.
        use_format_param = True
        thinking = kwargs.get("thinking", False)
        if model.model_family.startswith("gemma") and not thinking and self._is_ollama_backend():
            use_format_param = False

        try:
            timeout = self._health.effective_timeout
            if use_format_param:
                return await asyncio.wait_for(
                    self._backend.generate_structured(prompt, model.name, schema, **kwargs),
                    timeout=timeout,
                )
            else:
                # Text fallback: generate + json.loads
                response = await asyncio.wait_for(
                    self._backend.generate(prompt, model.name, **kwargs),
                    timeout=timeout,
                )
                if not response or not response.content:
                    return None
                try:
                    parsed: dict[str, Any] = json.loads(response.content)
                    if not self._validate_schema(parsed, schema):
                        logger.warning("Text fallback response failed schema validation")
                        return None
                    return parsed
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON from text response")
                    return None
        except TimeoutError:
            logger.warning("LLM generate_structured timed out")
            return None

    @staticmethod
    def _validate_schema(data: dict[str, Any], schema: dict[str, Any]) -> bool:
        """Lightweight schema validation for text fallback responses.

        Checks that required keys exist and top-level types match.
        Does NOT do full JSON Schema validation — uses simple isinstance checks.

        Args:
            data: Parsed JSON from LLM response.
            schema: JSON schema with optional 'required' and 'properties'.

        Returns:
            True if data passes structural validation.
        """
        type_map: dict[str, type | tuple[type, ...]] = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "array": list,
            "object": dict,
            "boolean": bool,
        }

        for key in schema.get("required", []):
            if key not in data:
                logger.warning("Schema validation: missing required key '%s'", key)
                return False

        for key, prop in schema.get("properties", {}).items():
            if key not in data:
                continue
            expected_type = prop.get("type")
            if isinstance(expected_type, str):
                python_type = type_map.get(expected_type)
                if python_type and not isinstance(data[key], python_type):
                    logger.warning(
                        "Schema validation: '%s' expected %s, got %s",
                        key,
                        expected_type,
                        type(data[key]).__name__,
                    )
                    return False

        return True

    def _is_ollama_backend(self) -> bool:
        """Check if the current backend is OllamaBackend."""
        try:
            from mirdan.llm.ollama import OllamaBackend

            return isinstance(self._backend, OllamaBackend)
        except ImportError:
            return False

    def is_role_available(self, role: ModelRole) -> bool:
        """Check if a model for the given role can be selected with current resources.

        Uses actual available RAM and detected architecture. This is the public
        API for engines to check model availability — engines must not access
        _selector directly.

        Args:
            role: FAST or BRAIN model role.

        Returns:
            True if a model for this role fits in current available memory.
        """
        if not self._hardware:
            return False
        available_ram = HardwareDetector.get_available_memory_mb()
        return (
            self._selector.select(role, available_ram, architecture=self._hardware.architecture)
            is not None
        )

    async def health(self) -> LLMHealth:
        """Get current LLM subsystem health.

        Returns:
            LLMHealth snapshot.
        """
        if self._health:
            return self._health.to_health()
        return LLMHealth(state=HealthState.UNAVAILABLE)

    async def shutdown(self) -> None:
        """Shutdown all LLM components and cleanup."""
        # Stop sidecar
        if self._sidecar:
            await self._sidecar.stop()

        # Cancel warmup
        if self._health:
            await self._health.close()

        # Close backend
        if self._backend:
            await self._backend.close()

        # Remove port file
        port_file = Path(".mirdan/sidecar.port")
        if port_file.exists():
            try:
                port_file.unlink()
            except OSError:
                pass

        logger.info("LLM subsystem shut down")
