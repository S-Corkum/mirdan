"""Hardware detection and LLM health monitoring."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import subprocess
from typing import Any

from mirdan.models import HardwareInfo, HardwareProfile, HealthState, LLMHealth

logger = logging.getLogger(__name__)


class HardwareDetector:
    """Detects CPU architecture, RAM, GPU, and determines hardware profile.

    All methods are static — no state, no dependencies.
    """

    @staticmethod
    def detect() -> HardwareInfo:
        """Detect hardware capabilities using stdlib only.

        Returns:
            HardwareInfo with architecture, RAM, GPU, and profile.
        """
        arch = platform.machine()
        total_ram = HardwareDetector._get_total_ram_mb()
        available_ram = HardwareDetector.get_available_memory_mb()
        gpu_type = HardwareDetector._detect_gpu()
        metal_capable = arch == "arm64" and platform.system() == "Darwin"
        profile = HardwareDetector._classify_profile(total_ram, arch)

        return HardwareInfo(
            architecture=arch,
            total_ram_mb=total_ram,
            available_ram_mb=available_ram,
            gpu_type=gpu_type,
            metal_capable=metal_capable,
            detected_profile=profile,
        )

    @staticmethod
    def get_available_memory_mb() -> int:
        """Get currently available (free) RAM in MB.

        Returns:
            Available memory in MB, or 0 if detection fails.
        """
        try:
            if platform.system() == "Darwin":
                # macOS: use vm_stat for available pages
                result = subprocess.run(
                    ["vm_stat"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    free_pages = 0
                    page_size = 16384  # Apple Silicon default
                    for line in result.stdout.splitlines():
                        if "page size of" in line:
                            parts = line.split()
                            for part in parts:
                                if part.isdigit():
                                    page_size = int(part)
                        if "Pages free:" in line or "Pages inactive:" in line:
                            val = line.split(":")[1].strip().rstrip(".")
                            free_pages += int(val)
                    return (free_pages * page_size) // (1024 * 1024)
            # Linux/other: use os.sysconf
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return (pages * page_size) // (1024 * 1024)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        return 0

    @staticmethod
    def _get_total_ram_mb() -> int:
        """Get total physical RAM in MB."""
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return (pages * page_size) // (1024 * 1024)
        except (OSError, ValueError):
            pass
        return 0

    @staticmethod
    def _detect_gpu() -> str | None:
        """Detect GPU type. macOS only for now."""
        if platform.system() != "Darwin":
            return None
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("Chipset Model:") or stripped.startswith(
                        "Chip:"
                    ):
                        return stripped.split(":", 1)[1].strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        return None

    @staticmethod
    def _classify_profile(total_ram_mb: int, architecture: str) -> HardwareProfile:
        """Map total RAM and architecture to a hardware profile.

        Args:
            total_ram_mb: Total physical RAM.
            architecture: CPU architecture string (e.g. "arm64", "x86_64").

        Returns:
            HardwareProfile tier.
        """
        if total_ram_mb > 49152 and architecture == "arm64":
            return HardwareProfile.FULL
        if total_ram_mb > 16384:
            return HardwareProfile.ENHANCED
        return HardwareProfile.STANDARD


# Hardware-profile-adjusted inference timeouts.
_PROFILE_TIMEOUTS: dict[HardwareProfile, float] = {
    HardwareProfile.STANDARD: 20.0,
    HardwareProfile.ENHANCED: 10.0,
    HardwareProfile.FULL: 5.0,
}


class HealthMonitor:
    """State machine monitoring LLM subsystem health with background warmup.

    States: STARTING → WARMING_UP → AVAILABLE | DEGRADED | UNAVAILABLE.
    The MCP server remains responsive during WARMING_UP — callers get None
    from LLMManager.generate() until the model is warm.
    """

    def __init__(self, hardware: HardwareInfo | None = None) -> None:
        self._state = HealthState.STARTING
        self._hardware = hardware
        self._warmup_task: asyncio.Task[None] | None = None
        self._last_check: float = 0.0
        self._check_interval: float = 30.0  # seconds between full health refreshes
        self._error: str | None = None

    @property
    def state(self) -> HealthState:
        """Current health state."""
        return self._state

    @property
    def effective_timeout(self) -> float:
        """Inference timeout adjusted for hardware profile."""
        if self._hardware:
            return _PROFILE_TIMEOUTS.get(
                self._hardware.detected_profile, 20.0
            )
        return 20.0

    def quick_check(self) -> HealthState:
        """Return cached state. <1ms, no I/O.

        Returns:
            Current HealthState from cache.
        """
        return self._state

    def transition(self, new_state: HealthState, error: str | None = None) -> None:
        """Transition to a new health state.

        Args:
            new_state: Target state.
            error: Optional error message for DEGRADED/UNAVAILABLE.
        """
        old = self._state
        self._state = new_state
        self._error = error
        if old != new_state:
            logger.info("Health state: %s → %s", old.value, new_state.value)
            if error:
                logger.warning("Health error: %s", error)

    async def warmup_background(self, backend: Any) -> None:
        """Start background model warmup as a non-blocking asyncio task.

        Sends a minimal prompt to the backend to trigger model loading.
        Transitions to AVAILABLE on success, DEGRADED on failure.

        Args:
            backend: An object implementing LocalLLMProtocol.
        """
        self.transition(HealthState.WARMING_UP)
        self._warmup_task = asyncio.create_task(self._do_warmup(backend))

    async def _do_warmup(self, backend: Any) -> None:
        """Execute warmup by sending a test prompt to the backend."""
        try:
            if not await backend.is_available():
                self.transition(
                    HealthState.UNAVAILABLE, error="Backend not available"
                )
                return

            # Send minimal prompt to trigger model loading
            response = await backend.generate("ok", "warmup")
            if response.content or response.tokens_used > 0:
                self.transition(HealthState.AVAILABLE)
            else:
                self.transition(HealthState.DEGRADED, error="Warmup returned empty")
        except Exception as exc:
            logger.warning("Warmup failed: %s", exc)
            self.transition(HealthState.DEGRADED, error=str(exc))

    def to_health(self) -> LLMHealth:
        """Build an LLMHealth snapshot from current state.

        Returns:
            LLMHealth with current state, hardware profile, and timeout.
        """
        return LLMHealth(
            state=self._state,
            hardware_profile=(
                self._hardware.detected_profile.value if self._hardware else "unknown"
            ),
            effective_timeout=self.effective_timeout,
            error=self._error,
        )

    async def close(self) -> None:
        """Cancel any running warmup task."""
        if self._warmup_task and not self._warmup_task.done():
            self._warmup_task.cancel()
