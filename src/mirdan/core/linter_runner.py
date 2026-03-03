"""Subprocess runner for external linters (ruff, ESLint, mypy).

Auto-detects available linters via ``shutil.which()``, runs them as
subprocesses with JSON output, and parses results into ``Violation``
objects using the parsers from ``linter_parsers.py``.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mirdan.core.linter_parsers import parse_eslint_output, parse_mypy_output, parse_ruff_output
from mirdan.models import Violation

logger = logging.getLogger(__name__)

# Default timeout per linter invocation (seconds)
_DEFAULT_TIMEOUT = 30.0


class LinterConfig:
    """Configuration for the linter runner."""

    def __init__(
        self,
        enabled_linters: list[str] | None = None,
        ruff_args: list[str] | None = None,
        eslint_args: list[str] | None = None,
        mypy_args: list[str] | None = None,
        auto_detect: bool = True,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.enabled_linters = enabled_linters or []
        self.ruff_args = ruff_args or []
        self.eslint_args = eslint_args or []
        self.mypy_args = mypy_args or []
        self.auto_detect = auto_detect
        self.timeout = timeout


# Mapping: language -> list of (linter_name, command_builder, parser)
_LINTER_REGISTRY: dict[str, list[tuple[str, Any, Any]]] = {
    "python": [
        (
            "ruff",
            lambda f, args: ["ruff", "check", "--output-format", "json", *args, str(f)],
            parse_ruff_output,
        ),
        (
            "mypy",
            lambda f, args: ["mypy", "--output", "json", "--no-error-summary", *args, str(f)],
            parse_mypy_output,
        ),
    ],
    "javascript": [
        ("eslint", lambda f, args: ["eslint", "--format", "json", *args, str(f)], parse_eslint_output),  # noqa: E501
    ],
    "typescript": [
        ("eslint", lambda f, args: ["eslint", "--format", "json", *args, str(f)], parse_eslint_output),  # noqa: E501
    ],
}


class LinterRunner:
    """Runs external linters and collects their violations."""

    def __init__(self, config: LinterConfig | None = None) -> None:
        self._config = config or LinterConfig()
        self._available: dict[str, str] = {}  # name -> path
        self._detected = False

    def _detect_available(self) -> None:
        """Detect which linters are available on PATH."""
        if self._detected:
            return
        for name in ("ruff", "mypy", "eslint"):
            path = shutil.which(name)
            if path:
                self._available[name] = path
                logger.debug("Linter '%s' found at %s", name, path)
            else:
                logger.debug("Linter '%s' not found on PATH", name)
        self._detected = True

    def is_available(self, linter_name: str) -> bool:
        """Check if a linter is available."""
        self._detect_available()
        return linter_name in self._available

    def available_linters(self) -> list[str]:
        """Return list of available linter names."""
        self._detect_available()
        return list(self._available.keys())

    async def run(self, file_path: Path, language: str) -> list[Violation]:
        """Run all applicable linters for the given file and language.

        Args:
            file_path: Path to the file to lint.
            language: Programming language (python, javascript, typescript).

        Returns:
            Combined list of violations from all linters.
        """
        self._detect_available()

        linters = _LINTER_REGISTRY.get(language, [])
        if not linters:
            return []

        all_violations: list[Violation] = []
        tasks = []

        for linter_name, cmd_builder, parser in linters:
            # Check if enabled (explicit list or auto-detect)
            if self._config.enabled_linters and linter_name not in self._config.enabled_linters:
                continue
            if not self.is_available(linter_name):
                logger.info("Linter '%s' not available, skipping", linter_name)
                continue

            extra_args = getattr(self._config, f"{linter_name}_args", [])
            cmd = cmd_builder(file_path, extra_args)
            tasks.append(self._run_linter(linter_name, cmd, parser))

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                all_violations.extend(result)
            elif isinstance(result, Exception):
                logger.warning("Linter raised exception: %s", result)

        return all_violations

    async def _run_linter(
        self,
        name: str,
        cmd: list[str],
        parser: Callable[[str], list[Violation]],
    ) -> list[Violation]:
        """Run a single linter subprocess."""
        logger.debug("Running linter '%s': %s", name, " ".join(cmd))
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._config.timeout,
            )
        except TimeoutError:
            logger.warning("Linter '%s' timed out after %.1fs", name, self._config.timeout)
            return []
        except FileNotFoundError:
            logger.warning("Linter '%s' not found", name)
            return []

        raw_output = stdout.decode("utf-8", errors="replace")

        # Some linters use non-zero exit codes for "found issues" vs errors
        if proc.returncode is not None and proc.returncode > 2:
            stderr_text = stderr.decode("utf-8", errors="replace")
            logger.warning(
                "Linter '%s' failed (exit %d): %s", name, proc.returncode, stderr_text[:200]
            )
            return []

        violations = parser(raw_output)
        logger.debug("Linter '%s' found %d violations", name, len(violations))
        return violations
