"""ScanConventions use case — extracted from server.py."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from mirdan.core.convention_extractor import ConventionExtractor

logger = logging.getLogger(__name__)


class ScanConventionsUseCase:
    """Scan a codebase to discover implicit conventions and patterns."""

    def __init__(self, convention_extractor: ConventionExtractor) -> None:
        self._convention_extractor = convention_extractor

    async def execute(
        self,
        directory: str = ".",
        language: str = "auto",
    ) -> dict[str, Any]:
        """Execute the scan_conventions use case.

        Validates multiple source files, aggregates results, and produces
        convention entries describing naming patterns, import styles,
        docstring conventions, and recurring violation patterns.

        Args:
            directory: Directory to scan (default: current directory)
            language: Language filter or "auto" to detect

        Returns:
            Scan result with discovered conventions and quality baselines
        """
        scan_dir = Path(directory).resolve()

        if not scan_dir.is_dir():
            return {"error": f"Not a directory: {directory}"}

        result = self._convention_extractor.scan(scan_dir, language=language)

        # Persist conventions for quality standards feedback loop
        try:
            conventions_dir = Path.cwd() / ".mirdan"
            conventions_dir.mkdir(parents=True, exist_ok=True)
            conventions_path = conventions_dir / "conventions.yaml"
            conventions_data = {
                "conventions": [e.to_dict() for e in result.conventions],
                "language": result.language,
            }
            with conventions_path.open("w") as f:
                yaml.dump(conventions_data, f, default_flow_style=False, allow_unicode=True)
        except Exception:
            logger.debug("Failed to persist conventions", exc_info=True)

        return result.to_dict()
