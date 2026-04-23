"""Usecase: validate a brief file against the brief-driven pipeline rubric."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mirdan.config import BriefConfig
from mirdan.core.brief_validator import BriefValidator


class ValidateBriefUseCase:
    """Thin wrapper around BriefValidator for the MCP tool surface."""

    def __init__(self, config: BriefConfig | None = None) -> None:
        self._validator = BriefValidator(config)

    async def execute(self, brief_path: str) -> dict[str, Any]:
        path = Path(brief_path)
        if not path.exists():
            return {
                "passed": False,
                "score": 0.0,
                "error": f"brief file not found: {brief_path}",
            }
        if not path.is_file():
            return {
                "passed": False,
                "score": 0.0,
                "error": f"not a file: {brief_path}",
            }
        result = self._validator.validate_file(path)
        return result.to_dict()
