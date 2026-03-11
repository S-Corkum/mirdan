"""GetQualityStandards use case — extracted from server.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mirdan.core.quality_standards import QualityStandards


class GetQualityStandardsUseCase:
    """Retrieve quality standards for a language/framework combination."""

    def __init__(self, quality_standards: QualityStandards) -> None:
        self._quality_standards = quality_standards

    async def execute(
        self,
        language: str,
        framework: str = "",
        category: str = "all",
    ) -> dict[str, Any]:
        """Execute the get_quality_standards use case.

        Args:
            language: Programming language (typescript, python, etc.)
            framework: Optional framework (react, fastapi, etc.)
            category: Filter to specific category (security|architecture|style|all)

        Returns:
            Quality standards for the specified language/framework
        """
        return self._quality_standards.get_all_standards(
            language=language, framework=framework, category=category
        )
