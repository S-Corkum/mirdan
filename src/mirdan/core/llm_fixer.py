"""LLM-powered auto-fixer — generates and applies search/replace fixes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mirdan.config import LLMConfig
from mirdan.llm.prompts.fix import FIX_SAMPLING, FIX_SCHEMA, build_fix_prompt
from mirdan.models import FileFixReport, ModelRole

if TYPE_CHECKING:
    from mirdan.llm.manager import LLMManager

logger = logging.getLogger(__name__)

# Minimum confidence for a fix to be applied
_MIN_FIX_CONFIDENCE = 0.7


class LLMFixer:
    """Applies LLM-generated search/replace fixes to files.

    The LLM receives the full file content and violations, generating
    exact search-and-replace pairs. Each fix is verified by checking
    that the search text exists in the file. After application, the
    caller is expected to re-run checks to verify no regressions.
    """

    def __init__(
        self,
        llm_manager: LLMManager | None = None,
        config: LLMConfig | None = None,
    ) -> None:
        self._llm = llm_manager
        self._config = config or LLMConfig()

    async def fix_file(
        self,
        file_path: str,
        violations: list[dict[str, Any]],
        language: str = "auto",
    ) -> FileFixReport:
        """Read a file, generate LLM fixes, apply them, write back.

        Args:
            file_path: Path to the source file to fix.
            violations: List of violation dicts with id, message, line.
            language: Programming language of the file.

        Returns:
            FileFixReport with applied and skipped fixes.
        """
        path = Path(file_path)
        if not path.exists():
            return FileFixReport(
                file=file_path,
                skipped=[v.get("id", "?") for v in violations],
            )

        code = path.read_text()

        # Get structured search/replace pairs from LLM
        fixes = await self._get_fixes(code, violations)
        if not fixes:
            return FileFixReport(
                file=file_path,
                skipped=[v.get("id", "?") for v in violations],
            )

        # Apply fixes that pass confidence threshold and search text anchor
        fixed_code = code
        applied: list[dict[str, Any]] = []
        skipped: list[str] = []
        fixed_violation_ids: set[str] = set()

        for fix in fixes:
            confidence = fix.get("confidence", 0.5)
            if confidence < _MIN_FIX_CONFIDENCE:
                skipped.append(fix.get("violation_id", "?"))
                continue

            search = fix.get("search", "")
            replace = fix.get("replace", "")

            if not search:
                skipped.append(fix.get("violation_id", "?"))
                continue

            # Search text anchor — fail-safe: if not found, skip
            if search not in fixed_code:
                logger.warning(
                    "Fix search text not found in %s for %s, skipping",
                    file_path,
                    fix.get("violation_id", "?"),
                )
                skipped.append(fix.get("violation_id", "?"))
                continue

            # Apply the fix (first occurrence only)
            fixed_code = fixed_code.replace(search, replace, 1)
            applied.append({
                "violation_id": fix.get("violation_id", ""),
                "search": search[:100],  # Truncate for display
                "replace": replace[:100],
                "confidence": confidence,
                "description": fix.get("description", ""),
            })
            fixed_violation_ids.add(fix.get("violation_id", ""))

        # Track violations we never tried to fix
        for v in violations:
            vid = v.get("id", "?")
            if vid not in fixed_violation_ids and vid not in skipped:
                skipped.append(vid)

        # Write back if we applied any fixes
        if applied:
            path.write_text(fixed_code)
            logger.info(
                "Applied %d LLM fixes to %s",
                len(applied),
                file_path,
            )

        return FileFixReport(
            file=file_path,
            applied=applied,
            skipped=skipped,
        )

    async def _get_fixes(
        self,
        code: str,
        violations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Ask the LLM to generate search/replace fix pairs.

        Args:
            code: Full source file content.
            violations: Violations to fix.

        Returns:
            List of fix dicts with violation_id, search, replace, confidence.
        """
        if not self._llm:
            return []

        violations_json = json.dumps(violations, indent=2)
        prompt = build_fix_prompt(code, violations_json)

        try:
            result = await self._llm.generate_structured(
                ModelRole.FAST, prompt, FIX_SCHEMA, **FIX_SAMPLING
            )
            if not result:
                return []

            fixes: list[dict[str, Any]] = result.get("fixes", [])
            return fixes
        except Exception:
            logger.warning("LLM fix generation failed", exc_info=True)
            return []

    async def revert_file(self, file_path: str, original_content: str) -> None:
        """Revert a file to its original content if verification fails.

        Args:
            file_path: Path to the file.
            original_content: Content to restore.
        """
        Path(file_path).write_text(original_content)
        logger.info("Reverted %s after failed verification", file_path)
