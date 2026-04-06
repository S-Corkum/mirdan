"""Smart validator — LLM-enriched code quality analysis."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from mirdan.config import LLMConfig

if TYPE_CHECKING:
    from mirdan.llm.manager import LLMManager
from mirdan.llm.prompts.validation import (
    COMBINED_ANALYSIS_SCHEMA,
    VALIDATION_SAMPLING,
    build_validation_prompt,
)
from mirdan.models import ModelRole, SmartValidationResult, Violation

logger = logging.getLogger(__name__)


class SmartValidator:
    """Enriches validation with false positive filtering, root cause grouping, and fixes.

    Uses a single combined LLM call (not three sequential) for efficiency.
    Includes injection mitigation via delimiters and sanity caps on FP ratio.
    """

    def __init__(
        self,
        llm_manager: LLMManager | None = None,
        config: LLMConfig | None = None,
        fix_validator: Callable[[str, str], list[Any]] | None = None,
    ) -> None:
        self._llm = llm_manager
        self._config = config or LLMConfig()
        self._fix_validator = fix_validator

    async def analyze(
        self,
        violations: list[Violation],
        code: str,
        language: str,
    ) -> SmartValidationResult | None:
        """Analyze violations with the local LLM for FP filtering and root causes.

        Args:
            violations: Detected violations from the rule engine.
            code: Source code that was validated.
            language: Detected programming language.

        Returns:
            SmartValidationResult with per-violation assessments and root causes,
            or None if LLM is unavailable or no violations to analyze.
        """
        if not self._llm or not self._config.smart_validation or not violations:
            return None

        # Build prompt with injection mitigation
        violations_json = json.dumps(
            [v.to_dict() for v in violations], indent=2
        )
        prompt = build_validation_prompt(
            code=code,
            violations_json=violations_json,
            supports_thinking=True,
        )

        # Single LLM call
        result = await self._llm.generate_structured(
            ModelRole.FAST, prompt, COMBINED_ANALYSIS_SCHEMA, **VALIDATION_SAMPLING
        )
        if not result:
            return None

        per_violation: list[dict[str, Any]] = result.get("per_violation", [])
        root_causes: list[dict[str, Any]] = result.get("root_causes", [])

        # Sanity check: cap false positive ratio
        was_capped = self._apply_sanity_cap(per_violation)

        # Sanity check: re-validate LLM-generated fixes
        if self._config.validate_llm_fixes and self._fix_validator:
            self._validate_fixes(per_violation, language)

        return SmartValidationResult(
            per_violation=per_violation,
            root_causes=root_causes,
            was_sanity_capped=was_capped,
        )

    def _apply_sanity_cap(self, per_violation: list[dict[str, Any]]) -> bool:
        """Reject all FP assessments if the ratio exceeds the configured maximum.

        Args:
            per_violation: Per-violation assessment list (mutated in place).

        Returns:
            True if the cap was triggered.
        """
        if not per_violation:
            return False

        fp_count = sum(
            1
            for v in per_violation
            if v.get("assessment") == "false_positive"
        )
        fp_ratio = fp_count / len(per_violation)

        if fp_ratio > self._config.max_false_positive_ratio:
            logger.warning(
                "FP ratio %.2f exceeds max %.2f — rejecting all FP assessments",
                fp_ratio,
                self._config.max_false_positive_ratio,
            )
            for item in per_violation:
                item["assessment"] = "confirmed"
                item.pop("false_positive_reason", None)
            return True

        return False

    def _validate_fixes(
        self, per_violation: list[dict[str, Any]], language: str
    ) -> None:
        """Re-validate LLM-generated fixes through the rule engine.

        Drops fixes that introduce new violations. Mutates per_violation in place.

        Args:
            per_violation: Per-violation list with potential fix_code entries.
            language: Programming language for re-validation.
        """
        assert self._fix_validator is not None
        for item in per_violation:
            fix_code = item.get("fix_code")
            if not fix_code:
                continue

            new_violations = self._fix_validator(fix_code, language)
            if new_violations:
                logger.info(
                    "Dropping LLM fix for %s — introduces %d new violations",
                    item.get("violation_id", "?"),
                    len(new_violations),
                )
                item.pop("fix_code", None)
                item["fix_confidence"] = 0.0
