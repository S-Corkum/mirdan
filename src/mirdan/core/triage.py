"""Triage engine — classify coding tasks before the paid model sees them."""

from __future__ import annotations

import logging
from typing import Any

from mirdan.config import LLMConfig
from mirdan.llm.prompts.triage import (
    TRIAGE_SAMPLING,
    TRIAGE_SCHEMA,
    build_triage_prompt,
)
from mirdan.models import (
    Intent,
    ModelRole,
    TaskClassification,
    TaskType,
    TriageResult,
)

logger = logging.getLogger(__name__)

# Confidence below this threshold triggers escalation to PAID_REQUIRED.
_LOW_CONFIDENCE_THRESHOLD = 0.7


class TriageEngine:
    """Classifies coding tasks as LOCAL_ONLY / LOCAL_ASSIST / PAID_MINIMAL / PAID_REQUIRED.

    Uses rules-based pre-filter for security/planning/ambiguous tasks,
    then calls the local FAST model for everything else.
    """

    def __init__(self, llm_manager: Any = None, config: LLMConfig | None = None) -> None:
        self._llm = llm_manager
        self._config = config or LLMConfig()

    async def classify(self, prompt: str, intent: Intent | None = None) -> TriageResult | None:
        """Classify a task into a triage category.

        Args:
            prompt: The developer's original task description.
            intent: Pre-analyzed intent (optional — provides pre-filter signals).

        Returns:
            TriageResult, or None if classification is not possible.
        """
        # Pre-filter: escalate security, high-ambiguity, and planning tasks
        if intent is not None:
            prefilter = self._prefilter(intent)
            if prefilter is not None:
                return prefilter

        # LLM classification
        if not self._llm or not self._config.triage:
            return None

        prompt_text = build_triage_prompt(prompt)
        result = await self._llm.generate_structured(
            ModelRole.FAST, prompt_text, TRIAGE_SCHEMA, **TRIAGE_SAMPLING
        )

        if not result:
            return None

        return self._parse_result(result)

    def _prefilter(self, intent: Intent) -> TriageResult | None:
        """Rules-based pre-filter that bypasses the LLM entirely.

        Args:
            intent: Analyzed intent with security/ambiguity signals.

        Returns:
            TriageResult for tasks that should always be PAID_REQUIRED, or None.
        """
        if intent.touches_security:
            return TriageResult(
                classification=TaskClassification.PAID_REQUIRED,
                confidence=0.95,
                reasoning="Security-sensitive task requires paid model",
            )

        if intent.ambiguity_score > 0.7:
            return TriageResult(
                classification=TaskClassification.PAID_REQUIRED,
                confidence=0.90,
                reasoning="High ambiguity requires paid model for clarification",
            )

        if intent.task_type == TaskType.PLANNING:
            return TriageResult(
                classification=TaskClassification.PAID_REQUIRED,
                confidence=0.90,
                reasoning="Planning tasks require deep reasoning",
            )

        return None

    def _parse_result(self, result: dict[str, Any]) -> TriageResult | None:
        """Parse and validate LLM classification result.

        Args:
            result: Parsed JSON from the LLM.

        Returns:
            Validated TriageResult, or None if unparseable.
        """
        try:
            classification = TaskClassification(
                result.get("classification", "paid_required")
            )
        except ValueError:
            logger.warning("Unknown classification: %s", result.get("classification"))
            return None

        confidence = float(result.get("confidence", 0.0))
        reasoning = str(result.get("reasoning", ""))

        # Low confidence → escalate to PAID_REQUIRED
        if confidence < _LOW_CONFIDENCE_THRESHOLD:
            logger.info(
                "Triage confidence %.2f < %.2f, escalating to PAID_REQUIRED",
                confidence,
                _LOW_CONFIDENCE_THRESHOLD,
            )
            return TriageResult(
                classification=TaskClassification.PAID_REQUIRED,
                confidence=confidence,
                reasoning=f"Low confidence ({confidence:.2f}): {reasoning}",
            )

        return TriageResult(
            classification=classification,
            confidence=confidence,
            reasoning=reasoning,
        )

    @staticmethod
    def get_ceremony_override(classification: TaskClassification) -> str | None:
        """Map triage classification to ceremony level override.

        Args:
            classification: Triage result.

        Returns:
            Ceremony level string, or None for PAID_REQUIRED (no override).
        """
        return {
            TaskClassification.LOCAL_ONLY: "micro",
            TaskClassification.LOCAL_ASSIST: "light",
            TaskClassification.PAID_MINIMAL: "standard",
        }.get(classification)

    @staticmethod
    def get_token_budget(classification: TaskClassification) -> int:
        """Map triage classification to token budget for the paid model.

        Args:
            classification: Triage result.

        Returns:
            Token budget (0 means unlimited).
        """
        return {
            TaskClassification.LOCAL_ONLY: 0,
            TaskClassification.LOCAL_ASSIST: 2000,
            TaskClassification.PAID_MINIMAL: 4000,
            TaskClassification.PAID_REQUIRED: 0,
        }.get(classification, 0)
