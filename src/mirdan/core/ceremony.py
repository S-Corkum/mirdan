"""Adaptive ceremony advisor — scales enhance_prompt guidance depth."""

from __future__ import annotations

from mirdan.config import CeremonyConfig
from mirdan.models import CeremonyLevel, CeremonyPolicy, Intent, SessionContext, TaskType


class CeremonyAdvisor:
    """Determines appropriate ceremony level based on task complexity.

    Pure logic: deterministic inputs → deterministic outputs.
    Depends only on public types from models.py and config.py.
    """

    POLICIES: dict[CeremonyLevel, CeremonyPolicy] = {
        CeremonyLevel.MICRO: CeremonyPolicy(
            level=CeremonyLevel.MICRO,
            enhancement_mode="analyze_only",
            context_level="none",
            recommended_validation="quick_essential",
            filter_tool_recs=False,
        ),
        CeremonyLevel.LIGHT: CeremonyPolicy(
            level=CeremonyLevel.LIGHT,
            enhancement_mode="enhance",
            context_level="minimal",
            recommended_validation="full",
            filter_tool_recs=True,
        ),
        CeremonyLevel.STANDARD: CeremonyPolicy(
            level=CeremonyLevel.STANDARD,
            enhancement_mode="enhance",
            context_level="auto",
            recommended_validation="full",
            filter_tool_recs=False,
        ),
        CeremonyLevel.THOROUGH: CeremonyPolicy(
            level=CeremonyLevel.THOROUGH,
            enhancement_mode="enhance",
            context_level="comprehensive",
            recommended_validation="full",
            filter_tool_recs=False,
        ),
    }

    _TASK_TYPE_SCORES: dict[TaskType, int] = {
        TaskType.GENERATION: 2,
        TaskType.REFACTOR: 2,
        TaskType.DEBUG: 1,
        TaskType.TEST: 1,
        TaskType.REVIEW: 1,
        TaskType.DOCUMENTATION: 0,
        TaskType.PLANNING: 4,
        TaskType.UNKNOWN: 1,
    }

    def __init__(self, config: CeremonyConfig | None = None) -> None:
        if config is None:
            config = CeremonyConfig()
        self._config = config

    def determine_level(
        self,
        intent: Intent,
        prompt_length: int,
        session: SessionContext | None = None,
    ) -> CeremonyLevel:
        """Determine ceremony level from intent signals and session state.

        Args:
            intent: Analyzed intent from the prompt.
            prompt_length: Character length of the original prompt.
            session: Optional session for escalation checks.

        Returns:
            CeremonyLevel appropriate for the task complexity.
        """
        if not self._config.enabled:
            return CeremonyLevel.STANDARD

        if self._config.default_level != "auto":
            base = CeremonyLevel[self._config.default_level.upper()]
        else:
            base = self._estimate_base_level(intent, prompt_length)

        escalated = self._apply_escalations(base, intent, session)

        # Clamp to min_level
        min_level = CeremonyLevel[self._config.min_level.upper()]
        return max(escalated, min_level)

    def get_policy(self, level: CeremonyLevel) -> CeremonyPolicy:
        """Look up the frozen policy for a ceremony level."""
        return self.POLICIES[level]

    def explain(self, level: CeremonyLevel, intent: Intent) -> str:
        """Human-readable explanation of why this level was chosen.

        Returns a short string for response metadata so the AI assistant
        understands the ceremony decision.
        """
        parts: list[str] = [f"{level.name}:"]

        # Task type
        task_desc = intent.task_type.value
        if intent.task_types and len(intent.task_types) > 1:
            task_desc = "+".join(t.value for t in intent.task_types)
        parts.append(f"{task_desc} task")

        # Frameworks
        if intent.frameworks:
            n = len(intent.frameworks)
            parts.append(f"with {n} framework{'s' if n > 1 else ''}")

        # Escalation reasons
        reasons: list[str] = []
        if intent.touches_security and self._config.security_escalation:
            reasons.append("security escalation")
        if intent.touches_rag:
            reasons.append("RAG escalation")
        if intent.touches_knowledge_graph:
            reasons.append("KG escalation")
        if (
            self._config.ambiguity_escalation
            and intent.ambiguity_score >= self._config.ambiguity_threshold
        ):
            reasons.append("ambiguity escalation")
        if intent.task_type == TaskType.PLANNING:
            reasons.append("planning always thorough")

        if reasons:
            parts.append(f"({', '.join(reasons)})")

        return " ".join(parts)

    def _estimate_base_level(self, intent: Intent, prompt_length: int) -> CeremonyLevel:
        """Score the task complexity and map to a base ceremony level."""
        score = self._TASK_TYPE_SCORES.get(intent.task_type, 1)

        # Compound task
        if intent.task_types and len(intent.task_types) > 1:
            score += 1

        # Frameworks (max +2)
        score += min(len(intent.frameworks), 2)

        # Prompt length
        if prompt_length >= 200:
            score += 1
        if prompt_length >= 500:
            score += 1

        # Entities detected
        if intent.entities:
            score += 1

        # Map score to level
        if score <= 1:
            return CeremonyLevel.MICRO
        if score == 2:
            return CeremonyLevel.LIGHT
        if score <= 5:
            return CeremonyLevel.STANDARD
        return CeremonyLevel.THOROUGH

    def _apply_escalations(
        self,
        base: CeremonyLevel,
        intent: Intent,
        session: SessionContext | None,
    ) -> CeremonyLevel:
        """Apply escalation rules. Can only escalate, never de-escalate."""
        level = base

        # Security, RAG, KG → min STANDARD
        if intent.touches_security and self._config.security_escalation:
            level = max(level, CeremonyLevel.STANDARD)
        if intent.touches_rag:
            level = max(level, CeremonyLevel.STANDARD)
        if intent.touches_knowledge_graph:
            level = max(level, CeremonyLevel.STANDARD)

        # Ambiguity → min STANDARD
        if (
            self._config.ambiguity_escalation
            and intent.ambiguity_score >= self._config.ambiguity_threshold
        ):
            level = max(level, CeremonyLevel.STANDARD)

        # Planning → always THOROUGH
        if intent.task_type == TaskType.PLANNING:
            level = max(level, CeremonyLevel.THOROUGH)

        # Persistent violations → +1 level
        if session and session.unresolved_errors > 0:
            next_level = min(level + 1, CeremonyLevel.THOROUGH)
            level = CeremonyLevel(next_level)

        return level
