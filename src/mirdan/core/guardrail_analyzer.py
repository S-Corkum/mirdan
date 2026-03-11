"""Cognitive guardrails — domain-aware pre-flight thinking prompts.

Surfaces 2-3 domain-specific considerations a senior engineer would
mention BEFORE coding starts. Different from quality_requirements
(coding rules) — these are THINKING prompts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import yaml

from mirdan.models import GuardrailAnalysis, Intent

if TYPE_CHECKING:
    from mirdan.config import GuardrailConfig

logger = logging.getLogger(__name__)


class GuardrailAnalyzer:
    """Analyze intent for domain-specific pre-flight guardrails.

    Template-based and fully deterministic — no LLM calls required.
    Loads guardrail domains from standards/guardrails.yaml at init time.
    """

    def __init__(self, config: GuardrailConfig) -> None:
        self._config = config
        self._domains: list[dict[str, Any]] = []
        self._load_domains()

    def _load_domains(self) -> None:
        """Load guardrail domains from standards/guardrails.yaml."""
        from importlib.resources import files

        try:
            standards_pkg = files("mirdan.standards")
        except ModuleNotFoundError:
            logger.warning("mirdan.standards package not found, skipping guardrail domains")
            return

        guardrails_file = standards_pkg.joinpath("guardrails.yaml")
        try:
            data = yaml.safe_load(guardrails_file.read_text(encoding="utf-8"))
            if data and isinstance(data, dict):
                self._domains = data.get("domains", [])
        except (yaml.YAMLError, OSError) as e:
            logger.warning("Failed to load guardrails.yaml: %s", e)

    def analyze(self, intent: Intent) -> list[GuardrailAnalysis]:
        """Match intent against guardrail domains, return matching guardrails.

        Args:
            intent: Analyzed intent with prompt text and entities.

        Returns:
            List of GuardrailAnalysis, with total guardrail items capped at max_guardrails.
        """
        if not self._config.enabled:
            return []

        # Build search text from prompt + entity values
        search_text = intent.original_prompt.lower()
        entity_text = " ".join(e.value.lower() for e in intent.entities)
        search_text = f"{search_text} {entity_text}"

        # Score each domain by trigger match count
        scored: list[tuple[int, dict[str, Any]]] = []
        for domain in self._domains:
            triggers = domain.get("triggers", [])
            score = sum(1 for t in triggers if t.lower() in search_text)
            if score > 0:
                scored.append((score, domain))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build results, capping total guardrail items at max_guardrails
        results: list[GuardrailAnalysis] = []
        total_items = 0
        for _, domain in scored:
            guardrails = domain.get("guardrails", [])
            remaining = self._config.max_guardrails - total_items
            if remaining <= 0:
                break
            capped = guardrails[:remaining]
            results.append(
                GuardrailAnalysis(
                    domain=domain.get("name", "unknown"),
                    guardrails=capped,
                )
            )
            total_items += len(capped)

        return results
