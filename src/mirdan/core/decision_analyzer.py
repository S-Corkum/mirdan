"""Decision intelligence — surfaces trade-off analysis for detected decision domains.

Loads YAML decision templates, matches triggers against intent and prompt text,
returns structured guidance. Config-gated, ceremony-gated in enhance_prompt.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import yaml

from mirdan.models import DecisionApproach, DecisionGuidance, Intent

if TYPE_CHECKING:
    from mirdan.config import DecisionConfig

logger = logging.getLogger(__name__)


class DecisionAnalyzer:
    """Analyze intent for decision domains and surface trade-off guidance.

    Template-based and fully deterministic — no LLM calls required.
    Loads YAML templates from standards/decisions/ at init time.
    """

    def __init__(self, config: DecisionConfig) -> None:
        self._config = config
        self._templates: list[dict[str, Any]] = []
        self._load_templates()

    def _load_templates(self) -> None:
        """Load all YAML templates from standards/decisions/ directory."""
        from importlib.resources import files

        try:
            standards_pkg = files("mirdan.standards")
        except ModuleNotFoundError:
            logger.warning("mirdan.standards package not found, skipping decision templates")
            return

        decisions_dir = standards_pkg.joinpath("decisions")
        try:
            for item in decisions_dir.iterdir():
                if item.name.endswith(".yaml"):
                    try:
                        data = yaml.safe_load(item.read_text(encoding="utf-8"))
                        if data and isinstance(data, dict):
                            self._templates.append(data)
                    except (yaml.YAMLError, OSError) as e:
                        logger.warning("Failed to load decision template %s: %s", item.name, e)
        except (OSError, TypeError) as e:
            logger.warning("Failed to iterate decisions directory: %s", e)

    def analyze(self, intent: Intent) -> list[DecisionGuidance]:
        """Match intent against decision templates, return top matches.

        Args:
            intent: Analyzed intent with prompt text and entities.

        Returns:
            List of DecisionGuidance, capped at max_decisions.
        """
        if not self._config.enabled:
            return []

        # Build search text from prompt + entity values
        search_text = intent.original_prompt.lower()
        entity_text = " ".join(e.value.lower() for e in intent.entities)
        search_text = f"{search_text} {entity_text}"

        # Score each template by trigger match count
        scored: list[tuple[int, dict[str, Any]]] = []
        for template in self._templates:
            triggers = template.get("triggers", [])
            score = sum(1 for t in triggers if t.lower() in search_text)
            if score > 0:
                scored.append((score, template))

        # Sort by score descending, take top N
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self._config.max_decisions]

        return [self._template_to_guidance(t) for _, t in top]

    def _template_to_guidance(self, template: dict[str, Any]) -> DecisionGuidance:
        """Convert a YAML template dict to a DecisionGuidance model."""
        approaches = [
            DecisionApproach(
                name=a["name"],
                when_best=a["when_best"],
                when_avoid=a["when_avoid"],
                complexity=a.get("complexity", "medium"),
            )
            for a in template.get("approaches", [])
        ]
        return DecisionGuidance(
            domain=template.get("name", "unknown"),
            approaches=approaches,
            senior_questions=template.get("senior_questions", []),
        )
