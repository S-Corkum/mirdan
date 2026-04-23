"""Brief Validator — enforces section gates per brief-driven pipeline rubric."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from mirdan.config import BriefConfig
from mirdan.models import BriefQualityScore


class BriefValidator:
    """Validates briefs for brief-first plan creation (2.1.0)."""

    SECTION_HEADING_PATTERN = re.compile(r"^##\s+(.+)$", re.MULTILINE)

    # Vague language patterns that fail the specificity gate.
    VAGUE_PATTERNS: list[tuple[str, str]] = [
        (r"\bfollow\s+best\s+practices\b", "Constraint too vague: specify what practices"),
        (r"\buse\s+good\s+patterns\b", "Constraint too vague: name the pattern"),
        (r"\bshould\s+be\s+fast\b", "AC untestable: specify latency threshold"),
        (r"\bshould\s+work\s+well\b", "AC untestable: specify observable outcome"),
    ]

    def __init__(self, config: BriefConfig | None = None) -> None:
        self._config = config or BriefConfig()

    def validate(self, brief_text: str) -> BriefQualityScore:
        """Validate a brief against the pipeline rubric.

        Pure structural + regex checks; semantic checks (AC testability via LLM)
        are layered in verify_plan_against_brief, not here.
        """
        sections = self._extract_sections(brief_text)
        missing_required = [
            s for s in self._config.required_sections if s not in sections
        ]
        thin_recommended = [
            s for s in self._config.recommended_sections if s not in sections
        ]
        gaps: list[dict[str, Any]] = [
            {"section": s, "severity": "error", "issue": "missing required section"}
            for s in missing_required
        ]

        if "Business Acceptance Criteria" in sections:
            ac_count = self._count_checklist_items(sections["Business Acceptance Criteria"])
            if ac_count < self._config.min_acs:
                gaps.append(
                    {
                        "section": "Business Acceptance Criteria",
                        "severity": "error",
                        "issue": f"only {ac_count} ACs, need {self._config.min_acs}",
                    }
                )

        for pattern, msg in self.VAGUE_PATTERNS:
            if re.search(pattern, brief_text, re.IGNORECASE):
                gaps.append({"section": "global", "severity": "warning", "issue": msg})

        passed = not missing_required and not any(
            g["severity"] == "error" for g in gaps
        )
        score = self._score(missing_required, thin_recommended, gaps)
        return BriefQualityScore(
            passed=passed,
            score=score,
            gaps=gaps,
            would_pass_after_fixes=len(missing_required) == 0,
            missing_required=missing_required,
            thin_recommended=thin_recommended,
        )

    def validate_file(self, brief_path: Path) -> BriefQualityScore:
        """Validate a brief at a filesystem path."""
        return self.validate(brief_path.read_text())

    def _extract_sections(self, text: str) -> dict[str, str]:
        """Map heading -> section body text."""
        matches = list(self.SECTION_HEADING_PATTERN.finditer(text))
        result: dict[str, str] = {}
        for i, m in enumerate(matches):
            heading = m.group(1).strip()
            normalized = self._normalize_heading(heading)
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            result[normalized] = text[start:end]
        return result

    @staticmethod
    def _normalize_heading(h: str) -> str:
        # "Constraints (Non-Negotiable)" -> "Constraints"
        return re.sub(r"\s*\(.*?\)\s*$", "", h).strip()

    @staticmethod
    def _count_checklist_items(body: str) -> int:
        return len(re.findall(r"^\s*-\s*\[\s*[ xX]?\s*\]", body, re.MULTILINE))

    def _score(
        self,
        missing: list[str],
        thin: list[str],
        gaps: list[dict[str, Any]],
    ) -> float:
        base = 1.0
        base -= 0.2 * len(missing)
        base -= 0.05 * len(thin)
        base -= 0.05 * sum(1 for g in gaps if g["severity"] == "warning")
        return max(0.0, min(1.0, base))
