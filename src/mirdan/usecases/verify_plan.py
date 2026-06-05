"""Usecase: verify a flat plan is internally executable (no brief, no LLM).

Mechanical self-check — does the plan reference files that exist, depend only on
steps that exist (no cycles), avoid vague cross-references, and give every step
its grounding fields? Deterministic and local; ~milliseconds on real plans.

Replaces the 2.1.0 ``verify_plan_against_brief`` for the brief-free workflow,
keeping its mechanical engine and dropping the brief-coupled semantic checks.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from mirdan.core import plan_mechanics
from mirdan.core.plan_validator import PlanValidator

# Flat-plan field-presence issues are emitted as "Step <n>: <message>".
_STEP_ISSUE_RE = re.compile(r"Step ([\w.\-]+):\s+(.+)$")


class VerifyPlanUseCase:
    """Verify a flat plan's internal consistency via mechanical checks."""

    def __init__(self, project_root: Path | None = None) -> None:
        self._plan_validator = PlanValidator()
        # Anchor file-existence checks here; defaults to CWD (the project root
        # during MCP-tool execution).
        self._project_root = project_root or Path.cwd()

    def execute(self, plan_path: str) -> dict[str, Any]:
        plan_p = Path(plan_path)
        if not plan_p.exists():
            return self._error(f"plan file not found: {plan_path}")

        plan_text = plan_p.read_text()

        score_result = self._plan_validator.validate(plan_text, template_mode="flat")
        missing_grounding = self._missing_grounding_from_issues(score_result.issues)

        steps = [(str(num), body) for num, body in self._plan_validator._extract_steps(plan_text)]
        phantom_files = plan_mechanics.check_phantom_files(steps, self._project_root)
        dependency_errors = plan_mechanics.check_dependencies(steps)
        vague_cross_references = plan_mechanics.check_vague_cross_references(steps)
        # lld_gaps is advisory (soft) — it does not fail `verified`.
        lld_gaps = plan_mechanics.check_lld_gaps(plan_text, steps)

        coverage_score = self._score(
            phantom_files=phantom_files,
            dependency_errors=dependency_errors,
            missing_grounding=missing_grounding,
            vague_cross_references=vague_cross_references,
        )
        verified = not (
            phantom_files or dependency_errors or vague_cross_references or missing_grounding
        )

        summary = self._summary(
            verified=verified,
            phantom_files_count=len(phantom_files),
            dependency_errors_count=len(dependency_errors),
            vague_cross_references_count=len(vague_cross_references),
            grounding_count=len(missing_grounding),
        )
        if lld_gaps:
            summary += f" · {len(lld_gaps)} low-level-design advisory(ies)"

        return {
            "verified": verified,
            "coverage_score": coverage_score,
            "phantom_files": phantom_files,
            "dependency_errors": dependency_errors,
            "vague_cross_references": vague_cross_references,
            "missing_grounding": missing_grounding,
            "lld_gaps": lld_gaps,
            "summary": summary,
        }

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _error(msg: str) -> dict[str, Any]:
        return {
            "verified": False,
            "coverage_score": 0.0,
            "error": msg,
            "phantom_files": [],
            "dependency_errors": [],
            "vague_cross_references": [],
            "missing_grounding": [],
            "lld_gaps": [],
        }

    @staticmethod
    def _missing_grounding_from_issues(issues: list[str]) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for issue in issues:
            m = _STEP_ISSUE_RE.match(issue)
            if m:
                result.append({"step_id": m.group(1), "issue": m.group(2)})
        return result

    @staticmethod
    def _score(
        phantom_files: list[dict[str, str]],
        dependency_errors: list[dict[str, str]],
        missing_grounding: list[dict[str, str]],
        vague_cross_references: list[dict[str, str]],
    ) -> float:
        score = 1.0
        # Phantom files are the severest mechanical finding.
        score -= 0.15 * len(phantom_files)
        score -= 0.10 * len(dependency_errors)
        score -= 0.05 * len(missing_grounding)
        score -= 0.03 * len(vague_cross_references)
        return max(0.0, round(score, 3))

    @staticmethod
    def _summary(
        verified: bool,
        phantom_files_count: int,
        dependency_errors_count: int,
        vague_cross_references_count: int,
        grounding_count: int,
    ) -> str:
        if verified:
            return "PASS — plan is internally consistent and executable"
        parts: list[str] = []
        if phantom_files_count:
            parts.append(f"{phantom_files_count} phantom file references")
        if dependency_errors_count:
            parts.append(f"{dependency_errors_count} dependency errors")
        if grounding_count:
            parts.append(f"{grounding_count} steps missing grounding")
        if vague_cross_references_count:
            parts.append(f"{vague_cross_references_count} vague cross-references")
        return "FAIL — " + "; ".join(parts)
