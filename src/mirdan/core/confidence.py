"""Confidence calibration for validated code.

Aggregates validation signals into a calibrated confidence level
(HIGH/MEDIUM/LOW) with a single attention_focus item pointing the
developer to the most important manual verification.
"""

from __future__ import annotations

from mirdan.models import ConfidenceAssessment, SemanticCheck, Violation


class ConfidenceCalibrator:
    """Assess confidence in validated code.

    Deterministic assessment — no LLM calls. Rules:
    - LOW: any error-severity violation, or any security violation
    - MEDIUM: >3 warnings, or warning-severity semantic checks, or no test file
    - HIGH: everything else
    """

    def assess(
        self,
        violations: list[Violation],
        semantic_checks: list[SemanticCheck],
        test_file: str = "",
    ) -> ConfidenceAssessment:
        """Produce a calibrated confidence assessment.

        Args:
            violations: Violations from code validation.
            semantic_checks: Semantic review checks generated.
            test_file: Path to associated test file, empty if none.

        Returns:
            ConfidenceAssessment with level, reason, and attention_focus.
        """
        errors = [v for v in violations if v.severity == "error"]
        security = [v for v in violations if v.category == "security"]
        warnings = [v for v in violations if v.severity == "warning"]
        warning_checks = [s for s in semantic_checks if s.severity == "warning"]

        # LOW: error violations or security violations
        if errors:
            focus = self._pick_focus(errors, semantic_checks)
            return ConfidenceAssessment(
                level="low",
                reason=f"{len(errors)} error-severity violation(s) detected",
                attention_focus=focus,
            )

        if security:
            focus = self._pick_focus(security, semantic_checks)
            return ConfidenceAssessment(
                level="low",
                reason=f"{len(security)} security violation(s) detected",
                attention_focus=focus,
            )

        # MEDIUM: >3 warnings, or warning semantic checks, or no test file
        if len(warnings) > 3:
            focus = self._pick_focus(warnings, semantic_checks)
            return ConfidenceAssessment(
                level="medium",
                reason=f"{len(warnings)} warning-severity violations",
                attention_focus=focus,
            )

        if warning_checks:
            return ConfidenceAssessment(
                level="medium",
                reason=f"{len(warning_checks)} semantic concern(s) need review",
                attention_focus=warning_checks[0].question,
            )

        if not test_file:
            return ConfidenceAssessment(
                level="medium",
                reason="No associated test file detected",
                attention_focus="Verify behavior manually or add tests",
            )

        # HIGH: clean
        return ConfidenceAssessment(
            level="high",
            reason="All checks passed with zero critical issues",
            attention_focus="None — code is clean",
        )

    def _pick_focus(
        self,
        violations: list[Violation],
        semantic_checks: list[SemanticCheck],
    ) -> str:
        """Pick the most important attention focus item."""
        # Prefer semantic check with highest severity
        severity_rank = {"critical": 0, "warning": 1, "info": 2}
        if semantic_checks:
            sorted_checks = sorted(
                semantic_checks,
                key=lambda s: severity_rank.get(s.severity, 99),
            )
            return sorted_checks[0].question

        # Fall back to first violation message
        if violations:
            return violations[0].message

        return "Review code manually"
