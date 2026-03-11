"""AI005: Inconsistent error handling detection rule."""

from __future__ import annotations

import re

from mirdan.core.rules.base import BaseRule, RuleContext
from mirdan.core.skip_regions import is_in_skip_region
from mirdan.models import Violation


class AI005ErrorHandlingRule(BaseRule):
    """Detect inconsistent error handling patterns (AI005)."""

    @property
    def id(self) -> str:
        return "AI005"

    @property
    def name(self) -> str:
        return "ai-inconsistent-errors"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "typescript", "javascript", "auto"})

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        """Detect inconsistent error handling patterns."""
        if language not in ("python", "typescript", "javascript", "auto"):
            return []

        violations: list[Violation] = []

        if language in ("python", "auto"):
            violations.extend(self._check_ai005_python(code, context.skip_regions))
        if language in ("typescript", "javascript"):
            violations.extend(self._check_ai005_typescript(code, context.skip_regions))

        return violations

    def _check_ai005_python(self, code: str, skip_regions: list[int]) -> list[Violation]:
        """AI005 checks for Python."""
        violations: list[Violation] = []

        # Detect bare except mixed with specific except in same file
        bare_excepts = [
            code[: m.start()].count("\n") + 1
            for m in re.finditer(r"^\s*except\s*:", code, re.MULTILINE)
            if not is_in_skip_region(m.start(), skip_regions)
        ]
        specific_excepts = [
            code[: m.start()].count("\n") + 1
            for m in re.finditer(r"^\s*except\s+\w+", code, re.MULTILINE)
            if not is_in_skip_region(m.start(), skip_regions)
        ]

        if bare_excepts and specific_excepts:
            violations.extend(
                Violation(
                    id="AI005",
                    rule="ai-inconsistent-errors",
                    category="ai_quality",
                    severity="warning",
                    message=(
                        "Bare 'except:' mixed with specific exception handlers in the"
                        " same file. Use specific exception types consistently."
                    ),
                    line=line_no,
                    suggestion="Replace bare 'except:' with specific exception types",
                )
                for line_no in bare_excepts
            )

        return violations

    def _check_ai005_typescript(self, code: str, skip_regions: list[int]) -> list[Violation]:
        """AI005 checks for TypeScript/JavaScript."""
        violations: list[Violation] = []

        # Detect empty catch blocks mixed with handled catch blocks
        empty_catches = [
            code[: m.start()].count("\n") + 1
            for m in re.finditer(r"catch\s*\([^)]*\)\s*\{\s*\}", code)
            if not is_in_skip_region(m.start(), skip_regions)
        ]
        handled_catches = [
            code[: m.start()].count("\n") + 1
            for m in re.finditer(r"catch\s*\([^)]*\)\s*\{[^}]+\}", code, re.DOTALL)
            if not is_in_skip_region(m.start(), skip_regions)
            and "{}" not in m.group(0).replace(" ", "")
        ]

        if empty_catches and handled_catches:
            violations.extend(
                Violation(
                    id="AI005",
                    rule="ai-inconsistent-errors",
                    category="ai_quality",
                    severity="warning",
                    message=(
                        "Empty catch block mixed with handled catch blocks."
                        " Handle or explicitly comment why the error is ignored."
                    ),
                    line=line_no,
                    suggestion="Add error handling or a comment explaining why it's ignored",
                )
                for line_no in empty_catches
            )

        return violations
