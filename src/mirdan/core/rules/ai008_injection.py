"""AI008: Injection vulnerability detection rule."""

from __future__ import annotations

import re

from mirdan.core.rules.base import BaseRule, RuleContext
from mirdan.core.skip_regions import is_in_skip_region
from mirdan.models import Violation


class AI008InjectionRule(BaseRule):
    """Detect injection vulnerabilities via f-string interpolation (AI008)."""

    _INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (
            re.compile(r"""f["'].*?(?:SELECT|INSERT|UPDATE|DELETE|DROP)\b.*?\{""", re.IGNORECASE),
            "SQL query built with f-string interpolation",
        ),
        (
            re.compile(r"""\beval\s*\(\s*f["']"""),
            "eval() with f-string input",
        ),
        (
            re.compile(r"""\bexec\s*\(\s*f["']"""),
            "exec() with f-string input",
        ),
        (
            re.compile(r"""\bos\.system\s*\(\s*f["']"""),
            "os.system() with f-string command",
        ),
        (
            re.compile(r"""\bsubprocess\.\w+\s*\([^)]*f["']"""),
            "subprocess call with f-string command",
        ),
    ]

    @property
    def id(self) -> str:
        return "AI008"

    @property
    def name(self) -> str:
        return "ai-injection-vulnerability"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "auto"})

    @property
    def is_quick(self) -> bool:
        return True

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        """Detect injection vulnerabilities via f-string interpolation."""
        # This rule applies to Python (f-strings) -- skip other languages
        if language not in ("python", "auto"):
            return []

        violations: list[Violation] = []

        for pattern, pattern_context in self._INJECTION_PATTERNS:
            for m in pattern.finditer(code):
                if is_in_skip_region(m.start(), context.skip_regions):
                    continue
                line_no = code[: m.start()].count("\n") + 1
                violations.append(
                    Violation(
                        id="AI008",
                        rule="ai-injection-vulnerability",
                        category="security",
                        severity="error",
                        message=(
                            f"Potential injection vulnerability: {pattern_context}."
                            " Use parameterized queries or input sanitization."
                        ),
                        line=line_no,
                        suggestion=(
                            "Use parameterized queries"
                            " (e.g., cursor.execute('SELECT ...', (param,)))"
                            " instead of f-string interpolation"
                        ),
                    )
                )

        return violations
