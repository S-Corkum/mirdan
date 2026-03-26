"""RS003: Undocumented unsafe block detection rule."""

from __future__ import annotations

import re

from mirdan.core.rules.base import BaseRule, RuleContext, RuleTier
from mirdan.core.skip_regions import is_in_skip_region
from mirdan.models import Violation

_UNSAFE_BLOCK = re.compile(r"unsafe\s*\{", re.MULTILINE)
_SAFETY_COMMENT = re.compile(r"//\s*SAFETY:", re.IGNORECASE)


class RS003UndocumentedUnsafeRule(BaseRule):
    """Detect unsafe blocks without SAFETY documentation."""

    @property
    def id(self) -> str:
        return "RS003"

    @property
    def name(self) -> str:
        return "undocumented-unsafe"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"rust", "auto"})

    @property
    def tier(self) -> RuleTier:
        return RuleTier.FULL

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        """Detect unsafe blocks without SAFETY comment in preceding lines."""
        if language not in ("rust", "auto"):
            return []

        violations: list[Violation] = []
        lines = code.split("\n")

        for m in _UNSAFE_BLOCK.finditer(code):
            if is_in_skip_region(m.start(), context.skip_regions):
                continue

            line_no = code[: m.start()].count("\n")
            # Check preceding 3 lines for SAFETY comment
            start_check = max(0, line_no - 3)
            preceding = "\n".join(lines[start_check : line_no + 1])
            if not _SAFETY_COMMENT.search(preceding):
                violations.append(
                    Violation(
                        id="RS003",
                        rule="undocumented-unsafe",
                        category="security",
                        severity="warning",
                        message="unsafe block without SAFETY comment documenting invariants",
                        line=line_no + 1,
                        suggestion="Add '// SAFETY: [explanation]' comment above the unsafe block",
                    )
                )

        return violations
