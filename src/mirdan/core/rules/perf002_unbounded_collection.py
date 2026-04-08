"""PERF002: Unbounded collection growth detection rule."""

from __future__ import annotations

import re

from mirdan.core.rules.base import BaseRule, RuleContext, RuleTier
from mirdan.core.skip_regions import is_in_skip_region
from mirdan.models import Violation

# Patterns: infinite or unbounded loop constructs
_INFINITE_LOOP = re.compile(
    r"(?:while\s+(?:True|true|1)\s*[:{]|for\s*\{|loop\s*\{)",
    re.MULTILINE,
)

_ACCUMULATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\.append\s*\("), "list.append()"),
    (re.compile(r"\.push\s*\("), "array.push()"),
    (re.compile(r"\.add\s*\("), "collection.add()"),
    (re.compile(r"\.Add\s*\("), "collection.Add()"),
    (re.compile(r"\+=\s*\["), "+= [list]"),
]


class PERF002UnboundedCollectionRule(BaseRule):
    """Detect unbounded collection growth in loops without size limits.

    Known limitation: inner matches are not checked against skip regions.
    """

    @property
    def id(self) -> str:
        return "PERF002"

    @property
    def name(self) -> str:
        return "unbounded-collection"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset(
            {"python", "typescript", "javascript", "go", "java", "rust", "csharp", "auto"}
        )

    @property
    def tier(self) -> RuleTier:
        return RuleTier.FULL

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        """Detect collection growth in unbounded loops."""
        violations: list[Violation] = []
        lines = code.split("\n")

        for loop_match in _INFINITE_LOOP.finditer(code):
            if is_in_skip_region(loop_match.start(), context.skip_regions):
                continue

            loop_line = code[: loop_match.start()].count("\n")
            loop_end = min(loop_line + 30, len(lines))
            loop_body = "\n".join(lines[loop_line:loop_end])

            # Check if there is a size/length check in the loop body
            has_size_check = bool(
                re.search(r"(?:len\([^)]*\)|\.length|\.size|\.Count|\.Len\(\))\s*[><=]", loop_body)
            )
            if has_size_check:
                continue

            for acc_pattern, acc_desc in _ACCUMULATION_PATTERNS:
                acc_match = acc_pattern.search(loop_body)
                if acc_match:
                    acc_line_offset = loop_body[: acc_match.start()].count("\n")
                    violations.append(
                        Violation(
                            id="PERF002",
                            rule="unbounded-collection",
                            category="performance",
                            severity="warning",
                            message=(
                                f"Unbounded collection growth: {acc_desc} in infinite loop "
                                "without size limit check. May exhaust memory under load."
                            ),
                            line=loop_line + acc_line_offset + 1,
                            suggestion=(
                                "Add a maximum size check: if len(collection) >= MAX_SIZE: break"
                            ),
                        )
                    )
                    break

        return violations
