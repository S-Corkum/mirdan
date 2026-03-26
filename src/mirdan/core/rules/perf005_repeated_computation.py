"""PERF005: Repeated computation in loops detection rule."""

from __future__ import annotations

import re

from mirdan.core.rules.base import BaseRule, RuleContext, RuleTier
from mirdan.core.skip_regions import is_in_skip_region
from mirdan.models import Violation

# Expensive function calls that are genuinely wasteful when repeated in loops.
# Excludes datetime.now()/Date.now() (intentional for timing) and
# json.loads()/JSON.parse() (different data each iteration).
_EXPENSIVE_CALLS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"re\.compile\s*\("), "re.compile()"),
    (re.compile(r"os\.environ\.get\s*\("), "os.environ.get()"),
    (re.compile(r"process\.env\.\w+"), "process.env access"),
    (re.compile(r"Environment\.GetEnvironmentVariable\s*\("), "GetEnvironmentVariable()"),
    (re.compile(r"os\.Getenv\s*\("), "os.Getenv()"),
    (re.compile(r"System\.getenv\s*\("), "System.getenv()"),
]

_LOOP_START = re.compile(
    r"(?:for\s*\(|for\s+\w+|while\s*\(|\.forEach\s*\(|\.map\s*\(|for\s+.*range|loop\s*\{)",
    re.MULTILINE,
)


class PERF005RepeatedComputationRule(BaseRule):
    """Detect repeated expensive computations inside loops.

    Known limitation: inner matches are not checked against skip regions.
    """

    @property
    def id(self) -> str:
        return "PERF005"

    @property
    def name(self) -> str:
        return "repeated-computation"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset(
            {"python", "typescript", "javascript", "go", "java", "rust", "csharp", "auto"}
        )

    @property
    def tier(self) -> RuleTier:
        return RuleTier.FULL

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        """Detect expensive calls inside loops that could be hoisted."""
        violations: list[Violation] = []
        lines = code.split("\n")

        for loop_match in _LOOP_START.finditer(code):
            if is_in_skip_region(loop_match.start(), context.skip_regions):
                continue

            loop_line = code[: loop_match.start()].count("\n")
            loop_end = min(loop_line + 20, len(lines))
            loop_body = "\n".join(lines[loop_line:loop_end])

            for call_pattern, call_desc in _EXPENSIVE_CALLS:
                call_match = call_pattern.search(loop_body)
                if call_match:
                    call_offset = loop_body[: call_match.start()].count("\n")
                    violations.append(
                        Violation(
                            id="PERF005",
                            rule="repeated-computation",
                            category="performance",
                            severity="info",
                            message=(
                                f"Repeated computation: {call_desc} inside loop. "
                                "Consider computing once before the loop."
                            ),
                            line=loop_line + call_offset + 1,
                            suggestion=(
                                f"Hoist {call_desc} outside the loop if the "
                                "result does not depend on the loop variable."
                            ),
                        )
                    )
                    break

        return violations
