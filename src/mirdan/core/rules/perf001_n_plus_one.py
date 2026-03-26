"""PERF001: N+1 query detection rule."""

from __future__ import annotations

import re

from mirdan.core.rules.base import BaseRule, RuleContext, RuleTier
from mirdan.core.skip_regions import is_in_skip_region
from mirdan.models import Violation

# Loop patterns per language family
_LOOP_PATTERN = re.compile(
    r"(?:for\s*\(|for\s+\w+|while\s*\(|\.forEach\s*\(|\.map\s*\(|for\s+.*range)",
    re.MULTILINE,
)

# Query patterns that indicate database access.
# Uses specific ORM patterns to avoid false positives on dict.get(), array.filter(), etc.
_QUERY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Python ORMs — specific patterns only
    (re.compile(r"session\.execute\s*\(|cursor\.execute\s*\(", re.MULTILINE), "SQL execute"),
    (re.compile(r"\.objects\.get\s*\(|\.objects\.filter\s*\(", re.MULTILINE), "Django ORM query"),
    # JavaScript/TypeScript ORMs — await + specific method names
    (
        re.compile(
            r"await\s+\w+\.findOne\s*\(|await\s+\w+\.findUnique\s*\(|await\s+\w+\.findFirst\s*\(",
            re.MULTILINE,
        ),
        "ORM findOne",
    ),
    # Go database
    (re.compile(r"db\.(?:Query|QueryRow|Exec)\s*\(", re.MULTILINE), "Go DB query"),
    # Java JPA/JDBC
    (
        re.compile(
            r"entityManager\.find\s*\(|\.createQuery\s*\(|\.createNativeQuery\s*\(",
            re.MULTILINE,
        ),
        "JPA query",
    ),
    # Rust sqlx
    (re.compile(r"sqlx::query\s*[!(]|\.fetch_one\s*\(", re.MULTILINE), "sqlx query"),
    # C# EF Core
    (
        re.compile(
            r"\.FindAsync\s*\(|\.FirstOrDefaultAsync\s*\(|\.SingleAsync\s*\(",
            re.MULTILINE,
        ),
        "EF Core query",
    ),
]


class PERF001NPlusOneRule(BaseRule):
    """Detect N+1 query patterns: database queries inside loops.

    Known limitation: inner matches are not checked against skip regions.
    Commented-out code inside loops may trigger false positives.
    """

    @property
    def id(self) -> str:
        return "PERF001"

    @property
    def name(self) -> str:
        return "n-plus-one-query"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset(
            {"python", "typescript", "javascript", "go", "java", "rust", "csharp", "auto"}
        )

    @property
    def tier(self) -> RuleTier:
        return RuleTier.FULL

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        """Detect database queries inside loop constructs."""
        violations: list[Violation] = []
        lines = code.split("\n")

        for loop_match in _LOOP_PATTERN.finditer(code):
            if is_in_skip_region(loop_match.start(), context.skip_regions):
                continue

            loop_line = code[: loop_match.start()].count("\n")
            # Search within a reasonable window after the loop start (next 20 lines)
            loop_end = min(loop_line + 20, len(lines))
            loop_body = "\n".join(lines[loop_line:loop_end])

            for query_pattern, query_desc in _QUERY_PATTERNS:
                query_match = query_pattern.search(loop_body)
                if query_match:
                    query_line_offset = loop_body[: query_match.start()].count("\n")
                    violations.append(
                        Violation(
                            id="PERF001",
                            rule="n-plus-one-query",
                            category="performance",
                            severity="warning",
                            message=(
                                f"Potential N+1 query: {query_desc} inside loop. "
                                "Each iteration executes a separate database query."
                            ),
                            line=loop_line + query_line_offset + 1,
                            suggestion=(
                                "Batch the query before the loop: fetch all needed data "
                                "in one query, then iterate over the results."
                            ),
                        )
                    )
                    break  # One violation per loop

        return violations
