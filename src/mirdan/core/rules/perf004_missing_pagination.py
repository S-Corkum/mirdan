"""PERF004: Missing pagination detection rule.

Known limitation: negative lookahead only checks the current line for
limit/skip/take. Multi-line queries with .limit() on a subsequent line
may produce false positives.
"""

from __future__ import annotations

import re

from mirdan.core.rules.base import BaseRule, RuleContext, RuleTier
from mirdan.core.skip_regions import is_in_skip_region
from mirdan.models import Violation

# Patterns that fetch all records without pagination
_UNBOUNDED_QUERIES: list[tuple[re.Pattern[str], str, str]] = [
    # Python ORMs
    (
        re.compile(r"\.objects\.all\s*\(\s*\)"),
        "QuerySet.all() without pagination",
        "Add .[:limit] slicing or use Paginator",
    ),
    (
        re.compile(r"session\.query\s*\([^)]*\)\.all\s*\(\s*\)"),
        "session.query().all() without limit",
        "Add .limit(N) before .all() or use pagination",
    ),
    # JavaScript/TypeScript ORMs
    (
        re.compile(r"\.findMany\s*\(\s*\{?\s*\}?\s*\)"),
        "findMany() without take/skip",
        "Add take and skip parameters for pagination",
    ),
    # Go
    (
        re.compile(r'db\.Query\s*\(\s*"SELECT\s+\*[^"]*"(?!.*LIMIT)'),
        "SELECT * without LIMIT",
        'Add "LIMIT $1" parameter to the query',
    ),
    # C#
    (
        re.compile(r"\.ToListAsync?\s*\(\s*\)(?!.*(?:Take|Skip))"),
        "ToList() without Take/Skip",
        "Add .Take(pageSize).Skip(offset) before ToList()",
    ),
]


class PERF004MissingPaginationRule(BaseRule):
    """Detect queries that load all records without pagination."""

    @property
    def id(self) -> str:
        return "PERF004"

    @property
    def name(self) -> str:
        return "missing-pagination"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "typescript", "javascript", "go", "java", "csharp", "auto"})

    @property
    def tier(self) -> RuleTier:
        return RuleTier.FULL

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        """Detect queries without pagination."""
        violations: list[Violation] = []

        for query_pattern, query_desc, query_fix in _UNBOUNDED_QUERIES:
            for match in query_pattern.finditer(code):
                if is_in_skip_region(match.start(), context.skip_regions):
                    continue
                line_no = code[: match.start()].count("\n") + 1
                violations.append(
                    Violation(
                        id="PERF004",
                        rule="missing-pagination",
                        category="performance",
                        severity="warning",
                        message=(
                            f"Potential unbounded query: {query_desc}. "
                            "Loading all records can exhaust memory with large datasets."
                        ),
                        line=line_no,
                        suggestion=query_fix,
                    )
                )

        return violations
