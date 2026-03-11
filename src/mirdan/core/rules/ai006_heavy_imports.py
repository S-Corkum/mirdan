"""AI006: Unnecessary heavy import detection rule."""

from __future__ import annotations

import re

from mirdan.core.rules.base import BaseRule, RuleContext
from mirdan.core.skip_regions import is_in_skip_region
from mirdan.models import Violation

# ---------------------------------------------------------------------------
# Heavy import alternatives (AI006)
# ---------------------------------------------------------------------------

_HEAVY_IMPORT_ALTERNATIVES: dict[str, dict[str, str]] = {
    "requests": {
        "pattern": r"\brequests\.get\s*\(",
        "alternative": "urllib.request.urlopen",
        "reason": "For simple GET requests, urllib.request avoids the requests dependency",
    },
    "pandas": {
        "pattern": r"\bpd\.read_csv\s*\([^)]*\)\s*$",
        "alternative": "csv module",
        "reason": "For simple CSV reading, the csv module avoids the heavy pandas dependency",
    },
    "numpy": {
        "pattern": r"\bnp\.(?:sum|mean|max|min|abs|sqrt)\s*\(\s*\[",
        "alternative": "math/statistics module",
        "reason": "For basic math on small lists, math/statistics avoids the numpy dependency",
    },
}


class AI006HeavyImportsRule(BaseRule):
    """Detect heavy library imports for trivially simple usage (AI006)."""

    @property
    def id(self) -> str:
        return "AI006"

    @property
    def name(self) -> str:
        return "ai-heavy-import"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "auto"})

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        """Detect heavy library imports for trivially simple usage."""
        if language not in ("python", "auto"):
            return []

        violations: list[Violation] = []

        for lib, info in _HEAVY_IMPORT_ALTERNATIVES.items():
            # Check if the library is imported
            import_match = re.search(
                rf"^(?:import\s+{lib}|from\s+{lib}\s+import)\b",
                code,
                re.MULTILINE,
            )
            if not import_match:
                continue
            if is_in_skip_region(import_match.start(), context.skip_regions):
                continue

            # Check if usage is trivially simple
            usage_pattern = re.compile(info["pattern"], re.MULTILINE)
            usages = list(usage_pattern.finditer(code))
            # Count total usages of the library
            all_usages = list(re.finditer(rf"\b{lib}\b", code))
            # Subtract the import line itself
            non_import_usages = [
                u
                for u in all_usages
                if u.start() != import_match.start()
                and not is_in_skip_region(u.start(), context.skip_regions)
            ]

            # Only flag if there are very few usages (1-2) and they match the simple pattern
            if len(non_import_usages) <= 2 and usages:
                line_no = code[: import_match.start()].count("\n") + 1
                violations.append(
                    Violation(
                        id="AI006",
                        rule="ai-heavy-import",
                        category="ai_quality",
                        severity="info",
                        message=(f"'{lib}' imported for simple usage. {info['reason']}."),
                        line=line_no,
                        suggestion=f"Consider using {info['alternative']} instead",
                    )
                )

        return violations
