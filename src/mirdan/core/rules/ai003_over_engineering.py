"""AI003: Over-engineering detection rule."""

from __future__ import annotations

import re

from mirdan.core.rules.base import BaseRule, RuleContext
from mirdan.core.skip_regions import is_in_skip_region
from mirdan.models import Violation


class AI003OverEngineeringRule(BaseRule):
    """Detect over-engineering patterns common in AI-generated code (AI003)."""

    _RE_ABSTRACT_CLASS = re.compile(
        r"^class\s+(\w+)\s*\(.*?\bABC\b.*?\)\s*:",
        re.MULTILINE,
    )
    _RE_CLASS_DEF = re.compile(r"^class\s+(\w+)\s*(?:\(([^)]*)\))?\s*:", re.MULTILINE)
    _RE_GENERIC_PARAMS = re.compile(r"\[([^\]]+)\]")
    _RE_FACTORY_FUNC = re.compile(
        r"^def\s+(create_|make_|build_|get_)\w+\s*\(",
        re.MULTILINE,
    )

    @property
    def id(self) -> str:
        return "AI003"

    @property
    def name(self) -> str:
        return "ai-over-engineering"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "typescript", "javascript", "java", "auto"})

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        """Detect over-engineering patterns common in AI-generated code."""
        if language not in ("python", "typescript", "javascript", "java", "auto"):
            return []

        violations: list[Violation] = []

        # 1. Abstract classes with only 1 concrete subclass in same file
        abstract_classes: dict[str, int] = {}
        for m in self._RE_ABSTRACT_CLASS.finditer(code):
            if is_in_skip_region(m.start(), context.skip_regions):
                continue
            abstract_classes[m.group(1)] = code[: m.start()].count("\n") + 1

        for abc_name, abc_line in abstract_classes.items():
            # Count classes that inherit from this abstract class
            subclass_count = 0
            pattern = re.compile(
                rf"^class\s+\w+\s*\([^)]*\b{re.escape(abc_name)}\b[^)]*\)\s*:",
                re.MULTILINE,
            )
            for m in pattern.finditer(code):
                if not is_in_skip_region(m.start(), context.skip_regions):
                    subclass_count += 1
            if subclass_count == 1:
                violations.append(
                    Violation(
                        id="AI003",
                        rule="ai-over-engineering",
                        category="ai_quality",
                        severity="warning",
                        message=(
                            f"Abstract class '{abc_name}' has only 1 concrete subclass"
                            " in this file. Consider using a concrete class instead."
                        ),
                        line=abc_line,
                        suggestion="Remove the abstraction unless more subclasses are planned",
                    )
                )

        # 2. Excessive generic type parameters (>5)
        for m in self._RE_CLASS_DEF.finditer(code):
            if is_in_skip_region(m.start(), context.skip_regions):
                continue
            class_name = m.group(1)
            bases = m.group(2) or ""
            generic_match = self._RE_GENERIC_PARAMS.search(bases)
            if generic_match:
                params = [p.strip() for p in generic_match.group(1).split(",")]
                if len(params) > 5:
                    line_no = code[: m.start()].count("\n") + 1
                    violations.append(
                        Violation(
                            id="AI003",
                            rule="ai-over-engineering",
                            category="ai_quality",
                            severity="warning",
                            message=(
                                f"Class '{class_name}' has {len(params)} generic type"
                                " parameters. Consider simplifying the type hierarchy."
                            ),
                            line=line_no,
                            suggestion="Reduce type parameters to 3-4 or use type aliases",
                        )
                    )

        # 3. Factory functions that return only 1 type
        for m in self._RE_FACTORY_FUNC.finditer(code):
            if is_in_skip_region(m.start(), context.skip_regions):
                continue
            func_start = m.start()
            line_no = code[:func_start].count("\n") + 1
            # Find function body (until next def/class at same indent or end)
            func_line = code[func_start:].split("\n")[0]
            indent = len(func_line) - len(func_line.lstrip())
            body_lines: list[str] = []
            for line in code[func_start:].split("\n")[1:]:
                if line.strip() == "":
                    body_lines.append(line)
                    continue
                line_indent = len(line) - len(line.lstrip())
                if line_indent <= indent and line.strip():
                    break
                body_lines.append(line)
            body = "\n".join(body_lines)
            # Count distinct return types/classes
            returns = re.findall(r"return\s+(\w+)\s*\(", body)
            unique_returns = set(returns)
            if len(returns) >= 2 and len(unique_returns) == 1:
                violations.append(
                    Violation(
                        id="AI003",
                        rule="ai-over-engineering",
                        category="ai_quality",
                        severity="warning",
                        message=(
                            f"Factory function at line {line_no} always returns the same"
                            f" type '{returns[0]}'. A factory may be unnecessary."
                        ),
                        line=line_no,
                        suggestion="Use a direct constructor call instead of a factory",
                    )
                )

        return violations
