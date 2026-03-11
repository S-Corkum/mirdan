"""TEST structure rules: TEST004, TEST006, TEST007, TEST008, TEST009.

Rules that analyze test file structure, naming, cross-references, and
isolation. All rules return [] when context.is_test is False.
"""

from __future__ import annotations

import ast
import re

from mirdan.core.rules.base import BaseRule, RuleContext, RuleTier
from mirdan.core.rules.test_body_rules import _extract_test_functions, _strip_docstring
from mirdan.models import Violation


class TEST004NoCoverageRule(BaseRule):
    """Detect tests that don't exercise any code from the implementation."""

    @property
    def id(self) -> str:
        return "TEST004"

    @property
    def name(self) -> str:
        return "no-code-under-test"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "auto"})

    @property
    def tier(self) -> RuleTier:
        return RuleTier.FULL

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        if not context.is_test:
            return []
        if language not in ("python", "auto"):
            return []
        if not context.implementation_code:
            return []

        # Extract public names from implementation
        impl_names = self._extract_public_names(context.implementation_code)
        if not impl_names:
            return []

        # Extract all referenced names in test function bodies
        test_funcs = _extract_test_functions(code)
        if not test_funcs:
            return []

        test_refs = self._extract_references(code, test_funcs)

        # Check if any implementation name is referenced
        if impl_names & test_refs:
            return []

        return [
            Violation(
                id="TEST004",
                rule="no-code-under-test",
                category="test_quality",
                severity="warning",
                message=(
                    "Tests do not appear to call any functions from the implementation under test."
                ),
                line=1,
                suggestion="Import and call the functions/classes being tested",
            )
        ]

    @staticmethod
    def _extract_public_names(code: str) -> set[str]:
        """Extract public function/class names from implementation code."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()

        names: set[str] = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ) and not node.name.startswith("_"):
                names.add(node.name)
        return names

    @staticmethod
    def _extract_references(
        code: str,
        test_funcs: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, int]],
    ) -> set[str]:
        """Extract all Name and Attribute references in test function bodies."""
        refs: set[str] = set()
        for _, node, _ in test_funcs:
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    refs.add(child.id)
                elif isinstance(child, ast.Attribute):
                    refs.add(child.attr)
        return refs


class TEST006DuplicateTestRule(BaseRule):
    """Detect test functions with identical AST structure."""

    @property
    def id(self) -> str:
        return "TEST006"

    @property
    def name(self) -> str:
        return "duplicate-test-logic"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "auto"})

    @property
    def tier(self) -> RuleTier:
        return RuleTier.FULL

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        if not context.is_test:
            return []
        if language not in ("python", "auto"):
            return []

        test_funcs = _extract_test_functions(code)
        if len(test_funcs) < 2:
            return []

        # Build AST dumps keyed by normalized structure
        dumps: dict[str, list[tuple[str, int]]] = {}
        for name, node, lineno in test_funcs:
            body = _strip_docstring(node.body)
            if not body:
                continue
            try:
                dump = ast.dump(ast.Module(body=body, type_ignores=[]))
            except Exception:  # noqa: S112
                continue
            dumps.setdefault(dump, []).append((name, lineno))

        violations: list[Violation] = []
        for entries in dumps.values():
            if len(entries) < 2:
                continue
            # Flag all duplicates after the first
            for name, lineno in entries[1:]:
                violations.append(
                    Violation(
                        id="TEST006",
                        rule="duplicate-test-logic",
                        category="test_quality",
                        severity="info",
                        message=(
                            f"Test function '{name}' has identical logic to "
                            f"'{entries[0][0]}'. Consider using parametrize."
                        ),
                        line=lineno,
                        suggestion=(
                            "Use @pytest.mark.parametrize or refactor shared setup into fixtures"
                        ),
                    )
                )
        return violations


class TEST007MissingEdgeCaseRule(BaseRule):
    """Detect test files that lack edge case tests."""

    _EDGE_CASE_INDICATORS = frozenset(
        {
            "empty",
            "none",
            "null",
            "zero",
            "negative",
            "boundary",
            "invalid",
            "error",
            "fail",
            "edge",
            "corner",
            "overflow",
            "large",
            "missing",
        }
    )

    @property
    def id(self) -> str:
        return "TEST007"

    @property
    def name(self) -> str:
        return "missing-edge-cases"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "auto"})

    @property
    def tier(self) -> RuleTier:
        return RuleTier.ESSENTIAL

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        if not context.is_test:
            return []
        if language not in ("python", "auto"):
            return []

        test_funcs = _extract_test_functions(code)
        if len(test_funcs) < 3:
            return []

        # Check if any test name contains an edge case indicator
        for name, _, _ in test_funcs:
            name_lower = name.lower()
            for indicator in self._EDGE_CASE_INDICATORS:
                if indicator in name_lower:
                    return []

        return [
            Violation(
                id="TEST007",
                rule="missing-edge-cases",
                category="test_quality",
                severity="info",
                message=(
                    f"Test file has {len(test_funcs)} tests but none appear "
                    "to cover edge cases (empty, null, error, boundary)."
                ),
                line=1,
                suggestion=("Add tests for empty inputs, None values, error paths, and boundaries"),
            )
        ]


class TEST008HardcodedDataRule(BaseRule):
    """Detect unexplained magic values in assertions."""

    _RE_ASSERT_MAGIC_NUMBER = re.compile(r"assert\w*\s*[^#\n]*\b(\d{2,})\b")
    _RE_COMMENT = re.compile(r"#.*$", re.MULTILINE)

    @property
    def id(self) -> str:
        return "TEST008"

    @property
    def name(self) -> str:
        return "unexplained-magic-values"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "auto"})

    @property
    def tier(self) -> RuleTier:
        return RuleTier.ESSENTIAL

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        if not context.is_test:
            return []
        if language not in ("python", "auto"):
            return []

        violations: list[Violation] = []
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped.startswith("assert"):
                continue

            m = self._RE_ASSERT_MAGIC_NUMBER.search(stripped)
            if not m:
                continue

            # Check if current or previous line has a comment
            has_comment = self._RE_COMMENT.search(stripped) is not None
            if not has_comment and i > 1:
                prev = lines[i - 2].strip()
                has_comment = prev.startswith("#")

            if not has_comment:
                violations.append(
                    Violation(
                        id="TEST008",
                        rule="unexplained-magic-values",
                        category="test_quality",
                        severity="info",
                        message=(f"Assertion contains unexplained numeric literal '{m.group(1)}'"),
                        line=i,
                        suggestion=(
                            "Add a comment or use a named constant to explain the expected value"
                        ),
                    )
                )
        return violations


class TEST009ExecutionOrderRule(BaseRule):
    """Detect tests that modify global or module-level state."""

    @property
    def id(self) -> str:
        return "TEST009"

    @property
    def name(self) -> str:
        return "test-execution-order"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "auto"})

    @property
    def tier(self) -> RuleTier:
        return RuleTier.FULL

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        if not context.is_test:
            return []
        if language not in ("python", "auto"):
            return []

        violations: list[Violation] = []
        for name, node, _lineno in _extract_test_functions(code):
            for child in ast.walk(node):
                if isinstance(child, (ast.Global, ast.Nonlocal)):
                    violations.append(
                        Violation(
                            id="TEST009",
                            rule="test-execution-order",
                            category="test_quality",
                            severity="warning",
                            message=(
                                f"Test function '{name}' modifies global/module-level "
                                "state, creating execution order dependency."
                            ),
                            line=child.lineno,
                            suggestion=(
                                "Use fixtures for setup/teardown. Avoid global/nonlocal in tests."
                            ),
                        )
                    )
                    break  # One violation per function is enough
        return violations
