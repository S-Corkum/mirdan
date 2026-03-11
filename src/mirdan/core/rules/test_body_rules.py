"""TEST body rules: TEST001, TEST002, TEST003, TEST005, TEST010.

Rules that analyze individual test function bodies for AI-generated
test anti-patterns. All rules return [] when context.is_test is False.
"""

from __future__ import annotations

import ast
import re

from mirdan.core.rules.base import BaseRule, RuleContext, RuleTier
from mirdan.models import Violation


def _extract_test_functions(
    code: str,
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, int]]:
    """Extract test functions from code via AST.

    Args:
        code: Python source code.

    Returns:
        List of (name, node, lineno) tuples for functions whose name starts
        with ``test_``. Returns an empty list on SyntaxError.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    return [
        (node.name, node, node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]


def _strip_docstring(
    body: list[ast.stmt],
) -> list[ast.stmt]:
    """Return body with an optional leading docstring removed."""
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


class TEST001EmptyTestRule(BaseRule):
    """Detect test functions with empty bodies (pass, ..., docstring-only)."""

    @property
    def id(self) -> str:
        return "TEST001"

    @property
    def name(self) -> str:
        return "empty-test-body"

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
        for name, node, lineno in _extract_test_functions(code):
            body = _strip_docstring(node.body)
            if not body:
                # Docstring-only
                violations.append(self._violation(name, lineno))
                continue
            if len(body) == 1:
                stmt = body[0]
                if isinstance(stmt, ast.Pass) or (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and stmt.value.value is ...
                ):
                    violations.append(self._violation(name, lineno))
        return violations

    def _violation(self, name: str, lineno: int) -> Violation:
        return Violation(
            id="TEST001",
            rule="empty-test-body",
            category="test_quality",
            severity="error",
            message=f"Test function '{name}' has an empty body — it verifies nothing.",
            line=lineno,
            suggestion="Add assertions that verify the expected behavior",
        )


class TEST002AssertTrueRule(BaseRule):
    """Detect tests whose only assertion is ``assert True`` or ``assert 1``."""

    _RE_ASSERT_TRUE_UNITTEST = re.compile(r"self\.assertTrue\s*\(\s*True\s*\)")

    @property
    def id(self) -> str:
        return "TEST002"

    @property
    def name(self) -> str:
        return "assert-true-only"

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
        for name, node, lineno in _extract_test_functions(code):
            body = _strip_docstring(node.body)
            if len(body) != 1:
                continue
            stmt = body[0]
            # assert True / assert 1
            if (
                isinstance(stmt, ast.Assert)
                and isinstance(stmt.test, ast.Constant)
                and (stmt.test.value is True or stmt.test.value == 1)
            ):
                violations.append(self._violation(name, lineno))
                continue
            # self.assertTrue(True)
            if isinstance(stmt, ast.Expr):
                line_text = ast.get_source_segment(code, stmt) or ""
                if self._RE_ASSERT_TRUE_UNITTEST.search(line_text):
                    violations.append(self._violation(name, lineno))
        return violations

    def _violation(self, name: str, lineno: int) -> Violation:
        return Violation(
            id="TEST002",
            rule="assert-true-only",
            category="test_quality",
            severity="error",
            message=(
                f"Test function '{name}' only asserts True — "
                "this is a placeholder that tests nothing."
            ),
            line=lineno,
            suggestion="Replace with assertions on actual return values or behavior",
        )


class TEST003NoAssertionsRule(BaseRule):
    """Detect test functions with no assertion statements at all."""

    _ASSERT_CALL_NAMES = frozenset(
        {
            "assert_called",
            "assert_called_with",
            "assert_called_once",
            "assert_called_once_with",
            "assert_any_call",
            "assert_has_calls",
            "assert_not_called",
            "assertEqual",
            "assertNotEqual",
            "assertTrue",
            "assertFalse",
            "assertIs",
            "assertIsNot",
            "assertIsNone",
            "assertIsNotNone",
            "assertIn",
            "assertNotIn",
            "assertIsInstance",
            "assertNotIsInstance",
            "assertRaises",
            "assertWarns",
            "assertAlmostEqual",
            "assertNotAlmostEqual",
            "assertGreater",
            "assertGreaterEqual",
            "assertLess",
            "assertLessEqual",
            "assertRegex",
            "assertNotRegex",
            "assertCountEqual",
        }
    )

    @property
    def id(self) -> str:
        return "TEST003"

    @property
    def name(self) -> str:
        return "no-assertions"

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
        for name, node, lineno in _extract_test_functions(code):
            if self._has_assertion(node):
                continue
            violations.append(
                Violation(
                    id="TEST003",
                    rule="no-assertions",
                    category="test_quality",
                    severity="warning",
                    message=(
                        f"Test function '{name}' has no assert statements, "
                        "pytest.raises, or mock assertions."
                    ),
                    line=lineno,
                    suggestion="Add assertions to verify expected outcomes",
                )
            )
        return violations

    def _has_assertion(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                return True
            if isinstance(child, ast.Call):
                func = child.func
                # pytest.raises / pytest.warns / pytest.approx
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "pytest"
                    and func.attr in ("raises", "warns", "approx")
                ):
                    return True
                # self.assert* or mock.assert_*
                if isinstance(func, ast.Attribute):
                    if func.attr in self._ASSERT_CALL_NAMES:
                        return True
                    if func.attr.startswith("assert"):
                        return True
        return False


class TEST005MockAbuseRule(BaseRule):
    """Detect test functions with excessive mocking."""

    _RE_PATCH_DECORATOR = re.compile(r"@(?:mock\.)?patch")

    @property
    def id(self) -> str:
        return "TEST005"

    @property
    def name(self) -> str:
        return "mock-everything"

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
        for name, node, lineno in _extract_test_functions(code):
            # Count @patch decorators on this function
            patch_count = sum(1 for d in node.decorator_list if self._is_patch_decorator(d))
            if patch_count >= 4:
                violations.append(
                    Violation(
                        id="TEST005",
                        rule="mock-everything",
                        category="test_quality",
                        severity="warning",
                        message=(
                            f"Test function '{name}' has {patch_count} mock patches — "
                            "likely testing mocks, not code."
                        ),
                        line=lineno,
                        suggestion=(
                            "Reduce mocks to external I/O boundaries only. "
                            "If everything is mocked, nothing is tested."
                        ),
                    )
                )
        return violations

    @staticmethod
    def _is_patch_decorator(decorator: ast.expr) -> bool:
        """Check if a decorator is @patch or @mock.patch."""
        if isinstance(decorator, ast.Call):
            decorator = decorator.func
        if isinstance(decorator, ast.Attribute) and decorator.attr == "patch":
            return True
        return isinstance(decorator, ast.Name) and decorator.id == "patch"


class TEST010BroadExceptionRule(BaseRule):
    """Detect pytest.raises(Exception) or assertRaises(Exception)."""

    _RE_PYTEST_RAISES_EXCEPTION = re.compile(r"pytest\.raises\s*\(\s*Exception\s*\)")
    _RE_ASSERT_RAISES_EXCEPTION = re.compile(r"assertRaises\s*\(\s*Exception\s*\)")

    @property
    def id(self) -> str:
        return "TEST010"

    @property
    def name(self) -> str:
        return "broad-exception-test"

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
        for pattern in (self._RE_PYTEST_RAISES_EXCEPTION, self._RE_ASSERT_RAISES_EXCEPTION):
            for m in pattern.finditer(code):
                line_no = code[: m.start()].count("\n") + 1
                violations.append(
                    Violation(
                        id="TEST010",
                        rule="broad-exception-test",
                        category="test_quality",
                        severity="warning",
                        message=(
                            "Testing for base Exception is too broad — "
                            "use a specific exception type."
                        ),
                        line=line_no,
                        suggestion="Use the specific exception type: pytest.raises(ValueError)",
                    )
                )
        return violations
