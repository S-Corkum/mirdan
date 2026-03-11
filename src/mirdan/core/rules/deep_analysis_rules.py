"""DEEP analysis rules: DEEP001, DEEP004.

Rules that use AST analysis to detect hard-to-catch error handling
patterns that cause production failures. All rules are Python-specific
and run at FULL tier.
"""

from __future__ import annotations

import ast

from mirdan.core.rules.base import BaseRule, RuleContext, RuleTier
from mirdan.models import Violation


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    """Return body with an optional leading docstring removed."""
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _is_empty_body(body: list[ast.stmt]) -> bool:
    """Check if a handler body is effectively empty (pass, return, return None, Ellipsis).

    Returns True only if the body (after docstring stripping) contains
    a single statement that is: pass, return, return None, or Ellipsis (...).
    Does NOT flag bodies containing any call expression (logging, raise, etc.).
    """
    stripped = _strip_docstring(body)
    if not stripped:
        return True
    if len(stripped) != 1:
        return False
    stmt = stripped[0]
    # pass
    if isinstance(stmt, ast.Pass):
        return True
    # ...
    if (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is ...
    ):
        return True
    # return / return None
    if isinstance(stmt, ast.Return):
        if stmt.value is None:
            return True
        if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
            return True
    return False


class DEEP001SwallowedExceptionRule(BaseRule):
    """Detect exception handlers that silently swallow exceptions.

    Flags except blocks whose body (after docstring stripping) is only:
    pass, return, return None, or Ellipsis (...).

    Does NOT flag if the body contains any call expression (logging, etc.),
    raise statement, or if contextlib.suppress is used instead.
    """

    @property
    def id(self) -> str:
        return "DEEP001"

    @property
    def name(self) -> str:
        return "swallowed-exception"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "auto"})

    @property
    def tier(self) -> RuleTier:
        return RuleTier.FULL

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        if language not in ("python", "auto"):
            return []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        violations: list[Violation] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if _is_empty_body(node.body):
                violations.append(
                    Violation(
                        id="DEEP001",
                        rule="swallowed-exception",
                        category="deep_analysis",
                        severity="warning",
                        message=(
                            "Exception handler silently swallows the error — "
                            "failures will be invisible."
                        ),
                        line=node.lineno,
                        suggestion=(
                            "At minimum, log the exception. If truly ignorable, "
                            "use contextlib.suppress() to document the intent."
                        ),
                    )
                )
        return violations


class DEEP004LostExceptionContextRule(BaseRule):
    """Detect re-raised exceptions that lose the original traceback.

    Flags except handlers that:
    1. Bind the exception (``as e``)
    2. Contain a Raise node with a different exception type
    3. The Raise node has ``cause=None`` (no ``from`` clause)

    Does NOT flag:
    - Bare ``raise`` (re-raise preserves context)
    - ``raise e`` (explicit re-raise of same exception)
    - ``raise ... from None`` (explicit suppression)
    - Handlers without ``as`` binding
    """

    @property
    def id(self) -> str:
        return "DEEP004"

    @property
    def name(self) -> str:
        return "lost-exception-context"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "auto"})

    @property
    def tier(self) -> RuleTier:
        return RuleTier.FULL

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        if language not in ("python", "auto"):
            return []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        violations: list[Violation] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            # Must bind the exception (as e)
            if not node.name:
                continue

            for child in ast.walk(node):
                if not isinstance(child, ast.Raise):
                    continue
                # Bare raise (re-raise) — preserves context
                if child.exc is None:
                    continue
                # raise e (re-raise same exception) — preserves context
                if isinstance(child.exc, ast.Name) and child.exc.id == node.name:
                    continue
                # raise ... from None — explicit suppression, intentional
                if isinstance(child.cause, ast.Constant) and child.cause.value is None:
                    continue
                # raise ... from <something> — has from clause, preserves context
                if child.cause is not None:
                    continue
                # No from clause — lost context
                violations.append(
                    Violation(
                        id="DEEP004",
                        rule="lost-exception-context",
                        category="deep_analysis",
                        severity="warning",
                        message=(
                            "Exception re-raised without `from` clause — "
                            "original traceback is lost."
                        ),
                        line=child.lineno,
                        suggestion=(
                            "Add `from` clause to preserve diagnostic chain: "
                            f"`raise ... from {node.name}`"
                        ),
                    )
                )
        return violations
