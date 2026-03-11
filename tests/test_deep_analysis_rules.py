"""Tests for deep analysis rules (DEEP001, DEEP004)."""

from __future__ import annotations

import pytest

from mirdan.core.rules.base import RuleContext
from mirdan.core.rules.deep_analysis_rules import (
    DEEP001SwallowedExceptionRule,
    DEEP004LostExceptionContextRule,
)


@pytest.fixture()
def context() -> RuleContext:
    """Standard rule context."""
    return RuleContext(skip_regions=[])


# --- DEEP001: Swallowed Exception ---


class TestDEEP001SwallowedException:
    def test_pass_in_except_body(self, context: RuleContext) -> None:
        code = """\
try:
    risky()
except ValueError:
    pass
"""
        rule = DEEP001SwallowedExceptionRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 1
        assert violations[0].id == "DEEP001"
        assert violations[0].line == 3

    def test_return_none_in_except(self, context: RuleContext) -> None:
        code = """\
try:
    risky()
except Exception:
    return None
"""
        rule = DEEP001SwallowedExceptionRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 1
        assert violations[0].id == "DEEP001"

    def test_return_bare_in_except(self, context: RuleContext) -> None:
        code = """\
try:
    risky()
except Exception:
    return
"""
        rule = DEEP001SwallowedExceptionRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 1

    def test_ellipsis_in_except(self, context: RuleContext) -> None:
        code = """\
try:
    risky()
except ValueError:
    ...
"""
        rule = DEEP001SwallowedExceptionRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 1

    def test_logging_in_except_no_violation(self, context: RuleContext) -> None:
        code = """\
try:
    risky()
except ValueError as e:
    logger.error(e)
"""
        rule = DEEP001SwallowedExceptionRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 0

    def test_raise_in_except_no_violation(self, context: RuleContext) -> None:
        code = """\
try:
    risky()
except ValueError:
    raise
"""
        rule = DEEP001SwallowedExceptionRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 0

    def test_contextlib_suppress_no_violation(self, context: RuleContext) -> None:
        # contextlib.suppress uses a with statement, not try/except
        code = """\
from contextlib import suppress
with suppress(KeyError):
    d["missing"]
"""
        rule = DEEP001SwallowedExceptionRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 0

    def test_non_python_skipped(self, context: RuleContext) -> None:
        code = "try { } catch(e) { }"
        rule = DEEP001SwallowedExceptionRule()
        violations = rule.check(code, "javascript", context)
        assert len(violations) == 0

    def test_syntax_error_no_crash(self, context: RuleContext) -> None:
        code = "def foo(\n"
        rule = DEEP001SwallowedExceptionRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 0

    def test_multiple_handlers(self, context: RuleContext) -> None:
        code = """\
try:
    risky()
except ValueError:
    pass
except TypeError:
    logger.error("type error")
except KeyError:
    return
"""
        rule = DEEP001SwallowedExceptionRule()
        violations = rule.check(code, "python", context)
        # ValueError: pass -> violation, TypeError: logging -> no, KeyError: return -> violation
        assert len(violations) == 2

    def test_docstring_then_pass(self, context: RuleContext) -> None:
        code = """\
try:
    risky()
except ValueError:
    \"\"\"Intentionally ignored.\"\"\"
    pass
"""
        rule = DEEP001SwallowedExceptionRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 1

    def test_pass_with_assignment_no_violation(self, context: RuleContext) -> None:
        code = """\
try:
    result = risky()
except ValueError:
    result = None
    pass
"""
        rule = DEEP001SwallowedExceptionRule()
        violations = rule.check(code, "python", context)
        # Body has 2 statements (assignment + pass), so not an "empty" body
        assert len(violations) == 0


# --- DEEP004: Lost Exception Context ---


class TestDEEP004LostExceptionContext:
    def test_raise_different_without_from(self, context: RuleContext) -> None:
        code = """\
try:
    risky()
except ValueError as e:
    raise TypeError("bad type")
"""
        rule = DEEP004LostExceptionContextRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 1
        assert violations[0].id == "DEEP004"
        assert "from" in violations[0].suggestion

    def test_raise_with_from_no_violation(self, context: RuleContext) -> None:
        code = """\
try:
    risky()
except ValueError as e:
    raise TypeError("bad type") from e
"""
        rule = DEEP004LostExceptionContextRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 0

    def test_bare_raise_no_violation(self, context: RuleContext) -> None:
        code = """\
try:
    risky()
except ValueError as e:
    raise
"""
        rule = DEEP004LostExceptionContextRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 0

    def test_raise_same_exception_no_violation(self, context: RuleContext) -> None:
        code = """\
try:
    risky()
except ValueError as e:
    raise e
"""
        rule = DEEP004LostExceptionContextRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 0

    def test_raise_from_none_no_violation(self, context: RuleContext) -> None:
        code = """\
try:
    risky()
except ValueError as e:
    raise TypeError("clean") from None
"""
        rule = DEEP004LostExceptionContextRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 0

    def test_no_binding_no_violation(self, context: RuleContext) -> None:
        code = """\
try:
    risky()
except ValueError:
    raise TypeError("bad type")
"""
        rule = DEEP004LostExceptionContextRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 0

    def test_non_python_skipped(self, context: RuleContext) -> None:
        code = "try { } catch(e) { throw new Error('test'); }"
        rule = DEEP004LostExceptionContextRule()
        violations = rule.check(code, "go", context)
        assert len(violations) == 0

    def test_syntax_error_no_crash(self, context: RuleContext) -> None:
        code = "def foo(\n"
        rule = DEEP004LostExceptionContextRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 0

    def test_raise_from_other_variable(self, context: RuleContext) -> None:
        code = """\
try:
    risky()
except ValueError as e:
    raise TypeError("bad type") from other_error
"""
        rule = DEEP004LostExceptionContextRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 0

    def test_suggestion_includes_exception_name(self, context: RuleContext) -> None:
        code = """\
try:
    risky()
except ValueError as exc:
    raise TypeError("bad type")
"""
        rule = DEEP004LostExceptionContextRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 1
        assert "exc" in violations[0].suggestion
