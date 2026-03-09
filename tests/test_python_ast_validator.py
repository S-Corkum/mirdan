"""Tests for mirdan.core.python_ast_validator."""

from __future__ import annotations

import textwrap

import pytest

from mirdan.core.code_validator import CodeValidator
from mirdan.core.python_ast_validator import validate_python_ast

# ---------------------------------------------------------------------------
# Unit tests: validate_python_ast standalone
# ---------------------------------------------------------------------------


class TestPY001Eval:
    """PY001-ast: eval() detection via AST."""

    def test_eval_call_detected(self) -> None:
        code = "result = eval(user_input)"
        violations, _ids, parsed = validate_python_ast(code)
        assert parsed is True
        py001 = [v for v in violations if v.id == "PY001"]
        assert len(py001) == 1
        assert py001[0].line == 1

    def test_eval_in_string_not_detected(self) -> None:
        code = 'msg = "eval(dangerous)"'
        violations, _ids, _parsed = validate_python_ast(code)
        py001 = [v for v in violations if v.id == "PY001"]
        assert len(py001) == 0

    def test_eval_method_on_object_not_detected(self) -> None:
        code = "df.eval('col > 0')"
        violations, _ids, _parsed = validate_python_ast(code)
        py001 = [v for v in violations if v.id == "PY001"]
        assert len(py001) == 0


class TestPY002Exec:
    """PY002-ast: exec() detection via AST."""

    def test_exec_call_detected(self) -> None:
        code = "exec(user_code)"
        violations, _ids, _parsed = validate_python_ast(code)
        py002 = [v for v in violations if v.id == "PY002"]
        assert len(py002) == 1

    def test_exec_in_string_not_detected(self) -> None:
        code = 'msg = "exec(dangerous)"'
        violations, _ids, _parsed = validate_python_ast(code)
        py002 = [v for v in violations if v.id == "PY002"]
        assert len(py002) == 0


class TestPY003BareExcept:
    """PY003-ast: bare except detection via AST."""

    def test_bare_except_detected(self) -> None:
        code = textwrap.dedent("""\
            try:
                pass
            except:
                pass
        """)
        violations, _ids, _parsed = validate_python_ast(code)
        py003 = [v for v in violations if v.id == "PY003"]
        assert len(py003) == 1

    def test_typed_except_not_detected(self) -> None:
        code = textwrap.dedent("""\
            try:
                pass
            except Exception:
                pass
        """)
        violations, _ids, _parsed = validate_python_ast(code)
        py003 = [v for v in violations if v.id == "PY003"]
        assert len(py003) == 0


class TestPY004MutableDefault:
    """PY004-ast: mutable default argument detection via AST."""

    def test_list_default_detected(self) -> None:
        code = "def f(x=[]):\n    pass"
        violations, _ids, _parsed = validate_python_ast(code)
        py004 = [v for v in violations if v.id == "PY004"]
        assert len(py004) == 1

    def test_dict_default_detected(self) -> None:
        code = "def f(x={}):\n    pass"
        violations, _ids, _parsed = validate_python_ast(code)
        py004 = [v for v in violations if v.id == "PY004"]
        assert len(py004) == 1

    def test_set_default_detected(self) -> None:
        code = "def f(x={1, 2}):\n    pass"
        violations, _ids, _parsed = validate_python_ast(code)
        py004 = [v for v in violations if v.id == "PY004"]
        assert len(py004) == 1

    def test_none_default_not_detected(self) -> None:
        code = "def f(x=None):\n    pass"
        violations, _ids, _parsed = validate_python_ast(code)
        py004 = [v for v in violations if v.id == "PY004"]
        assert len(py004) == 0

    def test_kwonly_mutable_default_detected(self) -> None:
        code = "def f(*, x=[]):\n    pass"
        violations, _ids, _parsed = validate_python_ast(code)
        py004 = [v for v in violations if v.id == "PY004"]
        assert len(py004) == 1


class TestPY014DeadImport:
    """PY014: unused import detection via AST."""

    def test_unused_import_detected(self) -> None:
        code = "import os\nx = 1"
        violations, _ids, _parsed = validate_python_ast(code)
        py014 = [v for v in violations if v.id == "PY014"]
        assert len(py014) == 1
        assert "os" in py014[0].message

    def test_used_import_not_detected(self) -> None:
        code = "import os\nos.path.exists('.')"
        violations, _ids, _parsed = validate_python_ast(code)
        py014 = [v for v in violations if v.id == "PY014"]
        assert len(py014) == 0

    def test_import_in_all_not_detected(self) -> None:
        code = textwrap.dedent("""\
            from mymodule import MyClass
            __all__ = ["MyClass"]
        """)
        violations, _ids, _parsed = validate_python_ast(code)
        py014 = [v for v in violations if v.id == "PY014"]
        assert len(py014) == 0

    def test_import_in_type_checking_not_detected(self) -> None:
        code = textwrap.dedent("""\
            from typing import TYPE_CHECKING
            if TYPE_CHECKING:
                import os
            x = 1
        """)
        violations, _ids, _parsed = validate_python_ast(code)
        py014 = [v for v in violations if v.id == "PY014"]
        # TYPE_CHECKING import should not be flagged
        assert not any(v for v in py014 if "os" in v.message)

    def test_import_in_skip_lines_not_detected(self) -> None:
        code = "import fake_module\nx = 1"
        violations, _ids, _parsed = validate_python_ast(code, skip_lines={1})
        py014 = [v for v in violations if v.id == "PY014"]
        assert len(py014) == 0

    def test_underscore_import_not_flagged(self) -> None:
        code = "from gettext import gettext as _\nx = 1"
        violations, _ids, _parsed = validate_python_ast(code)
        py014 = [v for v in violations if v.id == "PY014"]
        assert len(py014) == 0

    def test_from_import_used_as_attribute(self) -> None:
        code = textwrap.dedent("""\
            from pathlib import Path
            p = Path(".")
        """)
        violations, _ids, _parsed = validate_python_ast(code)
        py014 = [v for v in violations if v.id == "PY014"]
        assert len(py014) == 0

    def test_aliased_import_used(self) -> None:
        code = textwrap.dedent("""\
            import numpy as np
            x = np.array([1])
        """)
        violations, _ids, _parsed = validate_python_ast(code)
        py014 = [v for v in violations if v.id == "PY014"]
        assert len(py014) == 0

    def test_aliased_import_unused(self) -> None:
        code = textwrap.dedent("""\
            import numpy as np
            x = 1
        """)
        violations, _ids, _parsed = validate_python_ast(code)
        py014 = [v for v in violations if v.id == "PY014"]
        assert len(py014) == 1
        assert "np" in py014[0].message


class TestPY015UnreachableCode:
    """PY015: unreachable code detection via AST."""

    def test_code_after_return_detected(self) -> None:
        code = textwrap.dedent("""\
            def f():
                return 1
                x = 2
        """)
        violations, _ids, _parsed = validate_python_ast(code)
        py015 = [v for v in violations if v.id == "PY015"]
        assert len(py015) == 1
        assert py015[0].line == 3

    def test_code_after_raise_detected(self) -> None:
        code = textwrap.dedent("""\
            def f():
                raise ValueError("bad")
                x = 2
        """)
        violations, _ids, _parsed = validate_python_ast(code)
        py015 = [v for v in violations if v.id == "PY015"]
        assert len(py015) == 1

    def test_code_after_return_in_if_else_not_detected(self) -> None:
        """Code after if/else with returns is reachable if not all branches return."""
        code = textwrap.dedent("""\
            def f(x):
                if x:
                    return 1
                y = 2
                return y
        """)
        violations, _ids, _parsed = validate_python_ast(code)
        py015 = [v for v in violations if v.id == "PY015"]
        assert len(py015) == 0

    def test_code_after_break_in_loop_detected(self) -> None:
        code = textwrap.dedent("""\
            def f():
                for i in range(10):
                    break
                    x = 1
        """)
        violations, _ids, _parsed = validate_python_ast(code)
        py015 = [v for v in violations if v.id == "PY015"]
        assert len(py015) == 1

    def test_finally_block_not_checked(self) -> None:
        """Code in finally blocks is always reachable."""
        code = textwrap.dedent("""\
            def f():
                try:
                    return 1
                finally:
                    cleanup()
        """)
        violations, _ids, _parsed = validate_python_ast(code)
        py015 = [v for v in violations if v.id == "PY015"]
        assert len(py015) == 0


class TestSyntaxError:
    """Behavior on unparseable code."""

    def test_syntax_error_returns_empty(self) -> None:
        violations, ids, parsed = validate_python_ast("def f(")
        assert violations == []
        assert ids == set()
        assert parsed is False

    def test_none_input_returns_empty(self) -> None:
        violations, ids, parsed = validate_python_ast(None)  # type: ignore[arg-type]
        assert violations == []
        assert ids == set()
        assert parsed is False


class TestCheckedRuleIds:
    """checked_rule_ids is correct on success."""

    def test_checked_ids_on_success(self) -> None:
        _violations, ids, parsed = validate_python_ast("x = 1")
        assert parsed is True
        assert ids == {"PY001", "PY002", "PY003", "PY004"}


# ---------------------------------------------------------------------------
# Integration tests: through CodeValidator.validate()
# ---------------------------------------------------------------------------


class TestCodeValidatorIntegration:
    """Test AST checks integrated into CodeValidator.validate()."""

    @pytest.fixture()
    def validator(self) -> CodeValidator:
        from mirdan.core.quality_standards import QualityStandards

        return CodeValidator(QualityStandards())

    def test_eval_produces_ast_violation(self, validator: CodeValidator) -> None:
        code = textwrap.dedent("""\
            def main():
                result = eval(user_input)
        """)
        result = validator.validate(code, language="python")
        py001 = [v for v in result.violations if v.id == "PY001"]
        assert len(py001) >= 1
        # AST violations have verifiable=True (default)
        assert py001[0].verifiable is True

    def test_exec_produces_ast_violation(self, validator: CodeValidator) -> None:
        code = textwrap.dedent("""\
            def main():
                exec(user_code)
        """)
        result = validator.validate(code, language="python")
        py002 = [v for v in result.violations if v.id == "PY002"]
        assert len(py002) >= 1

    def test_eval_in_string_no_false_positive(self, validator: CodeValidator) -> None:
        """Regression: regex would flag eval() inside strings, AST should not."""
        code = textwrap.dedent("""\
            def main():
                msg = "Use eval(x) to evaluate"
                return msg
        """)
        result = validator.validate(code, language="python")
        py001 = [v for v in result.violations if v.id == "PY001"]
        assert len(py001) == 0

    def test_unused_import_produces_py014(self, validator: CodeValidator) -> None:
        code = textwrap.dedent("""\
            import json
            def main():
                return "hello"
        """)
        result = validator.validate(code, language="python")
        py014 = [v for v in result.violations if v.id == "PY014"]
        assert len(py014) == 1

    def test_syntax_error_falls_back_to_regex(self, validator: CodeValidator) -> None:
        """When AST fails, regex rules still fire."""
        code = "eval(x)\ndef incomplete("
        result = validator.validate(code, language="python")
        # AST can't parse this, so regex PY001 should still fire
        py001 = [v for v in result.violations if v.id == "PY001"]
        assert len(py001) >= 1

    def test_python_ast_in_standards_checked(self, validator: CodeValidator) -> None:
        code = textwrap.dedent("""\
            def main():
                eval(x)
        """)
        result = validator.validate(code, language="python")
        assert "python_ast" in result.standards_checked

    def test_non_python_no_ast_check(self, validator: CodeValidator) -> None:
        code = "function f() { eval('x'); }"
        result = validator.validate(code, language="javascript")
        assert "python_ast" not in result.standards_checked
