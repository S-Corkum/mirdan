"""Tests for AI-specific code quality checks (AI001, AI002, AI008)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mirdan.core.ai_quality_checker import PYTHON_STDLIB_MODULES, AIQualityChecker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def checker() -> AIQualityChecker:
    """Checker with no project context (AI002 disabled)."""
    return AIQualityChecker()


@pytest.fixture()
def checker_with_project(tmp_path: Path) -> AIQualityChecker:
    """Checker with a project dir containing pyproject.toml."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "myproject"\ndependencies = [\n'
        '    "fastapi>=0.100.0",\n'
        '    "sqlalchemy>=2.0",\n'
        '    "pydantic>=2.0",\n'
        "]\n"
    )
    # Create a local package so it's recognized
    pkg = tmp_path / "src" / "myproject"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").touch()
    return AIQualityChecker(project_dir=tmp_path)


# ---------------------------------------------------------------------------
# AI001: Placeholder Detection
# ---------------------------------------------------------------------------


class TestAI001Placeholders:
    """Tests for AI001 — placeholder code detection."""

    def test_catches_raise_not_implemented(self, checker: AIQualityChecker) -> None:
        code = "def process(data):\n    raise NotImplementedError\n"
        violations = checker.check(code, "python")
        ai001 = [v for v in violations if v.id == "AI001"]
        assert len(ai001) == 1
        assert "NotImplementedError" in ai001[0].message

    def test_catches_raise_not_implemented_with_message(self, checker: AIQualityChecker) -> None:
        code = 'def process(data):\n    raise NotImplementedError("todo")\n'
        violations = checker.check(code, "python")
        ai001 = [v for v in violations if v.id == "AI001"]
        assert len(ai001) == 1

    def test_catches_pass_with_todo(self, checker: AIQualityChecker) -> None:
        code = "def process(data):\n    # TODO: implement this\n    pass\n"
        violations = checker.check(code, "python")
        ai001 = [v for v in violations if v.id == "AI001"]
        assert len(ai001) == 1
        assert "pass" in ai001[0].message.lower() or "placeholder" in ai001[0].message.lower()

    def test_catches_pass_with_fixme(self, checker: AIQualityChecker) -> None:
        code = "def process(data):\n    pass  # FIXME: implement\n"
        violations = checker.check(code, "python")
        ai001 = [v for v in violations if v.id == "AI001"]
        assert len(ai001) == 1

    def test_catches_ellipsis_with_placeholder_comment(self, checker: AIQualityChecker) -> None:
        code = "def process(data):\n    ...  # placeholder\n"
        violations = checker.check(code, "python")
        ai001 = [v for v in violations if v.id == "AI001"]
        assert len(ai001) == 1
        assert "ellipsis" in ai001[0].message.lower()

    def test_skips_abstract_method(self, checker: AIQualityChecker) -> None:
        code = (
            "from abc import ABC, abstractmethod\n\n"
            "class Base(ABC):\n"
            "    @abstractmethod\n"
            "    def process(self, data):\n"
            "        raise NotImplementedError\n"
        )
        violations = checker.check(code, "python")
        ai001 = [v for v in violations if v.id == "AI001"]
        assert len(ai001) == 0

    def test_skips_non_python(self, checker: AIQualityChecker) -> None:
        code = "function process() { throw new Error('Not implemented'); }\n"
        violations = checker.check(code, "javascript")
        ai001 = [v for v in violations if v.id == "AI001"]
        assert len(ai001) == 0

    def test_pass_without_todo_is_fine(self, checker: AIQualityChecker) -> None:
        """A bare pass without any TODO comment should not trigger AI001."""
        code = "def process(data):\n    pass\n"
        violations = checker.check(code, "python")
        ai001 = [v for v in violations if v.id == "AI001"]
        assert len(ai001) == 0

    def test_not_implemented_in_comment_ignored(self, checker: AIQualityChecker) -> None:
        """NotImplementedError inside a comment should not trigger."""
        code = "# raise NotImplementedError if needed\ndef process(data):\n    return 42\n"
        violations = checker.check(code, "python")
        ai001 = [v for v in violations if v.id == "AI001"]
        assert len(ai001) == 0

    def test_severity_is_error(self, checker: AIQualityChecker) -> None:
        code = "def process(data):\n    raise NotImplementedError\n"
        violations = checker.check(code, "python")
        ai001 = [v for v in violations if v.id == "AI001"]
        assert ai001[0].severity == "error"


# ---------------------------------------------------------------------------
# AI002: Hallucinated Import Detection
# ---------------------------------------------------------------------------


class TestAI002HallucinatedImports:
    """Tests for AI002 — hallucinated import detection."""

    def test_catches_unknown_import(self, checker_with_project: AIQualityChecker) -> None:
        code = "import nonexistent_lib\n"
        violations = checker_with_project.check(code, "python")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert len(ai002) == 1
        assert "nonexistent_lib" in ai002[0].message

    def test_passes_stdlib_import(self, checker_with_project: AIQualityChecker) -> None:
        code = "import os\nimport sys\nimport pathlib\nimport json\n"
        violations = checker_with_project.check(code, "python")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert len(ai002) == 0

    def test_passes_project_dependency(self, checker_with_project: AIQualityChecker) -> None:
        code = "import fastapi\nfrom sqlalchemy import Column\nimport pydantic\n"
        violations = checker_with_project.check(code, "python")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert len(ai002) == 0

    def test_skips_when_no_project_dir(self, checker: AIQualityChecker) -> None:
        """Without project_dir, AI002 should not fire (can't verify)."""
        code = "import nonexistent_lib\n"
        violations = checker.check(code, "python")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert len(ai002) == 0

    def test_skips_relative_imports(self, checker_with_project: AIQualityChecker) -> None:
        code = "from . import utils\nfrom .models import User\n"
        violations = checker_with_project.check(code, "python")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert len(ai002) == 0

    def test_skips_future_imports(self, checker_with_project: AIQualityChecker) -> None:
        code = "from __future__ import annotations\n"
        violations = checker_with_project.check(code, "python")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert len(ai002) == 0

    def test_catches_from_unknown_import(self, checker_with_project: AIQualityChecker) -> None:
        code = "from fake_package import something\n"
        violations = checker_with_project.check(code, "python")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert len(ai002) == 1
        assert "fake_package" in ai002[0].message

    def test_severity_is_warning(self, checker_with_project: AIQualityChecker) -> None:
        code = "import nonexistent_lib\n"
        violations = checker_with_project.check(code, "python")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert ai002[0].severity == "warning"

    def test_passes_local_package(self, checker_with_project: AIQualityChecker) -> None:
        """Should recognize local packages under src/."""
        code = "import myproject\n"
        violations = checker_with_project.check(code, "python")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert len(ai002) == 0

    def test_normalizes_hyphenated_deps(self, tmp_path: Path) -> None:
        """Package names with hyphens should be normalized to underscores."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "test"\ndependencies = ["my-cool-lib>=1.0"]\n'
        )
        checker = AIQualityChecker(project_dir=tmp_path)
        code = "import my_cool_lib\n"
        violations = checker.check(code, "python")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert len(ai002) == 0


# ---------------------------------------------------------------------------
# AI008: Injection Vulnerability
# ---------------------------------------------------------------------------


class TestAI008Injection:
    """Tests for AI008 — injection vulnerability via f-string."""

    def test_catches_fstring_sql(self, checker: AIQualityChecker) -> None:
        code = 'cursor.execute(f"SELECT * FROM users WHERE id={user_id}")\n'
        violations = checker.check(code, "python")
        ai008 = [v for v in violations if v.id == "AI008"]
        assert len(ai008) == 1
        assert "SQL" in ai008[0].message

    def test_catches_fstring_sql_single_quote(self, checker: AIQualityChecker) -> None:
        code = "cursor.execute(f'DELETE FROM users WHERE id={user_id}')\n"
        violations = checker.check(code, "python")
        ai008 = [v for v in violations if v.id == "AI008"]
        assert len(ai008) == 1

    def test_catches_eval_fstring(self, checker: AIQualityChecker) -> None:
        code = 'result = eval(f"{user_input}")\n'
        violations = checker.check(code, "python")
        ai008 = [v for v in violations if v.id == "AI008"]
        assert len(ai008) == 1
        assert "eval" in ai008[0].message

    def test_catches_exec_fstring(self, checker: AIQualityChecker) -> None:
        code = 'exec(f"print({value})")\n'
        violations = checker.check(code, "python")
        ai008 = [v for v in violations if v.id == "AI008"]
        assert len(ai008) == 1

    def test_catches_os_system_fstring(self, checker: AIQualityChecker) -> None:
        code = 'os.system(f"rm {filename}")\n'
        violations = checker.check(code, "python")
        ai008 = [v for v in violations if v.id == "AI008"]
        assert len(ai008) == 1
        assert "os.system" in ai008[0].message

    def test_catches_subprocess_fstring(self, checker: AIQualityChecker) -> None:
        code = 'subprocess.run(f"echo {msg}", shell=True)\n'
        violations = checker.check(code, "python")
        ai008 = [v for v in violations if v.id == "AI008"]
        assert len(ai008) == 1

    def test_passes_parameterized_query(self, checker: AIQualityChecker) -> None:
        code = 'cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))\n'
        violations = checker.check(code, "python")
        ai008 = [v for v in violations if v.id == "AI008"]
        assert len(ai008) == 0

    def test_passes_eval_with_literal(self, checker: AIQualityChecker) -> None:
        code = 'result = eval("1 + 2")\n'
        violations = checker.check(code, "python")
        ai008 = [v for v in violations if v.id == "AI008"]
        assert len(ai008) == 0

    def test_severity_is_error(self, checker: AIQualityChecker) -> None:
        code = 'cursor.execute(f"SELECT * FROM users WHERE id={user_id}")\n'
        violations = checker.check(code, "python")
        ai008 = [v for v in violations if v.id == "AI008"]
        assert ai008[0].severity == "error"

    def test_skips_non_python(self, checker: AIQualityChecker) -> None:
        code = 'const query = `SELECT * FROM users WHERE id=${userId}`;\n'
        violations = checker.check(code, "javascript")
        ai008 = [v for v in violations if v.id == "AI008"]
        assert len(ai008) == 0


# ---------------------------------------------------------------------------
# Integration: AI rules run within validate() and validate_quick()
# ---------------------------------------------------------------------------


class TestIntegrationWithCodeValidator:
    """Tests that AI rules fire within the CodeValidator pipeline."""

    def test_ai001_in_full_validate(self) -> None:
        from mirdan.core.code_validator import CodeValidator
        from mirdan.core.quality_standards import QualityStandards

        cv = CodeValidator(QualityStandards())
        code = "def process(data):\n    raise NotImplementedError\n"
        result = cv.validate(code, language="python")
        ai001 = [v for v in result.violations if v.id == "AI001"]
        assert len(ai001) == 1

    def test_ai008_in_full_validate(self) -> None:
        from mirdan.core.code_validator import CodeValidator
        from mirdan.core.quality_standards import QualityStandards

        cv = CodeValidator(QualityStandards())
        code = 'cursor.execute(f"SELECT * FROM users WHERE id={uid}")\n'
        result = cv.validate(code, language="python")
        ai008 = [v for v in result.violations if v.id == "AI008"]
        assert len(ai008) == 1

    def test_ai001_in_validate_quick(self) -> None:
        from mirdan.core.code_validator import CodeValidator
        from mirdan.core.quality_standards import QualityStandards

        cv = CodeValidator(QualityStandards())
        code = "def process(data):\n    raise NotImplementedError\n"
        result = cv.validate_quick(code, language="python")
        ai001 = [v for v in result.violations if v.id == "AI001"]
        assert len(ai001) == 1

    def test_ai008_in_validate_quick(self) -> None:
        from mirdan.core.code_validator import CodeValidator
        from mirdan.core.quality_standards import QualityStandards

        cv = CodeValidator(QualityStandards())
        code = 'cursor.execute(f"SELECT * FROM users WHERE id={uid}")\n'
        result = cv.validate_quick(code, language="python")
        ai008 = [v for v in result.violations if v.id == "AI008"]
        assert len(ai008) == 1

    def test_ai002_not_in_validate_quick(self) -> None:
        """AI002 should NOT run in quick mode (it's not critical)."""
        from mirdan.core.code_validator import CodeValidator
        from mirdan.core.quality_standards import QualityStandards

        cv = CodeValidator(QualityStandards())
        code = "import nonexistent_lib\n"
        result = cv.validate_quick(code, language="python")
        ai002 = [v for v in result.violations if v.id == "AI002"]
        assert len(ai002) == 0

    def test_ai_quality_in_standards_checked(self) -> None:
        from mirdan.core.code_validator import CodeValidator
        from mirdan.core.quality_standards import QualityStandards

        cv = CodeValidator(QualityStandards())
        code = "def process(data):\n    raise NotImplementedError\n"
        result = cv.validate(code, language="python")
        assert "ai_quality" in result.standards_checked


# ---------------------------------------------------------------------------
# False positive tests
# ---------------------------------------------------------------------------


class TestFalsePositives:
    """Clean code should produce zero AI violations."""

    def test_clean_python_code(self, checker: AIQualityChecker) -> None:
        code = (
            "import os\n"
            "import json\n\n"
            "def process(data: dict) -> str:\n"
            '    return json.dumps(data)\n\n'
            "def main() -> None:\n"
            '    result = process({"key": "value"})\n'
            "    print(result)\n"
        )
        violations = checker.check(code, "python")
        assert len(violations) == 0

    def test_parameterized_sql_is_fine(self, checker: AIQualityChecker) -> None:
        code = (
            "import sqlite3\n\n"
            "def get_user(db, user_id: int):\n"
            '    cursor = db.execute("SELECT * FROM users WHERE id = ?", (user_id,))\n'
            "    return cursor.fetchone()\n"
        )
        violations = checker.check(code, "python")
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# PYTHON_STDLIB_MODULES constant
# ---------------------------------------------------------------------------


class TestStdlibModules:
    """Verify PYTHON_STDLIB_MODULES contains expected modules."""

    @pytest.mark.parametrize(
        "module",
        ["os", "sys", "pathlib", "json", "typing", "collections", "functools",
         "dataclasses", "logging", "re", "io", "math", "datetime", "hashlib",
         "unittest", "asyncio", "contextlib", "abc", "enum", "itertools"],
    )
    def test_common_stdlib_modules_present(self, module: str) -> None:
        assert module in PYTHON_STDLIB_MODULES

    def test_typing_extensions_included(self) -> None:
        """typing_extensions is so ubiquitous it should be in the skip list."""
        assert "typing_extensions" in PYTHON_STDLIB_MODULES


# ---------------------------------------------------------------------------
# check_quick only runs AI001 + AI008
# ---------------------------------------------------------------------------


class TestCheckQuick:
    """Verify check_quick runs only AI001 and AI008."""

    def test_quick_catches_ai001(self, checker: AIQualityChecker) -> None:
        code = "def process(data):\n    raise NotImplementedError\n"
        violations = checker.check_quick(code, "python")
        assert any(v.id == "AI001" for v in violations)

    def test_quick_catches_ai008(self, checker: AIQualityChecker) -> None:
        code = 'cursor.execute(f"SELECT * FROM users WHERE id={uid}")\n'
        violations = checker.check_quick(code, "python")
        assert any(v.id == "AI008" for v in violations)

    def test_quick_does_not_run_ai002(self, checker_with_project: AIQualityChecker) -> None:
        code = "import nonexistent_lib\n"
        violations = checker_with_project.check_quick(code, "python")
        assert not any(v.id == "AI002" for v in violations)

    def test_quick_empty_code(self, checker: AIQualityChecker) -> None:
        violations = checker.check_quick("", "python")
        assert len(violations) == 0

    def test_quick_clean_code(self, checker: AIQualityChecker) -> None:
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        violations = checker.check_quick(code, "python")
        assert len(violations) == 0

    def test_quick_catches_ai007(self, checker: AIQualityChecker) -> None:
        """AI007 should run in quick mode (security-critical)."""
        code = "hashed = hash(password)\n"
        violations = checker.check_quick(code, "python")
        assert any(v.id == "AI007" for v in violations)

    def test_quick_does_not_run_ai003(self, checker: AIQualityChecker) -> None:
        """AI003 should NOT run in quick mode."""
        code = (
            "from abc import ABC\n\n"
            "class Base(ABC):\n"
            "    pass\n\n"
            "class Concrete(Base):\n"
            "    pass\n"
        )
        violations = checker.check_quick(code, "python")
        assert not any(v.id == "AI003" for v in violations)

    def test_quick_does_not_run_ai004(self, checker: AIQualityChecker) -> None:
        """AI004 should NOT run in quick mode."""
        code = (
            "def func_a():\n"
            "    x = 1\n    y = 2\n    z = 3\n    a = 4\n    b = 5\n    return x + y + z + a + b\n\n"
            "def func_b():\n"
            "    x = 1\n    y = 2\n    z = 3\n    a = 4\n    b = 5\n    return x + y + z + a + b\n"
        )
        violations = checker.check_quick(code, "python")
        assert not any(v.id == "AI004" for v in violations)


# ---------------------------------------------------------------------------
# AI003: Over-Engineering Detection
# ---------------------------------------------------------------------------


class TestAI003OverEngineering:
    """Tests for AI003 — over-engineering detection."""

    def test_catches_abstract_with_single_subclass(self, checker: AIQualityChecker) -> None:
        code = (
            "from abc import ABC\n\n"
            "class BaseProcessor(ABC):\n"
            "    def process(self):\n"
            "        pass\n\n"
            "class ConcreteProcessor(BaseProcessor):\n"
            "    def process(self):\n"
            "        return 42\n"
        )
        violations = checker.check(code, "python")
        ai003 = [v for v in violations if v.id == "AI003"]
        assert len(ai003) == 1
        assert "BaseProcessor" in ai003[0].message
        assert "1 concrete subclass" in ai003[0].message

    def test_passes_abstract_with_multiple_subclasses(self, checker: AIQualityChecker) -> None:
        code = (
            "from abc import ABC\n\n"
            "class BaseProcessor(ABC):\n"
            "    def process(self):\n"
            "        pass\n\n"
            "class ProcessorA(BaseProcessor):\n"
            "    def process(self):\n"
            "        return 1\n\n"
            "class ProcessorB(BaseProcessor):\n"
            "    def process(self):\n"
            "        return 2\n"
        )
        violations = checker.check(code, "python")
        ai003 = [v for v in violations if v.id == "AI003"]
        assert len(ai003) == 0

    def test_catches_excessive_generics(self, checker: AIQualityChecker) -> None:
        code = "class MyClass(Generic[A, B, C, D, E, F]):\n    pass\n"
        violations = checker.check(code, "python")
        ai003 = [v for v in violations if v.id == "AI003"]
        assert len(ai003) == 1
        assert "6 generic type" in ai003[0].message

    def test_passes_reasonable_generics(self, checker: AIQualityChecker) -> None:
        code = "class MyClass(Generic[K, V]):\n    pass\n"
        violations = checker.check(code, "python")
        ai003 = [v for v in violations if v.id == "AI003"]
        assert len(ai003) == 0

    def test_severity_is_warning(self, checker: AIQualityChecker) -> None:
        code = "class MyClass(Generic[A, B, C, D, E, F]):\n    pass\n"
        violations = checker.check(code, "python")
        ai003 = [v for v in violations if v.id == "AI003"]
        assert ai003[0].severity == "warning"

    def test_skips_non_python(self, checker: AIQualityChecker) -> None:
        code = "class MyClass<A, B, C, D, E, F> {}\n"
        violations = checker.check(code, "go")
        ai003 = [v for v in violations if v.id == "AI003"]
        assert len(ai003) == 0


# ---------------------------------------------------------------------------
# AI004: Duplicate Code Block Detection
# ---------------------------------------------------------------------------


class TestAI004DuplicateBlocks:
    """Tests for AI004 — duplicate code block detection."""

    def test_catches_duplicate_function_bodies(self, checker: AIQualityChecker) -> None:
        code = (
            "def process_a(data):\n"
            "    result = []\n"
            "    for item in data:\n"
            "        if item > 0:\n"
            "            result.append(item * 2)\n"
            "        else:\n"
            "            result.append(0)\n"
            "    return result\n\n"
            "def process_b(data):\n"
            "    result = []\n"
            "    for item in data:\n"
            "        if item > 0:\n"
            "            result.append(item * 2)\n"
            "        else:\n"
            "            result.append(0)\n"
            "    return result\n"
        )
        violations = checker.check(code, "python")
        ai004 = [v for v in violations if v.id == "AI004"]
        assert len(ai004) == 1
        assert "duplicate" in ai004[0].message.lower()

    def test_passes_different_function_bodies(self, checker: AIQualityChecker) -> None:
        code = (
            "def func_a(data):\n"
            "    return sum(data)\n\n"
            "def func_b(data):\n"
            "    return len(data)\n"
        )
        violations = checker.check(code, "python")
        ai004 = [v for v in violations if v.id == "AI004"]
        assert len(ai004) == 0

    def test_ignores_short_bodies(self, checker: AIQualityChecker) -> None:
        """Bodies <= 5 lines should not be flagged."""
        code = (
            "def func_a():\n"
            "    return 42\n\n"
            "def func_b():\n"
            "    return 42\n"
        )
        violations = checker.check(code, "python")
        ai004 = [v for v in violations if v.id == "AI004"]
        assert len(ai004) == 0

    def test_severity_is_warning(self, checker: AIQualityChecker) -> None:
        code = (
            "def process_a(data):\n"
            "    result = []\n"
            "    for item in data:\n"
            "        if item > 0:\n"
            "            result.append(item * 2)\n"
            "        else:\n"
            "            result.append(0)\n"
            "    return result\n\n"
            "def process_b(data):\n"
            "    result = []\n"
            "    for item in data:\n"
            "        if item > 0:\n"
            "            result.append(item * 2)\n"
            "        else:\n"
            "            result.append(0)\n"
            "    return result\n"
        )
        violations = checker.check(code, "python")
        ai004 = [v for v in violations if v.id == "AI004"]
        assert ai004[0].severity == "warning"


# ---------------------------------------------------------------------------
# AI005: Inconsistent Error Handling
# ---------------------------------------------------------------------------


class TestAI005InconsistentErrors:
    """Tests for AI005 — inconsistent error handling detection."""

    def test_catches_bare_except_mixed_with_specific(self, checker: AIQualityChecker) -> None:
        code = (
            "try:\n"
            "    do_something()\n"
            "except ValueError:\n"
            "    handle_value_error()\n\n"
            "try:\n"
            "    do_other()\n"
            "except:\n"
            "    handle_any()\n"
        )
        violations = checker.check(code, "python")
        ai005 = [v for v in violations if v.id == "AI005"]
        assert len(ai005) >= 1
        assert "bare" in ai005[0].message.lower() or "inconsistent" in ai005[0].message.lower()

    def test_passes_consistent_specific_excepts(self, checker: AIQualityChecker) -> None:
        code = (
            "try:\n"
            "    do_something()\n"
            "except ValueError:\n"
            "    handle_value_error()\n\n"
            "try:\n"
            "    do_other()\n"
            "except TypeError:\n"
            "    handle_type_error()\n"
        )
        violations = checker.check(code, "python")
        ai005 = [v for v in violations if v.id == "AI005"]
        assert len(ai005) == 0

    def test_passes_all_bare_excepts(self, checker: AIQualityChecker) -> None:
        """If all are bare except, that's consistent (though not ideal)."""
        code = (
            "try:\n"
            "    do_something()\n"
            "except:\n"
            "    handle_any()\n\n"
            "try:\n"
            "    do_other()\n"
            "except:\n"
            "    handle_any_other()\n"
        )
        violations = checker.check(code, "python")
        ai005 = [v for v in violations if v.id == "AI005"]
        assert len(ai005) == 0

    def test_severity_is_warning(self, checker: AIQualityChecker) -> None:
        code = (
            "try:\n    x()\nexcept ValueError:\n    pass\n\n"
            "try:\n    y()\nexcept:\n    pass\n"
        )
        violations = checker.check(code, "python")
        ai005 = [v for v in violations if v.id == "AI005"]
        assert ai005[0].severity == "warning"

    def test_skips_non_python(self, checker: AIQualityChecker) -> None:
        code = "try { x() } catch(e) { } try { y() } catch(Error e) { handle(e) }\n"
        violations = checker.check(code, "go")
        ai005 = [v for v in violations if v.id == "AI005"]
        assert len(ai005) == 0


# ---------------------------------------------------------------------------
# AI006: Unnecessary Heavy Imports
# ---------------------------------------------------------------------------


class TestAI006HeavyImports:
    """Tests for AI006 — unnecessary heavy import detection."""

    def test_catches_requests_for_simple_get(self, checker: AIQualityChecker) -> None:
        code = (
            "import requests\n\n"
            "response = requests.get('https://example.com')\n"
        )
        violations = checker.check(code, "python")
        ai006 = [v for v in violations if v.id == "AI006"]
        assert len(ai006) == 1
        assert "requests" in ai006[0].message

    def test_passes_requests_with_complex_usage(self, checker: AIQualityChecker) -> None:
        code = (
            "import requests\n\n"
            "session = requests.Session()\n"
            "session.headers.update({'Authorization': 'Bearer token'})\n"
            "response = requests.post('https://api.example.com', json={'key': 'value'})\n"
            "other = requests.put('https://api.example.com/other', data=payload)\n"
        )
        violations = checker.check(code, "python")
        ai006 = [v for v in violations if v.id == "AI006"]
        assert len(ai006) == 0

    def test_severity_is_info(self, checker: AIQualityChecker) -> None:
        code = "import requests\nresponse = requests.get('https://example.com')\n"
        violations = checker.check(code, "python")
        ai006 = [v for v in violations if v.id == "AI006"]
        if ai006:
            assert ai006[0].severity == "info"

    def test_skips_non_python(self, checker: AIQualityChecker) -> None:
        code = "import requests from 'requests';\n"
        violations = checker.check(code, "typescript")
        ai006 = [v for v in violations if v.id == "AI006"]
        assert len(ai006) == 0


# ---------------------------------------------------------------------------
# AI007: Security Theater Detection
# ---------------------------------------------------------------------------


class TestAI007SecurityTheater:
    """Tests for AI007 — security theater detection."""

    def test_catches_hash_on_password(self, checker: AIQualityChecker) -> None:
        code = "hashed = hash(password)\n"
        violations = checker.check(code, "python")
        ai007 = [v for v in violations if v.id == "AI007"]
        assert len(ai007) == 1
        assert "hash()" in ai007[0].message

    def test_catches_hash_on_secret(self, checker: AIQualityChecker) -> None:
        code = "token_hash = hash(api_secret)\n"
        violations = checker.check(code, "python")
        ai007 = [v for v in violations if v.id == "AI007"]
        assert len(ai007) >= 1

    def test_catches_validate_always_true(self, checker: AIQualityChecker) -> None:
        code = "def validate_token(token):\n    return True\n"
        violations = checker.check(code, "python")
        ai007 = [v for v in violations if v.id == "AI007"]
        assert len(ai007) == 1
        assert "always returns True" in ai007[0].message

    def test_catches_md5_on_password(self, checker: AIQualityChecker) -> None:
        code = "password_hash = hashlib.md5(password.encode()).hexdigest()\n"
        violations = checker.check(code, "python")
        ai007 = [v for v in violations if v.id == "AI007"]
        assert len(ai007) >= 1
        assert "MD5" in ai007[0].message

    def test_passes_hash_on_regular_data(self, checker: AIQualityChecker) -> None:
        """hash() on non-sensitive data is fine."""
        code = "h = hash(my_data)\n"
        violations = checker.check(code, "python")
        ai007 = [v for v in violations if v.id == "AI007"]
        assert len(ai007) == 0

    def test_passes_real_validation(self, checker: AIQualityChecker) -> None:
        code = (
            "def validate_email(email: str) -> bool:\n"
            "    return '@' in email and '.' in email\n"
        )
        violations = checker.check(code, "python")
        ai007 = [v for v in violations if v.id == "AI007"]
        assert len(ai007) == 0

    def test_passes_md5_for_checksum(self, checker: AIQualityChecker) -> None:
        """MD5 used for file checksums (not security) should not flag."""
        code = "file_checksum = hashlib.md5(file_data).hexdigest()\n"
        violations = checker.check(code, "python")
        ai007 = [v for v in violations if v.id == "AI007"]
        assert len(ai007) == 0

    def test_severity_is_error(self, checker: AIQualityChecker) -> None:
        code = "hashed = hash(password)\n"
        violations = checker.check(code, "python")
        ai007 = [v for v in violations if v.id == "AI007"]
        assert ai007[0].severity == "error"

    def test_skips_non_python(self, checker: AIQualityChecker) -> None:
        code = "const h = hash(password);\n"
        violations = checker.check(code, "javascript")
        ai007 = [v for v in violations if v.id == "AI007"]
        assert len(ai007) == 0


# ---------------------------------------------------------------------------
# AI002: Multi-Language Support
# ---------------------------------------------------------------------------


class TestAI002MultiLanguage:
    """Tests for AI002 across multiple languages."""

    def test_typescript_catches_unknown_import(self, tmp_path: Path) -> None:
        pkg_json = tmp_path / "package.json"
        pkg_json.write_text('{"dependencies": {"express": "^4.0"}, "devDependencies": {"jest": "^29.0"}}')
        checker = AIQualityChecker(project_dir=tmp_path)
        code = "import { Router } from 'nonexistent-lib';\n"
        violations = checker.check(code, "typescript")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert len(ai002) == 1
        assert "nonexistent-lib" in ai002[0].message

    def test_typescript_passes_known_dependency(self, tmp_path: Path) -> None:
        pkg_json = tmp_path / "package.json"
        pkg_json.write_text('{"dependencies": {"express": "^4.0"}}')
        checker = AIQualityChecker(project_dir=tmp_path)
        code = "import express from 'express';\n"
        violations = checker.check(code, "typescript")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert len(ai002) == 0

    def test_typescript_passes_node_builtins(self, tmp_path: Path) -> None:
        pkg_json = tmp_path / "package.json"
        pkg_json.write_text('{"dependencies": {}}')
        checker = AIQualityChecker(project_dir=tmp_path)
        code = "import fs from 'fs';\nimport path from 'path';\n"
        violations = checker.check(code, "typescript")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert len(ai002) == 0

    def test_typescript_scoped_package(self, tmp_path: Path) -> None:
        pkg_json = tmp_path / "package.json"
        pkg_json.write_text('{"dependencies": {"@types/node": "^20.0"}}')
        checker = AIQualityChecker(project_dir=tmp_path)
        code = "import { Router } from '@types/node';\n"
        violations = checker.check(code, "typescript")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert len(ai002) == 0

    def test_rust_catches_unknown_crate(self, tmp_path: Path) -> None:
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[dependencies]\nserde = "1.0"\n')
        checker = AIQualityChecker(project_dir=tmp_path)
        code = "use nonexistent_crate::Something;\n"
        violations = checker.check(code, "rust")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert len(ai002) == 1
        assert "nonexistent_crate" in ai002[0].message

    def test_rust_passes_known_crate(self, tmp_path: Path) -> None:
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[dependencies]\nserde = "1.0"\n')
        checker = AIQualityChecker(project_dir=tmp_path)
        code = "use serde::Deserialize;\n"
        violations = checker.check(code, "rust")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert len(ai002) == 0

    def test_rust_passes_std(self, tmp_path: Path) -> None:
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[dependencies]\n')
        checker = AIQualityChecker(project_dir=tmp_path)
        code = "use std::collections::HashMap;\nuse core::fmt;\n"
        violations = checker.check(code, "rust")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert len(ai002) == 0

    def test_go_passes_stdlib(self, tmp_path: Path) -> None:
        go_mod = tmp_path / "go.mod"
        go_mod.write_text("module example.com/myproject\n\ngo 1.21\n")
        checker = AIQualityChecker(project_dir=tmp_path)
        code = 'import "fmt"\nimport "net/http"\n'
        violations = checker.check(code, "go")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert len(ai002) == 0

    def test_go_catches_unknown_module(self, tmp_path: Path) -> None:
        go_mod = tmp_path / "go.mod"
        go_mod.write_text(
            "module example.com/myproject\n\ngo 1.21\n\n"
            "require (\n\tgithub.com/gin-gonic/gin v1.9.0\n)\n"
        )
        checker = AIQualityChecker(project_dir=tmp_path)
        code = 'import "github.com/nonexistent/package"\n'
        violations = checker.check(code, "go")
        ai002 = [v for v in violations if v.id == "AI002"]
        assert len(ai002) == 1


# ---------------------------------------------------------------------------
# SEC014: Vulnerable Dependency Detection
# ---------------------------------------------------------------------------


class TestSEC014VulnerableDependency:
    """Tests for SEC014 — vulnerable dependency via cached vuln findings."""

    def test_sec014_cached_vuln_generates_violation(self, tmp_path: Path) -> None:
        """A cached vuln finding for an imported package should produce SEC014."""
        from unittest.mock import MagicMock

        from mirdan.models import PackageInfo, VulnFinding

        mock_parser = MagicMock()
        mock_parser.parse.return_value = [
            PackageInfo(name="requests", version="2.31.0", ecosystem="PyPI", source="test")
        ]

        mock_scanner = MagicMock()
        mock_scanner.check_cached.return_value = [
            VulnFinding(
                package="requests",
                version="2.31.0",
                ecosystem="PyPI",
                vuln_id="CVE-2024-1234",
                severity="high",
                summary="Test vuln",
                fixed_version="2.32.0",
            )
        ]

        checker = AIQualityChecker(
            manifest_parser=mock_parser, vuln_scanner=mock_scanner
        )
        code = "import requests\nrequests.get('https://example.com')\n"
        violations = checker.check(code, "python")
        sec014 = [v for v in violations if v.id == "SEC014"]
        assert len(sec014) == 1
        assert "requests" in sec014[0].message
        assert sec014[0].severity == "error"  # high → error

    def test_sec014_no_cache_no_violation(self, tmp_path: Path) -> None:
        """Empty cache should produce no SEC014 violations."""
        from unittest.mock import MagicMock

        from mirdan.models import PackageInfo

        mock_parser = MagicMock()
        mock_parser.parse.return_value = [
            PackageInfo(name="requests", version="2.31.0", ecosystem="PyPI", source="test")
        ]
        mock_scanner = MagicMock()
        mock_scanner.check_cached.return_value = []

        checker = AIQualityChecker(
            manifest_parser=mock_parser, vuln_scanner=mock_scanner
        )
        code = "import requests\n"
        violations = checker.check(code, "python")
        sec014 = [v for v in violations if v.id == "SEC014"]
        assert len(sec014) == 0

    def test_sec014_unrelated_import_no_violation(self, tmp_path: Path) -> None:
        """Cached vuln for 'requests' but code imports 'flask' → no SEC014."""
        from unittest.mock import MagicMock

        from mirdan.models import PackageInfo, VulnFinding

        mock_parser = MagicMock()
        mock_parser.parse.return_value = [
            PackageInfo(name="requests", version="2.31.0", ecosystem="PyPI", source="test"),
            PackageInfo(name="flask", version="3.0.0", ecosystem="PyPI", source="test"),
        ]
        mock_scanner = MagicMock()
        mock_scanner.check_cached.return_value = [
            VulnFinding(
                package="requests",
                version="2.31.0",
                ecosystem="PyPI",
                vuln_id="CVE-2024-1234",
                severity="high",
                summary="Test vuln",
                fixed_version="2.32.0",
            )
        ]

        checker = AIQualityChecker(
            manifest_parser=mock_parser, vuln_scanner=mock_scanner
        )
        code = "import flask\napp = flask.Flask(__name__)\n"
        violations = checker.check(code, "python")
        sec014 = [v for v in violations if v.id == "SEC014"]
        assert len(sec014) == 0

    def test_sec014_also_in_check_quick(self, tmp_path: Path) -> None:
        """SEC014 should also fire in check_quick (zero-cost cache lookup)."""
        from unittest.mock import MagicMock

        from mirdan.models import PackageInfo, VulnFinding

        mock_parser = MagicMock()
        mock_parser.parse.return_value = [
            PackageInfo(name="requests", version="2.31.0", ecosystem="PyPI", source="test")
        ]
        mock_scanner = MagicMock()
        mock_scanner.check_cached.return_value = [
            VulnFinding(
                package="requests",
                version="2.31.0",
                ecosystem="PyPI",
                vuln_id="CVE-2024-1234",
                severity="high",
                summary="Test vuln",
                fixed_version="2.32.0",
            )
        ]

        checker = AIQualityChecker(
            manifest_parser=mock_parser, vuln_scanner=mock_scanner
        )
        code = "import requests\n"
        violations = checker.check_quick(code, "python")
        sec014 = [v for v in violations if v.id == "SEC014"]
        assert len(sec014) == 1
