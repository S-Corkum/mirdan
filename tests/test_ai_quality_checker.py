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
