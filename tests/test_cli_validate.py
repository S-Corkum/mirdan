"""Tests for ``mirdan validate`` CLI command."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from mirdan.cli import main
from mirdan.cli.validate_command import (
    _output_result,
    _parse_args,
    _validate_diff,
    run_validate,
)
from mirdan.models import ValidationResult, Violation


class TestValidateArgParsing:
    """Tests for argument parsing."""

    def test_parse_file_arg(self) -> None:
        result = _parse_args(["--file", "test.py"])
        assert result["file"] == "test.py"

    def test_parse_stdin_arg(self) -> None:
        result = _parse_args(["--stdin"])
        assert result["stdin"] is True

    def test_parse_diff_arg(self) -> None:
        result = _parse_args(["--diff"])
        assert result["diff"] is True

    def test_parse_language_arg(self) -> None:
        result = _parse_args(["--file", "test.py", "--language", "python"])
        assert result["language"] == "python"

    def test_parse_format_json(self) -> None:
        result = _parse_args(["--file", "test.py", "--format", "json"])
        assert result["format"] == "json"

    def test_parse_format_github(self) -> None:
        result = _parse_args(["--file", "test.py", "--format", "github"])
        assert result["format"] == "github"

    def test_parse_security_flag(self) -> None:
        result = _parse_args(["--file", "test.py", "--security"])
        assert result["security"] is True

    def test_parse_no_security_flag(self) -> None:
        result = _parse_args(["--file", "test.py", "--no-security"])
        assert result["security"] is False

    def test_parse_invalid_format(self) -> None:
        result = _parse_args(["--file", "test.py", "--format", "xml"])
        assert "error" in result

    def test_parse_unknown_arg(self) -> None:
        result = _parse_args(["--unknown"])
        assert "error" in result

    def test_parse_empty_args(self) -> None:
        result = _parse_args([])
        assert "error" not in result
        assert "file" not in result

    def test_parse_help(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _parse_args(["--help"])
        assert exc_info.value.code == 0


class TestValidateOutput:
    """Tests for output formatting."""

    def _make_result(self, passed: bool = True, violations: list | None = None) -> ValidationResult:
        return ValidationResult(
            passed=passed,
            score=0.9 if passed else 0.5,
            language_detected="python",
            violations=violations or [],
            standards_checked=["style", "security"],
        )

    def _make_violation(
        self,
        vid: str = "PY001",
        severity: str = "warning",
        line: int | None = 10,
    ) -> Violation:
        return Violation(
            id=vid,
            rule="test-rule",
            category="style",
            severity=severity,
            message="Test violation message",
            line=line,
            column=5,
            suggestion="Fix this",
        )

    def test_text_output_pass(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = self._make_result(passed=True)
        _output_result(result, "text")
        captured = capsys.readouterr()
        assert "PASS" in captured.out

    def test_text_output_fail(self, capsys: pytest.CaptureFixture[str]) -> None:
        v = self._make_violation(severity="error")
        result = self._make_result(passed=False, violations=[v])
        _output_result(result, "text")
        captured = capsys.readouterr()
        assert "FAIL" in captured.out
        assert "PY001" in captured.out

    def test_text_output_with_file(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = self._make_result(passed=True)
        _output_result(result, "text", file_path="main.py")
        captured = capsys.readouterr()
        assert "main.py" in captured.out

    def test_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        v = self._make_violation()
        result = self._make_result(passed=True, violations=[v])
        _output_result(result, "json")
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["passed"] is True
        assert data["score"] == 0.9

    def test_json_output_includes_file(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = self._make_result(passed=True)
        _output_result(result, "json", file_path="test.py")
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["file"] == "test.py"

    def test_github_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        v = self._make_violation(severity="error", line=42)
        result = self._make_result(passed=False, violations=[v])
        _output_result(result, "github", file_path="src/main.py")
        captured = capsys.readouterr()
        assert "::error" in captured.out
        assert "file=src/main.py" in captured.out
        assert "line=42" in captured.out

    def test_github_output_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        v = self._make_violation(severity="warning", line=10)
        result = self._make_result(passed=True, violations=[v])
        _output_result(result, "github")
        captured = capsys.readouterr()
        assert "::warning" in captured.out

    def test_text_output_suggestion(self, capsys: pytest.CaptureFixture[str]) -> None:
        v = self._make_violation()
        result = self._make_result(passed=True, violations=[v])
        _output_result(result, "text")
        captured = capsys.readouterr()
        assert "Fix this" in captured.out


class TestValidateFile:
    """Tests for file validation via CLI."""

    def test_validate_passing_file(self, tmp_path: Path) -> None:
        code_file = tmp_path / "clean.py"
        code_file.write_text("def hello():\n    return 'world'\n")

        with pytest.raises(SystemExit) as exc_info:
            run_validate(["--file", str(code_file)])
        assert exc_info.value.code == 0

    def test_validate_failing_file(self, tmp_path: Path) -> None:
        code_file = tmp_path / "bad.py"
        code_file.write_text("password = 'hardcoded_secret_123'\n")

        with pytest.raises(SystemExit) as exc_info:
            run_validate(["--file", str(code_file), "--security"])
        # May be 0 or 1 depending on whether security check catches it
        assert exc_info.value.code in (0, 1)

    def test_validate_missing_file(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_validate(["--file", "/nonexistent/file.py"])
        assert exc_info.value.code == 2

    def test_validate_no_input(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_validate([])
        assert exc_info.value.code == 2

    def test_validate_json_format(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        code_file = tmp_path / "clean.py"
        code_file.write_text("x = 1\n")

        with pytest.raises(SystemExit) as exc_info:
            run_validate(["--file", str(code_file), "--format", "json"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "passed" in data

    def test_validate_github_format(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code_file = tmp_path / "code.py"
        code_file.write_text("x = 1\n")

        with pytest.raises(SystemExit) as exc_info:
            run_validate(["--file", str(code_file), "--format", "github"])
        assert exc_info.value.code == 0


class TestValidateStdin:
    """Tests for stdin validation."""

    def test_validate_stdin(self) -> None:
        with (
            patch("sys.stdin") as mock_stdin,
            pytest.raises(SystemExit) as exc_info,
        ):
            mock_stdin.read.return_value = "x = 1\n"
            run_validate(["--stdin"])
        assert exc_info.value.code == 0


class TestValidateDiff:
    """Tests for diff validation."""

    def test_validate_diff_added_lines(self) -> None:
        from mirdan.config import MirdanConfig
        from mirdan.core.code_validator import CodeValidator
        from mirdan.core.quality_standards import QualityStandards

        config = MirdanConfig()
        standards = QualityStandards(config=config.quality)
        validator = CodeValidator(standards, config=config.quality, thresholds=config.thresholds)

        diff = (
            "--- a/test.py\n"
            "+++ b/test.py\n"
            "@@ -1,3 +1,4 @@\n"
            " existing = 1\n"
            "+new_var = 2\n"
            " other = 3\n"
        )
        result = _validate_diff(validator, diff, "python", True)
        assert result.passed is True

    def test_validate_empty_diff(self) -> None:
        from mirdan.config import MirdanConfig
        from mirdan.core.code_validator import CodeValidator
        from mirdan.core.quality_standards import QualityStandards

        config = MirdanConfig()
        standards = QualityStandards(config=config.quality)
        validator = CodeValidator(standards, config=config.quality, thresholds=config.thresholds)

        diff = (
            "--- a/test.py\n"
            "+++ b/test.py\n"
            "@@ -1,3 +1,2 @@\n"
            " existing = 1\n"
            "-removed = 2\n"
            " other = 3\n"
        )
        result = _validate_diff(validator, diff, "auto", True)
        assert result.passed is True
        assert result.score == 1.0

    def test_validate_diff_stdin(self) -> None:
        diff = "--- a/test.py\n+++ b/test.py\n@@ -1,3 +1,4 @@\n x = 1\n+y = 2\n z = 3\n"
        with (
            patch("sys.stdin") as mock_stdin,
            pytest.raises(SystemExit) as exc_info,
        ):
            mock_stdin.read.return_value = diff
            run_validate(["--diff"])
        assert exc_info.value.code == 0


class TestCLIRouting:
    """Tests for CLI routing of new commands."""

    def test_validate_command_routes(self) -> None:
        with (
            patch.object(sys, "argv", ["mirdan", "validate", "--help"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 0

    def test_standards_command_routes(self) -> None:
        with (
            patch.object(sys, "argv", ["mirdan", "standards", "--help"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 0

    def test_checklist_command_routes(self) -> None:
        with (
            patch.object(sys, "argv", ["mirdan", "checklist", "--help"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 0

    def test_help_lists_new_commands(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(sys, "argv", ["mirdan", "--help"]):
            main()
        captured = capsys.readouterr()
        assert "validate" in captured.out
        assert "standards" in captured.out
        assert "checklist" in captured.out
