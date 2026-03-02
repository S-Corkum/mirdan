"""Tests for the linter runner (mocked subprocesses)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirdan.core.linter_runner import LinterConfig, LinterRunner


class TestLinterConfig:
    """Tests for LinterConfig defaults."""

    def test_default_config(self) -> None:
        config = LinterConfig()
        assert config.enabled_linters == []
        assert config.auto_detect is True
        assert config.timeout == 30.0

    def test_custom_config(self) -> None:
        config = LinterConfig(
            enabled_linters=["ruff"],
            ruff_args=["--fix"],
            timeout=10.0,
        )
        assert config.enabled_linters == ["ruff"]
        assert config.ruff_args == ["--fix"]
        assert config.timeout == 10.0


class TestLinterDetection:
    """Tests for linter availability detection."""

    def test_detects_available_linter(self) -> None:
        runner = LinterRunner()
        with patch("mirdan.core.linter_runner.shutil.which", return_value="/usr/bin/ruff"):
            assert runner.is_available("ruff") is True

    def test_detects_missing_linter(self) -> None:
        runner = LinterRunner()
        with patch("mirdan.core.linter_runner.shutil.which", return_value=None):
            assert runner.is_available("ruff") is False

    def test_available_linters_list(self) -> None:
        def mock_which(name: str) -> str | None:
            return "/usr/bin/ruff" if name == "ruff" else None

        runner = LinterRunner()
        with patch("mirdan.core.linter_runner.shutil.which", side_effect=mock_which):
            available = runner.available_linters()
        assert "ruff" in available
        assert "mypy" not in available


class TestLinterRunner:
    """Tests for running linters via subprocess."""

    async def test_run_ruff_success(self, tmp_path: Path) -> None:
        """Should parse ruff output into violations."""
        code_file = tmp_path / "test.py"
        code_file.write_text("x = 1\n")

        ruff_output = json.dumps([
            {"code": "E501", "message": "Line too long", "location": {"row": 1, "column": 89}},
        ])

        runner = LinterRunner()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(ruff_output.encode(), b""))
        mock_proc.returncode = 1  # ruff returns 1 when issues found

        with (
            patch("mirdan.core.linter_runner.shutil.which", return_value="/usr/bin/ruff"),
            patch("mirdan.core.linter_runner.asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            violations = await runner.run(code_file, "python")

        assert len(violations) == 1
        assert violations[0].id == "RUFF-E501"

    async def test_run_no_linters_available(self, tmp_path: Path) -> None:
        """Should return empty list when no linters are available."""
        code_file = tmp_path / "test.py"
        code_file.write_text("x = 1\n")

        runner = LinterRunner()
        with patch("mirdan.core.linter_runner.shutil.which", return_value=None):
            violations = await runner.run(code_file, "python")

        assert violations == []

    async def test_run_unknown_language(self, tmp_path: Path) -> None:
        """Should return empty list for unregistered languages."""
        code_file = tmp_path / "test.rs"
        code_file.write_text("fn main() {}\n")

        runner = LinterRunner()
        violations = await runner.run(code_file, "rust")
        assert violations == []

    async def test_run_linter_timeout(self, tmp_path: Path) -> None:
        """Should handle linter timeout gracefully."""
        code_file = tmp_path / "test.py"
        code_file.write_text("x = 1\n")

        runner = LinterRunner(LinterConfig(timeout=0.01))

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError)

        with (
            patch("mirdan.core.linter_runner.shutil.which", return_value="/usr/bin/ruff"),
            patch("mirdan.core.linter_runner.asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            violations = await runner.run(code_file, "python")

        assert violations == []

    async def test_run_linter_not_found(self, tmp_path: Path) -> None:
        """Should handle FileNotFoundError gracefully."""
        code_file = tmp_path / "test.py"
        code_file.write_text("x = 1\n")

        runner = LinterRunner()

        with (
            patch("mirdan.core.linter_runner.shutil.which", return_value="/usr/bin/ruff"),
            patch(
                "mirdan.core.linter_runner.asyncio.create_subprocess_exec",
                side_effect=FileNotFoundError,
            ),
        ):
            violations = await runner.run(code_file, "python")

        assert violations == []

    async def test_run_linter_high_exit_code(self, tmp_path: Path) -> None:
        """Should skip output for high exit codes (real errors)."""
        code_file = tmp_path / "test.py"
        code_file.write_text("x = 1\n")

        runner = LinterRunner()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"internal error"))
        mock_proc.returncode = 3

        with (
            patch("mirdan.core.linter_runner.shutil.which", return_value="/usr/bin/ruff"),
            patch("mirdan.core.linter_runner.asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            violations = await runner.run(code_file, "python")

        assert violations == []

    async def test_run_respects_enabled_linters(self, tmp_path: Path) -> None:
        """Should only run linters in the enabled list."""
        code_file = tmp_path / "test.py"
        code_file.write_text("x = 1\n")

        # Only enable mypy, not ruff
        config = LinterConfig(enabled_linters=["mypy"])
        runner = LinterRunner(config)

        def mock_which(name: str) -> str | None:
            return f"/usr/bin/{name}"

        mypy_output = json.dumps(
            {"file": "test.py", "line": 1, "column": 1, "severity": "error", "message": "err", "code": "E"}
        )

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(mypy_output.encode(), b""))
        mock_proc.returncode = 1

        with (
            patch("mirdan.core.linter_runner.shutil.which", side_effect=mock_which),
            patch("mirdan.core.linter_runner.asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            violations = await runner.run(code_file, "python")

        # Should only have mypy violations, not ruff
        assert all(v.id.startswith("MYPY") for v in violations)

    async def test_run_eslint_for_typescript(self, tmp_path: Path) -> None:
        """Should run eslint for TypeScript files."""
        code_file = tmp_path / "test.ts"
        code_file.write_text("const x = 1;\n")

        eslint_output = json.dumps([
            {
                "filePath": str(code_file),
                "messages": [
                    {"ruleId": "no-unused-vars", "severity": 1, "message": "unused", "line": 1, "column": 7},
                ],
            }
        ])

        runner = LinterRunner()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(eslint_output.encode(), b""))
        mock_proc.returncode = 1

        with (
            patch("mirdan.core.linter_runner.shutil.which", return_value="/usr/bin/eslint"),
            patch("mirdan.core.linter_runner.asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            violations = await runner.run(code_file, "typescript")

        assert len(violations) == 1
        assert violations[0].id == "ESLINT-no-unused-vars"


class TestValidateCommandLintFlag:
    """Tests for --lint flag in validate command."""

    def test_parse_lint_flag(self) -> None:
        from mirdan.cli.validate_command import _parse_args

        result = _parse_args(["--file", "test.py", "--lint"])
        assert result["lint"] is True

    def test_parse_no_lint_flag(self) -> None:
        from mirdan.cli.validate_command import _parse_args

        result = _parse_args(["--file", "test.py", "--no-lint"])
        assert result["lint"] is False

    def test_lint_default_false(self) -> None:
        from mirdan.cli.validate_command import _parse_args

        result = _parse_args(["--file", "test.py"])
        assert result.get("lint", False) is False
