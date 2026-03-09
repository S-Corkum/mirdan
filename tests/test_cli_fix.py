"""Tests for ``mirdan fix`` CLI command."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mirdan.cli.fix_command import _fix_file, _fix_staged, run_fix


class TestRunFix:
    """Tests for the run_fix entry point."""

    def test_help_flag_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_fix(["--help"])
        assert exc_info.value.code == 0

    def test_h_flag_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_fix(["-h"])
        assert exc_info.value.code == 0

    def test_no_args_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_fix([])
        assert exc_info.value.code == 1
        assert "Error:" in capsys.readouterr().out

    @patch("mirdan.cli.fix_command._fix_file")
    def test_routes_to_fix_file(self, mock_fix: MagicMock) -> None:
        run_fix(["test.py"])
        mock_fix.assert_called_once_with("test.py", dry_run=False, auto_apply=False)

    @patch("mirdan.cli.fix_command._fix_file")
    def test_dry_run_flag(self, mock_fix: MagicMock) -> None:
        run_fix(["test.py", "--dry-run"])
        mock_fix.assert_called_once_with("test.py", dry_run=True, auto_apply=False)

    @patch("mirdan.cli.fix_command._fix_file")
    def test_auto_flag(self, mock_fix: MagicMock) -> None:
        run_fix(["test.py", "--auto"])
        mock_fix.assert_called_once_with("test.py", dry_run=False, auto_apply=True)

    @patch("mirdan.cli.fix_command._fix_staged")
    def test_staged_flag(self, mock_staged: MagicMock) -> None:
        run_fix(["--staged"])
        mock_staged.assert_called_once_with(dry_run=False, auto_apply=False)

    @patch("mirdan.cli.fix_command._fix_staged")
    def test_staged_with_dry_run(self, mock_staged: MagicMock) -> None:
        run_fix(["--staged", "--dry-run"])
        mock_staged.assert_called_once_with(dry_run=True, auto_apply=False)


class TestFixFile:
    """Tests for _fix_file."""

    def test_file_not_found_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _fix_file("/nonexistent/file.py")
        assert exc_info.value.code == 1

    def test_no_violations(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        test_file = tmp_path / "clean.py"
        test_file.write_text("x = 1\n")

        with patch("mirdan.cli.fix_command.MirdanConfig") as mock_config:
            mock_config.find_config.return_value = MagicMock(
                quality=MagicMock(), thresholds=MagicMock()
            )
            with patch("mirdan.cli.fix_command.CodeValidator") as mock_validator:
                mock_result = MagicMock(passed=True, violations=[])
                mock_validator.return_value.validate.return_value = mock_result
                _fix_file(str(test_file))

        assert "No violations" in capsys.readouterr().out

    def test_no_auto_fixes_available(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        test_file = tmp_path / "bad.py"
        test_file.write_text("x = 1\n")
        violation = MagicMock(severity="error")

        with patch("mirdan.cli.fix_command.MirdanConfig") as mock_config:
            mock_config.find_config.return_value = MagicMock(
                quality=MagicMock(), thresholds=MagicMock()
            )
            with patch("mirdan.cli.fix_command.CodeValidator") as mock_validator:
                mock_result = MagicMock(passed=False, violations=[violation])
                mock_validator.return_value.validate.return_value = mock_result
                with patch("mirdan.cli.fix_command.AutoFixer") as mock_fixer:
                    mock_fixer.return_value.batch_fix.return_value = ("", [])
                    _fix_file(str(test_file))

        assert "No auto-fixes" in capsys.readouterr().out

    def test_dry_run_shows_fixes(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        test_file = tmp_path / "bad.py"
        test_file.write_text("x = 1\n")
        violation = MagicMock(severity="error")
        fix = MagicMock(confidence=0.9, fix_description="Remove unused import", fix_code="")

        with patch("mirdan.cli.fix_command.MirdanConfig") as mock_config:
            mock_config.find_config.return_value = MagicMock(
                quality=MagicMock(), thresholds=MagicMock()
            )
            with patch("mirdan.cli.fix_command.CodeValidator") as mock_validator:
                mock_result = MagicMock(passed=False, violations=[violation])
                mock_validator.return_value.validate.return_value = mock_result
                with patch("mirdan.cli.fix_command.AutoFixer") as mock_fixer:
                    mock_fixer.return_value.batch_fix.return_value = ("fixed", [fix])
                    _fix_file(str(test_file), dry_run=True)

        out = capsys.readouterr().out
        assert "Dry run" in out
        assert "1 auto-fix" in out

    def test_auto_apply(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        test_file = tmp_path / "bad.py"
        test_file.write_text("x = 1\n")
        violation = MagicMock(severity="error")
        fix = MagicMock(confidence=0.9, fix_description="Fix it", fix_code="fixed = True")

        with patch("mirdan.cli.fix_command.MirdanConfig") as mock_config:
            mock_config.find_config.return_value = MagicMock(
                quality=MagicMock(), thresholds=MagicMock()
            )
            with patch("mirdan.cli.fix_command.CodeValidator") as mock_validator:
                mock_result = MagicMock(passed=False, violations=[violation])
                mock_validator.return_value.validate.return_value = mock_result
                with patch("mirdan.cli.fix_command.AutoFixer") as mock_fixer:
                    # First call (dry_run=True preview), second call (actual apply)
                    mock_fixer.return_value.batch_fix.side_effect = [
                        ("fixed_code", [fix]),
                        ("fixed_code", [fix]),
                    ]
                    _fix_file(str(test_file), auto_apply=True)

        out = capsys.readouterr().out
        assert "Applied 1 fix" in out


class TestFixStaged:
    """Tests for _fix_staged."""

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_no_git_exits(self, _mock: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _fix_staged()
        assert exc_info.value.code == 1

    @patch("subprocess.run")
    def test_no_staged_files(self, mock_run: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        mock_run.return_value = MagicMock(stdout="\n", returncode=0)
        _fix_staged()
        assert "No staged files" in capsys.readouterr().out

    @patch("subprocess.run")
    def test_no_fixable_files(
        self, mock_run: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_run.return_value = MagicMock(stdout="readme.md\ndata.csv\n", returncode=0)
        _fix_staged()
        assert "No fixable files" in capsys.readouterr().out

    @patch("mirdan.cli.fix_command._fix_file")
    @patch("subprocess.run")
    def test_fixes_python_files(
        self,
        mock_run: MagicMock,
        mock_fix: MagicMock,
        tmp_path: Path,
    ) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        mock_run.return_value = MagicMock(stdout=f"{test_file}\n", returncode=0)
        _fix_staged(dry_run=True)
        mock_fix.assert_called_once()
