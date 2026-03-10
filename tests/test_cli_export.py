"""Tests for ``mirdan export`` CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mirdan.cli.export_command import (
    _export_badge,
    _export_json,
    _export_sarif,
    _get_changed_files,
    run_export,
)


class TestRunExport:
    """Tests for the run_export entry point."""

    def test_help_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_export(["--help"])
        out = capsys.readouterr().out
        assert "Usage:" in out

    def test_h_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_export(["-h"])
        out = capsys.readouterr().out
        assert "Usage:" in out

    def test_empty_args_shows_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_export([])
        out = capsys.readouterr().out
        assert "Usage:" in out

    def test_unknown_format_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_export(["--format", "xml"])
        assert exc_info.value.code == 2

    @patch("mirdan.cli.export_command._export_sarif")
    def test_routes_to_sarif(self, mock_sarif: MagicMock) -> None:
        run_export(["--format", "sarif"])
        mock_sarif.assert_called_once()

    @patch("mirdan.cli.export_command._export_badge")
    def test_routes_to_badge(self, mock_badge: MagicMock) -> None:
        run_export(["--format", "badge"])
        mock_badge.assert_called_once()

    @patch("mirdan.cli.export_command._export_json")
    def test_routes_to_json(self, mock_json: MagicMock) -> None:
        run_export(["--format", "json"])
        mock_json.assert_called_once()

    @patch("mirdan.cli.export_command._export_json")
    def test_default_format_is_json(self, mock_json: MagicMock, tmp_path: Path) -> None:
        # Pass a dummy arg to avoid help branch
        run_export(["--output", str(tmp_path / "out.json")])
        mock_json.assert_called_once()

    @patch("mirdan.cli.export_command._export_sarif")
    def test_output_path_passed(self, mock_sarif: MagicMock, tmp_path: Path) -> None:
        out_path = tmp_path / "out.sarif"
        run_export(["--format", "sarif", "--output", str(out_path)])
        mock_sarif.assert_called_once_with(out_path)

    def test_help_mid_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_export(["--format", "json", "--help"])
        out = capsys.readouterr().out
        assert "Usage:" in out


class TestExportJson:
    """Tests for JSON export."""

    @patch("mirdan.cli.export_command._get_changed_files", return_value=[])
    def test_no_changed_files(self, _mock: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        _export_json(None)
        assert "No changed files" in capsys.readouterr().out

    @patch("mirdan.cli.export_command._get_changed_files")
    def test_json_to_stdout(
        self,
        mock_files: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        mock_files.return_value = [str(test_file)]

        _export_json(None)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "files" in data
        assert "count" in data

    @patch("mirdan.cli.export_command._get_changed_files")
    def test_json_to_file(
        self,
        mock_files: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        mock_files.return_value = [str(test_file)]

        output = tmp_path / "output.json"
        _export_json(output)
        assert output.exists()
        data = json.loads(output.read_text())
        assert "files" in data

    @patch("mirdan.cli.export_command._get_changed_files")
    def test_nonexistent_file_skipped(
        self,
        mock_files: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_files.return_value = ["/nonexistent/file.py"]
        _export_json(None)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["count"] == 0


class TestExportSarif:
    """Tests for SARIF export."""

    @patch("mirdan.cli.export_command._get_changed_files", return_value=[])
    def test_no_changed_files(self, _mock: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        _export_sarif(None)
        assert "No changed files" in capsys.readouterr().out

    @patch("mirdan.cli.export_command._get_changed_files")
    def test_sarif_export(
        self,
        mock_files: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        mock_files.return_value = [str(test_file)]

        output = tmp_path / "results.sarif"
        _export_sarif(output)
        assert output.exists()
        data = json.loads(output.read_text())
        assert "$schema" in data or "runs" in data


class TestExportBadge:
    """Tests for badge export."""

    def test_badge_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        _export_badge()
        out = capsys.readouterr().out
        assert "Badge URL:" in out
        assert "Markdown:" in out


class TestGetChangedFiles:
    """Tests for git changed file detection."""

    @patch("subprocess.run")
    def test_returns_files(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="file1.py\nfile2.py\n")
        result = _get_changed_files()
        assert "file1.py" in result

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_handles_no_git(self, _mock: MagicMock) -> None:
        result = _get_changed_files()
        assert result == []

    @patch("subprocess.run")
    def test_fallback_on_failure(self, mock_run: MagicMock) -> None:
        fail_result = MagicMock(returncode=1, stdout="")
        success_result = MagicMock(returncode=0, stdout="staged.py\n")
        mock_run.side_effect = [fail_result, success_result]
        result = _get_changed_files()
        assert "staged.py" in result
