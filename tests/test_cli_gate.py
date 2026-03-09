"""Tests for ``mirdan gate`` CLI command."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mirdan.cli.gate_command import (
    _get_changed_files,
    _output_fail,
    _output_pass,
    _parse_gate_args,
    _print_workspace_grouped_results,
    run_gate,
)


class TestParseGateArgs:
    """Tests for gate argument parsing."""

    def test_empty_args(self) -> None:
        result = _parse_gate_args([])
        assert result == {}

    def test_format_arg(self) -> None:
        result = _parse_gate_args(["--format", "json"])
        assert result["format"] == "json"

    def test_help_flag(self) -> None:
        result = _parse_gate_args(["--help"])
        assert result["help"] is True

    def test_include_deps(self) -> None:
        result = _parse_gate_args(["--include-dependencies"])
        assert result["include_deps"] is True

    def test_include_deps_alias(self) -> None:
        result = _parse_gate_args(["--include-deps"])
        assert result["include_deps"] is True

    def test_unknown_args_ignored(self) -> None:
        result = _parse_gate_args(["--unknown"])
        assert "error" not in result


class TestOutputPass:
    """Tests for pass output."""

    def test_text_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        _output_pass(5, 0.95, "text", [])
        out = capsys.readouterr().out
        assert "PASS" in out
        assert "5 files" in out

    def test_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        _output_pass(5, 0.95, "json", [{"file": "test.py", "score": 0.95}])
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "PASS"
        assert data["files"] == 5


class TestOutputFail:
    """Tests for fail output."""

    def test_text_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        _output_fail(5, 3, 2, 0.6, "text", [])
        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "3 errors" in out

    def test_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        _output_fail(5, 3, 2, 0.6, "json", [])
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "FAIL"
        assert data["total_errors"] == 3

    def test_workspace_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        results = [
            {
                "file": "a/test.py",
                "score": 0.5,
                "passed": False,
                "errors": 2,
                "warnings": 1,
                "project": "project-a",
                "project_language": "python",
            }
        ]
        _output_fail(1, 2, 1, 0.5, "text", results, is_workspace=True)
        out = capsys.readouterr().out
        assert "[project-a]" in out


class TestWorkspaceGroupedResults:
    """Tests for workspace grouped output."""

    def test_groups_by_project(self, capsys: pytest.CaptureFixture[str]) -> None:
        results: list[dict[str, Any]] = [
            {
                "file": "a/foo.py",
                "score": 0.5,
                "errors": 1,
                "warnings": 0,
                "project": "proj-a",
                "project_language": "python",
            },
            {
                "file": "b/bar.ts",
                "score": 0.6,
                "errors": 0,
                "warnings": 1,
                "project": "proj-b",
                "project_language": "typescript",
            },
        ]
        _print_workspace_grouped_results(results)
        out = capsys.readouterr().out
        assert "[proj-a]" in out
        assert "[proj-b]" in out
        assert "(python)" in out
        assert "(typescript)" in out

    def test_root_project(self, capsys: pytest.CaptureFixture[str]) -> None:
        results: list[dict[str, Any]] = [
            {"file": "test.py", "score": 0.5, "errors": 1, "warnings": 0},
        ]
        _print_workspace_grouped_results(results)
        out = capsys.readouterr().out
        assert "(root)" in out


class TestGetChangedFiles:
    """Tests for git file detection."""

    @patch("subprocess.run")
    def test_combines_staged_and_unstaged(self, mock_run: MagicMock) -> None:
        diff_result = MagicMock(stdout="file1.py\nfile2.py\n")
        staged_result = MagicMock(stdout="file2.py\nfile3.py\n")
        mock_run.side_effect = [diff_result, staged_result]
        result = _get_changed_files()
        assert result == ["file1.py", "file2.py", "file3.py"]

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_handles_no_git(self, _mock: MagicMock) -> None:
        result = _get_changed_files()
        assert result == []


class TestRunGate:
    """Tests for the run_gate entry point."""

    def test_help_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_gate(["--help"])
        assert exc_info.value.code == 0

    @patch("mirdan.cli.gate_command._get_changed_files", return_value=[])
    def test_no_changes_passes(self, _mock: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_gate([])
        assert exc_info.value.code == 0
        assert "PASS" in capsys.readouterr().out

    @patch("mirdan.cli.gate_command._get_changed_files")
    def test_clean_files_pass(
        self,
        mock_files: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        mock_files.return_value = [str(test_file)]

        with pytest.raises(SystemExit) as exc_info:
            run_gate([])
        # Should pass (exit 0) since simple code has no errors
        assert exc_info.value.code == 0

    @patch("mirdan.cli.gate_command._get_changed_files")
    def test_json_format(
        self,
        mock_files: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        mock_files.return_value = [str(test_file)]

        with pytest.raises(SystemExit):
            run_gate(["--format", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "status" in data

    @patch("mirdan.cli.gate_command._get_changed_files")
    def test_nonexistent_files_skipped(
        self,
        mock_files: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_files.return_value = ["/nonexistent/file.py"]
        with pytest.raises(SystemExit) as exc_info:
            run_gate([])
        assert exc_info.value.code == 0
