"""Tests for ``mirdan report`` CLI command."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mirdan.cli.report_command import (
    _build_project_summaries,
    _discover_source_files,
    _output_markdown,
    _output_text_report,
    _parse_report_args,
    _run_compact_state_report,
    _run_session_report,
    run_report,
)


class TestParseReportArgs:
    """Tests for argument parsing."""

    def test_empty_args(self) -> None:
        result = _parse_report_args([])
        assert result == {}

    def test_format_arg(self) -> None:
        result = _parse_report_args(["--format", "json"])
        assert result["format"] == "json"

    def test_language_arg(self) -> None:
        result = _parse_report_args(["--language", "python"])
        assert result["language"] == "python"

    def test_session_flag(self) -> None:
        result = _parse_report_args(["--session"])
        assert result["session"] is True

    def test_compact_state_flag(self) -> None:
        result = _parse_report_args(["--compact-state"])
        assert result["compact_state"] is True

    def test_help_flag(self) -> None:
        result = _parse_report_args(["--help"])
        assert result["help"] is True

    def test_directory_positional(self) -> None:
        result = _parse_report_args(["/some/path"])
        assert result["directory"] == "/some/path"


class TestDiscoverSourceFiles:
    """Tests for source file discovery."""

    def test_finds_python_files(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("x = 1\n")
        (src / "utils.py").write_text("y = 2\n")

        files = _discover_source_files(tmp_path, "python")
        assert len(files) == 2

    def test_filters_by_language(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("x = 1\n")
        (tmp_path / "app.ts").write_text("const x = 1;\n")

        py_files = _discover_source_files(tmp_path, "python")
        ts_files = _discover_source_files(tmp_path, "typescript")
        assert len(py_files) == 1
        assert len(ts_files) == 1

    def test_skips_hidden_dirs(self, tmp_path: Path) -> None:
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("x = 1\n")

        files = _discover_source_files(tmp_path, "python")
        assert len(files) == 0

    def test_skips_node_modules(self, tmp_path: Path) -> None:
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "dep.js").write_text("var x = 1;\n")

        files = _discover_source_files(tmp_path)
        assert len(files) == 0

    def test_all_languages(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("x = 1\n")
        (tmp_path / "app.js").write_text("var x = 1;\n")

        files = _discover_source_files(tmp_path)
        assert len(files) == 2


class TestBuildProjectSummaries:
    """Tests for workspace project summaries."""

    def test_groups_by_project(self) -> None:
        results: list[dict[str, Any]] = [
            {
                "file": "a/foo.py",
                "score": 0.9,
                "passed": True,
                "errors": 0,
                "warnings": 0,
                "project": "proj-a",
                "project_language": "python",
            },
            {
                "file": "b/bar.ts",
                "score": 0.8,
                "passed": True,
                "errors": 0,
                "warnings": 1,
                "project": "proj-b",
                "project_language": "typescript",
            },
        ]
        summaries = _build_project_summaries(results)
        assert "proj-a" in summaries
        assert "proj-b" in summaries
        assert summaries["proj-a"]["files_analyzed"] == 1

    def test_root_project(self) -> None:
        results: list[dict[str, Any]] = [
            {"file": "test.py", "score": 0.9, "passed": True, "errors": 0, "warnings": 0}
        ]
        summaries = _build_project_summaries(results)
        assert "(root)" in summaries


class TestOutputTextReport:
    """Tests for text report output."""

    def test_basic_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        report: dict[str, Any] = {
            "directory": "/test",
            "files_analyzed": 5,
            "avg_score": 0.85,
            "pass_rate": 0.8,
            "total_violations": {"error": 2, "warning": 3},
            "files": [
                {"file": "bad.py", "score": 0.5, "passed": False, "errors": 2, "warnings": 1},
                {"file": "good.py", "score": 1.0, "passed": True, "errors": 0, "warnings": 0},
            ],
        }
        _output_text_report(report)
        out = capsys.readouterr().out
        assert "mirdan Quality Report" in out
        assert "Files analyzed: 5" in out
        assert "bad.py" in out

    def test_workspace_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        report: dict[str, Any] = {
            "directory": "/test",
            "files_analyzed": 5,
            "avg_score": 0.85,
            "pass_rate": 0.8,
            "total_violations": {"error": 0, "warning": 0},
            "files": [],
            "projects": {
                "proj-a": {
                    "language": "python",
                    "files_analyzed": 3,
                    "avg_score": 0.9,
                    "pass_rate": 1.0,
                    "errors": 0,
                    "warnings": 0,
                },
            },
        }
        _output_text_report(report, is_workspace=True)
        out = capsys.readouterr().out
        assert "Per-project summary:" in out
        assert "[proj-a]" in out


class TestOutputMarkdown:
    """Tests for markdown report output."""

    def test_markdown_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        report: dict[str, Any] = {
            "files_analyzed": 3,
            "avg_score": 0.9,
            "pass_rate": 1.0,
            "total_violations": {"error": 0, "warning": 1},
            "files": [
                {"file": "warn.py", "score": 0.8, "passed": False, "errors": 0, "warnings": 1},
            ],
        }
        _output_markdown(report)
        out = capsys.readouterr().out
        assert "# mirdan Quality Report" in out
        assert "| Metric | Value |" in out
        assert "`warn.py`" in out


class TestRunReport:
    """Tests for the run_report entry point."""

    def test_help_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_report(["--help"])
        assert exc_info.value.code == 0

    def test_invalid_directory_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_report(["/nonexistent/dir/does/not/exist"])
        assert exc_info.value.code == 2

    @patch("mirdan.cli.report_command._run_session_report")
    def test_session_flag_routes(self, mock_session: MagicMock) -> None:
        run_report(["--session"])
        mock_session.assert_called_once()

    @patch("mirdan.cli.report_command._run_compact_state_report")
    def test_compact_state_routes(self, mock_compact: MagicMock) -> None:
        run_report(["--compact-state"])
        mock_compact.assert_called_once()


class TestSessionReport:
    """Tests for session report."""

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_no_git_empty_report(
        self, _mock: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run_session_report({"format": "json"})
        data = json.loads(capsys.readouterr().out)
        assert data["validation_count"] == 0

    @patch("subprocess.run")
    def test_no_changes_empty_report(
        self, mock_run: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_run.return_value = MagicMock(stdout="\n")
        _run_session_report({"format": "json"})
        data = json.loads(capsys.readouterr().out)
        assert data["files_validated"] == 0

    @patch("subprocess.run")
    def test_with_changed_files(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        mock_run.return_value = MagicMock(stdout=f"{test_file}\n")

        _run_session_report({"format": "json"})
        data = json.loads(capsys.readouterr().out)
        assert data["files_validated"] >= 0


class TestCompactStateReport:
    """Tests for compact state report."""

    def test_json_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        _run_compact_state_report({"format": "json"})
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, dict)

    def test_text_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        _run_compact_state_report({"format": "text"})
        out = capsys.readouterr().out
        assert len(out) > 0
