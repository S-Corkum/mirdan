"""Tests for ``mirdan scan`` CLI command."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mirdan.cli.scan_command import (
    _output_text,
    _output_workspace_text,
    _parse_args,
    _save_conventions,
    _save_workspace_conventions,
    run_scan,
)
from mirdan.core.convention_extractor import ScanResult


def _make_scan_result(**kwargs: object) -> ScanResult:
    """Create a ScanResult for testing."""
    defaults = {
        "directory": "/test",
        "language": "python",
        "files_scanned": 5,
        "avg_score": 0.85,
        "pass_rate": 0.8,
        "common_violations": [],
        "conventions": [],
    }
    defaults.update(kwargs)
    return ScanResult(**defaults)


class TestParseArgs:
    """Tests for argument parsing."""

    def test_empty_args(self) -> None:
        result = _parse_args([])
        assert "error" not in result

    def test_language_arg(self) -> None:
        result = _parse_args(["--language", "python"])
        assert result["language"] == "python"

    def test_format_json(self) -> None:
        result = _parse_args(["--format", "json"])
        assert result["format"] == "json"

    def test_format_text(self) -> None:
        result = _parse_args(["--format", "text"])
        assert result["format"] == "text"

    def test_invalid_format(self) -> None:
        result = _parse_args(["--format", "xml"])
        assert "error" in result

    def test_directory_positional(self) -> None:
        result = _parse_args(["/some/path"])
        assert result["directory"] == "/some/path"

    def test_unknown_arg(self) -> None:
        result = _parse_args(["--unknown"])
        assert "error" in result

    def test_help_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _parse_args(["--help"])
        assert exc_info.value.code == 0


class TestRunScan:
    """Tests for the run_scan entry point."""

    def test_parse_error_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_scan(["--format", "xml"])
        assert exc_info.value.code == 2

    def test_invalid_directory_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_scan(["/nonexistent/dir/that/does/not/exist"])
        assert exc_info.value.code == 2

    @patch("mirdan.cli.scan_command._run_dependency_scan")
    def test_deps_flag_routes(self, mock_scan: MagicMock) -> None:
        run_scan(["--dependencies"])
        mock_scan.assert_called_once()

    @patch("mirdan.cli.scan_command._run_dependency_scan")
    def test_deps_alias_routes(self, mock_scan: MagicMock) -> None:
        run_scan(["--deps"])
        mock_scan.assert_called_once()

    @patch("mirdan.cli.scan_command._run_single_scan")
    @patch("mirdan.cli.scan_command.MirdanConfig")
    def test_single_scan_routing(
        self, mock_config: MagicMock, mock_scan: MagicMock, tmp_path: Path
    ) -> None:
        mock_config.find_config.return_value = MagicMock(is_workspace=False)
        run_scan([str(tmp_path)])
        mock_scan.assert_called_once()


class TestOutputText:
    """Tests for text output formatting."""

    def test_basic_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = _make_scan_result()
        _output_text(result)
        out = capsys.readouterr().out
        assert "Language:" in out
        assert "python" in out
        assert "Files scanned:" in out

    def test_with_violations(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = _make_scan_result(common_violations=[{"id": "AI001", "count": 5}])
        _output_text(result)
        out = capsys.readouterr().out
        assert "Common violations:" in out
        assert "AI001" in out

    def test_with_conventions(self, capsys: pytest.CaptureFixture[str]) -> None:
        convention = MagicMock(
            content_type="convention",
            content="Use snake_case",
            tags=["naming"],
            confidence=0.9,
        )
        result = _make_scan_result(conventions=[convention])
        _output_text(result)
        out = capsys.readouterr().out
        assert "1 convention" in out

    def test_no_conventions(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = _make_scan_result()
        _output_text(result)
        out = capsys.readouterr().out
        assert "No conventions discovered" in out


class TestWorkspaceText:
    """Tests for workspace text output."""

    def test_workspace_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        results = {
            "project-a": _make_scan_result(language="python", files_scanned=10),
            "project-b": _make_scan_result(language="typescript", files_scanned=5),
        }
        _output_workspace_text(results, 15)
        out = capsys.readouterr().out
        assert "Total files scanned: 15" in out
        assert "[project-a]" in out
        assert "[project-b]" in out


class TestSaveConventions:
    """Tests for convention saving."""

    def test_saves_yaml(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        result = _make_scan_result()
        path = tmp_path / "conventions.yaml"
        _save_conventions(result, path)
        assert path.exists()
        assert "Conventions saved" in capsys.readouterr().out

    def test_saves_workspace_yaml(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        results = {"proj-a": _make_scan_result()}
        path = tmp_path / "conventions.yaml"
        _save_workspace_conventions(results, path)
        assert path.exists()
        content = path.read_text()
        assert "proj-a" in content


class TestDependencyScan:
    """Tests for dependency scanning."""

    @patch("mirdan.core.manifest_parser.ManifestParser.parse", return_value=[])
    @patch("mirdan.cli.scan_command.MirdanConfig")
    def test_no_packages(
        self,
        mock_config: MagicMock,
        _mock_parse: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_config.find_config.return_value = MagicMock(dependencies=MagicMock(osv_cache_ttl=3600))

        from mirdan.cli.scan_command import _run_dependency_scan

        _run_dependency_scan([])
        assert "No dependency manifests" in capsys.readouterr().out

    @patch("asyncio.run", return_value=[])
    @patch("mirdan.core.manifest_parser.ManifestParser.parse")
    @patch("mirdan.cli.scan_command.MirdanConfig")
    def test_no_vulns(
        self,
        mock_config: MagicMock,
        mock_parse: MagicMock,
        _mock_asyncio: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_config.find_config.return_value = MagicMock(dependencies=MagicMock(osv_cache_ttl=3600))
        mock_parse.return_value = [MagicMock()]

        from mirdan.cli.scan_command import _run_dependency_scan

        _run_dependency_scan([])
        assert "No vulnerabilities" in capsys.readouterr().out

    @patch("asyncio.run")
    @patch("mirdan.core.manifest_parser.ManifestParser.parse")
    @patch("mirdan.cli.scan_command.MirdanConfig")
    def test_with_vulns(
        self,
        mock_config: MagicMock,
        mock_parse: MagicMock,
        mock_asyncio: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_config.find_config.return_value = MagicMock(dependencies=MagicMock(osv_cache_ttl=3600))
        mock_parse.return_value = [MagicMock()]
        finding = MagicMock(
            severity="high",
            package="urllib3",
            version="2.0.0",
            vuln_id="CVE-2024-1234",
            fixed_version="2.0.1",
            summary="A vulnerability",
        )
        mock_asyncio.return_value = [finding]

        from mirdan.cli.scan_command import _run_dependency_scan

        _run_dependency_scan([])
        out = capsys.readouterr().out
        assert "1 vulnerabilities" in out
        assert "urllib3" in out
