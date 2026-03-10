"""Tests for ``mirdan plugin`` CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mirdan.cli.plugin_command import run_plugin


class TestRunPlugin:
    """Tests for the run_plugin entry point."""

    def test_help_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_plugin(["--help"])
        out = capsys.readouterr().out
        assert "Usage:" in out

    def test_h_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_plugin(["-h"])
        out = capsys.readouterr().out
        assert "Usage:" in out

    def test_empty_args_shows_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_plugin([])
        out = capsys.readouterr().out
        assert "Usage:" in out

    def test_unknown_subcommand_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_plugin(["unknown"])
        assert exc_info.value.code == 1

    @patch("mirdan.cli.plugin_command._export")
    def test_routes_export(self, mock_export: MagicMock) -> None:
        run_plugin(["export"])
        mock_export.assert_called_once()


class TestExport:
    """Tests for plugin export."""

    def test_export_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_plugin(["export", "--help"])
        out = capsys.readouterr().out
        assert "Options:" in out

    @patch("mirdan.cli.plugin_command._export_claude_code")
    def test_default_exports_claude_code(self, mock_cc: MagicMock) -> None:
        run_plugin(["export"])
        mock_cc.assert_called_once()

    @patch("mirdan.cli.plugin_command._export_cursor")
    def test_cursor_flag(self, mock_cursor: MagicMock) -> None:
        run_plugin(["export", "--cursor"])
        mock_cursor.assert_called_once()

    @patch("mirdan.cli.plugin_command._export_cursor")
    @patch("mirdan.cli.plugin_command._export_claude_code")
    def test_all_flag(self, mock_cc: MagicMock, mock_cursor: MagicMock) -> None:
        run_plugin(["export", "--all"])
        mock_cc.assert_called_once()
        mock_cursor.assert_called_once()

    @patch("mirdan.cli.plugin_command._export_claude_code")
    def test_output_dir(self, mock_cc: MagicMock) -> None:
        run_plugin(["export", "--output-dir", "/tmp/out"])
        mock_cc.assert_called_once_with(Path("/tmp/out"))

    @patch("mirdan.integrations.claude_code.export_plugin")
    def test_claude_code_export_calls_integration(
        self, mock_export: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_export.return_value = tmp_path / "plugin"
        from mirdan.cli.plugin_command import _export_claude_code

        _export_claude_code(tmp_path)
        mock_export.assert_called_once_with(tmp_path)

    @patch("mirdan.integrations.cursor_plugin.CursorPluginExporter.export")
    def test_cursor_export_calls_integration(
        self, mock_export: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_export.return_value = tmp_path / "cursor"
        from mirdan.cli.plugin_command import _export_cursor

        _export_cursor(tmp_path)
        mock_export.assert_called_once_with(tmp_path)
