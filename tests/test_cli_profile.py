"""Tests for ``mirdan profile`` CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mirdan.cli.profile_command import _run_apply, _run_list, _run_suggest, run_profile


class TestRunProfile:
    """Tests for the run_profile entry point."""

    def test_help_flag_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_profile(["--help"])
        assert exc_info.value.code == 0

    def test_h_flag_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_profile(["-h"])
        assert exc_info.value.code == 0

    def test_empty_args_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_profile([])
        assert exc_info.value.code == 0

    def test_unknown_subcommand_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_profile(["unknown"])
        assert exc_info.value.code == 1

    @patch("mirdan.cli.profile_command._run_list")
    def test_routes_list(self, mock_list: MagicMock) -> None:
        run_profile(["list"])
        mock_list.assert_called_once()

    @patch("mirdan.cli.profile_command._run_suggest")
    def test_routes_suggest(self, mock_suggest: MagicMock) -> None:
        run_profile(["suggest"])
        mock_suggest.assert_called_once()

    @patch("mirdan.cli.profile_command._run_apply")
    def test_routes_apply(self, mock_apply: MagicMock) -> None:
        run_profile(["apply", "strict"])
        mock_apply.assert_called_once_with(["strict"])


class TestRunList:
    """Tests for profile listing."""

    @patch("mirdan.cli.profile_command.MirdanConfig")
    @patch("mirdan.cli.profile_command.list_profiles")
    def test_lists_profiles(
        self,
        mock_profiles: MagicMock,
        mock_config: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_config.find_config.return_value = MagicMock(
            quality_profile="default", custom_profiles={}
        )
        mock_profiles.return_value = [
            {"name": "default", "description": "Default profile"},
            {"name": "strict", "description": "Strict profile"},
        ]
        _run_list()
        out = capsys.readouterr().out
        assert "default" in out
        assert "strict" in out

    @patch("mirdan.cli.profile_command.MirdanConfig")
    @patch("mirdan.cli.profile_command.list_profiles")
    def test_shows_active_marker(
        self,
        mock_profiles: MagicMock,
        mock_config: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_config.find_config.return_value = MagicMock(
            quality_profile="strict", custom_profiles={}
        )
        mock_profiles.return_value = [
            {"name": "strict", "description": "Strict profile"},
        ]
        _run_list()
        out = capsys.readouterr().out
        assert "*" in out
        assert "currently active" in out


class TestRunSuggest:
    """Tests for profile suggestion."""

    def test_invalid_directory_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _run_suggest(["/nonexistent/dir"])
        assert exc_info.value.code == 2

    @patch("mirdan.cli.profile_command.get_profile")
    @patch("mirdan.cli.profile_command.suggest_profile")
    @patch("mirdan.core.convention_extractor.ConventionExtractor.scan")
    def test_suggests_profile(
        self,
        mock_scan: MagicMock,
        mock_suggest: MagicMock,
        mock_get: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {}
        mock_scan.return_value = mock_result
        mock_suggest.return_value = ("strict", 0.85)
        mock_get.return_value = MagicMock(
            description="Strict checks",
            security=0.9,
            architecture=0.8,
            testing=0.7,
            documentation=0.6,
            ai_slop_detection=0.8,
            performance=0.7,
            to_stringency=lambda v: "high" if v > 0.7 else "medium",
        )
        _run_suggest([str(tmp_path)])
        out = capsys.readouterr().out
        assert "Suggested profile: strict" in out
        assert "85%" in out


class TestRunApply:
    """Tests for profile application."""

    def test_no_name_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _run_apply([])
        assert exc_info.value.code == 2

    @patch("mirdan.cli.profile_command.MirdanConfig")
    @patch("mirdan.cli.profile_command.get_profile", side_effect=ValueError("not found"))
    def test_invalid_profile_exits(self, mock_get: MagicMock, mock_config: MagicMock) -> None:
        mock_config.find_config.return_value = MagicMock(custom_profiles={})
        with pytest.raises(SystemExit) as exc_info:
            _run_apply(["nonexistent"])
        assert exc_info.value.code == 2

    @patch("mirdan.cli.profile_command.MirdanConfig")
    @patch("mirdan.cli.profile_command.get_profile")
    def test_applies_profile(
        self,
        mock_get: MagicMock,
        mock_config: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config_instance = MagicMock(custom_profiles={}, quality_profile="default")
        mock_config.find_config.return_value = config_instance
        mock_config.find_config_with_path.return_value = (config_instance, tmp_path)
        mock_get.return_value = MagicMock()

        _run_apply(["strict"])
        out = capsys.readouterr().out
        assert "Applied quality profile: strict" in out
        config_instance.save.assert_called_once()

    @patch("mirdan.cli.profile_command.MirdanConfig")
    @patch("mirdan.cli.profile_command.get_profile")
    def test_no_config_dir_exits(
        self,
        mock_get: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        mock_config.find_config.return_value = MagicMock(custom_profiles={})
        mock_config.find_config_with_path.return_value = (MagicMock(), None)
        mock_get.return_value = MagicMock()

        with pytest.raises(SystemExit) as exc_info:
            _run_apply(["strict"])
        assert exc_info.value.code == 2
