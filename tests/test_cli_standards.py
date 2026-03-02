"""Tests for ``mirdan standards`` CLI command."""

from __future__ import annotations

import json

import pytest
import yaml

from mirdan.cli.standards_command import _parse_args, run_standards


class TestStandardsArgParsing:
    """Tests for argument parsing."""

    def test_parse_language(self) -> None:
        result = _parse_args(["--language", "python"])
        assert result["language"] == "python"

    def test_parse_framework(self) -> None:
        result = _parse_args(["--language", "python", "--framework", "fastapi"])
        assert result["framework"] == "fastapi"

    def test_parse_category(self) -> None:
        result = _parse_args(["--language", "python", "--category", "security"])
        assert result["category"] == "security"

    def test_parse_format_json(self) -> None:
        result = _parse_args(["--language", "python", "--format", "json"])
        assert result["format"] == "json"

    def test_parse_format_yaml(self) -> None:
        result = _parse_args(["--language", "python", "--format", "yaml"])
        assert result["format"] == "yaml"

    def test_parse_invalid_format(self) -> None:
        result = _parse_args(["--language", "python", "--format", "xml"])
        assert "error" in result

    def test_parse_unknown_arg(self) -> None:
        result = _parse_args(["--unknown"])
        assert "error" in result

    def test_parse_help(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _parse_args(["--help"])
        assert exc_info.value.code == 0


class TestStandardsCommand:
    """Tests for the standards command."""

    def test_missing_language(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_standards([])
        assert exc_info.value.code == 2

    def test_standards_yaml_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_standards(["--language", "python"])
        captured = capsys.readouterr()
        # Should be valid YAML
        data = yaml.safe_load(captured.out)
        assert isinstance(data, dict)

    def test_standards_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_standards(["--language", "python", "--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, dict)

    def test_standards_with_framework(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_standards(["--language", "python", "--framework", "fastapi", "--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, dict)

    def test_standards_with_category(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_standards(["--language", "python", "--category", "security", "--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, dict)
