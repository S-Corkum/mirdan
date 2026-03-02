"""Tests for ``mirdan checklist`` CLI command."""

from __future__ import annotations

import json

import pytest

from mirdan.cli.checklist_command import _parse_args, run_checklist


class TestChecklistArgParsing:
    """Tests for argument parsing."""

    def test_parse_task_type(self) -> None:
        result = _parse_args(["--task-type", "generation"])
        assert result["task_type"] == "generation"

    def test_parse_security(self) -> None:
        result = _parse_args(["--task-type", "generation", "--security"])
        assert result["security"] is True

    def test_parse_format_json(self) -> None:
        result = _parse_args(["--task-type", "generation", "--format", "json"])
        assert result["format"] == "json"

    def test_parse_invalid_format(self) -> None:
        result = _parse_args(["--task-type", "generation", "--format", "xml"])
        assert "error" in result

    def test_parse_unknown_arg(self) -> None:
        result = _parse_args(["--unknown"])
        assert "error" in result

    def test_parse_help(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _parse_args(["--help"])
        assert exc_info.value.code == 0


class TestChecklistCommand:
    """Tests for the checklist command."""

    def test_missing_task_type(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_checklist([])
        assert exc_info.value.code == 2

    def test_invalid_task_type(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_checklist(["--task-type", "invalid"])
        assert exc_info.value.code == 2

    def test_checklist_text_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_checklist(["--task-type", "generation"])
        captured = capsys.readouterr()
        assert "Checklist for generation:" in captured.out
        # Should have numbered steps
        assert "1." in captured.out

    def test_checklist_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_checklist(["--task-type", "generation", "--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["task_type"] == "generation"
        assert isinstance(data["checklist"], list)
        assert len(data["checklist"]) > 0

    def test_checklist_with_security(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_checklist(["--task-type", "generation", "--security", "--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        # Security flag should produce more checklist items
        assert len(data["checklist"]) > 0

    def test_checklist_review_type(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_checklist(["--task-type", "review", "--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["task_type"] == "review"

    def test_checklist_debug_type(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_checklist(["--task-type", "debug", "--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["task_type"] == "debug"

    def test_checklist_planning_type(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_checklist(["--task-type", "planning", "--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["task_type"] == "planning"
