"""Tests for Cursor hooks.json generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mirdan.integrations.cursor import (
    CURSOR_STRINGENCY_EVENTS,
    CursorHookStringency,
    generate_cursor_hooks,
)


class TestCursorHookStringency:
    """Tests for CursorHookStringency enum."""

    def test_enum_values(self) -> None:
        """Should have correct string values."""
        assert CursorHookStringency.MINIMAL.value == "minimal"
        assert CursorHookStringency.STANDARD.value == "standard"
        assert CursorHookStringency.COMPREHENSIVE.value == "comprehensive"

    def test_stringency_events_minimal(self) -> None:
        """MINIMAL should have 2 events."""
        events = CURSOR_STRINGENCY_EVENTS[CursorHookStringency.MINIMAL]
        assert len(events) == 2
        assert "afterFileEdit" in events
        assert "stop" in events

    def test_stringency_events_standard(self) -> None:
        """STANDARD should have 5 events."""
        events = CURSOR_STRINGENCY_EVENTS[CursorHookStringency.STANDARD]
        assert len(events) == 5
        assert "afterFileEdit" in events
        assert "preToolUse" in events
        assert "postToolUse" in events
        assert "sessionStart" in events
        assert "stop" in events

    def test_stringency_events_comprehensive(self) -> None:
        """COMPREHENSIVE should have 16 events."""
        events = CURSOR_STRINGENCY_EVENTS[CursorHookStringency.COMPREHENSIVE]
        assert len(events) == 16
        assert "afterFileEdit" in events
        assert "preToolUse" in events
        assert "postToolUse" in events
        assert "postToolUseFailure" in events
        assert "stop" in events
        assert "sessionStart" in events
        assert "beforeSubmitPrompt" in events
        assert "subagentStart" in events
        assert "preCompact" in events


class TestGenerateCursorHooks:
    """Tests for generate_cursor_hooks() function."""

    def test_produces_valid_json(self, tmp_path: Path) -> None:
        """Generated hooks.json should be valid JSON."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir)
        assert result is not None
        data = json.loads(result.read_text())
        assert isinstance(data, dict)

    def test_has_version_field(self, tmp_path: Path) -> None:
        """Generated hooks.json should have version: 1."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir)
        assert result is not None
        data = json.loads(result.read_text())
        assert data["version"] == 1

    def test_comprehensive_has_sixteen_events(self, tmp_path: Path) -> None:
        """COMPREHENSIVE should produce 16 hook events."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir, CursorHookStringency.COMPREHENSIVE)
        assert result is not None
        data = json.loads(result.read_text())
        assert len(data["hooks"]) == 16

    def test_standard_has_five_events(self, tmp_path: Path) -> None:
        """STANDARD should produce 5 hook events."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir, CursorHookStringency.STANDARD)
        assert result is not None
        data = json.loads(result.read_text())
        assert len(data["hooks"]) == 5

    def test_minimal_has_two_events(self, tmp_path: Path) -> None:
        """MINIMAL should produce 2 hook events."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir, CursorHookStringency.MINIMAL)
        assert result is not None
        data = json.loads(result.read_text())
        assert len(data["hooks"]) == 2

    def test_stop_hook_has_loop_limit(self, tmp_path: Path) -> None:
        """stop hook should have loop_limit field."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir)
        assert result is not None
        data = json.loads(result.read_text())
        stop_hooks = data["hooks"]["stop"]
        assert any("loop_limit" in h for h in stop_hooks)

    def test_after_file_edit_mentions_validate(self, tmp_path: Path) -> None:
        """afterFileEdit prompt should mention validate_code_quality."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir)
        assert result is not None
        data = json.loads(result.read_text())
        hooks = data["hooks"]["afterFileEdit"]
        prompt = hooks[0]["prompt"]
        assert "validate_code_quality" in prompt

    def test_all_hooks_use_prompt_type(self, tmp_path: Path) -> None:
        """All hooks should use type: prompt."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir)
        assert result is not None
        data = json.loads(result.read_text())
        for event_name, hook_list in data["hooks"].items():
            for hook in hook_list:
                assert hook["type"] == "prompt", f"{event_name} hook not prompt type"

    def test_idempotent_skips_existing(self, tmp_path: Path) -> None:
        """Should return None if hooks.json already exists."""
        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir(parents=True)
        (cursor_dir / "hooks.json").write_text('{"custom": true}')

        result = generate_cursor_hooks(cursor_dir)
        assert result is None

        # Original file should be preserved
        data = json.loads((cursor_dir / "hooks.json").read_text())
        assert data == {"custom": True}

    def test_creates_cursor_dir(self, tmp_path: Path) -> None:
        """Should create .cursor/ directory if it doesn't exist."""
        cursor_dir = tmp_path / ".cursor"
        assert not cursor_dir.exists()

        result = generate_cursor_hooks(cursor_dir)
        assert result is not None
        assert cursor_dir.exists()

    def test_default_stringency_is_comprehensive(self, tmp_path: Path) -> None:
        """Default stringency should be COMPREHENSIVE."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir)
        assert result is not None
        data = json.loads(result.read_text())
        # COMPREHENSIVE has 16 events
        assert len(data["hooks"]) == 16

    def test_pre_tool_use_has_matcher(self, tmp_path: Path) -> None:
        """preToolUse hook should have matcher field."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir, CursorHookStringency.STANDARD)
        assert result is not None
        data = json.loads(result.read_text())
        hooks = data["hooks"]["preToolUse"]
        assert any("matcher" in h for h in hooks)
