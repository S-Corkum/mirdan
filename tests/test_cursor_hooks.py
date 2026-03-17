"""Tests for Cursor hooks.json generation."""

from __future__ import annotations

import json
from pathlib import Path

from mirdan.integrations.cursor import (
    CURSOR_STRINGENCY_EVENTS,
    CursorHookStringency,
    generate_cursor_hook_scripts,
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
        """STANDARD should have 4 events."""
        events = CURSOR_STRINGENCY_EVENTS[CursorHookStringency.STANDARD]
        assert len(events) == 4
        assert "afterFileEdit" in events
        assert "postToolUse" in events
        assert "sessionStart" in events
        assert "stop" in events

    def test_stringency_events_comprehensive(self) -> None:
        """COMPREHENSIVE should have 7 events."""
        events = CURSOR_STRINGENCY_EVENTS[CursorHookStringency.COMPREHENSIVE]
        assert len(events) == 7
        assert "afterFileEdit" in events
        assert "postToolUseFailure" in events
        assert "stop" in events
        assert "sessionStart" in events
        assert "beforeShellExecution" in events
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

    def test_comprehensive_has_seven_events(self, tmp_path: Path) -> None:
        """COMPREHENSIVE should produce 7 hook events."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir, CursorHookStringency.COMPREHENSIVE)
        assert result is not None
        data = json.loads(result.read_text())
        assert len(data["hooks"]) == 7

    def test_standard_has_four_events(self, tmp_path: Path) -> None:
        """STANDARD should produce 4 hook events."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir, CursorHookStringency.STANDARD)
        assert result is not None
        data = json.loads(result.read_text())
        assert len(data["hooks"]) == 4

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

    def test_comprehensive_does_not_include_before_submit_prompt(self, tmp_path: Path) -> None:
        """beforeSubmitPrompt was removed from COMPREHENSIVE to avoid false positives."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir)
        assert result is not None
        data = json.loads(result.read_text())
        assert "beforeSubmitPrompt" not in data["hooks"]

    def test_all_hooks_have_valid_type(self, tmp_path: Path) -> None:
        """All hooks should use type: prompt or type: command."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir)
        assert result is not None
        data = json.loads(result.read_text())
        valid_types = {"prompt", "command"}
        for event_name, hook_list in data["hooks"].items():
            for hook in hook_list:
                assert hook["type"] in valid_types, (
                    f"{event_name} hook has invalid type: {hook['type']}"
                )

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
        # COMPREHENSIVE has 7 events
        assert len(data["hooks"]) == 7


class TestCommandTypeHooks:
    """Tests for command-type hook generation."""

    def test_before_shell_has_command_hook(self, tmp_path: Path) -> None:
        """beforeShellExecution should have a command-type hook."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir)
        assert result is not None
        data = json.loads(result.read_text())
        hooks = data["hooks"]["beforeShellExecution"]
        command_hooks = [h for h in hooks if h["type"] == "command"]
        assert len(command_hooks) == 1

    def test_stop_has_command_hook(self, tmp_path: Path) -> None:
        """stop should have a command-type hook."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir)
        assert result is not None
        data = json.loads(result.read_text())
        hooks = data["hooks"]["stop"]
        command_hooks = [h for h in hooks if h["type"] == "command"]
        assert len(command_hooks) == 1

    def test_command_hook_has_command_field(self, tmp_path: Path) -> None:
        """Command hooks must have a 'command' field with script path."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir)
        assert result is not None
        data = json.loads(result.read_text())
        hooks = data["hooks"]["beforeShellExecution"]
        cmd_hook = next(h for h in hooks if h["type"] == "command")
        assert "command" in cmd_hook
        assert "mirdan-shell-guard.sh" in cmd_hook["command"]

    def test_shell_guard_fail_closed(self, tmp_path: Path) -> None:
        """Shell guard should fail closed (block if script errors)."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir)
        assert result is not None
        data = json.loads(result.read_text())
        hooks = data["hooks"]["beforeShellExecution"]
        cmd_hook = next(h for h in hooks if h["type"] == "command")
        assert cmd_hook.get("failClosed") is True

    def test_stop_gate_has_loop_limit(self, tmp_path: Path) -> None:
        """Stop gate command hook should have loop_limit."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir)
        assert result is not None
        data = json.loads(result.read_text())
        hooks = data["hooks"]["stop"]
        cmd_hook = next(h for h in hooks if h["type"] == "command")
        assert "loop_limit" in cmd_hook

    def test_mixed_hooks_prompt_and_command(self, tmp_path: Path) -> None:
        """Events should have both prompt and command hooks."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir)
        assert result is not None
        data = json.loads(result.read_text())

        for event in ("beforeShellExecution", "stop"):
            hooks = data["hooks"][event]
            types = {h["type"] for h in hooks}
            assert "prompt" in types, f"{event} missing prompt hook"
            assert "command" in types, f"{event} missing command hook"

    def test_command_hook_listed_before_prompt(self, tmp_path: Path) -> None:
        """Command hooks should fire first (listed before prompt hooks)."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir)
        assert result is not None
        data = json.loads(result.read_text())

        hooks = data["hooks"]["beforeShellExecution"]
        assert hooks[0]["type"] == "command"

    def test_minimal_stringency_has_no_shell_command_hook(self, tmp_path: Path) -> None:
        """MINIMAL stringency doesn't include beforeShellExecution."""
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_hooks(cursor_dir, CursorHookStringency.MINIMAL)
        assert result is not None
        data = json.loads(result.read_text())
        assert "beforeShellExecution" not in data["hooks"]


class TestHookScriptGeneration:
    """Tests for generate_cursor_hook_scripts()."""

    def test_generates_two_scripts(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        paths = generate_cursor_hook_scripts(cursor_dir)
        assert len(paths) == 2

    def test_creates_hooks_directory(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        assert not (cursor_dir / "hooks").exists()
        generate_cursor_hook_scripts(cursor_dir)
        assert (cursor_dir / "hooks").is_dir()

    def test_scripts_are_executable(self, tmp_path: Path) -> None:
        import stat

        cursor_dir = tmp_path / ".cursor"
        generate_cursor_hook_scripts(cursor_dir)
        hooks_dir = cursor_dir / "hooks"
        for script in hooks_dir.iterdir():
            mode = script.stat().st_mode
            assert mode & stat.S_IXUSR, f"{script.name} must be executable"

    def test_scripts_have_shebang(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_hook_scripts(cursor_dir)
        hooks_dir = cursor_dir / "hooks"
        for script in hooks_dir.iterdir():
            content = script.read_text()
            assert content.startswith("#!/"), f"{script.name} must have shebang"

    def test_shell_guard_has_deny_patterns(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_hook_scripts(cursor_dir)
        content = (cursor_dir / "hooks" / "mirdan-shell-guard.sh").read_text()
        assert "rm" in content
        assert "DROP" in content
        assert "force" in content
        assert "reset --hard" in content

    def test_stop_gate_checks_git_changes(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_hook_scripts(cursor_dir)
        content = (cursor_dir / "hooks" / "mirdan-stop-gate.sh").read_text()
        assert "git diff" in content

    def test_idempotent_does_not_overwrite(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        first = generate_cursor_hook_scripts(cursor_dir)
        assert len(first) == 2

        target = cursor_dir / "hooks" / "mirdan-shell-guard.sh"
        target.write_text("#!/bin/bash\n# custom")

        second = generate_cursor_hook_scripts(cursor_dir)
        assert second == []
        assert "custom" in target.read_text()

    def test_expected_script_names(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_hook_scripts(cursor_dir)
        hooks_dir = cursor_dir / "hooks"
        names = {p.name for p in hooks_dir.iterdir()}
        assert names == {"mirdan-shell-guard.sh", "mirdan-stop-gate.sh"}
