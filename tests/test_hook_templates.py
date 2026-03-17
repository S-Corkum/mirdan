"""Tests for the hook template generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mirdan.integrations.hook_templates import (
    ALL_HOOK_EVENTS,
    HookConfig,
    HookTemplateGenerator,
)


@pytest.fixture
def default_generator() -> HookTemplateGenerator:
    """Create a generator with default config."""
    return HookTemplateGenerator()


@pytest.fixture
def full_generator() -> HookTemplateGenerator:
    """Create a generator with all events enabled."""
    config = HookConfig(
        enabled_events=["PostToolUse", "Stop"],
        session_hooks=True,
        subagent_hooks=True,
        compaction_resilience=True,
        notification_hooks=True,
    )
    return HookTemplateGenerator(config=config)


class TestHookConfig:
    """Tests for HookConfig defaults."""

    def test_default_events(self) -> None:
        """Default config should have 2 basic events."""
        config = HookConfig()
        assert "PostToolUse" in config.enabled_events
        assert "Stop" in config.enabled_events
        assert len(config.enabled_events) == 2

    def test_default_timeout(self) -> None:
        """Default timeout should be 5000ms."""
        config = HookConfig()
        assert config.quick_validate_timeout == 5000

    def test_auto_fix_suggestions_default(self) -> None:
        """Auto-fix suggestions should be on by default."""
        config = HookConfig()
        assert config.auto_fix_suggestions is True

    def test_advanced_features_off_by_default(self) -> None:
        """Advanced features should be off by default."""
        config = HookConfig()
        assert config.compaction_resilience is False
        assert config.multi_agent_awareness is False
        assert config.session_hooks is False
        assert config.subagent_hooks is False
        assert config.notification_hooks is False


class TestDefaultGeneration:
    """Tests for default (3-event) hook generation."""

    def test_generates_hooks_dict(self, default_generator: HookTemplateGenerator) -> None:
        """Should generate a dict with 'hooks' key."""
        result = default_generator.generate()
        assert "hooks" in result

    def test_has_post_tool_use(self, default_generator: HookTemplateGenerator) -> None:
        """Default should include PostToolUse."""
        hooks = default_generator.generate()["hooks"]
        assert "PostToolUse" in hooks

    def test_has_stop(self, default_generator: HookTemplateGenerator) -> None:
        """Default should include Stop."""
        hooks = default_generator.generate()["hooks"]
        assert "Stop" in hooks

    def test_no_session_hooks_by_default(self, default_generator: HookTemplateGenerator) -> None:
        """Default should not include SessionStart/SessionStop."""
        hooks = default_generator.generate()["hooks"]
        assert "SessionStart" not in hooks
        assert "SessionStop" not in hooks

    def test_no_subagent_hooks_by_default(self, default_generator: HookTemplateGenerator) -> None:
        """Default should not include SubagentStart/SubagentStop."""
        hooks = default_generator.generate()["hooks"]
        assert "SubagentStart" not in hooks
        assert "SubagentStop" not in hooks


class TestFullGeneration:
    """Tests for full (all-event) hook generation."""

    def test_has_all_core_events(self, full_generator: HookTemplateGenerator) -> None:
        """Full config should have all 8 events."""
        hooks = full_generator.generate()["hooks"]
        assert len(hooks) >= 8

    def test_has_session_start(self, full_generator: HookTemplateGenerator) -> None:
        """Full config should include SessionStart."""
        hooks = full_generator.generate()["hooks"]
        assert "SessionStart" in hooks

    def test_has_session_stop(self, full_generator: HookTemplateGenerator) -> None:
        """Full config should include SessionStop."""
        hooks = full_generator.generate()["hooks"]
        assert "SessionStop" in hooks

    def test_has_subagent_start(self, full_generator: HookTemplateGenerator) -> None:
        """Full config should include SubagentStart."""
        hooks = full_generator.generate()["hooks"]
        assert "SubagentStart" in hooks

    def test_has_subagent_stop(self, full_generator: HookTemplateGenerator) -> None:
        """Full config should include SubagentStop."""
        hooks = full_generator.generate()["hooks"]
        assert "SubagentStop" in hooks

    def test_has_pre_compact(self, full_generator: HookTemplateGenerator) -> None:
        """Full config should include PreCompact."""
        hooks = full_generator.generate()["hooks"]
        assert "PreCompact" in hooks

    def test_has_notification(self, full_generator: HookTemplateGenerator) -> None:
        """Full config should include Notification."""
        hooks = full_generator.generate()["hooks"]
        assert "Notification" in hooks


class TestPostToolUse:
    """Tests for PostToolUse hook generation."""

    def test_matches_write_edit_multiedit(self, default_generator: HookTemplateGenerator) -> None:
        """PostToolUse should match Write|Edit|MultiEdit."""
        hooks = default_generator.generate()["hooks"]
        post_tool = hooks["PostToolUse"]
        assert "Write|Edit|MultiEdit" in post_tool[0].get("matcher", "")

    def test_has_prompt_type(self, default_generator: HookTemplateGenerator) -> None:
        """PostToolUse should have a prompt type hook."""
        hooks = default_generator.generate()["hooks"]
        post_tool = hooks["PostToolUse"]
        hook_types = [h["type"] for h in post_tool[0]["hooks"]]
        assert "prompt" in hook_types

    def test_has_command_type(self, default_generator: HookTemplateGenerator) -> None:
        """PostToolUse should have a command type hook."""
        hooks = default_generator.generate()["hooks"]
        post_tool = hooks["PostToolUse"]
        hook_types = [h["type"] for h in post_tool[0]["hooks"]]
        assert "command" in hook_types

    def test_prompt_mentions_validate(self, default_generator: HookTemplateGenerator) -> None:
        """PostToolUse prompt hook should mention validate_code_quality."""
        hooks = default_generator.generate()["hooks"]
        post_tool = hooks["PostToolUse"]
        prompt_hooks = [h for h in post_tool[0]["hooks"] if h["type"] == "prompt"]
        assert len(prompt_hooks) > 0
        assert "validate_code_quality" in prompt_hooks[0]["prompt"]


class TestStop:
    """Tests for Stop hook generation."""

    def test_has_command_and_prompt(self, default_generator: HookTemplateGenerator) -> None:
        """Stop should have command+prompt combo for quality gate."""
        hooks = default_generator.generate()["hooks"]
        stop = hooks["Stop"]
        hook_types = [h["type"] for h in stop[0]["hooks"]]
        assert "command" in hook_types
        assert "prompt" in hook_types

    def test_command_runs_gate(self, default_generator: HookTemplateGenerator) -> None:
        """Stop command should run mirdan gate."""
        hooks = default_generator.generate()["hooks"]
        stop = hooks["Stop"]
        command_hooks = [h for h in stop[0]["hooks"] if h["type"] == "command"]
        assert len(command_hooks) > 0
        assert "gate" in command_hooks[0]["command"]


class TestGenerateAndWrite:
    """Tests for file writing."""

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        """Should write valid JSON to disk."""
        generator = HookTemplateGenerator()
        hooks_path = tmp_path / ".claude" / "hooks.json"
        result = generator.generate_and_write(hooks_path)
        assert result == hooks_path
        assert hooks_path.exists()

        data = json.loads(hooks_path.read_text())
        assert "hooks" in data

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Should create parent directories if they don't exist."""
        generator = HookTemplateGenerator()
        hooks_path = tmp_path / "deep" / "nested" / "hooks.json"
        generator.generate_and_write(hooks_path)
        assert hooks_path.exists()


class TestAllHookEvents:
    """Tests for the ALL_HOOK_EVENTS constant."""

    def test_has_sixteen_events(self) -> None:
        """Should define 16 hook events."""
        assert len(ALL_HOOK_EVENTS) == 16

    def test_includes_core_events(self) -> None:
        """Should include all core events."""
        for event in ("UserPromptSubmit", "PostToolUse", "Stop"):
            assert event in ALL_HOOK_EVENTS

    def test_includes_advanced_events(self) -> None:
        """Should include all advanced events."""
        for event in (
            "SessionStart",
            "SessionStop",
            "SubagentStart",
            "SubagentStop",
            "PreCompact",
            "Notification",
        ):
            assert event in ALL_HOOK_EVENTS
