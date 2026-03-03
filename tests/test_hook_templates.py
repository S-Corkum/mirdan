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
        enabled_events=["PreToolUse", "PostToolUse", "Stop"],
        session_hooks=True,
        subagent_hooks=True,
        compaction_resilience=True,
        notification_hooks=True,
    )
    return HookTemplateGenerator(config=config)


class TestHookConfig:
    """Tests for HookConfig defaults."""

    def test_default_events(self) -> None:
        """Default config should have 3 basic events."""
        config = HookConfig()
        assert "PreToolUse" in config.enabled_events
        assert "PostToolUse" in config.enabled_events
        assert "Stop" in config.enabled_events
        assert len(config.enabled_events) == 3

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

    def test_has_pre_tool_use(self, default_generator: HookTemplateGenerator) -> None:
        """Default should include PreToolUse."""
        hooks = default_generator.generate()["hooks"]
        assert "PreToolUse" in hooks

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
        """Full config should have all 9 events."""
        hooks = full_generator.generate()["hooks"]
        assert len(hooks) >= 9

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


class TestPreToolUse:
    """Tests for PreToolUse hook generation."""

    def test_matches_write_edit(self, default_generator: HookTemplateGenerator) -> None:
        """PreToolUse should match Write|Edit."""
        hooks = default_generator.generate()["hooks"]
        pre_tool = hooks["PreToolUse"]
        assert len(pre_tool) > 0
        assert "Write|Edit" in pre_tool[0].get("matcher", "")

    def test_has_prompt_type(self, default_generator: HookTemplateGenerator) -> None:
        """PreToolUse should use prompt type."""
        hooks = default_generator.generate()["hooks"]
        pre_tool = hooks["PreToolUse"]
        hook_item = pre_tool[0]["hooks"][0]
        assert hook_item["type"] == "prompt"

    def test_mentions_enhance_prompt(self, default_generator: HookTemplateGenerator) -> None:
        """PreToolUse prompt should mention enhance_prompt."""
        hooks = default_generator.generate()["hooks"]
        pre_tool = hooks["PreToolUse"]
        hook_item = pre_tool[0]["hooks"][0]
        assert "enhance_prompt" in hook_item["prompt"]


class TestPostToolUse:
    """Tests for PostToolUse hook generation."""

    def test_matches_write_edit(self, default_generator: HookTemplateGenerator) -> None:
        """PostToolUse should match Write|Edit."""
        hooks = default_generator.generate()["hooks"]
        post_tool = hooks["PostToolUse"]
        assert "Write|Edit" in post_tool[0].get("matcher", "")

    def test_has_command_type(self, default_generator: HookTemplateGenerator) -> None:
        """PostToolUse should have a command type hook."""
        hooks = default_generator.generate()["hooks"]
        post_tool = hooks["PostToolUse"]
        hook_types = [h["type"] for h in post_tool[0]["hooks"]]
        assert "command" in hook_types

    def test_uses_micro_format(self, default_generator: HookTemplateGenerator) -> None:
        """PostToolUse command should use micro format."""
        hooks = default_generator.generate()["hooks"]
        post_tool = hooks["PostToolUse"]
        for hook in post_tool[0]["hooks"]:
            if hook["type"] == "command":
                assert "micro" in hook["command"]

    def test_has_timeout(self, default_generator: HookTemplateGenerator) -> None:
        """PostToolUse command should have a timeout."""
        hooks = default_generator.generate()["hooks"]
        post_tool = hooks["PostToolUse"]
        for hook in post_tool[0]["hooks"]:
            if hook["type"] == "command":
                assert "timeout" in hook

    def test_auto_fix_suggestion_included(self) -> None:
        """With auto_fix_suggestions=True, should include fix prompt."""
        config = HookConfig(auto_fix_suggestions=True)
        generator = HookTemplateGenerator(config=config)
        hooks = generator.generate()["hooks"]
        post_tool = hooks["PostToolUse"]
        hook_types = [h["type"] for h in post_tool[0]["hooks"]]
        assert "prompt" in hook_types

    def test_auto_fix_suggestion_excluded(self) -> None:
        """With auto_fix_suggestions=False, should not include fix prompt."""
        config = HookConfig(auto_fix_suggestions=False)
        generator = HookTemplateGenerator(config=config)
        hooks = generator.generate()["hooks"]
        post_tool = hooks["PostToolUse"]
        hook_types = [h["type"] for h in post_tool[0]["hooks"]]
        assert hook_types == ["command"]


class TestStop:
    """Tests for Stop hook generation."""

    def test_has_command_type(self, default_generator: HookTemplateGenerator) -> None:
        """Stop should have a command hook."""
        hooks = default_generator.generate()["hooks"]
        stop = hooks["Stop"]
        assert stop[0]["hooks"][0]["type"] == "command"

    def test_validates_staged(self, default_generator: HookTemplateGenerator) -> None:
        """Stop command should validate staged files."""
        hooks = default_generator.generate()["hooks"]
        stop = hooks["Stop"]
        assert "--staged" in stop[0]["hooks"][0]["command"]


class TestCustomCommand:
    """Tests for custom mirdan command path."""

    def test_custom_command_in_post_tool_use(self) -> None:
        """Custom mirdan command should appear in hooks."""
        generator = HookTemplateGenerator(mirdan_command="uvx mirdan")
        hooks = generator.generate()["hooks"]
        post_tool = hooks["PostToolUse"]
        for hook in post_tool[0]["hooks"]:
            if hook["type"] == "command":
                assert "uvx mirdan" in hook["command"]

    def test_custom_command_in_stop(self) -> None:
        """Custom mirdan command should appear in stop hook."""
        generator = HookTemplateGenerator(mirdan_command="python -m mirdan")
        hooks = generator.generate()["hooks"]
        stop = hooks["Stop"]
        assert "python -m mirdan" in stop[0]["hooks"][0]["command"]


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

    def test_has_nine_events(self) -> None:
        """Should define 9 hook events."""
        assert len(ALL_HOOK_EVENTS) == 9

    def test_includes_core_events(self) -> None:
        """Should include all core events."""
        for event in ("PreToolUse", "PostToolUse", "Stop"):
            assert event in ALL_HOOK_EVENTS

    def test_includes_advanced_events(self) -> None:
        """Should include all advanced events."""
        for event in (
            "SessionStart", "SessionStop",
            "SubagentStart", "SubagentStop",
            "PreCompact", "Notification",
        ):
            assert event in ALL_HOOK_EVENTS
