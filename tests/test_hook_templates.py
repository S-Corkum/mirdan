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

    def test_has_command_backed_events(self, full_generator: HookTemplateGenerator) -> None:
        """Full config registers only events with a command-type backing.
        Prompt-only events are skipped because any prompt-type hook is
        LLM-evaluated as a blocking gate and locks continuation when
        the condition text isn't satisfied.
        """
        hooks = full_generator.generate()["hooks"]
        assert "PostToolUse" in hooks
        assert "Stop" in hooks

    def test_session_start_not_registered(
        self, full_generator: HookTemplateGenerator
    ) -> None:
        """SessionStart is prompt-only and would gate every session.

        Prompt-type hooks are LLM-evaluated as blocking conditions in
        the current harness; events without a command-type backing are
        skipped entirely rather than emitted as prompt-only hooks.
        """
        hooks = full_generator.generate()["hooks"]
        assert "SessionStart" not in hooks

    def test_has_session_stop(self, full_generator: HookTemplateGenerator) -> None:
        """Full config should include SessionStop."""
        hooks = full_generator.generate()["hooks"]
        assert "SessionStop" in hooks

    def test_subagent_start_not_registered(
        self, full_generator: HookTemplateGenerator
    ) -> None:
        """SubagentStart is prompt-only → skipped to avoid gate lockups."""
        hooks = full_generator.generate()["hooks"]
        assert "SubagentStart" not in hooks

    def test_subagent_stop_not_registered(
        self, full_generator: HookTemplateGenerator
    ) -> None:
        """SubagentStop is prompt-only → skipped."""
        hooks = full_generator.generate()["hooks"]
        assert "SubagentStop" not in hooks

    def test_pre_compact_not_registered(
        self, full_generator: HookTemplateGenerator
    ) -> None:
        """PreCompact is prompt-only → skipped."""
        hooks = full_generator.generate()["hooks"]
        assert "PreCompact" not in hooks

    def test_notification_not_registered(
        self, full_generator: HookTemplateGenerator
    ) -> None:
        """Notification is prompt-only → skipped."""
        hooks = full_generator.generate()["hooks"]
        assert "Notification" not in hooks


class TestPostToolUse:
    """Tests for PostToolUse hook generation."""

    def test_matches_write_edit_multiedit(self, default_generator: HookTemplateGenerator) -> None:
        """PostToolUse should match Write|Edit|MultiEdit."""
        hooks = default_generator.generate()["hooks"]
        post_tool = hooks["PostToolUse"]
        assert "Write|Edit|MultiEdit" in post_tool[0].get("matcher", "")

    def test_has_no_prompt_type(self, default_generator: HookTemplateGenerator) -> None:
        """PostToolUse must not emit prompt-type hooks.

        Every prompt-type hook is LLM-evaluated as a gating condition;
        a PostToolUse prompt that asks "if the file is a dep manifest,
        call scan_dependencies" blocks continuation whenever the file
        isn't a dep manifest (which is almost every edit).
        """
        hooks = default_generator.generate()["hooks"]
        post_tool = hooks["PostToolUse"]
        hook_types = [h["type"] for h in post_tool[0]["hooks"]]
        assert "prompt" not in hook_types

    def test_has_command_type(self, default_generator: HookTemplateGenerator) -> None:
        """PostToolUse should have a command type hook."""
        hooks = default_generator.generate()["hooks"]
        post_tool = hooks["PostToolUse"]
        hook_types = [h["type"] for h in post_tool[0]["hooks"]]
        assert "command" in hook_types

    def test_command_hook_runs_validate_script(
        self, default_generator: HookTemplateGenerator
    ) -> None:
        """The PostToolUse command-type hook invokes validate-file.sh,
        which in turn calls ``mirdan validate --quick``. Validation
        guidance for the assistant lives in ``.claude/rules/*.md``,
        not in an LLM-evaluated prompt-type hook.
        """
        hooks = default_generator.generate()["hooks"]
        post_tool = hooks["PostToolUse"]
        command_hooks = [h for h in post_tool[0]["hooks"] if h["type"] == "command"]
        assert len(command_hooks) == 1
        assert "validate-file.sh" in command_hooks[0]["command"]


class TestStop:
    """Tests for Stop hook generation."""

    def test_is_command_only(self, default_generator: HookTemplateGenerator) -> None:
        """Stop must be command-only — prompt-type hooks on blocking
        events are LLM-evaluated gates that lock turns regardless of
        wording. The command's exit code is the real gate.
        """
        hooks = default_generator.generate()["hooks"]
        stop = hooks["Stop"]
        hook_types = [h["type"] for h in stop[0]["hooks"]]
        assert hook_types == ["command"]

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
