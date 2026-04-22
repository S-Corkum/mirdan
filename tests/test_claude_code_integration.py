"""Tests for Claude Code integration module.

Tests cover:
- generate_claude_code_config() writes hooks into .claude/settings.json
  under the "hooks" key (the only location Claude Code loads hooks from)
- settings.json["hooks"]["PostToolUse"] uses the validate-file.sh helper
- Rule files are generated in .claude/rules/
- Existing settings.json "hooks" key not overwritten without --upgrade
- Non-hook keys in settings.json (permissions, etc.) are preserved on merge
- Legacy .claude/hooks.json is migrated to hooks.json.deprecated
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mirdan.cli.detect import DetectedProject
from mirdan.integrations.claude_code import generate_claude_code_config


@pytest.fixture()
def detected_python() -> DetectedProject:
    """A detected Python project."""
    return DetectedProject(
        project_name="test-project",
        project_type="application",
        primary_language="python",
        frameworks=["fastapi"],
        detected_ides=["claude-code"],
    )


@pytest.fixture()
def detected_typescript() -> DetectedProject:
    """A detected TypeScript project."""
    return DetectedProject(
        project_name="test-ts-project",
        project_type="application",
        primary_language="typescript",
        frameworks=["react"],
        detected_ides=["claude-code"],
    )


# ---------------------------------------------------------------------------
# settings.json hook-block generation
# ---------------------------------------------------------------------------


class TestHooksGeneration:
    """Tests for hook injection into .claude/settings.json."""

    def test_creates_settings_json(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """Should create .claude/settings.json with a hooks block."""
        generated = generate_claude_code_config(tmp_path, detected_python)
        settings_path = tmp_path / ".claude" / "settings.json"
        assert settings_path.exists()
        assert settings_path in generated
        data = json.loads(settings_path.read_text())
        assert "hooks" in data

    def test_does_not_create_legacy_hooks_json(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """Should NOT create .claude/hooks.json — Claude Code never reads it."""
        generate_claude_code_config(tmp_path, detected_python)
        legacy = tmp_path / ".claude" / "hooks.json"
        assert not legacy.exists()

    def test_hooks_has_post_tool_use(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """settings.json hooks should have PostToolUse entry."""
        generate_claude_code_config(tmp_path, detected_python)
        settings_path = tmp_path / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        assert "PostToolUse" in data["hooks"]

    def test_hooks_has_stop(self, tmp_path: Path, detected_python: DetectedProject) -> None:
        """settings.json hooks should have Stop entry for final quality check."""
        generate_claude_code_config(tmp_path, detected_python)
        settings_path = tmp_path / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        assert "Stop" in data["hooks"]

    def test_post_tool_use_is_command_only(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """PostToolUse must be command-only — any prompt-type hook
        here is evaluated as a gating condition and blocks continuation
        whenever the condition text doesn't match the just-edited file.
        """
        generate_claude_code_config(tmp_path, detected_python)
        settings_path = tmp_path / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        ptu_hooks = data["hooks"]["PostToolUse"][0]["hooks"]
        hook_types = [h["type"] for h in ptu_hooks]
        assert hook_types == ["command"]

    def test_stop_is_command_only(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """Stop hook must be command-only.

        Prompt-type hooks on blocking events (Stop, UserPromptSubmit)
        are LLM-evaluated as gating conditions by the hook-evaluator
        and lock users out whenever the condition can't be cleanly
        satisfied. The command's exit code is the real gate.
        """
        generate_claude_code_config(tmp_path, detected_python)
        settings_path = tmp_path / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        stop_hooks = data["hooks"]["Stop"][0]["hooks"]
        hook_types = [h["type"] for h in stop_hooks]
        assert hook_types == ["command"]

    def test_post_tool_use_matcher(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """PostToolUse should match Write|Edit|MultiEdit tools."""
        generate_claude_code_config(tmp_path, detected_python)
        settings_path = tmp_path / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        matcher = data["hooks"]["PostToolUse"][0]["matcher"]
        assert matcher == "Write|Edit|MultiEdit"

    def test_preserves_existing_settings_keys(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """Merging hooks should preserve permissions and other keys."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings_path = claude_dir / "settings.json"
        settings_path.write_text(
            json.dumps(
                {
                    "permissions": {"allow": ["WebSearch"]},
                    "enabledMcpjsonServers": ["mirdan"],
                }
            )
        )

        generate_claude_code_config(tmp_path, detected_python)

        data = json.loads(settings_path.read_text())
        assert data["permissions"] == {"allow": ["WebSearch"]}
        assert data["enabledMcpjsonServers"] == ["mirdan"]
        assert "hooks" in data

    def test_existing_hooks_key_not_overwritten_without_upgrade(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """An existing settings.json hooks key should be preserved."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings_path = claude_dir / "settings.json"
        settings_path.write_text(json.dumps({"hooks": {"Stop": "custom"}}))

        generated = generate_claude_code_config(tmp_path, detected_python)

        assert settings_path not in generated
        assert json.loads(settings_path.read_text()) == {"hooks": {"Stop": "custom"}}

    def test_upgrade_regenerates_hooks_and_backs_up(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """--upgrade should rewrite hooks and leave a .bak of prior settings."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings_path = claude_dir / "settings.json"
        settings_path.write_text(json.dumps({"hooks": {"Stop": "custom"}}))

        generate_claude_code_config(tmp_path, detected_python, upgrade=True)

        backup = settings_path.with_suffix(".json.bak")
        assert backup.exists()
        assert json.loads(backup.read_text()) == {"hooks": {"Stop": "custom"}}
        data = json.loads(settings_path.read_text())
        assert "PostToolUse" in data["hooks"]

    def test_legacy_hooks_json_is_renamed(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """A legacy .claude/hooks.json should be renamed to .deprecated."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        legacy = claude_dir / "hooks.json"
        legacy.write_text('{"legacy": true}')

        generate_claude_code_config(tmp_path, detected_python)

        assert not legacy.exists()
        deprecated = claude_dir / "hooks.json.deprecated"
        assert deprecated.exists()
        assert json.loads(deprecated.read_text()) == {"legacy": True}


# ---------------------------------------------------------------------------
# Rule files generation
# ---------------------------------------------------------------------------


class TestRulesGeneration:
    """Tests for .claude/rules/ generation."""

    def test_creates_rules_directory(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """Should create .claude/rules/ directory."""
        generate_claude_code_config(tmp_path, detected_python)
        rules_dir = tmp_path / ".claude" / "rules"
        assert rules_dir.is_dir()

    def test_generates_quality_rule(self, tmp_path: Path, detected_python: DetectedProject) -> None:
        """Should generate mirdan-quality.md rule file."""
        generate_claude_code_config(tmp_path, detected_python)
        quality_path = tmp_path / ".claude" / "rules" / "mirdan-quality.md"
        assert quality_path.exists()
        content = quality_path.read_text()
        assert "mirdan" in content.lower()

    def test_generates_security_rule(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """Should generate mirdan-security.md rule file."""
        generate_claude_code_config(tmp_path, detected_python)
        security_path = tmp_path / ".claude" / "rules" / "mirdan-security.md"
        assert security_path.exists()
        content = security_path.read_text()
        assert "security" in content.lower()

    def test_generates_python_rule(self, tmp_path: Path, detected_python: DetectedProject) -> None:
        """Should generate mirdan-python.md for Python projects."""
        generate_claude_code_config(tmp_path, detected_python)
        python_path = tmp_path / ".claude" / "rules" / "mirdan-python.md"
        assert python_path.exists()
        content = python_path.read_text()
        assert "python" in content.lower()

    def test_generates_typescript_rule(
        self, tmp_path: Path, detected_typescript: DetectedProject
    ) -> None:
        """Should generate mirdan-typescript.md for TypeScript projects."""
        generate_claude_code_config(tmp_path, detected_typescript)
        ts_path = tmp_path / ".claude" / "rules" / "mirdan-typescript.md"
        assert ts_path.exists()
        content = ts_path.read_text()
        assert "typescript" in content.lower()

    def test_no_python_rule_for_typescript(
        self, tmp_path: Path, detected_typescript: DetectedProject
    ) -> None:
        """Should NOT generate mirdan-python.md for TypeScript projects."""
        generate_claude_code_config(tmp_path, detected_typescript)
        python_path = tmp_path / ".claude" / "rules" / "mirdan-python.md"
        assert not python_path.exists()

    def test_rules_have_yaml_frontmatter(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """Rule files should have YAML frontmatter with description."""
        generate_claude_code_config(tmp_path, detected_python)
        quality_path = tmp_path / ".claude" / "rules" / "mirdan-quality.md"
        content = quality_path.read_text()
        assert content.startswith("---")
        assert "description:" in content

    def test_rules_overwritten_on_regeneration(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """Rule files should be overwritten on regeneration."""
        # First generation
        generate_claude_code_config(tmp_path, detected_python)
        quality_path = tmp_path / ".claude" / "rules" / "mirdan-quality.md"
        # Modify content
        quality_path.write_text("modified content")
        # Second generation should overwrite
        generate_claude_code_config(tmp_path, detected_python)
        assert quality_path.read_text() != "modified content"


# ---------------------------------------------------------------------------
# Full config generation
# ---------------------------------------------------------------------------


class TestFullConfigGeneration:
    """Tests for the complete generation flow."""

    def test_returns_all_generated_paths(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """Should return all generated file paths."""
        generated = generate_claude_code_config(tmp_path, detected_python)
        # settings.json + quality + security + python = 4 files
        assert len(generated) >= 4

    def test_all_paths_exist(self, tmp_path: Path, detected_python: DetectedProject) -> None:
        """All returned paths should exist on disk."""
        generated = generate_claude_code_config(tmp_path, detected_python)
        for path in generated:
            assert path.exists(), f"{path} does not exist"


# ---------------------------------------------------------------------------
# v0.2.0: Hook delegation, advanced events, self-managing integration
# ---------------------------------------------------------------------------


class TestHookDelegation:
    """Tests for hook generation delegating to HookTemplateGenerator."""

    def test_hooks_json_has_command_backed_events(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """Only events with a command-type backing are emitted.

        Prompt-only events (SessionStart, SubagentStart/Stop, PreCompact,
        Notification, TeammateIdle, PermissionRequest, ConfigChange,
        WorktreeCreate/Remove, PostToolUseFailure) are skipped because
        any prompt-type hook is LLM-evaluated as a blocking gate and
        locks continuation when the condition isn't satisfied. That
        leaves PostToolUse, Stop, SessionStop, and TaskCompleted.
        """
        generate_claude_code_config(tmp_path, detected_python)
        settings_path = tmp_path / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        hooks = data["hooks"]
        assert len(hooks) >= 2
        assert "PostToolUse" in hooks
        assert "Stop" in hooks

    def test_user_prompt_submit_absent_when_llm_disabled(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """UserPromptSubmit should not be registered without LLM triage.

        A non-LLM UserPromptSubmit hook is pure advisory text; because
        the hook-evaluator treats prompt-type hooks on blocking events
        as gates, any wording locks the turn when the evaluator can't
        confirm the condition. Without a command-type triage source,
        the event is simply not registered.
        """
        generate_claude_code_config(tmp_path, detected_python)
        settings_path = tmp_path / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        assert "UserPromptSubmit" not in data["hooks"]

    def test_subagent_start_not_registered(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """SubagentStart is prompt-only → skipped to avoid gate lockups."""
        generate_claude_code_config(tmp_path, detected_python)
        settings_path = tmp_path / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        assert "SubagentStart" not in data["hooks"]

    def test_pre_compact_not_registered(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """PreCompact is prompt-only → skipped."""
        generate_claude_code_config(tmp_path, detected_python)
        settings_path = tmp_path / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        assert "PreCompact" not in data["hooks"]

    def test_post_tool_use_has_no_prompt(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """PostToolUse must emit no prompt-type hooks."""
        generate_claude_code_config(tmp_path, detected_python)
        settings_path = tmp_path / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        post = data["hooks"]["PostToolUse"]
        hook_types = [h["type"] for h in post[0]["hooks"]]
        assert "prompt" not in hook_types

    def test_stop_is_command_only_comprehensive(
        self, tmp_path: Path, detected_python: DetectedProject
    ) -> None:
        """Stop hook must be command-only even at COMPREHENSIVE stringency."""
        generate_claude_code_config(tmp_path, detected_python)
        settings_path = tmp_path / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        stop = data["hooks"]["Stop"]
        hook_types = [h["type"] for h in stop[0]["hooks"]]
        assert hook_types == ["command"]
