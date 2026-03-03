"""Tests for Claude Code integration module.

Tests cover:
- generate_claude_code_config() creates hooks.json with correct structure
- hooks.json PostToolUse uses --quick flag
- Rule files are generated in .claude/rules/
- Hooks.json not overwritten if already exists
"""

from __future__ import annotations

import json

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
# hooks.json generation
# ---------------------------------------------------------------------------


class TestHooksGeneration:
    """Tests for hooks.json generation."""

    def test_creates_hooks_json(self, tmp_path, detected_python) -> None:
        """Should create .claude/hooks.json."""
        generated = generate_claude_code_config(tmp_path, detected_python)
        hooks_path = tmp_path / ".claude" / "hooks.json"
        assert hooks_path.exists()
        assert hooks_path in generated

    def test_hooks_has_post_tool_use(self, tmp_path, detected_python) -> None:
        """hooks.json should have PostToolUse entry."""
        generate_claude_code_config(tmp_path, detected_python)
        hooks_path = tmp_path / ".claude" / "hooks.json"
        data = json.loads(hooks_path.read_text())
        assert "PostToolUse" in data["hooks"]

    def test_hooks_has_stop(self, tmp_path, detected_python) -> None:
        """hooks.json should have Stop entry for final quality check."""
        generate_claude_code_config(tmp_path, detected_python)
        hooks_path = tmp_path / ".claude" / "hooks.json"
        data = json.loads(hooks_path.read_text())
        assert "Stop" in data["hooks"]

    def test_post_tool_use_uses_prompt(self, tmp_path, detected_python) -> None:
        """PostToolUse should use prompt-type hook for validation."""
        generate_claude_code_config(tmp_path, detected_python)
        hooks_path = tmp_path / ".claude" / "hooks.json"
        data = json.loads(hooks_path.read_text())
        ptu_hook = data["hooks"]["PostToolUse"][0]["hooks"][0]
        assert ptu_hook["type"] == "prompt"
        assert "validate_code_quality" in ptu_hook["prompt"]

    def test_stop_uses_prompt(self, tmp_path, detected_python) -> None:
        """Stop hook should use prompt-type for verification gate."""
        generate_claude_code_config(tmp_path, detected_python)
        hooks_path = tmp_path / ".claude" / "hooks.json"
        data = json.loads(hooks_path.read_text())
        stop_hook = data["hooks"]["Stop"][0]["hooks"][0]
        assert stop_hook["type"] == "prompt"
        assert "validate" in stop_hook["prompt"].lower()

    def test_post_tool_use_matcher(self, tmp_path, detected_python) -> None:
        """PostToolUse should match Write|Edit|MultiEdit tools."""
        generate_claude_code_config(tmp_path, detected_python)
        hooks_path = tmp_path / ".claude" / "hooks.json"
        data = json.loads(hooks_path.read_text())
        matcher = data["hooks"]["PostToolUse"][0]["matcher"]
        assert matcher == "Write|Edit|MultiEdit"

    def test_hooks_not_overwritten(self, tmp_path, detected_python) -> None:
        """Existing hooks.json should not be overwritten."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        hooks_path = claude_dir / "hooks.json"
        hooks_path.write_text('{"custom": true}')

        generated = generate_claude_code_config(tmp_path, detected_python)

        # hooks.json should not be in generated list
        assert hooks_path not in generated
        # Content should be preserved
        assert json.loads(hooks_path.read_text()) == {"custom": True}


# ---------------------------------------------------------------------------
# Rule files generation
# ---------------------------------------------------------------------------


class TestRulesGeneration:
    """Tests for .claude/rules/ generation."""

    def test_creates_rules_directory(self, tmp_path, detected_python) -> None:
        """Should create .claude/rules/ directory."""
        generate_claude_code_config(tmp_path, detected_python)
        rules_dir = tmp_path / ".claude" / "rules"
        assert rules_dir.is_dir()

    def test_generates_quality_rule(self, tmp_path, detected_python) -> None:
        """Should generate mirdan-quality.md rule file."""
        generate_claude_code_config(tmp_path, detected_python)
        quality_path = tmp_path / ".claude" / "rules" / "mirdan-quality.md"
        assert quality_path.exists()
        content = quality_path.read_text()
        assert "mirdan" in content.lower()

    def test_generates_security_rule(self, tmp_path, detected_python) -> None:
        """Should generate mirdan-security.md rule file."""
        generate_claude_code_config(tmp_path, detected_python)
        security_path = tmp_path / ".claude" / "rules" / "mirdan-security.md"
        assert security_path.exists()
        content = security_path.read_text()
        assert "security" in content.lower()

    def test_generates_python_rule(self, tmp_path, detected_python) -> None:
        """Should generate mirdan-python.md for Python projects."""
        generate_claude_code_config(tmp_path, detected_python)
        python_path = tmp_path / ".claude" / "rules" / "mirdan-python.md"
        assert python_path.exists()
        content = python_path.read_text()
        assert "python" in content.lower()

    def test_generates_typescript_rule(self, tmp_path, detected_typescript) -> None:
        """Should generate mirdan-typescript.md for TypeScript projects."""
        generate_claude_code_config(tmp_path, detected_typescript)
        ts_path = tmp_path / ".claude" / "rules" / "mirdan-typescript.md"
        assert ts_path.exists()
        content = ts_path.read_text()
        assert "typescript" in content.lower()

    def test_no_python_rule_for_typescript(self, tmp_path, detected_typescript) -> None:
        """Should NOT generate mirdan-python.md for TypeScript projects."""
        generate_claude_code_config(tmp_path, detected_typescript)
        python_path = tmp_path / ".claude" / "rules" / "mirdan-python.md"
        assert not python_path.exists()

    def test_rules_are_plain_markdown(self, tmp_path, detected_python) -> None:
        """Rule files should be plain markdown (no YAML frontmatter)."""
        generate_claude_code_config(tmp_path, detected_python)
        quality_path = tmp_path / ".claude" / "rules" / "mirdan-quality.md"
        content = quality_path.read_text()
        # Should NOT start with YAML frontmatter (---)
        assert not content.startswith("---")

    def test_rules_overwritten_on_regeneration(self, tmp_path, detected_python) -> None:
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

    def test_returns_all_generated_paths(self, tmp_path, detected_python) -> None:
        """Should return all generated file paths."""
        generated = generate_claude_code_config(tmp_path, detected_python)
        # hooks.json + quality + security + python = 4 files
        assert len(generated) >= 4

    def test_all_paths_exist(self, tmp_path, detected_python) -> None:
        """All returned paths should exist on disk."""
        generated = generate_claude_code_config(tmp_path, detected_python)
        for path in generated:
            assert path.exists(), f"{path} does not exist"


# ---------------------------------------------------------------------------
# v0.2.0: Hook delegation, advanced events, self-managing integration
# ---------------------------------------------------------------------------


class TestHookDelegation:
    """Tests for hook generation delegating to HookTemplateGenerator."""

    def test_hooks_json_has_comprehensive_events(self, tmp_path, detected_python) -> None:
        """Hooks should include 6 comprehensive lifecycle events."""
        generate_claude_code_config(tmp_path, detected_python)
        hooks_path = tmp_path / ".claude" / "hooks.json"
        data = json.loads(hooks_path.read_text())
        hooks = data["hooks"]
        assert len(hooks) >= 6

    def test_hooks_has_user_prompt_submit(self, tmp_path, detected_python) -> None:
        """Should include UserPromptSubmit hook."""
        generate_claude_code_config(tmp_path, detected_python)
        hooks_path = tmp_path / ".claude" / "hooks.json"
        data = json.loads(hooks_path.read_text())
        assert "UserPromptSubmit" in data["hooks"]

    def test_hooks_has_subagent_start(self, tmp_path, detected_python) -> None:
        """Should include SubagentStart hook."""
        generate_claude_code_config(tmp_path, detected_python)
        hooks_path = tmp_path / ".claude" / "hooks.json"
        data = json.loads(hooks_path.read_text())
        assert "SubagentStart" in data["hooks"]

    def test_hooks_has_pre_compact(self, tmp_path, detected_python) -> None:
        """Should include PreCompact hook for compaction resilience."""
        generate_claude_code_config(tmp_path, detected_python)
        hooks_path = tmp_path / ".claude" / "hooks.json"
        data = json.loads(hooks_path.read_text())
        assert "PreCompact" in data["hooks"]

    def test_hooks_has_pre_tool_use(self, tmp_path, detected_python) -> None:
        """Should include PreToolUse hook with prompt type."""
        generate_claude_code_config(tmp_path, detected_python)
        hooks_path = tmp_path / ".claude" / "hooks.json"
        data = json.loads(hooks_path.read_text())
        pre = data["hooks"]["PreToolUse"]
        assert len(pre) > 0
        assert pre[0]["hooks"][0]["type"] == "prompt"

    def test_post_tool_use_has_prompt(self, tmp_path, detected_python) -> None:
        """PostToolUse should use prompt-type hook for validation."""
        generate_claude_code_config(tmp_path, detected_python)
        hooks_path = tmp_path / ".claude" / "hooks.json"
        data = json.loads(hooks_path.read_text())
        post = data["hooks"]["PostToolUse"]
        hook_types = [h["type"] for h in post[0]["hooks"]]
        assert "prompt" in hook_types

    def test_stop_uses_prompt_type(self, tmp_path, detected_python) -> None:
        """Stop hook should use prompt type for verification gate."""
        generate_claude_code_config(tmp_path, detected_python)
        hooks_path = tmp_path / ".claude" / "hooks.json"
        data = json.loads(hooks_path.read_text())
        stop = data["hooks"]["Stop"]
        assert stop[0]["hooks"][0]["type"] == "prompt"
        assert "validate" in stop[0]["hooks"][0]["prompt"].lower()
