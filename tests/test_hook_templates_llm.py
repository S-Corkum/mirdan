"""Tests for LLM-aware hook template generation."""

from __future__ import annotations

from pathlib import Path

from mirdan.integrations.cursor import (
    CursorHookStringency,
    generate_cursor_hooks,
    generate_cursor_llm_rule,
)
from mirdan.integrations.hook_templates import (
    HookStringency,
    HookTemplateGenerator,
)

# ---------------------------------------------------------------------------
# Claude Code hooks with LLM
# ---------------------------------------------------------------------------


class TestClaudeCodeHooksLLMEnabled:
    """Claude Code hooks when llm_enabled=True."""

    def test_user_prompt_submit_has_command_hook(self) -> None:
        gen = HookTemplateGenerator()
        result = gen.generate_claude_code_hooks(
            stringency=HookStringency.STANDARD, llm_enabled=True
        )
        hooks = result["hooks"]
        assert "UserPromptSubmit" in hooks

        ups_hooks = hooks["UserPromptSubmit"]
        # Should have command-type hook (sidecar triage script)
        hook_types = [h["type"] for entry in ups_hooks for h in entry["hooks"]]
        assert "command" in hook_types

    def test_user_prompt_submit_has_triage_prompt(self) -> None:
        gen = HookTemplateGenerator()
        result = gen.generate_claude_code_hooks(
            stringency=HookStringency.STANDARD, llm_enabled=True
        )
        ups_hooks = result["hooks"]["UserPromptSubmit"]
        prompts = [
            h["prompt"] for entry in ups_hooks for h in entry["hooks"] if h["type"] == "prompt"
        ]
        assert any("triage" in p.lower() for p in prompts)

    def test_stop_has_check_command(self) -> None:
        gen = HookTemplateGenerator()
        result = gen.generate_claude_code_hooks(
            stringency=HookStringency.STANDARD, llm_enabled=True
        )
        stop_hooks = result["hooks"]["Stop"]
        commands = [
            h["command"] for entry in stop_hooks for h in entry["hooks"] if h["type"] == "command"
        ]
        assert any("check --smart" in c for c in commands)

    def test_post_tool_use_unchanged(self) -> None:
        gen = HookTemplateGenerator()
        llm_result = gen.generate_claude_code_hooks(
            stringency=HookStringency.STANDARD, llm_enabled=True
        )
        no_llm_result = gen.generate_claude_code_hooks(
            stringency=HookStringency.STANDARD, llm_enabled=False
        )
        # PostToolUse should be the same regardless of LLM
        assert llm_result["hooks"]["PostToolUse"] == no_llm_result["hooks"]["PostToolUse"]

    def test_hook_script_has_sidecar_fallback(self) -> None:
        gen = HookTemplateGenerator()
        result = gen.generate_claude_code_hooks(
            stringency=HookStringency.STANDARD, llm_enabled=True
        )
        ups_hooks = result["hooks"]["UserPromptSubmit"]
        commands = [
            h["command"] for entry in ups_hooks for h in entry["hooks"] if h["type"] == "command"
        ]
        # Should contain sidecar.port check and mirdan triage fallback
        cmd_text = " ".join(commands)
        assert "sidecar.port" in cmd_text
        assert "mirdan triage" in cmd_text


class TestClaudeCodeHooksLLMDisabled:
    """Claude Code hooks when llm_enabled=False (backward compat)."""

    def test_user_prompt_submit_is_prompt_only(self) -> None:
        gen = HookTemplateGenerator()
        result = gen.generate_claude_code_hooks(
            stringency=HookStringency.STANDARD, llm_enabled=False
        )
        ups_hooks = result["hooks"]["UserPromptSubmit"]
        hook_types = [h["type"] for entry in ups_hooks for h in entry["hooks"]]
        assert "command" not in hook_types
        assert "prompt" in hook_types

    def test_stop_is_gate_command(self) -> None:
        gen = HookTemplateGenerator()
        result = gen.generate_claude_code_hooks(
            stringency=HookStringency.STANDARD, llm_enabled=False
        )
        stop_hooks = result["hooks"]["Stop"]
        commands = [
            h.get("command", "") for entry in stop_hooks for h in entry["hooks"]
        ]
        assert any("gate" in c for c in commands)


# ---------------------------------------------------------------------------
# Cursor hooks with LLM
# ---------------------------------------------------------------------------


class TestCursorHooksLLMEnabled:
    """Cursor hooks when llm_enabled=True."""

    def test_session_start_has_llm_prompt(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_hooks(
            cursor_dir, CursorHookStringency.COMPREHENSIVE, llm_enabled=True
        )

        import json

        hooks_data = json.loads((cursor_dir / "hooks.json").read_text())
        session_hooks = hooks_data["hooks"].get("sessionStart", [])
        prompts = [h.get("prompt", "") for h in session_hooks]
        assert any("local LLM" in p for p in prompts)

    def test_generates_mdc_rule(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_hooks(
            cursor_dir, CursorHookStringency.COMPREHENSIVE, llm_enabled=True
        )

        mdc_path = cursor_dir / "rules" / "mirdan-llm.mdc"
        assert mdc_path.exists()
        content = mdc_path.read_text()
        assert "alwaysApply: true" in content
        assert "enhance_prompt" in content

    def test_no_mdc_rule_when_disabled(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_hooks(
            cursor_dir, CursorHookStringency.COMPREHENSIVE, llm_enabled=False
        )

        mdc_path = cursor_dir / "rules" / "mirdan-llm.mdc"
        assert not mdc_path.exists()


class TestCursorHooksLLMDisabled:
    """Cursor hooks when llm_enabled=False (backward compat)."""

    def test_no_llm_session_prompt(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_hooks(
            cursor_dir, CursorHookStringency.COMPREHENSIVE, llm_enabled=False
        )

        import json

        hooks_data = json.loads((cursor_dir / "hooks.json").read_text())
        session_hooks = hooks_data["hooks"].get("sessionStart", [])
        prompts = [h.get("prompt", "") for h in session_hooks]
        assert not any("local LLM" in p for p in prompts)


# ---------------------------------------------------------------------------
# Cursor LLM rule generation
# ---------------------------------------------------------------------------


class TestCursorLLMRule:
    """Tests for generate_cursor_llm_rule()."""

    def test_creates_mdc_file(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        result = generate_cursor_llm_rule(cursor_dir)

        assert result is not None
        assert result.exists()
        content = result.read_text()
        assert "alwaysApply: true" in content
        assert "MANDATORY" in content

    def test_skips_if_exists(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        rules_dir = cursor_dir / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "mirdan-llm.mdc").write_text("custom content")

        result = generate_cursor_llm_rule(cursor_dir)
        assert result is None
        # Original content preserved
        assert (rules_dir / "mirdan-llm.mdc").read_text() == "custom content"
