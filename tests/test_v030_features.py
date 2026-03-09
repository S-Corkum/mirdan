"""Tests for v0.3.0 features: comprehensive hooks, rules, skills, agents.

Tests cover:
- HookStringency enum and generate_claude_code_hooks method
- UserPromptSubmit event generation
- Rules content via template generators
- Modernized skills with context:fork and dynamic context
- Specialized agents with PROACTIVELY in descriptions
- Config quality_profile field
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mirdan.integrations.hook_templates import (
    STRINGENCY_EVENTS,
    HookStringency,
    HookTemplateGenerator,
)

# ---------------------------------------------------------------------------
# HookStringency and generate_claude_code_hooks
# ---------------------------------------------------------------------------


class TestHookStringency:
    """Tests for HookStringency enum and stringency-based generation."""

    def test_enum_values(self) -> None:
        assert HookStringency.MINIMAL.value == "minimal"
        assert HookStringency.STANDARD.value == "standard"
        assert HookStringency.COMPREHENSIVE.value == "comprehensive"

    def test_minimal_has_two_events(self) -> None:
        events = STRINGENCY_EVENTS[HookStringency.MINIMAL]
        assert len(events) == 2
        assert "PostToolUse" in events
        assert "Stop" in events

    def test_standard_has_five_events(self) -> None:
        events = STRINGENCY_EVENTS[HookStringency.STANDARD]
        assert len(events) == 5
        assert "UserPromptSubmit" in events
        assert "PreToolUse" in events
        assert "PostToolUse" in events
        assert "Stop" in events
        assert "SubagentStart" in events

    def test_comprehensive_events(self) -> None:
        events = STRINGENCY_EVENTS[HookStringency.COMPREHENSIVE]
        assert "UserPromptSubmit" in events
        assert "PreToolUse" in events
        assert "PostToolUse" in events
        assert "Stop" in events
        assert "PreCompact" in events
        assert "SubagentStart" in events


class TestGenerateClaudeCodeHooks:
    """Tests for generate_claude_code_hooks method."""

    def test_minimal_generates_two_hooks(self) -> None:
        generator = HookTemplateGenerator()
        result = generator.generate_claude_code_hooks(HookStringency.MINIMAL)
        assert len(result["hooks"]) == 2

    def test_standard_generates_five_hooks(self) -> None:
        generator = HookTemplateGenerator()
        result = generator.generate_claude_code_hooks(HookStringency.STANDARD)
        assert len(result["hooks"]) == 5

    def test_comprehensive_is_default(self) -> None:
        generator = HookTemplateGenerator()
        result = generator.generate_claude_code_hooks()
        assert "UserPromptSubmit" in result["hooks"]
        assert "PreCompact" in result["hooks"]

    def test_all_hooks_have_valid_type(self) -> None:
        generator = HookTemplateGenerator()
        result = generator.generate_claude_code_hooks(HookStringency.COMPREHENSIVE)
        for event_name, entries in result["hooks"].items():
            for entry in entries:
                for hook in entry.get("hooks", []):
                    assert hook["type"] in ("prompt", "command"), (
                        f"{event_name} has invalid hook type: {hook['type']}"
                    )


class TestUserPromptSubmit:
    """Tests for UserPromptSubmit event generator."""

    def test_generated_in_standard(self) -> None:
        generator = HookTemplateGenerator()
        result = generator.generate_claude_code_hooks(HookStringency.STANDARD)
        assert "UserPromptSubmit" in result["hooks"]

    def test_mentions_enhance_prompt(self) -> None:
        generator = HookTemplateGenerator()
        result = generator.generate_claude_code_hooks(HookStringency.STANDARD)
        ups = result["hooks"]["UserPromptSubmit"]
        prompt = ups[0]["hooks"][0]["prompt"]
        assert "enhance_prompt" in prompt

    def test_mentions_quality_standards(self) -> None:
        generator = HookTemplateGenerator()
        result = generator.generate_claude_code_hooks(HookStringency.STANDARD)
        ups = result["hooks"]["UserPromptSubmit"]
        prompt = ups[0]["hooks"][0]["prompt"]
        assert "quality_standards" in prompt


# ---------------------------------------------------------------------------
# Rules content validation (using templates from package)
# ---------------------------------------------------------------------------


class TestRulesContent:
    """Tests for the content of rule templates."""

    @pytest.fixture
    def rule_templates(self) -> dict[str, str]:
        """Load rule templates from package resources."""
        from importlib.resources import files as pkg_files

        templates: dict[str, str] = {}
        try:
            pkg = pkg_files("mirdan.integrations.templates.claude_code")
            for item in pkg.iterdir():
                if item.name.endswith(".md") and not item.is_dir():
                    templates[item.name] = item.read_text()
        except (ModuleNotFoundError, FileNotFoundError, TypeError):
            pytest.skip("Templates package not available")
        return templates

    def test_quality_rule_mentions_ai_rules(self, rule_templates: dict[str, str]) -> None:
        content = rule_templates.get("mirdan-quality.md", "")
        assert "AI001" in content
        assert "AI008" in content

    def test_security_rule_mentions_sec_rules(self, rule_templates: dict[str, str]) -> None:
        content = rule_templates.get("mirdan-security.md", "")
        assert "SEC001" in content
        assert "SEC013" in content


# ---------------------------------------------------------------------------
# Skills modernization
# ---------------------------------------------------------------------------


class TestModernizedSkills:
    """Tests for modernized skill SKILL.md files."""

    @pytest.fixture
    def skills_dir(self) -> Path:
        """Get skills template directory from package."""
        from importlib.resources import files as pkg_files

        try:
            pkg = pkg_files("mirdan.integrations.templates.claude_code")
            skills = pkg / "skills"
            return Path(str(skills))
        except (ModuleNotFoundError, FileNotFoundError, TypeError):
            pytest.skip("Templates package not available")

    def test_code_skill_has_model_inherit(self, skills_dir: Path) -> None:
        content = (skills_dir / "code" / "SKILL.md").read_text()
        assert "model: inherit" in content

    def test_code_skill_has_dynamic_context(self, skills_dir: Path) -> None:
        content = (skills_dir / "code" / "SKILL.md").read_text()
        assert "git diff --stat" in content

    def test_code_skill_references_current_tools(self, skills_dir: Path) -> None:
        content = (skills_dir / "code" / "SKILL.md").read_text()
        assert "mcp__mirdan__enhance_prompt" in content
        assert "mcp__mirdan__validate_code_quality" in content

    def test_plan_skill_has_context_fork(self, skills_dir: Path) -> None:
        content = (skills_dir / "plan" / "SKILL.md").read_text()
        assert "context: fork" in content

    def test_plan_skill_has_config_context(self, skills_dir: Path) -> None:
        content = (skills_dir / "plan" / "SKILL.md").read_text()
        assert ".mirdan/config.yaml" in content

    def test_quality_skill_has_context_fork(self, skills_dir: Path) -> None:
        content = (skills_dir / "quality" / "SKILL.md").read_text()
        assert "context: fork" in content

    def test_review_skill_has_context_fork(self, skills_dir: Path) -> None:
        content = (skills_dir / "review" / "SKILL.md").read_text()
        assert "context: fork" in content

    def test_debug_skill_references_current_tools(self, skills_dir: Path) -> None:
        content = (skills_dir / "debug" / "SKILL.md").read_text()
        assert "mcp__mirdan__enhance_prompt" in content

    def test_all_skills_exist(self, skills_dir: Path) -> None:
        for skill in ("code", "debug", "plan", "quality", "review"):
            assert (skills_dir / skill / "SKILL.md").exists()


# ---------------------------------------------------------------------------
# Specialized agents
# ---------------------------------------------------------------------------


class TestSpecializedAgents:
    """Tests for the 5 specialized agents."""

    @pytest.fixture
    def agents_dir(self) -> Path:
        """Get agents template directory from package."""
        from importlib.resources import files as pkg_files

        try:
            pkg = pkg_files("mirdan.integrations.templates.claude_code")
            agents = pkg / "agents"
            return Path(str(agents))
        except (ModuleNotFoundError, FileNotFoundError, TypeError):
            pytest.skip("Templates package not available")

    def test_all_agents_exist(self, agents_dir: Path) -> None:
        expected = {
            "quality-gate.md",
            "security-audit.md",
            "architecture-reviewer.md",
            "convention-check.md",
            "test-quality.md",
        }
        actual = {p.name for p in agents_dir.iterdir() if p.suffix == ".md"}
        assert expected == actual

    def test_security_scanner_is_proactive(self, agents_dir: Path) -> None:
        content = (agents_dir / "security-audit.md").read_text()
        assert "PROACTIVELY" in content

    def test_ai_slop_detector_is_proactive(self, agents_dir: Path) -> None:
        content = (agents_dir / "convention-check.md").read_text()
        assert "PROACTIVELY" in content

    def test_architecture_reviewer_uses_sonnet(self, agents_dir: Path) -> None:
        content = (agents_dir / "architecture-reviewer.md").read_text()
        assert "model: sonnet" in content

    def test_quality_validator_no_background(self, agents_dir: Path) -> None:
        """background: is not a valid Claude Code agent field — must be absent."""
        content = (agents_dir / "quality-gate.md").read_text()
        assert "background:" not in content

    def test_quality_validator_no_memory(self, agents_dir: Path) -> None:
        """memory: is not a valid Claude Code agent field — must be absent."""
        content = (agents_dir / "quality-gate.md").read_text()
        assert "memory:" not in content

    def test_test_auditor_no_memory(self, agents_dir: Path) -> None:
        """memory: is not a valid Claude Code agent field — must be absent."""
        content = (agents_dir / "test-quality.md").read_text()
        assert "memory:" not in content

    def test_agents_reference_current_tools(self, agents_dir: Path) -> None:
        for agent_file in agents_dir.iterdir():
            if agent_file.suffix == ".md":
                content = agent_file.read_text()
                assert "validate_code_quality" in content


# ---------------------------------------------------------------------------
# Plugin export validation
# ---------------------------------------------------------------------------


class TestPluginExport:
    """Tests for the plugin export function."""

    def test_export_creates_plugin_json(self, tmp_path: Path) -> None:
        from mirdan.integrations.claude_code import export_plugin

        export_plugin(tmp_path)
        manifest_path = tmp_path / ".claude-plugin" / "plugin.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["name"] == "mirdan"

    def test_export_version_matches_package(self, tmp_path: Path) -> None:
        from mirdan import __version__
        from mirdan.integrations.claude_code import export_plugin

        export_plugin(tmp_path)
        manifest = json.loads((tmp_path / ".claude-plugin" / "plugin.json").read_text())
        assert manifest["version"] == __version__

    def test_export_creates_skills(self, tmp_path: Path) -> None:
        from mirdan.integrations.claude_code import export_plugin

        export_plugin(tmp_path)
        for skill in ("code", "debug", "plan", "quality", "review"):
            assert (tmp_path / "skills" / skill / "SKILL.md").exists()

    def test_export_creates_agents(self, tmp_path: Path) -> None:
        from mirdan.integrations.claude_code import export_plugin

        export_plugin(tmp_path)
        agents_dir = tmp_path / "agents"
        assert agents_dir.exists()
        md_files = list(agents_dir.glob("*.md"))
        assert len(md_files) == 5


# ---------------------------------------------------------------------------
# Hooks.json validation (via generator, not static file)
# ---------------------------------------------------------------------------


class TestPluginHooks:
    """Tests for hooks.json generated by HookTemplateGenerator."""

    @pytest.fixture
    def hooks(self) -> dict[str, Any]:
        generator = HookTemplateGenerator()
        result = generator.generate_claude_code_hooks(HookStringency.COMPREHENSIVE)
        return result["hooks"]  # type: ignore[no-any-return]

    def test_has_core_events(self, hooks: dict[str, Any]) -> None:
        assert "UserPromptSubmit" in hooks
        assert "PreToolUse" in hooks
        assert "PostToolUse" in hooks
        assert "Stop" in hooks
        assert "PreCompact" in hooks
        assert "SubagentStart" in hooks

    def test_post_tool_use_matches_multiedit(self, hooks: dict[str, Any]) -> None:
        matcher = hooks["PostToolUse"][0]["matcher"]
        assert "MultiEdit" in matcher


# ---------------------------------------------------------------------------
# Config quality_profile field
# ---------------------------------------------------------------------------


class TestConfigQualityProfile:
    """Tests for the quality_profile field in MirdanConfig."""

    def test_default_profile_is_default(self) -> None:
        from mirdan.config import MirdanConfig

        config = MirdanConfig()
        assert config.quality_profile == "default"

    def test_custom_profiles_empty_by_default(self) -> None:
        from mirdan.config import MirdanConfig

        config = MirdanConfig()
        assert config.custom_profiles == {}

    def test_config_accepts_profile_name(self) -> None:
        from mirdan.config import MirdanConfig

        config = MirdanConfig(quality_profile="enterprise")
        assert config.quality_profile == "enterprise"
