"""Tests for v0.3.0 features: comprehensive hooks, rules, skills, agents.

Tests cover:
- HookStringency enum and generate_claude_code_hooks method
- UserPromptSubmit event generation
- Rules file content and path-scoped frontmatter
- Modernized skills with context:fork and dynamic context
- Specialized agents with PROACTIVELY in descriptions
- Config quality_profile field
"""

from __future__ import annotations

import json
from pathlib import Path

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

    def test_comprehensive_has_six_events(self) -> None:
        events = STRINGENCY_EVENTS[HookStringency.COMPREHENSIVE]
        assert len(events) == 6
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

    def test_comprehensive_generates_six_hooks(self) -> None:
        generator = HookTemplateGenerator()
        result = generator.generate_claude_code_hooks(HookStringency.COMPREHENSIVE)
        assert len(result["hooks"]) == 6

    def test_comprehensive_is_default(self) -> None:
        generator = HookTemplateGenerator()
        result = generator.generate_claude_code_hooks()
        assert "UserPromptSubmit" in result["hooks"]
        assert "PreCompact" in result["hooks"]

    def test_all_hooks_use_prompt_type(self) -> None:
        generator = HookTemplateGenerator()
        result = generator.generate_claude_code_hooks(HookStringency.COMPREHENSIVE)
        for event_name, entries in result["hooks"].items():
            for entry in entries:
                for hook in entry.get("hooks", []):
                    assert hook["type"] == "prompt", f"{event_name} has non-prompt hook"


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
# Rules content validation
# ---------------------------------------------------------------------------


class TestRulesContent:
    """Tests for the content of generated rules files."""

    def test_always_rule_no_frontmatter(self) -> None:
        """mirdan-always.md should NOT have YAML frontmatter."""
        content = Path(
            "/Users/seancorkum/projects/ai-assistance/mirdan-claude-code/rules/mirdan-always.md"
        ).read_text()
        assert not content.startswith("---")

    def test_always_rule_mentions_ai_rules(self) -> None:
        content = Path(
            "/Users/seancorkum/projects/ai-assistance/mirdan-claude-code/rules/mirdan-always.md"
        ).read_text()
        assert "AI001" in content
        assert "AI008" in content

    def test_security_rule_has_frontmatter(self) -> None:
        """mirdan-security.md should have paths: frontmatter."""
        content = Path(
            "/Users/seancorkum/projects/ai-assistance/mirdan-claude-code/rules/mirdan-security.md"
        ).read_text()
        assert content.startswith("---")
        assert "paths:" in content
        assert "**/auth*" in content

    def test_security_rule_mentions_sec_rules(self) -> None:
        content = Path(
            "/Users/seancorkum/projects/ai-assistance/mirdan-claude-code/rules/mirdan-security.md"
        ).read_text()
        assert "SEC001" in content
        assert "SEC013" in content

    def test_ai_quality_rule_has_frontmatter(self) -> None:
        content = Path(
            "/Users/seancorkum/projects/ai-assistance/mirdan-claude-code/rules/mirdan-ai-quality.md"
        ).read_text()
        assert content.startswith("---")
        assert "paths:" in content
        assert "**/*.py" in content
        assert "**/*.ts" in content

    def test_ai_quality_rule_mentions_all_ai_rules(self) -> None:
        content = Path(
            "/Users/seancorkum/projects/ai-assistance/mirdan-claude-code/rules/mirdan-ai-quality.md"
        ).read_text()
        for i in range(1, 9):
            assert f"AI00{i}" in content


# ---------------------------------------------------------------------------
# Skills modernization
# ---------------------------------------------------------------------------


class TestModernizedSkills:
    """Tests for modernized skill SKILL.md files."""

    @pytest.fixture
    def skills_dir(self) -> Path:
        return Path("/Users/seancorkum/projects/ai-assistance/mirdan-claude-code/skills")

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
        # Should NOT reference removed tools
        assert "get_verification_checklist" not in content

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
        # Should NOT reference removed tools
        assert "analyze_intent" not in content

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
        return Path("/Users/seancorkum/projects/ai-assistance/mirdan-claude-code/agents")

    def test_all_agents_exist(self, agents_dir: Path) -> None:
        expected = {
            "quality-validator.md",
            "security-scanner.md",
            "architecture-reviewer.md",
            "ai-slop-detector.md",
            "test-auditor.md",
        }
        actual = {p.name for p in agents_dir.iterdir() if p.suffix == ".md"}
        assert expected == actual

    def test_security_scanner_is_proactive(self, agents_dir: Path) -> None:
        content = (agents_dir / "security-scanner.md").read_text()
        assert "PROACTIVELY" in content

    def test_ai_slop_detector_is_proactive(self, agents_dir: Path) -> None:
        content = (agents_dir / "ai-slop-detector.md").read_text()
        assert "PROACTIVELY" in content

    def test_architecture_reviewer_uses_sonnet(self, agents_dir: Path) -> None:
        content = (agents_dir / "architecture-reviewer.md").read_text()
        assert "model: sonnet" in content

    def test_quality_validator_has_background(self, agents_dir: Path) -> None:
        content = (agents_dir / "quality-validator.md").read_text()
        assert "background: true" in content

    def test_quality_validator_has_memory(self, agents_dir: Path) -> None:
        content = (agents_dir / "quality-validator.md").read_text()
        assert "memory: project" in content

    def test_test_auditor_has_memory(self, agents_dir: Path) -> None:
        content = (agents_dir / "test-auditor.md").read_text()
        assert "memory: project" in content

    def test_agents_reference_current_tools(self, agents_dir: Path) -> None:
        for agent_file in agents_dir.iterdir():
            if agent_file.suffix == ".md":
                content = agent_file.read_text()
                assert "validate_code_quality" in content


# ---------------------------------------------------------------------------
# Plugin manifest validation
# ---------------------------------------------------------------------------


class TestPluginManifest:
    """Tests for the updated plugin manifest."""

    @pytest.fixture
    def plugin_dir(self) -> Path:
        return Path("/Users/seancorkum/projects/ai-assistance/mirdan-claude-code")

    def test_version_is_030(self, plugin_dir: Path) -> None:
        manifest = json.loads((plugin_dir / ".claude-plugin" / "plugin.json").read_text())
        assert manifest["version"] == "0.3.0"

    def test_provides_rules(self, plugin_dir: Path) -> None:
        manifest = json.loads((plugin_dir / ".claude-plugin" / "plugin.json").read_text())
        assert manifest["provides"].get("rules") is True

    def test_package_version_is_030(self, plugin_dir: Path) -> None:
        pkg = json.loads((plugin_dir / "package.json").read_text())
        assert pkg["version"] == "0.3.0"

    def test_package_includes_rules(self, plugin_dir: Path) -> None:
        pkg = json.loads((plugin_dir / "package.json").read_text())
        assert "rules/" in pkg["files"]

    def test_settings_has_current_tools(self, plugin_dir: Path) -> None:
        settings = json.loads((plugin_dir / "settings.json").read_text())
        tools = settings["permissions"]["allow"]
        assert "mcp__mirdan__enhance_prompt" in tools
        assert "mcp__mirdan__validate_code_quality" in tools
        assert "mcp__mirdan__validate_quick" in tools
        assert "mcp__mirdan__get_quality_standards" in tools
        assert "mcp__mirdan__get_quality_trends" in tools
        assert len(tools) == 5

    def test_settings_no_removed_tools(self, plugin_dir: Path) -> None:
        settings = json.loads((plugin_dir / "settings.json").read_text())
        tools = settings["permissions"]["allow"]
        # Should NOT have removed tools
        assert "mcp__mirdan__get_verification_checklist" not in tools
        assert "mcp__mirdan__analyze_intent" not in tools
        assert "mcp__mirdan__suggest_tools" not in tools
        assert "mcp__mirdan__validate_plan_quality" not in tools
        assert "mcp__mirdan__validate_diff" not in tools


# ---------------------------------------------------------------------------
# Hooks.json validation
# ---------------------------------------------------------------------------


class TestPluginHooks:
    """Tests for the plugin hooks.json."""

    @pytest.fixture
    def hooks(self) -> dict:
        hooks_path = Path(
            "/Users/seancorkum/projects/ai-assistance/mirdan-claude-code/hooks/hooks.json"
        )
        data = json.loads(hooks_path.read_text())
        return data["hooks"]

    def test_has_six_events(self, hooks: dict) -> None:
        assert len(hooks) == 6

    def test_has_user_prompt_submit(self, hooks: dict) -> None:
        assert "UserPromptSubmit" in hooks

    def test_has_pre_tool_use(self, hooks: dict) -> None:
        assert "PreToolUse" in hooks

    def test_has_post_tool_use(self, hooks: dict) -> None:
        assert "PostToolUse" in hooks

    def test_has_stop(self, hooks: dict) -> None:
        assert "Stop" in hooks

    def test_has_pre_compact(self, hooks: dict) -> None:
        assert "PreCompact" in hooks

    def test_has_subagent_start(self, hooks: dict) -> None:
        assert "SubagentStart" in hooks

    def test_post_tool_use_matches_multiedit(self, hooks: dict) -> None:
        matcher = hooks["PostToolUse"][0]["matcher"]
        assert "MultiEdit" in matcher

    def test_all_hooks_are_prompt_type(self, hooks: dict) -> None:
        for event, entries in hooks.items():
            for entry in entries:
                for hook in entry.get("hooks", []):
                    assert hook["type"] == "prompt", f"{event} has non-prompt hook"


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
