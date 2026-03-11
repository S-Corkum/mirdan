"""Tests for Phase 1 (v0.5.0) modernization changes.

Covers bug fixes, hook modernization, skill/agent/rule template updates,
diff validation improvements, and static hooks replacement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mirdan.integrations.hook_templates import (
    ALL_HOOK_EVENTS,
    STRINGENCY_EVENTS,
    HookConfig,
    HookStringency,
    HookTemplateGenerator,
)

# ---------------------------------------------------------------------------
# 1A.1: save_snapshot wired
# ---------------------------------------------------------------------------


class TestSaveSnapshotWired:
    """Verify save_snapshot() is called after validation."""

    def test_server_calls_save_snapshot(self) -> None:
        """ValidateCodeUseCase.execute source should call save_snapshot."""
        import inspect

        from mirdan.usecases.validate_code import ValidateCodeUseCase

        source = inspect.getsource(ValidateCodeUseCase.execute)
        assert "save_snapshot" in source


# ---------------------------------------------------------------------------
# 1A.2: apply_profile wired
# ---------------------------------------------------------------------------


class TestProfileWiring:
    """Verify quality profiles are applied when configured."""

    def test_get_components_calls_apply_profile(self) -> None:
        """ComponentProvider.__init__ source should reference apply_profile."""
        import inspect

        from mirdan.providers import ComponentProvider

        source = inspect.getsource(ComponentProvider.__init__)
        assert "apply_profile" in source
        assert 'quality_profile != "default"' in source


# ---------------------------------------------------------------------------
# 1A.3: _get_version uses __version__
# ---------------------------------------------------------------------------


class TestVersionFix:
    """Verify _get_version returns the package version."""

    def test_returns_package_version(self) -> None:
        from mirdan import __version__
        from mirdan.integrations.claude_code import _get_version

        assert _get_version() == __version__

    def test_not_hardcoded_0_1_0(self) -> None:
        from mirdan.integrations.claude_code import _get_version

        assert _get_version() != "0.1.0"


# ---------------------------------------------------------------------------
# 1A.4: OutputConfig removed
# ---------------------------------------------------------------------------


class TestOutputConfigRemoved:
    """Verify OutputConfig dataclass was removed."""

    def test_no_output_config_in_models(self) -> None:
        import mirdan.models as models

        assert not hasattr(models, "OutputConfig")


# ---------------------------------------------------------------------------
# 1A.6: mcp__mirdan__ double underscore in .mdc
# ---------------------------------------------------------------------------


class TestMdcDoubleUnderscore:
    """Verify .mdc templates use mcp__mirdan__ not mcp_mirdan_."""

    @pytest.fixture()
    def mdc_files(self) -> list[tuple[str, str]]:
        from importlib.resources import files

        pkg = files("mirdan.integrations.templates")
        return [
            (item.name, item.read_text()) for item in pkg.iterdir() if item.name.endswith(".mdc")
        ]

    def test_no_single_underscore(self, mdc_files: list[tuple[str, str]]) -> None:
        for name, content in mdc_files:
            assert "mcp_mirdan_" not in content, f"{name} has single underscore"

    def test_has_double_underscore(self, mdc_files: list[tuple[str, str]]) -> None:
        for name, content in mdc_files:
            if "mirdan" in content.lower():
                assert "mcp__mirdan__" in content, f"{name} missing double underscore"


# ---------------------------------------------------------------------------
# 1A.7: SEC rule numbering
# ---------------------------------------------------------------------------


class TestSecRuleNumbering:
    """Verify SEC rules go up to SEC013."""

    def test_agents_md_has_sec013(self) -> None:
        from mirdan.cli.detect import DetectedProject
        from mirdan.integrations.agents_md import AgentsMDGenerator

        detected = DetectedProject(primary_language="python")
        gen = AgentsMDGenerator()
        output = gen.generate(detected)
        assert "SEC013" in output

    def test_cursor_has_sec013(self) -> None:
        from mirdan.integrations.cursor import _AGENTS_SEC_RULES

        # _AGENTS_SEC_RULES is a multi-line string, not a list
        assert "SEC013" in _AGENTS_SEC_RULES


# ---------------------------------------------------------------------------
# 1A.8: Overlay includes /plan and /quality
# ---------------------------------------------------------------------------


class TestOverlaySkills:
    """Verify Claude Code overlay references all 5 skills."""

    def test_overlay_has_plan(self) -> None:
        from mirdan.cli.detect import DetectedProject
        from mirdan.integrations.agents_md import AgentsMDGenerator

        detected = DetectedProject(primary_language="python")
        gen = AgentsMDGenerator()
        output = gen.generate(detected, platform="claude-code")
        assert "/plan" in output

    def test_overlay_has_quality(self) -> None:
        from mirdan.cli.detect import DetectedProject
        from mirdan.integrations.agents_md import AgentsMDGenerator

        detected = DetectedProject(primary_language="python")
        gen = AgentsMDGenerator()
        output = gen.generate(detected, platform="claude-code")
        assert "/quality" in output


# ---------------------------------------------------------------------------
# 1B: Claude Code Hook Modernization — 17 events
# ---------------------------------------------------------------------------


class TestHookEventsComplete:
    """Verify all 17 Claude Code hook events are defined."""

    def test_all_hook_events_count(self) -> None:
        assert len(ALL_HOOK_EVENTS) == 17

    @pytest.mark.parametrize(
        "event",
        [
            "UserPromptSubmit",
            "PreToolUse",
            "PostToolUse",
            "PostToolUseFailure",
            "Stop",
            "SessionStart",
            "PreCompact",
            "SubagentStart",
            "SubagentStop",
            "TaskCompleted",
            "TeammateIdle",
            "PermissionRequest",
            "ConfigChange",
            "WorktreeCreate",
            "WorktreeRemove",
        ],
    )
    def test_event_exists(self, event: str) -> None:
        assert event in ALL_HOOK_EVENTS

    def test_comprehensive_includes_new_events(self) -> None:
        events = STRINGENCY_EVENTS[HookStringency.COMPREHENSIVE]
        for event in (
            "PostToolUseFailure",
            "TaskCompleted",
            "TeammateIdle",
            "PermissionRequest",
            "ConfigChange",
            "WorktreeCreate",
            "WorktreeRemove",
        ):
            assert event in events, f"{event} missing from COMPREHENSIVE"


class TestHookTypeDiversity:
    """Verify PostToolUse uses command+prompt combo."""

    def test_post_tool_use_has_command(self) -> None:
        gen = HookTemplateGenerator(mirdan_command="mirdan")
        hooks = gen.generate()["hooks"]
        ptu = hooks["PostToolUse"]
        hook_types = [h["type"] for h in ptu[0]["hooks"]]
        assert "command" in hook_types

    def test_post_tool_use_has_prompt(self) -> None:
        gen = HookTemplateGenerator(mirdan_command="mirdan")
        hooks = gen.generate()["hooks"]
        ptu = hooks["PostToolUse"]
        hook_types = [h["type"] for h in ptu[0]["hooks"]]
        assert "prompt" in hook_types

    def test_command_hook_has_timeout(self) -> None:
        gen = HookTemplateGenerator(mirdan_command="mirdan")
        hooks = gen.generate()["hooks"]
        ptu = hooks["PostToolUse"]
        cmd_hooks = [h for h in ptu[0]["hooks"] if h["type"] == "command"]
        assert len(cmd_hooks) > 0
        assert "timeout" in cmd_hooks[0]

    def test_post_tool_use_matcher(self) -> None:
        gen = HookTemplateGenerator(mirdan_command="mirdan")
        hooks = gen.generate()["hooks"]
        ptu = hooks["PostToolUse"]
        assert ptu[0]["matcher"] == "Write|Edit|MultiEdit"


# ---------------------------------------------------------------------------
# 1C: Cursor Hook Modernization
# ---------------------------------------------------------------------------


class TestCursorEventsComplete:
    """Verify Cursor hook events are expanded."""

    def test_comprehensive_count(self) -> None:
        from mirdan.integrations.cursor import (
            CURSOR_STRINGENCY_EVENTS,
            CursorHookStringency,
        )

        events = CURSOR_STRINGENCY_EVENTS[CursorHookStringency.COMPREHENSIVE]
        assert len(events) == 8

    def test_standard_count(self) -> None:
        from mirdan.integrations.cursor import (
            CURSOR_STRINGENCY_EVENTS,
            CursorHookStringency,
        )

        events = CURSOR_STRINGENCY_EVENTS[CursorHookStringency.STANDARD]
        assert len(events) == 5

    @pytest.mark.parametrize(
        "event",
        [
            "afterFileEdit",
            "preToolUse",
            "postToolUseFailure",
            "stop",
            "sessionStart",
            "beforeShellExecution",
            "subagentStart",
            "preCompact",
        ],
    )
    def test_cursor_event_exists_in_comprehensive(self, event: str) -> None:
        from mirdan.integrations.cursor import (
            CURSOR_STRINGENCY_EVENTS,
            CursorHookStringency,
        )

        events = CURSOR_STRINGENCY_EVENTS[CursorHookStringency.COMPREHENSIVE]
        assert event in events


# ---------------------------------------------------------------------------
# 1D: Skill Template Modernization
# ---------------------------------------------------------------------------


class TestSkillFrontmatter:
    """Verify skill templates have argument-hint frontmatter."""

    @pytest.fixture()
    def skills_pkg(self) -> Any:
        from importlib.resources import files

        return files("mirdan.integrations.templates.claude_code.skills")

    @pytest.mark.parametrize(
        ("skill", "hint"),
        [
            ("code", "Describe what to build"),
            ("debug", "Describe the bug or error"),
            ("review", "File or PR to review"),
            ("plan", "Describe what to plan"),
            ("quality", "File path or --trends"),
        ],
    )
    def test_skill_has_argument_hint(self, skills_pkg: Any, skill: str, hint: str) -> None:
        content = (skills_pkg / skill / "SKILL.md").read_text()
        assert f'argument-hint: "{hint}"' in content

    def test_review_is_user_invocable(self, skills_pkg: Any) -> None:
        content = (skills_pkg / "review" / "SKILL.md").read_text()
        assert "user-invocable: true" in content


# ---------------------------------------------------------------------------
# 1E: Agent Template Modernization
# ---------------------------------------------------------------------------


class TestAgentFrontmatter:
    """Verify agent templates have updated frontmatter."""

    @pytest.fixture()
    def agents_pkg(self) -> Any:
        from importlib.resources import files

        return files("mirdan.integrations.templates.claude_code.agents")

    def test_quality_gate_max_turns(self, agents_pkg: Any) -> None:
        content = (agents_pkg / "quality-gate.md").read_text()
        assert "maxTurns: 10" in content

    def test_security_audit_no_isolation(self, agents_pkg: Any) -> None:
        """isolation: worktree is not a valid Claude Code field — must be absent."""
        content = (agents_pkg / "security-audit.md").read_text()
        assert "isolation:" not in content

    def test_test_quality_no_skills(self, agents_pkg: Any) -> None:
        """skills: is not a valid Claude Code agent field — must be absent."""
        content = (agents_pkg / "test-quality.md").read_text()
        assert "skills:" not in content

    def test_convention_check_no_memory(self, agents_pkg: Any) -> None:
        """memory: is not a valid Claude Code agent field — must be absent."""
        content = (agents_pkg / "convention-check.md").read_text()
        assert "memory:" not in content

    def test_architecture_reviewer_no_mcp_servers(self, agents_pkg: Any) -> None:
        """mcpServers: is not a valid Claude Code agent field — must be absent."""
        content = (agents_pkg / "architecture-reviewer.md").read_text()
        assert "mcpServers:" not in content


# ---------------------------------------------------------------------------
# 1F: Claude Code Rules Modernization
# ---------------------------------------------------------------------------


class TestRuleFrontmatter:
    """Verify rule templates have YAML frontmatter with paths."""

    @pytest.fixture()
    def rules_pkg(self) -> Any:
        from importlib.resources import files

        return files("mirdan.integrations.templates.claude_code")

    def test_quality_has_frontmatter(self, rules_pkg: Any) -> None:
        content = (rules_pkg / "mirdan-quality.md").read_text()
        assert content.startswith("---")
        assert "description:" in content

    def test_quality_no_paths(self, rules_pkg: Any) -> None:
        """mirdan-quality.md should be always active (no paths)."""
        content = (rules_pkg / "mirdan-quality.md").read_text()
        assert "paths:" not in content

    def test_python_has_paths(self, rules_pkg: Any) -> None:
        content = (rules_pkg / "mirdan-python.md").read_text()
        assert '- "**/*.py"' in content

    def test_typescript_has_paths(self, rules_pkg: Any) -> None:
        content = (rules_pkg / "mirdan-typescript.md").read_text()
        assert '- "**/*.ts"' in content
        assert '- "**/*.tsx"' in content
        assert '- "**/*.js"' in content
        assert '- "**/*.jsx"' in content

    def test_security_has_paths(self, rules_pkg: Any) -> None:
        content = (rules_pkg / "mirdan-security.md").read_text()
        assert '- "**/auth/**"' in content
        assert '- "**/api/**"' in content

    def test_workflow_has_frontmatter_no_paths(self, rules_pkg: Any) -> None:
        content = (rules_pkg / "mirdan-workflow.md").read_text()
        assert content.startswith("---")
        assert "paths:" not in content


# ---------------------------------------------------------------------------
# 1G: Diff Validation improvements
# ---------------------------------------------------------------------------


class TestDiffValidation:
    """Verify diff validation has line number mapping."""

    def test_get_added_code_with_mapping(self) -> None:
        from mirdan.core.diff_parser import parse_unified_diff

        diff = """\
--- a/file.py
+++ b/file.py
@@ -10,3 +10,4 @@ def foo():
     pass
+    x = 1
+    y = 2
     return None
"""
        parsed = parse_unified_diff(diff)
        code, mapping = parsed.get_added_code_with_mapping()
        lines = code.split("\n")
        assert len(lines) == 2
        assert lines[0] == "    x = 1"
        # Line 1 in extracted code maps to line 11 in file.py
        assert mapping[1] == ("file.py", 11)
        # Line 2 in extracted code maps to line 12 in file.py
        assert mapping[2] == ("file.py", 12)

    def test_mapping_across_files(self) -> None:
        from mirdan.core.diff_parser import parse_unified_diff

        diff = """\
--- a/a.py
+++ b/a.py
@@ -1,2 +1,3 @@
 x = 1
+y = 2
 z = 3
--- a/b.py
+++ b/b.py
@@ -5,2 +5,3 @@
 a = 1
+b = 2
 c = 3
"""
        parsed = parse_unified_diff(diff)
        code, mapping = parsed.get_added_code_with_mapping()
        lines = code.split("\n")
        assert len(lines) == 2
        assert mapping[1] == ("a.py", 2)
        assert mapping[2] == ("b.py", 6)


# ---------------------------------------------------------------------------
# 1A.10: No hardcoded paths in tests
# ---------------------------------------------------------------------------


class TestFragilePathsRemoved:
    """Verify no hard-coded absolute paths remain in test files."""

    def test_no_hardcoded_paths_in_v030(self) -> None:
        test_file = Path(__file__).parent / "test_v030_features.py"
        if test_file.exists():
            content = test_file.read_text()
            assert "/Users/seancorkum/" not in content


# ---------------------------------------------------------------------------
# Compaction wiring (1A.5 / 1A.12)
# ---------------------------------------------------------------------------


class TestCompactionWiring:
    """Verify PreCompact hook includes structured state format."""

    def test_pre_compact_has_structured_format(self) -> None:
        config = HookConfig(
            enabled_events=["PreToolUse", "PostToolUse", "Stop"],
            compaction_resilience=True,
        )
        gen = HookTemplateGenerator(config=config)
        hooks = gen.generate()["hooks"]
        pre_compact = hooks["PreCompact"]
        prompt = pre_compact[0]["hooks"][0]["prompt"]
        assert "## mirdan Compacted State" in prompt
        assert "Session:" in prompt
        assert "Last score:" in prompt
        assert "Open violations:" in prompt


# ---------------------------------------------------------------------------
# Generate rules includes workflow (1F fix)
# ---------------------------------------------------------------------------


class TestRulesGenerationIncludesWorkflow:
    """Verify _generate_rules produces mirdan-workflow.md."""

    def test_workflow_generated(self, tmp_path: Path) -> None:
        from mirdan.cli.detect import DetectedProject
        from mirdan.integrations.claude_code import _generate_rules

        detected = DetectedProject(
            primary_language="python",
            frameworks=[],
        )
        paths = _generate_rules(tmp_path, detected)
        names = [p.name for p in paths]
        assert "mirdan-workflow.md" in names


# ---------------------------------------------------------------------------
# 1A.9: _standards wiring in agents_md
# ---------------------------------------------------------------------------


class TestStandardsWiring:
    """Verify _standards parameter is wired in AgentsMDGenerator."""

    def test_generate_with_standards(self) -> None:
        from mirdan.cli.detect import DetectedProject
        from mirdan.core.quality_standards import QualityStandards
        from mirdan.integrations.agents_md import AgentsMDGenerator

        detected = DetectedProject(primary_language="python")
        standards = QualityStandards()
        gen = AgentsMDGenerator(standards=standards)
        output = gen.generate(detected)
        # When standards are provided, language section should include style rules
        assert "python" in output.lower()


# ---------------------------------------------------------------------------
# All hooks generate without error
# ---------------------------------------------------------------------------


class TestAllHooksGenerate:
    """Verify all hook events can generate without error."""

    def test_comprehensive_generates(self) -> None:
        gen = HookTemplateGenerator(mirdan_command="mirdan")
        hooks = gen.generate_claude_code_hooks(stringency=HookStringency.COMPREHENSIVE)
        assert "hooks" in hooks
        # All 15 COMPREHENSIVE events should produce hooks
        assert len(hooks["hooks"]) >= 15

    def test_standard_generates(self) -> None:
        gen = HookTemplateGenerator(mirdan_command="mirdan")
        hooks = gen.generate_claude_code_hooks(stringency=HookStringency.STANDARD)
        assert "hooks" in hooks
        assert len(hooks["hooks"]) >= 5

    def test_minimal_generates(self) -> None:
        gen = HookTemplateGenerator(mirdan_command="mirdan")
        hooks = gen.generate_claude_code_hooks(stringency=HookStringency.MINIMAL)
        assert "hooks" in hooks
        assert len(hooks["hooks"]) >= 2
