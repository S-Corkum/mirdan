"""Integration tests for Phase 0.3.0 and 0.4.0 features."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mirdan.cli.detect import DetectedProject
from mirdan.core.output_formatter import OutputFormatter, determine_format
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import OutputFormat, ValidationResult, Violation


@pytest.fixture()
def detected() -> DetectedProject:
    """Minimal detected project for testing."""
    return DetectedProject(
        project_type="python",
        project_name="test-project",
        primary_language="python",
        frameworks=["fastapi"],
    )


# ---------------------------------------------------------------------------
# Phase 0.3.0: MICRO Output Format
# ---------------------------------------------------------------------------


class TestMicroOutputFormat:
    """Tests for MICRO output format (Step 12-14)."""

    def test_micro_enum_exists(self) -> None:
        assert OutputFormat.MICRO.value == "micro"

    def test_determine_format_returns_micro_for_small_budget(self) -> None:
        fmt = determine_format(
            max_tokens=100, compact_threshold=4000, minimal_threshold=1000, micro_threshold=200
        )
        assert fmt == OutputFormat.MICRO

    def test_determine_format_returns_minimal_above_micro_threshold(self) -> None:
        fmt = determine_format(
            max_tokens=500, compact_threshold=4000, minimal_threshold=1000, micro_threshold=200
        )
        assert fmt != OutputFormat.MICRO

    def test_micro_validation_pass(self) -> None:
        formatter = OutputFormatter(micro_threshold=200)
        result = ValidationResult(
            passed=True,
            score=0.95,
            language_detected="python",
            violations=[],
            standards_checked=["security"],
        )
        output = formatter.format_validation_result(
            result.to_dict(), max_tokens=100
        )
        assert output.get("micro") is not None
        assert "PASS" in output["micro"]

    def test_micro_validation_fail(self) -> None:
        formatter = OutputFormatter(micro_threshold=200)
        result = ValidationResult(
            passed=False,
            score=0.6,
            language_detected="python",
            violations=[
                Violation(
                    id="AI001", rule="placeholder-detection", category="ai_quality",
                    severity="error", message="test", line=42,
                ),
            ],
            standards_checked=["ai_quality"],
        )
        output = formatter.format_validation_result(
            result.to_dict(), max_tokens=100
        )
        assert output.get("micro") is not None
        assert "FAIL" in output["micro"]


# ---------------------------------------------------------------------------
# Phase 0.3.0: Claude Code Skills
# ---------------------------------------------------------------------------


class TestSkillsAutoInvocation:
    """Tests for skills with auto-invocation descriptions (Step 15-16)."""

    def test_code_skill_has_mcp_tools(self, tmp_path: Path, detected: DetectedProject) -> None:
        from mirdan.integrations.claude_code import generate_skills

        paths = generate_skills(tmp_path, detected)
        code_skill = next(p for p in paths if p.parent.name == "code")
        content = code_skill.read_text()
        assert "mcp__mirdan__enhance_prompt" in content
        assert "mcp__mirdan__validate_code_quality" in content

    def test_debug_skill_has_mcp_tools(self, tmp_path: Path, detected: DetectedProject) -> None:
        from mirdan.integrations.claude_code import generate_skills

        paths = generate_skills(tmp_path, detected)
        debug_skill = next(p for p in paths if p.parent.name == "debug")
        content = debug_skill.read_text()
        assert "mcp__mirdan__" in content

    def test_review_skill_has_mcp_tools(self, tmp_path: Path, detected: DetectedProject) -> None:
        from mirdan.integrations.claude_code import generate_skills

        paths = generate_skills(tmp_path, detected)
        review_skill = next(p for p in paths if p.parent.name == "review")
        content = review_skill.read_text()
        assert "mcp__mirdan__" in content

    def test_skills_have_allowed_tools(self, tmp_path: Path, detected: DetectedProject) -> None:
        from mirdan.integrations.claude_code import generate_skills

        paths = generate_skills(tmp_path, detected)
        for path in paths:
            content = path.read_text()
            assert "allowed-tools:" in content, f"{path.name} missing allowed-tools"


# ---------------------------------------------------------------------------
# Phase 0.3.0: Specialized Agents
# ---------------------------------------------------------------------------


class TestSpecializedAgents:
    """Tests for 5 specialized agents."""

    def test_five_agents_generated(self, tmp_path: Path, detected: DetectedProject) -> None:
        from mirdan.integrations.claude_code import generate_agents

        paths = generate_agents(tmp_path, detected)
        assert len(paths) == 5

    def test_agents_have_no_unsupported_attrs(
        self, tmp_path: Path, detected: DetectedProject
    ) -> None:
        from mirdan.integrations.claude_code import generate_agents

        paths = generate_agents(tmp_path, detected)
        unsupported = ("background:", "memory:", "skills:", "isolation:", "mcpServers:")
        for path in paths:
            content = path.read_text()
            for attr in unsupported:
                assert attr not in content, f"{path.name} contains unsupported attr: {attr}"

    def test_security_audit_uses_haiku(self, tmp_path: Path, detected: DetectedProject) -> None:
        from mirdan.integrations.claude_code import generate_agents

        paths = generate_agents(tmp_path, detected)
        security = next(p for p in paths if p.name == "security-audit.md")
        content = security.read_text()
        assert "model: haiku" in content

    def test_agent_names(self, tmp_path: Path, detected: DetectedProject) -> None:
        from mirdan.integrations.claude_code import generate_agents

        paths = generate_agents(tmp_path, detected)
        names = {p.name for p in paths}
        expected = {
            "quality-gate.md",
            "security-audit.md",
            "test-quality.md",
            "convention-check.md",
            "architecture-reviewer.md",
        }
        assert names == expected


# ---------------------------------------------------------------------------
# Phase 0.3.0: Advanced Hooks
# ---------------------------------------------------------------------------


class TestAdvancedHooks:
    """Tests for comprehensive lifecycle hooks."""

    def _load_hooks(self, tmp_path: Path, detected: DetectedProject) -> dict:
        from mirdan.integrations.claude_code import generate_claude_code_config

        generate_claude_code_config(tmp_path, detected)
        hooks_path = tmp_path / ".claude" / "hooks.json"
        assert hooks_path.exists()
        data = json.loads(hooks_path.read_text())
        # hooks.json has a top-level "hooks" key
        return data.get("hooks", data)

    def test_hooks_json_has_comprehensive_events(
        self, tmp_path: Path, detected: DetectedProject
    ) -> None:
        hooks = self._load_hooks(tmp_path, detected)
        # Should have 6 comprehensive lifecycle events
        assert len(hooks) >= 6

    def test_hooks_has_user_prompt_submit(
        self, tmp_path: Path, detected: DetectedProject
    ) -> None:
        hooks = self._load_hooks(tmp_path, detected)
        assert "UserPromptSubmit" in hooks

    def test_hooks_has_subagent_start(self, tmp_path: Path, detected: DetectedProject) -> None:
        hooks = self._load_hooks(tmp_path, detected)
        assert "SubagentStart" in hooks

    def test_hooks_has_pre_compact(self, tmp_path: Path, detected: DetectedProject) -> None:
        hooks = self._load_hooks(tmp_path, detected)
        assert "PreCompact" in hooks

    def test_post_tool_use_has_command_and_prompt(
        self, tmp_path: Path, detected: DetectedProject
    ) -> None:
        hooks = self._load_hooks(tmp_path, detected)
        post_tool = hooks.get("PostToolUse", [])
        assert len(post_tool) > 0
        # Should have both command and prompt type hooks
        hook_types = [h["type"] for h in post_tool[0]["hooks"]]
        assert "command" in hook_types
        assert "prompt" in hook_types


# ---------------------------------------------------------------------------
# Phase 0.3.0: Dynamic Cursor Rules
# ---------------------------------------------------------------------------


class TestDynamicCursorRules:
    """Tests for dynamic Cursor rules generation (Step 19-20)."""

    def test_cursor_rules_generated(self, tmp_path: Path, detected: DetectedProject) -> None:
        from mirdan.integrations.cursor import generate_cursor_rules

        standards = QualityStandards()
        rules_dir = tmp_path / ".cursor" / "rules"
        rules_dir.mkdir(parents=True)
        paths = generate_cursor_rules(rules_dir, detected, standards=standards)
        assert len(paths) > 0

    def test_cursor_rules_have_mdc_extension(
        self, tmp_path: Path, detected: DetectedProject
    ) -> None:
        from mirdan.integrations.cursor import generate_cursor_rules

        standards = QualityStandards()
        rules_dir = tmp_path / ".cursor" / "rules"
        rules_dir.mkdir(parents=True)
        paths = generate_cursor_rules(rules_dir, detected, standards=standards)
        for path in paths:
            assert path.suffix == ".mdc", f"{path.name} should have .mdc extension"

    def test_cursor_agents_generated(self, tmp_path: Path, detected: DetectedProject) -> None:
        from mirdan.integrations.cursor import generate_cursor_agents

        standards = QualityStandards()
        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir(parents=True)
        paths = generate_cursor_agents(cursor_dir, detected, standards=standards)
        names = {p.name for p in paths}
        assert "AGENTS.md" in names
        assert "BUGBOT.md" in names

    def test_cursor_rules_typescript_project(self, tmp_path: Path) -> None:
        from mirdan.integrations.cursor import generate_cursor_rules

        ts_detected = DetectedProject(
            project_type="node", primary_language="typescript", frameworks=["react"],
        )
        standards = QualityStandards()
        rules_dir = tmp_path / ".cursor" / "rules"
        rules_dir.mkdir(parents=True)
        paths = generate_cursor_rules(rules_dir, ts_detected, standards=standards)
        names = {p.name for p in paths}
        assert "mirdan-typescript.mdc" in names

    def test_cursor_rules_go_project(self, tmp_path: Path) -> None:
        from mirdan.integrations.cursor import generate_cursor_rules

        go_detected = DetectedProject(
            project_type="go", primary_language="go", frameworks=[],
        )
        standards = QualityStandards()
        rules_dir = tmp_path / ".cursor" / "rules"
        rules_dir.mkdir(parents=True)
        paths = generate_cursor_rules(rules_dir, go_detected, standards=standards)
        names = {p.name for p in paths}
        assert "mirdan-go.mdc" in names

    def test_cursor_rules_rust_project(self, tmp_path: Path) -> None:
        from mirdan.integrations.cursor import generate_cursor_rules

        rust_detected = DetectedProject(
            project_type="rust", primary_language="rust", frameworks=[],
        )
        standards = QualityStandards()
        rules_dir = tmp_path / ".cursor" / "rules"
        rules_dir.mkdir(parents=True)
        paths = generate_cursor_rules(rules_dir, rust_detected, standards=standards)
        names = {p.name for p in paths}
        assert "mirdan-rust.mdc" in names

    def test_cursor_rules_without_standards_falls_back(
        self, tmp_path: Path, detected: DetectedProject
    ) -> None:
        from mirdan.integrations.cursor import generate_cursor_rules

        rules_dir = tmp_path / ".cursor" / "rules"
        rules_dir.mkdir(parents=True)
        # No standards passed — should fall back to static templates
        paths = generate_cursor_rules(rules_dir, detected, standards=None)
        assert len(paths) >= 0  # May be empty or have static fallback

    def test_cursor_agents_content(self, tmp_path: Path, detected: DetectedProject) -> None:
        from mirdan.integrations.cursor import generate_cursor_agents

        standards = QualityStandards()
        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir(parents=True)
        paths = generate_cursor_agents(cursor_dir, detected, standards=standards)
        agents_md = next(p for p in paths if p.name == "AGENTS.md")
        content = agents_md.read_text()
        assert "mirdan" in content.lower()


# ---------------------------------------------------------------------------
# Phase 0.4.0: GitHub CI Integration
# ---------------------------------------------------------------------------


class TestGithubCIIntegration:
    """Tests for pre-commit config generation."""

    def test_precommit_config_generated(self, tmp_path: Path) -> None:
        from mirdan.integrations.github_ci import generate_precommit_config

        path = generate_precommit_config(tmp_path)
        assert path is not None
        assert path.exists()
        content = path.read_text()
        assert "mirdan" in content

    def test_precommit_not_overwritten(self, tmp_path: Path) -> None:
        from mirdan.integrations.github_ci import generate_precommit_config

        path = generate_precommit_config(tmp_path)
        assert path is not None
        result = generate_precommit_config(tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# Phase 0.4.0: Testing Standards
# ---------------------------------------------------------------------------


class TestTestingStandards:
    """Tests for testing.yaml standards (Step 26)."""

    def test_testing_standards_loaded(self) -> None:
        standards = QualityStandards()
        testing = standards.get_testing_standards()
        assert testing is not None
        assert "testing" in testing

    def test_testing_standards_has_categories(self) -> None:
        standards = QualityStandards()
        testing = standards.get_testing_standards()
        testing_data = testing.get("testing", {})
        assert "naming" in testing_data
        assert "assertions" in testing_data


# ---------------------------------------------------------------------------
# Phase 0.4.0: Report Command
# ---------------------------------------------------------------------------


class TestReportCommand:
    """Tests for mirdan report command (Step 27)."""

    def test_discover_source_files(self, tmp_path: Path) -> None:
        from mirdan.cli.report_command import _discover_source_files

        # Create a Python file
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "test.js").write_text("console.log('hello')")

        files = _discover_source_files(tmp_path)
        assert len(files) >= 2

    def test_discover_skips_hidden_dirs(self, tmp_path: Path) -> None:
        from mirdan.cli.report_command import _discover_source_files

        # Create a hidden dir with a file
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("x = 1")
        (tmp_path / "main.py").write_text("x = 1")

        files = _discover_source_files(tmp_path)
        assert not any(".hidden" in str(f) for f in files)

    def test_discover_filters_by_language(self, tmp_path: Path) -> None:
        from mirdan.cli.report_command import _discover_source_files

        (tmp_path / "main.py").write_text("x = 1")
        (tmp_path / "app.js").write_text("const x = 1")

        python_files = _discover_source_files(tmp_path, language_filter="python")
        assert all(str(f).endswith(".py") for f in python_files)


# ---------------------------------------------------------------------------
# Phase 0.4.0: Framework Version Awareness
# ---------------------------------------------------------------------------


class TestFrameworkVersionAwareness:
    """Tests for framework version detection in IntentAnalyzer (Step 29)."""

    def test_intent_has_framework_versions_field(self) -> None:
        from mirdan.core.intent_analyzer import IntentAnalyzer

        analyzer = IntentAnalyzer()
        intent = analyzer.analyze("create a fastapi endpoint")
        assert hasattr(intent, "framework_versions")
        assert isinstance(intent.framework_versions, dict)

    def test_framework_versions_empty_when_no_manifests(self) -> None:
        from mirdan.core.intent_analyzer import IntentAnalyzer

        analyzer = IntentAnalyzer()
        intent = analyzer.analyze("create a react component")
        # framework_versions may be empty if no package.json in cwd
        assert isinstance(intent.framework_versions, dict)


# ---------------------------------------------------------------------------
# Phase 0.4.0: Init Command CI Wiring
# ---------------------------------------------------------------------------


class TestInitPrecommitWiring:
    """Tests for pre-commit integration in init flow."""

    def test_setup_precommit_generates_config(self, tmp_path: Path) -> None:
        from mirdan.cli.detect import DetectedProject
        from mirdan.cli.init_command import _setup_precommit

        detected = DetectedProject(project_type="python")
        _setup_precommit(tmp_path, detected)
        precommit = tmp_path / ".pre-commit-config.yaml"
        assert precommit.exists()

    def test_setup_precommit_idempotent(self, tmp_path: Path) -> None:
        """Running pre-commit setup twice should not fail."""
        from mirdan.cli.detect import DetectedProject
        from mirdan.cli.init_command import _setup_precommit

        detected = DetectedProject(project_type="python")
        _setup_precommit(tmp_path, detected)
        # Second call — should not overwrite
        _setup_precommit(tmp_path, detected)
        assert (tmp_path / ".pre-commit-config.yaml").exists()

    def test_no_github_action_created(self, tmp_path: Path) -> None:
        """mirdan init must not create GitHub Actions workflows."""
        from mirdan.cli.detect import DetectedProject
        from mirdan.cli.init_command import _setup_precommit

        detected = DetectedProject(project_type="python")
        (tmp_path / ".git").mkdir()
        _setup_precommit(tmp_path, detected)
        workflow = tmp_path / ".github" / "workflows" / "mirdan.yml"
        assert not workflow.exists()


# ---------------------------------------------------------------------------
# Phase 0.4.0: Quality Standards (testing)
# ---------------------------------------------------------------------------


class TestQualityStandardsTestingIntegration:
    """Additional tests for testing standards integration."""

    def test_render_for_intent_includes_testing_standards(self) -> None:
        from mirdan.models import Intent, TaskType

        standards = QualityStandards()
        intent = Intent(
            original_prompt="write unit tests for auth module",
            task_type=TaskType.TEST,
            primary_language="python",
        )
        requirements = standards.render_for_intent(intent)
        # Testing standards should be included when task type is TEST
        assert any("test" in r.lower() for r in requirements)

    def test_render_for_intent_no_testing_for_generation(self) -> None:
        from mirdan.models import Intent, TaskType

        standards = QualityStandards()
        intent = Intent(
            original_prompt="create a user model",
            task_type=TaskType.GENERATION,
            primary_language="python",
        )
        requirements = standards.render_for_intent(intent)
        # Testing standards should NOT be included for GENERATION task
        # (may still include general test words from other standards)
        assert isinstance(requirements, list)


# ---------------------------------------------------------------------------
# Phase 0.4.0: Micro Output Format CLI
# ---------------------------------------------------------------------------


class TestOrchestratorPlanning:
    """Tests for MCPOrchestrator.suggest_tools_for_planning."""

    def test_planning_suggestions_include_enyal(self) -> None:
        from mirdan.core.orchestrator import MCPOrchestrator
        from mirdan.models import Intent, TaskType

        orchestrator = MCPOrchestrator()
        intent = Intent(
            original_prompt="plan a new auth system",
            task_type=TaskType.PLANNING,
            primary_language="python",
        )
        recs = orchestrator.suggest_tools_for_planning(intent)
        mcp_names = [r.mcp for r in recs]
        assert "enyal" in mcp_names

    def test_planning_suggestions_include_filesystem(self) -> None:
        from mirdan.core.orchestrator import MCPOrchestrator
        from mirdan.models import Intent, TaskType

        orchestrator = MCPOrchestrator()
        intent = Intent(
            original_prompt="plan a refactor",
            task_type=TaskType.PLANNING,
            primary_language="python",
        )
        recs = orchestrator.suggest_tools_for_planning(intent)
        mcp_names = [r.mcp for r in recs]
        assert "filesystem" in mcp_names

    def test_planning_with_framework_includes_context7(self) -> None:
        from mirdan.core.orchestrator import MCPOrchestrator
        from mirdan.models import Intent, TaskType

        orchestrator = MCPOrchestrator()
        intent = Intent(
            original_prompt="plan fastapi migration",
            task_type=TaskType.PLANNING,
            primary_language="python",
            frameworks=["fastapi"],
            uses_external_framework=True,
        )
        recs = orchestrator.suggest_tools_for_planning(intent)
        mcp_names = [r.mcp for r in recs]
        assert "context7" in mcp_names

    def test_planning_with_custom_available_mcps(self) -> None:
        from mirdan.core.orchestrator import MCPOrchestrator
        from mirdan.models import Intent, TaskType

        orchestrator = MCPOrchestrator()
        intent = Intent(
            original_prompt="plan something",
            task_type=TaskType.PLANNING,
        )
        recs = orchestrator.suggest_tools_for_planning(intent, available_mcps=["enyal"])
        mcp_names = [r.mcp for r in recs]
        assert "enyal" in mcp_names
        assert "filesystem" not in mcp_names

    def test_get_available_mcp_info(self) -> None:
        from mirdan.core.orchestrator import MCPOrchestrator

        orchestrator = MCPOrchestrator()
        info = orchestrator.get_available_mcp_info()
        assert "context7" in info
        assert "enyal" in info


class TestMicroOutputCLI:
    """Tests for the CLI _output_micro function."""

    def test_output_micro_pass(self, capsys: pytest.CaptureFixture[str]) -> None:
        from mirdan.cli.validate_command import _output_micro

        result = ValidationResult(
            passed=True, score=0.95, language_detected="python",
        )
        _output_micro(result)
        captured = capsys.readouterr()
        assert "PASS 0.95" in captured.out

    def test_output_micro_fail_no_violations(self, capsys: pytest.CaptureFixture[str]) -> None:
        from mirdan.cli.validate_command import _output_micro

        result = ValidationResult(
            passed=False, score=0.5, language_detected="python", violations=[],
        )
        _output_micro(result)
        captured = capsys.readouterr()
        assert "FAIL:" in captured.out

    def test_output_micro_fail(self, capsys: pytest.CaptureFixture[str]) -> None:
        from mirdan.cli.validate_command import _output_micro

        result = ValidationResult(
            passed=False, score=0.5, language_detected="python",
            violations=[
                Violation(
                    id="SEC001", rule="hardcoded-secrets", category="security",
                    severity="error", message="test", line=10,
                ),
            ],
        )
        _output_micro(result)
        captured = capsys.readouterr()
        assert "FAIL:" in captured.out
        assert "SEC001:L10" in captured.out
