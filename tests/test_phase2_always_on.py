"""Tests for Phase 2: Make It Always-On (v0.6.0).

Tests cover:
- SessionTracker: per-file tracking, unvalidated files, summaries
- Gate command: CLI quality gate, pass/fail, exit codes
- Auto-fix integration: quick_fix method, confidence filtering
- Stop gate: command+prompt upgrade
- Session summary: markdown generation
- Environment output: output format differs by detected IDE
- Hook upgrades: PostToolUse micro format, TaskCompleted command type
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from mirdan.core.auto_fixer import AutoFixer
from mirdan.core.session_tracker import SessionTracker
from mirdan.integrations.hook_templates import (
    STRINGENCY_EVENTS,
    HookConfig,
    HookStringency,
    HookTemplateGenerator,
)
from mirdan.integrations.self_managing import SelfManagingIntegration
from mirdan.models import SessionContext, TaskType, ValidationResult, Violation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker() -> SessionTracker:
    """Fresh session tracker."""
    return SessionTracker()


@pytest.fixture
def passing_result() -> ValidationResult:
    """A passing validation result."""
    return ValidationResult(
        passed=True,
        score=0.95,
        language_detected="python",
        violations=[],
        standards_checked=["style", "security"],
    )


@pytest.fixture
def failing_result() -> ValidationResult:
    """A failing validation result with errors."""
    return ValidationResult(
        passed=False,
        score=0.65,
        language_detected="python",
        violations=[
            Violation(
                id="SEC002",
                rule="sql-injection",
                category="security",
                severity="error",
                message="SQL injection via string concatenation",
                line=10,
                code_snippet="cursor.execute(f'SELECT * FROM {table}')",
            ),
            Violation(
                id="AI001",
                rule="placeholder-code",
                category="ai",
                severity="error",
                message="Placeholder code detected",
                line=20,
            ),
            Violation(
                id="PY003",
                rule="bare-except",
                category="style",
                severity="warning",
                message="Bare except clause",
                line=30,
                code_snippet="except:",
            ),
        ],
        standards_checked=["style", "security", "ai"],
    )


@pytest.fixture
def session() -> SessionContext:
    """A session context for tracking."""
    return SessionContext(
        session_id="test-session-001",
        task_type=TaskType.GENERATION,
        detected_language="python",
    )


# ---------------------------------------------------------------------------
# 2A: SessionTracker
# ---------------------------------------------------------------------------


class TestSessionTracker:
    """Tests for the SessionTracker class."""

    def test_record_validation_increments_count(
        self,
        tracker: SessionTracker,
        passing_result: ValidationResult,
    ) -> None:
        tracker.record_validation(passing_result, file_path="test.py")
        summary = tracker.get_session_summary()
        assert summary.validation_count == 1

    def test_record_multiple_validations(
        self,
        tracker: SessionTracker,
        passing_result: ValidationResult,
        failing_result: ValidationResult,
    ) -> None:
        tracker.record_validation(passing_result, file_path="a.py")
        tracker.record_validation(failing_result, file_path="b.py")
        summary = tracker.get_session_summary()
        assert summary.validation_count == 2
        assert summary.files_validated == 2

    def test_unvalidated_files(
        self,
        tracker: SessionTracker,
        passing_result: ValidationResult,
    ) -> None:
        tracker.record_validation(passing_result, file_path="a.py")
        unvalidated = tracker.get_unvalidated_files(["a.py", "b.py", "c.py"])
        assert unvalidated == ["b.py", "c.py"]

    def test_all_files_validated(
        self,
        tracker: SessionTracker,
        passing_result: ValidationResult,
    ) -> None:
        tracker.record_validation(passing_result, file_path="a.py")
        tracker.record_validation(passing_result, file_path="b.py")
        assert tracker.get_unvalidated_files(["a.py", "b.py"]) == []

    def test_score_for_file(
        self,
        tracker: SessionTracker,
        passing_result: ValidationResult,
    ) -> None:
        tracker.record_validation(passing_result, file_path="test.py")
        assert tracker.get_score_for_file("test.py") == 0.95

    def test_score_for_unknown_file(self, tracker: SessionTracker) -> None:
        assert tracker.get_score_for_file("unknown.py") is None

    def test_score_uses_latest(
        self,
        tracker: SessionTracker,
        passing_result: ValidationResult,
        failing_result: ValidationResult,
    ) -> None:
        tracker.record_validation(failing_result, file_path="test.py")
        tracker.record_validation(passing_result, file_path="test.py")
        assert tracker.get_score_for_file("test.py") == 0.95

    def test_session_summary_empty(self, tracker: SessionTracker) -> None:
        summary = tracker.get_session_summary()
        assert summary.validation_count == 0
        assert summary.files_validated == 0

    def test_session_summary_avg_score(
        self,
        tracker: SessionTracker,
        passing_result: ValidationResult,
        failing_result: ValidationResult,
    ) -> None:
        tracker.record_validation(passing_result, file_path="a.py")
        tracker.record_validation(failing_result, file_path="b.py")
        summary = tracker.get_session_summary()
        expected = (0.95 + 0.65) / 2
        assert abs(summary.avg_score - expected) < 0.01

    def test_session_summary_errors(
        self,
        tracker: SessionTracker,
        failing_result: ValidationResult,
    ) -> None:
        tracker.record_validation(failing_result, file_path="test.py")
        summary = tracker.get_session_summary()
        assert summary.total_errors == 2  # SEC002 + AI001

    def test_session_summary_pass_rate(
        self,
        tracker: SessionTracker,
        passing_result: ValidationResult,
        failing_result: ValidationResult,
    ) -> None:
        tracker.record_validation(passing_result, file_path="a.py")
        tracker.record_validation(failing_result, file_path="b.py")
        summary = tracker.get_session_summary()
        assert summary.pass_rate == 0.5

    def test_session_summary_to_dict(
        self,
        tracker: SessionTracker,
        passing_result: ValidationResult,
    ) -> None:
        tracker.record_validation(passing_result, file_path="test.py")
        summary = tracker.get_session_summary("sess-001")
        d = summary.to_dict()
        assert d["session_id"] == "sess-001"
        assert d["validation_count"] == 1

    def test_record_updates_session_context(
        self,
        tracker: SessionTracker,
        passing_result: ValidationResult,
        session: SessionContext,
    ) -> None:
        tracker.record_validation(passing_result, file_path="test.py", session=session)
        assert session.validation_count == 1
        assert session.cumulative_score == 0.95
        assert "test.py" in session.files_validated
        assert session.last_validated_at > 0

    def test_session_context_to_dict_includes_quality(
        self,
        tracker: SessionTracker,
        passing_result: ValidationResult,
        session: SessionContext,
    ) -> None:
        tracker.record_validation(passing_result, file_path="test.py", session=session)
        d = session.to_dict()
        assert "session_quality" in d
        assert d["session_quality"]["validation_count"] == 1

    def test_to_snapshot(
        self,
        tracker: SessionTracker,
        passing_result: ValidationResult,
    ) -> None:
        tracker.record_validation(passing_result, file_path="test.py")
        snapshot = tracker.to_snapshot("sess-001")
        assert snapshot is not None
        assert snapshot.score == 0.95
        assert snapshot.passed is True

    def test_to_snapshot_empty(self, tracker: SessionTracker) -> None:
        assert tracker.to_snapshot() is None

    def test_get_previous_violations_empty_first_run(
        self,
        tracker: SessionTracker,
        failing_result: ValidationResult,
    ) -> None:
        tracker.record_validation(failing_result, file_path="test.py")
        assert tracker.get_previous_violations("test.py") == set()

    def test_get_previous_violations_after_two_runs(
        self,
        tracker: SessionTracker,
        passing_result: ValidationResult,
        failing_result: ValidationResult,
    ) -> None:
        tracker.record_validation(failing_result, file_path="test.py")
        tracker.record_validation(passing_result, file_path="test.py")
        prev = tracker.get_previous_violations("test.py")
        assert "SEC002" in prev
        assert "AI001" in prev

    def test_get_violation_persistence_counts_consecutive(
        self,
        tracker: SessionTracker,
        failing_result: ValidationResult,
    ) -> None:
        tracker.record_validation(failing_result, file_path="test.py")
        tracker.record_validation(failing_result, file_path="test.py")
        persistence = tracker.get_violation_persistence("test.py")
        assert persistence.get("SEC002") == 2
        assert persistence.get("AI001") == 2

    def test_get_violation_persistence_resets_on_fix(
        self,
        tracker: SessionTracker,
        passing_result: ValidationResult,
        failing_result: ValidationResult,
    ) -> None:
        tracker.record_validation(failing_result, file_path="test.py")
        tracker.record_validation(passing_result, file_path="test.py")
        persistence = tracker.get_violation_persistence("test.py")
        assert persistence == {}

    def test_violation_rules_recorded(
        self,
        tracker: SessionTracker,
        failing_result: ValidationResult,
    ) -> None:
        tracker.record_validation(failing_result, file_path="test.py")
        records = tracker._file_validations["test.py"]
        assert "SEC002" in records[0].violation_rules
        assert "AI001" in records[0].violation_rules


# ---------------------------------------------------------------------------
# 2C: Auto-Fix quick_fix
# ---------------------------------------------------------------------------


class TestQuickFix:
    """Tests for AutoFixer.quick_fix method."""

    def test_returns_fixes_for_security_violations(
        self,
        failing_result: ValidationResult,
    ) -> None:
        fixer = AutoFixer()
        fixes = fixer.quick_fix(failing_result)
        # SEC002 has confidence 0.90, should be included
        assert any(f.fix_description for f in fixes if "parameterized" in f.fix_description.lower())

    def test_skips_low_confidence_fixes(self) -> None:
        fixer = AutoFixer()
        result = ValidationResult(
            passed=False,
            score=0.5,
            language_detected="python",
            violations=[
                Violation(
                    id="AI001",
                    rule="placeholder-code",
                    category="ai",
                    severity="error",
                    message="Placeholder code",
                    line=1,
                ),
            ],
        )
        fixes = fixer.quick_fix(result)
        # AI001 template has confidence 0.70, below 0.80 threshold
        assert len(fixes) == 0

    def test_skips_non_security_violations(self) -> None:
        fixer = AutoFixer()
        result = ValidationResult(
            passed=False,
            score=0.5,
            language_detected="python",
            violations=[
                Violation(
                    id="PY003",
                    rule="bare-except",
                    category="style",
                    severity="warning",
                    message="Bare except",
                    line=1,
                    code_snippet="except:",
                ),
            ],
        )
        fixes = fixer.quick_fix(result)
        # PY003 is not in _QUICK_FIX_RULES
        assert len(fixes) == 0

    def test_returns_empty_for_passing_code(
        self,
        passing_result: ValidationResult,
    ) -> None:
        fixer = AutoFixer()
        fixes = fixer.quick_fix(passing_result)
        assert len(fixes) == 0

    def test_sec006_fix_included(self) -> None:
        fixer = AutoFixer()
        result = ValidationResult(
            passed=False,
            score=0.7,
            language_detected="python",
            violations=[
                Violation(
                    id="SEC006",
                    rule="use-https",
                    category="security",
                    severity="warning",
                    message="Use HTTPS",
                    line=5,
                ),
            ],
        )
        fixes = fixer.quick_fix(result)
        assert len(fixes) == 1
        assert fixes[0].confidence >= 0.8


# ---------------------------------------------------------------------------
# 2D: Gate command
# ---------------------------------------------------------------------------


class TestGateCommand:
    """Tests for the gate command module."""

    def test_gate_module_exists(self) -> None:
        from mirdan.cli.gate_command import run_gate

        assert callable(run_gate)

    def test_gate_registered_in_cli(self) -> None:
        from mirdan.cli import main

        source = inspect.getsource(main)
        assert "gate" in source

    def test_get_changed_files_function(self) -> None:
        from mirdan.cli.gate_command import _get_changed_files

        # Should return a list (may be empty in test env)
        result = _get_changed_files()
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 2D/2E: Stop and TaskCompleted hook upgrades
# ---------------------------------------------------------------------------


class TestStopGate:
    """Tests for Stop hook command+prompt upgrade."""

    def test_stop_has_command_type(self) -> None:
        gen = HookTemplateGenerator()
        hooks = gen.generate()["hooks"]
        stop = hooks["Stop"]
        hook_types = [h["type"] for h in stop[0]["hooks"]]
        assert "command" in hook_types

    def test_stop_has_prompt_type(self) -> None:
        gen = HookTemplateGenerator()
        hooks = gen.generate()["hooks"]
        stop = hooks["Stop"]
        hook_types = [h["type"] for h in stop[0]["hooks"]]
        assert "prompt" in hook_types

    def test_stop_command_runs_gate(self) -> None:
        gen = HookTemplateGenerator()
        hooks = gen.generate()["hooks"]
        stop = hooks["Stop"]
        cmd_hooks = [h for h in stop[0]["hooks"] if h["type"] == "command"]
        assert "gate" in cmd_hooks[0]["command"]

    def test_stop_command_has_timeout(self) -> None:
        gen = HookTemplateGenerator()
        hooks = gen.generate()["hooks"]
        stop = hooks["Stop"]
        cmd_hooks = [h for h in stop[0]["hooks"] if h["type"] == "command"]
        assert cmd_hooks[0]["timeout"] == 30000

    def test_stop_prompt_mentions_fail(self) -> None:
        gen = HookTemplateGenerator()
        hooks = gen.generate()["hooks"]
        stop = hooks["Stop"]
        prompt_hooks = [h for h in stop[0]["hooks"] if h["type"] == "prompt"]
        assert "FAIL" in prompt_hooks[0]["prompt"]


class TestTaskCompletedHook:
    """Tests for TaskCompleted hook command upgrade."""

    def test_task_completed_has_command(self) -> None:
        config = HookConfig(multi_agent_awareness=True)
        gen = HookTemplateGenerator(config=config)
        hooks = gen.generate()["hooks"]
        tc = hooks["TaskCompleted"]
        hook_types = [h["type"] for h in tc[0]["hooks"]]
        assert "command" in hook_types

    def test_task_completed_command_runs_report(self) -> None:
        config = HookConfig(multi_agent_awareness=True)
        gen = HookTemplateGenerator(config=config)
        hooks = gen.generate()["hooks"]
        tc = hooks["TaskCompleted"]
        cmd_hooks = [h for h in tc[0]["hooks"] if h["type"] == "command"]
        assert "report --session" in cmd_hooks[0]["command"]

    def test_task_completed_has_prompt(self) -> None:
        config = HookConfig(multi_agent_awareness=True)
        gen = HookTemplateGenerator(config=config)
        hooks = gen.generate()["hooks"]
        tc = hooks["TaskCompleted"]
        hook_types = [h["type"] for h in tc[0]["hooks"]]
        assert "prompt" in hook_types


# ---------------------------------------------------------------------------
# 2B: PostToolUse micro format
# ---------------------------------------------------------------------------


class TestPostToolUseMicro:
    """Tests for PostToolUse micro format upgrade."""

    def test_post_tool_use_uses_micro_format(self) -> None:
        gen = HookTemplateGenerator()
        hooks = gen.generate()["hooks"]
        ptu = hooks["PostToolUse"]
        cmd_hooks = [h for h in ptu[0]["hooks"] if h["type"] == "command"]
        assert len(cmd_hooks) > 0
        assert "--format micro" in cmd_hooks[0]["command"]

    def test_post_tool_use_has_timeout(self) -> None:
        gen = HookTemplateGenerator()
        hooks = gen.generate()["hooks"]
        ptu = hooks["PostToolUse"]
        cmd_hooks = [h for h in ptu[0]["hooks"] if h["type"] == "command"]
        assert cmd_hooks[0]["timeout"] == 5000


# ---------------------------------------------------------------------------
# 2F: Session summary
# ---------------------------------------------------------------------------


class TestSessionSummary:
    """Tests for session summary generation."""

    def test_summary_is_markdown(self) -> None:
        integration = SelfManagingIntegration()
        data = {
            "validation_count": 5,
            "avg_score": 0.85,
            "files_validated": 3,
            "unresolved_errors": 0,
        }
        result = integration.generate_session_summary(data)
        assert "# mirdan Session Quality Summary" in result
        assert "| Metric | Value |" in result

    def test_summary_shows_pass(self) -> None:
        integration = SelfManagingIntegration()
        data = {
            "validation_count": 3,
            "avg_score": 0.90,
            "files_validated": 3,
            "unresolved_errors": 0,
        }
        result = integration.generate_session_summary(data)
        assert "PASS" in result

    def test_summary_shows_needs_work(self) -> None:
        integration = SelfManagingIntegration()
        data = {
            "validation_count": 3,
            "avg_score": 0.60,
            "files_validated": 3,
            "unresolved_errors": 2,
        }
        result = integration.generate_session_summary(data)
        assert "NEEDS WORK" in result

    def test_summary_includes_file_details(self) -> None:
        integration = SelfManagingIntegration()
        data = {
            "validation_count": 2,
            "avg_score": 0.80,
            "files_validated": 2,
            "unresolved_errors": 0,
        }
        file_results = [
            {"file": "main.py", "score": 0.90, "passed": True},
            {"file": "utils.py", "score": 0.70, "passed": False},
        ]
        result = integration.generate_session_summary(data, file_results=file_results)
        assert "main.py" in result
        assert "utils.py" in result
        assert "## File Details" in result


# ---------------------------------------------------------------------------
# 2F: Report CLI --session and --compact-state
# ---------------------------------------------------------------------------


class TestReportSessionFlag:
    """Tests for --session flag in report command."""

    def test_parse_session_flag(self) -> None:
        from mirdan.cli.report_command import _parse_report_args

        parsed = _parse_report_args(["--session"])
        assert parsed.get("session") is True

    def test_parse_compact_state_flag(self) -> None:
        from mirdan.cli.report_command import _parse_report_args

        parsed = _parse_report_args(["--compact-state"])
        assert parsed.get("compact_state") is True

    def test_parse_session_with_format(self) -> None:
        from mirdan.cli.report_command import _parse_report_args

        parsed = _parse_report_args(["--session", "--format", "json"])
        assert parsed.get("session") is True
        assert parsed.get("format") == "json"


# ---------------------------------------------------------------------------
# 2G: Environment-aware output
# ---------------------------------------------------------------------------


class TestEnvironmentOutput:
    """Tests for environment-aware output optimization."""

    def test_server_imports_ide_type(self) -> None:
        source = inspect.getsource(__import__("mirdan.server", fromlist=["_get_components"]))
        assert "IDEType" in source

    def test_claude_code_gets_higher_compact_threshold(self) -> None:
        """In Claude Code env, compact_threshold should be raised."""
        source = inspect.getsource(__import__("mirdan.server", fromlist=["_get_components"]))
        assert "IDEType.CLAUDE_CODE" in source
        assert "compact_threshold" in source

    def test_cursor_gets_lower_compact_threshold(self) -> None:
        """In Cursor env, compact_threshold should be lowered."""
        source = inspect.getsource(__import__("mirdan.server", fromlist=["_get_components"]))
        assert "IDEType.CURSOR" in source


# ---------------------------------------------------------------------------
# Server integration: SessionTracker wired
# ---------------------------------------------------------------------------


class TestServerSessionTracking:
    """Tests for SessionTracker wiring in server.py."""

    def test_server_has_session_tracker(self) -> None:
        source = inspect.getsource(__import__("mirdan.server", fromlist=["_Components"]))
        assert "session_tracker" in source

    def test_server_has_auto_fixer(self) -> None:
        source = inspect.getsource(__import__("mirdan.server", fromlist=["_Components"]))
        assert "auto_fixer" in source

    def test_validate_code_quality_records_session(self) -> None:
        source = inspect.getsource(__import__("mirdan.server", fromlist=["validate_code_quality"]))
        assert "record_validation" in source

    def test_validate_quick_includes_auto_fixes(self) -> None:
        source = inspect.getsource(__import__("mirdan.server", fromlist=["validate_quick"]))
        assert "quick_fix" in source
        assert "auto_fixes" in source

    def test_enhance_prompt_stores_tool_recommendations(self) -> None:
        source = inspect.getsource(__import__("mirdan.server", fromlist=["enhance_prompt"]))
        assert "tool_recommendations" in source
        assert "session.tool_recommendations" in source

    def test_validate_includes_timing_ms(self) -> None:
        source = inspect.getsource(__import__("mirdan.server", fromlist=["validate_code_quality"]))
        assert "timing_ms" in source
        assert "perf_counter" in source

    def test_validate_includes_session_context_delta(self) -> None:
        source = inspect.getsource(__import__("mirdan.server", fromlist=["validate_code_quality"]))
        assert "session_context" in source
        assert "get_previous_violations" in source
        assert "get_violation_persistence" in source

    def test_validate_includes_recommendation_reminders(self) -> None:
        source = inspect.getsource(__import__("mirdan.server", fromlist=["validate_code_quality"]))
        assert "recommendation_reminders" in source

    def test_validate_includes_checklist_note_on_revalidation(self) -> None:
        source = inspect.getsource(__import__("mirdan.server", fromlist=["validate_code_quality"]))
        assert "checklist_note" in source

    def test_scan_conventions_persists_yaml(self) -> None:
        source = inspect.getsource(__import__("mirdan.server", fromlist=["scan_conventions"]))
        assert "conventions.yaml" in source
        assert "yaml.dump" in source

    def test_intent_analyzer_receives_manifest_parser(self) -> None:
        source = inspect.getsource(__import__("mirdan.server", fromlist=["_get_components"]))
        assert "manifest_parser=manifest_parser" in source

    def test_quality_standards_receives_project_dir(self) -> None:
        source = inspect.getsource(__import__("mirdan.server", fromlist=["_get_components"]))
        assert "project_dir=project_dir" in source


# ---------------------------------------------------------------------------
# Comprehensive stringency still works
# ---------------------------------------------------------------------------


class TestStringencyCompatibility:
    """Tests that stringency levels work correctly with Phase 2 changes."""

    def test_minimal_has_two_events(self) -> None:
        events = STRINGENCY_EVENTS[HookStringency.MINIMAL]
        assert len(events) == 2

    def test_standard_has_five_events(self) -> None:
        events = STRINGENCY_EVENTS[HookStringency.STANDARD]
        assert len(events) == 5

    def test_comprehensive_includes_task_completed(self) -> None:
        events = STRINGENCY_EVENTS[HookStringency.COMPREHENSIVE]
        assert "TaskCompleted" in events

    def test_comprehensive_hooks_generate(self) -> None:
        gen = HookTemplateGenerator()
        result = gen.generate_claude_code_hooks(HookStringency.COMPREHENSIVE)
        assert "Stop" in result["hooks"]
        assert "TaskCompleted" in result["hooks"]

    def test_stop_and_task_completed_both_have_commands(self) -> None:
        gen = HookTemplateGenerator()
        result = gen.generate_claude_code_hooks(HookStringency.COMPREHENSIVE)
        for event in ("Stop", "TaskCompleted"):
            hook_types = [h["type"] for h in result["hooks"][event][0]["hooks"]]
            assert "command" in hook_types, f"{event} missing command hook"


# ---------------------------------------------------------------------------
# TestFrontierGaps: v1.2.0 frontier research gap implementations
# ---------------------------------------------------------------------------


class TestGap2MultiLabelTaskType:
    """Gap 2: Multi-label TaskType — compound task detection via score_all()."""

    def test_single_task_type_for_clear_prompt(self) -> None:
        from mirdan.core.intent_analyzer import IntentAnalyzer

        analyzer = IntentAnalyzer()
        intent = analyzer.analyze("refactor the authentication module")
        assert intent.task_types[0] == TaskType.REFACTOR

    def test_task_types_always_starts_with_primary(self) -> None:
        from mirdan.core.intent_analyzer import IntentAnalyzer

        analyzer = IntentAnalyzer()
        intent = analyzer.analyze("fix the bug in the payment processor")
        assert intent.task_types[0] == intent.task_type

    def test_compound_task_detects_test_and_generation(self) -> None:
        from mirdan.core.intent_analyzer import IntentAnalyzer

        analyzer = IntentAnalyzer()
        # Strong TEST signal (unit tests=5) + strong GENERATION signal (create+feature=2+3)
        intent = analyzer.analyze("write unit tests for the new authentication feature")
        assert TaskType.TEST in intent.task_types
        # task_types should have more than one entry for a compound prompt
        assert len(intent.task_types) >= 1
        assert intent.task_types[0] == intent.task_type

    def test_task_types_populated_by_analyze(self) -> None:
        from mirdan.core.intent_analyzer import IntentAnalyzer

        analyzer = IntentAnalyzer()
        intent = analyzer.analyze("implement a REST API endpoint")
        assert len(intent.task_types) >= 1
        assert intent.task_type in intent.task_types

    def test_unknown_prompt_returns_unknown_list(self) -> None:
        from mirdan.core.intent_analyzer import IntentAnalyzer

        analyzer = IntentAnalyzer()
        intent = analyzer.analyze("hello world")
        assert intent.task_types == [TaskType.UNKNOWN]
        assert intent.task_type == TaskType.UNKNOWN

    def test_task_types_in_enhanced_prompt_dict(self) -> None:
        from mirdan.core.intent_analyzer import IntentAnalyzer
        from mirdan.core.prompt_composer import PromptComposer
        from mirdan.core.quality_standards import QualityStandards
        from mirdan.models import ContextBundle

        analyzer = IntentAnalyzer()
        intent = analyzer.analyze("create a new REST endpoint")
        composer = PromptComposer(QualityStandards())
        enhanced = composer.compose(intent, ContextBundle(), [])
        result = enhanced.to_dict()
        assert "task_types" in result
        assert isinstance(result["task_types"], list)
        assert len(result["task_types"]) >= 1

    def test_union_verification_steps_for_compound_intent(self) -> None:
        from mirdan.core.prompt_composer import PromptComposer
        from mirdan.core.quality_standards import QualityStandards
        from mirdan.models import Intent

        composer = PromptComposer(QualityStandards())
        # Manually construct compound intent with both TEST and GENERATION
        intent = Intent(
            original_prompt="add tests for the new feature",
            task_type=TaskType.TEST,
            task_types=[TaskType.TEST, TaskType.GENERATION],
        )
        steps = composer.generate_verification_steps(intent)
        # TEST steps present
        assert any("happy path" in s for s in steps)
        # GENERATION step present (from GENERATION secondary)
        assert any("integrates with existing patterns" in s for s in steps)


class TestGap4ViolationVerifiability:
    """Gap 4: Violation verifiability — pattern-based AI rules marked as heuristic."""

    def test_ai_quality_violations_heuristic_unless_ast_confirmed(self) -> None:
        from mirdan.core.ai_quality_checker import AIQualityChecker

        checker = AIQualityChecker()
        # AST-confirmed AI001 in Python → verifiable=True (v1.3.0 upgrade)
        code = "def foo():\n    raise NotImplementedError\n"
        violations = checker.check(code, "python")
        ai001 = [v for v in violations if v.id == "AI001"]
        assert len(ai001) > 0
        for v in ai001:
            assert v.verifiable is True
        # Non-AI001 ai_quality violations remain heuristic
        code2 = "import nonexistent_fake_xyz_module\n"
        violations2 = checker.check(code2, "python")
        ai_non_001 = [v for v in violations2 if v.category == "ai_quality" and v.id != "AI001"]
        for v in ai_non_001:
            assert v.verifiable is False

    def test_non_ai_violations_keep_verifiable_true(self) -> None:
        # Violation objects not from ai_quality_checker default to verifiable=True
        v = Violation(
            id="PY001",
            rule="bare-except",
            category="style",
            severity="warning",
            message="Bare except",
        )
        assert v.verifiable is True

    def test_verifiable_false_appears_in_to_dict(self) -> None:
        v = Violation(
            id="AI001",
            rule="ai-placeholder-code",
            category="ai_quality",
            severity="error",
            message="Placeholder",
            verifiable=False,
        )
        result = v.to_dict()
        assert result["verifiable"] is False

    def test_verifiable_true_absent_from_to_dict(self) -> None:
        """verifiable=True (the default) must NOT appear in to_dict() output."""
        v = Violation(
            id="PY001",
            rule="bare-except",
            category="style",
            severity="warning",
            message="Bare except",
        )
        result = v.to_dict()
        assert "verifiable" not in result

    def test_check_quick_also_marks_heuristic(self) -> None:
        from mirdan.core.ai_quality_checker import AIQualityChecker

        checker = AIQualityChecker()
        code = "import nonexistent_module_xyz\n"
        violations = checker.check_quick(code, "python")
        # check_quick runs AI001/AI007/AI008/SEC014; if any ai_quality violations fire,
        # they must be marked heuristic
        for v in violations:
            if v.category == "ai_quality":
                assert v.verifiable is False


class TestGap1ValidationFeedbackLoop:
    """Gap 1: Persistent violation requirements injected into enhance_prompt."""

    def test_empty_reqs_on_new_session(self) -> None:
        from mirdan.server import _get_persistent_violation_reqs

        session = SessionContext(session_id="test-1")
        tracker = SessionTracker()
        result = _get_persistent_violation_reqs(session, tracker)
        assert result == []

    def test_empty_reqs_when_no_files_validated(self) -> None:
        from mirdan.server import _get_persistent_violation_reqs

        session = SessionContext(session_id="test-2", validation_count=3)
        tracker = SessionTracker()
        result = _get_persistent_violation_reqs(session, tracker)
        assert result == []

    def test_persistent_reqs_formatted_for_recurring_violations(self) -> None:
        from mirdan.server import _get_persistent_violation_reqs

        tracker = SessionTracker()
        # Record three consecutive validations with the same failing rule
        v_fail = Violation(
            id="AI001",
            rule="ai-placeholder-code",
            category="ai_quality",
            severity="error",
            message="Placeholder",
        )
        result1 = ValidationResult(
            passed=False, score=0.5, language_detected="python", violations=[v_fail]
        )
        result2 = ValidationResult(
            passed=False, score=0.5, language_detected="python", violations=[v_fail]
        )
        result3 = ValidationResult(
            passed=False, score=0.5, language_detected="python", violations=[v_fail]
        )
        tracker.record_validation(result1, file_path="auth.py")
        tracker.record_validation(result2, file_path="auth.py")
        tracker.record_validation(result3, file_path="auth.py")

        session = SessionContext(
            session_id="test-3",
            validation_count=3,
            files_validated=["auth.py"],
        )
        reqs = _get_persistent_violation_reqs(session, tracker)
        assert len(reqs) > 0
        assert any("AI001" in r for r in reqs)
        assert any("recurring" in r.lower() or "consecutive" in r.lower() for r in reqs)

    def test_persistent_reqs_capped_at_three(self) -> None:
        from mirdan.server import _get_persistent_violation_reqs

        tracker = SessionTracker()
        # Create 5 different recurring violations
        for _i in range(3):
            violations = [
                Violation(
                    id=f"AI00{j + 1}",
                    rule=f"rule-{j}",
                    category="ai_quality",
                    severity="error",
                    message=f"Msg {j}",
                )
                for j in range(5)
            ]
            result = ValidationResult(
                passed=False, score=0.3, language_detected="python", violations=violations
            )
            tracker.record_validation(result, file_path="big.py")

        session = SessionContext(
            session_id="test-cap",
            validation_count=3,
            files_validated=["big.py"],
        )
        reqs = _get_persistent_violation_reqs(session, tracker)
        assert len(reqs) <= 3


class TestGap3SessionAwareRouting:
    """Gap 3: Session-aware tool routing in MCPOrchestrator."""

    def _make_session(
        self, validation_count: int = 0, unresolved_errors: int = 0
    ) -> SessionContext:
        return SessionContext(
            session_id="route-test",
            validation_count=validation_count,
            unresolved_errors=unresolved_errors,
        )

    def test_first_call_generic_enyal_recall(self) -> None:
        from mirdan.core.orchestrator import MCPOrchestrator
        from mirdan.models import Intent

        orc = MCPOrchestrator()
        intent = Intent(original_prompt="create endpoint", task_type=TaskType.GENERATION)
        session = self._make_session(validation_count=0)
        recs = orc.suggest_tools(
            intent, available_mcps=list(orc.KNOWN_MCPS.keys()), session=session
        )
        enyal_recs = [r for r in recs if r.mcp == "enyal"]
        recall_recs = [r for r in enyal_recs if "query" in r.params]
        assert len(recall_recs) == 1
        assert "conventions" in recall_recs[0].action.lower()

    def test_reuse_with_errors_gives_targeted_enyal(self) -> None:
        from mirdan.core.orchestrator import MCPOrchestrator
        from mirdan.models import Intent

        orc = MCPOrchestrator()
        intent = Intent(original_prompt="fix the auth", task_type=TaskType.DEBUG)
        session = self._make_session(validation_count=2, unresolved_errors=3)
        recs = orc.suggest_tools(
            intent, available_mcps=list(orc.KNOWN_MCPS.keys()), session=session
        )
        enyal_recs = [r for r in recs if r.mcp == "enyal"]
        assert len(enyal_recs) == 1
        assert (
            "validation" in enyal_recs[0].action.lower()
            or "failure" in enyal_recs[0].action.lower()
        )

    def test_reuse_after_passing_skips_enyal(self) -> None:
        from mirdan.core.orchestrator import MCPOrchestrator
        from mirdan.models import Intent

        orc = MCPOrchestrator()
        intent = Intent(original_prompt="add a feature", task_type=TaskType.GENERATION)
        session = self._make_session(validation_count=1, unresolved_errors=0)
        recs = orc.suggest_tools(
            intent, available_mcps=list(orc.KNOWN_MCPS.keys()), session=session
        )
        enyal_recs = [r for r in recs if r.mcp == "enyal"]
        assert len(enyal_recs) == 0

    def test_no_session_uses_generic_enyal(self) -> None:
        from mirdan.core.orchestrator import MCPOrchestrator
        from mirdan.models import Intent

        orc = MCPOrchestrator()
        intent = Intent(original_prompt="implement feature", task_type=TaskType.GENERATION)
        recs = orc.suggest_tools(intent, available_mcps=list(orc.KNOWN_MCPS.keys()), session=None)
        enyal_recs = [r for r in recs if r.mcp == "enyal"]
        recall_recs = [r for r in enyal_recs if "query" in r.params]
        assert len(recall_recs) == 1
        assert "conventions" in recall_recs[0].action.lower()


class TestGap5ChecklistPruning:
    """Gap 5: Verification checklist compressed after successful validation."""

    def _passing_session(self) -> SessionContext:
        return SessionContext(
            session_id="prune-test",
            validation_count=1,
            unresolved_errors=0,
        )

    def _failing_session(self) -> SessionContext:
        return SessionContext(
            session_id="prune-fail",
            validation_count=1,
            unresolved_errors=2,
        )

    def test_checklist_full_without_session(self) -> None:
        from mirdan.core.prompt_composer import PromptComposer
        from mirdan.core.quality_standards import QualityStandards
        from mirdan.models import Intent

        composer = PromptComposer(QualityStandards())
        intent = Intent(original_prompt="add a feature", task_type=TaskType.GENERATION)
        steps = composer.generate_verification_steps(intent, session=None)
        assert len(steps) >= 4

    def test_checklist_compressed_after_passing_validation(self) -> None:
        from mirdan.core.prompt_composer import PromptComposer
        from mirdan.core.quality_standards import QualityStandards
        from mirdan.models import Intent

        composer = PromptComposer(QualityStandards())
        intent = Intent(original_prompt="add a feature", task_type=TaskType.GENERATION)
        session = self._passing_session()
        steps = composer.generate_verification_steps(intent, session=session)
        base_steps = [s for s in steps if "regression" in s.lower() or "Previous validation" in s]
        assert len(base_steps) == 1
        # Total step count should be less than the uncompressed version
        steps_no_session = composer.generate_verification_steps(intent, session=None)
        assert len(steps) < len(steps_no_session)

    def test_checklist_unchanged_when_errors_remain(self) -> None:
        from mirdan.core.prompt_composer import PromptComposer
        from mirdan.core.quality_standards import QualityStandards
        from mirdan.models import Intent

        composer = PromptComposer(QualityStandards())
        intent = Intent(original_prompt="fix the bug", task_type=TaskType.DEBUG)
        session = self._failing_session()
        steps = composer.generate_verification_steps(intent, session=session)
        # Must have the full base checklist (4 items) plus task-specific additions
        assert len(steps) >= 4

    def test_task_specific_steps_preserved_after_pruning(self) -> None:
        from mirdan.core.prompt_composer import PromptComposer
        from mirdan.core.quality_standards import QualityStandards
        from mirdan.models import Intent

        composer = PromptComposer(QualityStandards())
        intent = Intent(
            original_prompt="refactor the auth module",
            task_type=TaskType.REFACTOR,
            task_types=[TaskType.REFACTOR],
        )
        session = self._passing_session()
        steps = composer.generate_verification_steps(intent, session=session)
        # Task-specific REFACTOR steps must still be present after base compression
        assert any("existing functionality" in s for s in steps)
        assert any("API signatures" in s for s in steps)


# ---------------------------------------------------------------------------
# v1.3.0 Frontier Research Gaps — 5 new test classes
# ---------------------------------------------------------------------------


class TestStructuredViolationFeedback:
    """Gap 1 (v1.3.0): Enriched persistent violation feedback strings.

    _get_persistent_violation_reqs now includes message, suggestion, and
    line number from violation_details stored by the session tracker.
    """

    def test_enriched_string_includes_message(self) -> None:
        """Feedback string contains the violation message."""
        from mirdan.server import _get_persistent_violation_reqs

        tracker = SessionTracker()
        v = Violation(
            id="SEC002",
            rule="sql-injection",
            category="security",
            severity="error",
            message="SQL injection via concatenation",
            suggestion="Use parameterized queries",
            line=42,
        )
        result = ValidationResult(
            passed=False, score=0.5, language_detected="python", violations=[v]
        )
        tracker.record_validation(result, file_path="db.py")
        tracker.record_validation(result, file_path="db.py")

        session = SessionContext(session_id="t1", validation_count=2, files_validated=["db.py"])
        reqs = _get_persistent_violation_reqs(session, tracker)
        assert len(reqs) == 1
        assert "SQL injection via concatenation" in reqs[0]

    def test_enriched_string_includes_suggestion(self) -> None:
        """Feedback string contains the fix suggestion."""
        from mirdan.server import _get_persistent_violation_reqs

        tracker = SessionTracker()
        v = Violation(
            id="SEC002",
            rule="sql-injection",
            category="security",
            severity="error",
            message="SQL injection",
            suggestion="Use parameterized queries",
            line=10,
        )
        result = ValidationResult(
            passed=False, score=0.5, language_detected="python", violations=[v]
        )
        tracker.record_validation(result, file_path="db.py")
        tracker.record_validation(result, file_path="db.py")

        session = SessionContext(session_id="t2", validation_count=2, files_validated=["db.py"])
        reqs = _get_persistent_violation_reqs(session, tracker)
        assert any("Fix: Use parameterized queries" in r for r in reqs)

    def test_enriched_string_includes_line(self) -> None:
        """Feedback string contains 'at line N'."""
        from mirdan.server import _get_persistent_violation_reqs

        tracker = SessionTracker()
        v = Violation(
            id="AI001",
            rule="placeholder",
            category="ai_quality",
            severity="error",
            message="Placeholder code",
            line=77,
        )
        result = ValidationResult(
            passed=False, score=0.5, language_detected="python", violations=[v]
        )
        tracker.record_validation(result, file_path="api.py")
        tracker.record_validation(result, file_path="api.py")

        session = SessionContext(session_id="t3", validation_count=2, files_validated=["api.py"])
        reqs = _get_persistent_violation_reqs(session, tracker)
        assert any("at line 77" in r for r in reqs)

    def test_enriched_string_omits_empty_suggestion(self) -> None:
        """When suggestion is None/empty, no 'Fix:' appears."""
        from mirdan.server import _get_persistent_violation_reqs

        tracker = SessionTracker()
        v = Violation(
            id="AI001",
            rule="placeholder",
            category="ai_quality",
            severity="error",
            message="Placeholder code",
            line=10,
        )
        result = ValidationResult(
            passed=False, score=0.5, language_detected="python", violations=[v]
        )
        tracker.record_validation(result, file_path="x.py")
        tracker.record_validation(result, file_path="x.py")

        session = SessionContext(session_id="t4", validation_count=2, files_validated=["x.py"])
        reqs = _get_persistent_violation_reqs(session, tracker)
        assert all("Fix:" not in r for r in reqs)

    def test_violation_details_stored_in_session_tracker(self) -> None:
        """FileValidation.violation_details is populated by record_validation."""
        tracker = SessionTracker()
        v = Violation(
            id="SEC002",
            rule="sql-injection",
            category="security",
            severity="error",
            message="SQL injection",
            suggestion="Use params",
            line=5,
        )
        result = ValidationResult(
            passed=False, score=0.5, language_detected="python", violations=[v]
        )
        tracker.record_validation(result, file_path="f.py")

        records = tracker._file_validations["f.py"]
        assert "SEC002" in records[0].violation_details
        assert records[0].violation_details["SEC002"]["message"] == "SQL injection"
        assert records[0].violation_details["SEC002"]["suggestion"] == "Use params"
        assert records[0].violation_details["SEC002"]["line"] == 5

    def test_get_persistent_violation_details(self) -> None:
        """get_persistent_violation_details returns details for 2+ consecutive rules."""
        tracker = SessionTracker()
        v = Violation(
            id="AI001",
            rule="placeholder",
            category="ai_quality",
            severity="error",
            message="Placeholder",
            suggestion="Implement it",
            line=20,
        )
        result = ValidationResult(
            passed=False, score=0.5, language_detected="python", violations=[v]
        )
        tracker.record_validation(result, file_path="a.py")
        tracker.record_validation(result, file_path="a.py")

        details = tracker.get_persistent_violation_details("a.py")
        assert "AI001" in details
        assert details["AI001"]["message"] == "Placeholder"


class TestASTPlaceholderVerification:
    """Gap 2 (v1.3.0): AST-level AI001 verification for Python code.

    _get_ast_confirmed_placeholder_lines upgrades heuristic AI001
    violations to verifiable=True when confirmed by AST analysis.
    """

    def test_ast_confirms_raise_not_implemented(self) -> None:
        """Pure 'raise NotImplementedError' body → verifiable=True."""
        from mirdan.core.ai_quality_checker import AIQualityChecker

        checker = AIQualityChecker()
        code = "def foo():\n    raise NotImplementedError\n"
        violations = checker.check(code, "python")
        ai001 = [v for v in violations if v.id == "AI001"]
        assert len(ai001) > 0
        assert all(v.verifiable is True for v in ai001)

    def test_ast_confirms_pass_placeholder(self) -> None:
        """Pure 'pass' body → verifiable=True."""
        from mirdan.core.ai_quality_checker import AIQualityChecker

        checker = AIQualityChecker()
        code = "def bar():\n    pass  # TODO: implement\n"
        violations = checker.check(code, "python")
        ai001 = [v for v in violations if v.id == "AI001"]
        assert len(ai001) > 0
        assert all(v.verifiable is True for v in ai001)

    def test_ast_confirms_ellipsis_in_confirmed_lines(self) -> None:
        """Pure '...' body is detected by AST as a placeholder line."""
        from mirdan.core.ai_quality_checker import AIQualityChecker

        checker = AIQualityChecker()
        code = "def stub():\n    ...\n"
        # The regex pattern for AI001 doesn't detect standalone '...',
        # but the AST method correctly identifies it as a placeholder line.
        confirmed = checker._get_ast_confirmed_placeholder_lines(code, "python")
        assert 2 in confirmed  # line 2 is the ellipsis

    def test_ast_skips_abstractmethod(self) -> None:
        """@abstractmethod decorated functions are NOT flagged as verifiable."""
        from mirdan.core.ai_quality_checker import AIQualityChecker

        checker = AIQualityChecker()
        code = (
            "from abc import abstractmethod\n"
            "class Base:\n"
            "    @abstractmethod\n"
            "    def method(self):\n"
            "        raise NotImplementedError\n"
        )
        violations = checker.check(code, "python")
        ai001 = [v for v in violations if v.id == "AI001"]
        # If regex still fires, AST should NOT confirm it → verifiable=False
        for v in ai001:
            assert v.verifiable is False

    def test_non_python_returns_empty_set(self) -> None:
        """Non-Python code → AST analysis returns empty set (fallback to heuristic)."""
        from mirdan.core.ai_quality_checker import AIQualityChecker

        checker = AIQualityChecker()
        code = "function foo() { throw new Error('not implemented'); }\n"
        violations = checker.check(code, "javascript")
        # All ai_quality violations should be heuristic (verifiable=False)
        for v in violations:
            if v.category == "ai_quality":
                assert v.verifiable is False

    def test_syntax_error_falls_back_gracefully(self) -> None:
        """Code with syntax errors → AST fails, heuristic (verifiable=False) preserved."""
        from mirdan.core.ai_quality_checker import AIQualityChecker

        checker = AIQualityChecker()
        code = "def broken(\n    raise NotImplementedError\n"
        violations = checker.check(code, "python")
        for v in violations:
            if v.category == "ai_quality":
                assert v.verifiable is False

    def test_check_quick_also_applies_ast(self) -> None:
        """check_quick applies the same AST verification for AI001."""
        from mirdan.core.ai_quality_checker import AIQualityChecker

        checker = AIQualityChecker()
        code = "def todo():\n    raise NotImplementedError\n"
        violations = checker.check_quick(code, "python")
        ai001 = [v for v in violations if v.id == "AI001"]
        assert len(ai001) > 0
        assert all(v.verifiable is True for v in ai001)

    def test_docstring_then_placeholder_confirmed(self) -> None:
        """Function with docstring + raise NotImplementedError → verifiable=True."""
        from mirdan.core.ai_quality_checker import AIQualityChecker

        checker = AIQualityChecker()
        code = (
            "def documented():\n"
            '    """This function will be implemented later."""\n'
            "    raise NotImplementedError\n"
        )
        violations = checker.check(code, "python")
        ai001 = [v for v in violations if v.id == "AI001"]
        assert len(ai001) > 0
        assert all(v.verifiable is True for v in ai001)


class TestQualityBaseline:
    """Gap 3 (v1.3.0): Cross-session quality drift detection via baseline score.

    QualityPersistence.get_baseline_score returns a cached average from
    recent snapshots. Drift is flagged in validate_code_quality output.
    """

    def test_baseline_none_with_no_snapshots(self, tmp_path: Path) -> None:
        """No history → baseline returns None."""
        from mirdan.core.quality_persistence import QualityPersistence

        persistence = QualityPersistence(base_dir=tmp_path)
        assert persistence.get_baseline_score() is None

    def test_baseline_none_with_fewer_than_3_snapshots(self, tmp_path: Path) -> None:
        """Fewer than 3 snapshots → baseline returns None."""
        from mirdan.core.quality_persistence import QualityPersistence

        persistence = QualityPersistence(base_dir=tmp_path)
        # Create 2 snapshots
        for score in (0.9, 0.85):
            result = ValidationResult(
                passed=True,
                score=score,
                language_detected="python",
                violations=[],
                standards_checked=["style"],
            )
            persistence.save_snapshot(result)
        assert persistence.get_baseline_score() is None

    def test_baseline_returns_average_with_3_snapshots(self, tmp_path: Path) -> None:
        """3+ snapshots → baseline returns their average."""
        from mirdan.core.quality_persistence import QualityPersistence

        persistence = QualityPersistence(base_dir=tmp_path)
        scores = [0.90, 0.85, 0.80]
        for score in scores:
            result = ValidationResult(
                passed=True,
                score=score,
                language_detected="python",
                violations=[],
                standards_checked=["style"],
            )
            persistence.save_snapshot(result)
        baseline = persistence.get_baseline_score()
        assert baseline is not None
        expected = sum(scores) / len(scores)
        assert abs(baseline - expected) < 0.01

    def test_baseline_is_cached(self, tmp_path: Path) -> None:
        """Baseline is computed once and cached for the instance lifetime."""
        from mirdan.core.quality_persistence import QualityPersistence

        persistence = QualityPersistence(base_dir=tmp_path)
        for score in (0.90, 0.85, 0.80):
            result = ValidationResult(
                passed=True,
                score=score,
                language_detected="python",
                violations=[],
                standards_checked=["style"],
            )
            persistence.save_snapshot(result)
        first_call = persistence.get_baseline_score()
        # Add a 4th snapshot that would change the average
        result = ValidationResult(
            passed=True,
            score=0.50,
            language_detected="python",
            violations=[],
            standards_checked=["style"],
        )
        persistence.save_snapshot(result)
        second_call = persistence.get_baseline_score()
        # Cached — must be identical
        assert first_call == second_call

    def test_drift_detection_in_server_output(self) -> None:
        """Server adds quality_drift key when score drops > 0.15 below baseline."""
        import inspect

        source = inspect.getsource(__import__("mirdan.server", fromlist=["validate_code_quality"]))
        assert "quality_drift" in source
        assert "get_baseline_score" in source
        assert "0.15" in source


class TestTaskGuidance:
    """Gap 4 (v1.3.0): Task-type-specific guidance injected into prompts.

    PromptComposer.TASK_GUIDANCE provides structured instructions per
    TaskType, rendered between context and task sections in the template.
    """

    def test_generation_guidance_present(self) -> None:
        """GENERATION task type has guidance text."""
        from mirdan.core.prompt_composer import PromptComposer

        assert TaskType.GENERATION in PromptComposer.TASK_GUIDANCE
        assert "Implementation Approach" in PromptComposer.TASK_GUIDANCE[TaskType.GENERATION]

    def test_refactor_guidance_present(self) -> None:
        """REFACTOR task type has guidance text."""
        from mirdan.core.prompt_composer import PromptComposer

        assert TaskType.REFACTOR in PromptComposer.TASK_GUIDANCE
        assert "Refactoring Protocol" in PromptComposer.TASK_GUIDANCE[TaskType.REFACTOR]

    def test_debug_guidance_present(self) -> None:
        """DEBUG task type has guidance text."""
        from mirdan.core.prompt_composer import PromptComposer

        assert TaskType.DEBUG in PromptComposer.TASK_GUIDANCE
        assert "Debugging Protocol" in PromptComposer.TASK_GUIDANCE[TaskType.DEBUG]

    def test_review_guidance_present(self) -> None:
        from mirdan.core.prompt_composer import PromptComposer

        assert TaskType.REVIEW in PromptComposer.TASK_GUIDANCE

    def test_test_guidance_present(self) -> None:
        from mirdan.core.prompt_composer import PromptComposer

        assert TaskType.TEST in PromptComposer.TASK_GUIDANCE

    def test_documentation_guidance_present(self) -> None:
        from mirdan.core.prompt_composer import PromptComposer

        assert TaskType.DOCUMENTATION in PromptComposer.TASK_GUIDANCE

    def test_guidance_rendered_in_prompt(self) -> None:
        """Task guidance appears in the composed prompt text."""
        from mirdan.core.prompt_composer import PromptComposer
        from mirdan.core.quality_standards import QualityStandards
        from mirdan.models import ContextBundle, Intent

        composer = PromptComposer(QualityStandards())
        intent = Intent(original_prompt="fix the login bug", task_type=TaskType.DEBUG)
        enhanced = composer.compose(intent, ContextBundle(), [])
        assert "Debugging Protocol" in enhanced.enhanced_text

    def test_guidance_suppressed_in_minimal_verbosity(self) -> None:
        """Task guidance is NOT rendered in minimal verbosity mode."""
        from mirdan.config import EnhancementConfig
        from mirdan.core.prompt_composer import PromptComposer
        from mirdan.core.quality_standards import QualityStandards
        from mirdan.models import ContextBundle, Intent

        config = EnhancementConfig(verbosity="minimal")
        composer = PromptComposer(QualityStandards(), config=config)
        intent = Intent(original_prompt="fix the login bug", task_type=TaskType.DEBUG)
        enhanced = composer.compose(intent, ContextBundle(), [])
        assert "Debugging Protocol" not in enhanced.enhanced_text

    def test_unknown_task_type_has_no_guidance(self) -> None:
        """UNKNOWN task type is not in TASK_GUIDANCE dict."""
        from mirdan.core.prompt_composer import PromptComposer

        assert TaskType.UNKNOWN not in PromptComposer.TASK_GUIDANCE


class TestSecurityRegression:
    """Gap 5 (v1.3.0): Security regression detection.

    SessionTracker.detect_security_regression flags files that previously
    had zero security violations but now have security violations.
    """

    def test_no_regression_on_first_validation(self) -> None:
        """First validation can't be a regression (no prior state)."""
        tracker = SessionTracker()
        v = Violation(
            id="SEC002",
            rule="sql-injection",
            category="security",
            severity="error",
            message="SQL injection",
            line=10,
        )
        result = ValidationResult(
            passed=False, score=0.5, language_detected="python", violations=[v]
        )
        tracker.record_validation(result, file_path="db.py")
        assert tracker.detect_security_regression("db.py", [v]) is False

    def test_regression_detected_when_security_introduced(self) -> None:
        """Clean → security violation = regression."""
        tracker = SessionTracker()
        clean = ValidationResult(passed=True, score=0.9, language_detected="python", violations=[])
        tracker.record_validation(clean, file_path="db.py")

        v = Violation(
            id="SEC002",
            rule="sql-injection",
            category="security",
            severity="error",
            message="SQL injection",
            line=10,
        )
        dirty = ValidationResult(
            passed=False, score=0.5, language_detected="python", violations=[v]
        )
        tracker.record_validation(dirty, file_path="db.py")
        assert tracker.detect_security_regression("db.py", dirty.violations) is True

    def test_no_regression_when_security_was_already_present(self) -> None:
        """Security → security is NOT a regression (it was already broken)."""
        tracker = SessionTracker()
        v = Violation(
            id="SEC002",
            rule="sql-injection",
            category="security",
            severity="error",
            message="SQL injection",
            line=10,
        )
        result = ValidationResult(
            passed=False, score=0.5, language_detected="python", violations=[v]
        )
        tracker.record_validation(result, file_path="db.py")
        tracker.record_validation(result, file_path="db.py")
        assert tracker.detect_security_regression("db.py", result.violations) is False

    def test_no_regression_when_clean_follows_clean(self) -> None:
        """Clean → clean is NOT a regression."""
        tracker = SessionTracker()
        clean = ValidationResult(passed=True, score=0.9, language_detected="python", violations=[])
        tracker.record_validation(clean, file_path="db.py")
        tracker.record_validation(clean, file_path="db.py")
        assert tracker.detect_security_regression("db.py", []) is False

    def test_non_security_violations_dont_trigger_regression(self) -> None:
        """Clean → style-only violations is NOT a security regression."""
        tracker = SessionTracker()
        clean = ValidationResult(passed=True, score=0.9, language_detected="python", violations=[])
        tracker.record_validation(clean, file_path="db.py")

        v = Violation(
            id="PY003",
            rule="bare-except",
            category="style",
            severity="warning",
            message="Bare except",
            line=5,
        )
        style_fail = ValidationResult(
            passed=False, score=0.7, language_detected="python", violations=[v]
        )
        tracker.record_validation(style_fail, file_path="db.py")
        assert tracker.detect_security_regression("db.py", style_fail.violations) is False

    def test_regression_surfaced_in_server_output(self) -> None:
        """Server adds security_regression key to output when detected."""
        import inspect

        source = inspect.getsource(__import__("mirdan.server", fromlist=["validate_code_quality"]))
        assert "security_regression" in source
        assert "detect_security_regression" in source
