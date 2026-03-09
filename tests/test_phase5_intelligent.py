"""Tests for Phase 5: Make It Intelligent — explanations, forecasting, coordination, cross-project."""

from __future__ import annotations

from mirdan.core.cross_project import (
    CrossProjectIntelligence,
    QualityComparison,
    Suggestion,
)
from mirdan.core.quality_coordinator import QualityCoordinator
from mirdan.core.quality_forecaster import (
    QualityForecast,
    QualityForecaster,
    RegressionAlert,
)
from mirdan.core.violation_explainer import ViolationExplainer
from mirdan.models import Violation

# ── 5A: Contextual Violation Explanations ───────────────────────────


class TestViolationExplainer:
    """Tests for ViolationExplainer."""

    def test_explain_known_rule(self) -> None:
        explainer = ViolationExplainer()
        v = Violation(
            id="SEC001",
            rule="hardcoded-secret",
            category="security",
            severity="error",
            message="Found hardcoded API key",
        )
        explanation = explainer.explain(v)
        assert "hardcoded" in explanation.lower() or "secret" in explanation.lower()
        assert "security" in explanation.lower()

    def test_explain_unknown_rule(self) -> None:
        explainer = ViolationExplainer()
        v = Violation(
            id="CUSTOM999",
            rule="custom",
            category="style",
            severity="info",
            message="Custom violation",
        )
        explanation = explainer.explain(v)
        assert len(explanation) > 0

    def test_explain_with_override_counts(self) -> None:
        explainer = ViolationExplainer()
        v = Violation(
            id="PY001",
            rule="bare-except",
            category="style",
            severity="warning",
            message="Bare except",
        )
        explanation = explainer.explain(
            v,
            override_counts={"PY001": 7},
        )
        assert "overridden" in explanation.lower()
        assert "7" in explanation

    def test_explain_with_conventions(self) -> None:
        explainer = ViolationExplainer()
        v = Violation(
            id="SEC002",
            rule="sql-injection",
            category="security",
            severity="error",
            message="SQL injection risk",
        )
        conventions = [
            {"name": "use-parameterized-queries", "category": "security", "tags": []},
        ]
        explanation = explainer.explain(v, conventions=conventions)
        assert "convention" in explanation.lower() or "parameterized" in explanation.lower()

    def test_explain_dict_input(self) -> None:
        explainer = ViolationExplainer()
        v_dict = {
            "id": "AI001",
            "category": "ai_quality",
            "severity": "warning",
            "message": "Placeholder code",
        }
        explanation = explainer.explain(v_dict)
        assert "placeholder" in explanation.lower() or "todo" in explanation.lower()


class TestViolationExplainerEnrich:
    """Tests for ViolationExplainer.enrich_violations."""

    def test_enrich_sets_explanation(self) -> None:
        explainer = ViolationExplainer()
        violations = [
            Violation(
                id="SEC001",
                rule="secret",
                category="security",
                severity="error",
                message="Found secret",
            ),
        ]
        explainer.enrich_violations(violations)
        assert violations[0].explanation != ""
        assert "security" in violations[0].explanation.lower()

    def test_enrich_sets_related_violations(self) -> None:
        explainer = ViolationExplainer()
        violations = [
            Violation(id="SEC001", rule="a", category="security", severity="error", message="a"),
            Violation(id="SEC002", rule="b", category="security", severity="error", message="b"),
            Violation(id="PY001", rule="c", category="style", severity="warning", message="c"),
        ]
        explainer.enrich_violations(violations)
        # SEC001 should list SEC002 as related (same category)
        assert "SEC002" in violations[0].related_violations
        assert "SEC001" in violations[1].related_violations
        # PY001 should not have SEC violations as related
        assert "SEC001" not in violations[2].related_violations

    def test_enrich_sets_historical_frequency(self) -> None:
        explainer = ViolationExplainer()
        violations = [
            Violation(
                id="PY002",
                rule="mutable-default",
                category="style",
                severity="warning",
                message="m",
            ),
        ]
        explainer.enrich_violations(violations, override_counts={"PY002": 3})
        assert violations[0].historical_frequency == 3

    def test_enrich_empty_list(self) -> None:
        explainer = ViolationExplainer()
        violations: list[Violation] = []
        explainer.enrich_violations(violations)  # Should not raise


class TestViolationModelExtensions:
    """Tests for Violation model extensions (Phase 5A)."""

    def test_violation_has_explanation_field(self) -> None:
        v = Violation(
            id="T1",
            rule="test",
            category="style",
            severity="info",
            message="test",
            explanation="This is an explanation",
        )
        assert v.explanation == "This is an explanation"

    def test_violation_has_related_violations(self) -> None:
        v = Violation(
            id="T1",
            rule="test",
            category="style",
            severity="info",
            message="test",
            related_violations=["T2", "T3"],
        )
        assert v.related_violations == ["T2", "T3"]

    def test_violation_has_historical_frequency(self) -> None:
        v = Violation(
            id="T1",
            rule="test",
            category="style",
            severity="info",
            message="test",
            historical_frequency=5,
        )
        assert v.historical_frequency == 5

    def test_violation_to_dict_includes_explanation(self) -> None:
        v = Violation(
            id="T1",
            rule="test",
            category="style",
            severity="info",
            message="test",
            explanation="Explained",
        )
        d = v.to_dict()
        assert d["explanation"] == "Explained"

    def test_violation_to_dict_excludes_empty_explanation(self) -> None:
        v = Violation(
            id="T1",
            rule="test",
            category="style",
            severity="info",
            message="test",
        )
        d = v.to_dict()
        assert "explanation" not in d

    def test_violation_to_dict_includes_related(self) -> None:
        v = Violation(
            id="T1",
            rule="test",
            category="style",
            severity="info",
            message="test",
            related_violations=["T2"],
        )
        d = v.to_dict()
        assert d["related_violations"] == ["T2"]

    def test_violation_to_dict_includes_frequency(self) -> None:
        v = Violation(
            id="T1",
            rule="test",
            category="style",
            severity="info",
            message="test",
            historical_frequency=3,
        )
        d = v.to_dict()
        assert d["historical_frequency"] == 3


# ── 5D: Quality Forecasting ─────────────────────────────────────────


class TestQualityForecaster:
    """Tests for QualityForecaster."""

    def test_forecast_improving(self) -> None:
        forecaster = QualityForecaster()
        snapshots = [{"score": 0.6 + i * 0.02} for i in range(20)]
        forecast = forecaster.forecast(snapshots, days_ahead=7)
        assert isinstance(forecast, QualityForecast)
        assert forecast.direction == "improving"
        assert forecast.predicted_score > forecast.current_score
        assert forecast.slope > 0

    def test_forecast_declining(self) -> None:
        forecaster = QualityForecaster()
        snapshots = [{"score": 0.9 - i * 0.02} for i in range(20)]
        forecast = forecaster.forecast(snapshots, days_ahead=7)
        assert forecast.direction == "declining"
        assert forecast.predicted_score < forecast.current_score

    def test_forecast_stable(self) -> None:
        forecaster = QualityForecaster()
        snapshots = [{"score": 0.85} for _ in range(10)]
        forecast = forecaster.forecast(snapshots, days_ahead=7)
        assert forecast.direction == "stable"
        assert abs(forecast.predicted_score - 0.85) < 0.01

    def test_forecast_single_snapshot(self) -> None:
        forecaster = QualityForecaster()
        forecast = forecaster.forecast([{"score": 0.7}])
        assert forecast.confidence == 0.0
        assert forecast.current_score == 0.7
        assert forecast.direction == "stable"

    def test_forecast_empty(self) -> None:
        forecaster = QualityForecaster()
        forecast = forecaster.forecast([])
        assert forecast.confidence == 0.0
        assert forecast.current_score == 0.0

    def test_forecast_clamped_to_bounds(self) -> None:
        forecaster = QualityForecaster()
        # Very steep improving trend
        snapshots = [{"score": 0.5 + i * 0.1} for i in range(10)]
        forecast = forecaster.forecast(snapshots, days_ahead=100)
        assert forecast.predicted_score <= 1.0

    def test_forecast_to_dict(self) -> None:
        forecaster = QualityForecaster()
        snapshots = [{"score": 0.8 + i * 0.01} for i in range(5)]
        forecast = forecaster.forecast(snapshots, days_ahead=3)
        d = forecast.to_dict()
        assert "current_score" in d
        assert "predicted_score" in d
        assert "confidence" in d
        assert "direction" in d

    def test_detect_regression_critical(self) -> None:
        forecaster = QualityForecaster()
        # Earlier scores high, recent scores much lower
        snapshots = [{"score": 0.9}] * 10 + [{"score": 0.5}] * 7
        alerts = forecaster.detect_regression(snapshots, window=7)
        assert len(alerts) >= 1
        assert alerts[0].severity == "critical"
        assert isinstance(alerts[0], RegressionAlert)

    def test_detect_regression_warning(self) -> None:
        forecaster = QualityForecaster()
        snapshots = [{"score": 0.9}] * 10 + [{"score": 0.82}] * 7
        alerts = forecaster.detect_regression(snapshots, window=7)
        assert len(alerts) >= 1
        assert alerts[0].severity == "warning"

    def test_detect_no_regression(self) -> None:
        forecaster = QualityForecaster()
        snapshots = [{"score": 0.85}] * 20
        alerts = forecaster.detect_regression(snapshots)
        assert len(alerts) == 0

    def test_detect_regression_insufficient_data(self) -> None:
        forecaster = QualityForecaster()
        alerts = forecaster.detect_regression([{"score": 0.5}] * 3, window=7)
        assert len(alerts) == 0

    def test_calculate_velocity_improving(self) -> None:
        forecaster = QualityForecaster()
        snapshots = [{"score": 0.5 + i * 0.05} for i in range(10)]
        velocity = forecaster.calculate_velocity(snapshots)
        assert velocity > 0

    def test_calculate_velocity_stable(self) -> None:
        forecaster = QualityForecaster()
        snapshots = [{"score": 0.8}] * 10
        velocity = forecaster.calculate_velocity(snapshots)
        assert abs(velocity) < 0.001

    def test_calculate_velocity_single(self) -> None:
        forecaster = QualityForecaster()
        velocity = forecaster.calculate_velocity([{"score": 0.8}])
        assert velocity == 0.0

    def test_regression_alert_to_dict(self) -> None:
        alert = RegressionAlert(
            severity="warning",
            message="Quality declined",
            score_drop=0.1,
            period_days=7,
        )
        d = alert.to_dict()
        assert d["severity"] == "warning"
        assert d["score_drop"] == 0.1


# ── 5C: Cross-Project Quality Intelligence ──────────────────────────


class TestCrossProjectIntelligence:
    """Tests for CrossProjectIntelligence."""

    def test_discover_workspace_patterns(self) -> None:
        intel = CrossProjectIntelligence()
        entries = [
            {"content": "Use ruff for linting", "tags": ["linting"], "scope_path": "/project-a"},
            {"content": "Use ruff for linting", "tags": ["linting"], "scope_path": "/project-b"},
            {"content": "Use pytest", "tags": ["testing"], "scope_path": "/project-a"},
        ]
        patterns = intel.discover_workspace_patterns(entries)
        # "linting" tag appears in 2 projects
        assert len(patterns) >= 1
        linting_pattern = next((p for p in patterns if "linting" in p.get("tags", [])), None)
        assert linting_pattern is not None
        assert linting_pattern["scope"] == "workspace"

    def test_discover_no_cross_project_patterns(self) -> None:
        intel = CrossProjectIntelligence()
        entries = [
            {"content": "pattern A", "tags": ["a"], "scope_path": "/project-a"},
            {"content": "pattern B", "tags": ["b"], "scope_path": "/project-b"},
        ]
        patterns = intel.discover_workspace_patterns(entries)
        # No tags appear in multiple projects
        assert len(patterns) == 0

    def test_suggest_conventions(self) -> None:
        intel = CrossProjectIntelligence()
        project_entries = [
            {"content": "Use pytest", "content_type": "convention", "tags": ["testing"]},
        ]
        workspace_entries = [
            {
                "content": "Use ruff for linting",
                "content_type": "convention",
                "tags": ["linting"],
                "confidence": 0.9,
                "source_projects": ["/a", "/b"],
            },
        ]
        suggestions = intel.suggest_conventions(project_entries, workspace_entries)
        assert len(suggestions) >= 1
        assert isinstance(suggestions[0], Suggestion)
        assert suggestions[0].confidence == 0.9

    def test_suggest_no_new_conventions(self) -> None:
        intel = CrossProjectIntelligence()
        project_entries = [
            {"content": "Use ruff", "content_type": "convention", "tags": ["linting"]},
        ]
        workspace_entries = [
            {"content": "Use ruff", "content_type": "convention", "tags": ["linting"]},
        ]
        suggestions = intel.suggest_conventions(project_entries, workspace_entries)
        assert len(suggestions) == 0

    def test_compare_quality(self) -> None:
        intel = CrossProjectIntelligence()
        projects = [
            {"name": "project-a", "avg_score": 0.9, "violations": ["PY001", "SEC001"]},
            {"name": "project-b", "avg_score": 0.7, "violations": ["PY001", "AI002"]},
            {"name": "project-c", "avg_score": 0.8, "violations": ["PY001", "SEC003"]},
        ]
        comparison = intel.compare_quality(projects)
        assert isinstance(comparison, QualityComparison)
        assert comparison.best_project == "project-a"
        assert comparison.worst_project == "project-b"
        assert "PY001" in comparison.common_violations  # Common to all 3

    def test_compare_quality_empty(self) -> None:
        intel = CrossProjectIntelligence()
        comparison = intel.compare_quality([])
        assert comparison.best_project == ""
        assert comparison.avg_score == 0.0

    def test_compare_quality_to_dict(self) -> None:
        intel = CrossProjectIntelligence()
        projects = [
            {"name": "a", "avg_score": 0.9},
            {"name": "b", "avg_score": 0.7},
        ]
        comparison = intel.compare_quality(projects)
        d = comparison.to_dict()
        assert "best_project" in d
        assert "avg_score" in d

    def test_suggestion_to_dict(self) -> None:
        s = Suggestion(
            convention="Use ruff",
            confidence=0.85,
            source_projects=["/a", "/b"],
            category="style",
        )
        d = s.to_dict()
        assert d["confidence"] == 0.85
        assert len(d["source_projects"]) == 2


# ── 5B: Multi-Agent Quality Coordination ────────────────────────────


class TestQualityCoordinator:
    """Tests for QualityCoordinator."""

    def test_register_agent(self) -> None:
        coord = QualityCoordinator()
        result = coord.register_agent("agent-1", ["python", "security"])
        assert result.success is True
        assert result.data["agent_count"] == 1

    def test_register_duplicate(self) -> None:
        coord = QualityCoordinator()
        coord.register_agent("agent-1")
        result = coord.register_agent("agent-1")
        assert result.success is True
        assert "already" in result.message.lower()

    def test_unregister_agent(self) -> None:
        coord = QualityCoordinator()
        coord.register_agent("agent-1")
        result = coord.unregister_agent("agent-1")
        assert result.success is True

    def test_unregister_nonexistent(self) -> None:
        coord = QualityCoordinator()
        result = coord.unregister_agent("nope")
        assert result.success is False

    async def test_claim_file(self) -> None:
        coord = QualityCoordinator()
        coord.register_agent("agent-1")
        claimed = await coord.claim_file("agent-1", "src/app.py")
        assert claimed is True

    async def test_claim_file_already_claimed(self) -> None:
        coord = QualityCoordinator()
        coord.register_agent("agent-1")
        coord.register_agent("agent-2")
        await coord.claim_file("agent-1", "src/app.py")
        claimed = await coord.claim_file("agent-2", "src/app.py")
        assert claimed is False

    async def test_claim_unregistered_agent(self) -> None:
        coord = QualityCoordinator()
        claimed = await coord.claim_file("unknown", "src/app.py")
        assert claimed is False

    async def test_release_file(self) -> None:
        coord = QualityCoordinator()
        coord.register_agent("agent-1")
        await coord.claim_file("agent-1", "src/app.py")
        await coord.release_file("agent-1", "src/app.py", {"score": 0.9})
        # File should now be available for reclaim
        status = coord.get_status()
        assert len(status["active_claims"]) == 0
        assert status["completed_files"] == 1

    async def test_get_unassigned_files(self) -> None:
        coord = QualityCoordinator()
        coord.register_agent("agent-1")
        await coord.claim_file("agent-1", "src/a.py")

        changed = ["src/a.py", "src/b.py", "src/c.py"]
        unassigned = coord.get_unassigned_files(changed)
        assert "src/a.py" not in unassigned
        assert "src/b.py" in unassigned
        assert "src/c.py" in unassigned

    async def test_aggregate_results(self) -> None:
        coord = QualityCoordinator()
        coord.register_agent("agent-1")
        await coord.claim_file("agent-1", "a.py")
        await coord.release_file(
            "agent-1",
            "a.py",
            {"score": 0.8, "violations": [{"id": "X"}]},
        )
        await coord.claim_file("agent-1", "b.py")
        await coord.release_file(
            "agent-1",
            "b.py",
            {"score": 0.9, "violations": []},
        )
        agg = coord.aggregate_results()
        assert agg["files_validated"] == 2
        assert agg["avg_score"] == 0.85
        assert agg["total_violations"] == 1

    def test_aggregate_empty(self) -> None:
        coord = QualityCoordinator()
        agg = coord.aggregate_results()
        assert agg["files_validated"] == 0
        assert agg["avg_score"] == 0.0

    def test_get_status(self) -> None:
        coord = QualityCoordinator()
        coord.register_agent("agent-1", ["python"])
        status = coord.get_status()
        assert status["agent_count"] == 1
        assert len(status["agents"]) == 1
        assert status["agents"][0]["id"] == "agent-1"
        assert "python" in status["agents"][0]["capabilities"]

    async def test_unregister_releases_claims(self) -> None:
        coord = QualityCoordinator()
        coord.register_agent("agent-1")
        await coord.claim_file("agent-1", "src/x.py")
        result = coord.unregister_agent("agent-1")
        assert "src/x.py" in result.data["released_files"]
        # File should be available again
        assert coord.get_unassigned_files(["src/x.py"]) == ["src/x.py"]

    def test_coordination_result_to_dict(self) -> None:
        from mirdan.core.quality_coordinator import CoordinationResult

        result = CoordinationResult(
            success=True,
            message="OK",
            data={"count": 1},
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["count"] == 1


# ── 5E: GA Polish ───────────────────────────────────────────────────


class TestGAPolish:
    """Tests for GA version and classifiers."""

    def test_version_is_1_5_0(self) -> None:
        from mirdan import __version__

        assert __version__ == "1.5.0"

    def test_classifier_is_stable(self) -> None:
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text())
        classifiers = data["project"]["classifiers"]
        assert any("Production/Stable" in c for c in classifiers)


# ── Integration: Forecaster wired into get_quality_trends ────────────


class TestForecastInTrends:
    """Tests that get_quality_trends includes forecast data."""

    def test_get_quality_trends_has_format_param(self) -> None:
        import inspect

        import mirdan.server as server_mod

        fn = server_mod.get_quality_trends
        underlying = getattr(fn, "fn", fn)
        sig = inspect.signature(underlying)  # type: ignore[arg-type]
        assert "format" in sig.parameters


# ── Integration: Explainer wired into server ──────────────────────────


class TestExplainerInServer:
    """Tests that violation_explainer is wired into _Components."""

    def test_components_has_violation_explainer(self) -> None:
        import inspect

        import mirdan.server as server_mod

        source = inspect.getsource(server_mod._Components)
        assert "violation_explainer" in source

    def test_explainer_created(self) -> None:
        import mirdan.server as server_mod

        assert hasattr(server_mod, "_create_violation_explainer")
        explainer = server_mod._create_violation_explainer()
        assert hasattr(explainer, "enrich_violations")
