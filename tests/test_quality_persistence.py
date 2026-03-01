"""Tests for the quality persistence module."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mirdan.core.quality_persistence import QualityPersistence, _calculate_trend
from mirdan.models import QualitySnapshot, ValidationResult, Violation


def _make_result(
    score: float = 0.9,
    passed: bool = True,
    language: str = "python",
    violations: list[Violation] | None = None,
) -> ValidationResult:
    """Create a ValidationResult for testing."""
    return ValidationResult(
        passed=passed,
        score=score,
        language_detected=language,
        violations=violations or [],
        standards_checked=["security", "style"],
    )


class TestSaveSnapshot:
    """Tests for saving quality snapshots."""

    def test_save_creates_history_dir(self, tmp_path: Path) -> None:
        """Should create .mirdan/history/ directory."""
        persistence = QualityPersistence(base_dir=tmp_path)
        result = _make_result()
        persistence.save_snapshot(result)

        assert persistence.history_dir.exists()

    def test_save_creates_json_file(self, tmp_path: Path) -> None:
        """Should create a dated JSON file."""
        persistence = QualityPersistence(base_dir=tmp_path)
        result = _make_result()
        persistence.save_snapshot(result)

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        json_file = persistence.history_dir / f"{today}.json"
        assert json_file.exists()

    def test_save_returns_snapshot(self, tmp_path: Path) -> None:
        """Should return a QualitySnapshot with correct data."""
        persistence = QualityPersistence(base_dir=tmp_path)
        result = _make_result(score=0.85, language="typescript")
        snapshot = persistence.save_snapshot(result, project_path="/my/project")

        assert isinstance(snapshot, QualitySnapshot)
        assert snapshot.score == 0.85
        assert snapshot.language == "typescript"
        assert snapshot.project_path == "/my/project"
        assert snapshot.passed is True

    def test_save_counts_violations(self, tmp_path: Path) -> None:
        """Should correctly count violations by severity."""
        violations = [
            Violation(id="E1", rule="r1", category="security", severity="error", message="e"),
            Violation(id="W1", rule="r2", category="style", severity="warning", message="w"),
            Violation(id="W2", rule="r3", category="style", severity="warning", message="w2"),
            Violation(id="I1", rule="r4", category="style", severity="info", message="i"),
        ]
        result = _make_result(passed=False, violations=violations)
        persistence = QualityPersistence(base_dir=tmp_path)
        snapshot = persistence.save_snapshot(result)

        assert snapshot.violation_counts["error"] == 1
        assert snapshot.violation_counts["warning"] == 2
        assert snapshot.violation_counts["info"] == 1

    def test_save_appends_to_existing_file(self, tmp_path: Path) -> None:
        """Should append to existing day file, not overwrite."""
        persistence = QualityPersistence(base_dir=tmp_path)
        persistence.save_snapshot(_make_result(score=0.9))
        persistence.save_snapshot(_make_result(score=0.8))

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        json_file = persistence.history_dir / f"{today}.json"
        with json_file.open() as f:
            entries = json.load(f)
        assert len(entries) == 2

    def test_save_handles_corrupted_file(self, tmp_path: Path) -> None:
        """Should handle corrupted existing JSON file."""
        persistence = QualityPersistence(base_dir=tmp_path)
        persistence.history_dir.mkdir(parents=True)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        corrupted = persistence.history_dir / f"{today}.json"
        corrupted.write_text("not json {{{")

        # Should not raise, should start fresh
        snapshot = persistence.save_snapshot(_make_result())
        assert snapshot.score == 0.9


class TestGetSnapshots:
    """Tests for retrieving snapshots."""

    def test_get_snapshots_returns_saved(self, tmp_path: Path) -> None:
        """Should return previously saved snapshots."""
        persistence = QualityPersistence(base_dir=tmp_path)
        persistence.save_snapshot(_make_result(score=0.9))
        persistence.save_snapshot(_make_result(score=0.8))

        snapshots = persistence.get_snapshots(days=1)
        assert len(snapshots) == 2

    def test_get_snapshots_empty_dir(self, tmp_path: Path) -> None:
        """Should return empty list when no history exists."""
        persistence = QualityPersistence(base_dir=tmp_path)
        snapshots = persistence.get_snapshots()
        assert snapshots == []

    def test_get_snapshots_filters_by_project(self, tmp_path: Path) -> None:
        """Should filter by project_path when specified."""
        persistence = QualityPersistence(base_dir=tmp_path)
        persistence.save_snapshot(_make_result(), project_path="/project-a")
        persistence.save_snapshot(_make_result(), project_path="/project-b")

        snapshots = persistence.get_snapshots(project_path="/project-a")
        assert len(snapshots) == 1
        assert snapshots[0].project_path == "/project-a"

    def test_get_snapshots_sorted_oldest_first(self, tmp_path: Path) -> None:
        """Should return snapshots sorted oldest-first."""
        persistence = QualityPersistence(base_dir=tmp_path)
        persistence.save_snapshot(_make_result(score=0.9))
        persistence.save_snapshot(_make_result(score=0.8))

        snapshots = persistence.get_snapshots()
        # First saved should come first (older timestamp)
        assert snapshots[0].score == 0.9
        assert snapshots[1].score == 0.8

    def test_get_snapshots_skips_corrupted_files(self, tmp_path: Path) -> None:
        """Should skip corrupted files without crashing."""
        persistence = QualityPersistence(base_dir=tmp_path)
        persistence.save_snapshot(_make_result())

        # Create a corrupted file for yesterday
        yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
        corrupted = persistence.history_dir / f"{yesterday}.json"
        corrupted.write_text("corrupted!")

        snapshots = persistence.get_snapshots(days=7)
        assert len(snapshots) == 1  # Only today's valid snapshot


class TestGetTrends:
    """Tests for quality trend calculation."""

    def test_trends_empty_history(self, tmp_path: Path) -> None:
        """Should return zero-value trends when no history."""
        persistence = QualityPersistence(base_dir=tmp_path)
        trend = persistence.get_trends()

        assert trend.snapshot_count == 0
        assert trend.avg_score == 0.0
        assert trend.score_trend == "stable"

    def test_trends_single_snapshot(self, tmp_path: Path) -> None:
        """Should handle a single snapshot."""
        persistence = QualityPersistence(base_dir=tmp_path)
        persistence.save_snapshot(_make_result(score=0.85))

        trend = persistence.get_trends()
        assert trend.snapshot_count == 1
        assert trend.avg_score == 0.85
        assert trend.score_trend == "stable"

    def test_trends_calculates_averages(self, tmp_path: Path) -> None:
        """Should calculate correct average, min, max."""
        persistence = QualityPersistence(base_dir=tmp_path)
        persistence.save_snapshot(_make_result(score=0.7))
        persistence.save_snapshot(_make_result(score=0.8))
        persistence.save_snapshot(_make_result(score=0.9))

        trend = persistence.get_trends()
        assert trend.snapshot_count == 3
        assert abs(trend.avg_score - 0.8) < 0.01
        assert trend.min_score == 0.7
        assert trend.max_score == 0.9

    def test_trends_pass_rate(self, tmp_path: Path) -> None:
        """Should calculate correct pass rate."""
        persistence = QualityPersistence(base_dir=tmp_path)
        persistence.save_snapshot(_make_result(passed=True))
        persistence.save_snapshot(_make_result(passed=True))
        persistence.save_snapshot(_make_result(passed=False))

        trend = persistence.get_trends()
        assert abs(trend.pass_rate - 2 / 3) < 0.01

    def test_trends_to_dict(self, tmp_path: Path) -> None:
        """Should convert to dict correctly."""
        persistence = QualityPersistence(base_dir=tmp_path)
        persistence.save_snapshot(_make_result(score=0.9))

        trend = persistence.get_trends()
        d = trend.to_dict()
        assert "avg_score" in d
        assert "score_trend" in d
        assert "snapshots" in d
        assert len(d["snapshots"]) == 1


class TestCalculateTrend:
    """Tests for the trend calculation helper."""

    def test_improving_trend(self) -> None:
        scores = [0.6, 0.65, 0.7, 0.8, 0.85, 0.9]
        assert _calculate_trend(scores) == "improving"

    def test_declining_trend(self) -> None:
        scores = [0.9, 0.85, 0.8, 0.7, 0.65, 0.6]
        assert _calculate_trend(scores) == "declining"

    def test_stable_trend(self) -> None:
        scores = [0.8, 0.81, 0.79, 0.8, 0.8, 0.81]
        assert _calculate_trend(scores) == "stable"

    def test_single_value_is_stable(self) -> None:
        assert _calculate_trend([0.8]) == "stable"

    def test_empty_list_is_stable(self) -> None:
        assert _calculate_trend([]) == "stable"

    def test_two_values_improving(self) -> None:
        assert _calculate_trend([0.5, 0.9]) == "improving"

    def test_two_values_declining(self) -> None:
        assert _calculate_trend([0.9, 0.5]) == "declining"
