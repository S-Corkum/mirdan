"""Quality persistence — stores and queries quality snapshots over time.

Snapshots are stored as JSON files in `.mirdan/history/YYYY-MM-DD.json`.
Each file contains an array of snapshot objects for that day.
No external dependencies required.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mirdan.models import QualitySnapshot, QualityTrend, ValidationResult

logger = logging.getLogger(__name__)

_HISTORY_DIR = ".mirdan/history"


class QualityPersistence:
    """Persists validation results as quality snapshots.

    Stores snapshots as JSON in `.mirdan/history/YYYY-MM-DD.json`.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize with a base directory for history storage.

        Args:
            base_dir: Project root directory. Defaults to cwd.
        """
        self._base_dir = base_dir or Path.cwd()
        self._history_dir = self._base_dir / _HISTORY_DIR
        self._baseline_score: float | None = None
        self._baseline_loaded: bool = False

    @property
    def history_dir(self) -> Path:
        """Return the history directory path."""
        return self._history_dir

    def save_snapshot(
        self,
        result: ValidationResult,
        project_path: str = "",
    ) -> QualitySnapshot:
        """Save a validation result as a quality snapshot.

        Args:
            result: The validation result to persist.
            project_path: Optional project path identifier.

        Returns:
            The saved QualitySnapshot.
        """
        now = datetime.now(UTC)
        snapshot = QualitySnapshot(
            timestamp=now.isoformat(),
            project_path=project_path or str(self._base_dir),
            language=result.language_detected,
            score=result.score,
            passed=result.passed,
            violation_counts={
                "error": sum(1 for v in result.violations if v.severity == "error"),
                "warning": sum(1 for v in result.violations if v.severity == "warning"),
                "info": sum(1 for v in result.violations if v.severity == "info"),
            },
            standards_checked=list(result.standards_checked),
        )

        self._write_snapshot(snapshot, now)
        return snapshot

    def get_snapshots(
        self,
        days: int = 30,
        project_path: str | None = None,
    ) -> list[QualitySnapshot]:
        """Retrieve snapshots within a time range.

        Args:
            days: Number of days of history to retrieve.
            project_path: Optional filter by project path.

        Returns:
            List of QualitySnapshot objects, oldest first.
        """
        snapshots: list[QualitySnapshot] = []
        now = datetime.now(UTC)

        for day_offset in range(days):
            date = now - timedelta(days=day_offset)
            day_file = self._history_dir / f"{date.strftime('%Y-%m-%d')}.json"

            if not day_file.exists():
                continue

            try:
                with day_file.open() as f:
                    entries: list[dict[str, Any]] = json.load(f)

                for entry in entries:
                    snapshot = _dict_to_snapshot(entry)
                    if project_path and snapshot.project_path != project_path:
                        continue
                    snapshots.append(snapshot)

            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning("Skipping corrupted history file: %s", day_file)

        # Sort oldest-first
        snapshots.sort(key=lambda s: s.timestamp)
        return snapshots

    def get_trends(
        self,
        days: int = 30,
        project_path: str | None = None,
    ) -> QualityTrend:
        """Calculate quality trends from historical snapshots.

        Args:
            days: Number of days to analyze.
            project_path: Optional filter by project path.

        Returns:
            QualityTrend with aggregated statistics.
        """
        snapshots = self.get_snapshots(days=days, project_path=project_path)

        if not snapshots:
            return QualityTrend(
                project_path=project_path or str(self._base_dir),
                days=days,
                snapshot_count=0,
                avg_score=0.0,
                min_score=0.0,
                max_score=0.0,
                pass_rate=0.0,
                score_trend="stable",
                snapshots=[],
            )

        scores = [s.score for s in snapshots]
        passed_count = sum(1 for s in snapshots if s.passed)

        # Determine trend direction
        score_trend = _calculate_trend(scores)

        return QualityTrend(
            project_path=project_path or str(self._base_dir),
            days=days,
            snapshot_count=len(snapshots),
            avg_score=sum(scores) / len(scores),
            min_score=min(scores),
            max_score=max(scores),
            pass_rate=passed_count / len(snapshots),
            score_trend=score_trend,
            snapshots=snapshots,
        )

    def get_baseline_score(self, window: int = 10) -> float | None:
        """Get the average score from recent snapshots as a quality baseline.

        Used for cross-session quality drift detection. The baseline represents
        the project's historical quality level — if the current validation score
        drops significantly below it, a quality regression is flagged.

        Cached for the lifetime of this instance to avoid repeated disk reads
        and to keep the baseline stable within a session.

        Args:
            window: Maximum number of recent snapshots to average.

        Returns:
            Average score, or None if fewer than 3 snapshots exist.
        """
        if self._baseline_loaded:
            return self._baseline_score

        self._baseline_loaded = True
        snapshots = self.get_snapshots(days=7)
        if len(snapshots) < 3:
            self._baseline_score = None
            return None

        recent = snapshots[-window:]
        self._baseline_score = sum(s.score for s in recent) / len(recent)
        return self._baseline_score

    def _write_snapshot(self, snapshot: QualitySnapshot, now: datetime) -> None:
        """Write a snapshot to the day's JSON file.

        Args:
            snapshot: The snapshot to persist.
            now: Current datetime for file naming.
        """
        self._history_dir.mkdir(parents=True, exist_ok=True)
        day_file = self._history_dir / f"{now.strftime('%Y-%m-%d')}.json"

        existing: list[dict[str, Any]] = []
        if day_file.exists():
            try:
                with day_file.open() as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Corrupted history file, starting fresh: %s", day_file)
                existing = []

        existing.append(snapshot.to_dict())

        with day_file.open("w") as f:
            json.dump(existing, f, indent=2)


def _dict_to_snapshot(data: dict[str, Any]) -> QualitySnapshot:
    """Convert a dictionary to a QualitySnapshot.

    Args:
        data: Dictionary from JSON storage.

    Returns:
        QualitySnapshot instance.
    """
    return QualitySnapshot(
        timestamp=data["timestamp"],
        project_path=data.get("project_path", ""),
        language=data.get("language", ""),
        score=data.get("score", 0.0),
        passed=data.get("passed", False),
        violation_counts=data.get("violation_counts", {}),
        standards_checked=data.get("standards_checked", []),
    )


def _calculate_trend(scores: list[float]) -> str:
    """Determine whether scores are improving, declining, or stable.

    Compares the average of the first half to the second half.
    Requires at least 2 data points.

    Args:
        scores: List of scores in chronological order.

    Returns:
        "improving", "declining", or "stable".
    """
    if len(scores) < 2:
        return "stable"

    mid = len(scores) // 2
    first_half_avg = sum(scores[:mid]) / mid
    second_half_avg = sum(scores[mid:]) / (len(scores) - mid)

    diff = second_half_avg - first_half_avg
    threshold = 0.05  # 5% change threshold

    if diff > threshold:
        return "improving"
    if diff < -threshold:
        return "declining"
    return "stable"
