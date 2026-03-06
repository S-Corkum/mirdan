"""Session-level quality tracking across validation calls.

Tracks per-file validation results within a session, identifies
unvalidated files, and generates session quality summaries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from mirdan.models import QualitySnapshot, SessionContext, ValidationResult


@dataclass
class FileValidation:
    """A single validation record for a file."""

    file_path: str
    score: float
    passed: bool
    error_count: int
    warning_count: int
    timestamp: float
    language: str = ""
    violation_rules: list[str] = field(default_factory=list)


@dataclass
class SessionQualitySummary:
    """Aggregate quality summary for a session."""

    session_id: str
    validation_count: int
    files_validated: int
    avg_score: float
    min_score: float
    max_score: float
    total_errors: int
    total_warnings: int
    unvalidated_files: list[str] = field(default_factory=list)
    pass_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "session_id": self.session_id,
            "validation_count": self.validation_count,
            "files_validated": self.files_validated,
            "avg_score": round(self.avg_score, 3),
            "min_score": round(self.min_score, 3),
            "max_score": round(self.max_score, 3),
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "unvalidated_files": self.unvalidated_files,
            "pass_rate": round(self.pass_rate, 3),
        }


class SessionTracker:
    """Tracks per-file validation results within a session.

    Records each validation call, tracks which files have been
    validated, and generates session-level quality summaries.
    """

    def __init__(self) -> None:
        self._file_validations: dict[str, list[FileValidation]] = {}
        self._validation_count: int = 0

    def record_validation(
        self,
        result: ValidationResult,
        file_path: str = "",
        session: SessionContext | None = None,
    ) -> None:
        """Record a validation result.

        Args:
            result: The validation result to record.
            file_path: Optional file path being validated.
            session: Optional session to update with tracking data.
        """
        now = time.time()
        error_count = sum(1 for v in result.violations if v.severity == "error")
        warning_count = sum(1 for v in result.violations if v.severity == "warning")

        record = FileValidation(
            file_path=file_path or "<inline>",
            score=result.score,
            passed=result.passed,
            error_count=error_count,
            warning_count=warning_count,
            timestamp=now,
            language=result.language_detected,
            violation_rules=[v.id for v in result.violations],
        )

        key = record.file_path
        if key not in self._file_validations:
            self._file_validations[key] = []
        self._file_validations[key].append(record)
        self._validation_count += 1

        # Update session tracking fields
        if session is not None:
            session.validation_count += 1
            session.cumulative_score += result.score
            session.unresolved_errors = error_count
            session.last_validated_at = now
            if file_path and file_path not in session.files_validated:
                session.files_validated.append(file_path)

    def get_unvalidated_files(self, changed_files: list[str]) -> list[str]:
        """Get files from a change list that haven't been validated.

        Args:
            changed_files: List of file paths that were changed.

        Returns:
            Files not yet validated in this session.
        """
        return [f for f in changed_files if f not in self._file_validations]

    def get_score_for_file(self, file_path: str) -> float | None:
        """Get the most recent validation score for a file.

        Args:
            file_path: The file path to check.

        Returns:
            Most recent score, or None if never validated.
        """
        records = self._file_validations.get(file_path)
        if not records:
            return None
        return records[-1].score

    def get_previous_violations(self, file_path: str) -> set[str]:
        """Get violation rule IDs from the previous validation of a file.

        Must be called AFTER record_validation() for the current run,
        since records[-1] is the current and records[-2] is the previous.

        Args:
            file_path: The file path to check (empty string for inline code).

        Returns:
            Set of rule IDs from the previous validation, or empty set
            if fewer than 2 validations exist for this file.
        """
        records = self._file_validations.get(file_path or "<inline>")
        if not records or len(records) < 2:
            return set()
        return set(records[-2].violation_rules)

    def get_violation_persistence(self, file_path: str) -> dict[str, int]:
        """Count consecutive validations each current rule has appeared in.

        Returns a count only for rules present in the most recent validation.

        Args:
            file_path: The file path to check (empty string for inline code).

        Returns:
            Mapping of rule_id -> consecutive occurrence count.
        """
        records = self._file_validations.get(file_path or "<inline>")
        if not records:
            return {}
        current_rules = set(records[-1].violation_rules)
        persistence: dict[str, int] = {}
        for rule in current_rules:
            count = 0
            for rec in reversed(records):
                if rule in rec.violation_rules:
                    count += 1
                else:
                    break
            persistence[rule] = count
        return persistence

    def get_session_summary(self, session_id: str = "") -> SessionQualitySummary:
        """Generate a quality summary for the current session.

        Args:
            session_id: The session ID to include in the summary.

        Returns:
            Aggregate quality summary.
        """
        if not self._file_validations:
            return SessionQualitySummary(
                session_id=session_id,
                validation_count=0,
                files_validated=0,
                avg_score=0.0,
                min_score=0.0,
                max_score=0.0,
                total_errors=0,
                total_warnings=0,
            )

        all_scores: list[float] = []
        total_errors = 0
        total_warnings = 0
        passed_count = 0

        for records in self._file_validations.values():
            # Use most recent validation per file
            latest = records[-1]
            all_scores.append(latest.score)
            total_errors += latest.error_count
            total_warnings += latest.warning_count
            if latest.passed:
                passed_count += 1

        files_validated = len(self._file_validations)

        return SessionQualitySummary(
            session_id=session_id,
            validation_count=self._validation_count,
            files_validated=files_validated,
            avg_score=sum(all_scores) / len(all_scores) if all_scores else 0.0,
            min_score=min(all_scores) if all_scores else 0.0,
            max_score=max(all_scores) if all_scores else 0.0,
            total_errors=total_errors,
            total_warnings=total_warnings,
            pass_rate=passed_count / files_validated if files_validated else 0.0,
        )

    def to_snapshot(self, session_id: str = "") -> QualitySnapshot | None:
        """Convert the latest session state to a QualitySnapshot.

        Returns:
            QualitySnapshot or None if no validations recorded.
        """
        summary = self.get_session_summary(session_id)
        if summary.validation_count == 0:
            return None

        from datetime import UTC, datetime

        return QualitySnapshot(
            timestamp=datetime.now(tz=UTC).isoformat(),
            project_path="",
            language="",
            score=summary.avg_score,
            passed=summary.total_errors == 0,
            violation_counts={
                "error": summary.total_errors,
                "warning": summary.total_warnings,
            },
        )
