"""Track violation overrides and suggest severity adjustments.

Records when rules are overridden (ignored by the AI agent) and
suggests severity reductions for frequently overridden rules.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Number of overrides before suggesting severity reduction
_OVERRIDE_THRESHOLD = 5


@dataclass
class SeverityAdjustment:
    """Suggestion to change a rule's severity."""

    rule_id: str
    current_severity: str
    suggested_severity: str
    override_count: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "current_severity": self.current_severity,
            "suggested_severity": self.suggested_severity,
            "override_count": self.override_count,
            "reason": self.reason,
        }


@dataclass
class OverrideRecord:
    """A single override record."""

    rule_id: str
    file_path: str
    reason: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "file_path": self.file_path,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


# Severity downgrade mapping
_SEVERITY_DOWNGRADE = {
    "error": "warning",
    "warning": "info",
    "info": "info",  # Can't downgrade further
}


class ViolationTracker:
    """Tracks violation overrides and suggests severity adjustments.

    Stores override records in ``.mirdan/overrides.json`` and suggests
    severity reductions for rules that are frequently overridden.
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        """Initialize tracker.

        Args:
            storage_path: Path to the overrides JSON file.
                Defaults to ``.mirdan/overrides.json`` relative to cwd.
        """
        self._storage_path = storage_path or Path(".mirdan/overrides.json")
        self._overrides: list[OverrideRecord] = []
        self._load()

    def _load(self) -> None:
        """Load existing overrides from disk."""
        if not self._storage_path.exists():
            return

        try:
            data = json.loads(self._storage_path.read_text())
            for record in data.get("overrides", []):
                self._overrides.append(
                    OverrideRecord(
                        rule_id=record["rule_id"],
                        file_path=record.get("file_path", ""),
                        reason=record.get("reason", ""),
                        timestamp=record.get("timestamp", 0.0),
                    )
                )
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.warning("Failed to load overrides from %s: %s", self._storage_path, e)

    def _save(self) -> None:
        """Persist overrides to disk."""
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"overrides": [r.to_dict() for r in self._overrides]}
            self._storage_path.write_text(json.dumps(data, indent=2))
        except OSError as e:
            logger.warning("Failed to save overrides to %s: %s", self._storage_path, e)

    def record_override(
        self,
        rule_id: str,
        file_path: str = "",
        reason: str = "",
    ) -> None:
        """Record that a rule was overridden.

        Args:
            rule_id: The rule ID that was overridden (e.g., "PY005").
            file_path: The file where the override occurred.
            reason: Optional reason for the override.
        """
        record = OverrideRecord(
            rule_id=rule_id,
            file_path=file_path,
            reason=reason,
        )
        self._overrides.append(record)
        self._save()

    def get_override_count(self, rule_id: str) -> int:
        """Get the number of times a rule has been overridden.

        Args:
            rule_id: The rule ID to check.

        Returns:
            Number of override records for this rule.
        """
        return sum(1 for r in self._overrides if r.rule_id == rule_id)

    def get_all_counts(self) -> dict[str, int]:
        """Get override counts for all rules.

        Returns:
            Dict mapping rule_id to override count.
        """
        counts: dict[str, int] = {}
        for r in self._overrides:
            counts[r.rule_id] = counts.get(r.rule_id, 0) + 1
        return counts

    def suggest_severity_changes(
        self,
        threshold: int = _OVERRIDE_THRESHOLD,
    ) -> list[SeverityAdjustment]:
        """Suggest severity reductions for frequently overridden rules.

        Rules overridden >= threshold times are candidates for
        severity reduction (error → warning, warning → info).

        Args:
            threshold: Minimum override count to trigger suggestion.

        Returns:
            List of SeverityAdjustment suggestions.
        """
        suggestions: list[SeverityAdjustment] = []
        counts = self.get_all_counts()

        for rule_id, count in counts.items():
            if count < threshold:
                continue

            # Determine current severity from the rule_id prefix
            current = self._infer_severity(rule_id)
            suggested = _SEVERITY_DOWNGRADE.get(current, current)

            if suggested == current:
                continue  # Can't downgrade further

            suggestions.append(
                SeverityAdjustment(
                    rule_id=rule_id,
                    current_severity=current,
                    suggested_severity=suggested,
                    override_count=count,
                    reason=(
                        f"Rule {rule_id} has been overridden {count} times. "
                        f"Consider reducing severity from '{current}' to '{suggested}'."
                    ),
                )
            )

        return suggestions

    def _infer_severity(self, rule_id: str) -> str:
        """Infer the default severity of a rule from its ID prefix.

        SEC* rules default to error, PY* rules vary, etc.
        This is a best-effort heuristic.

        Args:
            rule_id: Rule ID like "SEC001", "PY005".

        Returns:
            Inferred severity level.
        """
        if rule_id.startswith("SEC"):
            return "error"
        error_prefixes = ("PY001", "PY002", "PY003", "PY004", "PY007", "PY008", "PY009", "PY010")
        if rule_id.startswith(error_prefixes):
            return "error"
        return "warning"
