"""Output parsers for external linters (ruff, ESLint, mypy).

Each parser converts linter-specific JSON output into a list of
``Violation`` objects compatible with Mirdan's validation pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mirdan.models import Violation

logger = logging.getLogger(__name__)


def parse_ruff_output(raw: str) -> list[Violation]:
    """Parse ruff JSON output into Violations.

    ruff output format (--output-format json):
        [{"code": "E501", "message": "...", "location": {"row": 1, "column": 1}, ...}]
    """
    try:
        entries: list[dict[str, Any]] = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse ruff JSON output")
        return []

    violations: list[Violation] = []
    for entry in entries:
        code = entry.get("code", "")
        location = entry.get("location", {})
        violations.append(
            Violation(
                id=f"RUFF-{code}",
                rule=code,
                category="style",
                severity=_ruff_severity(code),
                message=entry.get("message", ""),
                line=location.get("row"),
                column=location.get("column"),
            )
        )
    return violations


def parse_mypy_output(raw: str) -> list[Violation]:
    """Parse mypy JSON output into Violations.

    mypy output format (--output json):
        Each line is a JSON object:
        {"file": "...", "line": 1, "column": 1, "severity": "error", ...}
    """
    violations: list[Violation] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry: dict[str, Any] = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue

        severity = entry.get("severity", "error")
        if severity == "note":
            severity = "info"
        violations.append(
            Violation(
                id=f"MYPY-{entry.get('code', 'E')}",
                rule=entry.get("code", "error"),
                category="style",
                severity=severity,
                message=entry.get("message", ""),
                line=entry.get("line"),
                column=entry.get("column"),
            )
        )
    return violations


def parse_eslint_output(raw: str) -> list[Violation]:
    """Parse ESLint JSON output into Violations.

    ESLint output format (--format json):
        [{"filePath": "...", "messages": [{"ruleId": "...", "severity": 1|2, ...}]}]
    """
    try:
        files: list[dict[str, Any]] = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse ESLint JSON output")
        return []

    violations: list[Violation] = []
    for file_entry in files:
        for msg in file_entry.get("messages", []):
            rule_id = msg.get("ruleId", "unknown")
            severity_num = msg.get("severity", 1)
            severity = "error" if severity_num == 2 else "warning"
            violations.append(
                Violation(
                    id=f"ESLINT-{rule_id}",
                    rule=rule_id or "parse-error",
                    category="style",
                    severity=severity,
                    message=msg.get("message", ""),
                    line=msg.get("line"),
                    column=msg.get("column"),
                )
            )
    return violations


def _ruff_severity(code: str) -> str:
    """Map ruff rule code to severity level."""
    # S-series (bandit security) and E9-series (syntax errors) are errors
    if code.startswith("S") or code.startswith("E9"):
        return "error"
    # E/W/F are typically warnings
    if code.startswith(("E", "W", "F")):
        return "warning"
    return "warning"
