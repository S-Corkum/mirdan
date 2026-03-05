"""Generate semantic review questions and analysis protocols."""

from __future__ import annotations

import re
from typing import Any

from mirdan.config import SemanticConfig
from mirdan.models import AnalysisProtocol, SemanticCheck, Violation

# Pattern detectors — regex patterns to detect code structures
_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "sql": [
        (r"(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b", "SQL query"),
        (r"\.(?:execute|query|prepare)\s*\(", "database call"),
    ],
    "auth": [
        (
            r"(?:authenticat\w*|authoriz\w*|login|logout|session|token|permission|role)\b",
            "auth logic",
        ),
    ],
    "file_io": [
        (r"\bopen\s*\(", "file open"),
        (r"(?:read|write|readFile|writeFile|fs\.)", "file I/O"),
    ],
    "crypto": [
        (
            r"(?:encrypt\w*|decrypt\w*|hashlib|hmac|cipher|aes|rsa|sha\d*)\b",
            "crypto operation",
        ),
    ],
    "network": [
        (
            r"(?:requests\.(?:get|post|put|delete)|fetch\s*\(|axios\.|http\.(?:Get|Post))",
            "network call",
        ),
    ],
    "error_handling": [
        (r"(?:try\s*[:{]|except\s|catch\s*\(|\.catch\s*\()", "error handler"),
    ],
    "loops": [
        (r"(?:for\s+\w+\s+in\b|while\s|for\s*\()", "loop"),
    ],
}

# Question templates for each detected pattern
_QUESTION_TEMPLATES: dict[str, str] = {
    "sql": (
        "Line {line}: {context}. If this query incorporates any external input, "
        "verify it uses parameterized queries — not string concatenation or f-strings."
    ),
    "auth": (
        "Line {line}: {context}. Verify this check executes BEFORE any data access. "
        "Check: can this be bypassed by passing null/empty values?"
    ),
    "file_io": (
        "Line {line}: {context}. Verify the file is closed in ALL exit paths "
        "including exceptions. For Python, prefer context managers (with statement)."
    ),
    "crypto": (
        "Line {line}: {context}. Verify the algorithm is currently recommended "
        "(SHA-256+ for hashing, AES-256-GCM for encryption, RSA-2048+ for asymmetric)."
    ),
    "network": (
        "Line {line}: {context}. Verify error handling for network failures "
        "(timeouts, DNS failures, non-2xx responses). Check if SSL verification is enabled."
    ),
    "error_handling": (
        "Line {line}: {context}. Verify caught exception types match what the "
        "called functions actually raise. Check for silently swallowed errors."
    ),
    "loops": (
        "Line {line}: {context}. Verify loop termination condition is guaranteed. "
        "Check for off-by-one errors and empty collection edge cases."
    ),
}

# Violation-informed follow-ups — existing violations trigger deeper questions
_VIOLATION_FOLLOW_UPS: dict[str, str] = {
    "SEC004": (
        "SQL string concatenation detected. Trace the concatenated variable "
        "backward to its SOURCE — where does it enter the system? Is it ever sanitized?"
    ),
    "SEC005": (
        "SQL f-string detected. Same as SEC004 — trace the interpolated "
        "variable to its origin."
    ),
    "SEC008": (
        "Shell format injection. Trace {var} backward — can an attacker "
        "control its value? Check ALL code paths that assign to it."
    ),
    "AI001": (
        "Placeholder detected. What SPECIFIC implementation should replace this? "
        "Check the function's callers to understand the expected contract and return type."
    ),
    "AI003": (
        "Over-engineering flagged. Count actual call sites for this "
        "abstraction — if fewer than 3, inline it."
    ),
}

# Severity mapping for pattern types
_SECURITY_PATTERNS = frozenset({"sql", "auth", "crypto"})


class SemanticAnalyzer:
    """Generate semantic review questions from code patterns and violations.

    Template-based and fully deterministic — no LLM calls required.
    Produces targeted review questions that guide the calling LLM to
    investigate specific concerns.
    """

    def __init__(self, config: SemanticConfig | None = None) -> None:
        self._config = config or SemanticConfig()

    def generate_checks(
        self,
        code: str,
        language: str,
        violations: list[Violation],
    ) -> list[SemanticCheck]:
        """Generate semantic review questions from code patterns and violations."""
        if not self._config.enabled:
            return []

        if not code or not code.strip():
            return []

        checks: list[SemanticCheck] = []
        lines = code.split("\n")

        # Phase 1: Pattern-based checks
        for pattern_type, patterns in _PATTERNS.items():
            for line_num, line_text in enumerate(lines, 1):
                for regex, context_label in patterns:
                    if re.search(regex, line_text, re.IGNORECASE):
                        template = _QUESTION_TEMPLATES.get(pattern_type)
                        if template:
                            checks.append(
                                SemanticCheck(
                                    concern=pattern_type,
                                    question=template.format(
                                        line=line_num,
                                        context=f"{context_label} detected:"
                                        f" `{line_text.strip()[:80]}`",
                                    ),
                                    severity=(
                                        "warning"
                                        if pattern_type in _SECURITY_PATTERNS
                                        else "info"
                                    ),
                                    focus_lines=[line_num],
                                )
                            )

        # Phase 2: Violation-informed follow-ups
        for violation in violations:
            follow_up = _VIOLATION_FOLLOW_UPS.get(violation.id)
            if follow_up and violation.line:
                checks.append(
                    SemanticCheck(
                        concern="violation_deep_dive",
                        question=f"Line {violation.line}: {follow_up}",
                        severity=violation.severity,
                        related_violation=violation.id,
                        focus_lines=[violation.line],
                    )
                )

        # Deduplicate by line+concern
        seen: set[tuple[int, str]] = set()
        deduped: list[SemanticCheck] = []
        for check in checks:
            key = (check.focus_lines[0] if check.focus_lines else 0, check.concern)
            if key not in seen:
                seen.add(key)
                deduped.append(check)
        return deduped

    def generate_analysis_protocol(
        self,
        code: str,
        language: str,
        violations: list[Violation],
        semantic_checks: list[SemanticCheck],
    ) -> AnalysisProtocol | None:
        """Generate structured analysis protocol for security-critical code."""
        focus_areas: list[dict[str, Any]] = [
            {
                "concern": check.concern,
                "question": check.question,
                "focus_lines": check.focus_lines,
            }
            for check in semantic_checks
            if check.concern in ("sql", "auth", "crypto", "violation_deep_dive")
        ]

        if not focus_areas:
            return None

        return AnalysisProtocol(
            type="security_flow_analysis",
            focus_areas=focus_areas,
            response_format={
                "findings": [
                    {
                        "line": "int",
                        "severity": "str",
                        "issue": "str",
                        "recommendation": "str",
                    }
                ]
            },
        )
