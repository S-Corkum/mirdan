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
    "test_quality": [
        (r"def\s+test_\w+.*:\s*\n\s+(?:pass|\.\.\.)\s*$", "empty test function"),
        (r"assert\s+True\s*$", "assert True placeholder"),
        (r"@patch.*\n.*@patch.*\n.*@patch.*\n.*@patch", "heavily mocked test"),
    ],
    "concurrency": [
        (
            r"(?:async\s+def|asyncio\.gather|asyncio\.create_task"
            r"|threading\.Thread|concurrent\.futures)",
            "concurrent code",
        ),
        (r"(?:global\s+\w+\s*$|Lock\(\)|Semaphore\(\)|RLock\(\))", "synchronization primitive"),
    ],
    "boundary": [
        (r"(?<=[\w)\]])\s*/\s*[a-zA-Z_]\w*(?!\w)", "division with variable denominator"),
        (r"\[\s*[a-zA-Z_]\w*\s*(?:[+-]\s*\d+\s*)?\]", "dynamic index access"),
        (
            r"(?:\bint\s*\(|\bfloat\s*\(|parseFloat\s*\(|parseInt\s*\("
            r"|strconv\.(?:Atoi|ParseFloat))",
            "numeric parsing from string",
        ),
    ],
    "error_propagation": [
        (
            r"except\s+\w[\w.]*(?:\s+as\s+\w+)?:\s*$",
            "exception handler (verify error is not swallowed)",
        ),
        (
            r"\.catch\s*\(\s*(?:\(\s*\w*\s*\)\s*=>|function\s*\()\s*\{",
            "JS catch handler (verify error is handled)",
        ),
    ],
    "state_machine": [
        (
            r"\b(?:status|state|phase|stage|mode)\s*(?:==|===|!=|!==)\s*['\"]",
            "string-based state comparison",
        ),
        (r"(?:\.status|\.state|\.phase|\.stage)\s*=\s*['\"]", "direct string state assignment"),
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
    "test_quality": (
        "Line {line}: {context}. Verify this test exercises real behavior. "
        "Check: does it assert meaningful outcomes? Are mocks minimal and realistic?"
    ),
    "concurrency": (
        "Line {line}: {context}. If this code runs concurrently (multiple "
        "coroutines, threads, or requests), verify shared mutable state is "
        "protected. Check: can two executions interleave and corrupt data?"
    ),
    "boundary": (
        "Line {line}: {context}. Verify behavior at boundary values: "
        "zero denominator, empty collection, negative index, max-int overflow. "
        "Add explicit guards if the caller cannot guarantee safe inputs."
    ),
    "error_propagation": (
        "Line {line}: {context}. Verify this error path preserves diagnostic "
        "context (stack trace, original exception, relevant variables). "
        "Check: does the caller receive enough information to diagnose failures?"
    ),
    "state_machine": (
        "Line {line}: {context}. Verify all valid state transitions are handled "
        "and invalid transitions are rejected. Check: what happens if the state "
        "is an unexpected value? Consider using an enum instead of strings."
    ),
}

# Violation-informed follow-ups — existing violations trigger deeper questions
_VIOLATION_FOLLOW_UPS: dict[str, str] = {
    "SEC004": (
        "SQL string concatenation detected. Trace the concatenated variable "
        "backward to its SOURCE — where does it enter the system? Is it ever sanitized?"
    ),
    "SEC005": (
        "SQL f-string detected. Same as SEC004 — trace the interpolated variable to its origin."
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
    "TEST001": (
        "Empty test detected. What SPECIFIC behavior should this test verify? "
        "Check the corresponding implementation function's contract and edge cases."
    ),
    "TEST002": (
        "Assert True placeholder. Replace with an assertion that verifies "
        "the ACTUAL return value or side effect of the function under test."
    ),
    "TEST003": (
        "Test has no assertions. Add assertions that verify: (1) return values, "
        "(2) expected side effects, (3) exception behavior for invalid inputs."
    ),
    "TEST005": (
        "Excessive mocking detected. Consider: are you testing the mocks or the code? "
        "Remove mocks for units that are fast and deterministic. Mock only I/O boundaries."
    ),
    "DEEP001": (
        "Swallowed exception detected. What failure does this exception represent? "
        "At minimum, log the exception. If truly ignorable, use contextlib.suppress() "
        "to document the intent explicitly."
    ),
    "DEEP004": (
        "Exception re-raised without `from` clause — original traceback is lost. "
        "Add `from original_exception` to preserve the diagnostic chain."
    ),
}

# Severity mapping for pattern types
_PATTERN_SEVERITY: dict[str, str] = {
    "sql": "warning",
    "auth": "warning",
    "crypto": "warning",
    "concurrency": "warning",
    "error_propagation": "warning",
    "boundary": "info",
    "state_machine": "info",
}


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

        # Deep analysis patterns (gated by config)
        deep_pattern_types = frozenset(
            {
                "concurrency",
                "boundary",
                "error_propagation",
                "state_machine",
            }
        )

        # Phase 1: Pattern-based checks
        for pattern_type, patterns in _PATTERNS.items():
            if pattern_type in deep_pattern_types and not self._config.deep_analysis:
                continue
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
                                    severity=_PATTERN_SEVERITY.get(pattern_type, "info"),
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
        security_concerns = {"sql", "auth", "crypto", "violation_deep_dive"}
        deep_concerns = {"concurrency", "error_propagation", "boundary", "state_machine"}
        protocol_concerns = security_concerns | deep_concerns

        focus_areas: list[dict[str, Any]] = [
            {
                "concern": check.concern,
                "question": check.question,
                "focus_lines": check.focus_lines,
            }
            for check in semantic_checks
            if check.concern in protocol_concerns
        ]

        if not focus_areas:
            return None

        concern_types = {fa["concern"] for fa in focus_areas}
        has_security = bool(concern_types & security_concerns)
        has_deep = bool(concern_types & deep_concerns)
        if has_security and has_deep:
            protocol_type = "comprehensive_analysis"
        elif has_deep:
            protocol_type = "deep_analysis"
        else:
            protocol_type = "security_flow_analysis"

        return AnalysisProtocol(
            type=protocol_type,
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
