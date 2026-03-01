"""TypeScript/JavaScript architecture validator.

Provides regex-based architecture checks for TS/JS code.
When tree-sitter is installed (``pip install mirdan[ast]``),
uses AST walking for more accurate results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mirdan.models import Violation

# Regex patterns for TS/JS structural analysis
_FUNCTION_DECL = re.compile(
    r"^[ \t]*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*(?:<[^>]*>)?\s*\(",
    re.MULTILINE,
)
_ARROW_NAMED = re.compile(
    r"^[ \t]*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*"
    r"(?::\s*[^=]+)?\s*=\s*(?:async\s+)?(?:\([^)]*\)|[a-zA-Z_]\w*)\s*(?::\s*\w[^=]*)?\s*=>",
    re.MULTILINE,
)
_METHOD_DECL = re.compile(
    r"^[ \t]*(?:async\s+)?(\w+)\s*(?:<[^>]*>)?\s*\([^)]*\)\s*(?::\s*\S+)?\s*\{",
    re.MULTILINE,
)
_NESTING_OPEN = re.compile(r"\b(?:if|for|while|switch|try)\s*[\(\{]")
_RETURN_TYPE = re.compile(
    r"^[ \t]*(?:export\s+)?(?:async\s+)?function\s+\w+\s*(?:<[^>]*>)?\s*\([^)]*\)\s*:\s*\S",
    re.MULTILINE,
)
_ARROW_RETURN_TYPE = re.compile(
    r"^[ \t]*(?:export\s+)?(?:const|let|var)\s+\w+\s*:\s*",
    re.MULTILINE,
)

# Comment line patterns for TS/JS
_COMMENT_LINE = re.compile(r"^\s*(?://|/\*|\*)")


@dataclass
class TSValidationConfig:
    """Configuration for TS/JS architecture checks."""

    max_function_length: int = 30
    max_file_length: int = 300
    max_nesting_depth: int = 4


def validate_ts_architecture(
    code: str,
    language: str = "typescript",
    config: TSValidationConfig | None = None,
) -> tuple[list[Violation], list[str]]:
    """Validate TypeScript/JavaScript code architecture.

    Uses regex-based heuristics to detect:
    - TSARCH001: function-too-long
    - TSARCH002: file-too-long
    - TSARCH003: excessive-nesting
    - TSARCH004: missing-return-type (TypeScript only)

    Args:
        code: Source code to validate.
        language: "typescript" or "javascript".
        config: Optional configuration overrides.

    Returns:
        Tuple of (violations, limitations).
    """
    if config is None:
        config = TSValidationConfig()

    violations: list[Violation] = []
    limitations: list[str] = []
    lines = code.split("\n")

    # TSARCH001: function-too-long
    violations.extend(_check_function_length(code, lines, config.max_function_length))

    # TSARCH002: file-too-long
    violations.extend(_check_file_length(lines, config.max_file_length))

    # TSARCH003: excessive-nesting
    violations.extend(_check_nesting_depth(code, lines, config.max_nesting_depth))

    # TSARCH004: missing-return-type (TS only)
    if language == "typescript":
        violations.extend(_check_missing_return_type(code, lines))

    limitations.append(
        "TS/JS architecture checks use regex heuristics; "
        "install mirdan[ast] for tree-sitter-based AST analysis"
    )

    return violations, limitations


def _check_function_length(code: str, lines: list[str], max_length: int) -> list[Violation]:
    """Check for functions exceeding max length."""
    violations: list[Violation] = []

    for match in _FUNCTION_DECL.finditer(code):
        name = match.group(1)
        start_line = code[: match.start()].count("\n") + 1
        func_len = _estimate_block_length(lines, start_line - 1)
        if func_len > max_length:
            violations.append(
                Violation(
                    id="TSARCH001",
                    rule="function-too-long",
                    category="architecture",
                    severity="warning",
                    message=f"Function '{name}' is ~{func_len} lines (max {max_length})",
                    line=start_line,
                    column=1,
                    code_snippet=lines[start_line - 1].strip() if start_line <= len(lines) else "",
                    suggestion="Break this function into smaller, focused functions",
                )
            )

    for match in _ARROW_NAMED.finditer(code):
        name = match.group(1)
        start_line = code[: match.start()].count("\n") + 1
        func_len = _estimate_block_length(lines, start_line - 1)
        if func_len > max_length:
            violations.append(
                Violation(
                    id="TSARCH001",
                    rule="function-too-long",
                    category="architecture",
                    severity="warning",
                    message=f"Function '{name}' is ~{func_len} lines (max {max_length})",
                    line=start_line,
                    column=1,
                    code_snippet=lines[start_line - 1].strip() if start_line <= len(lines) else "",
                    suggestion="Break this function into smaller, focused functions",
                )
            )

    return violations


def _check_file_length(lines: list[str], max_length: int) -> list[Violation]:
    """Check if file exceeds max non-empty, non-comment lines."""
    non_empty = sum(1 for line in lines if line.strip() and not _COMMENT_LINE.match(line))
    if non_empty > max_length:
        return [
            Violation(
                id="TSARCH002",
                rule="file-too-long",
                category="architecture",
                severity="warning",
                message=f"File has {non_empty} non-empty lines (max {max_length})",
                line=1,
                column=1,
                code_snippet="",
                suggestion="Split this file into smaller, focused modules",
            )
        ]
    return []


def _check_nesting_depth(code: str, lines: list[str], max_depth: int) -> list[Violation]:
    """Check for excessive nesting depth using brace counting."""
    violations: list[Violation] = []
    seen_functions: set[int] = set()

    # Find functions and check their nesting
    for match in _FUNCTION_DECL.finditer(code):
        start_line = code[: match.start()].count("\n")
        if start_line in seen_functions:
            continue
        seen_functions.add(start_line)
        depth = _measure_max_nesting(lines, start_line)
        if depth > max_depth:
            name = match.group(1)
            violations.append(
                Violation(
                    id="TSARCH003",
                    rule="excessive-nesting",
                    category="architecture",
                    severity="warning",
                    message=f"Function '{name}' has nesting depth ~{depth} (max {max_depth})",
                    line=start_line + 1,
                    column=1,
                    code_snippet=lines[start_line].strip() if start_line < len(lines) else "",
                    suggestion="Reduce nesting by extracting conditions or using early returns",
                )
            )

    return violations


def _check_missing_return_type(code: str, lines: list[str]) -> list[Violation]:
    """Check for exported functions missing return type annotations (TS only)."""
    violations: list[Violation] = []

    for match in _FUNCTION_DECL.finditer(code):
        name = match.group(1)
        # Skip private/underscore functions
        if name.startswith("_"):
            continue
        line_text = lines[code[: match.start()].count("\n")]
        # Check if function has return type annotation
        # Find the closing paren and check for ":"
        paren_depth = 0
        found_close = False
        rest = code[match.start() :]
        for i, ch in enumerate(rest):
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth -= 1
                if paren_depth == 0:
                    # Check what follows the closing paren
                    after = rest[i + 1 :].lstrip()
                    if after and after[0] == ":":
                        found_close = True
                    break

        if not found_close:
            start_line = code[: match.start()].count("\n") + 1
            violations.append(
                Violation(
                    id="TSARCH004",
                    rule="missing-return-type",
                    category="architecture",
                    severity="info",
                    message=f"Function '{name}' is missing a return type annotation",
                    line=start_line,
                    column=1,
                    code_snippet=line_text.strip(),
                    suggestion="Add return type: function name(): ReturnType { ... }",
                )
            )

    return violations


def _estimate_block_length(lines: list[str], start_idx: int) -> int:
    """Estimate the length of a brace-delimited block starting at start_idx.

    Counts from the first ``{`` to its matching ``}``.
    """
    brace_depth = 0
    started = False

    for i in range(start_idx, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                brace_depth += 1
                started = True
            elif ch == "}":
                brace_depth -= 1
                if started and brace_depth == 0:
                    return i - start_idx + 1

    # If no matching brace found, return lines to end of file
    return len(lines) - start_idx


def _measure_max_nesting(lines: list[str], start_idx: int) -> int:
    """Measure maximum nesting depth within a function body."""
    brace_depth = 0
    max_depth = 0
    started = False
    # Track depth relative to function body (subtract 1 for function's own braces)
    base_depth = -1

    for i in range(start_idx, len(lines)):
        line = lines[i]
        # Skip comment lines
        if _COMMENT_LINE.match(line):
            continue
        for ch in line:
            if ch == "{":
                brace_depth += 1
                if not started:
                    started = True
                    base_depth = brace_depth
                else:
                    relative = brace_depth - base_depth
                    if relative > max_depth:
                        max_depth = relative
            elif ch == "}":
                brace_depth -= 1
                if started and brace_depth < base_depth:
                    return max_depth

    return max_depth
