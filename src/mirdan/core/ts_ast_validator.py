"""TypeScript/JavaScript architecture validator.

Provides architecture checks for TS/JS code. Uses tree-sitter AST
walking when installed (``pip install mirdan[ast]``), otherwise falls
back to regex-based heuristics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mirdan.models import Violation

_HAS_TREE_SITTER = False
try:
    import tree_sitter_javascript as _tsjs
    import tree_sitter_typescript as _tsts
    from tree_sitter import Language, Parser

    _HAS_TREE_SITTER = True
except ImportError:
    pass

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

    Uses tree-sitter AST walking when available, otherwise falls back to
    regex-based heuristics. Checks:

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

    if _HAS_TREE_SITTER:
        return _validate_with_tree_sitter(code, language, config)

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


# ---------------------------------------------------------------------------
# Tree-sitter code path (used when tree-sitter is installed)
# ---------------------------------------------------------------------------

_NESTING_NODE_TYPES = frozenset({
    "if_statement",
    "for_statement",
    "for_in_statement",
    "while_statement",
    "do_statement",
    "switch_statement",
    "try_statement",
})

_FUNCTION_NODE_TYPES = frozenset({
    "function_declaration",
    "arrow_function",
    "method_definition",
    "function_expression",
})


def _validate_with_tree_sitter(
    code: str,
    language: str,
    config: TSValidationConfig,
) -> tuple[list[Violation], list[str]]:
    """Validate TS/JS architecture using tree-sitter AST."""
    try:
        if language in ("typescript", "tsx"):
            if language == "tsx":
                lang_obj = Language(_tsts.language_tsx())
            else:
                lang_obj = Language(_tsts.language_typescript())
        else:
            lang_obj = Language(_tsjs.language())

        parser = Parser(lang_obj)
        tree = parser.parse(code.encode("utf-8"))
    except Exception:
        # Fall back to regex if tree-sitter parsing fails
        violations: list[Violation] = []
        limitations: list[str] = []
        lines = code.split("\n")
        violations.extend(_check_function_length(code, lines, config.max_function_length))
        violations.extend(_check_file_length(lines, config.max_file_length))
        violations.extend(_check_nesting_depth(code, lines, config.max_nesting_depth))
        if language == "typescript":
            violations.extend(_check_missing_return_type(code, lines))
        limitations.append(
            "TS/JS architecture checks use regex heuristics; "
            "tree-sitter parsing failed"
        )
        return violations, limitations

    root = tree.root_node
    violations = []

    # Collect all function-like nodes
    func_nodes = _collect_function_nodes(root)

    # TSARCH001: function-too-long
    for node, name in func_nodes:
        func_len = node.end_point.row - node.start_point.row + 1
        if func_len > config.max_function_length:
            violations.append(
                Violation(
                    id="TSARCH001",
                    rule="function-too-long",
                    category="architecture",
                    severity="warning",
                    message=(
                        f"Function '{name}' is {func_len} lines"
                        f" (max {config.max_function_length})"
                    ),
                    line=node.start_point.row + 1,
                    column=node.start_point.column + 1,
                    suggestion="Break this function into smaller, focused functions",
                )
            )

    # TSARCH002: file-too-long
    lines = code.split("\n")
    non_empty = sum(
        1 for line in lines if line.strip() and not _COMMENT_LINE.match(line)
    )
    if non_empty > config.max_file_length:
        violations.append(
            Violation(
                id="TSARCH002",
                rule="file-too-long",
                category="architecture",
                severity="warning",
                message=f"File has {non_empty} non-empty lines (max {config.max_file_length})",
                line=1,
                column=1,
                suggestion="Split this file into smaller, focused modules",
            )
        )

    # TSARCH003: excessive-nesting
    for node, name in func_nodes:
        depth = _ts_max_nesting(node)
        if depth > config.max_nesting_depth:
            violations.append(
                Violation(
                    id="TSARCH003",
                    rule="excessive-nesting",
                    category="architecture",
                    severity="warning",
                    message=(
                        f"Function '{name}' has nesting depth"
                        f" {depth} (max {config.max_nesting_depth})"
                    ),
                    line=node.start_point.row + 1,
                    column=node.start_point.column + 1,
                    suggestion="Reduce nesting by extracting conditions or using early returns",
                )
            )

    # TSARCH004: missing-return-type (TypeScript only)
    if language in ("typescript", "tsx"):
        for node, name in func_nodes:
            if name.startswith("_"):
                continue
            if not _ts_has_return_type(node):
                violations.append(
                    Violation(
                        id="TSARCH004",
                        rule="missing-return-type",
                        category="architecture",
                        severity="info",
                        message=f"Function '{name}' is missing a return type annotation",
                        line=node.start_point.row + 1,
                        column=node.start_point.column + 1,
                        suggestion="Add return type: function name(): ReturnType { ... }",
                    )
                )

    return violations, []  # No limitations when tree-sitter is available


def _collect_function_nodes(root) -> list[tuple]:
    """Collect all function-like nodes with their names from the AST."""
    results: list[tuple] = []

    def _walk(node) -> None:
        if node.type == "function_declaration":
            name = _ts_get_child_text(node, "identifier") or "<anonymous>"
            results.append((node, name))
        elif node.type == "method_definition":
            name = _ts_get_child_text(node, "property_identifier") or "<anonymous>"
            results.append((node, name))
        elif node.type == "arrow_function":
            # Arrow functions get their name from the parent variable declarator
            name = _ts_arrow_name(node) or "<anonymous>"
            results.append((node, name))
        elif node.type == "function_expression":
            # Named or anonymous function expression
            name = _ts_get_child_text(node, "identifier") or "<anonymous>"
            results.append((node, name))

        for child in node.children:
            _walk(child)

    _walk(root)
    return results


def _ts_get_child_text(node, child_type: str) -> str | None:
    """Get the text of the first child with the given type."""
    for child in node.children:
        if child.type == child_type:
            return child.text.decode("utf-8") if child.text else None
    return None


def _ts_arrow_name(node) -> str | None:
    """Get the name of an arrow function from its parent variable declarator."""
    parent = node.parent
    if parent and parent.type == "variable_declarator":
        return _ts_get_child_text(parent, "identifier")
    return None


def _ts_max_nesting(node, current_depth: int = 0) -> int:
    """Measure maximum nesting depth within a function node."""
    depth = current_depth
    if node.type in _NESTING_NODE_TYPES:
        current_depth += 1
        depth = max(depth, current_depth)

    for child in node.children:
        # Don't descend into nested function definitions
        if child.type in _FUNCTION_NODE_TYPES:
            continue
        depth = max(depth, _ts_max_nesting(child, current_depth))

    return depth


def _ts_has_return_type(node) -> bool:
    """Check if a function node has a return type annotation."""
    return any(child.type == "type_annotation" for child in node.children)
