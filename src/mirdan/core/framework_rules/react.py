"""React-specific validation rules.

Detects common React anti-patterns:
- REACT001: hooks-in-conditional — React hooks called inside if/for/while
- REACT002: hooks-not-top-level — hooks called inside nested functions
- REACT003: missing-key-prop — map() without key prop in JSX
"""

from __future__ import annotations

import re

from mirdan.models import Violation

# React hook function names
_HOOK_PATTERN = re.compile(r"\buse[A-Z]\w*\s*\(")

# Conditional/loop blocks
_CONDITIONAL_BLOCK = re.compile(
    r"\b(if|else\s+if|else|for|while|switch)\s*(?:\([^)]*\))?\s*\{",
)

# Nested function declarations
_NESTED_FUNC = re.compile(
    r"(?:function\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?(?:\([^)]*\)|[a-zA-Z_]\w*)\s*=>)\s*\{",
)

# JSX map without key
_MAP_WITHOUT_KEY = re.compile(
    r"\.map\s*\(\s*(?:\([^)]*\)|[a-zA-Z_]\w*)\s*=>\s*(?:\(?\s*<)(?:(?!key\s*=).)*?\)",
    re.DOTALL,
)

# Component function pattern (starts with uppercase)
_COMPONENT_FUNC = re.compile(
    r"^[ \t]*(?:export\s+)?(?:default\s+)?(?:function\s+([A-Z]\w*)|"
    r"const\s+([A-Z]\w*)\s*(?::\s*\w[^=]*)?\s*=)",
    re.MULTILINE,
)


def check_react_rules(code: str) -> list[Violation]:
    """Check code against React-specific rules.

    Args:
        code: Source code to validate.

    Returns:
        List of React violations.
    """
    violations: list[Violation] = []
    lines = code.split("\n")

    violations.extend(_check_hooks_in_conditionals(code, lines))
    violations.extend(_check_hooks_not_top_level(code, lines))
    violations.extend(_check_missing_key_prop(code, lines))

    return violations


def _check_hooks_in_conditionals(code: str, lines: list[str]) -> list[Violation]:
    """REACT001: Detect hooks called inside conditional/loop blocks."""
    violations: list[Violation] = []

    # Simple approach: find conditional blocks, check if hooks are called within
    brace_depth = 0
    in_conditional = False
    conditional_depth = 0

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track if we're entering a conditional block
        if _CONDITIONAL_BLOCK.search(stripped) and not in_conditional:
            in_conditional = True
            conditional_depth = brace_depth

        # Count braces
        for ch in stripped:
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if in_conditional and brace_depth <= conditional_depth:
                    in_conditional = False

        # Check for hooks in conditional context
        if in_conditional and _HOOK_PATTERN.search(stripped):
            hook_match = _HOOK_PATTERN.search(stripped)
            if hook_match:
                hook_name = stripped[hook_match.start() : stripped.index("(", hook_match.start())]
                violations.append(
                    Violation(
                        id="REACT001",
                        rule="hooks-in-conditional",
                        category="style",
                        severity="error",
                        message=f"React hook '{hook_name}' called inside a conditional block",
                        line=line_num,
                        column=hook_match.start() + 1,
                        code_snippet=stripped,
                        suggestion=(
                            "Move hooks to the top level of your component. "
                            "Hooks must be called in the same order every render."
                        ),
                    )
                )

    return violations


def _check_hooks_not_top_level(code: str, lines: list[str]) -> list[Violation]:
    """REACT002: Detect hooks called inside nested functions."""
    violations: list[Violation] = []

    # Track function nesting depth
    func_depth = 0
    brace_depth = 0
    func_brace_depths: list[int] = []

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()

        # Detect function declarations
        if _NESTED_FUNC.search(stripped) or re.match(r"^\s*(?:async\s+)?function\s+\w+", stripped):
            func_depth += 1
            # Record the brace depth where this function starts
            func_brace_depths.append(brace_depth)

        # Count braces
        for ch in stripped:
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                # Check if we're exiting a function scope
                if func_brace_depths and brace_depth <= func_brace_depths[-1]:
                    func_depth -= 1
                    func_brace_depths.pop()

        # Check for hooks in nested function (depth > 1)
        if func_depth > 1 and _HOOK_PATTERN.search(stripped):
            hook_match = _HOOK_PATTERN.search(stripped)
            if hook_match:
                hook_name = stripped[hook_match.start() : stripped.index("(", hook_match.start())]
                violations.append(
                    Violation(
                        id="REACT002",
                        rule="hooks-not-top-level",
                        category="style",
                        severity="error",
                        message=f"React hook '{hook_name}' called inside a nested function",
                        line=line_num,
                        column=hook_match.start() + 1,
                        code_snippet=stripped,
                        suggestion=(
                            "Move hooks to the top level of your component function. "
                            "Don't call hooks inside nested functions or callbacks."
                        ),
                    )
                )

    return violations


def _check_missing_key_prop(code: str, lines: list[str]) -> list[Violation]:
    """REACT003: Detect .map() rendering JSX without key prop."""
    violations: list[Violation] = []

    # Look for .map( patterns that contain JSX (< tag) but no key=
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        # Simple heuristic: line has .map( and < but no key=
        if ".map(" in stripped and "<" in stripped and "key=" not in stripped:
            violations.append(
                Violation(
                    id="REACT003",
                    rule="missing-key-prop",
                    category="style",
                    severity="warning",
                    message="JSX element in .map() is missing a 'key' prop",
                    line=line_num,
                    column=stripped.index(".map(") + 1,
                    code_snippet=stripped,
                    suggestion=(
                        "Add a unique 'key' prop to the outermost JSX element: "
                        "<Element key={item.id} />"
                    ),
                )
            )

    return violations
