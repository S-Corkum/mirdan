"""Next.js-specific validation rules.

Detects common Next.js anti-patterns:
- NEXT001: missing-use-client — client-side APIs used without 'use client' directive
"""

from __future__ import annotations

import re

from mirdan.models import Violation

# Client-side APIs that require 'use client'
_CLIENT_APIS = re.compile(
    r"\b(?:"
    r"useState|useEffect|useRef|useCallback|useMemo|useReducer|useContext|"
    r"useLayoutEffect|useImperativeHandle|useDebugValue|useSyncExternalStore|"
    r"useTransition|useDeferredValue|useId|useOptimistic|useFormStatus|"
    r"onClick|onChange|onSubmit|onFocus|onBlur|onKeyDown|onKeyUp|onMouseEnter"
    r")\b"
)

# 'use client' directive at top of file
_USE_CLIENT = re.compile(r"""^['"]use client['"];?\s*$""", re.MULTILINE)


def check_nextjs_rules(code: str) -> list[Violation]:
    """Check code against Next.js-specific rules.

    Args:
        code: Source code to validate.

    Returns:
        List of Next.js violations.
    """
    violations: list[Violation] = []
    lines = code.split("\n")

    violations.extend(_check_missing_use_client(code, lines))

    return violations


def _check_missing_use_client(code: str, lines: list[str]) -> list[Violation]:
    """NEXT001: Detect client-side APIs without 'use client' directive."""
    # Check if 'use client' is already present
    if _USE_CLIENT.search(code):
        return []

    # Look for client-side API usage
    client_apis_found: list[tuple[int, str]] = []
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        # Skip import lines
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        # Skip comment lines
        if stripped.startswith("//") or stripped.startswith("/*"):
            continue

        match = _CLIENT_APIS.search(stripped)
        if match:
            client_apis_found.append((line_num, match.group(0)))

    if client_apis_found:
        apis = sorted({api for _, api in client_apis_found})
        api_list = ", ".join(apis[:5])
        if len(apis) > 5:
            api_list += f" (+{len(apis) - 5} more)"

        return [
            Violation(
                id="NEXT001",
                rule="missing-use-client",
                category="style",
                severity="error",
                message=(f"Client-side APIs used without 'use client' directive: {api_list}"),
                line=1,
                column=1,
                code_snippet=lines[0].strip() if lines else "",
                suggestion=(
                    "Add 'use client'; at the top of the file, or move "
                    "client-side logic to a separate client component."
                ),
            )
        ]

    return []
