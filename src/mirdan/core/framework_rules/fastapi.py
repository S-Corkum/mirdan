"""FastAPI-specific validation rules.

Detects common FastAPI anti-patterns:
- FAPI001: sync-endpoint — synchronous route handler (should be async)
- FAPI002: missing-response-model — endpoint without response_model
"""

from __future__ import annotations

import re

from mirdan.models import Violation

# FastAPI route decorators
_ROUTE_DECORATOR = re.compile(
    r"^[ \t]*@(?:app|router)\.(?:get|post|put|delete|patch|options|head)\s*\(",
    re.MULTILINE,
)

# Sync function definition (no async keyword)
_SYNC_DEF = re.compile(r"^[ \t]*def\s+(\w+)\s*\(", re.MULTILINE)

# Async function definition
_ASYNC_DEF = re.compile(r"^[ \t]*async\s+def\s+(\w+)\s*\(", re.MULTILINE)

# response_model in decorator
_RESPONSE_MODEL = re.compile(r"response_model\s*=")


def check_fastapi_rules(code: str) -> list[Violation]:
    """Check code against FastAPI-specific rules.

    Args:
        code: Source code to validate.

    Returns:
        List of FastAPI violations.
    """
    violations: list[Violation] = []
    lines = code.split("\n")

    violations.extend(_check_sync_endpoints(code, lines))
    violations.extend(_check_missing_response_model(code, lines))

    return violations


def _check_sync_endpoints(code: str, lines: list[str]) -> list[Violation]:
    """FAPI001: Detect synchronous FastAPI endpoint handlers."""
    violations: list[Violation] = []

    for match in _ROUTE_DECORATOR.finditer(code):
        # Find the function definition following this decorator
        rest = code[match.end() :]
        # Skip to end of decorator (may span multiple lines)
        paren_depth = 1
        end_idx = 0
        for idx, ch in enumerate(rest):
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth -= 1
                if paren_depth == 0:
                    end_idx = idx
                    break

        after_decorator = rest[end_idx + 1 :].lstrip()

        # Check if the next function is sync
        sync_match = _SYNC_DEF.match(after_decorator)
        async_match = _ASYNC_DEF.match(after_decorator)

        if sync_match and not async_match:
            func_name = sync_match.group(1)
            # Find actual line number of the def
            func_line = code[: code.index(after_decorator[:20], match.end())].count("\n") + 1
            violations.append(
                Violation(
                    id="FAPI001",
                    rule="sync-endpoint",
                    category="style",
                    severity="warning",
                    message=f"FastAPI endpoint '{func_name}' is synchronous",
                    line=func_line,
                    column=1,
                    code_snippet=lines[func_line - 1].strip() if func_line <= len(lines) else "",
                    suggestion=(
                        "Use 'async def' for FastAPI endpoints to avoid "
                        "blocking the event loop. If using blocking I/O, "
                        "use run_in_executor or a task queue."
                    ),
                )
            )

    return violations


def _check_missing_response_model(code: str, lines: list[str]) -> list[Violation]:
    """FAPI002: Detect endpoints without response_model parameter."""
    violations: list[Violation] = []

    for match in _ROUTE_DECORATOR.finditer(code):
        decorator_line = code[: match.start()].count("\n") + 1
        # Get the full decorator text
        rest = code[match.start() :]
        paren_depth = 0
        end_idx = 0
        for i, ch in enumerate(rest):
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth -= 1
                if paren_depth == 0:
                    end_idx = i
                    break

        decorator_text = rest[: end_idx + 1]

        if not _RESPONSE_MODEL.search(decorator_text):
            violations.append(
                Violation(
                    id="FAPI002",
                    rule="missing-response-model",
                    category="style",
                    severity="info",
                    message="FastAPI endpoint missing response_model parameter",
                    line=decorator_line,
                    column=1,
                    code_snippet=lines[decorator_line - 1].strip()
                    if decorator_line <= len(lines)
                    else "",
                    suggestion=(
                        "Add response_model to the decorator for automatic "
                        "response serialization and OpenAPI documentation: "
                        "@app.get('/path', response_model=MyModel)"
                    ),
                )
            )

    return violations
