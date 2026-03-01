"""Framework-specific validation rules registry.

Dispatches validation to framework-specific rule modules based on
detected frameworks in the code.
"""

from __future__ import annotations

from collections.abc import Callable

from mirdan.models import Violation


def check_framework_rules(
    code: str,
    language: str,
    frameworks: list[str] | None = None,
) -> list[Violation]:
    """Run framework-specific validation rules.

    Args:
        code: Source code to validate.
        language: Detected programming language.
        frameworks: List of detected frameworks. If None, auto-detects
            from code content.

    Returns:
        List of framework-specific violations.
    """
    if frameworks is None:
        frameworks = _detect_frameworks(code, language)

    violations: list[Violation] = []

    for framework in frameworks:
        checker = _FRAMEWORK_CHECKERS.get(framework)
        if checker:
            violations.extend(checker(code))

    return violations


def _detect_frameworks(code: str, language: str) -> list[str]:
    """Auto-detect frameworks from code imports and patterns."""
    detected: list[str] = []

    if language in ("typescript", "javascript"):
        if "from 'react'" in code or 'from "react"' in code or "require('react')" in code:
            detected.append("react")
        if "from 'next" in code or 'from "next' in code:
            detected.append("next.js")

    if language == "python" and ("from fastapi" in code or "import fastapi" in code):
        detected.append("fastapi")

    return detected


def _check_react(code: str) -> list[Violation]:
    from mirdan.core.framework_rules.react import check_react_rules

    return check_react_rules(code)


def _check_nextjs(code: str) -> list[Violation]:
    from mirdan.core.framework_rules.nextjs import check_nextjs_rules

    return check_nextjs_rules(code)


def _check_fastapi(code: str) -> list[Violation]:
    from mirdan.core.framework_rules.fastapi import check_fastapi_rules

    return check_fastapi_rules(code)


_FRAMEWORK_CHECKERS: dict[str, Callable[[str], list[Violation]]] = {
    "react": _check_react,
    "next.js": _check_nextjs,
    "fastapi": _check_fastapi,
}
