"""Default lint/typecheck/test commands per detected project language."""

from __future__ import annotations

from mirdan.config import CheckRunnerConfig

DEFAULT_CHECKS_BY_LANGUAGE: dict[str, CheckRunnerConfig] = {
    "python": CheckRunnerConfig(
        lint_command="ruff check",
        typecheck_command="mypy",
        test_command="pytest -x --tb=short",
    ),
    "typescript": CheckRunnerConfig(
        lint_command="eslint .",
        typecheck_command="tsc --noEmit",
        test_command="npm test --silent",
    ),
    "javascript": CheckRunnerConfig(
        lint_command="eslint .",
        typecheck_command="true",
        test_command="npm test --silent",
    ),
    "rust": CheckRunnerConfig(
        lint_command="cargo clippy --quiet",
        typecheck_command="cargo check --quiet",
        test_command="cargo test --quiet",
    ),
    "go": CheckRunnerConfig(
        lint_command="go vet ./...",
        typecheck_command="true",
        test_command="go test ./...",
    ),
    "java": CheckRunnerConfig(
        lint_command="mvn checkstyle:check -q",
        typecheck_command="mvn compile -q",
        test_command="mvn test -q",
    ),
}


def defaults_for_language(language: str) -> CheckRunnerConfig | None:
    """Return a fresh copy of the default checks for ``language``.

    Case-insensitive. Returns None when ``language`` is not in the table
    (caller should fall back to class defaults).
    """
    entry = DEFAULT_CHECKS_BY_LANGUAGE.get(language.lower())
    return entry.model_copy() if entry is not None else None
