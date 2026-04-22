"""Tests for the multi-language default check-command lookup table."""

from __future__ import annotations

import pytest

from mirdan.config import CheckRunnerConfig
from mirdan.core.check_defaults import (
    DEFAULT_CHECKS_BY_LANGUAGE,
    defaults_for_language,
)


@pytest.mark.parametrize(
    "language,expected_lint_prefix",
    [
        ("python", "ruff"),
        ("typescript", "eslint"),
        ("javascript", "eslint"),
        ("rust", "cargo"),
        ("go", "go"),
        ("java", "mvn"),
    ],
)
def test_language_has_expected_lint_tool(
    language: str, expected_lint_prefix: str
) -> None:
    cfg = defaults_for_language(language)
    assert cfg is not None
    assert cfg.lint_command.startswith(expected_lint_prefix)


def test_unknown_language_returns_none() -> None:
    assert defaults_for_language("cobol") is None
    assert defaults_for_language("fortran") is None


def test_lookup_is_case_insensitive() -> None:
    assert defaults_for_language("PYTHON").lint_command == "ruff check"
    assert defaults_for_language("TypeScript").lint_command == "eslint ."


def test_defaults_return_a_copy() -> None:
    """Mutating the returned config must not poison the shared table entry."""
    first = defaults_for_language("python")
    assert first is not None
    first.lint_command = "custom"
    second = defaults_for_language("python")
    assert second is not None
    assert second.lint_command == "ruff check"


def test_all_entries_are_check_runner_configs() -> None:
    for lang, cfg in DEFAULT_CHECKS_BY_LANGUAGE.items():
        assert isinstance(cfg, CheckRunnerConfig), lang


def test_for_language_factory_on_config_class() -> None:
    assert CheckRunnerConfig.for_language("rust").lint_command == "cargo clippy --quiet"
    # Unknown language falls back to class defaults (Python)
    assert CheckRunnerConfig.for_language("cobol").lint_command == "ruff check"


def test_noop_typecheck_for_languages_without_separate_step() -> None:
    """Go and JS bundle typechecking into build/test; use POSIX ``true``."""
    assert defaults_for_language("go").typecheck_command == "true"
    assert defaults_for_language("javascript").typecheck_command == "true"
