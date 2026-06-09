"""Tests for the multi-language runtime fallback in ``check_command`` and
``CheckRunner``'s ``checks`` parameter.

Covers: a legacy config that detected a non-Python project (e.g. TypeScript)
but lacks an explicit ``checks`` block must still produce the right check
commands at runtime.
"""

from __future__ import annotations

from mirdan.cli.check_command import _resolve_checks
from mirdan.config import CheckRunnerConfig, MirdanConfig, ProjectConfig
from mirdan.core.check_runner import CheckRunner


class TestResolveChecks:
    """Tests for ``check_command._resolve_checks``."""

    def test_legacy_typescript_config_swaps_to_eslint(self) -> None:
        """primary_language: typescript + default Python checks → eslint."""
        cfg = MirdanConfig(project=ProjectConfig(primary_language="typescript"))
        resolved = _resolve_checks(cfg)
        assert resolved.lint_command == "eslint ."
        assert resolved.typecheck_command == "tsc --noEmit"

    def test_custom_user_lint_is_preserved(self) -> None:
        """If the user customised any of the three commands, don't override."""
        cfg = MirdanConfig(
            project=ProjectConfig(primary_language="typescript"),
            checks=CheckRunnerConfig(lint_command="my-custom-lint"),
        )
        resolved = _resolve_checks(cfg)
        assert resolved.lint_command == "my-custom-lint"

    def test_unknown_language_falls_back_to_python_defaults(self) -> None:
        """An unrecognised language leaves the Pydantic class defaults alone."""
        cfg = MirdanConfig(project=ProjectConfig(primary_language="cobol"))
        resolved = _resolve_checks(cfg)
        assert resolved.lint_command == "ruff check"

    def test_python_primary_language_no_swap(self) -> None:
        """Python projects stay on the canonical Python toolchain."""
        cfg = MirdanConfig(project=ProjectConfig(primary_language="python"))
        resolved = _resolve_checks(cfg)
        assert resolved.lint_command == "ruff check"


class TestCheckRunnerChecks:
    """Tests for the ``checks`` parameter on ``CheckRunner``."""

    def test_check_runner_accepts_checks(self) -> None:
        """A TS-shaped CheckRunnerConfig wires into the runner."""
        runner = CheckRunner(CheckRunnerConfig.for_language("typescript"))
        assert runner._checks.lint_command == "eslint ."

    def test_check_runner_default_checks(self) -> None:
        """An explicit CheckRunnerConfig is used as-is."""
        runner = CheckRunner(CheckRunnerConfig(lint_command="baseline"))
        assert runner._checks.lint_command == "baseline"
