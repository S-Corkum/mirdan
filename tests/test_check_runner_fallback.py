"""Tests for the multi-language runtime fallback in ``check_command`` and
``CheckRunner``'s ``checks_override`` parameter.

Covers the 2.0.7 story: a legacy config written under 2.0.x that detected
a non-Python project (e.g. TypeScript) but lacks an explicit ``llm.checks``
block must still produce the right check commands at runtime.
"""

from __future__ import annotations

from mirdan.cli.check_command import _resolve_checks
from mirdan.config import (
    CheckRunnerConfig,
    LLMConfig,
    MirdanConfig,
    ProjectConfig,
)
from mirdan.core.check_runner import CheckRunner


class TestResolveChecks:
    """Tests for ``check_command._resolve_checks``."""

    def test_legacy_typescript_config_swaps_to_eslint(self) -> None:
        """primary_language: typescript + default Python checks → eslint."""
        cfg = MirdanConfig(
            project=ProjectConfig(primary_language="typescript"),
            llm=LLMConfig(),
        )
        resolved = _resolve_checks(cfg)
        assert resolved.lint_command == "eslint ."
        assert resolved.typecheck_command == "tsc --noEmit"

    def test_custom_user_lint_is_preserved(self) -> None:
        """If the user customised any of the three commands, don't override."""
        cfg = MirdanConfig(
            project=ProjectConfig(primary_language="typescript"),
            llm=LLMConfig(checks=CheckRunnerConfig(lint_command="my-custom-lint")),
        )
        resolved = _resolve_checks(cfg)
        assert resolved.lint_command == "my-custom-lint"

    def test_unknown_language_falls_back_to_python_defaults(self) -> None:
        """An unrecognised language leaves the Pydantic class defaults alone."""
        cfg = MirdanConfig(
            project=ProjectConfig(primary_language="cobol"),
            llm=LLMConfig(),
        )
        resolved = _resolve_checks(cfg)
        assert resolved.lint_command == "ruff check"

    def test_python_primary_language_no_swap(self) -> None:
        """Python projects stay on the canonical Python toolchain."""
        cfg = MirdanConfig(
            project=ProjectConfig(primary_language="python"),
            llm=LLMConfig(),
        )
        resolved = _resolve_checks(cfg)
        assert resolved.lint_command == "ruff check"


class TestCheckRunnerOverride:
    """Tests for the ``checks_override`` parameter on ``CheckRunner``."""

    def test_check_runner_accepts_override(self) -> None:
        """Passing a TS-shaped CheckRunnerConfig wires it into the runner."""
        runner = CheckRunner(
            llm_manager=None,
            config=LLMConfig(),
            checks_override=CheckRunnerConfig.for_language("typescript"),
        )
        assert runner._config.checks.lint_command == "eslint ."

    def test_check_runner_without_override_keeps_config_checks(self) -> None:
        """Omitting ``checks_override`` means the LLMConfig's own checks win."""
        cfg = LLMConfig(checks=CheckRunnerConfig(lint_command="baseline"))
        runner = CheckRunner(llm_manager=None, config=cfg)
        assert runner._config.checks.lint_command == "baseline"
