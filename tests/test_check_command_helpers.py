"""Tests for check_command helpers.

Covers the three issues that produced the unsatisfiable Stop-hook false
negative: bare-mypy target fallback, test-file filtering for pytest, and
the configurable subprocess timeout surfaced in the result payload.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from mirdan.cli.check_command import (
    _filter_test_files,
    _run_subprocess,
    _typecheck_target,
)


class TestFilterTestFiles:
    def test_empty_input_returns_empty(self) -> None:
        assert _filter_test_files([]) == []

    def test_filters_test_prefixed_files(self) -> None:
        files = ["tests/test_foo.py", "src/bar.py"]
        assert _filter_test_files(files) == ["tests/test_foo.py"]

    def test_detects_tests_directory(self) -> None:
        files = ["tests/integration/runner.py", "src/app.py"]
        assert _filter_test_files(files) == ["tests/integration/runner.py"]

    def test_detects_underscore_test_suffix(self) -> None:
        files = ["src/foo_test.py", "src/foo.py"]
        assert _filter_test_files(files) == ["src/foo_test.py"]


class TestTypecheckTarget:
    def test_explicit_files_used_as_target(self) -> None:
        assert _typecheck_target("mypy", ["src/a.py"]) == ["src/a.py"]

    def test_bare_mypy_no_files_falls_back_to_dot(self) -> None:
        assert _typecheck_target("mypy", []) == ["."]

    def test_mypy_with_explicit_target_in_command_untouched(self) -> None:
        assert _typecheck_target("mypy src/", []) == []

    def test_non_mypy_commands_get_no_fallback(self) -> None:
        assert _typecheck_target("pyright", []) == []
        assert _typecheck_target("tsc --noEmit", []) == []


class TestRunSubprocessTimeout:
    def test_timeout_surfaced_in_stderr(self) -> None:
        fake_err = subprocess.TimeoutExpired(cmd="pytest", timeout=123)
        with patch("mirdan.cli.check_command.subprocess.run", side_effect=fake_err):
            result = _run_subprocess("pytest", [], timeout=123)
        assert "timeout after 123s" in result["stderr"]
        assert result["returncode"] == -1

    def test_default_timeout_is_60(self) -> None:
        captured: dict[str, int] = {}

        def _fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
            captured["timeout"] = kwargs["timeout"]

            class _R:
                returncode = 0
                stdout = ""
                stderr = ""

            return _R()

        with patch("mirdan.cli.check_command.subprocess.run", side_effect=_fake_run):
            _run_subprocess("ruff check", [])
        assert captured["timeout"] == 60

    def test_custom_timeout_plumbed_through(self) -> None:
        captured: dict[str, int] = {}

        def _fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
            captured["timeout"] = kwargs["timeout"]

            class _R:
                returncode = 0
                stdout = ""
                stderr = ""

            return _R()

        with patch("mirdan.cli.check_command.subprocess.run", side_effect=_fake_run):
            _run_subprocess("pytest", [], timeout=600)
        assert captured["timeout"] == 600


@pytest.mark.parametrize(
    "command,files,expected_substring",
    [
        ("ruff check", ["src/a.py"], "ruff check src/a.py"),
        ("mypy", ["src/a.py"], "mypy src/a.py"),
    ],
)
def test_run_subprocess_builds_display_command(
    command: str, files: list[str], expected_substring: str
) -> None:
    def _fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        return _R()

    with patch("mirdan.cli.check_command.subprocess.run", side_effect=_fake_run):
        result = _run_subprocess(command, files)
    assert expected_substring in result["command"]
