"""Tests for check_command helpers.

Covers the three issues that produced the unsatisfiable Stop-hook false
negative: bare-mypy target fallback, test-file filtering for pytest, and
the configurable subprocess timeout surfaced in the result payload.
"""

from __future__ import annotations

import subprocess
from typing import Any
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


class TestRunSubprocessClassification:
    """Tests for the ``classification`` field added to _run_subprocess output."""

    def test_success_is_classified_ok(self) -> None:
        def _fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
            class _R:
                returncode = 0
                stdout = ""
                stderr = ""

            return _R()

        with patch("mirdan.cli.check_command.subprocess.run", side_effect=_fake_run):
            result = _run_subprocess("ruff check", [])
        assert result["classification"] == "ok"

    def test_nonzero_returncode_is_code_quality(self) -> None:
        def _fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
            class _R:
                returncode = 1
                stdout = "E501 line too long"
                stderr = ""

            return _R()

        with patch("mirdan.cli.check_command.subprocess.run", side_effect=_fake_run):
            result = _run_subprocess("ruff check", ["x.py"])
        assert result["classification"] == "code_quality"

    def test_timeout_is_infrastructure(self) -> None:
        fake_err = subprocess.TimeoutExpired(cmd="pytest", timeout=5)
        with patch("mirdan.cli.check_command.subprocess.run", side_effect=fake_err):
            result = _run_subprocess("pytest", [], timeout=5)
        assert result["classification"] == "infrastructure"

    def test_missing_binary_is_infrastructure(self) -> None:
        with patch(
            "mirdan.cli.check_command.subprocess.run", side_effect=FileNotFoundError("nope")
        ):
            result = _run_subprocess("nonexistent-linter", [])
        assert result["classification"] == "infrastructure"


class TestCheckJsonShape:
    """Verify ``mirdan check --smart`` JSON output exposes the new keys."""

    def _run_and_capture(
        self,
        capsys: pytest.CaptureFixture[str],
        lint_classification: str,
        typecheck_classification: str,
        test_classification: str,
    ) -> dict[str, Any]:
        """Drive run_check with mocked subprocess results and parse the JSON."""
        import json as _json

        from mirdan.cli.check_command import run_check
        from mirdan.config import (
            CheckRunnerConfig,
            LLMConfig,
            MirdanConfig,
            ProjectConfig,
        )

        def _make_result(clf: str) -> dict[str, Any]:
            rc = {"ok": 0, "code_quality": 1, "infrastructure": -1}[clf]
            return {
                "command": "fake",
                "returncode": rc,
                "stdout": "",
                "stderr": "",
                "classification": clf,
            }

        # Pre-classified results for lint/typecheck/test in that call order.
        # Auto-fix is disabled in the test config so _run_subprocess fires
        # exactly three times even when lint reports code_quality.
        results_by_slot = {
            0: _make_result(lint_classification),
            1: _make_result(typecheck_classification),
            2: _make_result(test_classification),
        }
        call_count = {"n": 0}

        def _fake(*args, **kwargs):  # type: ignore[no-untyped-def]
            idx = call_count["n"]
            call_count["n"] += 1
            return results_by_slot.get(idx, _make_result("ok"))

        test_config = MirdanConfig(
            project=ProjectConfig(primary_language="python"),
            llm=LLMConfig(
                checks=CheckRunnerConfig(auto_fix_lint=False),
            ),
        )

        with (
            patch("mirdan.cli.check_command._run_subprocess", side_effect=_fake),
            patch("mirdan.cli.check_command._try_sidecar", return_value=None),
            patch(
                "mirdan.cli.check_command.MirdanConfig.find_config",
                return_value=test_config,
            ),
        ):
            run_check([])

        captured = capsys.readouterr()
        output = captured.out.strip()
        start = output.find("{")
        parsed: dict[str, Any] = _json.loads(output[start:])
        return parsed

    def test_check_json_includes_new_keys(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Code-quality failure surfaces all new keys correctly."""
        data = self._run_and_capture(capsys, "code_quality", "ok", "ok")
        assert data["all_pass"] is False
        assert data["code_quality_pass"] is False
        assert data["infra_ok"] is True
        assert data["infra_failures"] == []

    def test_check_json_infra_failure_surfaces(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Missing-binary path produces an infra_failures entry."""
        data = self._run_and_capture(capsys, "infrastructure", "ok", "ok")
        assert data["all_pass"] is False
        assert data["code_quality_pass"] is True
        assert data["infra_ok"] is False
        assert data["infra_failures"] == ["lint"]

    def test_check_json_backwards_compat_keys_present(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """All original keys still appear for downstream Stop-hook parsers."""
        data = self._run_and_capture(capsys, "ok", "ok", "ok")
        for key in ("lint", "typecheck", "test", "all_pass", "auto_fixed", "summary"):
            assert key in data, f"missing legacy key {key}"
