"""Tests for CheckRunner subprocess orchestration."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirdan.config import CheckRunnerConfig, LLMConfig
from mirdan.core.check_runner import CheckRunner
from mirdan.models import SubprocessResult


def _make_subprocess_result(
    command: str = "test",
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> SubprocessResult:
    return SubprocessResult(
        command=command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


# ---------------------------------------------------------------------------
# run_all()
# ---------------------------------------------------------------------------


class TestCheckRunnerRunAll:
    """Tests for CheckRunner.run_all()."""

    @pytest.mark.asyncio
    async def test_all_pass(self) -> None:
        runner = CheckRunner()

        async def mock_run(cmd: str, files: Any, cwd: str, **kw: Any) -> SubprocessResult:
            return _make_subprocess_result(command=cmd, returncode=0)

        with patch.object(runner, "_run_cmd", side_effect=mock_run):
            result = await runner.run_all()

        assert result.all_pass is True
        assert "all checks pass" in result.summary

    @pytest.mark.asyncio
    async def test_lint_fails(self) -> None:
        runner = CheckRunner()
        call_count = 0

        async def mock_run(cmd: str, files: Any, cwd: str, **kw: Any) -> SubprocessResult:
            nonlocal call_count
            call_count += 1
            if "ruff" in cmd and "--fix" not in cmd:
                return _make_subprocess_result(command=cmd, returncode=1, stdout="E001 error")
            return _make_subprocess_result(command=cmd, returncode=0)

        with patch.object(runner, "_run_cmd", side_effect=mock_run):
            result = await runner.run_all()

        assert result.all_pass is False
        assert "lint errors" in result.summary

    @pytest.mark.asyncio
    async def test_typecheck_fails(self) -> None:
        runner = CheckRunner()

        async def mock_run(cmd: str, files: Any, cwd: str, **kw: Any) -> SubprocessResult:
            if "mypy" in cmd:
                return _make_subprocess_result(command=cmd, returncode=1)
            return _make_subprocess_result(command=cmd, returncode=0)

        with patch.object(runner, "_run_cmd", side_effect=mock_run):
            result = await runner.run_all()

        assert result.all_pass is False
        assert "type errors" in result.summary

    @pytest.mark.asyncio
    async def test_test_fails(self) -> None:
        runner = CheckRunner()

        async def mock_run(cmd: str, files: Any, cwd: str, **kw: Any) -> SubprocessResult:
            if "pytest" in cmd:
                return _make_subprocess_result(command=cmd, returncode=1)
            return _make_subprocess_result(command=cmd, returncode=0)

        with patch.object(runner, "_run_cmd", side_effect=mock_run):
            result = await runner.run_all()

        assert result.all_pass is False
        assert "test failures" in result.summary

    @pytest.mark.asyncio
    async def test_multiple_failures(self) -> None:
        runner = CheckRunner()

        async def mock_run(cmd: str, files: Any, cwd: str, **kw: Any) -> SubprocessResult:
            return _make_subprocess_result(command=cmd, returncode=1)

        with patch.object(runner, "_run_cmd", side_effect=mock_run):
            result = await runner.run_all()

        assert result.all_pass is False
        assert "lint errors" in result.summary
        assert "type errors" in result.summary
        assert "test failures" in result.summary


# ---------------------------------------------------------------------------
# Auto-fix
# ---------------------------------------------------------------------------


class TestCheckRunnerAutoFix:
    """Tests for lint auto-fix behavior."""

    @pytest.mark.asyncio
    async def test_auto_fix_when_lint_fails(self) -> None:
        config = LLMConfig(checks=CheckRunnerConfig(auto_fix_lint=True))
        runner = CheckRunner(config=config)
        commands_run: list[str] = []

        async def mock_run(cmd: str, files: Any, cwd: str, **kw: Any) -> SubprocessResult:
            commands_run.append(cmd)
            if "ruff check --fix" in cmd:
                return _make_subprocess_result(command=cmd, returncode=0)
            if "ruff check" in cmd and len(commands_run) <= 1:
                return _make_subprocess_result(command=cmd, returncode=1)
            return _make_subprocess_result(command=cmd, returncode=0)

        with patch.object(runner, "_run_cmd", side_effect=mock_run):
            result = await runner.run_all()

        assert any("--fix" in c for c in commands_run)
        assert result.auto_fixed == ["lint auto-fixed"]

    @pytest.mark.asyncio
    async def test_no_auto_fix_when_disabled(self) -> None:
        config = LLMConfig(checks=CheckRunnerConfig(auto_fix_lint=False))
        runner = CheckRunner(config=config)
        commands_run: list[str] = []

        async def mock_run(cmd: str, files: Any, cwd: str, **kw: Any) -> SubprocessResult:
            commands_run.append(cmd)
            if "ruff" in cmd:
                return _make_subprocess_result(command=cmd, returncode=1)
            return _make_subprocess_result(command=cmd, returncode=0)

        with patch.object(runner, "_run_cmd", side_effect=mock_run):
            await runner.run_all()

        assert not any("--fix" in c for c in commands_run)


# ---------------------------------------------------------------------------
# LLM analysis
# ---------------------------------------------------------------------------


class TestCheckRunnerLLMAnalysis:
    """Tests for LLM-enhanced analysis."""

    @pytest.mark.asyncio
    async def test_uses_llm_when_available(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = {
            "issues": [
                {
                    "tool": "lint",
                    "file": "main.py",
                    "line": 5,
                    "message": "unused import",
                    "classification": "trivial",
                }
            ],
            "summary": "1 trivial lint issue",
        }

        config = LLMConfig(check_runner=True)
        runner = CheckRunner(llm_manager=mock_llm, config=config)

        async def mock_run(cmd: str, files: Any, cwd: str, **kw: Any) -> SubprocessResult:
            return _make_subprocess_result(command=cmd, returncode=0)

        with patch.object(runner, "_run_cmd", side_effect=mock_run):
            result = await runner.run_all()

        assert result.summary == "1 trivial lint issue"
        assert len(result.needs_attention) == 1
        assert result.needs_attention[0]["classification"] == "trivial"

    @pytest.mark.asyncio
    async def test_falls_back_without_llm(self) -> None:
        runner = CheckRunner(llm_manager=None)

        async def mock_run(cmd: str, files: Any, cwd: str, **kw: Any) -> SubprocessResult:
            return _make_subprocess_result(command=cmd, returncode=0)

        with patch.object(runner, "_run_cmd", side_effect=mock_run):
            result = await runner.run_all()

        assert result.summary == "all checks pass"
        assert result.needs_attention == []

    @pytest.mark.asyncio
    async def test_llm_returns_none_uses_basic_summary(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = None

        config = LLMConfig(check_runner=True)
        runner = CheckRunner(llm_manager=mock_llm, config=config)

        async def mock_run(cmd: str, files: Any, cwd: str, **kw: Any) -> SubprocessResult:
            if "ruff" in cmd:
                return _make_subprocess_result(command=cmd, returncode=1)
            return _make_subprocess_result(command=cmd, returncode=0)

        with patch.object(runner, "_run_cmd", side_effect=mock_run):
            result = await runner.run_all()

        assert "lint errors" in result.summary


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------


class TestCheckRunnerSubprocess:
    """Tests for _run_cmd subprocess execution."""

    @pytest.mark.asyncio
    async def test_captures_stdout_stderr(self) -> None:
        runner = CheckRunner()

        async def mock_exec(*args: Any, **kw: Any) -> MagicMock:
            proc = MagicMock()
            proc.communicate = AsyncMock(return_value=(b"output", b"error"))
            proc.returncode = 0
            return proc

        with patch(
            "mirdan.core.check_runner.asyncio.create_subprocess_exec", side_effect=mock_exec
        ):
            result = await runner._run_cmd("echo hello", None, "/tmp")

        assert result.stdout == "output"
        assert result.stderr == "error"
        assert result.returncode == 0

    @pytest.mark.asyncio
    async def test_timeout_kills_process(self) -> None:
        runner = CheckRunner()

        async def mock_exec(*args: Any, **kw: Any) -> MagicMock:
            proc = MagicMock()

            async def slow_communicate() -> tuple[bytes, bytes]:
                await asyncio.sleep(100)
                return b"", b""

            proc.communicate = slow_communicate
            proc.kill = MagicMock()
            return proc

        with patch(
            "mirdan.core.check_runner.asyncio.create_subprocess_exec", side_effect=mock_exec
        ):
            result = await runner._run_cmd("slow", None, "/tmp", timeout=0)

        assert result.returncode == -1
        assert "TIMEOUT" in result.stderr

    @pytest.mark.asyncio
    async def test_command_not_found(self) -> None:
        runner = CheckRunner()

        with patch(
            "mirdan.core.check_runner.asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("not found"),
        ):
            result = await runner._run_cmd("nonexistent", None, "/tmp")

        assert result.returncode == -1
        assert "Command not found" in result.stderr

    @pytest.mark.asyncio
    async def test_passes_file_paths(self) -> None:
        runner = CheckRunner()
        captured_args: list[Any] = []

        async def mock_exec(*args: Any, **kw: Any) -> MagicMock:
            captured_args.extend(args)
            proc = MagicMock()
            proc.communicate = AsyncMock(return_value=(b"", b""))
            proc.returncode = 0
            return proc

        with patch(
            "mirdan.core.check_runner.asyncio.create_subprocess_exec", side_effect=mock_exec
        ):
            await runner._run_cmd("ruff check", ["a.py", "b.py"], "/tmp")

        assert "a.py" in captured_args
        assert "b.py" in captured_args


# ---------------------------------------------------------------------------
# Basic summary
# ---------------------------------------------------------------------------


class TestBasicSummary:
    """Tests for _basic_summary fallback."""

    def test_all_pass(self) -> None:
        r = _make_subprocess_result(returncode=0)
        assert CheckRunner._basic_summary(r, r, r) == "all checks pass"

    def test_single_failure(self) -> None:
        ok = _make_subprocess_result(returncode=0)
        fail = _make_subprocess_result(returncode=1)
        assert "lint errors" in CheckRunner._basic_summary(fail, ok, ok)

    def test_all_fail(self) -> None:
        fail = _make_subprocess_result(returncode=1)
        summary = CheckRunner._basic_summary(fail, fail, fail)
        assert "lint errors" in summary
        assert "type errors" in summary
        assert "test failures" in summary
