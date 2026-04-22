"""Check runner — orchestrates lint + typecheck + test subprocesses."""

from __future__ import annotations

import asyncio
import logging
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mirdan.config import CheckRunnerConfig, LLMConfig

if TYPE_CHECKING:
    from mirdan.llm.manager import LLMManager
from mirdan.llm.prompts.checks import CHECK_SAMPLING, CHECK_SCHEMA, build_check_analysis_prompt
from mirdan.models import CheckResult, ModelRole, SubprocessResult

logger = logging.getLogger(__name__)


class CheckRunner:
    """Runs lint, typecheck, and test tools as subprocesses.

    Optionally uses the local FAST model to parse combined output
    into classified issues. Falls back to basic summary from exit codes
    when the LLM is unavailable.
    """

    def __init__(
        self,
        llm_manager: LLMManager | None = None,
        config: LLMConfig | None = None,
        checks_override: CheckRunnerConfig | None = None,
    ) -> None:
        self._llm = llm_manager
        self._config = config or LLMConfig()
        if checks_override is not None:
            self._config = self._config.model_copy(update={"checks": checks_override})

    async def run_all(
        self,
        file_paths: list[str] | None = None,
        working_dir: str | None = None,
    ) -> CheckResult:
        """Run lint + typecheck + test and return structured results.

        Args:
            file_paths: Specific files to check (passed to lint/typecheck).
            working_dir: Working directory for subprocesses.

        Returns:
            CheckResult with per-tool output and overall status.
        """
        cwd = working_dir or str(Path.cwd())
        checks = self._config.checks

        # 1. Lint
        lint_result = await self._run_cmd(checks.lint_command, file_paths, cwd)

        # 2. Auto-fix lint if configured and lint failed
        auto_fixed: list[str] = []
        if checks.auto_fix_lint and lint_result.returncode != 0:
            fix_result = await self._run_cmd(checks.lint_command + " --fix", file_paths, cwd)
            if fix_result.returncode == 0:
                auto_fixed.append("lint auto-fixed")
            # Re-run lint to get current state
            lint_result = await self._run_cmd(checks.lint_command, file_paths, cwd)

        # 3. Typecheck
        typecheck_result = await self._run_cmd(checks.typecheck_command, file_paths, cwd)

        # 4. Test (with configurable timeout)
        test_result = await self._run_cmd(
            checks.test_command, None, cwd, timeout=checks.test_timeout
        )

        # 5. LLM analysis (optional)
        needs_attention: list[dict[str, Any]] = []
        summary = ""
        if self._llm and self._config.check_runner:
            analysis = await self._analyze(lint_result, typecheck_result, test_result)
            if analysis:
                needs_attention = analysis.get("issues", [])
                summary = analysis.get("summary", "")

        if not summary:
            summary = self._basic_summary(lint_result, typecheck_result, test_result)

        results_list = [lint_result, typecheck_result, test_result]
        all_pass = all(r.returncode == 0 for r in results_list)
        code_quality_pass = all(r.classification != "code_quality" for r in results_list)
        infra_ok = all(r.classification != "infrastructure" for r in results_list)

        return CheckResult(
            lint=lint_result,
            typecheck=typecheck_result,
            test=test_result,
            all_pass=all_pass,
            auto_fixed=auto_fixed,
            needs_attention=needs_attention,
            code_quality_pass=code_quality_pass,
            infra_ok=infra_ok,
            summary=summary,
        )

    async def _run_cmd(
        self,
        command: str,
        files: list[str] | None,
        cwd: str,
        timeout: int = 60,
    ) -> SubprocessResult:
        """Run a shell command as an async subprocess.

        Args:
            command: Shell command string (will be split with shlex).
            files: Optional file paths to append.
            cwd: Working directory.
            timeout: Maximum seconds before killing the process.

        Returns:
            SubprocessResult with exit code and captured output.
        """
        args = shlex.split(command) + (files or [])
        full_command = " ".join(args)

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            rc = proc.returncode or 0
            return SubprocessResult(
                command=full_command,
                returncode=rc,
                stdout=stdout_bytes.decode(errors="replace"),
                stderr=stderr_bytes.decode(errors="replace"),
                classification="ok" if rc == 0 else "code_quality",
            )
        except TimeoutError:
            proc.kill()
            return SubprocessResult(
                command=full_command,
                returncode=-1,
                stdout="",
                stderr=f"TIMEOUT after {timeout}s",
                classification="infrastructure",
            )
        except FileNotFoundError:
            return SubprocessResult(
                command=full_command,
                returncode=-1,
                stdout="",
                stderr=f"Command not found: {args[0]}",
                classification="infrastructure",
            )

    async def _analyze(
        self,
        lint: SubprocessResult,
        typecheck: SubprocessResult,
        test: SubprocessResult,
    ) -> dict[str, Any] | None:
        """Use local LLM to classify issues from combined tool output.

        Args:
            lint: Lint subprocess result.
            typecheck: Typecheck subprocess result.
            test: Test subprocess result.

        Returns:
            Parsed analysis dict, or None if LLM unavailable.
        """
        if not self._llm:
            return None
        prompt = build_check_analysis_prompt(
            lint.stdout,
            lint.stderr,
            typecheck.stdout,
            typecheck.stderr,
            test.stdout,
            test.stderr,
        )
        result: dict[str, Any] | None = await self._llm.generate_structured(
            ModelRole.FAST, prompt, CHECK_SCHEMA, **CHECK_SAMPLING
        )
        return result

    @staticmethod
    def _basic_summary(
        lint: SubprocessResult,
        typecheck: SubprocessResult,
        test: SubprocessResult,
    ) -> str:
        """Generate a basic summary from exit codes when LLM is unavailable."""
        parts: list[str] = []
        if lint.returncode != 0:
            parts.append("lint errors")
        if typecheck.returncode != 0:
            parts.append("type errors")
        if test.returncode != 0:
            parts.append("test failures")

        if not parts:
            return "all checks pass"
        return f"Issues found: {', '.join(parts)}"
