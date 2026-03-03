"""Shared linter orchestration utilities.

Provides reusable functions for merging linter violations into
validation results and creating LinterRunner instances from config.
Used by both the MCP server (server.py) and CLI (validate_command.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mirdan.core.linter_runner import LinterConfig, LinterRunner
from mirdan.models import ValidationResult, Violation

if TYPE_CHECKING:
    from mirdan.config import MirdanConfig, ThresholdsConfig


def merge_linter_violations(
    base_result: ValidationResult,
    linter_violations: list[Violation],
    thresholds: ThresholdsConfig,
) -> ValidationResult:
    """Merge external linter violations into a validation result.

    Recalculates score using config-based severity weights from
    ThresholdsConfig rather than hardcoded values.

    Args:
        base_result: The original validation result (from CodeValidator).
        linter_violations: Violations from external linters.
        thresholds: Threshold config with severity weights.

    Returns:
        New ValidationResult with merged violations and recalculated score.
    """
    if not linter_violations:
        return base_result

    all_violations = list(base_result.violations) + linter_violations

    # Recalculate score using config-based weights
    error_count = sum(1 for v in all_violations if v.severity == "error")
    warning_count = sum(1 for v in all_violations if v.severity == "warning")
    info_count = sum(1 for v in all_violations if v.severity == "info")
    score = max(
        0.0,
        1.0
        - (
            error_count * thresholds.severity_error_weight
            + warning_count * thresholds.severity_warning_weight
            + info_count * thresholds.severity_info_weight
        ),
    )

    passed = error_count == 0

    return ValidationResult(
        passed=passed,
        score=score,
        language_detected=base_result.language_detected,
        violations=all_violations,
        standards_checked=[*base_result.standards_checked, "external-linters"],
        limitations=base_result.limitations,
    )


def create_linter_runner(config: MirdanConfig) -> LinterRunner:
    """Create a LinterRunner from MirdanConfig.

    Converts LinterOrcConfig (config.py) to LinterConfig (linter_runner.py).

    Args:
        config: The Mirdan configuration.

    Returns:
        Configured LinterRunner instance.
    """
    linter_config = LinterConfig(
        enabled_linters=config.linters.enabled_linters,
        ruff_args=config.linters.ruff_args,
        eslint_args=config.linters.eslint_args,
        mypy_args=config.linters.mypy_args,
        auto_detect=config.linters.auto_detect,
        timeout=config.linters.timeout,
    )
    return LinterRunner(linter_config)
