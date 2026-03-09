"""Tests for the linter orchestrator shared utilities.

Tests cover:
- merge_linter_violations() score calculation with ThresholdsConfig
- create_linter_runner() config conversion
- validate_code_quality MCP tool with file_path triggering linters
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import mirdan.server as server_mod
from mirdan.config import MirdanConfig, ThresholdsConfig
from mirdan.core.linter_orchestrator import create_linter_runner, merge_linter_violations
from mirdan.models import ValidationResult, Violation

# Extract raw async function from FastMCP FunctionTool wrapper
_validate_code_quality = server_mod.validate_code_quality.fn


@pytest.fixture(autouse=True)
def _reset_components() -> None:
    """Reset the server singleton before each test."""
    server_mod._components = None
    yield
    server_mod._components = None


@pytest.fixture()
def base_result() -> ValidationResult:
    """A clean base validation result."""
    return ValidationResult(
        passed=True,
        score=1.0,
        language_detected="python",
        violations=[],
        standards_checked=["python_style", "security"],
    )


@pytest.fixture()
def thresholds() -> ThresholdsConfig:
    """Default thresholds config."""
    return ThresholdsConfig()


# ---------------------------------------------------------------------------
# merge_linter_violations
# ---------------------------------------------------------------------------


class TestMergeLinterViolations:
    """Tests for merge_linter_violations()."""

    def test_empty_violations_returns_base(
        self, base_result: ValidationResult, thresholds: ThresholdsConfig
    ) -> None:
        """No linter violations should return the base result unchanged."""
        result = merge_linter_violations(base_result, [], thresholds)
        assert result is base_result

    def test_single_error_violation(
        self, base_result: ValidationResult, thresholds: ThresholdsConfig
    ) -> None:
        """Single error violation should fail and reduce score."""
        violations = [
            Violation(
                id="RUFF001",
                rule="unused-import",
                category="style",
                severity="error",
                message="unused import",
            )
        ]
        result = merge_linter_violations(base_result, violations, thresholds)
        assert result.passed is False
        assert result.score < 1.0
        assert "external-linters" in result.standards_checked

    def test_warning_violations_still_pass(
        self, base_result: ValidationResult, thresholds: ThresholdsConfig
    ) -> None:
        """Warning-only violations should still pass."""
        violations = [
            Violation(
                id="RUFF002",
                rule="line-too-long",
                category="style",
                severity="warning",
                message="line too long",
            )
        ]
        result = merge_linter_violations(base_result, violations, thresholds)
        assert result.passed is True
        assert result.score < 1.0

    def test_score_uses_config_weights(self) -> None:
        """Score should use ThresholdsConfig weights, not hardcoded values."""
        custom_thresholds = ThresholdsConfig(
            severity_error_weight=0.5,
            severity_warning_weight=0.1,
            severity_info_weight=0.01,
        )
        base = ValidationResult(
            passed=True,
            score=1.0,
            language_detected="python",
            violations=[],
            standards_checked=["python_style"],
        )
        violations = [
            Violation(id="E1", rule="err", category="style", severity="error", message="err"),
        ]
        result = merge_linter_violations(base, violations, custom_thresholds)
        # With 1 error at 0.5 weight: score = 1.0 - 0.5 = 0.5
        assert result.score == pytest.approx(0.5)

    def test_multiple_violations_merge(
        self, base_result: ValidationResult, thresholds: ThresholdsConfig
    ) -> None:
        """Multiple violations from different linters should all merge."""
        violations = [
            Violation(id="RUFF001", rule="r1", category="style", severity="warning", message="w1"),
            Violation(id="MYPY001", rule="m1", category="style", severity="error", message="e1"),
            Violation(id="RUFF002", rule="r2", category="style", severity="info", message="i1"),
        ]
        result = merge_linter_violations(base_result, violations, thresholds)
        assert len(result.violations) == 3
        assert result.passed is False  # has error

    def test_preserves_base_violations(self, thresholds: ThresholdsConfig) -> None:
        """Existing violations in base_result should be preserved."""
        base = ValidationResult(
            passed=False,
            score=0.75,
            language_detected="python",
            violations=[
                Violation(
                    id="SEC001",
                    rule="sec",
                    category="security",
                    severity="error",
                    message="sec issue",
                ),
            ],
            standards_checked=["security"],
        )
        linter_violations = [
            Violation(id="RUFF001", rule="r1", category="style", severity="warning", message="w1"),
        ]
        result = merge_linter_violations(base, linter_violations, thresholds)
        assert len(result.violations) == 2
        assert any(v.id == "SEC001" for v in result.violations)
        assert any(v.id == "RUFF001" for v in result.violations)

    def test_preserves_limitations(self, thresholds: ThresholdsConfig) -> None:
        """Base result limitations should be preserved."""
        base = ValidationResult(
            passed=True,
            score=1.0,
            language_detected="python",
            limitations=["test limitation"],
            standards_checked=["security"],
        )
        violations = [
            Violation(id="R1", rule="r", category="style", severity="warning", message="w"),
        ]
        result = merge_linter_violations(base, violations, thresholds)
        assert "test limitation" in result.limitations


# ---------------------------------------------------------------------------
# create_linter_runner
# ---------------------------------------------------------------------------


class TestCreateLinterRunner:
    """Tests for create_linter_runner()."""

    def test_creates_runner(self) -> None:
        """Should create a LinterRunner from MirdanConfig."""
        config = MirdanConfig()
        runner = create_linter_runner(config)
        assert isinstance(runner, server_mod.LinterRunner)

    def test_passes_config_values(self) -> None:
        """Should pass linter config values through."""
        config = MirdanConfig()
        config.linters.timeout = 60.0
        config.linters.enabled_linters = ["ruff"]
        runner = create_linter_runner(config)
        assert runner._config.timeout == 60.0
        assert runner._config.enabled_linters == ["ruff"]


# ---------------------------------------------------------------------------
# validate_code_quality MCP tool with file_path
# ---------------------------------------------------------------------------


class TestValidateCodeQualityWithLinters:
    """Tests for validate_code_quality tool with file_path parameter."""

    async def test_file_path_triggers_linters(self, tmp_path) -> None:
        """Providing file_path should trigger external linter run."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")

        c = server_mod._get_components()
        mock_run = AsyncMock(return_value=[])
        with patch.object(c.linter_runner, "run", mock_run):
            await _validate_code_quality("x = 1\n", language="python", file_path=str(test_file))

        mock_run.assert_awaited_once()

    async def test_no_file_path_skips_linters(self) -> None:
        """Without file_path, linters should not run."""
        c = server_mod._get_components()
        mock_run = AsyncMock(return_value=[])
        with patch.object(c.linter_runner, "run", mock_run):
            await _validate_code_quality("x = 1\n", language="python")

        mock_run.assert_not_awaited()

    async def test_linter_violations_merged_into_result(self, tmp_path) -> None:
        """Linter violations should appear in the final result."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")

        linter_violations = [
            Violation(
                id="RUFF001",
                rule="unused-import",
                category="style",
                severity="warning",
                message="unused import",
            )
        ]

        c = server_mod._get_components()
        mock_run = AsyncMock(return_value=linter_violations)
        with patch.object(c.linter_runner, "run", mock_run):
            result = await _validate_code_quality(
                "x = 1\n", language="python", file_path=str(test_file)
            )

        # Should have the linter violation
        ruff_violations = [v for v in result.get("violations", []) if v.get("id") == "RUFF001"]
        assert len(ruff_violations) == 1
        assert "external-linters" in result["standards_checked"]

    async def test_nonexistent_file_path_skips_linters(self) -> None:
        """Non-existent file_path should skip linters gracefully."""
        c = server_mod._get_components()
        mock_run = AsyncMock(return_value=[])
        with patch.object(c.linter_runner, "run", mock_run):
            result = await _validate_code_quality(
                "x = 1\n", language="python", file_path="/nonexistent/file.py"
            )

        mock_run.assert_not_awaited()
        assert result["passed"] is True
