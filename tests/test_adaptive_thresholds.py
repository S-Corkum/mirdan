"""Tests for adaptive file-path thresholds."""

from __future__ import annotations

import pytest

from mirdan.config import FileThresholdOverride, ThresholdsConfig
from mirdan.core.code_validator import CodeValidator
from mirdan.core.quality_standards import QualityStandards


class TestFileThresholdOverride:
    """Tests for FileThresholdOverride model."""

    def test_creation(self) -> None:
        override = FileThresholdOverride(pattern="tests/**", arch_max_function_length=50)
        assert override.pattern == "tests/**"
        assert override.arch_max_function_length == 50
        assert override.severity_error_weight is None

    def test_serialization(self) -> None:
        override = FileThresholdOverride(
            pattern="**/auth/**",
            severity_error_weight=0.35,
        )
        data = override.model_dump()
        assert data["pattern"] == "**/auth/**"
        assert data["severity_error_weight"] == 0.35
        assert data["arch_max_function_length"] is None


class TestResolveForFile:
    """Tests for ThresholdsConfig.resolve_for_file()."""

    def test_no_overrides_returns_self(self) -> None:
        config = ThresholdsConfig()
        result = config.resolve_for_file("src/main.py")
        assert result is config

    def test_matching_pattern_returns_overridden_config(self) -> None:
        config = ThresholdsConfig(
            file_overrides=[
                FileThresholdOverride(pattern="tests/**", arch_max_function_length=50),
            ]
        )
        result = config.resolve_for_file("tests/test_foo.py")
        assert result.arch_max_function_length == 50
        # Non-overridden fields retain defaults
        assert result.arch_max_file_length == 300
        assert result.arch_max_nesting_depth == 4

    def test_no_matching_pattern_returns_self(self) -> None:
        config = ThresholdsConfig(
            file_overrides=[
                FileThresholdOverride(pattern="tests/**", arch_max_function_length=50),
            ]
        )
        result = config.resolve_for_file("src/main.py")
        assert result is config

    def test_first_match_wins(self) -> None:
        config = ThresholdsConfig(
            file_overrides=[
                FileThresholdOverride(pattern="tests/**", arch_max_function_length=50),
                FileThresholdOverride(pattern="tests/**", arch_max_function_length=100),
            ]
        )
        result = config.resolve_for_file("tests/test_foo.py")
        assert result.arch_max_function_length == 50

    def test_only_set_fields_override(self) -> None:
        config = ThresholdsConfig(
            arch_max_function_length=25,
            file_overrides=[
                FileThresholdOverride(pattern="tests/**", arch_max_file_length=500),
            ]
        )
        result = config.resolve_for_file("tests/test_foo.py")
        # arch_max_file_length overridden
        assert result.arch_max_file_length == 500
        # arch_max_function_length retains base value
        assert result.arch_max_function_length == 25

    def test_auth_pattern_matches(self) -> None:
        config = ThresholdsConfig(
            file_overrides=[
                FileThresholdOverride(
                    pattern="**/auth/**",
                    severity_error_weight=0.35,
                ),
            ]
        )
        result = config.resolve_for_file("src/auth/login.py")
        assert result.severity_error_weight == 0.35

    def test_severity_weights_override(self) -> None:
        config = ThresholdsConfig(
            file_overrides=[
                FileThresholdOverride(
                    pattern="**/security/**",
                    severity_error_weight=0.5,
                    severity_warning_weight=0.15,
                ),
            ]
        )
        result = config.resolve_for_file("src/security/validator.py")
        assert result.severity_error_weight == 0.5
        assert result.severity_warning_weight == 0.15
        # Info weight retains default
        assert result.severity_info_weight == 0.02


class TestModelCopyCorrectness:
    """Tests that model_copy produces correct results."""

    def test_non_overridden_fields_retain_base(self) -> None:
        config = ThresholdsConfig(
            entity_base_confidence=0.9,
            file_overrides=[
                FileThresholdOverride(pattern="tests/**", arch_max_function_length=50),
            ]
        )
        result = config.resolve_for_file("tests/test_foo.py")
        assert result.entity_base_confidence == 0.9
        assert result.arch_max_function_length == 50


class TestCodeValidatorIntegration:
    """Tests for adaptive thresholds through CodeValidator.validate()."""

    @pytest.fixture()
    def validator(self) -> CodeValidator:
        return CodeValidator(QualityStandards())

    def test_thresholds_param_lowers_arch001_threshold(self, validator: CodeValidator) -> None:
        """Lower arch_max_function_length should produce ARCH001 where default wouldn't."""
        # 20-line function: passes default (30) but fails at threshold=15
        lines = ["def func() -> None:"] + ["    x = 1"] * 19
        code = "\n".join(lines)

        # Default thresholds — should pass
        result_default = validator.validate(code, language="python")
        arch001 = [v for v in result_default.violations if v.id == "ARCH001"]
        assert len(arch001) == 0

        # Overridden thresholds — should fail
        overridden = ThresholdsConfig(arch_max_function_length=15)
        result_strict = validator.validate(code, language="python", thresholds=overridden)
        arch001_strict = [v for v in result_strict.violations if v.id == "ARCH001"]
        assert len(arch001_strict) == 1

    def test_thresholds_none_uses_defaults(self, validator: CodeValidator) -> None:
        """thresholds=None should behave identically to no parameter."""
        code = "x = 1"
        result1 = validator.validate(code, language="python")
        result2 = validator.validate(code, language="python", thresholds=None)
        assert result1.score == result2.score
        assert len(result1.violations) == len(result2.violations)
