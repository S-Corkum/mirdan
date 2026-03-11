"""Tests for ConfidenceCalibrator."""

from mirdan.core.confidence import ConfidenceCalibrator
from mirdan.models import SemanticCheck, Violation


def _make_violation(
    severity: str = "warning",
    category: str = "style",
    **kwargs: str,
) -> Violation:
    return Violation(
        id=kwargs.get("id", "TEST001"),
        rule="test-rule",
        category=category,
        severity=severity,
        message=kwargs.get("message", "Test violation"),
        line=1,
    )


def _make_semantic_check(
    severity: str = "info",
    concern: str = "test",
    question: str = "Is this correct?",
) -> SemanticCheck:
    return SemanticCheck(
        concern=concern,
        question=question,
        severity=severity,
    )


class TestHighConfidence:
    """Tests for HIGH confidence assessment."""

    def test_no_violations_no_checks(self) -> None:
        calibrator = ConfidenceCalibrator()
        result = calibrator.assess([], [], test_file="test_foo.py")
        assert result.level == "high"

    def test_clean_code_with_test_file(self) -> None:
        calibrator = ConfidenceCalibrator()
        result = calibrator.assess([], [], test_file="tests/test_main.py")
        assert result.level == "high"
        assert "passed" in result.reason.lower()


class TestLowConfidence:
    """Tests for LOW confidence assessment."""

    def test_error_violations(self) -> None:
        calibrator = ConfidenceCalibrator()
        violations = [_make_violation(severity="error")]
        result = calibrator.assess(violations, [], test_file="test.py")
        assert result.level == "low"
        assert "error" in result.reason.lower()

    def test_security_violations(self) -> None:
        calibrator = ConfidenceCalibrator()
        violations = [_make_violation(severity="warning", category="security")]
        result = calibrator.assess(violations, [], test_file="test.py")
        assert result.level == "low"
        assert "security" in result.reason.lower()

    def test_multiple_errors(self) -> None:
        calibrator = ConfidenceCalibrator()
        violations = [
            _make_violation(severity="error", message="Error 1"),
            _make_violation(severity="error", message="Error 2"),
        ]
        result = calibrator.assess(violations, [], test_file="test.py")
        assert result.level == "low"
        assert "2" in result.reason


class TestMediumConfidence:
    """Tests for MEDIUM confidence assessment."""

    def test_many_warnings(self) -> None:
        calibrator = ConfidenceCalibrator()
        violations = [_make_violation(severity="warning") for _ in range(4)]
        result = calibrator.assess(violations, [], test_file="test.py")
        assert result.level == "medium"
        assert "warning" in result.reason.lower()

    def test_warning_semantic_checks(self) -> None:
        calibrator = ConfidenceCalibrator()
        checks = [_make_semantic_check(severity="warning")]
        result = calibrator.assess([], checks, test_file="test.py")
        assert result.level == "medium"
        assert "semantic" in result.reason.lower()

    def test_no_test_file(self) -> None:
        calibrator = ConfidenceCalibrator()
        result = calibrator.assess([], [], test_file="")
        assert result.level == "medium"
        assert "test" in result.reason.lower()

    def test_three_warnings_is_not_medium(self) -> None:
        """Exactly 3 warnings should not trigger medium (threshold is >3)."""
        calibrator = ConfidenceCalibrator()
        violations = [_make_violation(severity="warning") for _ in range(3)]
        result = calibrator.assess(violations, [], test_file="test.py")
        assert result.level == "high"


class TestAttentionFocus:
    """Tests for attention_focus selection."""

    def test_picks_highest_severity_semantic_check(self) -> None:
        calibrator = ConfidenceCalibrator()
        violations = [_make_violation(severity="error")]
        checks = [
            _make_semantic_check(severity="info", question="Minor issue"),
            _make_semantic_check(severity="critical", question="Critical concern"),
            _make_semantic_check(severity="warning", question="Warning issue"),
        ]
        result = calibrator.assess(violations, checks, test_file="test.py")
        assert result.attention_focus == "Critical concern"

    def test_falls_back_to_violation_message(self) -> None:
        calibrator = ConfidenceCalibrator()
        violations = [_make_violation(severity="error", message="SQL injection found")]
        result = calibrator.assess(violations, [], test_file="test.py")
        assert result.attention_focus == "SQL injection found"

    def test_default_when_no_checks_no_violations(self) -> None:
        calibrator = ConfidenceCalibrator()
        result = calibrator.assess([], [], test_file="test.py")
        assert "clean" in result.attention_focus.lower() or "none" in result.attention_focus.lower()


class TestEmptyInputs:
    """Tests with empty inputs."""

    def test_empty_everything(self) -> None:
        calibrator = ConfidenceCalibrator()
        result = calibrator.assess([], [])
        # No test file → medium
        assert result.level == "medium"

    def test_to_dict(self) -> None:
        calibrator = ConfidenceCalibrator()
        result = calibrator.assess([], [], test_file="test.py")
        d = result.to_dict()
        assert "level" in d
        assert "reason" in d
        assert "attention_focus" in d
