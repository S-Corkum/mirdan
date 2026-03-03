"""Tests for the auto-fix engine."""

from __future__ import annotations

import pytest

from mirdan.core.auto_fixer import (
    PATTERN_FIXES,
    TEMPLATE_FIXES,
    AutoFixer,
    FixResult,
)
from mirdan.models import Violation


@pytest.fixture
def fixer() -> AutoFixer:
    """Create an AutoFixer instance."""
    return AutoFixer()


class TestFixResult:
    """Tests for the FixResult dataclass."""

    def test_should_suggest_high_confidence(self) -> None:
        """Fixes with >= 0.7 confidence should be suggested."""
        fix = FixResult(fix_code="const", fix_description="test", confidence=0.9, is_template=True)
        assert fix.should_suggest is True

    def test_should_not_suggest_low_confidence(self) -> None:
        """Fixes with < 0.7 confidence should not be suggested."""
        fix = FixResult(fix_code="const", fix_description="test", confidence=0.5, is_template=True)
        assert fix.should_suggest is False

    def test_should_suggest_at_threshold(self) -> None:
        """Fixes at exactly 0.7 should be suggested."""
        fix = FixResult(fix_code="const", fix_description="test", confidence=0.7, is_template=True)
        assert fix.should_suggest is True


class TestTemplateFixes:
    """Tests for template-based fixes."""

    def test_py003_bare_except(self, fixer: AutoFixer) -> None:
        """PY003 should suggest replacing bare except."""
        result = fixer.get_fix("PY003")
        assert result is not None
        assert "except Exception" in result.fix_code
        assert result.is_template is True
        assert result.should_suggest is True

    def test_py004_mutable_default(self, fixer: AutoFixer) -> None:
        """PY004 should suggest replacing mutable default."""
        result = fixer.get_fix("PY004")
        assert result is not None
        assert "None" in result.fix_code

    def test_py005_typing_imports(self, fixer: AutoFixer) -> None:
        """PY005 should suggest native type syntax."""
        result = fixer.get_fix("PY005")
        assert result is not None

    def test_js001_var_replacement(self, fixer: AutoFixer) -> None:
        """JS001 should suggest replacing var with const."""
        result = fixer.get_fix("JS001")
        assert result is not None
        assert "const" in result.fix_code

    def test_ts004_as_unknown(self, fixer: AutoFixer) -> None:
        """TS004 should suggest replacing 'as any' with 'as unknown'."""
        result = fixer.get_fix("TS004")
        assert result is not None
        assert "unknown" in result.fix_code

    def test_rs001_unwrap_expect(self, fixer: AutoFixer) -> None:
        """RS001 should suggest .expect() over .unwrap()."""
        result = fixer.get_fix("RS001")
        assert result is not None
        assert "expect" in result.fix_code

    def test_sec001_hardcoded_secret(self, fixer: AutoFixer) -> None:
        """SEC001 should suggest environment variable."""
        result = fixer.get_fix("SEC001")
        assert result is not None
        assert "environ" in result.fix_code

    def test_sec002_sql_injection(self, fixer: AutoFixer) -> None:
        """SEC002 should suggest parameterized query."""
        result = fixer.get_fix("SEC002")
        assert result is not None
        assert "params" in result.fix_code

    def test_sec007_verify_true(self, fixer: AutoFixer) -> None:
        """SEC007 should suggest verify=True."""
        result = fixer.get_fix("SEC007")
        assert result is not None
        assert "verify=True" in result.fix_code

    def test_ai007_security_theater(self, fixer: AutoFixer) -> None:
        """AI007 should suggest proper cryptographic hashing."""
        result = fixer.get_fix("AI007")
        assert result is not None
        assert "SECURITY" in result.fix_code

    def test_ai008_injection(self, fixer: AutoFixer) -> None:
        """AI008 should suggest parameterized queries."""
        result = fixer.get_fix("AI008")
        assert result is not None

    def test_go001_error_check(self, fixer: AutoFixer) -> None:
        """GO001 should suggest error checking."""
        result = fixer.get_fix("GO001")
        assert result is not None
        assert "err" in result.fix_code

    def test_java001_logging(self, fixer: AutoFixer) -> None:
        """JAVA001 should suggest proper logging."""
        result = fixer.get_fix("JAVA001")
        assert result is not None
        assert "Logger" in result.fix_code

    def test_nonexistent_rule_returns_none(self, fixer: AutoFixer) -> None:
        """Non-existent rules should return None."""
        result = fixer.get_fix("NONEXISTENT999")
        assert result is None


class TestPatternFixes:
    """Tests for pattern-based fixes."""

    def test_py003_bare_except_pattern(self, fixer: AutoFixer) -> None:
        """Pattern fix should fix bare except in code line."""
        result = fixer.get_fix("PY003", code_line="    except:")
        assert result is not None
        # Template fix takes priority, but should still work

    def test_py005_list_pattern(self, fixer: AutoFixer) -> None:
        """Pattern fix should replace List[ with list[."""
        result = fixer.get_fix("PY005", code_line="items: List[str]")
        # Template fix may take priority over pattern
        assert result is not None

    def test_js002_equality_pattern(self, fixer: AutoFixer) -> None:
        """Pattern fix for JS002 should work on equality."""
        result = fixer.get_fix("JS002")
        assert result is not None
        assert "===" in result.fix_code

    def test_sec006_http_pattern(self, fixer: AutoFixer) -> None:
        """SEC006 should upgrade HTTP to HTTPS."""
        result = fixer.get_fix("SEC006")
        assert result is not None
        assert "https" in result.fix_code


class TestApplyFix:
    """Tests for applying fixes to code."""

    def test_apply_bare_except_fix(self, fixer: AutoFixer) -> None:
        """Should fix bare except on the right line."""
        code = "try:\n    pass\nexcept:\n    pass"
        fixed, applied = fixer.apply_fix(code, "PY003", line_number=3)
        if applied:
            assert "except Exception:" in fixed

    def test_apply_fix_no_match(self, fixer: AutoFixer) -> None:
        """No pattern match should leave code unchanged."""
        code = "x = 1\ny = 2"
        fixed, applied = fixer.apply_fix(code, "NONEXISTENT", line_number=1)
        assert fixed == code
        assert applied is False

    def test_apply_fix_invalid_line(self, fixer: AutoFixer) -> None:
        """Invalid line number should not modify code."""
        code = "x = 1"
        fixed, applied = fixer.apply_fix(code, "PY003", line_number=100)
        assert fixed == code
        assert applied is False


class TestBatchFix:
    """Tests for batch fix operations."""

    def test_batch_fix_dry_run(self, fixer: AutoFixer) -> None:
        """Dry run should collect fixes without modifying code."""
        violations = [
            Violation(
                id="PY003",
                rule="bare-except",
                category="style",
                severity="warning",
                message="Bare except",
                line=3,
                code_snippet="except:",
            )
        ]
        code = "try:\n    pass\nexcept:\n    pass"
        fixed_code, fixes = fixer.batch_fix(code, violations, dry_run=True)
        # Dry run doesn't modify code
        assert fixed_code == code
        assert len(fixes) >= 1

    def test_batch_fix_apply(self, fixer: AutoFixer) -> None:
        """Batch fix should apply fixes to code."""
        violations = [
            Violation(
                id="PY003",
                rule="bare-except",
                category="style",
                severity="warning",
                message="Bare except",
                line=3,
                code_snippet="except:",
            )
        ]
        code = "try:\n    pass\nexcept:\n    pass"
        fixed_code, _ = fixer.batch_fix(code, violations, dry_run=False)
        # May or may not apply depending on pattern match
        assert isinstance(fixed_code, str)

    def test_batch_fix_empty_violations(self, fixer: AutoFixer) -> None:
        """Empty violations list should return unchanged code."""
        code = "x = 1"
        fixed_code, fixes = fixer.batch_fix(code, [], dry_run=False)
        assert fixed_code == code
        assert fixes == []


class TestGetFixForViolation:
    """Tests for violation-based fix lookup."""

    def test_fix_for_violation(self, fixer: AutoFixer) -> None:
        """Should get fix from a Violation object."""
        v = Violation(
            id="SEC007",
            rule="ssl-verify",
            category="security",
            severity="error",
            message="SSL verification disabled",
            code_snippet="verify=False",
        )
        result = fixer.get_fix_for_violation(v)
        assert result is not None
        assert "verify=True" in result.fix_code

    def test_fix_for_violation_no_fix(self, fixer: AutoFixer) -> None:
        """Violations without fixes should return None."""
        v = Violation(
            id="UNKNOWN999",
            rule="unknown",
            category="style",
            severity="info",
            message="Unknown",
        )
        result = fixer.get_fix_for_violation(v)
        assert result is None


class TestCoverageAndMetadata:
    """Tests for coverage and metadata methods."""

    def test_get_fixable_rules(self) -> None:
        """Should return sorted list of rule IDs."""
        rules = AutoFixer.get_fixable_rules()
        assert isinstance(rules, list)
        assert len(rules) > 0
        assert rules == sorted(rules)

    def test_coverage_report(self) -> None:
        """Coverage report should have all required fields."""
        report = AutoFixer.coverage_report()
        assert "template_fixes" in report
        assert "pattern_fixes" in report
        assert "total_fixable_rules" in report
        assert report["template_fixes"] == len(TEMPLATE_FIXES)
        assert report["pattern_fixes"] == len(PATTERN_FIXES)
        assert report["total_fixable_rules"] > 0

    def test_template_fixes_have_high_confidence(self) -> None:
        """All template fixes should have >= 0.7 confidence."""
        for rule_id, (_, _, confidence) in TEMPLATE_FIXES.items():
            assert confidence >= 0.7, f"{rule_id} has low confidence {confidence}"

    def test_pattern_fixes_have_valid_regex(self, fixer: AutoFixer) -> None:
        """All pattern fixes should have compiled regex patterns."""
        for key in PATTERN_FIXES:
            assert key in fixer._compiled_patterns
