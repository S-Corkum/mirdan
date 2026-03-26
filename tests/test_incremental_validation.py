"""Tests for incremental validation: RuleTier, check_by_tier, changed_lines, scope."""

from __future__ import annotations

import pytest

from mirdan.core.rules.base import BaseRule, RuleContext, RuleRegistry, RuleTier
from mirdan.models import Violation
from mirdan.server import _parse_changed_lines

# ---------------------------------------------------------------------------
# Test helpers: stub rules at different tiers
# ---------------------------------------------------------------------------


class _QuickStubRule(BaseRule):
    @property
    def id(self) -> str:
        return "STUB_QUICK"

    @property
    def name(self) -> str:
        return "stub-quick"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "auto"})

    @property
    def is_quick(self) -> bool:
        return True

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        return [
            Violation(
                id="STUB_QUICK",
                rule="stub-quick",
                category="test",
                severity="warning",
                message="quick stub",
                line=1,
            )
        ]


class _EssentialStubRule(BaseRule):
    @property
    def id(self) -> str:
        return "STUB_ESSENTIAL"

    @property
    def name(self) -> str:
        return "stub-essential"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "auto"})

    @property
    def tier(self) -> RuleTier:
        return RuleTier.ESSENTIAL

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        return [
            Violation(
                id="STUB_ESSENTIAL",
                rule="stub-essential",
                category="test",
                severity="warning",
                message="essential stub",
                line=5,
            )
        ]


class _FullStubRule(BaseRule):
    @property
    def id(self) -> str:
        return "STUB_FULL"

    @property
    def name(self) -> str:
        return "stub-full"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "auto"})

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        return [
            Violation(
                id="STUB_FULL",
                rule="stub-full",
                category="test",
                severity="warning",
                message="full stub",
                line=10,
            )
        ]


class _MultiLineStubRule(BaseRule):
    """Stub that produces violations on lines 5, 10, 20, 30."""

    @property
    def id(self) -> str:
        return "STUB_MULTI"

    @property
    def name(self) -> str:
        return "stub-multi"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "auto"})

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        return [
            Violation(
                id="STUB_MULTI",
                rule="stub-multi",
                category="test",
                severity="warning",
                message=f"line {ln}",
                line=ln,
            )
            for ln in (5, 10, 20, 30)
        ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRuleTier:
    def test_ordering(self) -> None:
        assert RuleTier.QUICK < RuleTier.ESSENTIAL < RuleTier.FULL

    def test_values(self) -> None:
        assert RuleTier.QUICK.value == 0
        assert RuleTier.ESSENTIAL.value == 1
        assert RuleTier.FULL.value == 2


class TestRuleTierDefaulting:
    def test_quick_rule_gets_quick_tier(self) -> None:
        rule = _QuickStubRule()
        assert rule.is_quick is True
        assert rule.tier == RuleTier.QUICK

    def test_non_quick_rule_gets_full_tier(self) -> None:
        rule = _FullStubRule()
        assert rule.is_quick is False
        assert rule.tier == RuleTier.FULL

    def test_explicit_essential_tier(self) -> None:
        rule = _EssentialStubRule()
        assert rule.tier == RuleTier.ESSENTIAL


class TestCheckByTier:
    @pytest.fixture()
    def registry(self) -> RuleRegistry:
        reg = RuleRegistry()
        reg.register(_QuickStubRule())
        reg.register(_EssentialStubRule())
        reg.register(_FullStubRule())
        return reg

    def test_quick_only(self, registry: RuleRegistry) -> None:
        ctx = RuleContext(skip_regions=[])
        violations = registry.check_by_tier("x = 1", "python", ctx, max_tier=RuleTier.QUICK)
        ids = {v.id for v in violations}
        assert ids == {"STUB_QUICK"}

    def test_essential_includes_quick(self, registry: RuleRegistry) -> None:
        ctx = RuleContext(skip_regions=[])
        violations = registry.check_by_tier("x = 1", "python", ctx, max_tier=RuleTier.ESSENTIAL)
        ids = {v.id for v in violations}
        assert ids == {"STUB_QUICK", "STUB_ESSENTIAL"}

    def test_full_includes_all(self, registry: RuleRegistry) -> None:
        ctx = RuleContext(skip_regions=[])
        violations = registry.check_by_tier("x = 1", "python", ctx, max_tier=RuleTier.FULL)
        ids = {v.id for v in violations}
        assert ids == {"STUB_QUICK", "STUB_ESSENTIAL", "STUB_FULL"}

    def test_backward_compat_check_all(self, registry: RuleRegistry) -> None:
        ctx = RuleContext(skip_regions=[])
        violations = registry.check_all("x = 1", "python", ctx)
        ids = {v.id for v in violations}
        assert ids == {"STUB_QUICK", "STUB_ESSENTIAL", "STUB_FULL"}


class TestParseChangedLines:
    def test_empty_returns_none(self) -> None:
        assert _parse_changed_lines("") is None

    def test_single_line(self) -> None:
        assert _parse_changed_lines("5") == frozenset({5})

    def test_multiple_lines(self) -> None:
        assert _parse_changed_lines("1,5,10") == frozenset({1, 5, 10})

    def test_range(self) -> None:
        assert _parse_changed_lines("10-15") == frozenset({10, 11, 12, 13, 14, 15})

    def test_mixed(self) -> None:
        assert _parse_changed_lines("1,5,10-12,20") == frozenset({1, 5, 10, 11, 12, 20})

    def test_reversed_range(self) -> None:
        assert _parse_changed_lines("15-10") == frozenset({10, 11, 12, 13, 14, 15})

    def test_invalid_skipped(self) -> None:
        assert _parse_changed_lines("abc") is None

    def test_negative_skipped(self) -> None:
        assert _parse_changed_lines("-5") is None

    def test_whitespace(self) -> None:
        assert _parse_changed_lines("  ") is None

    def test_mixed_valid_invalid(self) -> None:
        result = _parse_changed_lines("5,abc,10")
        assert result == frozenset({5, 10})


class TestValidateQuickScope:
    """Test validate_quick scope parameter via CodeValidator."""

    def test_security_scope_default_behavior(self) -> None:
        from mirdan.core.code_validator import CodeValidator
        from mirdan.core.quality_standards import QualityStandards

        validator = CodeValidator(QualityStandards())
        # Bare except is a Python style rule (PY003), not a security rule.
        # Security scope should NOT catch it.
        code = "try:\n    pass\nexcept:\n    pass\n"
        result = validator.validate_quick(code=code, language="python", scope="security")
        py_style_violations = [v for v in result.violations if v.id == "PY003"]
        assert len(py_style_violations) == 0

    def test_essential_scope_catches_style(self) -> None:
        from mirdan.core.code_validator import CodeValidator
        from mirdan.core.quality_standards import QualityStandards

        validator = CodeValidator(QualityStandards())
        # Essential scope SHOULD catch PY003 (bare except)
        code = "try:\n    pass\nexcept:\n    pass\n"
        result = validator.validate_quick(code=code, language="python", scope="essential")
        py_style_violations = [v for v in result.violations if v.id == "PY003"]
        assert len(py_style_violations) >= 1

    def test_invalid_scope_falls_back(self) -> None:
        from mirdan.core.code_validator import CodeValidator
        from mirdan.core.quality_standards import QualityStandards

        validator = CodeValidator(QualityStandards())
        code = "try:\n    pass\nexcept:\n    pass\n"
        result = validator.validate_quick(code=code, language="python", scope="foobar")
        # Falls back to security scope — no PY003
        py_style_violations = [v for v in result.violations if v.id == "PY003"]
        assert len(py_style_violations) == 0


class TestChangedLinesFiltering:
    """Test changed_lines filtering in CodeValidator.validate()."""

    def test_filters_to_changed_lines(self) -> None:
        from mirdan.core.code_validator import CodeValidator
        from mirdan.core.quality_standards import QualityStandards

        validator = CodeValidator(QualityStandards())
        # Bare except is on line 3. changed_lines={100} is far outside ±2 buffer.
        code = "try:\n    pass\nexcept:\n    pass\n"
        result_all = validator.validate(code=code, language="python")
        result_filtered = validator.validate(
            code=code, language="python", changed_lines=frozenset({100})
        )
        # Line 3 violation should be excluded when only line 100 is changed
        py003_all = [v for v in result_all.violations if v.id == "PY003"]
        py003_filtered = [v for v in result_filtered.violations if v.id == "PY003"]
        assert len(py003_all) >= 1
        assert len(py003_filtered) == 0

    def test_buffer_includes_nearby(self) -> None:
        from mirdan.core.code_validator import CodeValidator
        from mirdan.core.quality_standards import QualityStandards

        validator = CodeValidator(QualityStandards())
        # Bare except is on line 3. changed_lines={1} with ±2 buffer includes lines -1..3
        code = "try:\n    pass\nexcept:\n    pass\n"
        result = validator.validate(code=code, language="python", changed_lines=frozenset({1}))
        py003 = [v for v in result.violations if v.id == "PY003"]
        # Line 3 is within ±2 of line 1, so it should be included
        assert len(py003) >= 1

    def test_none_returns_all(self) -> None:
        from mirdan.core.code_validator import CodeValidator
        from mirdan.core.quality_standards import QualityStandards

        validator = CodeValidator(QualityStandards())
        code = "try:\n    pass\nexcept:\n    pass\n"
        result = validator.validate(code=code, language="python", changed_lines=None)
        py003 = [v for v in result.violations if v.id == "PY003"]
        assert len(py003) >= 1


class TestValidateQuickChangedLines:
    """Test changed_lines in validate_quick."""

    def test_filters_violations(self) -> None:
        from mirdan.core.code_validator import CodeValidator
        from mirdan.core.quality_standards import QualityStandards

        validator = CodeValidator(QualityStandards())
        # In essential scope, PY003 fires on line 3
        code = "try:\n    pass\nexcept:\n    pass\n"
        result = validator.validate_quick(
            code=code,
            language="python",
            scope="essential",
            changed_lines=frozenset({100}),  # Far from line 3
        )
        py003 = [v for v in result.violations if v.id == "PY003"]
        assert len(py003) == 0
