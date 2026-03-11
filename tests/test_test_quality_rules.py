"""Tests for test quality rules (TEST001-TEST010)."""

from __future__ import annotations

import pytest

from mirdan.core.ai_quality_checker import AIQualityChecker
from mirdan.core.rules.base import RuleContext, RuleTier
from mirdan.core.rules.test_body_rules import (
    TEST001EmptyTestRule,
    TEST002AssertTrueRule,
    TEST003NoAssertionsRule,
    TEST005MockAbuseRule,
    TEST010BroadExceptionRule,
)
from mirdan.core.rules.test_structure_rules import (
    TEST004NoCoverageRule,
    TEST006DuplicateTestRule,
    TEST007MissingEdgeCaseRule,
    TEST008HardcodedDataRule,
    TEST009ExecutionOrderRule,
)


@pytest.fixture()
def test_context() -> RuleContext:
    """RuleContext with is_test=True."""
    return RuleContext(skip_regions=[], is_test=True)


@pytest.fixture()
def non_test_context() -> RuleContext:
    """RuleContext with is_test=False."""
    return RuleContext(skip_regions=[], is_test=False)


class TestTEST001EmptyTest:
    def test_detects_pass_body(self, test_context: RuleContext) -> None:
        code = "def test_something():\n    pass\n"
        rule = TEST001EmptyTestRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 1
        assert violations[0].id == "TEST001"

    def test_detects_ellipsis_body(self, test_context: RuleContext) -> None:
        code = "def test_something():\n    ...\n"
        rule = TEST001EmptyTestRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 1

    def test_detects_docstring_only(self, test_context: RuleContext) -> None:
        code = 'def test_something():\n    """A test."""\n'
        rule = TEST001EmptyTestRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 1

    def test_no_trigger_normal_test(self, test_context: RuleContext) -> None:
        code = "def test_something():\n    assert 1 + 1 == 2\n"
        rule = TEST001EmptyTestRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 0

    def test_no_trigger_non_test_function(self, test_context: RuleContext) -> None:
        code = "def helper():\n    pass\n"
        rule = TEST001EmptyTestRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 0


class TestTEST002AssertTrue:
    def test_detects_assert_true(self, test_context: RuleContext) -> None:
        code = "def test_something():\n    assert True\n"
        rule = TEST002AssertTrueRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 1
        assert violations[0].id == "TEST002"

    def test_detects_assert_one(self, test_context: RuleContext) -> None:
        code = "def test_something():\n    assert 1\n"
        rule = TEST002AssertTrueRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 1

    def test_no_trigger_real_assertion(self, test_context: RuleContext) -> None:
        code = "def test_something():\n    assert result == 42\n"
        rule = TEST002AssertTrueRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 0

    def test_no_trigger_multiple_statements(self, test_context: RuleContext) -> None:
        code = "def test_something():\n    x = 1\n    assert True\n"
        rule = TEST002AssertTrueRule()
        violations = rule.check(code, "python", test_context)
        # Not flagged because assert True is not the ONLY statement
        assert len(violations) == 0


class TestTEST003NoAssertions:
    def test_detects_no_asserts(self, test_context: RuleContext) -> None:
        code = "def test_something():\n    x = 1 + 1\n    print(x)\n"
        rule = TEST003NoAssertionsRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 1
        assert violations[0].id == "TEST003"

    def test_no_trigger_with_assert(self, test_context: RuleContext) -> None:
        code = "def test_something():\n    assert 1 + 1 == 2\n"
        rule = TEST003NoAssertionsRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 0

    def test_no_trigger_pytest_raises(self, test_context: RuleContext) -> None:
        code = "def test_something():\n    with pytest.raises(ValueError):\n        func()\n"
        rule = TEST003NoAssertionsRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 0

    def test_no_trigger_mock_assert(self, test_context: RuleContext) -> None:
        code = "def test_something():\n    mock.assert_called_once()\n"
        rule = TEST003NoAssertionsRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 0


class TestTEST004NoCoverage:
    def test_detects_no_implementation_refs(self, test_context: RuleContext) -> None:
        impl_code = "def calculate_total(items):\n    return sum(items)\n"
        test_code = "def test_something():\n    assert 1 + 1 == 2\n"
        ctx = RuleContext(skip_regions=[], is_test=True, implementation_code=impl_code)
        rule = TEST004NoCoverageRule()
        violations = rule.check(test_code, "python", ctx)
        assert len(violations) == 1
        assert violations[0].id == "TEST004"

    def test_no_trigger_with_ref(self, test_context: RuleContext) -> None:
        impl_code = "def calculate_total(items):\n    return sum(items)\n"
        test_code = (
            "def test_total():\n    result = calculate_total([1, 2, 3])\n    assert result == 6\n"
        )
        ctx = RuleContext(skip_regions=[], is_test=True, implementation_code=impl_code)
        rule = TEST004NoCoverageRule()
        violations = rule.check(test_code, "python", ctx)
        assert len(violations) == 0

    def test_returns_empty_without_implementation(self, test_context: RuleContext) -> None:
        test_code = "def test_something():\n    assert True\n"
        rule = TEST004NoCoverageRule()
        violations = rule.check(test_code, "python", test_context)
        assert len(violations) == 0


class TestTEST005MockAbuse:
    def test_detects_excessive_patches(self, test_context: RuleContext) -> None:
        code = (
            "from unittest.mock import patch\n"
            '@patch("mod.a")\n'
            '@patch("mod.b")\n'
            '@patch("mod.c")\n'
            '@patch("mod.d")\n'
            "def test_something(mock_d, mock_c, mock_b, mock_a):\n"
            "    assert True\n"
        )
        rule = TEST005MockAbuseRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 1
        assert violations[0].id == "TEST005"

    def test_no_trigger_few_patches(self, test_context: RuleContext) -> None:
        code = '@patch("mod.a")\ndef test_something(mock_a):\n    assert True\n'
        rule = TEST005MockAbuseRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 0


class TestTEST006DuplicateTest:
    def test_detects_duplicate_bodies(self, test_context: RuleContext) -> None:
        code = (
            "def test_first():\n"
            "    result = func(1)\n"
            "    assert result == 2\n"
            "\n"
            "def test_second():\n"
            "    result = func(1)\n"
            "    assert result == 2\n"
        )
        rule = TEST006DuplicateTestRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 1
        assert violations[0].id == "TEST006"

    def test_no_trigger_different_bodies(self, test_context: RuleContext) -> None:
        code = (
            "def test_first():\n"
            "    assert func(1) == 2\n"
            "\n"
            "def test_second():\n"
            "    assert func(2) == 4\n"
        )
        rule = TEST006DuplicateTestRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 0


class TestTEST007MissingEdgeCases:
    def test_detects_no_edge_cases(self, test_context: RuleContext) -> None:
        code = (
            "def test_create_user():\n    assert True\n\n"
            "def test_update_user():\n    assert True\n\n"
            "def test_delete_user():\n    assert True\n"
        )
        rule = TEST007MissingEdgeCaseRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 1
        assert violations[0].id == "TEST007"

    def test_no_trigger_with_edge_case(self, test_context: RuleContext) -> None:
        code = (
            "def test_create_user():\n    assert True\n\n"
            "def test_update_user():\n    assert True\n\n"
            "def test_empty_name():\n    assert True\n"
        )
        rule = TEST007MissingEdgeCaseRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 0

    def test_no_trigger_few_tests(self, test_context: RuleContext) -> None:
        code = "def test_create_user():\n    assert True\n"
        rule = TEST007MissingEdgeCaseRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 0


class TestTEST008HardcodedData:
    def test_detects_magic_number(self, test_context: RuleContext) -> None:
        code = "def test_total():\n    assert calculate_total([1, 2]) == 42\n"
        rule = TEST008HardcodedDataRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 1
        assert violations[0].id == "TEST008"

    def test_no_trigger_with_comment(self, test_context: RuleContext) -> None:
        code = (
            "def test_total():\n"
            "    # 42 is the expected sum\n"
            "    assert calculate_total([1, 2]) == 42\n"
        )
        rule = TEST008HardcodedDataRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 0


class TestTEST009ExecutionOrder:
    def test_detects_global_statement(self, test_context: RuleContext) -> None:
        code = (
            "counter = 0\n"
            "def test_increment():\n"
            "    global counter\n"
            "    counter += 1\n"
            "    assert counter == 1\n"
        )
        rule = TEST009ExecutionOrderRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 1
        assert violations[0].id == "TEST009"

    def test_no_trigger_normal_test(self, test_context: RuleContext) -> None:
        code = "def test_something():\n    x = 1\n    assert x == 1\n"
        rule = TEST009ExecutionOrderRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 0


class TestTEST010BroadException:
    def test_detects_pytest_raises_exception(self, test_context: RuleContext) -> None:
        code = "def test_raises():\n    with pytest.raises(Exception):\n        func()\n"
        rule = TEST010BroadExceptionRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 1
        assert violations[0].id == "TEST010"

    def test_no_trigger_specific_exception(self, test_context: RuleContext) -> None:
        code = "def test_raises():\n    with pytest.raises(ValueError):\n        func()\n"
        rule = TEST010BroadExceptionRule()
        violations = rule.check(code, "python", test_context)
        assert len(violations) == 0


class TestNonTestCodeSkipped:
    """Verify all 10 rules return [] when is_test=False."""

    @pytest.mark.parametrize(
        "rule_cls",
        [
            TEST001EmptyTestRule,
            TEST002AssertTrueRule,
            TEST003NoAssertionsRule,
            TEST004NoCoverageRule,
            TEST005MockAbuseRule,
            TEST006DuplicateTestRule,
            TEST007MissingEdgeCaseRule,
            TEST008HardcodedDataRule,
            TEST009ExecutionOrderRule,
            TEST010BroadExceptionRule,
        ],
    )
    def test_skips_non_test_code(self, rule_cls: type, non_test_context: RuleContext) -> None:
        code = "def test_something():\n    pass\n"
        rule = rule_cls()
        violations = rule.check(code, "python", non_test_context)
        assert violations == []


class TestRuleTierAssignments:
    """Verify each rule's tier property."""

    @pytest.mark.parametrize(
        ("rule_cls", "expected_tier"),
        [
            (TEST001EmptyTestRule, RuleTier.ESSENTIAL),
            (TEST002AssertTrueRule, RuleTier.ESSENTIAL),
            (TEST003NoAssertionsRule, RuleTier.ESSENTIAL),
            (TEST004NoCoverageRule, RuleTier.FULL),
            (TEST005MockAbuseRule, RuleTier.ESSENTIAL),
            (TEST006DuplicateTestRule, RuleTier.FULL),
            (TEST007MissingEdgeCaseRule, RuleTier.ESSENTIAL),
            (TEST008HardcodedDataRule, RuleTier.ESSENTIAL),
            (TEST009ExecutionOrderRule, RuleTier.FULL),
            (TEST010BroadExceptionRule, RuleTier.ESSENTIAL),
        ],
    )
    def test_tier(self, rule_cls: type, expected_tier: RuleTier) -> None:
        assert rule_cls().tier == expected_tier


class TestIntegrationViaChecker:
    """Test via AIQualityChecker.check() with is_test and max_tier."""

    def test_essential_tier_catches_empty_test(self) -> None:
        checker = AIQualityChecker()
        code = "def test_something():\n    pass\n"
        violations = checker.check(code, "python", is_test=True, max_tier=RuleTier.ESSENTIAL)
        test_violations = [v for v in violations if v.id.startswith("TEST")]
        assert any(v.id == "TEST001" for v in test_violations)

    def test_essential_tier_skips_full_rules(self) -> None:
        checker = AIQualityChecker()
        # TEST009 (FULL tier) should not fire at ESSENTIAL tier
        code = "counter = 0\ndef test_increment():\n    global counter\n    counter += 1\n"
        violations = checker.check(code, "python", is_test=True, max_tier=RuleTier.ESSENTIAL)
        assert not any(v.id == "TEST009" for v in violations)

    def test_full_tier_includes_all(self) -> None:
        checker = AIQualityChecker()
        code = "counter = 0\ndef test_increment():\n    global counter\n    counter += 1\n"
        violations = checker.check(code, "python", is_test=True, max_tier=RuleTier.FULL)
        assert any(v.id == "TEST009" for v in violations)
