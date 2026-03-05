"""Tests for auto-fix generation in code validator."""

from mirdan.config import MirdanConfig
from mirdan.core.auto_fixer import TEMPLATE_FIXES, AutoFixer
from mirdan.core.code_validator import AUTO_FIX_TEMPLATES, CodeValidator
from mirdan.core.quality_standards import QualityStandards


class TestAutoFixTemplates:
    """Tests for the AUTO_FIX_TEMPLATES dictionary."""

    def test_required_rules_have_fixes(self) -> None:
        """All 8 planned rules should have auto-fix templates."""
        expected_rules = ["PY003", "PY004", "PY005", "PY011", "JS001", "TS004", "RS001", "SEC007"]
        for rule_id in expected_rules:
            assert rule_id in AUTO_FIX_TEMPLATES, f"Missing auto-fix for {rule_id}"

    def test_fix_templates_have_descriptions(self) -> None:
        """Each fix template should have a non-empty description."""
        for rule_id, (fix_code, fix_desc) in AUTO_FIX_TEMPLATES.items():
            assert fix_code, f"Empty fix_code for {rule_id}"
            assert fix_desc, f"Empty fix_description for {rule_id}"


class TestAutoFixInViolations:
    """Tests that violations include auto-fix data."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        config = MirdanConfig()
        self.quality_standards = QualityStandards(config=config.quality)
        self.code_validator = CodeValidator(
            self.quality_standards, config=config.quality, thresholds=config.thresholds
        )

    def test_bare_except_has_fix(self) -> None:
        """PY003 bare except should include fix_code."""
        code = """
def process():
    try:
        do_thing()
    except:
        pass
"""
        result = self.code_validator.validate(code, language="python")
        py003 = [v for v in result.violations if v.id == "PY003"]
        assert len(py003) >= 1
        assert py003[0].fix_code == "except Exception:"
        assert py003[0].fix_description != ""

    def test_var_has_fix(self) -> None:
        """JS001 var should include fix_code."""
        code = "var counter = 0;"
        result = self.code_validator.validate(code, language="javascript")
        js001 = [v for v in result.violations if v.id == "JS001"]
        assert len(js001) >= 1
        assert js001[0].fix_code == "const"

    def test_as_any_has_fix(self) -> None:
        """TS004 as any should include fix_code."""
        code = "const x = value as any;"
        result = self.code_validator.validate(code, language="typescript")
        ts004 = [v for v in result.violations if v.id == "TS004"]
        assert len(ts004) >= 1
        assert ts004[0].fix_code == "as unknown"

    def test_unwrap_has_fix(self) -> None:
        """RS001 unwrap should include fix_code."""
        code = "let value = result.unwrap();"
        result = self.code_validator.validate(code, language="rust")
        rs001 = [v for v in result.violations if v.id == "RS001"]
        assert len(rs001) >= 1
        assert ".expect(" in rs001[0].fix_code

    def test_verify_false_has_fix(self) -> None:
        """SEC007 verify=False should include fix_code."""
        code = "requests.get(url, verify=False)"
        result = self.code_validator.validate(code, language="python")
        sec007 = [v for v in result.violations if v.id == "SEC007"]
        assert len(sec007) >= 1
        assert sec007[0].fix_code == "verify=True"

    def test_rule_without_fix_has_empty_fix_code(self) -> None:
        """Rules without auto-fix should have empty fix_code."""
        code = "result = eval(user_input)"
        result = self.code_validator.validate(code, language="python")
        py001 = [v for v in result.violations if v.id == "PY001"]
        assert len(py001) >= 1
        assert py001[0].fix_code == ""

    def test_fix_code_in_to_dict(self) -> None:
        """Violation.to_dict() should include fix_code when present."""
        code = """
def process():
    try:
        do_thing()
    except:
        pass
"""
        result = self.code_validator.validate(code, language="python")
        py003 = [v for v in result.violations if v.id == "PY003"]
        assert len(py003) >= 1
        d = py003[0].to_dict()
        assert "fix_code" in d
        assert d["fix_code"] == "except Exception:"

    def test_no_fix_code_not_in_to_dict(self) -> None:
        """Violation.to_dict() should NOT include fix_code when empty."""
        code = "result = eval(user_input)"
        result = self.code_validator.validate(code, language="python")
        py001 = [v for v in result.violations if v.id == "PY001"]
        assert len(py001) >= 1
        d = py001[0].to_dict()
        assert "fix_code" not in d

    def test_os_path_has_fix(self) -> None:
        """PY011 os.path should include fix_code."""
        code = "result = os.path.join(base, name)"
        result = self.code_validator.validate(code, language="python")
        py011 = [v for v in result.violations if v.id == "PY011"]
        assert len(py011) >= 1
        assert "pathlib" in py011[0].fix_code


class TestSEC014AutoFix:
    """Tests for SEC014 auto-fix behavior."""

    def test_sec014_fix_available(self) -> None:
        """SEC014 should have a TEMPLATE_FIXES entry with confidence=0.5."""
        assert "SEC014" in TEMPLATE_FIXES
        _fix_code, _fix_desc, confidence = TEMPLATE_FIXES["SEC014"]
        assert confidence == 0.5

    def test_sec014_not_in_quick_fix_rules(self) -> None:
        """SEC014 should NOT be in AutoFixer._QUICK_FIX_RULES (confidence < 0.8)."""
        assert "SEC014" not in AutoFixer._QUICK_FIX_RULES
