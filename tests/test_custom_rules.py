"""Tests for custom rule definitions loading."""

from pathlib import Path
from textwrap import dedent

from mirdan.config import MirdanConfig, QualityConfig
from mirdan.core.code_validator import CodeValidator
from mirdan.core.quality_standards import QualityStandards


class TestCustomRuleLoading:
    """Tests for loading custom rules from YAML files."""

    def _create_rules_dir(self, tmp_path: Path, rules_yaml: str) -> Path:
        """Create a temp rules directory with a YAML file."""
        rules_dir = tmp_path / ".mirdan" / "rules"
        rules_dir.mkdir(parents=True)
        rules_file = rules_dir / "custom.yaml"
        rules_file.write_text(rules_yaml)
        return rules_dir

    def test_load_custom_rule(self, tmp_path: Path) -> None:
        """Should load and apply a custom rule."""
        rules_yaml = dedent("""\
            rules:
              - id: CUSTOM001
                rule: no-print
                pattern: "\\\\bprint\\\\s*\\\\("
                severity: warning
                message: "print() detected - use logging instead"
                suggestion: "Use logging.info() or logging.debug()"
        """)
        rules_dir = self._create_rules_dir(tmp_path, rules_yaml)

        config = QualityConfig(custom_rules_dir=str(rules_dir))
        standards = QualityStandards()
        validator = CodeValidator(standards, config=config)

        code = 'print("hello")'
        result = validator.validate(code, language="python")

        custom = [v for v in result.violations if v.id == "CUSTOM001"]
        assert len(custom) == 1
        assert "print()" in custom[0].message

    def test_custom_rule_with_fix_template(self, tmp_path: Path) -> None:
        """Should load custom rule with fix template."""
        rules_yaml = dedent("""\
            rules:
              - id: CUSTOM002
                rule: no-todo-comments
                pattern: "#\\\\s*TODO"
                severity: info
                message: "TODO comment found"
                suggestion: "Create an issue instead"
                fix_template: "# ISSUE: "
                fix_description: "Replace TODO with issue reference"
        """)
        rules_dir = self._create_rules_dir(tmp_path, rules_yaml)

        config = QualityConfig(custom_rules_dir=str(rules_dir))
        standards = QualityStandards()
        validator = CodeValidator(standards, config=config)

        code = "# TODO fix this later"
        result = validator.validate(code, language="python")

        custom = [v for v in result.violations if v.id == "CUSTOM002"]
        assert len(custom) == 1
        assert custom[0].fix_code == "# ISSUE: "

    def test_custom_rule_override_severity(self, tmp_path: Path) -> None:
        """Should override built-in rule severity."""
        rules_yaml = dedent("""\
            rules:
              - id: PY003
                severity: warning
        """)
        rules_dir = self._create_rules_dir(tmp_path, rules_yaml)

        config = QualityConfig(custom_rules_dir=str(rules_dir))
        standards = QualityStandards()
        validator = CodeValidator(standards, config=config)

        code = """
def process():
    try:
        do_thing()
    except:
        pass
"""
        result = validator.validate(code, language="python")

        py003 = [v for v in result.violations if v.id == "PY003"]
        assert len(py003) >= 1
        assert py003[0].severity == "warning"  # Overridden from error

    def test_nonexistent_rules_dir_no_error(self) -> None:
        """Should not error when rules directory doesn't exist."""
        config = QualityConfig(custom_rules_dir="/nonexistent/path")
        standards = QualityStandards()
        # Should not raise
        validator = CodeValidator(standards, config=config)
        result = validator.validate("x = 1", language="python")
        assert result.passed is True

    def test_invalid_yaml_skipped(self, tmp_path: Path) -> None:
        """Should skip YAML files with parse errors."""
        rules_dir = tmp_path / ".mirdan" / "rules"
        rules_dir.mkdir(parents=True)
        bad_file = rules_dir / "bad.yaml"
        bad_file.write_text("invalid: yaml: [unclosed")

        config = QualityConfig(custom_rules_dir=str(rules_dir))
        standards = QualityStandards()
        # Should not raise
        validator = CodeValidator(standards, config=config)
        result = validator.validate("x = 1", language="python")
        assert result.passed is True

    def test_invalid_regex_skipped(self, tmp_path: Path) -> None:
        """Should skip rules with invalid regex patterns."""
        rules_yaml = dedent("""\
            rules:
              - id: BAD001
                rule: bad-regex
                pattern: "[unclosed"
                severity: warning
                message: "Bad regex"
                suggestion: "Fix it"
        """)
        rules_dir = self._create_rules_dir(tmp_path, rules_yaml)

        config = QualityConfig(custom_rules_dir=str(rules_dir))
        standards = QualityStandards()
        # Should not raise
        validator = CodeValidator(standards, config=config)
        result = validator.validate("x = 1", language="python")
        assert result.passed is True

    def test_custom_rules_checked_in_standards(self, tmp_path: Path) -> None:
        """Custom rules should appear in standards_checked."""
        rules_yaml = dedent("""\
            rules:
              - id: CUSTOM003
                rule: no-magic-numbers
                pattern: "=\\\\s*\\\\d{3,}"
                severity: info
                message: "Magic number detected"
                suggestion: "Use a named constant"
        """)
        rules_dir = self._create_rules_dir(tmp_path, rules_yaml)

        config = QualityConfig(custom_rules_dir=str(rules_dir))
        standards = QualityStandards()
        validator = CodeValidator(standards, config=config)

        code = "timeout = 3600"
        result = validator.validate(code, language="python")
        assert "custom" in result.standards_checked

    def test_empty_rules_file_handled(self, tmp_path: Path) -> None:
        """Should handle empty rules files."""
        rules_dir = tmp_path / ".mirdan" / "rules"
        rules_dir.mkdir(parents=True)
        empty_file = rules_dir / "empty.yaml"
        empty_file.write_text("")

        config = QualityConfig(custom_rules_dir=str(rules_dir))
        standards = QualityStandards()
        validator = CodeValidator(standards, config=config)
        result = validator.validate("x = 1", language="python")
        assert result.passed is True

    def test_multiple_custom_rules_files(self, tmp_path: Path) -> None:
        """Should load rules from multiple YAML files."""
        rules_dir = tmp_path / ".mirdan" / "rules"
        rules_dir.mkdir(parents=True)

        file1 = rules_dir / "security.yaml"
        file1.write_text(
            dedent("""\
            rules:
              - id: SEC_CUSTOM1
                rule: no-debug-mode
                pattern: "DEBUG\\\\s*=\\\\s*True"
                severity: error
                message: "Debug mode enabled"
                suggestion: "Disable debug mode in production"
        """)
        )

        file2 = rules_dir / "style.yaml"
        file2.write_text(
            dedent("""\
            rules:
              - id: STYLE_CUSTOM1
                rule: no-print
                pattern: "\\\\bprint\\\\s*\\\\("
                severity: warning
                message: "print() detected"
                suggestion: "Use logging"
        """)
        )

        config = QualityConfig(custom_rules_dir=str(rules_dir))
        standards = QualityStandards()
        validator = CodeValidator(standards, config=config)

        code = 'DEBUG = True\nprint("test")'
        result = validator.validate(code, language="python")

        rule_ids = {v.id for v in result.violations}
        assert "SEC_CUSTOM1" in rule_ids
        assert "STYLE_CUSTOM1" in rule_ids


class TestDefaultCustomRulesDir:
    """Tests for default custom_rules_dir config."""

    def test_default_config_has_custom_rules_dir(self) -> None:
        """Default config should set custom_rules_dir."""
        config = MirdanConfig()
        assert config.quality.custom_rules_dir == ".mirdan/rules"
