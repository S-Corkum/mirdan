"""Tests for the TypeScript/JavaScript architecture validator."""

from mirdan.core.ts_ast_validator import (
    TSValidationConfig,
    validate_ts_architecture,
)


class TestFunctionLength:
    """TSARCH001: function-too-long checks."""

    def test_short_function_passes(self) -> None:
        """Functions within limit should not trigger violations."""
        code = "function greet(name) {\n" + "  console.log(name);\n" * 5 + "}\n"
        violations, _ = validate_ts_architecture(code)
        assert not any(v.id == "TSARCH001" for v in violations)

    def test_long_function_detected(self) -> None:
        """Functions exceeding max_function_length should trigger TSARCH001."""
        lines = ["function processData(input) {"]
        lines.extend(["  const x = 1;"] * 35)
        lines.append("}")
        code = "\n".join(lines)
        config = TSValidationConfig(max_function_length=30)
        violations, _ = validate_ts_architecture(code, config=config)
        assert any(v.id == "TSARCH001" for v in violations)
        match = next(v for v in violations if v.id == "TSARCH001")
        assert "processData" in match.message
        assert match.rule == "function-too-long"

    def test_long_arrow_function_detected(self) -> None:
        """Arrow functions exceeding limit should trigger TSARCH001."""
        lines = ["export const handler = (req) => {"]
        lines.extend(["  const x = 1;"] * 35)
        lines.append("}")
        code = "\n".join(lines)
        config = TSValidationConfig(max_function_length=30)
        violations, _ = validate_ts_architecture(code, config=config)
        assert any(v.id == "TSARCH001" for v in violations)
        match = next(v for v in violations if v.id == "TSARCH001")
        assert "handler" in match.message

    def test_custom_max_length(self) -> None:
        """Custom max_function_length should be respected."""
        lines = ["function short() {"]
        lines.extend(["  const x = 1;"] * 12)
        lines.append("}")
        code = "\n".join(lines)
        # Should pass with default (30)
        violations_default, _ = validate_ts_architecture(code)
        assert not any(v.id == "TSARCH001" for v in violations_default)
        # Should fail with strict limit
        config = TSValidationConfig(max_function_length=10)
        violations_strict, _ = validate_ts_architecture(code, config=config)
        assert any(v.id == "TSARCH001" for v in violations_strict)


class TestFileLength:
    """TSARCH002: file-too-long checks."""

    def test_short_file_passes(self) -> None:
        """Files within limit should not trigger violations."""
        code = "\n".join(f"const x{i} = {i};" for i in range(50))
        violations, _ = validate_ts_architecture(code)
        assert not any(v.id == "TSARCH002" for v in violations)

    def test_long_file_detected(self) -> None:
        """Files exceeding max_file_length should trigger TSARCH002."""
        code = "\n".join(f"const x{i} = {i};" for i in range(350))
        config = TSValidationConfig(max_file_length=300)
        violations, _ = validate_ts_architecture(code, config=config)
        assert any(v.id == "TSARCH002" for v in violations)
        match = next(v for v in violations if v.id == "TSARCH002")
        assert match.rule == "file-too-long"
        assert "300" in match.message

    def test_empty_lines_excluded(self) -> None:
        """Empty and comment lines should not count toward file length."""
        real_lines = [f"const x{i} = {i};" for i in range(50)]
        padding = ["", "  ", "// comment"] * 100
        code = "\n".join(real_lines + padding)
        config = TSValidationConfig(max_file_length=100)
        violations, _ = validate_ts_architecture(code, config=config)
        assert not any(v.id == "TSARCH002" for v in violations)


class TestNestingDepth:
    """TSARCH003: excessive-nesting checks."""

    def test_shallow_nesting_passes(self) -> None:
        """Shallow nesting should not trigger violations."""
        code = """\
function simple() {
  if (true) {
    console.log("ok");
  }
}
"""
        violations, _ = validate_ts_architecture(code)
        assert not any(v.id == "TSARCH003" for v in violations)

    def test_deep_nesting_detected(self) -> None:
        """Deeply nested code should trigger TSARCH003."""
        code = """\
function deepNest() {
  if (a) {
    if (b) {
      if (c) {
        if (d) {
          if (e) {
            console.log("too deep");
          }
        }
      }
    }
  }
}
"""
        config = TSValidationConfig(max_nesting_depth=4)
        violations, _ = validate_ts_architecture(code, config=config)
        assert any(v.id == "TSARCH003" for v in violations)
        match = next(v for v in violations if v.id == "TSARCH003")
        assert "deepNest" in match.message


class TestMissingReturnType:
    """TSARCH004: missing-return-type checks (TypeScript only)."""

    def test_function_with_return_type_passes(self) -> None:
        """Functions with return types should not trigger violations."""
        code = "function greet(name: string): string {\n  return name;\n}\n"
        violations, _ = validate_ts_architecture(code, language="typescript")
        assert not any(v.id == "TSARCH004" for v in violations)

    def test_function_without_return_type_detected(self) -> None:
        """Functions missing return types should trigger TSARCH004."""
        code = "function greet(name: string) {\n  return name;\n}\n"
        violations, _ = validate_ts_architecture(code, language="typescript")
        assert any(v.id == "TSARCH004" for v in violations)
        match = next(v for v in violations if v.id == "TSARCH004")
        assert "greet" in match.message

    def test_skipped_for_javascript(self) -> None:
        """TSARCH004 should not fire for JavaScript."""
        code = "function greet(name) {\n  return name;\n}\n"
        violations, _ = validate_ts_architecture(code, language="javascript")
        assert not any(v.id == "TSARCH004" for v in violations)

    def test_underscore_functions_skipped(self) -> None:
        """Private/underscore functions should not trigger TSARCH004."""
        code = "function _helper(x) {\n  return x;\n}\n"
        violations, _ = validate_ts_architecture(code, language="typescript")
        assert not any(v.id == "TSARCH004" for v in violations)


class TestLimitations:
    """Tests for limitation messages."""

    def test_includes_limitation_message(self) -> None:
        """Should include regex heuristics limitation."""
        code = "const x = 1;"
        _, limitations = validate_ts_architecture(code)
        assert len(limitations) >= 1
        assert any("regex" in lim.lower() for lim in limitations)

    def test_returns_violations_and_limitations_tuple(self) -> None:
        """Should return a (violations, limitations) tuple."""
        code = "const x = 1;"
        result = validate_ts_architecture(code)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestConfigDefaults:
    """Tests for TSValidationConfig defaults."""

    def test_default_values(self) -> None:
        """Default config should have expected values."""
        config = TSValidationConfig()
        assert config.max_function_length == 30
        assert config.max_file_length == 300
        assert config.max_nesting_depth == 4
