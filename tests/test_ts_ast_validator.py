"""Tests for the TypeScript/JavaScript architecture validator."""

import pytest

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
        """When tree-sitter is not available, should include regex limitation."""
        from mirdan.core.ts_ast_validator import _HAS_TREE_SITTER

        code = "const x = 1;"
        _, limitations = validate_ts_architecture(code)
        if _HAS_TREE_SITTER:
            # Tree-sitter path has no limitations
            assert len(limitations) == 0
        else:
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


# ---------------------------------------------------------------------------
# Tree-sitter-specific tests (skipped when tree-sitter not installed)
# ---------------------------------------------------------------------------

try:
    import tree_sitter_typescript  # noqa: F401

    HAS_TS = True
except ImportError:
    HAS_TS = False


@pytest.mark.skipif(not HAS_TS, reason="tree-sitter not installed")
class TestTreeSitterValidation:
    """Tests that only run when tree-sitter is installed."""

    def test_function_length_ast(self) -> None:
        """Tree-sitter should accurately measure function length."""
        code = (
            "function longFunc(x: number): string {\n"
            + "    console.log(x);\n" * 35
            + "    return x.toString();\n"
            + "}\n"
        )
        violations, _ = validate_ts_architecture(code, "typescript")
        tsarch001 = [v for v in violations if v.id == "TSARCH001"]
        assert len(tsarch001) == 1
        assert "longFunc" in tsarch001[0].message
        # AST gives exact count: 38 lines
        assert "38 lines" in tsarch001[0].message

    def test_nesting_depth_ast(self) -> None:
        """Tree-sitter should accurately measure nesting depth."""
        code = """function nested(x: number): void {
    if (x > 0) {
        for (let i = 0; i < x; i++) {
            while (true) {
                if (i > 5) {
                    console.log(i);
                }
            }
        }
    }
}"""
        config = TSValidationConfig(max_nesting_depth=3)
        violations, _ = validate_ts_architecture(code, "typescript", config)
        tsarch003 = [v for v in violations if v.id == "TSARCH003"]
        assert len(tsarch003) == 1
        assert "nested" in tsarch003[0].message

    def test_missing_return_type_ast(self) -> None:
        """Tree-sitter should detect missing return types."""
        code = "function noReturn(x: number) {\n    console.log(x);\n}"
        violations, _ = validate_ts_architecture(code, "typescript")
        tsarch004 = [v for v in violations if v.id == "TSARCH004"]
        assert len(tsarch004) == 1
        assert "noReturn" in tsarch004[0].message

    def test_function_with_return_type_ast(self) -> None:
        """Tree-sitter should not flag functions with return types."""
        code = "function hasReturn(x: number): string {\n    return x.toString();\n}"
        violations, _ = validate_ts_architecture(code, "typescript")
        tsarch004 = [v for v in violations if v.id == "TSARCH004"]
        assert len(tsarch004) == 0

    def test_no_limitation_message(self) -> None:
        """When tree-sitter is available, no 'install mirdan[ast]' limitation."""
        code = "const x = 1;"
        _, limitations = validate_ts_architecture(code)
        assert len(limitations) == 0

    def test_arrow_function_detected(self) -> None:
        """Tree-sitter should detect arrow functions."""
        code = "const myArrow = (x: number): void => {\n" + "    console.log(x);\n" * 35 + "};\n"
        violations, _ = validate_ts_architecture(code, "typescript")
        tsarch001 = [v for v in violations if v.id == "TSARCH001"]
        assert len(tsarch001) == 1
        assert "myArrow" in tsarch001[0].message

    def test_javascript_no_return_type_check(self) -> None:
        """TSARCH004 should not fire for JavaScript."""
        code = "function noReturn(x) {\n    console.log(x);\n}"
        violations, _ = validate_ts_architecture(code, "javascript")
        tsarch004 = [v for v in violations if v.id == "TSARCH004"]
        assert len(tsarch004) == 0

    def test_method_definition_detected(self) -> None:
        """Tree-sitter should detect class method definitions."""
        code = (
            "class MyClass {\n"
            + "    longMethod(): void {\n"
            + "        console.log('hello');\n" * 35
            + "    }\n"
            + "}\n"
        )
        violations, _ = validate_ts_architecture(code, "typescript")
        tsarch001 = [v for v in violations if v.id == "TSARCH001"]
        assert len(tsarch001) == 1
        assert "longMethod" in tsarch001[0].message
