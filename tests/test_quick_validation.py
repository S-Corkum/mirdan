"""Tests for the validate_quick fast validation path.

Tests cover:
- CodeValidator.validate_quick() — security-only checks
- validate_quick MCP tool handler
- Edge cases: empty code, unknown language
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import mirdan.server as server_mod
from mirdan.config import MirdanConfig
from mirdan.core.code_validator import CodeValidator
from mirdan.core.quality_standards import QualityStandards

# Extract raw async function from FastMCP FunctionTool wrapper
_validate_quick = server_mod.validate_quick.fn


@pytest.fixture(autouse=True)
def _reset_components() -> Iterator[None]:
    """Reset the server singleton before each test."""
    server_mod._components = None
    yield
    server_mod._components = None


@pytest.fixture()
def validator() -> CodeValidator:
    """Create a CodeValidator instance for direct testing."""
    config = MirdanConfig()
    standards = QualityStandards(config=config.quality)
    return CodeValidator(standards, config=config.quality, thresholds=config.thresholds)


# ---------------------------------------------------------------------------
# CodeValidator.validate_quick() unit tests
# ---------------------------------------------------------------------------


class TestValidateQuickMethod:
    """Tests for CodeValidator.validate_quick()."""

    def test_returns_only_security_violations(self, validator: CodeValidator) -> None:
        """validate_quick should only return SEC* violations, not style/arch."""
        # eval() triggers PY001 (style), SQL concat triggers SEC004 (security)
        code = 'query = "SELECT * FROM users WHERE id=" + user_id\nresult = eval(user_input)\n'
        result = validator.validate_quick(code, language="python")
        # Should have security violations (SEC004 for SQL concat)
        assert any(v.id.startswith("SEC") for v in result.violations)
        # Should NOT have style violations (PY001 for eval)
        assert not any(v.id.startswith("PY") for v in result.violations)
        # Standards checked should only be security
        assert result.standards_checked == ["security"]

    def test_skips_architecture_checks(self, validator: CodeValidator) -> None:
        """validate_quick should not trigger architecture violations."""
        # Create a very long function that would trigger ARCH001
        lines = ["def long_function() -> int:"] + ["    x = 1"] * 40 + ["    return x"]
        code = "\n".join(lines)
        result = validator.validate_quick(code, language="python")
        arch_violations = [v for v in result.violations if v.id.startswith("ARCH")]
        assert len(arch_violations) == 0

    def test_empty_code(self, validator: CodeValidator) -> None:
        """validate_quick should handle empty code gracefully."""
        result = validator.validate_quick("", language="python")
        assert result.passed is True
        assert result.score == 1.0
        assert result.standards_checked == ["security"]

    def test_whitespace_only_code(self, validator: CodeValidator) -> None:
        """validate_quick should handle whitespace-only code."""
        result = validator.validate_quick("   \n\n  ", language="python")
        assert result.passed is True

    def test_clean_code_passes(self, validator: CodeValidator) -> None:
        """Clean code with no security issues should pass."""
        code = 'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b\n'
        result = validator.validate_quick(code, language="python")
        assert result.passed is True
        assert result.score == 1.0

    def test_auto_language_detection(self, validator: CodeValidator) -> None:
        """validate_quick should auto-detect language."""
        code = "const x: number = eval(input);\n"
        result = validator.validate_quick(code, language="auto")
        # Should detect as typescript/javascript and still find security issues
        assert result.language_detected in ("typescript", "javascript")

    def test_unknown_language(self, validator: CodeValidator) -> None:
        """validate_quick should handle unknown languages."""
        code = "some code here"
        result = validator.validate_quick(code, language="brainfuck")
        # Should still run security checks
        assert result.standards_checked == ["security"]
        assert "not fully supported" in result.limitations[0]

    def test_sql_injection_detected(self, validator: CodeValidator) -> None:
        """validate_quick should catch SQL injection patterns."""
        code = 'query = "SELECT * FROM users WHERE id=" + user_id\n'
        result = validator.validate_quick(code, language="python")
        sec_violations = [v for v in result.violations if v.id.startswith("SEC")]
        assert len(sec_violations) > 0


# ---------------------------------------------------------------------------
# validate_quick MCP tool tests
# ---------------------------------------------------------------------------


class TestValidateQuickTool:
    """Tests for the validate_quick MCP tool handler."""

    async def test_clean_code_passes(self) -> None:
        """Clean code should pass quick validation."""
        server_mod._get_components()
        code = 'def add(a: int, b: int) -> int:\n    """Add."""\n    return a + b\n'
        result = await _validate_quick(code, language="python")
        assert result["passed"] is True
        assert result["score"] > 0.8

    async def test_security_violation_detected(self) -> None:
        """Code with security issues should fail."""
        server_mod._get_components()
        code = 'query = "SELECT * FROM users WHERE id=" + user_id\n'
        result = await _validate_quick(code, language="python")
        sec_violations = [
            v for v in result.get("violations", []) if v.get("id", "").startswith("SEC")
        ]
        assert len(sec_violations) > 0

    async def test_no_style_violations_returned(self) -> None:
        """Quick validation should not include style violations."""
        server_mod._get_components()
        # Code that would trigger style rules but not security rules
        code = "from typing import List\nx: List[int] = []\n"
        result = await _validate_quick(code, language="python")
        style_violations = [
            v for v in result.get("violations", []) if v.get("category", "") == "style"
        ]
        assert len(style_violations) == 0

    async def test_rejects_oversized_code(self) -> None:
        """Should return error for code exceeding max length."""
        oversized = "x" * (server_mod._MAX_CODE_LENGTH + 1)
        result = await _validate_quick(oversized)
        assert "error" in result

    async def test_standards_checked_is_security_only(self) -> None:
        """standards_checked should only contain security."""
        server_mod._get_components()
        code = "x = 1\n"
        result = await _validate_quick(code, language="python")
        assert result["standards_checked"] == ["security"]
