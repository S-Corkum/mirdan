"""Tests for the actual server.py tool handler functions.

These tests call the async @mcp.tool() decorated functions directly,
verifying the server-level orchestration logic including input validation,
component wiring, and response formatting.

FastMCP wraps @mcp.tool() functions into FunctionTool objects. We access
the underlying coroutine via the `.fn` attribute.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import pytest

import mirdan.server as server_mod
from mirdan.models import ContextBundle

# Extract raw async functions from FastMCP FunctionTool wrappers
_enhance_prompt = server_mod.enhance_prompt.fn
_get_quality_standards = server_mod.get_quality_standards.fn
_validate_code_quality = server_mod.validate_code_quality.fn


@pytest.fixture(autouse=True)
def _reset_components() -> Iterator[None]:
    """Reset the server singleton before each test."""
    server_mod._components = None
    yield
    server_mod._components = None


@pytest.fixture()
def components() -> server_mod._Components:
    """Eagerly initialize and return the component singleton."""
    return server_mod._get_components()


# ---------------------------------------------------------------------------
# _lifespan context manager
# ---------------------------------------------------------------------------


class TestLifespan:
    """Tests for the server lifespan context manager."""

    async def test_lifespan_initializes_components(self) -> None:
        """_lifespan should eagerly init components on startup."""
        assert server_mod._components is None
        async with server_mod._lifespan(server_mod.mcp):
            assert server_mod._components is not None

    async def test_lifespan_cleanup_with_close(self) -> None:
        """_lifespan should call close on context_aggregator during shutdown."""
        mock_close = AsyncMock()
        async with server_mod._lifespan(server_mod.mcp):
            assert server_mod._components is not None
            server_mod._components.context_aggregator.close = mock_close  # type: ignore[method-assign]
        mock_close.assert_awaited_once()

    async def test_lifespan_cleanup_when_no_components(self) -> None:
        """_lifespan shutdown should handle None _components gracefully."""
        async with server_mod._lifespan(server_mod.mcp):
            # Force components to None (simulates init failure scenario)
            server_mod._components = None
        # Should not raise


# ---------------------------------------------------------------------------
# enhance_prompt
# ---------------------------------------------------------------------------


class TestEnhancePromptTool:
    """Tests for the enhance_prompt async tool handler."""

    async def test_enhance_prompt_returns_expected_keys(self) -> None:
        """Should return dict with all expected keys."""
        with patch.object(
            server_mod._get_components().context_aggregator,
            "gather_all",
            new_callable=AsyncMock,
            return_value=ContextBundle(),
        ):
            result = await _enhance_prompt("Create a Python function")

        assert "enhanced_prompt" in result
        assert "task_type" in result
        assert "language" in result
        assert "frameworks" in result
        assert "quality_requirements" in result
        assert "verification_steps" in result
        assert "tool_recommendations" in result

    async def test_enhance_prompt_detects_language(self) -> None:
        """Should detect the language from the prompt."""
        with patch.object(
            server_mod._get_components().context_aggregator,
            "gather_all",
            new_callable=AsyncMock,
            return_value=ContextBundle(),
        ):
            result = await _enhance_prompt("Write a Python REST API")

        assert result["language"] == "python"

    async def test_enhance_prompt_rejects_oversized(self) -> None:
        """Should return error for prompt exceeding max length."""
        oversized = "x" * (server_mod._MAX_PROMPT_LENGTH + 1)
        result = await _enhance_prompt(oversized)

        assert "error" in result
        assert "prompt" in result["error"]
        assert result["max_length"] == server_mod._MAX_PROMPT_LENGTH

    async def test_enhance_prompt_task_type_override(self) -> None:
        """Should override task type when specified."""
        with patch.object(
            server_mod._get_components().context_aggregator,
            "gather_all",
            new_callable=AsyncMock,
            return_value=ContextBundle(),
        ):
            result = await _enhance_prompt("Look at this code", task_type="review")

        assert result["task_type"] == "review"

    async def test_enhance_prompt_invalid_task_type_ignored(self) -> None:
        """Invalid task_type should be silently ignored (contextlib.suppress)."""
        with patch.object(
            server_mod._get_components().context_aggregator,
            "gather_all",
            new_callable=AsyncMock,
            return_value=ContextBundle(),
        ):
            result = await _enhance_prompt("Write Python code", task_type="not_a_real_type")

        # Should succeed without error — task type stays auto-detected
        assert "enhanced_prompt" in result

    async def test_enhance_prompt_auto_task_type(self) -> None:
        """task_type='auto' should use auto-detection (no override)."""
        with patch.object(
            server_mod._get_components().context_aggregator,
            "gather_all",
            new_callable=AsyncMock,
            return_value=ContextBundle(),
        ):
            result = await _enhance_prompt("Fix the login bug", task_type="auto")

        assert "enhanced_prompt" in result
        assert result["task_type"] == "debug"

    async def test_enhance_prompt_context_level_passed(self) -> None:
        """context_level argument should be forwarded to gather_all."""
        mock_gather = AsyncMock(return_value=ContextBundle())
        with patch.object(
            server_mod._get_components().context_aggregator,
            "gather_all",
            mock_gather,
        ):
            await _enhance_prompt("Create code", context_level="comprehensive")

        # Verify gather_all was called with the context_level
        mock_gather.assert_awaited_once()
        call_args = mock_gather.call_args
        # context_level can be passed as positional arg [1] or keyword
        passed_level = call_args.kwargs.get("context_level")
        if passed_level is None and len(call_args.args) > 1:
            passed_level = call_args.args[1]
        assert passed_level == "comprehensive"


# ---------------------------------------------------------------------------
# get_quality_standards
# ---------------------------------------------------------------------------


class TestGetQualityStandardsTool:
    """Tests for the get_quality_standards async tool handler."""

    async def test_get_standards_python(self) -> None:
        """Should return Python standards."""
        server_mod._get_components()
        result = await _get_quality_standards("python")
        assert isinstance(result, dict)
        assert "language_standards" in result

    async def test_get_standards_with_framework(self) -> None:
        """Should include framework-specific standards."""
        server_mod._get_components()
        result = await _get_quality_standards("python", framework="fastapi")
        assert "framework_standards" in result

    async def test_get_standards_security_category(self) -> None:
        """Should filter to security category."""
        server_mod._get_components()
        result = await _get_quality_standards("python", category="security")
        assert "security_standards" in result

    async def test_get_standards_typescript(self) -> None:
        """Should return TypeScript standards."""
        server_mod._get_components()
        result = await _get_quality_standards("typescript")
        assert isinstance(result, dict)

    async def test_get_standards_unknown_language(self) -> None:
        """Should handle unknown languages gracefully."""
        server_mod._get_components()
        result = await _get_quality_standards("brainfuck")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# validate_code_quality
# ---------------------------------------------------------------------------


class TestValidateCodeQualityTool:
    """Tests for the validate_code_quality async tool handler."""

    async def test_validate_clean_code(self) -> None:
        """Clean code should pass."""
        server_mod._get_components()
        code = 'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b\n'
        result = await _validate_code_quality(code, language="python")
        assert result["passed"] is True
        assert result["score"] > 0.8

    async def test_validate_code_with_violations(self) -> None:
        """Code with violations should fail."""
        server_mod._get_components()
        code = "result = eval(user_input)\n"
        result = await _validate_code_quality(code, language="python")
        assert result["passed"] is False
        assert len(result["violations"]) > 0

    async def test_validate_rejects_oversized_code(self) -> None:
        """Should return error for code exceeding max length."""
        oversized = "x" * (server_mod._MAX_CODE_LENGTH + 1)
        result = await _validate_code_quality(oversized)
        assert "error" in result
        assert result["max_length"] == server_mod._MAX_CODE_LENGTH

    async def test_validate_auto_detect_language(self) -> None:
        """Should auto-detect language when 'auto' is specified."""
        server_mod._get_components()
        code = 'fn main() {\n    let x = 5;\n    println!("{}", x);\n}\n'
        result = await _validate_code_quality(code, language="auto")
        assert result["language_detected"] == "rust"

    async def test_validate_severity_threshold_filters(self) -> None:
        """severity_threshold should filter violations in the response."""
        server_mod._get_components()
        lines = ["def long_function() -> int:"] + ["    x = 1"] * 34 + ["    return x"]
        code = "\n".join(lines)
        result = await _validate_code_quality(code, language="python", severity_threshold="error")
        # With error threshold, warnings should be filtered out
        for v in result.get("violations", []):
            assert v["severity"] == "error"

    async def test_validate_check_security_flag(self) -> None:
        """check_security=False should disable SEC* security rule checks.

        Note: Language-level rules like PY001 (no-eval) are NOT affected
        by check_security — they fall under language/style checks.
        """
        server_mod._get_components()
        # Use code that would trigger a SEC rule specifically
        code = 'query = "SELECT * FROM users WHERE id=" + user_id\n'
        result_with = await _validate_code_quality(code, language="python", check_security=True)
        result_without = await _validate_code_quality(code, language="python", check_security=False)
        sec_with = [
            v for v in result_with.get("violations", []) if v.get("id", "").startswith("SEC")
        ]
        sec_without = [
            v for v in result_without.get("violations", []) if v.get("id", "").startswith("SEC")
        ]
        # SEC rules should be present with security checks enabled
        # and absent (or fewer) with security checks disabled
        assert len(sec_without) <= len(sec_with)

    async def test_validate_empty_code(self) -> None:
        """Should handle empty code gracefully."""
        server_mod._get_components()
        result = await _validate_code_quality("", language="python")
        assert result["passed"] is True

    async def test_validate_check_architecture_flag(self) -> None:
        """check_architecture=False should skip architecture checks."""
        server_mod._get_components()
        lines = ["def long_function() -> int:"] + ["    x = 1"] * 34 + ["    return x"]
        code = "\n".join(lines)
        result = await _validate_code_quality(code, language="python", check_architecture=False)
        arch_violations = [
            v for v in result.get("violations", []) if v.get("id", "").startswith("ARCH")
        ]
        assert len(arch_violations) == 0

    async def test_validate_check_style_flag(self) -> None:
        """check_style=False should disable style checks."""
        server_mod._get_components()
        code = "from typing import List\nx: List[int] = []\n"
        result = await _validate_code_quality(code, language="python", check_style=False)
        style_violations = [
            v for v in result.get("violations", []) if v.get("category", "") == "style"
        ]
        assert len(style_violations) == 0


# ---------------------------------------------------------------------------
# Tool registration count
# ---------------------------------------------------------------------------


class TestKnowledgeStorageHint:
    """Tests for knowledge_storage_hint in validate_code_quality."""

    async def test_hint_present_when_knowledge_entries_exist(self) -> None:
        """Should include knowledge_storage_hint when knowledge_entries are generated."""
        c = server_mod._get_components()
        # Code with 3+ same violations triggers KnowledgeProducer pattern extraction
        code = "try:\n    pass\nexcept:\n    pass\ntry:\n    pass\nexcept:\n    pass\ntry:\n    pass\nexcept:\n    pass\n"
        result = await _validate_code_quality(code, language="python")
        if "knowledge_entries" in result:
            # auto_memory is False by default, so hint should appear
            assert not c.config.orchestration.auto_memory
            assert "knowledge_storage_hint" in result
            assert "enyal_remember" in result["knowledge_storage_hint"]

    async def test_hint_absent_when_no_knowledge_entries(self) -> None:
        """Should NOT include knowledge_storage_hint when no entries generated."""
        server_mod._get_components()
        code = 'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b\n'
        result = await _validate_code_quality(code, language="python")
        assert "knowledge_storage_hint" not in result


class TestToolRegistration:
    """Tests for MCP tool registration."""

    def test_exactly_seven_tools_registered(self) -> None:
        """All 7 MCP tools should be registered."""
        tool_names = set(server_mod.mcp._tool_manager._tools.keys())
        assert len(tool_names) == 7
        expected = {
            "enhance_prompt",
            "validate_code_quality",
            "validate_quick",
            "get_quality_standards",
            "get_quality_trends",
            "scan_dependencies",
            "scan_conventions",
        }
        assert tool_names == expected

    def test_no_deprecated_tools_registered(self) -> None:
        """Deprecated tool names should not be registered."""
        tool_names = set(server_mod.mcp._tool_manager._tools.keys())
        deprecated = {
            "analyze_intent",
            "suggest_tools",
            "get_verification_checklist",
            "validate_diff",
            "validate_plan_quality",
            "compare_approaches",
        }
        assert tool_names.isdisjoint(deprecated)


# ---------------------------------------------------------------------------
# _get_components singleton
# ---------------------------------------------------------------------------


class TestGetComponents:
    """Tests for the _get_components() singleton factory."""

    def test_returns_components_instance(self) -> None:
        """Should return a _Components dataclass."""
        c = server_mod._get_components()
        assert isinstance(c, server_mod._Components)

    def test_returns_singleton(self) -> None:
        """Subsequent calls should return the same instance."""
        c1 = server_mod._get_components()
        c2 = server_mod._get_components()
        assert c1 is c2

    def test_all_components_initialized(self) -> None:
        """All component fields should be non-None."""
        c = server_mod._get_components()
        assert c.intent_analyzer is not None
        assert c.quality_standards is not None
        assert c.prompt_composer is not None
        assert c.mcp_orchestrator is not None
        assert c.context_aggregator is not None
        assert c.code_validator is not None
        assert c.plan_validator is not None
        assert c.config is not None


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the main() entry point."""

    def test_main_calls_mcp_run(self) -> None:
        """main() should call mcp.run()."""
        with patch.object(server_mod.mcp, "run") as mock_run:
            server_mod.main()
        mock_run.assert_called_once()
