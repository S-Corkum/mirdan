"""Tests for M2 tool consolidation — verify new modes and deprecated aliases."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

import mirdan.server as server_mod

# Extract raw async functions from FastMCP FunctionTool wrappers
_enhance_prompt = server_mod.enhance_prompt.fn
_validate_code_quality = server_mod.validate_code_quality.fn


@pytest.fixture(autouse=True)
def _reset_components() -> None:
    """Reset the server singleton before each test."""
    server_mod._components = None
    yield
    server_mod._components = None


# ---------------------------------------------------------------------------
# enhance_prompt: analyze_only mode (replaces analyze_intent)
# ---------------------------------------------------------------------------


class TestEnhancePromptAnalyzeOnly:
    """Tests for enhance_prompt(task_type='analyze_only')."""

    async def test_returns_intent_keys(self) -> None:
        """Should return dict with all analyze_intent keys."""
        server_mod._get_components()
        result = await _enhance_prompt("Write a login function in Python", task_type="analyze_only")
        expected_keys = {
            "task_type",
            "language",
            "frameworks",
            "touches_security",
            "ambiguity_score",
            "ambiguity_level",
            "extracted_entities",
            "clarifying_questions",
        }
        assert expected_keys.issubset(result.keys())

    async def test_detects_security(self) -> None:
        server_mod._get_components()
        result = await _enhance_prompt(
            "Implement password hashing for user authentication",
            task_type="analyze_only",
        )
        assert result["touches_security"] is True

    async def test_detects_rag(self) -> None:
        server_mod._get_components()
        result = await _enhance_prompt(
            "Build a RAG pipeline with vector embeddings",
            task_type="analyze_only",
        )
        assert result["touches_rag"] is True

    async def test_ambiguity_levels(self) -> None:
        server_mod._get_components()
        result = await _enhance_prompt("fix it", task_type="analyze_only")
        assert result["ambiguity_level"] in ("low", "medium", "high")


# ---------------------------------------------------------------------------
# enhance_prompt: plan_validation mode (replaces validate_plan_quality)
# ---------------------------------------------------------------------------


class TestEnhancePromptPlanValidation:
    """Tests for enhance_prompt(task_type='plan_validation')."""

    async def test_returns_plan_quality_keys(self) -> None:
        """Should return plan quality score dict."""
        server_mod._get_components()
        plan = """
## Research Notes (Pre-Plan Verification)
### Files Verified
- `src/auth.py`: line 45

### Step 1: Add import
**File:** `src/auth.py`
**Action:** Edit
**Details:** Add import bcrypt
**Verify:** Read file
**Grounding:** Read of src/auth.py
"""
        result = await _enhance_prompt(plan, task_type="plan_validation")
        assert "overall_score" in result
        assert "issues" in result
        assert "ready_for_cheap_model" in result

    async def test_vague_plan_low_score(self) -> None:
        server_mod._get_components()
        result = await _enhance_prompt(
            "I think we should probably fix it",
            task_type="plan_validation",
        )
        assert result["overall_score"] < 1.0
        assert len(result["issues"]) > 0


# ---------------------------------------------------------------------------
# enhance_prompt: context_level="none" (M5 preview)
# ---------------------------------------------------------------------------


class TestEnhancePromptContextNone:
    """Tests for enhance_prompt(context_level='none')."""

    async def test_returns_without_context(self) -> None:
        """context_level='none' should skip context gathering."""
        server_mod._get_components()
        # Should NOT call gather_all
        with patch.object(
            server_mod._get_components().context_aggregator,
            "gather_all",
            new_callable=AsyncMock,
        ) as mock_gather:
            result = await _enhance_prompt("Create a Python function", context_level="none")

        mock_gather.assert_not_awaited()
        assert "enhanced_prompt" in result
        assert "session_id" in result


# ---------------------------------------------------------------------------
# validate_code_quality: input_type="diff" (replaces validate_diff)
# ---------------------------------------------------------------------------


class TestValidateCodeQualityDiff:
    """Tests for validate_code_quality(input_type='diff')."""

    async def test_diff_validation(self) -> None:
        server_mod._get_components()
        diff = "--- a/test.py\n+++ b/test.py\n@@ -1,3 +1,4 @@\n x = 1\n+y = 2\n z = 3\n"
        result = await _validate_code_quality(diff, language="python", input_type="diff")
        assert "passed" in result

    async def test_empty_diff(self) -> None:
        server_mod._get_components()
        diff = "--- a/test.py\n+++ b/test.py\n@@ -1,3 +1,2 @@\n x = 1\n-removed = 2\n z = 3\n"
        result = await _validate_code_quality(diff, input_type="diff")
        assert result["passed"] is True
        assert result["score"] == 1.0


# ---------------------------------------------------------------------------
# validate_code_quality: compare=True (replaces compare_approaches)
# ---------------------------------------------------------------------------


class TestValidateCodeQualityCompare:
    """Tests for validate_code_quality(compare=True)."""

    async def test_compare_two_implementations(self) -> None:
        server_mod._get_components()
        impls = json.dumps(
            [
                "def greet(name: str) -> str:\n    return f'Hello, {name}'\n",
                "result = eval(user_input)\n",
            ]
        )
        result = await _validate_code_quality(impls, language="python", compare=True)
        assert "winner" in result
        assert "entries" in result
        assert len(result["entries"]) == 2

    async def test_compare_invalid_json(self) -> None:
        server_mod._get_components()
        result = await _validate_code_quality("not json", compare=True)
        assert "error" in result

    async def test_compare_not_list(self) -> None:
        server_mod._get_components()
        result = await _validate_code_quality(json.dumps({"a": 1}), compare=True)
        assert "error" in result

    async def test_compare_too_few(self) -> None:
        server_mod._get_components()
        result = await _validate_code_quality(json.dumps(["single"]), compare=True)
        assert "error" in result


# ---------------------------------------------------------------------------
# validate_code_quality: checklist in output
# ---------------------------------------------------------------------------


class TestValidateCodeQualityChecklist:
    """Tests for checklist field in validate_code_quality output."""

    async def test_output_includes_checklist(self) -> None:
        server_mod._get_components()
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        result = await _validate_code_quality(code, language="python")
        assert "checklist" in result
        assert isinstance(result["checklist"], list)
        assert len(result["checklist"]) > 0


# ---------------------------------------------------------------------------
# Consolidated modes cover all former deprecated tool functionality
# ---------------------------------------------------------------------------
# analyze_intent → enhance_prompt(task_type="analyze_only")  [tested above]
# validate_plan_quality → enhance_prompt(task_type="plan_validation")  [tested above]
# validate_diff → validate_code_quality(input_type="diff")  [tested above]
# compare_approaches → validate_code_quality(compare=True)  [tested above]
# suggest_tools → enhance_prompt includes tool_recommendations  [tested in test_server_tools]
# get_verification_checklist → validate_code_quality includes checklist  [tested above]
