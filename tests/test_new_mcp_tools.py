"""Integration tests for 2.1.0 brief-driven pipeline MCP tools.

Tests cover:
- validate_brief (no LLM needed)
- verify_plan_against_brief (mechanical checks, LLM graceful degradation)
- propose_subtask_diff (fail-closed when LLM unavailable)
- mirdan_health (shape assertions)
- enhance_prompt brief_path merging

@pytest.mark.local_llm tests require a running Gemma 4; they are skipped
when the model is not available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mirdan.usecases.propose_subtask_diff import ProposeSubtaskDiffUseCase
from mirdan.usecases.validate_brief import ValidateBriefUseCase
from mirdan.usecases.verify_plan_against_brief import VerifyPlanAgainstBriefUseCase

# ---------------------------------------------------------------------------
# Fixtures — sample brief + sample plan
# ---------------------------------------------------------------------------


_BRIEF = """# Brief: example

## Outcome
Ship with latency under 200ms.

## Users & Scenarios
Primary: backend engineer.
Scenario: issues requests to the cache endpoint.

## Business Acceptance Criteria
- [ ] Endpoint responds under 200ms at p95
- [ ] Cache invalidates on write
- [ ] Metrics emitted for hit rate

## Constraints
- Must not modify Redis schema
- Must pass existing auth middleware

## Out of Scope
- Migration of legacy cache tier
- Multi-region replication
"""


_PLAN = """---
plan: example
brief: docs/briefs/example.md
---

# Plan: example

## Research Notes
Verified files.

## Epic Layer
Outcome: latency.

## Story Layer

### Story 1 — cache
- **As** backend engineer
- **I want** cache endpoint
- **So that** latency drops

**Acceptance Criteria:**
- [ ] cache returns under 200ms
- [ ] invalidation on writes
- [ ] hit-rate metric emitted

#### Subtasks

##### 1.1 — wire cache
**File:** src/cache.py
**Action:** Write
**Details:** implement cache layer
**Depends on:** —
**Verify:** run test_cache
**Grounding:** Read redis.py
"""


@pytest.fixture
def brief_file(tmp_path: Path) -> Path:
    p = tmp_path / "brief.md"
    p.write_text(_BRIEF)
    return p


@pytest.fixture
def plan_file(tmp_path: Path) -> Path:
    p = tmp_path / "plan.md"
    p.write_text(_PLAN)
    return p


# ---------------------------------------------------------------------------
# validate_brief
# ---------------------------------------------------------------------------


class TestValidateBrief:
    @pytest.mark.asyncio
    async def test_passing_brief(self, brief_file: Path) -> None:
        uc = ValidateBriefUseCase()
        result = await uc.execute(str(brief_file))
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_missing_section_fails(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.md"
        p.write_text("# Brief\n\n## Outcome\nx\n")
        uc = ValidateBriefUseCase()
        result = await uc.execute(str(p))
        assert result["passed"] is False
        assert len(result["missing_required"]) >= 4

    @pytest.mark.asyncio
    async def test_missing_file_errors(self) -> None:
        uc = ValidateBriefUseCase()
        result = await uc.execute("/tmp/does_not_exist_12345.md")
        assert result["passed"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# verify_plan_against_brief — mechanical path (no LLM)
# ---------------------------------------------------------------------------


class TestVerifyPlanMechanical:
    @pytest.mark.asyncio
    async def test_semantic_check_skipped_when_no_llm(
        self, brief_file: Path, plan_file: Path
    ) -> None:
        uc = VerifyPlanAgainstBriefUseCase(llm_manager=None)
        result = await uc.execute(str(plan_file), str(brief_file))
        assert result["semantic_check_skipped"] is True
        # Grounding is complete in fixture plan → no missing_grounding errors
        assert result["missing_grounding"] == []

    @pytest.mark.asyncio
    async def test_missing_grounding_detected(
        self, brief_file: Path, tmp_path: Path
    ) -> None:
        bad_plan = tmp_path / "bad.md"
        bad_plan.write_text(
            _PLAN.replace("**Grounding:** Read redis.py", "")
        )
        uc = VerifyPlanAgainstBriefUseCase(llm_manager=None)
        result = await uc.execute(str(bad_plan), str(brief_file))
        assert any(
            g["issue"].startswith("Subtask missing 'Grounding'")
            or "Grounding" in g["issue"]
            for g in result["missing_grounding"]
        )

    @pytest.mark.asyncio
    async def test_plan_not_found(self, brief_file: Path) -> None:
        uc = VerifyPlanAgainstBriefUseCase(llm_manager=None)
        result = await uc.execute("/tmp/does_not_exist_plan.md", str(brief_file))
        assert result["verified"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# propose_subtask_diff — fail-closed without LLM
# ---------------------------------------------------------------------------


class TestProposeSubtaskDiffFailClosed:
    @pytest.mark.asyncio
    async def test_halts_when_llm_unavailable(self) -> None:
        uc = ProposeSubtaskDiffUseCase(llm_manager=None)
        result = await uc.execute(
            subtask_yaml="File: x.py\nAction: Edit\n",
            file_context={"x.py": "# content"},
        )
        assert result["halted"] is True
        assert result["diff"] == ""
        assert "local_llm" in result["halt_reason"]

    @pytest.mark.asyncio
    async def test_halts_when_no_role_available(self) -> None:
        class FakeMgr:
            def is_role_available(self, role):  # type: ignore[no-untyped-def]
                return False

            async def generate(self, *a, **kw):  # type: ignore[no-untyped-def]
                return None

        uc = ProposeSubtaskDiffUseCase(llm_manager=FakeMgr())  # type: ignore[arg-type]
        result = await uc.execute(
            subtask_yaml="File: x.py\n",
            file_context={"x.py": "# content"},
        )
        assert result["halted"] is True


# ---------------------------------------------------------------------------
# mirdan_health
# ---------------------------------------------------------------------------


class TestMirdanHealth:
    @pytest.mark.asyncio
    async def test_returns_expected_shape(self) -> None:
        from mirdan.server import mirdan_health

        result = await mirdan_health()
        expected_keys = {
            "local_llm_available",
            "model_in_use",
            "vram_gb",
            "recommended_mode",
            "backend_kind",
        }
        # FastMCP tools expose via a wrapper — if result is a dict, check keys
        if isinstance(result, dict):
            assert expected_keys.issubset(set(result.keys()))
            assert result["recommended_mode"] in ("inline", "proxied")


# ---------------------------------------------------------------------------
# enhance_prompt brief_path merging
# ---------------------------------------------------------------------------


class TestEnhancePromptBriefMerge:
    @pytest.mark.asyncio
    async def test_brief_path_populates_out_of_scope(
        self, brief_file: Path
    ) -> None:
        """When brief_path is provided, enhance_prompt output includes out_of_scope."""
        from mirdan.providers import ComponentProvider

        provider = ComponentProvider()
        uc = provider.create_enhance_prompt_usecase(set())
        result = await uc.execute(
            prompt="add caching to the user endpoint",
            brief_path=str(brief_file),
        )
        assert isinstance(result, dict)
        # At minimum, one of the brief-derived fields must be present
        has_oos = "out_of_scope" in result
        has_brief_prefix = any(
            isinstance(v, str) and "[from brief]" in v
            for v in result.values()
        ) or any(
            isinstance(v, list)
            and any(isinstance(x, str) and "[from brief]" in x for x in v)
            for v in result.values()
        )
        assert has_oos or has_brief_prefix

    @pytest.mark.asyncio
    async def test_no_brief_path_preserves_existing_behavior(self) -> None:
        """Default behavior (no brief_path) must be unchanged."""
        from mirdan.providers import ComponentProvider

        provider = ComponentProvider()
        uc = provider.create_enhance_prompt_usecase(set())
        result = await uc.execute(prompt="add caching to the user endpoint")
        assert isinstance(result, dict)
        assert "out_of_scope" not in result


# ---------------------------------------------------------------------------
# Local-LLM integration tests (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.local_llm
class TestLocalLLMIntegration:
    """Require a running Gemma 4. Skipped unless --run-local-llm is passed."""

    @pytest.mark.asyncio
    async def test_verify_plan_full(self, brief_file: Path, plan_file: Path) -> None:
        from mirdan.config import LLMConfig
        from mirdan.llm.manager import LLMManager

        mgr = LLMManager.create_if_enabled(LLMConfig(enabled=True))
        if mgr is None:
            pytest.skip("LLM not enabled")
        uc = VerifyPlanAgainstBriefUseCase(llm_manager=mgr)
        result = await uc.execute(str(plan_file), str(brief_file))
        # Semantic path runs — result may verify or fail, but key must be present
        assert "semantic_check_skipped" in result
