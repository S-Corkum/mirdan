"""Cross-IDE parity tests for the brief-driven pipeline.

Claude Code and Cursor must produce byte-identical results for brief-driven
pipeline operations that go through MCP tools. Parity is enforced by the
fact that both IDEs call the same MCP tool; this test suite pins the
invariant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mirdan.usecases.validate_brief import ValidateBriefUseCase
from mirdan.usecases.verify_plan_against_brief import VerifyPlanAgainstBriefUseCase

_BRIEF = """# Brief: parity fixture

## Outcome
Response time under 200ms at p95.

## Users & Scenarios
Primary: backend engineer.
Scenario: user calls /cache endpoint.

## Business Acceptance Criteria
- [ ] Endpoint responds under 200ms at p95
- [ ] Cache invalidates on write
- [ ] Hit-rate metric emitted

## Constraints
- Must not modify Redis schema

## Out of Scope
- Multi-region replication
"""


_PLAN = """---
plan: parity
brief: docs/briefs/parity.md
---

# Plan: parity

## Research Notes
verified 2026-04-23

## Epic Layer
Ship cache endpoint.

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
**Details:** implement cache
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
# Parity — validate_brief
# ---------------------------------------------------------------------------


class TestValidateBriefParity:
    """Both IDEs call the same MCP tool path, so output must be identical."""

    @pytest.mark.asyncio
    async def test_deterministic_across_calls(self, brief_file: Path) -> None:
        """Same input → same output across repeat calls."""
        uc = ValidateBriefUseCase()
        first = await uc.execute(str(brief_file))
        second = await uc.execute(str(brief_file))
        assert first == second


# ---------------------------------------------------------------------------
# Parity — verify_plan_against_brief (mechanical path)
# ---------------------------------------------------------------------------


class TestVerifyPlanParity:
    """Mechanical verification must be deterministic and schema-stable."""

    @pytest.mark.asyncio
    async def test_deterministic_across_calls(
        self, brief_file: Path, plan_file: Path
    ) -> None:
        uc = VerifyPlanAgainstBriefUseCase(llm_manager=None)
        first = await uc.execute(str(plan_file), str(brief_file))
        second = await uc.execute(str(plan_file), str(brief_file))
        assert first == second

    @pytest.mark.asyncio
    async def test_output_shape_stable(
        self, brief_file: Path, plan_file: Path
    ) -> None:
        """Every downstream tool expects these keys — they must not drift."""
        uc = VerifyPlanAgainstBriefUseCase(llm_manager=None)
        result = await uc.execute(str(plan_file), str(brief_file))
        expected_keys = {
            "verified",
            "coverage_score",
            "unmapped_acs",
            "missing_grounding",
            "out_of_scope_violations",
            "invest_failures",
            "semantic_check_skipped",
            "summary",
        }
        assert expected_keys.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# Parity — shared plan-review rubric
# ---------------------------------------------------------------------------


class TestSharedRubricParity:
    """The plan-review rubric is the contract that forces identical output shape."""

    def test_rubric_file_exists(self) -> None:
        from importlib.resources import files

        pkg = files("mirdan.templates")
        rubric = pkg / "plan-review-rubric.md"
        assert rubric.is_file()

    def test_rubric_has_exact_5_sections(self) -> None:
        from importlib.resources import files

        pkg = files("mirdan.templates")
        content = (pkg / "plan-review-rubric.md").read_text()
        for heading in (
            "## unmapped_acs",
            "## constraint_violations",
            "## scope_violations",
            "## grounding_gaps",
            "## risks",
        ):
            assert heading in content, f"rubric missing section: {heading}"
        assert "**Verdict:**" in content

    def test_cc_plan_reviewer_agent_references_rubric(self) -> None:
        """Claude Code plan-reviewer agent enforces the rubric."""
        from importlib.resources import files

        pkg = files("mirdan.integrations.templates.claude_code.agents")
        content = (pkg / "plan-reviewer.md").read_text()
        assert "plan-review-rubric.md" in content
        for heading in (
            "unmapped_acs",
            "constraint_violations",
            "scope_violations",
            "grounding_gaps",
            "risks",
        ):
            assert heading in content

    def test_cursor_plan_review_command_references_rubric(self) -> None:
        """Cursor plan-review command enforces the rubric."""
        from importlib.resources import files

        pkg = files("mirdan.integrations.templates.cursor_commands")
        content = (pkg / "plan-review.md").read_text()
        assert "plan-review-rubric.md" in content
        for heading in (
            "unmapped_acs",
            "constraint_violations",
            "scope_violations",
            "grounding_gaps",
            "risks",
        ):
            assert heading in content
