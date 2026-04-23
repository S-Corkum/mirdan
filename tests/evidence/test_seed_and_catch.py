"""Seed-and-catch: plant known defects; measure verifier detection rate.

Not a unit test of the verifier internals — this is evidence that the
verifier actually detects the issues it claims to detect, across a variety of
plausible defect types.

Pass threshold: 100% detection of seeded structural defects (the verifier
operates on rules, not heuristics, so misses would be bugs). False-positive
rate on clean plans should be 0%.
"""

from __future__ import annotations

import re

import pytest

from mirdan.usecases.verify_plan_against_brief import VerifyPlanAgainstBriefUseCase

# A clean, minimal brief + plan pair. The baseline for all defect seeding.
_CLEAN_BRIEF = """# Brief: example

## Outcome
Response time under 200ms at p95.

## Users & Scenarios
Primary: backend engineer.
Scenario: call /cache endpoint.

## Business Acceptance Criteria
- [ ] Endpoint responds under 200ms at p95
- [ ] Cache invalidates on write
- [ ] Hit-rate metric emitted

## Constraints
- Must not modify Redis schema

## Out of Scope
- Multi-region replication
"""


_CLEAN_PLAN = """---
plan: example
brief: /tmp/example-brief.md
---

# Plan: example

## Research Notes
Verified 2026-04-23.

## Epic Layer
Ship cache endpoint.

## Story Layer

### Story 1 — cache endpoint
- **As** backend engineer
- **I want** a cache endpoint
- **So that** latency drops

**Acceptance Criteria:**
- [ ] cache returns under 200ms
- [ ] invalidation on writes
- [ ] hit-rate metric emitted

#### Subtasks

##### 1.1 — implement cache layer
**File:** src/cache.py
**Action:** Write
**Details:** implement cache
**Depends on:** —
**Verify:** run tests/test_cache.py
**Grounding:** Read src/redis.py 2026-04-23
"""


@pytest.fixture
def brief_path(tmp_path):
    p = tmp_path / "brief.md"
    p.write_text(_CLEAN_BRIEF)
    return p


@pytest.fixture
def plan_path(tmp_path):
    p = tmp_path / "plan.md"
    p.write_text(_CLEAN_PLAN)
    return p


# ---------------------------------------------------------------------------
# Baseline: clean plan should verify cleanly (no false positives).
# ---------------------------------------------------------------------------


class TestFalsePositivesOnCleanPlan:
    @pytest.mark.asyncio
    async def test_clean_plan_has_no_grounding_gaps(self, brief_path, plan_path):
        uc = VerifyPlanAgainstBriefUseCase(llm_manager=None)
        r = await uc.execute(str(plan_path), str(brief_path))
        assert r["missing_grounding"] == [], (
            f"False-positive: clean plan flagged {len(r['missing_grounding'])} "
            f"grounding gaps: {r['missing_grounding']}"
        )

    @pytest.mark.asyncio
    async def test_clean_plan_has_no_invest_failures(self, brief_path, plan_path):
        uc = VerifyPlanAgainstBriefUseCase(llm_manager=None)
        r = await uc.execute(str(plan_path), str(brief_path))
        assert r["invest_failures"] == [], (
            f"False-positive: clean plan flagged {len(r['invest_failures'])} "
            f"INVEST failures: {r['invest_failures']}"
        )

    @pytest.mark.asyncio
    async def test_clean_plan_has_no_scope_violations(self, brief_path, plan_path):
        uc = VerifyPlanAgainstBriefUseCase(llm_manager=None)
        r = await uc.execute(str(plan_path), str(brief_path))
        assert r["out_of_scope_violations"] == []


# ---------------------------------------------------------------------------
# Seeded grounding defects (one field removed at a time).
# ---------------------------------------------------------------------------


_GROUNDING_FIELDS = [
    ("**File:**", "File"),
    ("**Action:**", "Action"),
    ("**Details:**", "Details"),
    ("**Depends on:**", "Depends on"),
    ("**Verify:**", "Verify"),
    ("**Grounding:**", "Grounding"),
]


class TestSeededGroundingDefects:
    """Remove each grounding field in turn; assert verifier catches each."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("marker", "field_name"), _GROUNDING_FIELDS)
    async def test_removed_field_detected(
        self, tmp_path, brief_path, marker, field_name
    ):
        # Build a plan with this specific field removed from subtask 1.1
        defective = re.sub(
            rf"{re.escape(marker)}[^\n]*\n",
            "",
            _CLEAN_PLAN,
            count=1,
        )
        plan = tmp_path / "defective.md"
        plan.write_text(defective)

        uc = VerifyPlanAgainstBriefUseCase(llm_manager=None)
        r = await uc.execute(str(plan), str(brief_path))

        assert r["missing_grounding"], (
            f"DEFECT ESCAPED: removing {field_name} did not trip the verifier"
        )
        found = any(
            field_name in item["issue"] for item in r["missing_grounding"]
        )
        assert found, (
            f"DEFECT MISCLASSIFIED: removed {field_name} but verifier reported "
            f"{r['missing_grounding']}"
        )


# ---------------------------------------------------------------------------
# Seeded INVEST defects (remove story fields).
# ---------------------------------------------------------------------------


_INVEST_FIELDS = [
    ("**As** backend engineer", "As"),
    ("**I want** a cache endpoint", "I want"),
    ("**So that** latency drops", "So that"),
    ("**Acceptance Criteria:**", "Acceptance Criteria"),
]


class TestSeededInvestDefects:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("marker", "field_name"), _INVEST_FIELDS)
    async def test_removed_invest_field_detected(
        self, tmp_path, brief_path, marker, field_name
    ):
        defective = _CLEAN_PLAN.replace(marker, "")
        plan = tmp_path / "defective.md"
        plan.write_text(defective)

        uc = VerifyPlanAgainstBriefUseCase(llm_manager=None)
        r = await uc.execute(str(plan), str(brief_path))

        assert r["invest_failures"], (
            f"DEFECT ESCAPED: removing {field_name} did not trip the verifier"
        )


# ---------------------------------------------------------------------------
# Seeded scope-violation defects.
# ---------------------------------------------------------------------------


class TestSeededScopeViolations:
    @pytest.mark.asyncio
    async def test_out_of_scope_item_in_plan_detected(
        self, tmp_path, brief_path
    ):
        # Brief says Multi-region replication is out of scope.
        # Sneak it into the plan body in a way that suggests it's IN scope.
        defective = _CLEAN_PLAN.replace(
            "## Research Notes\nVerified 2026-04-23.",
            (
                "## Research Notes\n"
                "Verified 2026-04-23.\n\n"
                "This plan adds Multi-region replication across 3 datacenters "
                "with Multi-region replication failover logic."
            ),
        )
        plan = tmp_path / "defective.md"
        plan.write_text(defective)

        uc = VerifyPlanAgainstBriefUseCase(llm_manager=None)
        r = await uc.execute(str(plan), str(brief_path))

        assert r["out_of_scope_violations"], (
            "DEFECT ESCAPED: plan touches 'Multi-region replication' "
            "but verifier did not flag a scope violation"
        )
