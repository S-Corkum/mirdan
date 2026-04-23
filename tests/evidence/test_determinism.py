"""Determinism proof: verifier output is byte-identical across runs.

Byte-identical output is what lets Claude Code and Cursor both call the same
MCP tool and get interchangeable results. If this breaks, the "IDE parity"
claim collapses.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from mirdan.usecases.validate_brief import ValidateBriefUseCase
from mirdan.usecases.verify_plan_against_brief import VerifyPlanAgainstBriefUseCase


def _hash(obj: dict) -> str:
    """Stable hash of a dict — preserves ordering under json sort."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode()
    ).hexdigest()


_BRIEF = """# Brief

## Outcome
x

## Users & Scenarios
y

## Business Acceptance Criteria
- [ ] a
- [ ] b
- [ ] c

## Constraints
- x

## Out of Scope
- y
"""


_PLAN = """---
brief: /tmp/b.md
---

# Plan

## Research Notes
ok

## Epic Layer
e

## Story Layer

### Story 1 — t
- **As** u
- **I want** x
- **So that** y

**Acceptance Criteria:**
- [ ] a

#### Subtasks
##### 1.1 — act
**File:** x
**Action:** Edit
**Details:** d
**Depends on:** —
**Verify:** v
**Grounding:** g
"""


@pytest.fixture
def brief_path(tmp_path):
    p = tmp_path / "b.md"
    p.write_text(_BRIEF)
    return p


@pytest.fixture
def plan_path(tmp_path):
    p = tmp_path / "p.md"
    p.write_text(_PLAN)
    return p


N_RUNS = 20


class TestDeterminism:
    @pytest.mark.asyncio
    async def test_validate_brief_deterministic_across_20_runs(self, brief_path):
        uc = ValidateBriefUseCase()
        hashes: set[str] = set()
        for _ in range(N_RUNS):
            r = await uc.execute(str(brief_path))
            hashes.add(_hash(r))
        assert len(hashes) == 1, (
            f"validate_brief non-deterministic: {len(hashes)} distinct outputs "
            f"across {N_RUNS} runs"
        )

    @pytest.mark.asyncio
    async def test_verify_plan_deterministic_across_20_runs(
        self, brief_path, plan_path
    ):
        uc = VerifyPlanAgainstBriefUseCase(llm_manager=None)
        hashes: set[str] = set()
        for _ in range(N_RUNS):
            r = await uc.execute(str(plan_path), str(brief_path))
            hashes.add(_hash(r))
        assert len(hashes) == 1, (
            f"verify_plan_against_brief non-deterministic: {len(hashes)} "
            f"distinct outputs across {N_RUNS} runs"
        )

    @pytest.mark.asyncio
    async def test_verify_plan_deterministic_across_different_instances(
        self, brief_path, plan_path
    ):
        """Same output regardless of which usecase instance runs it.

        This is the 'Cursor and Claude Code call different instances of the
        MCP server but get the same result' invariant.
        """
        hashes: set[str] = set()
        for _ in range(5):
            uc = VerifyPlanAgainstBriefUseCase(llm_manager=None)  # fresh instance
            r = await uc.execute(str(plan_path), str(brief_path))
            hashes.add(_hash(r))
        assert len(hashes) == 1
