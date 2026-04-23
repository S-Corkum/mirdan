"""Seed-and-catch for the 2.1.0 mechanical upgrades.

Three new mechanical checks in VerifyPlanAgainstBriefUseCase — this file
proves each one catches its target defect and doesn't false-positive on
clean plans.

1. File existence verification — phantom_files
2. Cross-subtask dependency validation — dependency_errors
3. Vague cross-reference detection — vague_cross_references

Evidence threshold: 100% detection on seeded defects; 0% false positives
on the clean fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mirdan.usecases.verify_plan_against_brief import VerifyPlanAgainstBriefUseCase

_BRIEF = """# Brief: mechanical upgrade fixture

## Outcome
Ship cache endpoint.

## Users & Scenarios
Primary: backend engineer.
Scenario: call /cache.

## Business Acceptance Criteria
- [ ] Endpoint responds under 200ms
- [ ] Cache invalidates on write
- [ ] Hit-rate metric emitted

## Constraints
- No Redis schema changes

## Out of Scope
- Multi-region replication
"""


def _build_plan(
    file_path_11: str = "src/mirdan/models.py",
    file_path_12: str = "NEW: src/mirdan/cache.py",
    depends_on_11: str = "—",
    depends_on_12: str = "1.1",
    details_11: str = "implement cache",
) -> str:
    """Construct a plan where each field under test is swappable.

    Default path values point at REAL files in the mirdan repo so the
    clean-fixture tests actually verify against the filesystem.
    """
    return f"""---
plan: fixture
brief: /tmp/brief.md
---

# Plan: fixture

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
- [ ] under 200ms
- [ ] invalidation
- [ ] metric

#### Subtasks

##### 1.1 — wire cache module
**File:** {file_path_11}
**Action:** Read
**Details:** {details_11}
**Depends on:** {depends_on_11}
**Verify:** run tests
**Grounding:** Read 2026-04-23

##### 1.2 — create new cache layer
**File:** {file_path_12}
**Action:** Write
**Details:** new cache module
**Depends on:** {depends_on_12}
**Verify:** import cache
**Grounding:** Write 2026-04-23
"""


@pytest.fixture
def brief_path(tmp_path: Path) -> Path:
    p = tmp_path / "brief.md"
    p.write_text(_BRIEF)
    return p


@pytest.fixture
def mirdan_repo() -> Path:
    """Point verifier at the mirdan submodule so real file paths resolve."""
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Clean plan → zero false positives
# ---------------------------------------------------------------------------


class TestCleanPlanFalsePositives:
    @pytest.fixture
    def uc(self, mirdan_repo) -> VerifyPlanAgainstBriefUseCase:
        return VerifyPlanAgainstBriefUseCase(project_root=mirdan_repo)

    @pytest.mark.asyncio
    async def test_no_phantom_files_on_clean_plan(
        self, uc, tmp_path, brief_path
    ):
        plan = tmp_path / "plan.md"
        plan.write_text(_build_plan())  # real mirdan files, valid NEW parent dir
        r = await uc.execute(str(plan), str(brief_path))
        assert r["phantom_files"] == [], (
            f"false-positive: {r['phantom_files']}"
        )

    @pytest.mark.asyncio
    async def test_no_dependency_errors_on_clean_plan(
        self, uc, tmp_path, brief_path
    ):
        plan = tmp_path / "plan.md"
        plan.write_text(_build_plan())
        r = await uc.execute(str(plan), str(brief_path))
        assert r["dependency_errors"] == []

    @pytest.mark.asyncio
    async def test_no_vague_cross_refs_on_clean_plan(
        self, uc, tmp_path, brief_path
    ):
        plan = tmp_path / "plan.md"
        plan.write_text(_build_plan())
        r = await uc.execute(str(plan), str(brief_path))
        assert r["vague_cross_references"] == []


# ---------------------------------------------------------------------------
# Seeded phantom files — detection
# ---------------------------------------------------------------------------


class TestPhantomFileDetection:
    @pytest.fixture
    def uc(self, mirdan_repo) -> VerifyPlanAgainstBriefUseCase:
        return VerifyPlanAgainstBriefUseCase(project_root=mirdan_repo)

    @pytest.mark.asyncio
    async def test_nonexistent_file_detected(
        self, uc, tmp_path, brief_path
    ):
        plan = tmp_path / "plan.md"
        plan.write_text(_build_plan(file_path_11="src/mirdan/does_not_exist.py"))
        r = await uc.execute(str(plan), str(brief_path))
        phantoms = [p["path"] for p in r["phantom_files"]]
        assert "src/mirdan/does_not_exist.py" in phantoms

    @pytest.mark.asyncio
    async def test_new_file_with_missing_parent_dir_detected(
        self, uc, tmp_path, brief_path
    ):
        plan = tmp_path / "plan.md"
        plan.write_text(
            _build_plan(file_path_12="NEW: src/mirdan/nowhere/impossible.py")
        )
        r = await uc.execute(str(plan), str(brief_path))
        paths = [p["path"] for p in r["phantom_files"]]
        assert "src/mirdan/nowhere/impossible.py" in paths

    @pytest.mark.asyncio
    async def test_new_file_with_valid_parent_accepted(
        self, uc, tmp_path, brief_path
    ):
        """NEW: file with existing parent dir is OK — not phantom."""
        plan = tmp_path / "plan.md"
        plan.write_text(_build_plan(file_path_12="NEW: src/mirdan/new_module.py"))
        r = await uc.execute(str(plan), str(brief_path))
        paths = [p["path"] for p in r["phantom_files"]]
        assert "src/mirdan/new_module.py" not in paths


# ---------------------------------------------------------------------------
# Seeded dependency errors — detection
# ---------------------------------------------------------------------------


class TestDependencyErrorDetection:
    @pytest.fixture
    def uc(self, mirdan_repo) -> VerifyPlanAgainstBriefUseCase:
        return VerifyPlanAgainstBriefUseCase(project_root=mirdan_repo)

    @pytest.mark.asyncio
    async def test_dangling_reference_detected(
        self, uc, tmp_path, brief_path
    ):
        plan = tmp_path / "plan.md"
        plan.write_text(_build_plan(depends_on_12="9.9"))  # 9.9 doesn't exist
        r = await uc.execute(str(plan), str(brief_path))
        missing = [e["missing_dep"] for e in r["dependency_errors"]]
        assert "9.9" in missing

    @pytest.mark.asyncio
    async def test_cycle_detected(self, uc, tmp_path, brief_path):
        # 1.1 depends on 1.2, 1.2 depends on 1.1 → cycle
        plan = tmp_path / "plan.md"
        plan.write_text(
            _build_plan(depends_on_11="1.2", depends_on_12="1.1")
        )
        r = await uc.execute(str(plan), str(brief_path))
        cycle_errors = [e for e in r["dependency_errors"] if "cycle" in e["issue"].lower()]
        assert cycle_errors, (
            f"cycle not detected; dependency_errors = {r['dependency_errors']}"
        )


# ---------------------------------------------------------------------------
# Seeded vague cross-references — detection
# ---------------------------------------------------------------------------


class TestVagueCrossReferenceDetection:
    @pytest.fixture
    def uc(self, mirdan_repo) -> VerifyPlanAgainstBriefUseCase:
        return VerifyPlanAgainstBriefUseCase(project_root=mirdan_repo)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "phrase",
        [
            "as discussed",
            "as mentioned above",
            "from before",
            "like Step 3",
            "the function from earlier",
            "see above",
        ],
    )
    async def test_phrase_detected(
        self, uc, tmp_path, brief_path, phrase
    ):
        plan = tmp_path / "plan.md"
        plan.write_text(
            _build_plan(details_11=f"implement the cache — {phrase}")
        )
        r = await uc.execute(str(plan), str(brief_path))
        assert r["vague_cross_references"], (
            f"phrase '{phrase}' not detected as vague"
        )


# ---------------------------------------------------------------------------
# Output shape contract — new fields must be present
# ---------------------------------------------------------------------------


class TestOutputShape:
    @pytest.mark.asyncio
    async def test_new_fields_present_in_result(
        self, tmp_path, brief_path, mirdan_repo
    ):
        uc = VerifyPlanAgainstBriefUseCase(project_root=mirdan_repo)
        plan = tmp_path / "plan.md"
        plan.write_text(_build_plan())
        r = await uc.execute(str(plan), str(brief_path))
        for field in ("phantom_files", "dependency_errors", "vague_cross_references"):
            assert field in r, f"missing {field} in output"
            assert isinstance(r[field], list)
