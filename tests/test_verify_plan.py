"""Tests for the brief-free mechanical plan verifier (``verify_plan``).

Re-homes the 2.1.0 mechanical-check evidence — phantom files, dependency errors,
vague cross-references, missing grounding — onto flat ``### Step N`` plans, plus a
determinism proof. All checks are deterministic and make no network/LLM calls.
"""

from __future__ import annotations

from pathlib import Path

from mirdan.usecases.verify_plan import VerifyPlanUseCase


def _write(root: Path, body: str) -> str:
    p = root / "plan.md"
    p.write_text(body)
    return str(p)


_CLEAN_PLAN = """# Plan

## Research Notes
### Files Verified
- real.py exists

## Plan Steps

### Step 1: edit the real file
**File:** `real.py`
**Action:** Edit
**Details:** change the value at line 1
**Depends On:** —
**Verify:** read it back
**Grounding:** Read real.py
"""


class TestCleanPlan:
    def test_clean_flat_plan_verifies(self, tmp_path: Path) -> None:
        (tmp_path / "real.py").write_text("x = 1\n")
        r = VerifyPlanUseCase(project_root=tmp_path).execute(_write(tmp_path, _CLEAN_PLAN))
        assert r["verified"] is True
        assert r["coverage_score"] == 1.0
        assert r["phantom_files"] == []
        assert r["dependency_errors"] == []
        assert r["vague_cross_references"] == []
        assert r["missing_grounding"] == []


class TestPhantomFiles:
    def test_detects_phantom_file(self, tmp_path: Path) -> None:
        body = (
            "# Plan\n## Research Notes\n### Files Verified\n- x\n\n## Plan Steps\n\n"
            "### Step 1: edit missing file\n**File:** `does/not/exist.py`\n"
            "**Action:** Edit\n**Details:** d\n**Depends On:** —\n"
            "**Verify:** v\n**Grounding:** g\n"
        )
        r = VerifyPlanUseCase(project_root=tmp_path).execute(_write(tmp_path, body))
        assert len(r["phantom_files"]) == 1
        assert "exist.py" in r["phantom_files"][0]["path"]
        assert r["verified"] is False

    def test_backtick_wrapped_real_file_not_flagged(self, tmp_path: Path) -> None:
        """Flat plans wrap File paths in backticks; a real file must not be phantom."""
        (tmp_path / "real.py").write_text("x = 1\n")
        r = VerifyPlanUseCase(project_root=tmp_path).execute(_write(tmp_path, _CLEAN_PLAN))
        assert r["phantom_files"] == []

    def test_new_file_with_existing_parent_ok(self, tmp_path: Path) -> None:
        (tmp_path / "pkg").mkdir()
        body = (
            "# Plan\n## Research Notes\n### Files Verified\n- x\n\n## Plan Steps\n\n"
            "### Step 1: create file\n**File:** NEW: `pkg/new.py`\n"
            "**Action:** Write\n**Details:** d\n**Depends On:** —\n"
            "**Verify:** v\n**Grounding:** Glob pkg/\n"
        )
        r = VerifyPlanUseCase(project_root=tmp_path).execute(_write(tmp_path, body))
        assert r["phantom_files"] == []

    def test_new_file_missing_parent_flagged(self, tmp_path: Path) -> None:
        body = (
            "# Plan\n## Research Notes\n### Files Verified\n- x\n\n## Plan Steps\n\n"
            "### Step 1: create file\n**File:** NEW: `nope/new.py`\n"
            "**Action:** Write\n**Details:** d\n**Depends On:** —\n"
            "**Verify:** v\n**Grounding:** Glob\n"
        )
        r = VerifyPlanUseCase(project_root=tmp_path).execute(_write(tmp_path, body))
        assert len(r["phantom_files"]) == 1


class TestDependencies:
    def test_dangling_dependency(self, tmp_path: Path) -> None:
        (tmp_path / "real.py").write_text("x\n")
        body = (
            "# Plan\n## Research Notes\n### Files Verified\n- x\n\n## Plan Steps\n\n"
            "### Step 1: a\n**File:** `real.py`\n**Action:** Edit\n**Details:** d\n"
            "**Depends On:** Step 9\n**Verify:** v\n**Grounding:** g\n"
        )
        r = VerifyPlanUseCase(project_root=tmp_path).execute(_write(tmp_path, body))
        assert len(r["dependency_errors"]) == 1
        assert r["dependency_errors"][0]["missing_dep"] == "9"

    def test_dependency_cycle(self, tmp_path: Path) -> None:
        (tmp_path / "real.py").write_text("x\n")
        body = (
            "# Plan\n## Research Notes\n### Files Verified\n- x\n\n## Plan Steps\n\n"
            "### Step 1: a\n**File:** `real.py`\n**Action:** Edit\n**Details:** d\n"
            "**Depends On:** Step 2\n**Verify:** v\n**Grounding:** g\n\n"
            "### Step 2: b\n**File:** `real.py`\n**Action:** Edit\n**Details:** d\n"
            "**Depends On:** Step 1\n**Verify:** v\n**Grounding:** g\n"
        )
        r = VerifyPlanUseCase(project_root=tmp_path).execute(_write(tmp_path, body))
        assert any("cycle" in e["issue"] for e in r["dependency_errors"])


class TestVagueAndGrounding:
    def test_vague_cross_reference(self, tmp_path: Path) -> None:
        (tmp_path / "real.py").write_text("x\n")
        body = (
            "# Plan\n## Research Notes\n### Files Verified\n- x\n\n## Plan Steps\n\n"
            "### Step 1: a\n**File:** `real.py`\n**Action:** Edit\n"
            "**Details:** do it as discussed\n**Depends On:** —\n"
            "**Verify:** v\n**Grounding:** g\n"
        )
        r = VerifyPlanUseCase(project_root=tmp_path).execute(_write(tmp_path, body))
        assert len(r["vague_cross_references"]) == 1

    def test_missing_grounding_field(self, tmp_path: Path) -> None:
        (tmp_path / "real.py").write_text("x\n")
        body = (
            "# Plan\n## Research Notes\n### Files Verified\n- x\n\n## Plan Steps\n\n"
            "### Step 1: a\n**File:** `real.py`\n**Action:** Edit\n**Details:** d\n"
            "**Depends On:** —\n**Verify:** v\n"  # no **Grounding:**
        )
        r = VerifyPlanUseCase(project_root=tmp_path).execute(_write(tmp_path, body))
        assert any("Grounding" in g["issue"] for g in r["missing_grounding"])
        assert r["verified"] is False


_LLD_HEADER = (
    "# Plan\n## Research Notes\n### Files Verified\n- ok\n\n"
    "## Low-Level Design\n### Interfaces & Signatures\n"
)
_LLD_STEPS = (
    "\n## Plan Steps\n\n### Step 1: implement login\n**File:** `real.py`\n"
    "**Action:** Edit\n**Details:** add the login handler\n**Depends On:** —\n"
    "**Verify:** v\n**Grounding:** g\n"
)


class TestLLDGaps:
    """Advisory (soft) LLD anti-hallucination checks. Never fail `verified`."""

    def test_no_lld_section_is_not_a_gap(self, tmp_path: Path) -> None:
        (tmp_path / "real.py").write_text("x\n")
        r = VerifyPlanUseCase(project_root=tmp_path).execute(_write(tmp_path, _CLEAN_PLAN))
        assert r["lld_gaps"] == []

    def test_clean_lld_has_no_gaps(self, tmp_path: Path) -> None:
        (tmp_path / "real.py").write_text("x\n")
        body = (
            _LLD_HEADER
            + "- `def login(req)` [NEW]\n"
            + "- `UserRepo.get(id)` [EXISTING] — src/db.py:42\n"
            + _LLD_STEPS
        )
        r = VerifyPlanUseCase(project_root=tmp_path).execute(_write(tmp_path, body))
        assert r["lld_gaps"] == []
        assert r["verified"] is True  # advisory check does not fail verification

    def test_existing_interface_without_citation_flagged(self, tmp_path: Path) -> None:
        (tmp_path / "real.py").write_text("x\n")
        body = _LLD_HEADER + "- `UserRepo.get(id)` [EXISTING]\n" + _LLD_STEPS
        r = VerifyPlanUseCase(project_root=tmp_path).execute(_write(tmp_path, body))
        assert any("[EXISTING]" in g["issue"] for g in r["lld_gaps"])
        # still verified — lld_gaps is advisory
        assert r["verified"] is True

    def test_new_interface_not_in_steps_flagged(self, tmp_path: Path) -> None:
        (tmp_path / "real.py").write_text("x\n")
        body = _LLD_HEADER + "- `def orphan_fn(x)` [NEW]\n" + _LLD_STEPS
        r = VerifyPlanUseCase(project_root=tmp_path).execute(_write(tmp_path, body))
        assert any(g["interface"] == "orphan_fn" for g in r["lld_gaps"])


class TestDeterminismAndErrors:
    def test_deterministic_across_runs(self, tmp_path: Path) -> None:
        (tmp_path / "real.py").write_text("x = 1\n")
        plan = _write(tmp_path, _CLEAN_PLAN)
        uc = VerifyPlanUseCase(project_root=tmp_path)
        results = [uc.execute(plan) for _ in range(5)]
        assert all(r == results[0] for r in results)

    def test_missing_plan_file_returns_error(self, tmp_path: Path) -> None:
        r = VerifyPlanUseCase(project_root=tmp_path).execute(str(tmp_path / "nope.md"))
        assert r["verified"] is False
        assert "error" in r
