"""Tests for the Haiku-proof mechanical checks (format_version 2).

Deterministic, no LLM. Covers the new checks plus their false-positive guards
(capable-step exemption, fenced/quoted/negated masking).
"""

from __future__ import annotations

from pathlib import Path

from mirdan.core import plan_mechanics as pm

_ANCHORED_EDIT = (
    "**File:** `x.py`\n**Action:** Edit\n**Details:** swap it\n"
    "```anchor\nold = 1\n```\n```replace\nold = 2\n```\n"
)


def _step(details: str) -> list[pm.Step]:
    return [("1", f"**File:** `a.py`\n**Action:** Edit\n**Details:** {details}\n**Verify:** v\n")]


# -- read_format_version -----------------------------------------------------


def test_format_version_absent_defaults_to_1() -> None:
    assert pm.read_format_version("# Plan\nno frontmatter\n") == 1


def test_format_version_v2() -> None:
    assert pm.read_format_version("---\nformat_version: 2\nplan: x\n---\n# Plan\n") == 2


def test_format_version_malformed_defaults_to_1() -> None:
    assert pm.read_format_version("---\n:\t: bad: yaml: :\n---\n") == 1


def test_format_version_non_int_defaults_to_1() -> None:
    assert pm.read_format_version("---\nformat_version: latest\n---\n") == 1


# -- step_is_capable ---------------------------------------------------------


def test_step_is_capable_true() -> None:
    assert pm.step_is_capable("**Action:** Edit  **[target: capable]**\n") is True


def test_step_is_capable_false() -> None:
    assert pm.step_is_capable("**Action:** Edit\n") is False


# -- check_edit_anchors ------------------------------------------------------


def test_edit_without_anchor_flagged() -> None:
    out = pm.check_edit_anchors([("1", "**File:** `x.py`\n**Action:** Edit\n**Details:** d\n")])
    assert len(out) == 1


def test_anchored_edit_ok() -> None:
    assert pm.check_edit_anchors([("1", _ANCHORED_EDIT)]) == []


def test_capable_edit_without_anchor_exempt() -> None:
    body = "**File:** `x.py`\n**Action:** Edit  **[target: capable]**\n**Details:** rewrite\n"
    assert pm.check_edit_anchors([("1", body)]) == []


def test_empty_anchor_flagged() -> None:
    body = "**File:** `x.py`\n**Action:** Edit\n**Details:** d\n```anchor\n```\n```replace\ny\n```\n"
    out = pm.check_edit_anchors([("1", body)])
    assert len(out) == 1
    assert "empty" in out[0]["issue"]


def test_write_action_needs_no_anchor() -> None:
    body = "**File:** NEW: `x.py`\n**Action:** Write\n**Details:** whole file\n"
    assert pm.check_edit_anchors([("1", body)]) == []


# -- check_anchor_uniqueness -------------------------------------------------


def test_anchor_unique_ok(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("a = 1\nold = 1\nb = 2\n")
    assert pm.check_anchor_uniqueness([("1", _ANCHORED_EDIT)], tmp_path) == []


def test_anchor_nonunique_flagged(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("old = 1\nold = 1\n")
    out = pm.check_anchor_uniqueness([("1", _ANCHORED_EDIT)], tmp_path)
    assert len(out) == 1
    assert "unique" in out[0]["issue"]


def test_anchor_not_found_flagged(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("totally different\n")
    out = pm.check_anchor_uniqueness([("1", _ANCHORED_EDIT)], tmp_path)
    assert len(out) == 1
    assert "not found" in out[0]["issue"]


# -- check_atomicity ---------------------------------------------------------


def test_atomicity_multiple_files() -> None:
    body = "**File:** `a.py`\n**Action:** Edit\n**Details:** d\n**File:** `b.py`\n"
    assert any("File" in f["issue"] for f in pm.check_atomicity([("1", body)]))


def test_atomicity_and_then_in_details() -> None:
    body = "**File:** `a.py`\n**Action:** Edit\n**Details:** add x and then add y\n**Verify:** v\n"
    assert any("compound Details" in f["issue"] for f in pm.check_atomicity([("1", body)]))


def test_atomicity_clean_step_ok() -> None:
    body = "**File:** `a.py`\n**Action:** Edit\n**Details:** add the field\n**Verify:** v\n"
    assert pm.check_atomicity([("1", body)]) == []


def test_atomicity_ignores_and_then_inside_anchor() -> None:
    body = (
        "**File:** `a.py`\n**Action:** Edit\n**Details:** swap\n"
        "```anchor\nx = 1  # first do this and then that\n```\n```replace\nx = 2\n```\n"
        "**Verify:** v\n"
    )
    assert pm.check_atomicity([("1", body)]) == []


def test_atomicity_capable_exempt() -> None:
    body = "**File:** `a.py`\n**Action:** Edit  **[target: capable]**\n**Details:** add x and then y\n"
    assert pm.check_atomicity([("1", body)]) == []


# -- check_unresolved_decisions ----------------------------------------------


def test_unresolved_tbd_flagged() -> None:
    out = pm.check_unresolved_decisions("# Plan\n", _step("the format is TBD"))
    assert any("TBD" in f["issue"] for f in out)


def test_unresolved_either_or_flagged() -> None:
    out = pm.check_unresolved_decisions("# Plan\n", _step("use either redis or memcached"))
    assert any("either/or" in f["issue"] for f in out)


def test_unresolved_quoted_ignored() -> None:
    assert pm.check_unresolved_decisions("# Plan\n", _step('the rule bans "TBD" markers')) == []


def test_unresolved_negated_ignored() -> None:
    assert pm.check_unresolved_decisions("# Plan\n", _step("ensure no TBD remains")) == []


def test_unresolved_fenced_either_or_ignored() -> None:
    details = "swap\n```anchor\n# use either a or b here\n```\n```replace\nx = 2\n```"
    assert pm.check_unresolved_decisions("# Plan\n", _step(details)) == []


def test_unresolved_clean_ok() -> None:
    assert pm.check_unresolved_decisions("# Plan\n", _step("add the cache field")) == []
