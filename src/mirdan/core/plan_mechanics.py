"""Mechanical plan checks — deterministic, no LLM, format-agnostic.

Extracted from the 2.1.0 brief verifier (which proved 100% detection / 0% false
positives / byte-deterministic on seeded defects) and decoupled from the brief.
These operate on a list of ``(step_id, step_body)`` tuples, so they work on flat
``### Step N`` plans as well as any other step-structured document.

Three checks:
- :func:`check_phantom_files` — a step's ``**File:**`` field points at a path that
  does not exist (for ``NEW:`` files, the parent directory is missing). The severest
  finding: an executor cannot operate on a path that does not exist.
- :func:`check_dependencies` — a ``**Depends on:**`` field references a step that is
  not in the plan, or the dependency graph has a cycle.
- :func:`check_vague_cross_references` — phrases like "as discussed" / "see above"
  that a downstream reader cannot resolve.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

# A step's File field: "**File:** path" or "**File:** NEW: path".
_FILE_FIELD_RE = re.compile(
    r"\*\*File:\*\*\s*(?P<label>NEW:\s*)?(?P<path>[^\s(]+)",
)
# A step's Depends-on field (flat plans use "Depends On"; case-insensitive).
_DEPENDS_FIELD_RE = re.compile(
    r"\*\*Depends\s+[Oo]n:?\*\*\s*(?P<deps>[^\n]+)",
    re.IGNORECASE,
)
# Tokens that mean "no dependencies".
_NO_DEPS = {"—", "-", "", "none", "n/a", "na"}

_VAGUE_CROSS_REF_PATTERNS: list[tuple[str, str]] = [
    (r"\bas\s+discussed\b", "'as discussed' — spell out what was discussed"),
    (r"\bas\s+mentioned\s+above\b", "'as mentioned above' — restate it inline"),
    (r"\bfrom\s+before\b", "'from before' — cite the step number"),
    (r"\bthe\s+function\s+from\s+earlier\b", "vague function reference"),
    (r"\bsee\s+above\b", "'see above' — restate inline"),
]

Step = tuple[str, str]


def check_phantom_files(steps: list[Step], project_root: Path) -> list[dict[str, str]]:
    """Verify each step's ``**File:**`` path exists (or, for ``NEW:``, its parent).

    Placeholder tokens from templates (e.g. ``path/to/file.py``, bare ``x``) are
    skipped: a real claim has at least one path separator and a file extension.
    """
    phantom: list[dict[str, str]] = []
    for step_id, body in steps:
        m = _FILE_FIELD_RE.search(body)
        if not m:
            continue
        # Flat plans wrap the path in backticks (`path`); strip markdown wrapping
        # and any trailing sentence punctuation without touching leading dotfiles.
        raw_path = m.group("path").strip().strip("`").strip("\"'").rstrip(".,;:")
        is_new = bool(m.group("label"))
        candidate = (project_root / raw_path).resolve()

        if is_new:
            parent = candidate.parent
            if not parent.exists():
                phantom.append(
                    {
                        "step_id": step_id,
                        "path": raw_path,
                        "issue": f"NEW: file — parent directory {parent} does not exist",
                    }
                )
        elif not candidate.exists():
            looks_like_real_path = "/" in raw_path and "." in raw_path.split("/")[-1]
            if looks_like_real_path:
                phantom.append(
                    {
                        "step_id": step_id,
                        "path": raw_path,
                        "issue": f"File referenced but not found at {candidate}",
                    }
                )
    return phantom


def check_dependencies(steps: list[Step]) -> list[dict[str, str]]:
    """Validate the ``**Depends on:**`` graph — no dangling refs, no cycles.

    Flat plans reference steps by number ("Step 2", "Steps 1, 2"); the step IDs
    produced by the flat extractor are those numbers as strings. Any integer in a
    Depends-on field that is not a step in the plan is a dangling reference.
    """
    errors: list[dict[str, str]] = []
    ids = {sid for sid, _ in steps}
    graph: dict[str, set[str]] = {}

    for step_id, body in steps:
        m = _DEPENDS_FIELD_RE.search(body)
        if not m:
            graph[step_id] = set()
            continue
        raw = m.group("deps").strip()
        if raw.lower() in _NO_DEPS:
            graph[step_id] = set()
            continue
        deps: set[str] = set()
        # Flat step references are bare integers ("Step 2" / "Steps 1, 2").
        for ref in re.findall(r"\d+", raw):
            if ref in ids:
                deps.add(ref)
            else:
                errors.append(
                    {
                        "step_id": step_id,
                        "missing_dep": ref,
                        "issue": f"depends on Step {ref} which is not a step in this plan",
                    }
                )
        graph[step_id] = deps

    errors.extend(_detect_cycles(graph))
    return errors


def _detect_cycles(graph: dict[str, set[str]]) -> list[dict[str, str]]:
    """DFS 3-coloring cycle detection over the dependency graph."""
    errors: list[dict[str, str]] = []
    white, gray, black = 0, 1, 2
    color = dict.fromkeys(graph, white)

    def visit(node: str, stack: list[str]) -> None:
        if color.get(node) == gray:
            cycle_start = stack.index(node) if node in stack else 0
            cycle = " → ".join([*stack[cycle_start:], node])
            errors.append(
                {
                    "step_id": node,
                    "missing_dep": "(cycle)",
                    "issue": f"dependency cycle: {cycle}",
                }
            )
            return
        if color.get(node) == black:
            return
        color[node] = gray
        for nxt in graph.get(node, ()):
            visit(nxt, [*stack, node])
        color[node] = black

    for n in graph:
        if color[n] == white:
            visit(n, [])
    return errors


def check_vague_cross_references(steps: list[Step]) -> list[dict[str, str]]:
    """Detect vague cross-references a downstream reader can't resolve."""
    findings: list[dict[str, str]] = []
    for step_id, body in steps:
        for pattern, msg in _VAGUE_CROSS_REF_PATTERNS:
            if re.search(pattern, body, re.IGNORECASE):
                findings.append({"step_id": step_id, "issue": msg})
    return findings


_LLD_HEADING_RE = re.compile(r"^##\s+Low-?Level Design\s*$", re.MULTILINE | re.IGNORECASE)
_NEXT_H2_RE = re.compile(r"^##\s+\S", re.MULTILINE)
_INTERFACE_TAG_RE = re.compile(r"\[(NEW|EXISTING)\]")
# A real file citation: a path (has a "/") with an extension, or a "file.ext:line".
# A bare dotted name like "UserRepo.get" is NOT a citation.
_CITATION_RE = re.compile(r"/[\w.-]+\.[A-Za-z]{1,5}\b|[\w.-]+\.[A-Za-z]{1,5}:\d+")
# A symbol name inside a backtick span, skipping a leading def/class keyword.
_NEW_SYMBOL_RE = re.compile(r"`(?:def |class |async def )?([A-Za-z_][\w.]*)")


def check_lld_gaps(plan_text: str, steps: list[Step]) -> list[dict[str, str]]:
    """Advisory anti-hallucination check for the Low-Level Design section.

    Soft (warn-only): absence of an LLD section is not a gap. When present:
    - an interface marked ``[EXISTING]`` must carry a ``file:line`` / path citation
      (else it may be a fabricated signature), and
    - an interface marked ``[NEW]`` should be created by some step (its symbol name
      must appear in a step body), else it is ungrounded.
    """
    m = _LLD_HEADING_RE.search(plan_text)
    if not m:
        return []
    nxt = _NEXT_H2_RE.search(plan_text, m.end())
    section = plan_text[m.end() : nxt.start() if nxt else len(plan_text)]
    steps_text = "\n".join(body for _, body in steps)

    gaps: list[dict[str, str]] = []
    for line in section.splitlines():
        tag = _INTERFACE_TAG_RE.search(line)
        if not tag:
            continue
        snippet = line.strip()[:80]
        if tag.group(1) == "EXISTING":
            if not _CITATION_RE.search(line):
                gaps.append(
                    {
                        "interface": snippet,
                        "issue": "[EXISTING] interface without a file:line citation",
                    }
                )
        else:  # NEW
            sym = _NEW_SYMBOL_RE.search(line)
            if sym and sym.group(1) not in steps_text:
                gaps.append(
                    {
                        "interface": sym.group(1),
                        "issue": "[NEW] interface not created by any step",
                    }
                )
    return gaps


# ---------------------------------------------------------------------------
# Haiku-proof checks (format_version 2). Deterministic, no LLM.
#
# On v1 / frontmatter-less plans the caller treats these as soft (reported, not
# scored), so the historical corpus keeps passing. On v2 plans they are hard.
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\n(?P<body>.*?)\n---", re.DOTALL)


def read_format_version(plan_text: str) -> int:
    """Return the plan's declared ``format_version`` (frontmatter), defaulting to 1.

    Any failure (no frontmatter, malformed YAML, non-int value) returns 1 — the
    v1 / soft path — so a missing or broken stamp never hard-fails a legacy plan.
    """
    m = _FRONTMATTER_RE.match(plan_text)
    if not m:
        return 1
    try:
        meta = yaml.safe_load(m.group("body"))
    except Exception:  # any YAML / parse error means "treat as v1"
        return 1
    if not isinstance(meta, dict):
        return 1
    try:
        return int(meta.get("format_version", 1))
    except (TypeError, ValueError):
        return 1


# A step is "capable-targeted" when its Action line carries a [target: capable]
# tag — judgment / un-anchorable / wholesale edits, exempt from anchor + atomicity
# enforcement (and reported so a haiku-target reader knows it is not cold-executable).
_CAPABLE_TAG_RE = re.compile(r"\*\*Action:\*\*[^\n]*\[target:\s*capable\]", re.IGNORECASE)


def step_is_capable(step_body: str) -> bool:
    """True if the step's Action line is tagged ``[target: capable]``."""
    return bool(_CAPABLE_TAG_RE.search(step_body))


_ACTION_EDIT_RE = re.compile(r"\*\*Action:\*\*\s*Edit\b", re.IGNORECASE)
_ANCHOR_BLOCK_RE = re.compile(r"```anchor\b(?P<body>.*?)```", re.DOTALL)
_REPLACE_BLOCK_RE = re.compile(r"```replace\b(?P<body>.*?)```", re.DOTALL)
_FENCED_SPAN_RE = re.compile(r"```.*?```", re.DOTALL)


def check_edit_anchors(steps: list[Step]) -> list[dict[str, str]]:
    """Every non-capable ``Action: Edit`` step must carry one literal find/replace.

    A cold Haiku executor cannot compute line numbers; it can only do an
    unambiguous literal find-and-replace. So each Edit step needs exactly one
    non-empty ```anchor``` block (the verbatim existing text) and one ```replace```
    block. Steps tagged ``[target: capable]`` are exempt.
    """
    findings: list[dict[str, str]] = []
    for step_id, body in steps:
        if not _ACTION_EDIT_RE.search(body) or step_is_capable(body):
            continue
        anchors = _ANCHOR_BLOCK_RE.findall(body)
        replaces = _REPLACE_BLOCK_RE.findall(body)
        if len(anchors) != 1 or len(replaces) != 1:
            findings.append(
                {
                    "step_id": step_id,
                    "issue": "Edit step needs exactly one ```anchor``` and one "
                    "```replace``` block (or mark the step [target: capable])",
                }
            )
        elif not anchors[0].strip():
            findings.append(
                {
                    "step_id": step_id,
                    "issue": "```anchor``` block is empty — copy the verbatim existing text",
                }
            )
    return findings


def check_anchor_uniqueness(steps: list[Step], project_root: Path) -> list[dict[str, str]]:
    """Each anchored Edit's ```anchor``` text must occur exactly once in its file.

    First-occurrence find-and-replace is only safe when the anchor is unique; a
    non-unique (or absent) anchor would land the edit in the wrong place. Reads the
    target file. ``NEW:`` / missing files are skipped (check_phantom_files covers them),
    as are capable and presence-failing steps (check_edit_anchors covers those).
    """
    findings: list[dict[str, str]] = []
    for step_id, body in steps:
        if not _ACTION_EDIT_RE.search(body) or step_is_capable(body):
            continue
        anchors = _ANCHOR_BLOCK_RE.findall(body)
        if len(anchors) != 1 or not anchors[0].strip():
            continue
        fm = _FILE_FIELD_RE.search(body)
        if not fm or fm.group("label"):  # no File field, or a NEW: file
            continue
        raw_path = fm.group("path").strip().strip("`").strip("\"'").rstrip(".,;:")
        target = (project_root / raw_path).resolve()
        if not target.exists():
            continue
        try:
            content = target.read_text()
        except OSError:
            continue
        anchor_text = anchors[0].strip("\n")
        count = content.count(anchor_text)
        if count == 0:
            findings.append(
                {
                    "step_id": step_id,
                    "path": raw_path,
                    "issue": "anchor text not found verbatim in the target file",
                }
            )
        elif count > 1:
            findings.append(
                {
                    "step_id": step_id,
                    "path": raw_path,
                    "issue": f"anchor text matches {count} places — make the anchor unique",
                }
            )
    return findings


_ACTION_CONJ_RE = re.compile(r"\*\*Action:\*\*[^\n]*\b(?:and|then|also|plus)\b", re.IGNORECASE)
_DETAILS_FIELD_RE = re.compile(
    r"\*\*Details:\*\*(?P<body>.*?)"
    r"(?=\n\*\*(?:Depends\s+[Oo]n|Verify|Grounding|Action|File):|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_DETAIL_COMPOUND_RES = [
    re.compile(r"\band\s+then\b", re.IGNORECASE),
    re.compile(r"\bthen\s+also\b", re.IGNORECASE),
    re.compile(r"\bfirst\b[^.\n]{0,60}\bthen\b", re.IGNORECASE),
]


def _details_body(step_body: str) -> str:
    m = _DETAILS_FIELD_RE.search(step_body)
    return m.group("body") if m else ""


def check_atomicity(steps: list[Step]) -> list[dict[str, str]]:
    """Flag non-atomic steps (more than one file or action). Capable steps exempt.

    Low-false-positive by construction:
    - more than one ``**File:**`` target in a step;
    - a conjunction verb in the ``**Action:**`` field ("Edit and add");
    - a compound connective ("and then" / "first…then") in the Details prose —
      fenced ```anchor```/```replace``` code is stripped first, so real code
      containing those words does not trip.
    """
    findings: list[dict[str, str]] = []
    for step_id, body in steps:
        if step_is_capable(body):
            continue
        if len(re.findall(r"\*\*File:\*\*", body)) > 1:
            findings.append(
                {
                    "step_id": step_id,
                    "issue": "multiple **File:** targets — one file per step "
                    "(or mark [target: capable])",
                }
            )
        if _ACTION_CONJ_RE.search(body):
            findings.append(
                {"step_id": step_id, "issue": "compound Action verb — one action per step"}
            )
        details = _FENCED_SPAN_RE.sub(" ", _details_body(body))
        if any(rgx.search(details) for rgx in _DETAIL_COMPOUND_RES):
            findings.append(
                {
                    "step_id": step_id,
                    "issue": "compound Details ('and then' / 'first…then') — "
                    "split into atomic steps",
                }
            )
    return findings


_DESIGN_DECISIONS_RE = re.compile(r"^#{2,4}\s+Design Decisions\s*$", re.MULTILINE | re.IGNORECASE)
_ANY_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_DQUOTED_RE = re.compile(r"\"[^\"\n]*\"")
_NEGATION_PREFIX_RE = re.compile(
    r"\b(?:no|not|without|resolve[ds]?|avoid|never)\b[^.\n]{0,12}$", re.IGNORECASE
)
_UNRESOLVED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bTBD\b"), "unresolved marker 'TBD'"),
    (re.compile(r"\bTODO\b"), "unresolved marker 'TODO'"),
    (re.compile(r"\bFIXME\b"), "unresolved marker 'FIXME'"),
    (re.compile(r"\bTBC\b"), "unresolved marker 'TBC'"),
    (re.compile(r"\?\?\?"), "unresolved marker '???'"),
    (
        re.compile(r"\bdecide\s+(?:at|during|in)\s+(?:review|impl\w*)\b", re.IGNORECASE),
        "decision deferred ('decide at review/impl')",
    ),
    (re.compile(r"\beither\b[^.\n]{0,40}\bor\b", re.IGNORECASE), "unresolved either/or"),
    (
        re.compile(r"\boption\s+[A-Z]\b[^.\n]{0,40}\bor\b", re.IGNORECASE),
        "unresolved 'option A or B'",
    ),
    (
        re.compile(r"\bif\s+\w+[^.\n]{0,40}\bthen\b[^.\n]{0,40}\belse\b", re.IGNORECASE),
        "conditional 'if…then…else' — pre-resolve the decision",
    ),
]


def _mask_prose(text: str) -> str:
    """Blank out fenced code, inline code, and double-quoted spans so that
    quoted/illustrative marker mentions do not false-positive."""
    text = _FENCED_SPAN_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    return _DQUOTED_RE.sub(" ", text)


def check_unresolved_decisions(plan_text: str, steps: list[Step]) -> list[dict[str, str]]:
    """Flag decisions left for the executor (TBD / either-or / 'decide later').

    A cold Haiku executor must not have to *choose* — an unresolved decision forces
    invention. Scans only the ``Design Decisions`` section and each step's Details
    (never Verify/Grounding). Markers inside code spans or double quotes, or
    immediately preceded by a negation ("no TBD"), are ignored.
    """
    segments: list[str] = []
    dd = _DESIGN_DECISIONS_RE.search(plan_text)
    if dd:
        nxt = _ANY_HEADING_RE.search(plan_text, dd.end())
        segments.append(plan_text[dd.end() : nxt.start() if nxt else len(plan_text)])
    segments.extend(_details_body(body) for _sid, body in steps)

    findings: list[dict[str, str]] = []
    seen: set[str] = set()
    for seg in segments:
        masked = _mask_prose(seg)
        for rgx, msg in _UNRESOLVED_PATTERNS:
            if msg in seen:
                continue
            for m in rgx.finditer(masked):
                line_prefix = masked[: m.start()].rsplit("\n", 1)[-1]
                if _NEGATION_PREFIX_RE.search(line_prefix):
                    continue
                seen.add(msg)
                findings.append({"issue": msg})
                break
    return findings
