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
