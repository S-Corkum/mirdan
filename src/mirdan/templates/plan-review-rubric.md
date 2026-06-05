# Plan Review Rubric (shared by Claude Code and Cursor)

Judgment review of a **flat plan** (Research Notes → Low-Level Design → grounded
steps). This complements `/plan-verify`'s mechanical check — it covers what
judgment catches and mechanics can't.

Reviews MUST produce a markdown report with EXACTLY these 5 sections in this order.
Section headings are literal — do not rename, reorder, or add sections.

## design_gaps

Low-Level Design problems: a missing or unsound interface/signature, an incomplete
error taxonomy, an unclear contract or data shape, an interface marked `[EXISTING]`
without a `file:line` citation, or `[NEW]` created by no step.
Each entry: `- <interface or design element>: <the problem>`.
Empty → `- (none)`.

## grounding_gaps

Steps missing one or more grounding fields (File, Action, Details, Verify,
Grounding), or that reference a file / API / line not verified in Research Notes.
Each entry: `- <step id>: <what is missing or unverified>`.
Empty → `- (none)`.

## completeness_gaps

Pieces the plan needs but omits: tests for new behavior, import/export updates,
config changes, error handling, edge cases.
Each entry: `- <missing piece>: <where it should go>`.
Empty → `- (none)`.

## safety

Security and data-safety concerns the plan does not address: auth, input
validation, injection, secret handling, data migration / rollback.
Each entry: `- <concern>: <where in the plan>`.
Empty → `- (none)`.

## risks

Emergent / cross-step risks a reviewer identifies that mechanical verification
(`/plan-verify`) would miss — concurrency, ordering, integration, rollback,
security blind spots.
Each entry: `- <risk summary>: <rationale>`.
Empty → `- (none)`.

---

After the 5 sections, append a one-line verdict:

`**Verdict:** pass | fail | revise`

- `pass` — design_gaps, grounding_gaps, completeness_gaps, safety all empty; risks acceptable or absent
- `fail` — any of the first 4 sections is non-empty with blocking severity
- `revise` — issues exist but are addressable without replanning
