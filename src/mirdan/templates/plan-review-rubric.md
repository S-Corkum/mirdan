# Plan Review Rubric (shared by Claude Code and Cursor)

Reviews MUST produce a markdown report with EXACTLY these 5 sections in this order.
Section headings are literal — do not rename, reorder, or add sections.

## unmapped_acs

List brief Business ACs that no story AC in the plan covers.
Empty → `- (none)`.

## constraint_violations

List brief Constraints the plan violates or fails to honor.
Each entry: `- <constraint name>: <what the plan does that violates it>`.
Empty → `- (none)`.

## scope_violations

List items the plan does that are in the brief's Out of Scope section.
Each entry: `- <excluded item>: <where it appears in the plan>`.
Empty → `- (none)`.

## grounding_gaps

List subtasks missing one or more of the 6 grounding fields (File, Action,
Details, Depends on, Verify, Grounding).
Each entry: `- <subtask id>: missing [<comma-separated field names>]`.
Empty → `- (none)`.

## risks

List emergent / cross-story risks the reviewer identifies that mechanical
verification would miss (concurrency, rollback, ordering, integration,
security blind spots).
Each entry: `- <risk summary>: <rationale>`.
Empty → `- (none)`.

---

After the 5 sections, append a one-line verdict:

`**Verdict:** pass | fail | revise`

- `pass` — all first 4 sections empty; risks are acceptable or absent
- `fail` — any of the first 4 sections is non-empty with blocking severity
- `revise` — issues exist but are addressable without replanning
