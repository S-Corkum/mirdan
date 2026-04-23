---
plan: [REPLACE: slug]
brief: [REPLACE: docs/briefs/slug.md]
status: draft
author: [REPLACE: author]
created: [REPLACE: YYYY-MM-DD]
target_version: [REPLACE: version]
target_executor: [REPLACE: haiku | sonnet | local]
---

# Plan: [REPLACE: title]

This plan implements [REPLACE: docs/briefs/slug.md]. Every subtask is
self-contained — executors do not need to read the brief to execute any
individual subtask.

## Research Notes (Pre-Plan Verification)

All facts below verified via Read/Grep/tool-call on [REPLACE: YYYY-MM-DD].

### Files Verified

- `path/to/file.py`: [REPLACE: key structures found, line numbers]

### Dependencies Confirmed

- `library-name`: version X.Y.Z (from pyproject.toml line N)

### API Documentation

- `library.method()`: [REPLACE: verified signature]

### Conventions

- `pattern-name`: [REPLACE: how this project does X — cite enyal entry if from memory]

---

## Epic Layer

**Outcome:** [REPLACE: single-sentence outcome tied to brief metric]

**Metric:** [REPLACE: observable metric from brief]

**Scope boundary:** [REPLACE: what's explicitly in and out]

**Epic grounding:**
- Problem evidence: [REPLACE: tool citation]
- Existing infrastructure: [REPLACE: verified module paths]
- Constraining decisions: [REPLACE: enyal entry IDs]

---

## Story Layer

<!-- Each story decomposes the epic into shippable slices. Use INVEST check. -->

### Story 1 — [REPLACE: title]

- **As** [REPLACE: persona]
- **I want** [REPLACE: capability]
- **So that** [REPLACE: value]

**Acceptance Criteria:**
- [ ] [REPLACE: testable criterion]
- [ ] [REPLACE: testable criterion]

**INVEST:** I✓ N✓ V✓ E✓ S✓ T✓

#### Subtasks

##### 1.1 — [REPLACE: action title]

**File:** [REPLACE: exact/path.py (verified via Read YYYY-MM-DD) OR NEW: path (parent Glob-confirmed)]

**Action:** [REPLACE: Edit | Write | Bash | specific MCP tool]

**Details:** [REPLACE: specific changes — line numbers, function names, exact
modifications. "Add validation" is bad; "Add validate_token(token) after line
45 calling jwt.decode" is good.]

**Depends on:** [REPLACE: subtask IDs, or —]

**Verify:** [REPLACE: binary tool-based check — "Read file, confirm function at line X"]

**Grounding:** [REPLACE: which tool call verified each fact]
- File exists: Read on YYYY-MM-DD
- API signature: context7 query for [topic]
- Pattern: Read of similar implementation in [file]

**If blocked:** [REPLACE: halt instruction — do NOT guess. E.g., "If line 45
is empty, STOP and report: grounding assumed function at line 45."]

<!-- Add more subtasks: 1.2, 1.3, ... -->

<!-- Add more stories: ### Story 2 — ... -->

---

## Plan Self-Check

- [ ] Research Notes present with tool citations
- [ ] Every subtask has File, Action, Details, Depends on, Verify, Grounding, If-blocked
- [ ] No vague language ("should", "probably", "maybe")
- [ ] Every file path verified or marked NEW with parent Glob
- [ ] Subtasks are atomic — one action each
- [ ] Self-contained: executor doesn't need epic/brief context

## Execution Order

```
Story 1 → Story 2 → ...
```
