# Enyal Deep Integration — Phase 1: Orchestrator + Templates

**Status:** READY FOR IMPLEMENTATION
**Created:** 2026-03-10
**Scope:** Low effort, high impact changes to surface enyal's graph capabilities through mirdan

## Research Notes (Pre-Plan Verification)

### Files Verified
- `src/mirdan/core/orchestrator.py` (320 lines): MCPOrchestrator with `suggest_tools()` (lines 42-195) and `suggest_tools_for_planning()` (lines 225-315). KNOWN_MCPS at lines 21-40. Enyal capabilities registered as `["project_context", "decisions", "conventions"]`. Only `enyal_recall` recommended currently.
- `src/mirdan/server.py` (~1180 lines): 6 MCP tools. `validate_code_quality` builds output dict at lines 577-663. `knowledge_entries` added at lines 634-637 via `_process_knowledge_entries()`. `auto_store` flag exists but no storage guidance text.
- `src/mirdan/models.py` (~660 lines): `ToolRecommendation` has `mcp`, `action`, `priority`, `params`, `reason` fields. `KnowledgeEntry` has `content`, `content_type`, `tags`, `scope`, `scope_path`, `confidence`. No model changes needed.
- `src/mirdan/core/knowledge_producer.py` (234 lines): Extracts entries, returns them. Docstring: "Does NOT call enyal directly — loose coupling by design."
- `src/mirdan/core/gatherers/enyal.py` (201 lines): EnyalGatherer with task-specific queries. Uses `file_path` for scope-aware recall. No tag/content_type filtering (Phase 2).

### Templates Verified
- `/code` SKILL.md: Has `enyal_recall` + `enyal_remember` in allowed-tools
- `/debug` SKILL.md: Has `enyal_recall` + `enyal_remember` in allowed-tools
- `/scan` SKILL.md: Has `enyal_recall` + `enyal_remember` in allowed-tools
- `/review` SKILL.md: **Missing enyal entirely**
- `/quality` SKILL.md: **Missing enyal entirely**
- `/gate` SKILL.md: **Missing enyal entirely**
- `/plan` SKILL.md: **Missing enyal entirely** (uses `context: fork`)
- `mirdan-quality.md`: References enyal only in passing ("call context7, sequential-thinking, enyal as suggested")
- `mirdan-workflow.md`: No enyal section at all

### Tests Verified
- `tests/test_orchestrator.py`: 18 tests, 100% coverage on orchestrator module
- `tests/test_knowledge_producer.py`: 20 tests covering all extraction types
- Total test baseline: 2390 tests passing

### Conventions (enyal)
- Session-aware logic lives in `server.py`, not validators (per 2026-03-06 agentic gap decision)
- Zero new dependencies policy
- All changes must pass ruff + mypy
- Existing recommendation pattern: `ToolRecommendation(mcp=, action=, priority=, params=, reason=)`

---

## Implementation Steps

### Step 1: Update KNOWN_MCPS enyal capabilities

**File:** `src/mirdan/core/orchestrator.py` (verified via Read)
**Action:** Edit lines 34-36
**Details:**
Update the enyal capabilities list to reflect its full feature set:
```python
"enyal": {
    "capabilities": [
        "project_context", "decisions", "conventions",
        "knowledge_graph", "impact_analysis", "knowledge_maintenance",
    ],
},
```
**Depends On:** None
**Verify:** Read file, confirm capabilities list updated. Run `test_contains_all_known_mcps` — passes (checks key presence, not values).
**Grounding:** Read of orchestrator.py lines 21-40; test_orchestrator.py line 310-318

---

### Step 2: Add enyal graph recommendations for REFACTOR tasks

**File:** `src/mirdan/core/orchestrator.py` (verified via Read)
**Action:** Edit — add new block after line 99 (after existing enyal recall block, before sequential-thinking block at line 101)
**Details:**
Add a new conditional block for graph-aware enyal recommendations:
```python
# Enyal graph operations for architecture-aware tasks
if "enyal" in available_mcps and intent.task_type == TaskType.REFACTOR:
    recommendations.append(
        ToolRecommendation(
            mcp="enyal",
            action=(
                "Traverse knowledge graph around the refactored area "
                "to understand related conventions and dependencies"
            ),
            priority="high",
            params={"tool": "enyal_traverse", "max_depth": 2},
            reason="Refactoring may break assumptions stored in the knowledge graph",
        )
    )
    recommendations.append(
        ToolRecommendation(
            mcp="enyal",
            action=(
                "Check impact of changing existing patterns — "
                "find entries that depend on current conventions"
            ),
            priority="high",
            params={"tool": "enyal_impact"},
            reason="Understand what depends on patterns being refactored",
        )
    )
```
**Depends On:** None
**Verify:** Read file, confirm new block exists after existing enyal recall and before sequential-thinking. Run new tests (Step 12).
**Grounding:** Read of orchestrator.py lines 72-138; enyal traverse/impact tool signatures from exploration

---

### Step 3: Add enyal_remember post-task recommendation

**File:** `src/mirdan/core/orchestrator.py` (verified via Read)
**Action:** Edit — add new block after the Step 2 block, before sequential-thinking block
**Details:**
Add a post-task remember recommendation for first-call scenarios:
```python
# Post-task knowledge persistence (first call only)
if "enyal" in available_mcps and not (
    session and session.validation_count > 0
):
    recommendations.append(
        ToolRecommendation(
            mcp="enyal",
            action=(
                "After completing the task, store decisions and patterns "
                "via enyal_remember with appropriate tags and scope"
            ),
            priority="medium",
            params={"tool": "enyal_remember", "when": "after_completion"},
            reason="Persist insights for future sessions",
        )
    )
```
Condition: Only on first enhance_prompt call (no prior validations). Skipped on re-enhancement cycles.
**Depends On:** None
**Verify:** Read file, confirm block added. Run new tests (Step 12).
**Grounding:** Read of orchestrator.py lines 72-99 (existing session-aware logic pattern); SessionContext.validation_count field at models.py line 102

---

### Step 4: Add enyal_traverse to suggest_tools_for_planning()

**File:** `src/mirdan/core/orchestrator.py` (verified via Read)
**Action:** Edit — add new recommendation after the existing enyal recall block (after line 267, before filesystem block at line 269)
**Details:**
```python
        # Traverse knowledge graph for full architecture context
        if "enyal" in available_mcps:
            recommendations.append(
                ToolRecommendation(
                    mcp="enyal",
                    action=(
                        "Traverse knowledge graph around the planned area "
                        "to discover related decisions and dependencies"
                    ),
                    priority="high",
                    params={"tool": "enyal_traverse", "max_depth": 2},
                    reason="Plans must account for existing architecture decisions and their dependencies",
                )
            )
```
**Depends On:** None
**Verify:** Read file, confirm block added in suggest_tools_for_planning. Run new tests (Step 12).
**Grounding:** Read of orchestrator.py lines 225-315; existing PLANNING enyal recall at lines 257-267

---

### Step 5: Add knowledge_storage_hint to validate_code_quality

**File:** `src/mirdan/server.py` (verified via Read)
**Action:** Edit — add after line 637 (`output["knowledge_entries"] = ...`), inside the existing `if knowledge_entries:` block
**Details:**
```python
    if not c.config.orchestration.auto_memory:
        output["knowledge_storage_hint"] = (
            "Store entries marked auto_store=true via enyal_remember. "
            "For convention entries, use suggest_supersedes=true to detect "
            "and link superseded decisions. Tags and scope are pre-set."
        )
```
The `auto_memory` guard ensures the hint only appears when the agent needs to manually store (not when server-side auto-storage is enabled).
**Depends On:** None
**Verify:** Read file, confirm hint added inside `if knowledge_entries:` block with auto_memory guard. Run new test (Step 13).
**Grounding:** Read of server.py lines 634-637; _process_knowledge_entries at lines 1132-1162; OrchestrationConfig.auto_memory at config.py line 71-74

---

### Step 6: Update /review SKILL.md — add enyal

**File:** `src/mirdan/integrations/templates/claude_code/skills/review/SKILL.md` (verified via Read)
**Action:** Edit
**Details:**
1. Add `mcp__enyal__enyal_recall` to allowed-tools line (line 8)
2. Add workflow step after "Read" (before "Validate"):
```
1.5. **Recall** — Call `mcp__enyal__enyal_recall` with the file path to get project code conventions
   - Use `file_path` parameter for scope-weighted results
   - Use `content_type="convention"` to filter to conventions only
```
**Depends On:** None
**Verify:** Read file, confirm allowed-tools includes enyal_recall and workflow includes Recall step.
**Grounding:** Read of SKILL.md (8 lines allowed-tools, 42 lines total)

---

### Step 7: Update /code SKILL.md — add enyal_traverse

**File:** `src/mirdan/integrations/templates/claude_code/skills/code/SKILL.md` (verified via Read)
**Action:** Edit
**Details:**
Add `mcp__enyal__enyal_traverse` to allowed-tools line (line 6). The /code skill already has enyal_recall and enyal_remember — traverse enables the agent to follow orchestrator recommendations for REFACTOR tasks routed through /code.
**Depends On:** None
**Verify:** Read file, confirm allowed-tools includes enyal_traverse.
**Grounding:** Read of SKILL.md line 6 (current allowed-tools list)

---

### Step 8: Update /quality SKILL.md — add enyal

**File:** `src/mirdan/integrations/templates/claude_code/skills/quality/SKILL.md` (verified via Read)
**Action:** Edit
**Details:**
1. Add `mcp__enyal__enyal_recall` to allowed-tools line (line 7)
2. Add workflow step after "Identify" (before "Read"):
```
1.5. **Conventions** — Call `mcp__enyal__enyal_recall("quality conventions")` to load project quality standards for comparison
```
**Depends On:** None
**Verify:** Read file, confirm allowed-tools includes enyal_recall and workflow includes Conventions step.
**Grounding:** Read of SKILL.md (7 lines allowed-tools, 35 lines total)

---

### Step 9: Update /gate SKILL.md — add enyal

**File:** `src/mirdan/integrations/templates/claude_code/skills/gate/SKILL.md` (verified via Read)
**Action:** Edit
**Details:**
1. Add `mcp__enyal__enyal_recall, mcp__enyal__enyal_remember` to allowed-tools (line 6)
2. Add workflow step after "Re-gate" (before "Complete"):
```
5.5. **Persist** — If validation produced `knowledge_entries`, store them via `mcp__enyal__enyal_remember` with the suggested tags and scope
```
**Depends On:** None
**Verify:** Read file, confirm allowed-tools includes both enyal tools and workflow includes Persist step.
**Grounding:** Read of SKILL.md (6 lines allowed-tools, 35 lines total)

---

### Step 10: Update /plan SKILL.md — add enyal with traverse

**File:** `src/mirdan/integrations/templates/claude_code/skills/plan/SKILL.md` (verified via Read)
**Action:** Edit
**Details:**
1. Add `mcp__enyal__enyal_recall, mcp__enyal__enyal_remember, mcp__enyal__enyal_traverse` to allowed-tools (line 7)
2. Add workflow step after "Enhance" (before "Think"):
```
1.5. **Context** — Call `mcp__enyal__enyal_recall("architecture conventions decisions")` to load project context, then `mcp__enyal__enyal_traverse` with the planned area to discover related decisions and dependencies
```
3. Add to step 6 "Review":
```
   - Architecture decisions from enyal are respected
```
**Depends On:** None
**Verify:** Read file, confirm allowed-tools includes all 3 enyal tools and workflow includes Context step.
**Grounding:** Read of SKILL.md (7 lines allowed-tools, 47 lines total); /plan uses `context: fork` so enyal is the ONLY way to get project memory

---

### Step 11: Enhance mirdan-quality.md — Knowledge Persistence section

**File:** `src/mirdan/integrations/templates/claude_code/mirdan-quality.md` (verified via Read)
**Action:** Edit — add new section after "Reading Validation Output" (after line 45)
**Details:**
Add:
```markdown

## Knowledge Persistence

When `validate_code_quality` returns `knowledge_entries` with `auto_store: true`:
- Call `enyal_remember` for each entry — tags, scope, and confidence are pre-set
- For convention entries, use `suggest_supersedes=true` to detect and link entries this supersedes
- Deduplication and conflict detection are on by default in enyal
```
**Depends On:** None
**Verify:** Read file, confirm new section exists.
**Grounding:** Read of mirdan-quality.md (45 lines); enyal remember tool signature from exploration

---

### Step 12: Enhance mirdan-workflow.md — Enyal Integration section

**File:** `src/mirdan/integrations/templates/claude_code/mirdan-workflow.md` (verified via Read)
**Action:** Edit — add new section after "Convention Learning Cycle" (after line 44), update Convention Learning Cycle text
**Details:**
1. Update Convention Learning Cycle (lines 42-44) to mention enyal:
```markdown
Run `scan_conventions` after establishing code patterns in a project. Discovered conventions
are persisted to `.mirdan/conventions.yaml` and automatically included in future
`enhance_prompt` quality requirements. Store high-confidence conventions in enyal via
`enyal_remember` so they're available across sessions with scope-aware retrieval.
```

2. Add new section:
```markdown

## Enyal Knowledge Integration

When enyal is available, mirdan leverages it for persistent quality intelligence:

| Enyal Tool | When to Use | Example |
|------------|-------------|---------|
| `enyal_recall` | Before coding — get conventions | `recall("conventions", file_path=current_file)` |
| `enyal_traverse` | Before refactoring — understand dependencies | `traverse(start_query="auth patterns", max_depth=2)` |
| `enyal_impact` | Before architecture changes — check what depends on a decision | `impact(query="database convention")` |
| `enyal_remember` | After task completion — store decisions and patterns | With `suggest_supersedes=true` for evolving conventions |
| `enyal_status` | Periodically — monitor knowledge health | `status(view="health")` — act if score < 0.7 |
```
**Depends On:** None
**Verify:** Read file, confirm updated Convention Learning Cycle and new Enyal Knowledge Integration section.
**Grounding:** Read of mirdan-workflow.md (54 lines); enyal tool signatures from exploration

---

### Step 13: Write orchestrator tests

**File:** `tests/test_orchestrator.py` (verified via Read)
**Action:** Edit — add new test class after `TestSequentialThinkingRecommendations` (after line 297)
**Details:**
Add `TestEnyalGraphRecommendations` class with 6 tests:
1. `test_enyal_traverse_for_refactor` — REFACTOR task gets traverse recommendation
2. `test_enyal_impact_for_refactor` — REFACTOR task gets impact recommendation
3. `test_no_enyal_graph_for_generation` — GENERATION task does NOT get traverse/impact
4. `test_enyal_remember_on_first_call` — First call (no session) gets remember recommendation
5. `test_no_enyal_remember_after_validation` — Session with validation_count > 0 skips remember
6. `test_enyal_traverse_in_planning` — PLANNING task gets traverse recommendation

Each test follows the existing pattern: create Intent, call suggest_tools, check mcp_names.
**Depends On:** Steps 2, 3, 4
**Verify:** Run `uv run pytest tests/test_orchestrator.py -v` — all pass including new tests.
**Grounding:** Read of test_orchestrator.py (319 lines, existing test patterns)

---

### Step 14: Write server test for knowledge_storage_hint

**File:** `tests/test_server_tools.py` (verified via Glob)
**Action:** Edit — add test for knowledge_storage_hint
**Details:**
Add test verifying:
1. When `validate_code_quality` returns knowledge_entries and auto_memory is False → `knowledge_storage_hint` is present
2. When no knowledge_entries → `knowledge_storage_hint` is absent

Follow existing test patterns in test_server_tools.py.
**Depends On:** Step 5
**Verify:** Run `uv run pytest tests/test_server_tools.py -v -k knowledge_storage` — passes.
**Grounding:** Read of server.py lines 634-637; Glob confirmed test_server_tools.py exists

---

### Step 15: Run full test suite + linters

**File:** N/A
**Action:** Bash
**Details:**
1. `uv run pytest tests/ -q` — all 2390+ tests pass (plus ~8 new)
2. `uv run ruff check src/ tests/` — clean
3. `uv run mypy src/mirdan/` — clean
**Depends On:** All previous steps
**Verify:** Zero failures, zero lint errors, zero type errors.
**Grounding:** Baseline: 2390 tests passing, ruff+mypy clean (verified via Bash)

---

## Summary

| Area | Files Changed | Lines Added | Tests Added |
|------|--------------|-------------|-------------|
| Orchestrator | `orchestrator.py` | ~40 | 6 |
| Server | `server.py` | ~5 | 2 |
| Skills | 5 SKILL.md files | ~25 | 0 |
| Rules | 2 rule .md files | ~25 | 0 |
| **Total** | **9 files** | **~95** | **8** |

### What This Changes
- REFACTOR tasks now get enyal traverse + impact recommendations
- PLANNING tasks now get enyal traverse recommendations
- All first-call tasks get enyal remember post-task recommendation
- validate_code_quality guides agents to store knowledge entries properly
- /review, /quality, /gate, /plan skills can now access enyal
- Templates teach users about traverse, impact, status, supersedes

### What This Does NOT Change
- No new models, dependencies, or config options
- No changes to KnowledgeProducer (loose coupling preserved)
- No changes to EnyalGatherer (Phase 2: tag/content_type filtering)
- No changes to agent templates (lightweight agents stay lean)
- No changes to convention_extractor.py (Phase 2: enyal sync)
- Existing enyal recall logic untouched (session-aware routing preserved)

### Tech Debt Assessment: NONE
- All changes follow existing patterns exactly
- Backward compatible (works with or without enyal available)
- auto_memory edge case handled (hint suppressed when auto-storage active)
- Every Python change has corresponding test coverage
