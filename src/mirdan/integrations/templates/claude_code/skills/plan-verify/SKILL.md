---
name: plan-verify
description: "Verify a flat plan is internally executable (mechanical, local, no LLM)"
argument-hint: "<plan-path>"
model: inherit
allowed-tools: mcp__mirdan__verify_plan, Read, Glob
---

# /plan-verify — Mechanical Plan Self-Check

Run mechanical verification of a flat plan against **itself** — no brief, no LLM,
deterministic, milliseconds. Confirms the plan is internally executable before you
implement it.

## Workflow

1. **Parse args** — `<plan-path>` required.

2. **Call MCP** — `mcp__mirdan__verify_plan(plan_path)`.

3. **Render report** — present the result as markdown with these sections:

   ```
   ## Verification: <plan-path>

   ### Score: <coverage_score>/1.0 — <verified ? PASS : FAIL>

   #### phantom_files
   Steps whose `**File:**` points at a path that doesn't exist (the most severe
   finding — an executor cannot operate on a missing path). `NEW:`-marked files
   must have an existing parent directory.

   #### dependency_errors
   `**Depends On:**` references to steps that don't exist, plus dependency cycles.

   #### vague_cross_references
   "as discussed", "see above", "from before" — a reader can't resolve these.

   #### missing_grounding
   Steps missing any of File / Action / Details / Verify / Grounding.

   #### lld_gaps (advisory — does not fail verification)
   Low-Level Design interfaces marked [EXISTING] without a file:line citation, or
   [NEW] without a step that creates them.

   ### Summary
   <summary field>
   ```

4. **Exit code** — 0 if `verified=true`, non-zero otherwise, so CI/hooks can gate.

## Escape hatch

For judgment review (auth, payments, regulated code, emergent cross-step risk),
use `/plan-review --stakes high <plan-path>` — model-judgment review against the
shared rubric, complementary to this mechanical check.
