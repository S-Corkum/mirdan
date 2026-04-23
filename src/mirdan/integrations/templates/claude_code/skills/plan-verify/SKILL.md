---
name: plan-verify
description: "Verify a three-layer plan against its brief via local Gemma 4"
argument-hint: "<plan-path>"
model: inherit
allowed-tools: mcp__mirdan__verify_plan_against_brief, Read, Glob
---

# /plan-verify — Brief-Driven Plan Verification

Run mechanical verification against the plan's brief. Cheap (local Gemma 4
via mirdan MCP), deterministic, ≤ 30s on mid-tier hardware for a 30-subtask
plan. This replaces `/plan-review` as the default quality check.

## Workflow

1. **Parse args** — `<plan-path>` required.

2. **Read plan frontmatter** — extract the `brief:` field. If absent, emit:
   `Plan has no brief reference. /plan-verify only supports brief-driven
   (three-layer) plans. Use /plan-review for flat-template plans.` and exit.

3. **Call MCP** — `mcp__mirdan__verify_plan_against_brief(plan_path, brief_path)`.

4. **Render report** — present result as markdown with ALL these sections:

   ```
   ## Verification: <plan-path>

   ### Coverage Score: <coverage_score>/1.0
   ### Semantic Check: <"ran" | "skipped — BRAIN-tier LLM not available">

   ### Mechanical findings (no LLM required — reliable at any hardware tier)

   #### phantom_files
   Subtasks referencing files that don't exist (the most severe finding —
   cheap executor cannot operate on a missing path). `NEW:`-marked files
   must have an existing parent directory.

   #### dependency_errors
   Dangling `Depends on:` references to subtasks that don't exist in the
   plan, plus any circular dependencies.

   #### vague_cross_references
   Language like "as discussed", "like Step 3", "the function from earlier"
   — cheap executors can't resolve these.

   #### missing_grounding
   Subtasks missing any of the 6 required grounding fields.

   #### out_of_scope_violations
   Brief Out-of-Scope items appearing in plan body.

   #### invest_failures
   Stories missing INVEST structural fields.

   ### Semantic findings (BRAIN-tier LLM only)

   #### unmapped_acs
   Brief Business ACs with no matching story AC (by BRAIN semantic judgment
   with confidence >= 0.6). Empty if semantic check was skipped.

   ### Summary
   <summary field>
   ```

5. **Exit code** — 0 if `verified=true`, non-zero otherwise so CI / hooks
   can gate on it.

## Escape hatch

For judgment review (auth, payments, regulated code), use
`/plan-review --stakes high <plan-path>`. That invokes the `plan-reviewer`
subagent and produces a different 5-section output enforcing the shared
rubric at `templates/plan-review-rubric.md`.
