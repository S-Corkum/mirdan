# /plan-verify — Brief-Driven Plan Verification

Run mechanical verification of a three-layer plan against its brief via
local Gemma 4. Cheap, deterministic, ≤ 30s on mid-tier hardware for a
30-subtask plan. Replaces `/plan-review` as the default quality check.

## Usage

`/plan-verify <plan-path>`

## Workflow

1. Read the plan file; extract the `brief:` field from the frontmatter.
2. If no `brief:` field: emit `Plan has no brief reference. /plan-verify
   only supports brief-driven plans. Use /plan-review for flat-template
   plans.` and exit.
3. Call `mcp__mirdan__verify_plan_against_brief(plan_path, brief_path)`.
4. Render the structured report with ALL these sections:

   ```
   ## Verification: <plan-path>

   ### Coverage Score: <score>/1.0
   ### Semantic Check: <ran | skipped — BRAIN-tier LLM not available>

   ### Mechanical findings (no LLM — reliable at any hardware tier)
   #### phantom_files           — files referenced but not found
   #### dependency_errors       — dangling Depends-on refs or cycles
   #### vague_cross_references  — "as discussed", "like Step N", etc.
   #### missing_grounding       — subtasks missing any of 6 grounding fields
   #### out_of_scope_violations — brief Out-of-Scope items appearing in plan
   #### invest_failures         — stories missing INVEST structural fields

   ### Semantic findings (BRAIN-tier LLM only; conf >= 0.6)
   #### unmapped_acs            — brief ACs with no story AC mapping

   ### Summary
   <summary field>
   ```

5. Exit non-zero if `verified=false` so CI / hooks can gate on it.

## Escape hatch

For judgment review (auth, payments, regulated code), use
`/plan-review <plan-path>` which instructs the user-selected Cursor model to
produce the same 5-section output shape against the shared rubric at
`templates/plan-review-rubric.md`.
