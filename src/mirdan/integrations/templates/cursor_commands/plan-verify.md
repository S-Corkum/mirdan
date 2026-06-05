# /plan-verify — Mechanical Plan Self-Check

Verify a flat plan against **itself** — no brief, no LLM, deterministic,
milliseconds. Confirms the plan is internally executable before you implement it.

## Usage

`/plan-verify <plan-path>`

## Workflow

1. Call `mcp__mirdan__verify_plan(plan_path)`.
2. Render the structured report:

   ```
   ## Verification: <plan-path>

   ### Score: <coverage_score>/1.0 — <PASS | FAIL>

   #### phantom_files          — File field points at a path that doesn't exist
   #### dependency_errors      — Depends-On refs to missing steps, or cycles
   #### vague_cross_references — "as discussed", "see above", "from before"
   #### missing_grounding      — steps missing File/Action/Details/Verify/Grounding
   #### lld_gaps (advisory)    — [EXISTING] interface w/o file:line citation, or
                                 [NEW] interface created by no step

   ### Summary
   <summary field>
   ```

3. Exit non-zero if `verified=false` so CI / hooks can gate on it.

## Escape hatch

For judgment review (auth, payments, regulated code, cross-step emergent risk),
use `/plan-review --stakes high <plan-path>` against the shared rubric at
`templates/plan-review-rubric.md`.
