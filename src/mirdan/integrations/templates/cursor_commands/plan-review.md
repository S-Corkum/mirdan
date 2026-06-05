# /plan-review — Deep Plan Review (Escape Hatch)

Judgment review of a flat plan. Use this for high-stakes plans where model
judgment is needed (auth, payments, data migration, regulated code). For routine
checking, prefer `/plan-verify` (local, mechanical, zero API cost).

## Output Contract (mandatory)

You MUST produce output conforming exactly to the rubric at
`<mirdan-install>/templates/plan-review-rubric.md`. The rubric defines 5
mandatory sections in a fixed order:

1. `## design_gaps`
2. `## grounding_gaps`
3. `## completeness_gaps`
4. `## safety`
5. `## risks`

Followed by a single line: `**Verdict:** pass | fail | revise`.

Do NOT reorder, rename, merge, or add sections. Do NOT add a preamble before
`## design_gaps`. Empty sections use `- (none)`. The output must be byte-identical
in structure to Claude Code's `/plan-review --stakes high` output so both IDEs
produce interchangeable review artifacts.

If you cannot read the rubric file, reproduce the 5 section headings verbatim
from this instruction.

## Review inputs

The flat plan at the user-provided path — read its Research Notes, Low-Level
Design, and every step. There is no brief; the plan is reviewed against its own
stated outcome, design, and grounding.

## Apply the rubric

- `design_gaps`: unsound/missing interfaces, error taxonomy, contracts; `[EXISTING]`
  interfaces without a `file:line` citation, `[NEW]` created by no step
- `grounding_gaps`: steps missing File/Action/Details/Verify/Grounding, or referencing
  unverified files/APIs
- `completeness_gaps`: missing tests, imports/exports, config, error handling, edge cases
- `safety`: auth, input validation, injection, secrets, migration/rollback
- `risks`: judgment — emergent cross-step risks `/plan-verify` would miss

## Quality scaling

Review quality scales with your selected Cursor model. Mirdan does NOT proxy this
call. For strongest review, use Opus / Gemini 3 Pro / GPT-5 Pro class.
