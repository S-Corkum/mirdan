# /plan-review — Deep Plan Review (Escape Hatch)

Review a plan against its referenced brief. Use this for high-stakes plans
where judgment review is needed (auth, payments, data migration, regulated
code). For routine coverage checking, prefer `/plan-verify` (local Gemma 4,
zero API cost, mechanical).

## Output Contract (mandatory)

You MUST produce output conforming exactly to the rubric at
`<mirdan-install>/templates/plan-review-rubric.md`. The rubric defines 5
mandatory sections in a fixed order:

1. `## unmapped_acs`
2. `## constraint_violations`
3. `## scope_violations`
4. `## grounding_gaps`
5. `## risks`

Followed by a single line: `**Verdict:** pass | fail | revise`.

Do NOT reorder, rename, merge, or add sections. Do NOT add a preamble before
`## unmapped_acs`. Empty sections use `- (none)`. The output of this command
must be byte-identical in structure to Claude Code's `/plan-review --stakes high`
output so both IDEs produce interchangeable review artifacts.

If you cannot read the rubric file, reproduce the 5 section headings
verbatim from this instruction.

## Review inputs

1. Plan at the user-provided path — read all sections and subtasks.
2. Brief at the path in plan frontmatter's `brief:` field.

## Apply the rubric

- `unmapped_acs`: brief ACs with no story AC mapping
- `constraint_violations`: brief Constraints the plan violates
- `scope_violations`: brief Out-of-Scope items present in plan
- `grounding_gaps`: subtasks missing any of 6 grounding fields
- `risks`: judgment — emergent cross-story risks that mechanical verification
  via `mcp__mirdan__verify_plan_against_brief` would miss

## Quality scaling

Review quality scales with your selected Cursor model. Mirdan does NOT
proxy this call. For strongest review, use Opus / Gemini 3 Pro / GPT-5 Pro
class.
