# /brief — Structured Brief Authoring

Author a brief that constrains downstream plan generation. Shifts
frontier-token spend upstream (to constraints) so plan creation + execution
can use cheaper models reliably.

## Usage

`/brief <slug> "<one-line outcome>"`

## Workflow

1. Parse `<slug>` (filename-safe) and outcome. Resolve target to
   `docs/briefs/<slug>.md` (override: `briefs.dir` in `.mirdan/config.yaml`).
2. If target dir does not exist but `briefs/` or `docs/plans/briefs/` does,
   prompt the user once to confirm before creating.
3. Scaffold from `<mirdan-install>/templates/brief.md` or use the template
   embedded in the `mirdan-brief.mdc` rule.
4. Call `mcp__enyal__enyal_recall` with the outcome for Prior Art. Call
   `mcp__mirdan__enhance_prompt` with `task_type="planning"` for framework
   context.
5. Draft required sections iteratively with the user:
   Outcome → Users & Scenarios → Business ACs → Constraints → Out of Scope.
6. Call `mcp__mirdan__validate_brief(brief_path)`. Resolve all `error`-severity
   gaps. Re-validate until passed.
7. On pass: auto-store to enyal with `content_type="brief"`, `scope="project"`,
   tags `[<slug>, *detected_frameworks, "freshness:active"]`. If a prior entry
   tagged `slug:<slug>` has `freshness:active`, update it to
   `freshness:superseded` first.
8. Emit: `Brief saved to docs/briefs/<slug>.md. Run /plan --brief
   docs/briefs/<slug>.md to generate the implementation plan.`

## Required sections (hard gates)

- Outcome · Users & Scenarios · Business Acceptance Criteria · Constraints · Out of Scope

## Recommended sections (soft gates)

- Prior Art · Known Pitfalls · Quality Bar · Non-Goals
