---
name: brief
description: "Author a structured brief to constrain plan generation"
argument-hint: "<slug> \"<one-line outcome>\""
model: inherit
allowed-tools: mcp__mirdan__enhance_prompt, mcp__mirdan__validate_brief, mcp__enyal__enyal_recall, mcp__enyal__enyal_remember, mcp__enyal__enyal_traverse, Read, Write, Glob, Grep
---

# /brief — Structured Brief Authoring

Create a brief that constrains downstream plan creation. Briefs shift
frontier-token spend upstream (to constraints) so downstream plan execution
can use cheaper models reliably.

## Dynamic Context

Briefs directory:
```
!`ls docs/briefs/ 2>/dev/null | head -10`
```

## Workflow

1. **Parse args** — `<slug>` (filename-safe) and outcome statement.
   Resolve target to `docs/briefs/<slug>.md` (or `briefs.dir` config override).
   **Dir-collision check:** if target dir doesn't exist but `briefs/` or
   `docs/plans/briefs/` does, prompt the user once to confirm/reconfigure
   before creating. Record confirmation at `.mirdan/.briefs-dir-confirmed`.

2. **Template** — scaffold from `<mirdan-install>/templates/brief.md`, or use
   the inline template at the bottom of this skill if the packaged template is
   unavailable.

3. **Recall prior art** — `mcp__enyal__enyal_recall` with the outcome and any
   keywords. Include past decisions, relevant conventions, and superseded
   entries worth referencing.

4. **Enhance context** — `mcp__mirdan__enhance_prompt(prompt=<outcome>,
   task_type="planning")`. Use `detected_frameworks`, `touches_security`, and
   `quality_requirements` to inform the draft.

5. **Draft iteratively** — work through sections in this order with the user:
   Outcome → Users & Scenarios → Business ACs → Constraints → Out of Scope.
   Recommended sections (Prior Art, Pitfalls, Quality Bar, Non-Goals) follow.

6. **Validate** — `mcp__mirdan__validate_brief(brief_path)`. If `passed=false`,
   resolve every `error`-severity gap. Re-validate until it passes.

7. **Auto-store** — on pass, call `mcp__enyal__enyal_remember` with
   `content_type="brief"`, `scope="project"`, and tags including the slug +
   any detected frameworks + `freshness:active`. **If an existing entry tagged
   `slug:<slug>` already has `freshness:active`:** call `enyal_update` on the
   old entry to change that tag to `freshness:superseded` before writing the
   new one. This keeps `enyal_recall` results biased toward the current brief.

8. **Emit hand-off** — print exactly:
   `Brief saved to docs/briefs/<slug>.md. Run /plan --brief docs/briefs/<slug>.md to generate the implementation plan.`

## Required Sections (hard gates — `/plan` refuses invalid briefs)

- **Outcome** — specific metric or observable outcome (no "should be fast")
- **Users & Scenarios** — ≥ 1 named persona + ≥ 1 scenario
- **Business Acceptance Criteria** — ≥ 3 testable criteria
- **Constraints** — ≥ 1 specific constraint (no "follow best practices")
- **Out of Scope** — ≥ 1 explicit exclusion

## Recommended Sections (soft gates — warnings only)

- Prior Art · Known Pitfalls · Quality Bar · Non-Goals
