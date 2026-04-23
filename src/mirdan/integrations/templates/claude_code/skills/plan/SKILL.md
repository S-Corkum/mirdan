---
name: plan
description: "Brief-driven three-layer plan generation"
argument-hint: "--brief <path> | --no-brief (exploratory)"
model: inherit
context: fork
allowed-tools: mcp__mirdan__enhance_prompt, mcp__mirdan__validate_brief, mcp__mirdan__validate_code_quality, mcp__mirdan__get_quality_standards, mcp__sequential-thinking__sequentialthinking, mcp__enyal__enyal_recall, mcp__enyal__enyal_remember, mcp__enyal__enyal_traverse, Read, Glob, Grep, Write
---

# /plan — Brief-Driven Three-Layer Plan

Generate an implementation plan structured as epic → stories → subtasks,
constrained by a brief. Plans output to `docs/plans/<slug>.md` with the brief
referenced in frontmatter.

## Dynamic Context

Project config:
```
!`cat .mirdan/config.yaml 2>/dev/null | head -40`
```

Recent briefs:
```
!`ls -t docs/briefs/*.md 2>/dev/null | head -5`
```

## Workflow

1. **Brief-first gate** — if no `--brief <path>` argument and not `--no-brief`:
   emit one line and exit:
   `Brief-first is the default. Run /brief <slug> first, or pass --no-brief for exploratory plans.`

2. **If `--brief <path>`:** call `mcp__mirdan__validate_brief(brief_path)`.
   If `passed=false`, display gaps with the user's fix guidance and exit.
   Do not proceed with an invalid brief.

3. **If `--no-brief`:** warn — "Exploratory plan without brief constraint.
   Template will be flat (legacy). Three-layer templates require a brief." —
   then proceed with the legacy flat planning flow.

4. **Read brief** — extract Outcome, Constraints, ACs, Out-of-Scope, Prior Art.

5. **Enhance** — `mcp__mirdan__enhance_prompt(prompt=<derived task>,
   task_type="planning", brief_path=<path>)`. Brief constraints merge into
   `quality_requirements` automatically.

6. **Recall** — `mcp__enyal__enyal_recall` with `file_path=<brief path>`
   weighted to project scope. `mcp__enyal__enyal_traverse` for architecture
   clusters around the brief's domain.

7. **Ground** — read every file referenced in brief's Prior Art. Glob directory
   structure for file paths the plan will cite. context7 for external APIs.

8. **Draft three-layer plan** using template at
   `<mirdan-install>/templates/plan-three-layer.md`:
   - Research Notes (verified facts with tool citations)
   - Epic Layer (outcome + metric + scope boundary + epic grounding)
   - Story Layer (each with INVEST check + AC list)
   - Subtasks per story (File, Action, Details, Depends on, Verify,
     Grounding, If-blocked — all 6 fields mandatory)

9. **Self-validate** — run `mcp__mirdan__validate_code_quality` on any code
   snippets embedded in the plan. Glob-verify every cited file path.

10. **Write** — derive slug from brief filename. Save to
    `docs/plans/<slug>.md` with frontmatter `brief: docs/briefs/<slug>.md`.

11. **Emit hand-off** — print exactly:
    `Plan saved to docs/plans/<slug>.md. Run /plan-verify docs/plans/<slug>.md to confirm coverage against the brief.`

## Subtask Self-Containment Contract

Subtasks MUST be executable without reading the epic or stories above them.
Compress the "why" from the parent story's AC into a one-line context note
inside each subtask when needed. Cheap executors should not have to scroll up.

## Vague-Language Ban

No "should", "probably", "likely", "maybe", "around line X", "somewhere in",
"I think", "I believe", "assume", "might", "possibly". Verify and state facts.
