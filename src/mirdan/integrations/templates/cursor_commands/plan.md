# /plan — Brief-Driven Three-Layer Plan

Generate an implementation plan structured as epic → stories → subtasks,
constrained by a brief. Plans output to `docs/plans/<slug>.md` with the brief
referenced in frontmatter.

## Usage

- `/plan --brief <path>` (required by default)
- `/plan --no-brief` (exploratory; legacy flat template; NOT eligible for
  `/plan-execute` or `/plan-verify`)

## Workflow

1. **Brief-first gate**: without `--brief` and without `--no-brief`, emit:
   `Brief-first is the default. Run /brief <slug> first, or pass --no-brief
   for exploratory plans.` and exit.
2. With `--brief <path>`: call `mcp__mirdan__validate_brief(brief_path)`.
   If `passed=false`, display gaps and exit.
3. With `--no-brief`: warn and proceed with legacy flat flow.
4. Read the brief. Extract Outcome, Constraints, ACs, Out-of-Scope, Prior Art.
5. Call `mcp__mirdan__enhance_prompt(prompt=<derived task>,
   task_type="planning", brief_path=<path>)` — brief constraints merge into
   `quality_requirements` automatically.
6. Call `mcp__enyal__enyal_recall` and `mcp__enyal__enyal_traverse` for
   architecture clusters and past decisions.
7. Read every file referenced in the brief's Prior Art. Glob directory
   structure for cited paths.
8. Draft the plan using `<mirdan-install>/templates/plan-three-layer.md`:
   Research Notes → Epic Layer → Story Layer (INVEST) → Subtasks (6 grounding
   fields each, all mandatory).
9. Self-validate: run `mcp__mirdan__validate_code_quality` on any embedded
   code snippets. Glob-verify every cited file path.
10. Write to `docs/plans/<slug>.md` with frontmatter
    `brief: docs/briefs/<slug>.md` — slug derived from the brief filename.
11. Emit: `Plan saved to docs/plans/<slug>.md. Run /plan-verify
    docs/plans/<slug>.md to confirm coverage against the brief.`

## Subtask self-containment

Every subtask must be executable without reading its parent story or the
epic. Compress the "why" from the parent story's AC into a one-line context
note inside each subtask when needed.

## Vague-language ban

No "should", "probably", "likely", "maybe", "around line N", "somewhere in",
"I think", "I believe", "assume", "might", "possibly". Verify and state facts.
