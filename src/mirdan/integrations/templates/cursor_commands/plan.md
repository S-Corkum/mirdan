# /plan — Flat Grounded Plan with Low-Level Design

Generate an implementation plan as **Research Notes → Low-Level Design → atomic
grounded steps**. Output to `docs/plans/<slug>.md`. No brief required.

## Usage

`/plan <slug> "<what to build>"`

## Workflow

1. Call `mcp__mirdan__enhance_prompt(prompt=<description>, task_type="planning")`.
   Use `detected_frameworks`, `touches_security`, and any `decision_guidance`
   (design domains) to shape the design.
2. Call `mcp__enyal__enyal_recall` and `mcp__enyal__enyal_traverse` for
   architecture clusters and past decisions.
3. Research before any step: Glob the structure, Read every file you'll modify,
   Read the dependency manifest and similar implementations, context7 every
   external API. You cannot write a step for a file you haven't Read.
4. Write **Research Notes** — verified facts with tool citations.
5. Write the **Low-Level Design** (schema below). Include a subsection ONLY if it
   applies — delete inapplicable headings, do not write "N/A".
6. Write atomic `### Step N` steps with **File / Action / Details / Depends On /
   Verify / Grounding**.
7. Self-validate: `mcp__mirdan__validate_code_quality` on embedded snippets;
   Glob-verify every cited path.
8. Write to `docs/plans/<slug>.md`.
9. Emit: `Plan saved to docs/plans/<slug>.md. Run /plan-verify docs/plans/<slug>.md to confirm it is internally executable.`

## Low-Level Design schema

`## Low-Level Design` section. **Mandatory:** Interfaces & Signatures (each tagged
[NEW]/[EXISTING]; [EXISTING] cites `file:line`, [NEW] created by a step), Error
Taxonomy, Design Decisions (one per surfaced design domain + rationale).
**Applicability-gated:** Data Model (only if a persisted/serialized shape changes),
Module Boundaries (only if >1 module touched).

## Vague-language ban

No "should", "probably", "likely", "maybe", "around line N", "somewhere in",
"I think", "assume", "might", "possibly". Verify and state facts.
