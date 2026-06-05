---
name: plan
description: "Flat grounded implementation plan with a low-level design"
argument-hint: "<slug> <description>"
model: inherit
context: fork
allowed-tools: mcp__mirdan__enhance_prompt, mcp__mirdan__verify_plan, mcp__mirdan__validate_code_quality, mcp__mirdan__get_quality_standards, mcp__sequential-thinking__sequentialthinking, mcp__enyal__enyal_recall, mcp__enyal__enyal_remember, mcp__enyal__enyal_traverse, Read, Glob, Grep, Write
---

# /plan — Flat Grounded Plan with Low-Level Design

Generate an implementation plan as **Research Notes → Low-Level Design → atomic
grounded steps**. Plans output to `docs/plans/<slug>.md`. No brief required.

## Dynamic Context

Project config:
```
!`cat .mirdan/config.yaml 2>/dev/null | head -30`
```

Recent plans:
```
!`ls -t docs/plans/*.md 2>/dev/null | head -5`
```

## Workflow

1. **Parse args** — `<slug>` (filename-safe) and a one-line description of the work.

2. **Enhance** — `mcp__mirdan__enhance_prompt(prompt=<description>, task_type="planning")`.
   Use `detected_frameworks`, `touches_security`, and any `decision_guidance`
   (design domains) to shape the design.

3. **Recall** — `mcp__enyal__enyal_recall` (with `file_path` of the area you'll
   touch) and `enyal_traverse` for architecture clusters around the domain.

4. **Research (MANDATORY before any step)** — Glob the structure, Read every file
   you will modify, Read the dependency manifest, Read similar implementations,
   context7 every external API. You cannot write a step for a file you haven't Read.

5. **Write Research Notes** — verified facts with tool citations (Files Verified
   with line numbers, Dependencies Confirmed, API Documentation, Conventions).

6. **Write the Low-Level Design** (see schema below) — interfaces and signatures
   first, then the error taxonomy and the design decisions. Include a subsection
   ONLY if it applies; delete inapplicable headings — do not write "N/A".

7. **Write atomic steps** — `### Step N` with **File / Action / Details /
   Depends On / Verify / Grounding**. Each step single-action, grounded, no vague
   language ("should", "probably", "around line X", "I think", "assume").

8. **Self-validate** — run `mcp__mirdan__validate_code_quality` on any embedded code
   snippets; Glob-verify every cited file path.

9. **Write** — save to `docs/plans/<slug>.md`.

10. **Emit hand-off** — print exactly:
    `Plan saved to docs/plans/<slug>.md. Run /plan-verify docs/plans/<slug>.md to confirm it is internally executable.`

## Low-Level Design schema

Write a `## Low-Level Design` section. **Mandatory** subsections:

- **Interfaces & Signatures** — exact name, params with types, return type for each
  function/class the plan touches. Tag each **[NEW]** or **[EXISTING]**. For
  **[EXISTING]**, cite the `file:line` you Read it from. For **[NEW]**, it must be
  created by a step below. (An interface that is neither cited nor created by a step
  is hallucinated — remove it. `/plan-verify` flags these.)
- **Error Taxonomy** — each failure mode and the exception/return for it; which layer
  raises vs handles.
- **Design Decisions** — for each design domain surfaced by `enhance_prompt`, the
  approach chosen + one-line rationale grounded in this codebase.

**Applicability-gated** (include only if relevant):

- **Data Model** — only if you introduce/change a persisted or serialized shape.
- **Module Boundaries** — only if more than one module/file is touched.

## Subtask Self-Containment

Each step must be executable without scrolling up. If a step needs context from the
design, restate the one relevant line inside the step.
