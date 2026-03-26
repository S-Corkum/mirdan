---
name: plan-review
description: "Staff-engineer-grade review of AI-generated implementation plans — verifies grounding, completeness, and cheap-model executability"
argument-hint: "Path to plan file, or 'last' to review the most recent plan"
user-invocable: true
model: inherit
context: fork
allowed-tools: >-
  mcp__mirdan__enhance_prompt,
  mcp__mirdan__validate_code_quality,
  mcp__mirdan__get_quality_standards,
  mcp__enyal__enyal_recall,
  Read, Glob, Grep
---

# /plan-review — Staff-Engineer Plan Review

Review an AI-generated implementation plan for factual accuracy and cheap-model executability. This skill is the Judge half of a Judge/Planner separation — `/plan` creates plans, `/plan-review` reviews them.

## Usage

- `/plan-review docs/plans/auth.md` — Review a specific plan
- `/plan-review last` — Review the most recent plan in docs/plans/
- `/plan-review docs/plans/auth.md --target flash` — Review for a Flash-class executor

## Dynamic Context

Most recent plan:
```
!`ls -t docs/plans/*.md 2>/dev/null | head -1`
```

## Effort Calibration

- **Small plans (1-5 steps):** Verify ALL references with tools
- **Medium plans (6-15 steps):** Verify all file paths and line numbers; sample-verify functions and imports
- **Large plans (16+ steps):** Verify file paths for all steps; deep-verify the 5 highest-risk steps
- **Budget:** Maximum 20 tool calls for grounding verification

## Workflow

### 1. Read Plan

Read the plan file from the provided path. If the argument is `last`, use `Glob("docs/plans/*.md")` and read the most recently modified file.

### 2. Structural Validate

Call `mcp__mirdan__enhance_prompt` with the full plan text as the `prompt` parameter and `task_type="plan_validation"`. Record the returned scores:
- `clarity_score`, `completeness_score`, `atomicity_score`, `grounding_score`
- `issues` list — structural problems detected
- `ready_for_cheap_model` — boolean

### 3. Load Context

Call `mcp__enyal__enyal_recall` with query `"architecture conventions decisions"` to load project standards. Use these to inform the Architecture assessment in Step 7.

### 4. Extract References

Scan the plan text and build a reference table. Look for these patterns:

| Type | Pattern to Match | Example |
|------|-----------------|---------|
| File path | Backtick-wrapped string containing `/` with a file extension | `` `src/main.py` `` |
| Line number | Number preceded by `line `, `:`, or `L` | "line 45", ":30", "L12" |
| Function/class | Backtick-wrapped `def name()` or `class Name` | `` `def process()` `` |
| Import | Backtick-wrapped `from X import Y` or `import Z` | `` `from utils import helper` `` |
| Step ref | "Step N" or "Depends On: Step N" | "Depends On: Step 2" |
| Directory | Path ending in `/` | `` `src/models/` `` |

Record each reference with its step number for the verification pass.

### 5. Grounding Verification

This is the most critical step. For each extracted reference, verify it with the appropriate tool.

**BEFORE verifying, apply false-positive prevention rules:**
1. If a step's Action is "Write" or "Create", or its File field starts with "NEW:", classify its file references as **EXPECTED_NEW**. Verify the PARENT directory exists instead.
2. If Step N creates a file or function, and Step M (where M > N) references it, classify Step M's reference as **EXPECTED_NEW** (cumulative state tracking).
3. If a step inserts lines into a file, note that subsequent steps targeting the same file may have shifted line numbers. Flag as a warning, not an error.

**Verification actions:**

| Reference Type | Tool | What to Check |
|---|---|---|
| File path | `Read` | File exists; note first few lines for context |
| Line number | `Read` with offset/limit | Content at the cited line matches the plan's description |
| Function/class | `Grep` in project | Symbol exists somewhere in the codebase |
| Import path | `Grep` for module | Module exists and is importable |
| Directory | `Glob` with pattern | Directory exists |
| Step ref | Cross-check | Referenced step number exists in the plan |

**Classify each reference as one of:**
- **VERIFIED** — Tool confirmed the claim is accurate
- **MISMATCH** — Reference exists but content differs from what the plan claims (include what was actually found)
- **NOT_FOUND** — Reference does not exist in the codebase
- **EXPECTED_NEW** — Plan intends to create this (false-positive rules applied)
- **UNVERIFIABLE** — Cannot be checked with available tools (external services, runtime config)

### 6. Dependency Analysis

Build a mental step graph from "Depends On:" fields:
- Do all referenced step numbers actually exist in the plan?
- Are there circular dependencies (Step A depends on B, B depends on A)?
- Do file creation steps come BEFORE modification steps for the same file?
- Do import addition steps come BEFORE code that uses those imports?

### 7. Semantic Review

Assess dimensions that require judgment:

- **COMPLETENESS**: For each new function or file in the plan, is there a corresponding test step? Are `__init__.py` exports updated when new modules are added? Are config changes included? Are type annotations addressed?
- **EXECUTABILITY**: Read each step as if you are Claude Haiku with no prior context. Does the step contain ALL information needed to execute it? Flag any step using "as discussed", "the function from before", "this file", or other ambiguous references.
- **SAFETY**: If the plan touches auth, user input, database queries, or APIs — are input validation, error handling, and injection prevention addressed?
- **ARCHITECTURE**: Do the changes respect the project conventions loaded from enyal in Step 3?

### 8. Synthesize Report

Calculate dimension scores using the rubric below and output the report in the exact template format.

## Scoring Rubric

### Grounding (30% weight)
- **1.0**: All references VERIFIED or EXPECTED_NEW, zero MISMATCH or NOT_FOUND
- **0.8**: All critical references verified, 1-2 minor UNVERIFIABLE items
- **0.5**: Some verified, some MISMATCH found
- **0.0**: Majority of references unverified or wrong

### Completeness (20%)
- **1.0**: All sections present, no detectable gaps
- **0.75**: Sections present, 1-2 minor gaps (missing test step, missing export)
- **0.5**: Some sections missing, notable gaps
- **0.0**: Structurally incomplete

### Atomicity (15%)
- **1.0**: All steps are single-action, single-file
- **0.7**: 1-2 compound steps detected ("and then", "first...then")
- **0.3**: Multiple compound steps
- **0.0**: Steps are mixed-action paragraphs

### Clarity (10%)
- **1.0**: Zero vague language instances
- Deduct **0.1** per instance of: "should", "probably", "likely", "maybe", "around line", "somewhere", "I think", "I believe", "might", "possibly"

### Dependency Order (10%)
- **1.0**: Valid ordering, all dependencies satisfied
- **0.7**: Minor ordering issues
- **0.3**: Missing dependency declarations
- **0.0**: Circular dependencies or fundamentally wrong order

### Executability (10%)
- **1.0**: Every step is self-contained and unambiguous
- **0.7**: Most steps clear, some need minor additional context
- **0.3**: Multiple steps require interpretation by the executor
- **0.0**: Steps are vague descriptions, not executable instructions

### Safety (5%)
- **1.0**: All security concerns addressed (or plan does not touch security)
- **0.7**: Concerns identified but not fully addressed in steps
- **0.3**: Security-relevant changes with no security considerations
- **0.0**: Active security risks present in plan (hardcoded secrets, injection patterns)

## Verdict Thresholds

Target model defaults to **haiku** if not specified in the argument.

| Target Model | Overall | Grounding | Atomicity |
|-------------|---------|-----------|-----------|
| haiku | >= 0.95 | 1.0 | >= 0.9 |
| flash | >= 0.90 | >= 0.9 | >= 0.9 |
| capable | >= 0.75 | >= 0.7 | >= 0.7 |

- **PASS**: Meets ALL thresholds for the target model
- **REVISE**: Overall >= 0.5 but below PASS threshold — fixable issues with specific feedback
- **FAIL**: Overall < 0.5 — fundamental problems requiring replanning

## Report Template

Output your review in EXACTLY this format:

```
## Plan Review Report

**Verdict:** PASS | REVISE | FAIL
**Overall Score:** X.XX / 1.00
**Target Model:** haiku | flash | capable
**Grounding Coverage:** N/M references verified (X%)

### Dimension Scores

| Dimension | Score | Weight | Issues | Status |
|-----------|-------|--------|--------|--------|
| Grounding | X.XX | 30% | N | PASS/FAIL |
| Completeness | X.XX | 20% | N | PASS/FAIL |
| Atomicity | X.XX | 15% | N | PASS/FAIL |
| Clarity | X.XX | 10% | N | PASS/FAIL |
| Dependency Order | X.XX | 10% | N | PASS/FAIL |
| Executability | X.XX | 10% | N | PASS/FAIL |
| Safety | X.XX | 5% | N | PASS/FAIL |

### Grounding Results

#### Verified (N references)
- Step X: `path/to/file.py` — EXISTS, content confirmed
- ...

#### Mismatches (N — MUST FIX)
- Step X: `path/to/file.py:30` — File exists but line 30 is `import os`, not `def helper()` as claimed. Actual location: line 47.
- ...

#### Not Found (N — MUST FIX)
- Step X: `path/to/missing.py` — File does not exist. Did you mean `path/to/actual.py`?
- ...

#### Expected New (N — OK)
- Step X: `path/to/new_file.py` — Marked as NEW, parent directory `path/to/` EXISTS.
- ...

#### Unverifiable (N — Manual check recommended)
- Step X: "Redis connection pool" — External service, cannot verify configuration.
- ...

### Issues by Step

#### Step N: [title from plan]
- **[GROUNDING/HIGH]** Description — Fix: specific recommendation
- **[COMPLETENESS/MEDIUM]** Description — Fix: specific recommendation
- ...

### Missing Steps
- No step updates `src/__init__.py` exports after adding new module in Step 4
- No test step for authentication logic introduced in Steps 4-6
- ...

### Recommendations (Priority Ordered)
1. Fix all GROUNDING mismatches — these cause execution failure
2. ...
```

## Example Finding

A correctly formatted finding looks like:

- **[GROUNDING/HIGH]** File `src/auth.py` exists but line 45 is `import os`, not `def validate_token()` as claimed. Actual location: line 72. Fix: Update step to reference line 72.
- **[COMPLETENESS/MEDIUM]** Step 4 creates `src/services/auth.py` but no step updates `src/services/__init__.py` to export it. Fix: Add a step after Step 4 to update the `__init__.py`.
- **[ATOMICITY/DETERMINISTIC]** Step 8 contains "add the import AND update the function" — compound action. Fix: Split into Step 8a (add import) and Step 8b (update function).
