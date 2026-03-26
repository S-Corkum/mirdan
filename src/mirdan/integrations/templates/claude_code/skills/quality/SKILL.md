---
name: quality
description: "On-demand code quality validation with mirdan"
argument-hint: "File path or --trends"
model: inherit
context: fork
allowed-tools: mcp__mirdan__validate_code_quality, mcp__mirdan__validate_quick, mcp__mirdan__get_quality_standards, mcp__mirdan__get_quality_trends, mcp__enyal__enyal_recall, Read, Glob, Grep
---

# /quality — On-Demand Quality Check

Run mirdan quality validation on specific code or files in an isolated context.

## Usage

- `/quality` — Validate the most recently changed file
- `/quality src/auth.py` — Validate a specific file
- `/quality --trends` — Show quality score trends

## Workflow

1. **Identify** — Determine which code to validate (file, selection, or recent changes)

2. **Conventions** — Call `mcp__enyal__enyal_recall` with `input: { query: "quality conventions" }` to load project quality standards for comparison

3. **Read** — Read the target code

4. **Validate** — Call `mcp__mirdan__validate_code_quality` with:
   - `check_security=true` for auth/input/API code
   - `check_architecture=true` for structural analysis
   - `severity_threshold="info"` for comprehensive results

5. **Present** — Show results organized by severity:
   - **Errors** — Must fix before merge
   - **Warnings** — Should fix, note if intentional
   - **Info** — Suggestions for improvement
   - **Score** — Overall quality score (0.00-1.00)
