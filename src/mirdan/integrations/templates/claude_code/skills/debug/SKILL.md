---
description: "Use when debugging, fixing bugs, or investigating errors. Analyzes the problem, applies a fix, and validates the fix meets quality standards."
allowed-tools: Read, Edit, Write, Grep, Glob, Bash, mcp__mirdan__enhance_prompt, mcp__mirdan__validate_code_quality, mcp__mirdan__get_verification_checklist
---

# /debug - Quality-Validated Debugging

Follow this workflow to debug the issue:

## 1. Analyze intent
Call `mcp__mirdan__enhance_prompt` with the bug description and `task_type="debug"`.
Note the `quality_requirements` and `touches_security` fields.

## 2. Investigate
Use Read, Grep, and Glob to trace the issue. Identify the root cause before changing code.

## 3. Fix
Apply the minimal fix needed. Do not refactor or add features beyond the bug fix.

## 4. Validate
Call `mcp__mirdan__validate_code_quality` on the modified code.
Pass `check_security=true` if the fix touches security-sensitive areas.

## 5. Verify
Run relevant tests to confirm the fix works and no regressions were introduced.

## 6. Report
Run `mcp__mirdan__get_verification_checklist` for the debug task type.
Summarize the root cause, the fix applied, and the validation score.
