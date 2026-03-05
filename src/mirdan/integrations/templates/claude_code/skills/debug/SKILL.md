---
name: debug
description: "Mirdan-assisted debugging with quality validation"
argument-hint: "Describe the bug or error"
model: inherit
allowed-tools: mcp__mirdan__enhance_prompt, mcp__mirdan__validate_code_quality, mcp__mirdan__get_quality_standards, mcp__enyal__enyal_recall, mcp__enyal__enyal_remember, mcp__context7__resolve-library-id, mcp__context7__query-docs, Read, Write, Edit, Glob, Grep, Bash
---

# /debug — Quality-Aware Debugging

Debug issues with mirdan quality analysis to prevent introducing new problems.

## Dynamic Context

Recent changes:
```
!`git diff --stat HEAD 2>/dev/null | tail -5`
```

## Workflow

1. **Analyze** — Call `mcp__mirdan__enhance_prompt` with the bug description (task_type=debug)
   - Get security context and quality requirements
   - Note if touches_security is flagged

2. **Check Known Issues** — Call `mcp__enyal__enyal_recall` with the bug description to check if a similar issue was previously solved

3. **Investigate** — Read relevant code, trace the issue through the codebase

4. **Fix** — Apply the fix following quality requirements from enhance_prompt

5. **Validate** — Call `mcp__mirdan__validate_code_quality` on the modified code
   - Set `check_security=true` if touches_security was flagged
   - Ensure the fix doesn't introduce new violations

6. **Verify** — Confirm:
   - Root cause addressed (not just symptoms)
   - No new validation errors introduced
   - Regression test coverage considered
