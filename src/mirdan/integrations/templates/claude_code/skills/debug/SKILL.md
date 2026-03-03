---
name: debug
description: "Mirdan-assisted debugging with quality validation"
model: inherit
allowed-tools: mcp__mirdan__enhance_prompt, mcp__mirdan__validate_code_quality, mcp__mirdan__get_quality_standards, Read, Write, Edit, Glob, Grep, Bash
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

2. **Investigate** — Read relevant code, trace the issue through the codebase

3. **Fix** — Apply the fix following quality requirements from enhance_prompt

4. **Validate** — Call `mcp__mirdan__validate_code_quality` on the modified code
   - Set `check_security=true` if touches_security was flagged
   - Ensure the fix doesn't introduce new violations

5. **Verify** — Confirm:
   - Root cause addressed (not just symptoms)
   - No new validation errors introduced
   - Regression test coverage considered
