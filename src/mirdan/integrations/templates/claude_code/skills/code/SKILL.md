---
name: code
description: "Enhanced coding with automatic mirdan quality orchestration"
model: inherit
allowed-tools: mcp__mirdan__enhance_prompt, mcp__mirdan__validate_code_quality, mcp__mirdan__get_quality_standards, mcp__mirdan__validate_quick, Read, Write, Edit, Glob, Grep, Bash
---

# /code — Quality-Orchestrated Coding

Execute coding tasks with automatic quality enforcement via mirdan.

## Dynamic Context

Recent changes:
```
!`git diff --stat HEAD 2>/dev/null | tail -5`
```

## Workflow

1. **Analyze** — Call `mcp__mirdan__enhance_prompt` with the user's task to get:
   - Detected language and frameworks
   - Quality requirements to follow
   - Security sensitivity (touches_security)
   - A session_id for the task

2. **Standards** — Call `mcp__mirdan__get_quality_standards` for the detected language/framework

3. **Implement** — Write the code following the quality_requirements from step 1

4. **Validate** — Call `mcp__mirdan__validate_code_quality` on each changed file:
   - Set `check_security=true` if touches_security was flagged
   - Fix all errors immediately
   - Note warnings for review

5. **Complete** — Confirm:
   - All validation errors resolved
   - Quality requirements from enhance_prompt satisfied
   - No placeholder code (AI001) or hallucinated imports (AI002)
