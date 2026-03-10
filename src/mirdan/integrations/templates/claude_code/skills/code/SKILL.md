---
name: code
description: "Enhanced coding with automatic mirdan quality orchestration"
argument-hint: "Describe what to build"
model: inherit
allowed-tools: mcp__mirdan__enhance_prompt, mcp__mirdan__validate_code_quality, mcp__mirdan__get_quality_standards, mcp__mirdan__validate_quick, mcp__sequential-thinking__sequentialthinking, mcp__enyal__enyal_recall, mcp__enyal__enyal_remember, mcp__enyal__enyal_traverse, mcp__context7__resolve-library-id, mcp__context7__query-docs, Read, Write, Edit, Glob, Grep, Bash
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

3. **Plan** (if complex) — If the task involves security, multi-file changes, or architectural decisions, use `mcp__sequential-thinking__sequentialthinking` to plan the approach before implementing

4. **Implement** — Write the code following the quality_requirements from step 1

5. **Validate** — Call `mcp__mirdan__validate_code_quality` on each changed file:
   - Set `check_security=true` if touches_security was flagged
   - Fix all errors immediately
   - Note warnings for review

6. **Complete** — Confirm:
   - All validation errors resolved
   - Quality requirements from enhance_prompt satisfied
   - No placeholder code (AI001) or hallucinated imports (AI002)
