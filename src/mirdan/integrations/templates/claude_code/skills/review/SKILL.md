---
name: review
description: "Code review with mirdan quality standards enforcement"
argument-hint: "File or PR to review"
user-invocable: true
model: inherit
context: fork
allowed-tools: mcp__mirdan__validate_code_quality, mcp__mirdan__get_quality_standards, mcp__mirdan__get_quality_trends, Read, Glob, Grep
---

# /review — Quality Standards Code Review

Review code or diffs against mirdan quality standards in an isolated context.

## Dynamic Context

Recent changes:
```
!`git diff --stat HEAD 2>/dev/null | tail -10`
```

## Workflow

1. **Read** — Read the code to review (specific file or recent changes)

2. **Validate** — Call `mcp__mirdan__validate_code_quality` on the code
   - Set `check_security=true` for security-sensitive files
   - Set `check_architecture=true` for structural analysis

3. **Standards** — Call `mcp__mirdan__get_quality_standards` for the language
   - Compare code against language best practices

4. **Report** — Present findings organized by severity:
   - **Errors** — Must fix before merge
   - **Warnings** — Should fix, note if intentional
   - **Info** — Suggestions for improvement

5. **Summary** — Overall assessment:
   - Quality score
   - Key areas of concern
   - Positive observations
