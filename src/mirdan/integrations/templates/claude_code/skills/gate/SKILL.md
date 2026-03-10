---
name: gate
description: "Quality gate check — validate all changed files pass mirdan standards before committing or marking task complete"
argument-hint: "Optional: --format text|json"
model: inherit
allowed-tools: mcp__mirdan__validate_code_quality, mcp__mirdan__get_quality_standards, mcp__mirdan__validate_quick, mcp__enyal__enyal_recall, mcp__enyal__enyal_remember, Read, Glob, Grep, Bash
---

# /gate — Quality Gate

Run the full quality gate before committing or completing a task.

## Dynamic Context

Changed files:
```
!`git diff --stat HEAD 2>/dev/null | tail -10`
```

## Workflow

1. **Identify** — Find all files changed in this session:
   - `git diff --name-only HEAD` for uncommitted changes
   - `git diff --staged --name-only` for staged files

2. **Gate** — Run `Bash("uvx mirdan gate --format text")` to check overall quality gate status

3. **Validate** — For any file that failed, call `mcp__mirdan__validate_code_quality` with `check_security=true` to get specific violations

4. **Fix** — Address all errors. Warnings should be reviewed but may be accepted with justification.

5. **Re-gate** — Run the gate again to confirm PASS status

6. **Persist** — If validation produced `knowledge_entries`, store them via `mcp__enyal__enyal_remember` with the suggested tags and scope

7. **Complete** — Only mark task complete if gate returns PASS
