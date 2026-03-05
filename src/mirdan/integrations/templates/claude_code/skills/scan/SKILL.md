---
name: scan
description: "Convention scanning — scan codebase for patterns, violations, and AI quality issues"
argument-hint: "Optional: path or language to scope the scan"
model: inherit
allowed-tools: mcp__mirdan__scan_conventions, mcp__mirdan__validate_code_quality, mcp__mirdan__get_quality_standards, mcp__enyal__enyal_recall, mcp__enyal__enyal_remember, Read, Glob, Grep
---

# /scan — Convention Scanner

Scan the codebase for quality violations and convention patterns.

## Dynamic Context

Recent changes:
```
!`git diff --stat HEAD 2>/dev/null | tail -5`
```

## Workflow

1. **Scope** — Identify files to scan (all Python, staged changes, or specific path from argument)

2. **Recall** — Call `mcp__enyal__enyal_recall("conventions patterns")` to load known project conventions for comparison

3. **Scan** — Call `mcp__mirdan__scan_conventions` to discover patterns and violations across the codebase

4. **Standards** — Call `mcp__mirdan__get_quality_standards` for the project language to verify findings against rules

5. **Report** — Summarize:
   - Violations by rule ID and count
   - New patterns discovered
   - Conventions that differ from stored knowledge

6. **Store** — If new high-confidence conventions discovered, call `mcp__enyal__enyal_remember` to persist them
