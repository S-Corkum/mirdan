---
name: debug
description: "[DEPRECATED] Retired in 2.1.0 — debug inline and run /plan-verify if the fix touches a plan-tracked file"
argument-hint: "(deprecated)"
model: inherit
allowed-tools:
---

# /debug — DEPRECATED

**DEPRECATED: /debug is retired in 2.1.0, use inline debugging instead**

Debug inline using Read, Grep, and the main chat. After fixing, call
`mcp__mirdan__validate_code_quality` on the modified file to confirm no new
violations were introduced. If the fix touches a file tracked in a
three-layer plan at `docs/plans/`, run `/plan-verify <path>` afterward to
confirm plan coverage is still intact.

This stub will be removed in 2.2.0. Update any automation or muscle memory now.

See `CHANGELOG.md` "2.1.0" section under "Migration from 2.0.x" for the full
retirement map.
