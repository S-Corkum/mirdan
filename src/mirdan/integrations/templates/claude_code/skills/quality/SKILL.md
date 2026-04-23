---
name: quality
description: "[DEPRECATED] Retired in 2.1.0 — call mcp__mirdan__validate_code_quality directly"
argument-hint: "(deprecated)"
model: inherit
allowed-tools:
---

# /quality — DEPRECATED

**DEPRECATED: /quality is retired in 2.1.0, call mcp__mirdan__validate_code_quality directly**

For quality validation of a single file or snippet, call
`mcp__mirdan__validate_code_quality` directly with the code and language.
For plan-level quality gating, use `/plan-verify <plan-path>`.

This stub will be removed in 2.2.0. Update any automation or muscle memory now.

See `CHANGELOG.md` "2.1.0" section under "Migration from 2.0.x" for the full
retirement map.
