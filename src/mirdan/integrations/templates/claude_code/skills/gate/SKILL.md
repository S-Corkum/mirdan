---
name: gate
description: "[DEPRECATED] Retired in 2.1.0 — call validate_code_quality + validate_quick directly, or /plan-verify for plan-level gating"
argument-hint: "(deprecated)"
model: inherit
allowed-tools:
---

# /gate — DEPRECATED

**DEPRECATED: /gate is retired in 2.1.0, use validate_code_quality + validate_quick directly**

For per-file quality gating, call `mcp__mirdan__validate_code_quality` and
`mcp__mirdan__validate_quick` directly. For plan-level gating (before
`/plan-execute`), use `/plan-verify <plan-path>` which runs automatically as
a pre-flight check when executing a plan.

This stub will be removed in 2.2.0. Update any automation or muscle memory now.

See `CHANGELOG.md` "2.1.0" section under "Migration from 2.0.x" for the full
retirement map.
