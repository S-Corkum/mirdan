---
name: review
description: "[DEPRECATED] Retired in 2.1.0 — use /plan-review --stakes high OR call validate_code_quality directly"
argument-hint: "(deprecated)"
user-invocable: true
model: inherit
allowed-tools:
---

# /review — DEPRECATED

**DEPRECATED: /review is retired in 2.1.0, use /plan-review --stakes high instead**

For judgment review of a plan: `/plan-review --stakes high <plan-path>` —
produces the 5-section structured output against the shared rubric at
`templates/plan-review-rubric.md`.

For reviewing arbitrary files or PRs without a plan context: call
`mcp__mirdan__validate_code_quality` directly on the file contents.

This stub will be removed in 2.2.0. Update any automation or muscle memory now.

See `CHANGELOG.md` "2.1.0" section under "Migration from 2.0.x" for the full
retirement map.
