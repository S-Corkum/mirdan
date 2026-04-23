---
name: scan
description: "[DEPRECATED] Retired in 2.1.0 — call scan_conventions / scan_dependencies MCP tools directly"
argument-hint: "(deprecated)"
model: inherit
allowed-tools:
---

# /scan — DEPRECATED

**DEPRECATED: /scan is retired in 2.1.0, call scan_conventions / scan_dependencies directly**

For convention scanning, call `mcp__mirdan__scan_conventions` directly.
For dependency vulnerability scanning, call `mcp__mirdan__scan_dependencies`.
Both MCP tools take parameters directly and return structured results.

This stub will be removed in 2.2.0. Update any automation or muscle memory now.

See `CHANGELOG.md` "2.1.0" section under "Migration from 2.0.x" for the full
retirement map.
