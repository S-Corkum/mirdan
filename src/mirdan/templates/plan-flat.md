---
plan: <slug>
status: draft
author: <author>
created: <YYYY-MM-DD>
target: <version or milestone>
---

# Plan: <title>

<!--
This is a WORKED EXAMPLE. Replace the content but keep the shape and the level of
concreteness. Every File path is real (Read/Glob-verified). Every interface is
tagged [NEW] or [EXISTING]; [EXISTING] cites file:line, [NEW] is created by a step.
Include a Low-Level Design subsection ONLY if it applies — delete inapplicable
headings, do not write "N/A".
-->

## Research Notes (Pre-Plan Verification)

All facts verified via Read/Grep/tool-call on 2026-06-04.

### Files Verified
- `src/app/cache.py`: `class LruCache` at line 12; `get(key)` at line 28, no TTL support.
- `src/app/config.py`: `Settings` model at line 9; no `cache_ttl` field.

### Dependencies Confirmed
- `cachetools`: 5.3.2 (pyproject.toml line 24).

### API Documentation
- `cachetools.TTLCache(maxsize, ttl)` — evicts entries older than `ttl` seconds.

### Conventions
- Config fields are pydantic `Field(default=...)` with a description (enyal: config-convention).

## Low-Level Design

### Interfaces & Signatures
- `def get(self, key: str) -> Any | None` [EXISTING] — `src/app/cache.py:28`
- `class TtlCache(maxsize: int, ttl_seconds: int)` [NEW] — created by Step 2
- `Settings.cache_ttl_seconds: int` [NEW] — created by Step 1

### Error Taxonomy
- Missing key → return `None` (not an exception; expected outcome).
- `ttl_seconds <= 0` → raise `ValueError` at construction (config validation layer).

### Design Decisions
- **Caching strategy:** use `cachetools.TTLCache` rather than hand-rolled expiry —
  the dependency is already present and battle-tested (rationale grounded in
  pyproject.toml:24).

## Plan Steps

### Step 1: Add cache_ttl_seconds to Settings
**File:** `src/app/config.py`
**Action:** Edit
**Details:** After line 9 add `cache_ttl_seconds: int = Field(default=300, gt=0, description="Cache entry TTL in seconds")`.
**Depends On:** —
**Verify:** Read config.py; confirm field present and `Settings().cache_ttl_seconds == 300`.
**Grounding:** Read of config.py:9 (Settings model); convention from enyal config-convention.

### Step 2: Add TtlCache
**File:** NEW: `src/app/ttl_cache.py`
**Action:** Write
**Details:** Define `class TtlCache` wrapping `cachetools.TTLCache(maxsize, ttl_seconds)`; expose `get(key)` and `set(key, value)`.
**Depends On:** Step 1
**Verify:** Read ttl_cache.py; confirm `TtlCache(8, 1)` evicts after 1s.
**Grounding:** context7 for `cachetools.TTLCache` signature; parent dir `src/app/` Glob-confirmed.

## Self-Check
- [ ] Research Notes present with tool citations
- [ ] Every interface tagged [NEW]/[EXISTING]; [EXISTING] cites file:line; [NEW] created by a step
- [ ] Every step has File, Action, Details, Depends On, Verify, Grounding
- [ ] No vague language ("should", "probably", "around line X", "I think", "assume")
- [ ] Every File path verified or marked NEW with parent Glob-confirmed
- [ ] Steps are atomic — one action each
