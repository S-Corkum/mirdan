# /plan-execute — Hardware-Adaptive Plan Execution

Walk a three-layer plan's subtasks in dependency order. On Cursor, routes
between Option A (inline execution on your selected model) and Option B
(MCP-proxied diff generation via local Gemma 4) based on hardware and local
LLM health.

## Usage

`/plan-execute <plan-path> [--skip-verify] [--mode inline|proxied] [--dry-run]`

## Workflow

1. Read the plan; extract `brief:` from frontmatter.
2. **Template-shape gate**: if no `brief:` field OR body lacks both
   `## Epic Layer` and `## Story Layer` sections, emit: `/plan-execute only
   supports three-layer plans. Flat-template plans use the legacy execution
   path.` and exit non-zero.
3. Read the brief.
4. **Pre-flight verify** — unless `--skip-verify`:
   call `mcp__mirdan__verify_plan_against_brief(plan_path, brief_path)`.
   If `verified=false`, display the coverage report and exit.
5. Call `mcp__mirdan__mirdan_health` ONCE. Cache `recommended_mode`,
   `local_llm_available`, `vram_gb`, `model_in_use` for this run.
6. Parse subtasks. Build a dependency graph from `Depends on:` fields.
   Topologically sort. Reject on cycles.
7. If `--dry-run`: print ordered subtask list with dependencies, exit.
8. **Dispatch loop** — for each subtask in order:
   - If `--mode inline` OR `recommended_mode == "inline"` OR
     `local_llm_available == false`: **Option A** — execute the subtask
     inline with your (Cursor-selected) model via Edit/Write.
   - Else (`--mode proxied` OR `recommended_mode == "proxied"`): **Option B**
     — call `mcp__mirdan__propose_subtask_diff(subtask_yaml, file_context)`.
     If `halted == false`: review the returned diff; apply via Edit.
     If `halted == true`: do NOT silently fall back. Surface the halt
     (`halt_reason`, `reason`) to the user and stop the loop.
   - After success, call `mcp__mirdan__validate_code_quality` on the file
     touched.
9. **Summary** — on loop completion, report: N subtasks completed, M halted,
   files modified, validation issues.

## Halt semantics

Subtask halt is a signal, not a failure. Cheap models halt when the plan's
grounding no longer matches reality (file moved, API changed, line number
drifted). Do NOT automatically retry, auto-fall-back from B to A, or guess
an alternative. Surface the halt to the user so they can correct the plan.

## Mode overrides

- `--mode inline` — force Option A even if local LLM is healthy.
- `--mode proxied` — force Option B even if hardware is marginal.

Default is auto-routing from `mirdan_health` output.
