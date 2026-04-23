---
name: plan-execute
description: "Execute a three-layer plan by dispatching subtasks to cheap executor"
argument-hint: "<plan-path> [--skip-verify] [--dry-run]"
model: inherit
allowed-tools: mcp__mirdan__verify_plan_against_brief, mcp__mirdan__validate_code_quality, Read, Grep, Bash, Task
---

# /plan-execute — Dispatch Three-Layer Plan to Cheap Executor

Walk a three-layer plan's subtasks in dependency order, dispatching each to
the `cheap-executor` agent (Haiku). Halts on grounding mismatch.

## Dynamic Context

Recent plans:
```
!`ls -t docs/plans/*.md 2>/dev/null | head -5`
```

## Workflow

1. **Parse args** — `<plan-path>` required. Flags: `--skip-verify`, `--dry-run`.

2. **Template-shape gate** — read plan frontmatter and body. If frontmatter
   lacks `brief:` field OR body lacks both `## Epic Layer` and `## Story Layer`
   sections: emit:
   `/plan-execute only supports three-layer plans. Flat-template plans use
   the legacy execution path. See CHANGELOG migration notes.` and exit
   non-zero.

3. **Read brief** — from the plan's frontmatter `brief:` field.

4. **Pre-flight verify** — unless `--skip-verify`:
   call `mcp__mirdan__verify_plan_against_brief(plan_path, brief_path)`.
   If `verified=false`, display the coverage report and exit. Do not dispatch.

5. **Parse subtasks** — extract every `##### <id> — <title>` block with its
   6 grounding fields. Build a dependency graph from each subtask's
   `Depends on:` field.

6. **Compute order** — topological sort over the dependency graph. Reject on
   cycles.

7. **If `--dry-run`** — print the ordered subtask list with dependency arrows
   and exit.

8. **Dispatch loop** — for each subtask in order:
   - Assemble subtask block as self-contained YAML-like text
   - `Agent(subagent_type="cheap-executor", prompt=<subtask block>)`
   - Wait for result
   - On success: call `mcp__mirdan__validate_code_quality` on the file touched
   - On halted=true: STOP the loop; emit the halted subtask details; user
     decides retry / fix / abandon

9. **Summary** — on loop completion, report: N subtasks completed, M halted,
   files modified, validation issues (if any).

## Halt semantics

Subtask halt is a signal, not a failure. Cheap models halt because the plan's
grounding no longer matches reality (file moved, API changed, line number
drifted). Do NOT automatically retry, fall back, or guess an alternative.
Surface the halt to the user so they can correct the plan.
