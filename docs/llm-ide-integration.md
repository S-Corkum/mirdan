# IDE Integration Guide

## Claude Code

### Automatic setup

```bash
mirdan init --claude-code
```

### What it configures

- `.mcp.json` — mirdan MCP server
- `.claude/hooks.json` — command hooks for triage + check + validation
- `.claude/rules/` — quality enforcement rules

### Hook behavior

When `llm.enabled: true`:

| Event | Hook Type | Action |
|-------|-----------|--------|
| UserPromptSubmit | command | Triage task via sidecar. Output injected into Claude's context. |
| PostToolUse | command | Quick rule-based validation after Write/Edit. |
| Stop | command | Run `mirdan check --smart`. Results injected before completion. |

Claude Code hooks inject stdout directly into the model's context. This is the strongest integration — the paid model sees triage results and check output without spending tokens to request them.

### How the sidecar works

The hook script tries the HTTP sidecar first (reuses the warm model from the MCP server, <5ms). If the sidecar isn't running, it falls back to `mirdan triage --stdin` (cold start, 3-8s).

```bash
# Generated hook script
PORT_FILE=".mirdan/sidecar.port"
if [ -f "$PORT_FILE" ]; then
  curl -s --max-time 10 "http://localhost:$(cat $PORT_FILE)/triage" --data-binary @-
else
  mirdan triage --stdin 2>/dev/null
fi
```

## Cursor IDE

### Automatic setup

```bash
mirdan init --cursor
```

### What it configures

- `.cursor/mcp.json` — mirdan MCP server with `MIRDAN_TOOL_BUDGET=3`
- `.cursor/hooks.json` — command hooks
- `.cursor/rules/*.mdc` — quality enforcement rules with `alwaysApply: true`

### Key difference from Claude Code

Cursor hooks **cannot inject context into the model**. The `beforeSubmitPrompt` hook returns a boolean only — it cannot add text to the model's context.

mirdan's Cursor strategy uses **`.mdc` rules** as the primary injection mechanism. Rules with `alwaysApply: true` are included in every prompt like a system instruction:

```yaml
---
description: mirdan quality enforcement with local LLM optimization
alwaysApply: true
---
MANDATORY: Before writing ANY code, call the enhance_prompt MCP tool.
This tool gathers context locally using a free local model, saving significant
paid API tokens. After writing code, call validate_code_quality for enriched
validation with false-positive filtering and root-cause analysis.
```

### Tool budget

Cursor exposes 3 MCP tools. mirdan's priority order:

1. `validate_code_quality` — most valuable (quality gate)
2. `validate_quick` — fast validation (hooks)
3. `enhance_prompt` — context enrichment

### Hook behavior

| Event | Action |
|-------|--------|
| sessionStart | Injects static context about mirdan quality and local LLM |
| beforeShellExecution | Shell governance (prevents duplicate lint/test runs) |
| afterFileEdit | Quick rule-based validation |
| stop | Quality gate check |

## Cursor CLI

Same configuration as Cursor IDE — Cursor CLI reads `.cursor/` config.

```bash
mirdan init --cursor  # Same command
```

## Why Claude Code gets higher savings

Claude Code hooks inject triage results **before** the model processes — the model never wastes tokens exploring trivial tasks. Cursor hooks can't inject, so the model calls MCP tools itself (spending tokens to read the enriched response).

| IDE | Mechanism | Savings (16GB) |
|-----|-----------|----------------|
| Claude Code | Hook injection + MCP | 30-45% |
| Cursor IDE | Rules + MCP | 20-30% |
| Cursor CLI | Rules + MCP | 20-30% |

Both benefit from local LLM enrichment, but Claude Code eliminates more exploration tokens.
