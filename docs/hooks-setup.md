# Mirdan Hook Setup Guide

Mirdan hooks automatically validate code quality after AI edits and before task completion.

## Quick Setup

```bash
# Cursor (recommended for Cursor 1.7+)
mirdan init --cursor

# Claude Code
mirdan init --claude-code

# Both platforms
mirdan init --all
```

## Platform-Specific Setup

### Cursor (1.7+)

```bash
mirdan init --cursor
```

Generates `.cursor/hooks.json` with **prompt-type hooks** that leverage Cursor's native hook system:

| Event | Behavior |
|-------|----------|
| `afterFileEdit` | Calls `validate_code_quality` on changed code, fixes errors automatically |
| `preToolUse` | Security review before Write/Edit (SQL injection, secrets, command injection) |
| `stop` | Final quality gate — verifies all files validated before completion (loop_limit: 3) |
| `beforeSubmitPrompt` | Suggests `enhance_prompt` for quality requirements |

Also generates:
- `.cursor/mcp.json` — MCP server registration with `MIRDAN_TOOL_BUDGET`
- `.cursor/rules/*.mdc` — Language-specific project rules
- `AGENTS.md` — Enhanced with quality checkpoints, AI/security rules, quality thresholds
- `BUGBOT.md` — Structured PR review rules with regex patterns

#### Hook Stringency

Hooks are generated at COMPREHENSIVE level by default. The stringency levels are:

| Level | Events | Best For |
|-------|--------|----------|
| MINIMAL | afterFileEdit, stop | Low-friction onboarding |
| STANDARD | + preToolUse | Daily development |
| COMPREHENSIVE | + beforeSubmitPrompt | Teams, production projects |

#### Tool Budget

The generated `.cursor/mcp.json` includes `MIRDAN_TOOL_BUDGET=3` by default, exposing the top 3 priority tools:

1. `validate_code_quality` (always included)
2. `validate_quick` (always included with budget >= 2)
3. `enhance_prompt` (included with budget >= 3)
4. `get_quality_standards` (budget >= 4)
5. `get_quality_trends` (budget = 5 or unset)

Adjust via the `MIRDAN_TOOL_BUDGET` env var in `.cursor/mcp.json`.

### Claude Code

```bash
mirdan init --claude-code
```

Generates:
- `.claude/hooks.json` — PostToolUse and Stop hook configuration
- `.mcp.json` — MCP server registration
- `.claude/rules/mirdan-*.md` — Quality rules
- `.claude/skills/` — `/mirdan:code`, `/mirdan:debug`, `/mirdan:review`
- `.claude/agents/quality-gate.md` — Background quality validation

### Pre-commit (any platform)

```bash
mirdan init --hooks
```

Generates `.pre-commit-config.yaml` at project root.

Install with:
```bash
pip install pre-commit
pre-commit install
```

## Hook Behavior

### Cursor Hooks (prompt-type)

All Cursor hooks use the **prompt type**, meaning the LLM evaluates the hook instruction in context (with access to MCP tools). This is more powerful than command-type hooks because the AI can reason about what to validate.

- **afterFileEdit**: Non-blocking — reports and fixes issues inline
- **preToolUse**: Advisory — provides security guidance before writes
- **stop**: Blocking loop — re-runs up to 3 times until all files are validated
- **beforeSubmitPrompt**: Advisory — suggests quality enhancement

### Claude Code Hooks (command-type)

- **PostToolUse**: Runs `mirdan validate-quick` after Write/Edit tool calls
- **Stop**: Runs validation summary before completing

### Pre-commit Hooks

- **Blocking**: Prevents commits with quality errors (exit 1)
- **Format**: Text output for human readability

## Configuration

Hooks use `.mirdan/config.yaml` for settings. Key options:

```yaml
linters:
  auto_detect: true
  enabled_linters: [ruff, mypy]
  timeout: 30.0
```

## Idempotent Generation

`mirdan init --cursor` will **skip** generating `.cursor/hooks.json` if it already exists, respecting any customizations you've made. To regenerate, delete the file first.

## Adding --lint to hooks

To include external linter checks in hooks, modify the command:

```bash
mirdan validate --file "$FILE" --format text --lint
```
