# mirdan Claude Code Plugin

AI Code Quality Runtime for Claude Code.

## Installation

```bash
claude --plugin-dir ./mirdan-plugin
```

Or install from exported plugin:

```bash
mirdan plugin export --output-dir ./mirdan-plugin
claude --plugin-dir ./mirdan-plugin
```

## What's Included

- **MCP Server**: mirdan quality validation tools (5 tools)
- **Skills**: `/code`, `/debug`, `/review` slash commands
- **Agents**: `quality-gate` subagent for automated validation
- **Hooks**: Pre/Post tool-use quality checks

## Tools Available

| Tool | Purpose |
|------|---------|
| `enhance_prompt` | Quality-aware prompt enhancement |
| `validate_code_quality` | Full code quality validation |
| `validate_quick` | Fast security-only validation |
| `get_quality_standards` | Language/framework standards |
| `get_quality_trends` | Quality score history |
