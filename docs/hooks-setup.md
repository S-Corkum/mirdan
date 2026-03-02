# Mirdan Hook Setup Guide

Mirdan hooks automatically validate code quality after AI edits and before commits.

## Quick Setup

```bash
mirdan init --hooks
```

This auto-detects your IDE and installs appropriate hooks.

## Platform-Specific Setup

### Cursor

```bash
mirdan init --hooks --cursor
```

Installs:
- `.cursor/hooks/post-edit.sh` — validates files after Write/Edit tool calls
- `.git/hooks/pre-commit` — validates staged files before commit

### Claude Code

```bash
mirdan init --hooks --claude-code
```

Installs:
- `.claude/hooks.json` — PostToolUse hook configuration

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

- **Non-blocking by default**: hooks report issues but don't prevent edits
- **Pre-commit hooks**: block commits with errors (exit 1)
- **Format**: text output for human readability

## Manual Installation

### Cursor post-edit hook

Copy `hooks/cursor/post-edit.sh` to `.cursor/hooks/post-edit.sh`:
```bash
cp $(python -c "import mirdan; print(mirdan.__file__)")/../../hooks/cursor/post-edit.sh .cursor/hooks/
chmod +x .cursor/hooks/post-edit.sh
```

### Claude Code hooks

Copy `hooks/claude-code/hooks.json` to `.claude/hooks.json`:
```bash
cp $(python -c "import mirdan; print(mirdan.__file__)")/../../hooks/claude-code/hooks.json .claude/hooks.json
```

### Git pre-commit

```bash
cp hooks/cursor/pre-commit.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## Configuration

Hooks use `.mirdan/config.yaml` for settings. Key options:

```yaml
linters:
  auto_detect: true
  enabled_linters: [ruff, mypy]
  timeout: 30.0
```

## Adding --lint to hooks

To include external linter checks in hooks, modify the command:

```bash
mirdan validate --file "$FILE" --format text --lint
```
