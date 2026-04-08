# Local Intelligence Layer — Quickstart

Get mirdan's local LLM running in 5 minutes. Saves 30-45% of your paid API tokens.

## Prerequisites

- mirdan installed (`pip install mirdan`)
- 16GB+ RAM (works on any modern laptop)
- One of: Ollama installed OR Python with pip

## Step 1: Run the setup wizard

```bash
mirdan llm setup
```

The wizard detects your hardware, recommends the right model, downloads it, and configures everything. Follow the prompts.

## Step 2: Configure your IDE

```bash
mirdan init --claude-code   # For Claude Code
mirdan init --cursor        # For Cursor IDE or Cursor CLI
```

This generates hooks that automatically call mirdan before and after the paid model runs.

## Step 3: Start coding

That's it. mirdan now:

- **Triages** your tasks (trivial ones never hit the paid model)
- **Runs lint/typecheck/tests** locally and only sends failures to the paid model
- **Enriches validation** with false-positive filtering and root cause analysis

## Verify it's working

```bash
mirdan llm status
```

You should see your hardware profile, loaded model, and enabled features.

## What's happening under the hood

When you type a prompt in your IDE, mirdan's hook fires first. A small local model (Gemma 4, ~3-5GB) classifies your task. Trivial tasks like "fix unused import" are tagged as LOCAL_ONLY and handled with zero paid tokens.

For tasks that need the paid model, mirdan enriches the quality context — gathering conventions, running linters, checking types — so the paid model can focus on writing code instead of exploring the codebase.

After you write code, mirdan runs ruff, mypy, and pytest locally, parses the output with the local LLM, auto-fixes lint issues, and only reports complex failures to the paid model.

## Next steps

- [Configuration Reference](llm-configuration.md) — tune for your workflow
- [IDE Integration](llm-ide-integration.md) — detailed per-IDE setup
- [Troubleshooting](llm-troubleshooting.md) — if something isn't working
