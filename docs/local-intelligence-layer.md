# Local Intelligence Layer

## What it does

mirdan 2.0.0 adds a Local Intelligence Layer that offloads mundane work from expensive paid LLMs to a small, fast model running on the developer's own machine. The paid model (Claude Opus, Sonnet, etc.) focuses exclusively on what it's best at — complex reasoning, architecture decisions, and writing code. Everything else — task triage, linting, type checking, running tests, parsing failures, validating output — happens locally for free.

**Realistic savings:** Claude Code: 30-45% fewer paid tokens on 16GB. Cursor IDE/CLI: 20-30% on 16GB. Up to 50-70% on 64GB+ Apple Silicon with full features.

## How it works

```
User prompt → Hook fires → Local LLM triages task
                              ├── LOCAL_ONLY → handle locally, zero paid tokens
                              └── PAID_REQUIRED → enrich context, send to paid model

Paid model writes code → Hook fires → Local LLM runs checks
                                         ├── ruff (lint) → auto-fix
                                         ├── mypy (typecheck)
                                         └── pytest (tests)
                                      → Only complex failures sent to paid model
```

Two integration paths exist depending on your IDE:

**Hook path (Claude Code):** IDE hooks call mirdan before/after the paid model. Local LLM handles triage and checks. Results are injected into the paid model's context. <5ms overhead via HTTP sidecar.

**MCP path (Cursor IDE/CLI):** Paid model calls mirdan's MCP tools (enhance_prompt, validate_code_quality). mirdan enriches the response with local LLM analysis. `.mdc` rules mandate the tool calls.

## Hardware requirements

| RAM | Backend | Model | Active Memory | Features |
|-----|---------|-------|---------------|----------|
| 16GB | llama-cpp-python | Gemma 4 E4B Q3 | ~4.5GB | Triage + Checks + Validation |
| 16GB | Ollama | Gemma 4 E2B Q4 | ~3.5GB | Same (simpler setup) |
| 32GB | Either | Gemma 4 E4B Q4+ | ~5GB | Better quality |
| 64GB+ AS | Either | E4B + 31B | ~22GB | + Optimization + Research |

Everything runs locally. No remote servers. Compatible with Enyal (~1GB).

The model unloads after 5 minutes idle (configurable via `model_keep_alive`), freeing RAM for other work. Reloads in 2-5 seconds on next use.

## Supported IDEs

| Feature | Claude Code | Cursor IDE | Cursor CLI |
|---------|------------|-----------|------------|
| MCP tools | All 7 | 3 (budget) | 3 (budget) |
| Hook context injection | Yes | No | No |
| Local LLM triage | Yes | Yes | Yes |
| Local LLM checks | Yes | Yes | Yes |
| Primary injection path | Hooks | .mdc rules | .mdc rules |

## Feature breakdown

### Triage

Classifies tasks into four categories before the paid model sees them:

- **LOCAL_ONLY** (0 paid tokens): "fix unused import", "format with black"
- **LOCAL_ASSIST** (2K tokens): "add docstrings", "add type annotations"
- **PAID_MINIMAL** (4K tokens): "add GET /users endpoint", "write unit tests"
- **PAID_REQUIRED** (unlimited): "implement JWT auth", "refactor to Strategy pattern"

Security-sensitive, highly ambiguous, and planning tasks always go to PAID_REQUIRED regardless of LLM classification. Low-confidence (<0.7) classifications are escalated.

### Check Runner

Runs lint, typecheck, and tests as subprocesses:

1. `ruff check` — auto-fixes if configured
2. `mypy` — type checking
3. `pytest -x --tb=short` — tests with configurable timeout

When the local LLM is available, it parses combined output and classifies each issue as auto_fixed, trivial, or complex. Only complex issues are surfaced to the paid model.

### Smart Validation

Enriches validate_code_quality with a single combined LLM call:

- **False positive filtering** — identifies rule violations that aren't actual issues
- **Root cause grouping** — clusters related violations under a single cause
- **Fix suggestions** — generates code fixes (re-validated through the rule engine)
- **Sanity cap** — if >40% of violations are marked as false positives, all FP assessments are rejected

### Prompt Optimization (64GB+ Apple Silicon only)

The 31B BRAIN model crafts optimized prompts per target paid model:

- **Opus:** Concise, constraints-focused, XML sections
- **Sonnet:** Structured with examples, clear success criteria
- **Haiku:** Step-by-step, all context inline, explicit format

Includes context pruning — IDE-aware (Cursor gets more aggressive pruning since it has no context budget visibility).

### Research Agent (64GB+ Apple Silicon, experimental)

The BRAIN model autonomously gathers context by calling MCPs (context7, enyal, github) in an agentic loop. Max 5 iterations, 10K token budget. Replaces ContextAggregator for THOROUGH ceremony tasks.

## Privacy and security

- Everything runs locally — no data leaves the developer's machine
- Prompt injection mitigated via `<CODE_FOR_ANALYSIS>` delimiters
- False positive ratio capped at 40% to prevent model hallucination
- LLM-generated fixes are re-validated through the rule engine before suggestion
- All features degrade gracefully — rules-only mode is always available
