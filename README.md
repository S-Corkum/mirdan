# Mirdan

AI Code Quality Orchestrator — **saves 30-45% of your paid AI coding tokens** by running a local intelligence layer that handles triage, linting, type checking, test running, and validation so expensive models like Claude Opus focus on writing code.

[![PyPI version](https://img.shields.io/pypi/v/mirdan.svg)](https://pypi.org/project/mirdan/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

```bash
uv tool install mirdan                  # Install mirdan
mirdan llm setup                        # Auto-installs backend, downloads model, configures
mirdan init --claude-code               # or --cursor
# Done. Quality enforcement + local intelligence is now automatic.
```

Works with **Claude Code**, **Cursor IDE**, and **Cursor CLI**. Runs on 16GB laptops. Everything stays local.

---

## Local Intelligence Layer (New in 2.0)

Mirdan 2.0 offloads mundane work to a small local model (Gemma 4) running on your machine. The paid model focuses exclusively on complex reasoning and writing code.

**Before you code:**
- **Triage** — classifies tasks. Trivial tasks (fix an import, format a file) never hit the paid model. Zero tokens spent.
- **Research** — gathers codebase context, library docs, and project conventions locally (64GB+ only).

**After you code:**
- **Check Runner** — runs ruff, mypy, pytest locally. LLM parses output, auto-fixes lint, reports only complex failures.
- **Smart Validation** — 64 quality rules + LLM false-positive filtering, root cause grouping, and fix suggestions.
- **Auto-Fix** — `mirdan check --smart --fix` applies LLM-generated search/replace fixes with verification.

| Hardware | What Runs | Token Savings |
|----------|-----------|---------------|
| 16GB laptop | Gemma 4 E4B Q3 — triage, checks, validation, auto-fix | 30-45% |
| 32GB | Gemma 4 E4B Q3 — same features, more headroom | 35-50% |
| 64GB+ Apple Silicon | + Gemma 4 31B for prompt optimization and research | 50-70% |

### Quick Setup

```bash
mirdan llm setup              # Detects hardware, installs backend, downloads model, configures
mirdan init --claude-code     # or --cursor
mirdan llm status             # Verify it's working
```

Everything runs locally. No remote servers. No data leaves your machine.

---

## Why Mirdan?

AI coding assistants generate code fast, but without guardrails they produce **slop**: hardcoded secrets, SQL injection, placeholder functions, hallucinated imports, bare except blocks, and code that looks right but fails in production.

Mirdan fixes this by intercepting your AI workflow at two points:

1. **Before coding** — `enhance_prompt` enriches your task with quality requirements, security constraints, and framework-specific standards so the AI produces better code from the start
2. **After coding** — `validate_code_quality` catches what slipped through: 64 rules covering security vulnerabilities, AI-specific antipatterns, and language best practices

Once installed, it runs invisibly through IDE hooks. You just code normally.

### What It Catches

Here's what mirdan flags on a single function of typical AI-generated code:

```python
API_KEY = "sk-proj-abc123456789"           # SEC001: hardcoded API key
def get_users(user_id):
    query = f"SELECT * FROM users WHERE id={user_id}"  # SEC005 + AI008: SQL injection
    result = eval(user_input)              # PY001: code injection via eval()
    data = requests.get(url, verify=False) # SEC007: SSL verification disabled
    try:
        process(data)
    except:                                # PY003: bare except
        pass
```

**Result:** Score 0.0/1.0, 6 errors, 3 warnings. With auto-fix, mirdan resolves 5 of these automatically.

### What It Adds to Your Prompts

When you ask an AI assistant to "Create a user auth endpoint in FastAPI with JWT tokens", mirdan's `enhance_prompt` detects this touches security and injects:

- **Framework standards:** "Use `Depends()` with `Annotated` for type-safe dependency injection"
- **Security constraints:** "Check that no secrets or credentials are hardcoded"
- **Quality requirements:** "Use Pydantic models for all request bodies and response schemas"
- **Verification steps:** "Ensure error handling covers all async operations"

The AI gets structured guidance instead of a bare prompt, producing better code on the first try.

---

## Quick Start

### Install

```bash
uv tool install mirdan                   # Install mirdan
mirdan llm setup                         # Auto-installs LLM backend + downloads model

# Optional extras:
uv tool install 'mirdan[ast]'            # + tree-sitter for TS/JS AST analysis
uv tool install 'mirdan[enterprise]'     # + truststore for corporate SSL inspection

# Upgrade:
uv tool upgrade mirdan

# Or with pip:
pip install mirdan
```

### Set Up Your IDE

```bash
mirdan init --claude-code    # Claude Code: hooks, rules, skills, agents
mirdan init --cursor         # Cursor: hooks, rules, AGENTS.md, BUGBOT.md
mirdan init --all            # Both IDEs
```

This generates everything — MCP server config, quality hooks, rules files, and agent definitions. When LLM is enabled, hooks also configure local triage and check runner. After init, your IDE automatically:

1. Enriches prompts with quality requirements before coding tasks
2. Validates code against security and quality rules after every edit
3. Runs a final quality gate before task completion

### Use From the Command Line

```bash
mirdan validate --file src/auth.py     # Validate a file
mirdan validate --staged               # Validate git staged changes
mirdan fix --file src/auth.py          # Auto-fix violations (pattern-based)
mirdan check --smart                   # Run lint + typecheck + test with LLM analysis
mirdan check --smart --fix src/        # Run checks AND auto-fix with local LLM
mirdan gate                            # CI/CD quality gate (exit 0 or 1)
mirdan scan --dependencies             # Check deps for known CVEs
mirdan scan --directory src/           # Discover codebase conventions
mirdan llm setup                       # Configure local LLM
mirdan llm status                      # Show LLM health, model, hardware
mirdan llm metrics                     # Token savings dashboard
```

---

## How It Works

Mirdan is an [MCP server](https://modelcontextprotocol.io/) — it connects to AI coding assistants (Claude Code, Cursor, Claude Desktop, or any MCP client) and provides quality enforcement tools.

```
┌──────────────────────────────────────────────────┐
│  Your AI Assistant (Claude Code / Cursor / etc)  │
│                                                  │
│  1. You type a coding task                       │
│  2. Hook triages task via local LLM ────┐        │
│  3. AI generates code with guidance     │        │
│  4. Hook runs lint/typecheck/test ◄─────┘        │
│  5. AI fixes only complex issues                 │
│  6. Quality gate passes → task complete          │
└──────────────────────────────────────────────────┘
         │                        ▲
         ▼                        │
┌──────────────────────────────────────────────────┐
│  Mirdan MCP Server + Local Intelligence Layer    │
│                                                  │
│  MCP Tools (unchanged):                          │
│  enhance_prompt         → Quality requirements   │
│  validate_code_quality  → 64 rules + LLM enrich │
│  validate_quick         → Fast security checks   │
│  get_quality_standards  → Language/framework ref  │
│  get_quality_trends     → Historical analysis    │
│  scan_dependencies      → CVE detection (OSV)    │
│  scan_conventions       → Convention discovery   │
│                                                  │
│  Local LLM (Gemma 4, runs on your machine):      │
│  Triage          → Classify tasks, save tokens   │
│  Check Runner    → Run ruff/mypy/pytest locally  │
│  Smart Validator → FP filtering, root causes     │
│  Auto-Fix        → Search/replace code fixes     │
│  HTTP Sidecar    → <5ms hook integration         │
└──────────────────────────────────────────────────┘
```

---

## Validation Rules

Mirdan ships with **64 rules** across 10 categories. No external services required — all rules run locally.

### AI Quality (AI001–AI008)

Rules that catch patterns unique to AI-generated code:

| Rule | What It Catches |
|------|-----------------|
| AI001 | Placeholder code — `raise NotImplementedError`, `pass` with TODO (skips `@abstractmethod`) |
| AI002 | Hallucinated imports — packages not in stdlib or project dependencies |
| AI003 | Over-engineering — unnecessary abstractions for simple operations |
| AI004 | Duplicate code blocks |
| AI005 | Inconsistent error handling patterns |
| AI006 | Unnecessary heavy imports where lighter alternatives exist |
| AI007 | Security theater — patterns that look secure but provide no protection |
| AI008 | Injection via f-strings — SQL, eval, exec, os.system with interpolation |

### Security (SEC001–SEC014)

| Rule | What It Catches |
|------|-----------------|
| SEC001–003 | Hardcoded secrets — API keys, passwords, AWS keys |
| SEC004–006 | SQL injection — string concat, f-strings, template literals |
| SEC007 | SSL/TLS verification disabled |
| SEC008–009 | Shell command injection via string formatting |
| SEC010 | JWT verification disabled |
| SEC011–013 | Graph database injection — Neo4j Cypher, Gremlin |
| SEC014 | Vulnerable dependencies — packages with known CVEs |

### Language-Specific

| Language | Rules | Key Checks |
|----------|-------|------------|
| Python | PY001–PY015 | eval/exec, bare except, mutable defaults, deprecated typing, unsafe pickle/yaml, subprocess shell, dead imports, unreachable code |
| JavaScript | JS001–JS005 | `var`, eval, document.write, innerHTML, child_process.exec |
| TypeScript | TS001–TS005 | eval, Function constructor, @ts-ignore, `as any`, innerHTML |
| Go | GO001–GO003 | Ignored errors, panic(), SQL via fmt.Sprintf |
| Java | JV001–JV007 | String ==, generic Exception, System.exit, Runtime.exec, unsafe deserialization |
| Rust | RS001–RS002 | .unwrap(), empty .expect() |

Plus **ARCH001–003** / **TSARCH001–004** (function length, file length, nesting depth, missing return types), **RAG001–002** (chunk overlap, deprecated loaders).

Python rules PY001–PY004 use **AST-based validation** — eliminating false positives from strings and comments. When `mirdan[ast]` is installed, TypeScript/JavaScript architecture checks use **tree-sitter** for accurate function length, nesting depth, and return type analysis.

**32 rules** support automatic fixes via `mirdan fix`.

---

## Language and Framework Support

**Languages:** Python, TypeScript, JavaScript, Go, Java, Rust

**33 framework standards** — mirdan knows the idioms, best practices, and common pitfalls for each:

React, React Native, Next.js, Nuxt, Vue, SvelteKit, Astro, Flutter, Tailwind, FastAPI, Django, Express, NestJS, Echo, Gin, Spring Boot, Micronaut, Quarkus, Drizzle, Neo4j, Supabase, Convex, Pinecone, Qdrant, Milvus, Weaviate, ChromaDB, FAISS, LangChain, LangGraph, CrewAI, DSPy, tRPC

When `enhance_prompt` detects a framework, it injects framework-specific quality requirements (e.g., "Use `Depends()` with `Annotated`" for FastAPI, "Prefer server components" for Next.js).

---

## Quality Profiles

Profiles tune enforcement levels across 8 dimensions. Choose one that matches your project:

| Profile | Security | Architecture | Testing | AI Slop | Dep Security | Best For |
|---------|----------|-------------|---------|---------|-------------|----------|
| **default** | 0.7 | 0.5 | 0.7 | 0.7 | 0.7 | General-purpose projects |
| **startup** | 0.7 | 0.3 | 0.5 | 0.8 | 0.5 | Moving fast with safety nets |
| **enterprise** | 1.0 | 0.9 | 0.9 | 1.0 | 1.0 | Production enterprise code |
| **fintech** | 1.0 | 0.8 | 1.0 | 1.0 | 1.0 | Financial-grade correctness |
| **library** | 0.8 | 0.9 | 0.9 | 0.8 | 0.8 | Public APIs and packages |
| **data-science** | 0.7 | 0.3 | 0.5 | 0.6 | 0.5 | Exploration with data safety |
| **prototype** | 0.5 | 0.2 | 0.2 | 0.5 | 0.3 | Rapid prototyping |

Scale: **0.0–0.3** permissive | **0.3–0.7** moderate | **0.7–1.0** strict

```bash
mirdan init --quality-profile enterprise
mirdan profile apply fintech          # Change later
mirdan profile suggest                # Let mirdan recommend one
```

---

## IDE Integration

### Claude Code

```bash
mirdan init --claude-code
```

Generates `.mcp.json`, hooks, rules, 7 skills (`/code`, `/debug`, `/review`, `/plan`, `/quality`, `/scan`, `/gate`), and 5 agents (quality-gate, security-audit, test-quality, convention-check, architecture-reviewer).

Hook stringency levels control how aggressively mirdan intervenes:

| Level | Hooks | Best For |
|-------|-------|----------|
| MINIMAL | 2 (PostToolUse, Stop) | Low-friction onboarding |
| STANDARD | 5 (+ UserPromptSubmit, PreToolUse, SubagentStart) | Daily development |
| COMPREHENSIVE | 15 (full lifecycle including compaction, worktrees) | Teams and production |

### Cursor

```bash
mirdan init --cursor
```

Generates a complete Cursor 2.x integration:

- **Rules** — `.cursor/rules/*.mdc` (always-on, security, planning, debug, agent, language-specific)
- **Hooks** — `.cursor/hooks.json` with prompt-type + command-type hooks, `.cursor/hooks/*.sh` scripts
- **Subagents** — `.cursor/agents/*.md` (quality-validator, security-scanner, test-auditor, slop-detector, architecture-reviewer)
- **Skills** — `.cursor/skills/*/SKILL.md` following the [Agent Skills Standard](https://agentskills.io) (code, debug, review, plan, quality, scan, gate)
- **Commands** — `.cursor/commands/*.md` slash commands (`/code`, `/debug`, `/review`, `/plan`, `/quality`, `/scan`, `/gate`)
- **Environment** — `.cursor/environment.json` for Cloud Agent environments
- **Config** — `.cursor/mcp.json`, `AGENTS.md`, `BUGBOT.md`

Cursor has tool slot limits. Set `MIRDAN_TOOL_BUDGET` to control which tools are exposed (2 = validation only, 5+ = all tools).

### Claude Desktop / Any MCP Client

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "mirdan": {
      "command": "uvx",
      "args": ["mirdan"]
    }
  }
}
```

**Corporate networks** (Netskope, Zscaler, Artifactory proxy): pass env vars for SSL and model downloads:

```json
{
  "mcpServers": {
    "mirdan": {
      "command": "uvx",
      "args": ["mirdan"],
      "env": {
        "MIRDAN_HF_ENDPOINT": "https://artifactory.corp.com/hf",
        "MIRDAN_HF_TOKEN": "your-artifactory-token",
        "MIRDAN_SSL_CERT_FILE": "/path/to/corporate-ca-bundle.crt"
      }
    }
  }
}
```

### Enterprise Deployment

For organization-wide enforcement via managed configuration:

**macOS:** `/Library/Application Support/ClaudeCode/managed-mcp.json`
**Linux:** `/etc/claude-code/managed-mcp.json`

```json
{
  "mcpServers": {
    "mirdan": {
      "command": "uvx",
      "args": ["mirdan"]
    }
  }
}
```

---

## CI/CD Integration

### GitHub Actions

Add this workflow to `.github/workflows/mirdan.yml`:

```yaml
name: Mirdan Quality Gate
on: [pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv tool install mirdan
      - run: mirdan gate
```

### SARIF Export for GitHub Code Scanning

```yaml
- run: mirdan export --format sarif > results.sarif
- uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: results.sarif
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: mirdan
        name: mirdan quality gate
        entry: mirdan validate --staged --quick
        language: system
        types: [python]
```

### Quality Badges

```bash
mirdan export --format badge > .mirdan/badge.json
```

---

## Configuration

`mirdan init` generates `.mirdan/config.yaml`. Key sections:

```yaml
version: "1.0"

project:
  name: "MyApp"
  primary_language: "python"
  frameworks: ["fastapi", "react"]

# Quality enforcement levels
quality:
  security: "strict"           # strict|moderate|permissive
  architecture: "moderate"
  documentation: "moderate"
  testing: "strict"

# Or use a named profile (overrides quality section)
quality_profile: "default"

# Semantic validation and dependency scanning
semantic:
  enabled: true
  analysis_protocol: "security"  # none|security|comprehensive

dependencies:
  enabled: true
  osv_cache_ttl: 86400           # 24 hours
  scan_on_gate: true
  fail_on_severity: "high"       # critical|high|medium|low|none

# Score thresholds
thresholds:
  severity_error_weight: 0.25
  severity_warning_weight: 0.08
  arch_max_function_length: 30
  arch_max_file_length: 300
  # Per-file threshold overrides (glob patterns)
  file_overrides:
    - pattern: "tests/**"
      arch_max_function_length: 60
    - pattern: "scripts/**"
      arch_max_file_length: 500

# Hook behavior
hooks:
  enabled_events: ["PreToolUse", "PostToolUse", "Stop"]
  quick_validate_timeout: 5000
  auto_fix_suggestions: true
```

LLM configuration goes in `.mirdan.yaml` (written by `mirdan llm setup`):

```yaml
llm:
  enabled: true
  backend: llamacpp                # llamacpp (default) or ollama
  model_keep_alive: 5m             # Unload model after idle (saves RAM)
```

See the full [LLM configuration reference](https://github.com/S-Corkum/mirdan/blob/main/docs/llm-configuration.md) or run `mirdan llm setup` to auto-configure.

---

## Advanced Features

### AST-Based Validation

Python rules PY001–PY004 are verified via the `ast` module, eliminating false positives from eval/exec in strings, bare except in comments, etc. Two additional AST rules:

- **PY014** (dead-import) — detects unused imports, respecting `TYPE_CHECKING` blocks, `__all__`, and aliased imports
- **PY015** (unreachable-code) — detects code after `return`/`raise`/`break`/`continue`, skipping `finally` blocks

For TypeScript/JavaScript, install the optional `ast` extra to enable tree-sitter parsing:

```bash
uv tool install 'mirdan[ast]'
```

This gives accurate function length, nesting depth, and missing return type detection instead of regex approximation. Falls back gracefully to regex when tree-sitter is not installed.

### Full-File Diff Validation

When validating diffs (via hooks or `validate_code_quality` with `input_type="diff"`), mirdan reads the full file from disk when available. This enables architecture checks (function length, nesting depth) that require full-file context. Violations are filtered to changed lines only, and file-scope rules (file-too-long) are excluded from diff results.

### Adaptive File-Path Thresholds

Override thresholds for specific file patterns using `file_overrides` in your config:

```yaml
thresholds:
  arch_max_function_length: 30
  file_overrides:
    - pattern: "tests/**"
      arch_max_function_length: 60    # Tests can be longer
    - pattern: "migrations/**"
      arch_max_file_length: 1000      # Migration files are naturally long
```

Patterns use glob syntax and are matched against file paths. Overrides only replace the fields they specify — all other thresholds inherit from the base config.

### Convention Discovery

Scan your codebase to discover implicit patterns and generate custom rules:

```bash
mirdan init --learn              # During init
mirdan scan --directory src/     # Standalone
```

Discovers naming patterns, import styles, docstring conventions, and recurring patterns. Generates `.mirdan/rules/conventions.yaml` with project-specific rules.

### Dependency Vulnerability Scanning

Check dependencies against the [OSV database](https://osv.dev) (free, no API key required):

```bash
mirdan scan --dependencies                # Standalone scan
mirdan gate --include-dependencies        # Quality gate + vuln check
```

Supports PyPI, npm, crates.io, Go, and Maven. Results are cached for 24 hours. Vulnerabilities in imported packages trigger SEC014 violations during code validation.

### Semantic Validation

`validate_code_quality` returns `semantic_checks` — targeted review questions generated from code patterns (SQL queries, auth logic, crypto operations, file I/O). These guide the AI to investigate specific concerns rather than doing shallow pattern matching. For security-critical code, an `analysis_protocol` provides structured deep-analysis steps.

### Quality Forecasting

`get_quality_trends` analyzes validation history to track scores over time, forecast trajectory, detect regressions between sessions, and calculate pass rates.

### Session Tracking and Feedback Loop

Each `enhance_prompt` call returns a `session_id`. Pass it to `validate_code_quality` to track quality across the full task lifecycle. Pass it back to `enhance_prompt` on the next call to close the feedback loop:

```
enhance_prompt(task)                   → session_id, enhanced_prompt
  ↓ implement code
validate_code_quality(code, session_id) → violations, session_context
  ↓ fix issues, iterate
enhance_prompt(task, session_id=...)   → persistent violations injected
                                          as priority quality requirements
```

When a violation recurs across two or more consecutive validations, mirdan surfaces it as a priority `quality_requirement` in the next enhanced prompt — ensuring the AI addresses the root cause rather than adding new code on top of broken foundations.

### Multi-Agent Coordination

Hook configurations provide guardrails for autonomous agents (Cursor Background Agents, Claude Code subagents), ensuring quality enforcement without human oversight.

### Cross-Project Intelligence

When combined with [enyal](https://github.com/S-Corkum/enyal) (persistent knowledge graph MCP), mirdan stores project conventions as knowledge entries and recalls patterns across projects.

### Upgrading

```bash
uv tool upgrade mirdan     # Upgrade to latest version
mirdan init --upgrade      # Regenerate IDE integration files
```

`uv tool upgrade` updates the package. `mirdan init --upgrade` merges new configuration fields into existing `.mirdan/config.yaml`, regenerates integration files, and preserves your customizations.

---

## MCP Tools Reference

### enhance_prompt

Entry point for coding tasks. Enriches a prompt with quality requirements, security constraints, and tool recommendations.

```
Parameters:
  prompt (required)     — The coding task description
  task_type             — generation|refactor|debug|review|test|planning|auto
  context_level         — minimal|auto|comprehensive
  max_tokens            — Token budget (0=unlimited)
  model_tier            — auto|opus|sonnet|haiku
  session_id            — Resume an existing session to thread validation
                          feedback into this prompt. Persistent violations
                          from prior validate_code_quality calls are injected
                          as priority quality requirements.

Returns:
  enhanced_prompt       — Enriched prompt with quality guidance
  detected_language     — Primary language detected
  detected_frameworks   — Frameworks to query docs for
  task_type             — Primary detected task type
  task_types            — All detected task types (compound detection). A
                          prompt like "add tests for the new feature" returns
                          ["test", "generation"] and unions verification steps
                          from both types.
  touches_security      — Whether task involves security-sensitive code
  quality_requirements  — Constraints to follow during implementation
  verification_steps    — Checklist before marking complete. Compressed to a
                          single re-validation step when a prior session passed,
                          reducing context waste on iterative work.
  tool_recommendations  — Which MCPs to call for context. Session-aware:
                          targets enyal recall to failure patterns on re-calls
                          with errors; suppresses redundant recalls after a pass.
```

### validate_code_quality

Exit gate — validates code against quality standards. Returns score, violations, and semantic review questions.

```
Parameters:
  code (required)       — Code to validate
  language              — python|typescript|javascript|rust|go|java|auto
  check_security        — Enable security rules (default: true)
  check_architecture    — Enable architecture rules (default: true)
  check_style           — Enable style rules (default: true)
  severity_threshold    — error|warning|info
  input_type            — code|diff|compare
  session_id            — Session ID from enhance_prompt

Returns:
  passed                — Whether validation passed
  score                 — Quality score (0.0–1.0)
  violations            — List of rule violations with details. Each violation
                          includes verifiable: false when the check is
                          pattern-based (AI001–AI008) rather than AST-verified,
                          so the AI knows to confirm semantically before fixing.
  semantic_checks       — Targeted review questions from code patterns
  summary               — Human-readable summary
```

### validate_quick

Fast security-only validation (<500ms) for hook integration. Runs SEC001–SEC014, AI001, and AI008.

### get_quality_standards

Look up quality standards for a language/framework combination.

### get_quality_trends

Quality score trends and forecasting from validation history.

### scan_dependencies

Scan project dependencies for known vulnerabilities via the OSV database.

### scan_conventions

Discover implicit codebase conventions and generate custom rules.

---

## CLI Reference

| Command | Purpose |
|---------|---------|
| `mirdan serve` | Start the MCP server (default) |
| `mirdan init` | Initialize project — generates config, hooks, rules, IDE integrations |
| `mirdan validate` | Validate code quality (`--file`, `--staged`, `--stdin`, `--diff`, `--quick`) |
| `mirdan gate` | Quality gate for CI/CD (`--include-dependencies` for vuln check) |
| `mirdan fix` | Auto-fix violations (`--dry-run`, `--auto`, `--staged`) |
| `mirdan check` | Run lint + typecheck + test (`--smart` for LLM analysis, `--fix` for auto-fix) |
| `mirdan scan` | Discover conventions (`--directory`) or scan deps (`--dependencies`) |
| `mirdan profile` | Manage quality profiles (`list`, `suggest`, `apply`) |
| `mirdan export` | Export results (`--format sarif\|badge\|json`) |
| `mirdan report` | Quality reports (`--session`, `--compact-state`, `--format`) |
| `mirdan standards` | View quality standards for a language |
| `mirdan checklist` | View verification checklists for a task type |
| `mirdan plugin` | Plugin export for standalone distribution |
| `mirdan llm setup` | Configure local LLM — installs backend, downloads model |
| `mirdan llm status` | Show LLM health, loaded model, hardware profile (`--json`) |
| `mirdan llm warmup` | Pre-load model into memory for faster first inference |
| `mirdan llm metrics` | Token savings dashboard (`--days N`, `--json`) |
| `mirdan triage` | Classify a task via local LLM (`--stdin`, used by hooks) |
| `mirdan fine-tune` | Training data management (`status`, `export`) |

---

## Troubleshooting

### Server Not Connecting

1. **Check uvx is available:** `uvx --version`
2. **Test server manually:** `uvx mirdan` (should start without errors)
3. **Check status in Claude Code:** `/mcp`

### Debug Logging

```json
{
  "mcpServers": {
    "mirdan": {
      "command": "uvx",
      "args": ["mirdan"],
      "env": { "FASTMCP_DEBUG": "true" }
    }
  }
}
```

### Common Issues

| Issue | Solution |
|-------|----------|
| `command not found: uvx` | Install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `command not found: mirdan` | `uv tool install mirdan` |
| Server starts but no tools appear | Restart your IDE after config changes |
| Python version error | Ensure Python 3.11+ is installed |
| Hook not firing | Check hook stringency level — MINIMAL only fires on 2 events |
| Tool budget limiting tools | Set `MIRDAN_TOOL_BUDGET=5` or remove the env var |
| "LLM backend unavailable" | Run `mirdan llm setup` — it auto-installs the backend and model |
| "Model not found" | Run `mirdan llm setup` to download the recommended model |
| Slow LLM inference (<5 tok/s) | Verify Metal: `CMAKE_ARGS="-DGGML_METAL=ON" pip install --force-reinstall llama-cpp-python` |
| High memory usage | Set `model_keep_alive: 2m` in `.mirdan.yaml`. Model unloads after idle. |
| Hooks not calling local LLM | Run `mirdan init --claude-code` (or `--cursor`) to regenerate hooks |
| Corporate network download fails | Set `MIRDAN_HF_ENDPOINT` and `MIRDAN_HF_TOKEN` (see [troubleshooting guide](https://github.com/S-Corkum/mirdan/blob/main/docs/llm-troubleshooting.md)) |

---

## Development

```bash
git clone https://github.com/S-Corkum/mirdan.git
cd mirdan
uv sync --all-extras         # Includes tree-sitter for TS/JS AST
uv run pytest                # 3050+ tests
uv run mirdan                # Run server locally
```

---

## License

MIT
