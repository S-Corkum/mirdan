# Mirdan

AI Code Quality Orchestrator — prevent AI slop before it reaches your codebase.

[![PyPI version](https://img.shields.io/pypi/v/mirdan.svg)](https://pypi.org/project/mirdan/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

```bash
pip install mirdan
mirdan init --claude-code   # or --cursor, or --all
# Done. Quality enforcement is now automatic.
```

---

## Why Mirdan?

AI coding assistants generate code fast, but without guardrails they produce **slop**: hardcoded secrets, SQL injection, placeholder functions, hallucinated imports, bare except blocks, and code that looks right but fails in production.

Mirdan fixes this by intercepting your AI workflow at two points:

1. **Before coding** — `enhance_prompt` enriches your task with quality requirements, security constraints, and framework-specific standards so the AI produces better code from the start
2. **After coding** — `validate_code_quality` catches what slipped through: 62 rules covering security vulnerabilities, AI-specific antipatterns, and language best practices

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
pip install mirdan       # or: uv pip install mirdan --system
```

### Set Up Your IDE

```bash
mirdan init --claude-code    # Claude Code: hooks, rules, skills, agents
mirdan init --cursor         # Cursor: hooks, rules, AGENTS.md, BUGBOT.md
mirdan init --all            # Both IDEs
```

This generates everything — MCP server config, quality hooks, rules files, and agent definitions. After init, your IDE automatically:

1. Enriches prompts with quality requirements before coding tasks
2. Validates code against security and quality rules after every edit
3. Runs a final quality gate before task completion

### Use From the Command Line

```bash
mirdan validate --file src/auth.py     # Validate a file
mirdan validate --staged               # Validate git staged changes
mirdan fix --file src/auth.py          # Auto-fix violations
mirdan gate                            # CI/CD quality gate (exit 0 or 1)
mirdan scan --dependencies             # Check deps for known CVEs
mirdan scan --directory src/           # Discover codebase conventions
```

---

## How It Works

Mirdan is an [MCP server](https://modelcontextprotocol.io/) — it connects to AI coding assistants (Claude Code, Cursor, Claude Desktop, or any MCP client) and provides quality enforcement tools.

```
┌──────────────────────────────────────────────────┐
│  Your AI Assistant (Claude Code / Cursor / etc)  │
│                                                  │
│  1. You type a coding task                       │
│  2. Hook calls enhance_prompt ──────────┐        │
│  3. AI generates code with guidance     │        │
│  4. Hook calls validate_code_quality ◄──┘        │
│  5. AI fixes violations automatically            │
│  6. Quality gate passes → task complete          │
└──────────────────────────────────────────────────┘
         │                        ▲
         ▼                        │
┌──────────────────────────────────────────────────┐
│  Mirdan MCP Server                               │
│                                                  │
│  enhance_prompt         → Quality requirements   │
│  validate_code_quality  → 62 rules, scoring      │
│  validate_quick         → Fast security checks   │
│  get_quality_standards  → Language/framework ref  │
│  get_quality_trends     → Historical analysis    │
│  scan_dependencies      → CVE detection (OSV)    │
│  scan_conventions       → Convention discovery   │
└──────────────────────────────────────────────────┘
```

---

## Validation Rules

Mirdan ships with **62 rules** across 10 categories. No external services required — all rules run locally.

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
| Python | PY001–PY013 | eval/exec, bare except, mutable defaults, deprecated typing, unsafe pickle/yaml, subprocess shell |
| JavaScript | JS001–JS005 | `var`, eval, document.write, innerHTML, child_process.exec |
| TypeScript | TS001–TS005 | eval, Function constructor, @ts-ignore, `as any`, innerHTML |
| Go | GO001–GO003 | Ignored errors, panic(), SQL via fmt.Sprintf |
| Java | JV001–JV007 | String ==, generic Exception, System.exit, Runtime.exec, unsafe deserialization |
| Rust | RS001–RS002 | .unwrap(), empty .expect() |

Plus **ARCH001–003** (function length, file length, nesting depth), **RAG001–002** (chunk overlap, deprecated loaders).

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

Generates `.mcp.json`, hooks, rules, 5 skills (`/mirdan:code`, `/mirdan:debug`, `/mirdan:review`, `/mirdan:plan`, `/mirdan:quality`), and 5 agents (quality-gate, security-audit, test-quality, convention-check, architecture-reviewer).

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

Generates `.cursor/mcp.json`, hooks, `.mdc` rules, `AGENTS.md`, and `BUGBOT.md` for structured PR review.

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

`mirdan init` generates a workflow, or create one manually:

```yaml
name: Mirdan Quality Gate
on: [pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv pip install mirdan --system
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

# Hook behavior
hooks:
  enabled_events: ["PreToolUse", "PostToolUse", "Stop"]
  quick_validate_timeout: 5000
  auto_fix_suggestions: true
```

See the full configuration reference in [docs/configuration.md](docs/configuration.md) or run `mirdan init` to generate a commented config file.

---

## Advanced Features

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

### Session Tracking

Each `enhance_prompt` call creates a session. Pass `session_id` to subsequent `validate_code_quality` calls to track quality across the full task lifecycle.

### Multi-Agent Coordination

Hook configurations provide guardrails for autonomous agents (Cursor Background Agents, Claude Code subagents), ensuring quality enforcement without human oversight.

### Cross-Project Intelligence

When combined with [enyal](https://github.com/S-Corkum/enyal) (persistent knowledge graph MCP), mirdan stores project conventions as knowledge entries and recalls patterns across projects.

### Upgrading

```bash
mirdan init --upgrade
```

Merges new configuration fields into existing `.mirdan/config.yaml`, regenerates integration files, and preserves your customizations.

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

Returns:
  enhanced_prompt       — Enriched prompt with quality guidance
  detected_language     — Primary language detected
  detected_frameworks   — Frameworks to query docs for
  touches_security      — Whether task involves security-sensitive code
  quality_requirements  — Constraints to follow during implementation
  verification_steps    — Checklist before marking complete
  tool_recommendations  — Which MCPs to call for context
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
  violations            — List of rule violations with details
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
| `mirdan scan` | Discover conventions (`--directory`) or scan deps (`--dependencies`) |
| `mirdan profile` | Manage quality profiles (`list`, `suggest`, `apply`) |
| `mirdan export` | Export results (`--format sarif\|badge\|json`) |
| `mirdan report` | Quality reports (`--session`, `--compact-state`, `--format`) |
| `mirdan standards` | View quality standards for a language |
| `mirdan checklist` | View verification checklists for a task type |
| `mirdan plugin` | Plugin export for standalone distribution |

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
| Server starts but no tools appear | Restart your IDE after config changes |
| Python version error | Ensure Python 3.11+ is installed |
| Hook not firing | Check hook stringency level — MINIMAL only fires on 2 events |
| Tool budget limiting tools | Set `MIRDAN_TOOL_BUDGET=5` or remove the env var |

---

## Development

```bash
git clone https://github.com/S-Corkum/mirdan.git
cd mirdan
uv sync --all-extras
uv run pytest               # 1932 tests
uv run mirdan               # Run server locally
```

---

## License

MIT
