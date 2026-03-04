# Mirdan

AI Code Quality Orchestrator — prevent AI slop before it reaches your codebase.

[![PyPI version](https://img.shields.io/pypi/v/mirdan.svg)](https://pypi.org/project/mirdan/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Problem

AI coding assistants produce "slop" not because the models are incapable, but because developers provide prompts that lack context, structure, and quality constraints. Research shows properly structured prompts achieve 15-74% better results.

The result: placeholder functions, hallucinated imports, hardcoded secrets, over-engineered abstractions, and code that looks right but fails in production.

## The Solution

Mirdan is an MCP server that intercepts your workflow at two critical points:

1. **Before you code** — `enhance_prompt` enriches your task with quality requirements, security constraints, and framework-specific standards
2. **After you code** — `validate_code_quality` catches AI slop patterns, security vulnerabilities, and style violations before they land

Combined with automatic IDE hooks, mirdan enforces quality invisibly — you just code normally.

## Features

### 6 MCP Tools

| Tool | Purpose |
|------|---------|
| `enhance_prompt` | Entry point — enriches prompts with quality requirements and tool recommendations |
| `validate_code_quality` | Exit gate — validates code against security, architecture, AI quality, and style rules |
| `validate_quick` | Fast validation (<500ms) for hook integration — security-critical rules only |
| `get_quality_standards` | Retrieve language/framework-specific quality standards |
| `get_quality_trends` | Quality score trends, forecasting, and regression detection |
| `scan_conventions` | Discover implicit codebase conventions and generate custom rules |

### 12 CLI Commands

| Command | Purpose |
|---------|---------|
| `mirdan serve` | Start the MCP server (default when running bare `mirdan`) |
| `mirdan init` | Initialize project — generates config, hooks, rules, IDE integrations |
| `mirdan validate` | Validate code quality from the command line |
| `mirdan gate` | Quality gate for CI/CD — exit code 0 (pass) or 1 (fail) |
| `mirdan fix` | Auto-fix violations with `--dry-run`, `--auto`, `--staged` |
| `mirdan scan` | Scan codebase to discover conventions |
| `mirdan profile` | Manage quality profiles — `list`, `suggest`, `apply` |
| `mirdan export` | Export results — `--format sarif\|badge\|json` |
| `mirdan report` | Generate quality reports — `--session`, `--compact-state`, `--format` |
| `mirdan standards` | View quality standards for a language |
| `mirdan checklist` | View verification checklists for a task type |
| `mirdan plugin` | Plugin management — `export` for standalone distribution |

### Validation Engine

- **8 AI quality rules** (AI001–AI008) — placeholder detection, hallucinated imports, over-engineering, duplicate code, injection vulnerabilities
- **13 security rules** (SEC001–SEC013) — hardcoded secrets, SQL injection, command injection, SSL bypass, JWT bypass, graph DB injection
- **Language rules** — Python (PY001–PY013), JavaScript (JS001–JS005), TypeScript (TS001–TS005), Go (GO001–GO003), Java (JV001–JV007), Rust (RS001–RS002)
- **RAG/ML rules** (RAG001–RAG002) — chunk overlap, deprecated loaders
- **31 auto-fixable rules** with template and pattern-based fixes
- **61 total validation rules**

### 6 Languages, 33 Framework Standards

**Languages:** Python, TypeScript, JavaScript, Go, Java, Rust

**Frameworks:** React, React Native, Next.js, Nuxt, Vue, SvelteKit, Astro, Flutter, Tailwind, FastAPI, Django, Express, NestJS, Echo, Gin, Spring Boot, Micronaut, Quarkus, Drizzle, Neo4j, Supabase, Convex, Pinecone, Qdrant, Milvus, Weaviate, ChromaDB, FAISS, LangChain, LangGraph, CrewAI, DSPy, tRPC

### 7 Quality Profiles

| Profile | Focus |
|---------|-------|
| `default` | Balanced quality for general-purpose projects |
| `startup` | Move fast with essential safety nets |
| `enterprise` | Maximum enforcement for enterprise codebases |
| `fintech` | Financial-grade security and correctness |
| `library` | High-quality public API with documentation focus |
| `data-science` | Flexible for exploration, strict on data handling |
| `prototype` | Minimal enforcement for rapid prototyping |

### IDE Integration

- **Claude Code** — hooks (up to 15 events), 5 skills, 5 agents, quality rules
- **Cursor** — hooks (up to 16 events), rules (.mdc), AGENTS.md, BUGBOT.md
- **Claude Desktop** — MCP server configuration
- **Any MCP client** — standard MCP stdio transport

### Quality Intelligence

- Violation explanations with fix suggestions
- Quality forecasting and regression detection
- Convention discovery and custom rule generation
- Session tracking with historical trend analysis
- Cross-project intelligence (with enyal MCP)

### CI/CD

- SARIF 2.1.0 export for GitHub Code Scanning
- GitHub Actions workflow generation
- Pre-commit hook configuration
- Quality badge generation

---

## Installation

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### From PyPI (Recommended)

```bash
# Using uv (recommended)
uv pip install mirdan --system

# Or using pip
pip install mirdan

# Verify installation
mirdan --help
```

### From Source (Development)

```bash
git clone https://github.com/S-Corkum/mirdan.git
cd mirdan
uv sync
uv run mirdan
```

---

## Quick Start

```bash
# Install
pip install mirdan

# Initialize for your IDE
mirdan init --claude-code    # Claude Code
mirdan init --cursor         # Cursor
mirdan init --all            # Both

# That's it. Quality gates are now automatic.
```

After `mirdan init`, your IDE automatically:
- Enhances prompts with quality requirements before coding
- Validates code against security and quality rules after every edit
- Runs a final quality gate before task completion

### Manual Usage

```bash
# Validate a file
mirdan validate --file src/auth.py

# Validate staged changes
mirdan validate --staged

# Auto-fix violations
mirdan fix --file src/auth.py

# Quality gate (CI/CD)
mirdan gate

# Discover conventions
mirdan scan --directory src/

# Export SARIF for GitHub Code Scanning
mirdan export --format sarif > results.sarif
```

---

## MCP Tools Reference

### enhance_prompt

Entry point for all coding tasks. Enriches a prompt with quality requirements, security constraints, and tool recommendations.

```
Parameters:
  prompt (required)     — The coding task description
  task_type             — generation|refactor|debug|review|test|planning|auto
  context_level         — minimal|auto|comprehensive
  max_tokens            — Token budget (0=unlimited)
  model_tier            — auto|opus|sonnet|haiku

Returns:
  detected_language     — Primary language detected
  detected_frameworks   — Frameworks to query docs for
  touches_security      — Whether task involves security-sensitive code
  quality_requirements  — Constraints to follow during implementation
  tool_recommendations  — Which MCPs to call for context
  verification_steps    — Checklist before marking complete
```

### validate_code_quality

Exit gate — validates code against quality standards. Supports standard validation, diff mode, and multi-implementation comparison.

```
Parameters:
  code (required)       — Code to validate
  language              — python|typescript|javascript|rust|go|java|auto
  check_security        — Validate against security standards (default: true)
  check_architecture    — Validate against architecture standards (default: true)
  check_style           — Validate against style standards (default: true)
  severity_threshold    — error|warning|info
  input_type            — code|diff|compare
  session_id            — Session ID from enhance_prompt
  max_tokens            — Token budget
  model_tier            — Target model tier

Returns:
  passed                — Whether validation passed
  score                 — Quality score (0.0–1.0)
  violations            — List of rule violations with details
  summary               — Human-readable summary
```

### validate_quick

Fast security-only validation (<500ms target) for hook integration. Runs SEC001–SEC013, AI001, and AI008 only.

```
Parameters:
  code (required)       — Code to validate
  language              — Language override (auto-detected by default)
  max_tokens            — Token budget
  model_tier            — Target model tier
```

### get_quality_standards

Look up quality standards for a language/framework combination.

```
Parameters:
  language (required)   — Programming language
  framework             — Framework name (optional)
  category              — security|architecture|style|all
```

### get_quality_trends

Quality score trends from stored validation history. Reads from `.mirdan/history/`.

```
Parameters:
  project_path          — Project directory
  days                  — Number of days to analyze
  format                — Output format
```

### scan_conventions

Scan a codebase to discover implicit conventions and patterns. Aggregates results into convention entries.

```
Parameters:
  directory (required)  — Directory to scan
  language              — Language filter
```

---

## CLI Reference

### `mirdan init`

Initialize mirdan for a project. Detects language, frameworks, and IDE, then generates configuration and integration files.

```bash
mirdan init [directory] [flags]
```

| Flag | Description |
|------|-------------|
| `--claude-code` | Generate Claude Code integration (hooks, rules, skills, agents) |
| `--cursor` | Generate Cursor integration (hooks, rules, AGENTS.md, BUGBOT.md) |
| `--all` | Generate both Claude Code and Cursor integrations |
| `--quality-profile NAME` | Set quality profile (e.g., `enterprise`, `startup`) |
| `--learn` | Scan codebase and generate custom rules from discovered conventions |
| `--upgrade` | Upgrade existing config — merges new fields, regenerates integration files |
| `--hooks` | Install hook scripts for auto-validation |

### `mirdan validate`

Validate code quality from the command line.

```bash
mirdan validate --file src/main.py
mirdan validate --staged           # Validate git staged changes
mirdan validate --stdin             # Read from stdin
mirdan validate --diff              # Validate unified diff
mirdan validate --quick             # Security-only fast mode
mirdan validate --lint              # Include linter results
mirdan validate --format json       # Output format
```

### `mirdan gate`

Quality gate for CI/CD pipelines. Validates all changed files and exits with code 0 (pass) or 1 (fail).

```bash
mirdan gate                        # Check all changed files
```

### `mirdan fix`

Auto-fix violations using template and pattern-based fixes.

```bash
mirdan fix --file src/main.py      # Fix a specific file
mirdan fix --staged                # Fix staged changes
mirdan fix --auto                  # Apply fixes automatically
mirdan fix --dry-run               # Show what would be fixed
```

### `mirdan profile`

Manage quality profiles.

```bash
mirdan profile list                # List all available profiles
mirdan profile suggest [dir]       # Suggest a profile based on codebase analysis
mirdan profile apply enterprise    # Apply a named profile
```

### `mirdan export`

Export validation results in various formats.

```bash
mirdan export --format sarif       # SARIF 2.1.0 for GitHub Code Scanning
mirdan export --format badge       # Quality badge
mirdan export --format json        # JSON results
```

### `mirdan report`

Generate quality reports with session history.

```bash
mirdan report                      # Full quality report
mirdan report --session ID         # Report for a specific session
mirdan report --compact-state      # Compact state summary
mirdan report --format json        # JSON output
```

### `mirdan scan`

Scan codebase to discover conventions.

```bash
mirdan scan --directory src/
```

### `mirdan standards`

View quality standards for a language.

```bash
mirdan standards --language python
```

### `mirdan checklist`

View verification checklists for a task type.

```bash
mirdan checklist --task-type review
```

### `mirdan plugin`

Plugin management for standalone distribution.

```bash
mirdan plugin export --output-dir ./mirdan-plugin
mirdan plugin export --cursor      # Cursor-specific export
mirdan plugin export --all         # All platforms
```

---

## Quality Profiles

Each profile defines enforcement levels across 6 dimensions scored 0.0–1.0:

| Profile | Security | Architecture | Testing | Documentation | AI Slop | Performance |
|---------|----------|-------------|---------|--------------|---------|-------------|
| **default** | 0.7 | 0.5 | 0.7 | 0.5 | 0.7 | 0.5 |
| **startup** | 0.7 | 0.3 | 0.5 | 0.2 | 0.8 | 0.3 |
| **enterprise** | 1.0 | 0.9 | 0.9 | 0.8 | 1.0 | 0.7 |
| **fintech** | 1.0 | 0.8 | 1.0 | 0.7 | 1.0 | 0.8 |
| **library** | 0.8 | 0.9 | 0.9 | 0.9 | 0.8 | 0.7 |
| **data-science** | 0.7 | 0.3 | 0.5 | 0.5 | 0.6 | 0.4 |
| **prototype** | 0.5 | 0.2 | 0.2 | 0.1 | 0.5 | 0.2 |

Scale: **0.0–0.3** permissive | **0.3–0.7** moderate | **0.7–1.0** strict

Select a profile during init or apply later:

```bash
mirdan init --quality-profile enterprise
mirdan profile apply fintech
```

---

## Validation Rules

### AI Quality Rules (AI001–AI008)

| Rule | Name | Severity | Description |
|------|------|----------|-------------|
| AI001 | ai-placeholder-code | error | Detects `raise NotImplementedError`, `pass` with TODO/FIXME (skips `@abstractmethod`) |
| AI002 | hallucinated-imports | warning | Flags imports not in stdlib or project dependencies |
| AI003 | over-engineering-detection | warning | Detects unnecessary abstractions and over-complex implementations |
| AI004 | duplicate-code-block | warning | Detects duplicate or near-duplicate code blocks |
| AI005 | inconsistent-error-handling | warning | Detects inconsistent error handling patterns |
| AI006 | unnecessary-heavy-imports | info | Detects heavy library imports where lighter alternatives exist |
| AI007 | security-theater-detection | warning | Detects false-positive security patterns with no actual protection |
| AI008 | injection-fstring | error | Catches f-string SQL injection, `eval`/`exec`/`os.system` with f-strings |

AI001 and AI008 run in both full and quick validation modes (security-critical).

### Security Rules (SEC001–SEC013)

| Rule | Name | Description |
|------|------|-------------|
| SEC001 | hardcoded-api-key | Hardcoded API keys |
| SEC002 | hardcoded-password | Hardcoded passwords |
| SEC003 | aws-access-key | AWS access keys (AKIA prefix) |
| SEC004 | sql-concat-python | SQL via string concatenation (Python) |
| SEC005 | sql-fstring-python | SQL via f-strings (Python) |
| SEC006 | sql-template-js | SQL via template literals (JavaScript) |
| SEC007 | ssl-verify-disabled | Disabled SSL/TLS verification |
| SEC008 | shell-format-injection | Shell command injection via string formatting |
| SEC009 | shell-fstring-injection | Shell command injection via f-strings |
| SEC010 | jwt-no-verify | JWT verification disabled |
| SEC011 | cypher-fstring-injection | Neo4j Cypher injection via f-strings |
| SEC012 | cypher-concat-injection | Neo4j Cypher injection via string concatenation |
| SEC013 | gremlin-fstring-injection | Gremlin query injection via f-strings |

### Language Rules

| Language | Rules | Key Checks |
|----------|-------|------------|
| Python | PY001–PY013 | `eval`/`exec`, bare except, mutable defaults, deprecated typing, unsafe pickle/yaml, subprocess shell, wildcard imports |
| JavaScript | JS001–JS005 | `var` usage, `eval`, `document.write`, innerHTML, child_process.exec |
| TypeScript | TS001–TS005 | `eval`, Function constructor, `@ts-ignore`, `as any`, innerHTML |
| Go | GO001–GO003 | Ignored errors, `panic()`, SQL via fmt.Sprintf |
| Java | JV001–JV007 | String ==, generic Exception, System.exit, empty catch, Runtime.exec, unsafe deserialization, XMLDecoder |
| Rust | RS001–RS002 | `.unwrap()`, empty `.expect()` |

### RAG/ML Rules

| Rule | Name | Description |
|------|------|-------------|
| RAG001 | chunk-overlap-zero | Zero overlap in RAG chunk splitting |
| RAG002 | deprecated-langchain-loader | Deprecated LangChain loaders |

### Auto-Fixable Rules

31 rules support automatic fixes via `mirdan fix`: AI001, AI003, AI006, AI007, AI008, SEC001–SEC008, PY003–PY006, PY008, PY009, PY011, PY012, JS001–JS003, TS001, TS004, RS001, RS002, GO001, GO002, JAVA001.

---

## IDE Integration

### Claude Code

```bash
mirdan init --claude-code
```

Generates:

| File | Purpose |
|------|---------|
| `.mcp.json` | Registers mirdan as MCP server |
| `.claude/hooks.json` | Automatic quality hooks |
| `.claude/rules/mirdan-*.md` | Path-specific quality rules |
| `.claude/skills/mirdan/*.md` | 5 slash command skills |
| `.claude/agents/*.md` | 5 specialized agents |

**5 Skills:** `/mirdan:code`, `/mirdan:debug`, `/mirdan:review`, `/mirdan:plan`, `/mirdan:quality`

**5 Agents:** `quality-gate`, `security-audit`, `test-quality`, `convention-check`, `architecture-reviewer`

**Hook stringency levels:**

| Level | Events | Best For |
|-------|--------|----------|
| MINIMAL | 2 (PostToolUse, Stop) | Low-friction onboarding |
| STANDARD | 5 (+ UserPromptSubmit, PreToolUse, SubagentStart) | Daily development |
| COMPREHENSIVE | 15 (full lifecycle) | Teams, production projects |

**Project settings (`.claude/settings.json`):**

```json
{
  "permissions": {
    "allow": [
      "mcp__mirdan__enhance_prompt",
      "mcp__mirdan__validate_code_quality",
      "mcp__mirdan__validate_quick",
      "mcp__mirdan__get_quality_standards",
      "mcp__mirdan__get_quality_trends",
      "mcp__mirdan__scan_conventions"
    ]
  }
}
```

### Cursor

```bash
mirdan init --cursor
```

Generates:

| File | Purpose |
|------|---------|
| `.cursor/mcp.json` | MCP server registration with tool budget |
| `.cursor/hooks.json` | Prompt hooks for automatic enforcement |
| `.cursor/rules/*.mdc` | Language-specific quality rules |
| `AGENTS.md` | Quality checkpoints + inline AI/security rules |
| `BUGBOT.md` | Structured PR review rules with regex patterns |

**Hook stringency levels:**

| Level | Events | Best For |
|-------|--------|----------|
| MINIMAL | 2 (afterFileEdit, stop) | Low-friction onboarding |
| STANDARD | 5 (+ preToolUse, postToolUse, sessionStart) | Daily development |
| COMPREHENSIVE | 16 (all events) | Teams, production projects |

**Tool budget** — Cursor has tool slot limits. The `MIRDAN_TOOL_BUDGET` env var controls which tools are exposed:

| Budget | Tools Exposed |
|--------|--------------|
| 2 | validate_code_quality, validate_quick |
| 3 | + enhance_prompt |
| 4 | + get_quality_standards |
| 5+ | All tools (default) |

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

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

### Any MCP Client

Mirdan uses standard MCP stdio transport:

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

For development installs:

```json
{
  "mcpServers": {
    "mirdan": {
      "command": "uv",
      "args": ["--directory", "/path/to/mirdan", "run", "mirdan"]
    }
  }
}
```

### Enterprise Deployment

For organization-wide enforcement via managed configuration:

**File (macOS):** `/Library/Application Support/ClaudeCode/managed-mcp.json`
**File (Linux):** `/etc/claude-code/managed-mcp.json`

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

## Configuration

Create `.mirdan/config.yaml` in your project (auto-generated by `mirdan init`):

```yaml
version: "1.0"

# Project metadata (auto-detected)
project:
  name: "MyApp"
  type: "application"              # application|library|tool|data
  primary_language: "python"
  frameworks: ["fastapi", "react"]

# Quality enforcement levels
quality:
  security: "strict"               # strict|moderate|permissive
  architecture: "moderate"
  documentation: "moderate"
  testing: "strict"
  framework: "moderate"
  language: "moderate"
  custom_rules_dir: ".mirdan/rules"

# Named quality profile (overrides quality section)
quality_profile: "default"

# IDE platform profile
platform:
  name: "claude-code"              # generic|cursor|claude-code
  context_level: "auto"            # auto|none
  tool_budget_aware: false

# MCP orchestration
orchestration:
  prefer_mcps: ["context7", "filesystem"]
  auto_invoke: []
  gather_timeout: 10.0
  gatherer_timeout: 3.0
  auto_memory: false
  auto_memory_threshold: 0.8

# Enhancement behavior
enhancement:
  mode: "auto"                     # auto|confirm|manual
  verbosity: "balanced"            # minimal|balanced|comprehensive
  include_verification: true
  include_tool_hints: true

# Plan validation (for cheap model handoff)
planning:
  target_model: "haiku"            # haiku|flash|cheap|capable
  min_grounding_score: 0.9
  min_completeness_score: 0.9
  min_clarity_score: 0.95
  require_research_notes: true
  require_step_grounding: true
  require_verification_per_step: true
  reject_vague_language: true
  max_words_per_step_detail: 100

# Token budgets
tokens:
  default_max_tokens: 0            # 0=unlimited
  compact_threshold: 4000
  minimal_threshold: 1000
  micro_threshold: 200

# Centralized thresholds
thresholds:
  severity_error_weight: 0.25
  severity_warning_weight: 0.08
  severity_info_weight: 0.02
  arch_max_function_length: 30
  arch_max_file_length: 300
  arch_max_nesting_depth: 4
  arch_max_class_methods: 10

# External linter orchestration
linters:
  enabled_linters: []              # Empty = auto-detect
  auto_detect: true
  timeout: 30.0
  ruff_args: []
  eslint_args: []
  mypy_args: []

# Hook generation settings
hooks:
  enabled_events: ["PreToolUse", "PostToolUse", "Stop"]
  quick_validate_timeout: 5000
  auto_fix_suggestions: true
  compaction_resilience: false
  multi_agent_awareness: false
  session_hooks: false
  notification_hooks: false

# Session management
session:
  ttl_seconds: 1800               # 30 minutes
  max_sessions: 100
```

---

## CI/CD Integration

### GitHub Actions

`mirdan init` generates `.github/workflows/mirdan.yml`. You can also create it manually:

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

### Quality Badges

```bash
mirdan export --format badge > .mirdan/badge.json
```

### Pre-commit Hooks

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

---

## Advanced Features

### Convention Discovery (`--learn`)

Scan your codebase to discover implicit conventions and generate custom rules:

```bash
mirdan init --learn
# or standalone:
mirdan scan --directory src/
```

Discovers naming patterns, import styles, docstring conventions, and recurring patterns. Generates `.mirdan/rules/conventions.yaml` with custom rules derived from your actual code.

### Quality Forecasting and Regression Detection

`get_quality_trends` analyzes validation history from `.mirdan/history/` to:
- Track quality scores over time
- Forecast quality trajectory
- Detect regressions between sessions
- Calculate pass rates and trend direction

### Session Tracking

Each `enhance_prompt` call creates a session. Pass the `session_id` to `validate_code_quality` to track quality across the full task lifecycle.

### Multi-Agent Coordination

AGENTS.md and hook configurations provide guardrails for autonomous agents (Cursor Background Agents, Claude Code subagents), ensuring quality enforcement without human oversight.

### Cross-Project Intelligence

When combined with [enyal](https://github.com/S-Corkum/enyal) (persistent knowledge graph MCP), mirdan can:
- Store project conventions as knowledge entries
- Recall patterns across projects
- Build a cross-project quality knowledge base

### Violation Explanations

Validation results include detailed explanations for each violation: what the rule checks, why it matters, and how to fix it.

### Upgrade Existing Projects

```bash
mirdan init --upgrade
```

Merges new configuration fields into existing `.mirdan/config.yaml`, regenerates integration files, and preserves your customizations.

---

## Troubleshooting

### Server Not Connecting

1. **Check uvx is available:**
   ```bash
   uvx --version
   ```

2. **Test server manually:**
   ```bash
   uvx mirdan
   # Should start without errors, waiting for MCP protocol
   ```

3. **Check server status in Claude Code:**
   ```
   /mcp
   ```

### Debug Logging

Enable verbose output:

```json
{
  "mcpServers": {
    "mirdan": {
      "command": "uvx",
      "args": ["mirdan"],
      "env": {
        "FASTMCP_DEBUG": "true"
      }
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
| Hook not firing | Check hook stringency level — MINIMAL only fires 2 events |
| Tool budget limiting tools | Set `MIRDAN_TOOL_BUDGET=5` or remove the env var |

---

## Development

```bash
# Clone and install
git clone https://github.com/S-Corkum/mirdan.git
cd mirdan
uv sync --all-extras

# Run tests
uv run pytest

# Run the server locally
uv run mirdan
```

---

## License

MIT
