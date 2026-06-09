# Mirdan

AI Code Quality Orchestrator — **deterministic quality enforcement and Haiku-proof planning** for Claude Code and Cursor. 64 security and quality rules, an AI-slop detector, and a no-LLM plan verifier run locally as zero-token hooks — so cheap models can plan and build without shipping slop.

[![PyPI version](https://img.shields.io/pypi/v/mirdan.svg)](https://pypi.org/project/mirdan/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

```bash
uv tool install mirdan                  # Install mirdan
mirdan init --claude-code               # or --cursor
# Done. Quality enforcement is now automatic.
```

Works with **Claude Code**, **Cursor IDE**, and **Cursor CLI**. Deterministic checks — everything stays local, no external API calls.

---

## Planning Pipeline — Haiku-proof plans

Mirdan's planning workflow is **flat, grounded, and design-first** — no briefs, no business
ceremony. A plan is Research Notes → a Low-Level Design → atomic grounded steps, formatted so a
**cheap model can execute it cold** and verified mechanically before you implement it.

This is the heart of the 2.3.0 (Claude Code) and 2.4.0 (Cursor) realignment. Both editors now ship
native plan modes that **plan with a strong model and build with a cheap one** — Claude Code's
`opusplan` (Opus plans, Sonnet/Haiku builds), Cursor's cross-model Plan Mode (build with Haiku /
Composer). Mirdan supplies what makes that safe: a plan format a cheap executor won't hallucinate
against, plus a deterministic, no-LLM verifier.

```
/plan <slug> <description>     → flat grounded plan (format_version 2) with a Low-Level Design
/plan-verify <plan-path>       → mechanical self-check (local, deterministic, no LLM)
/plan-review --stakes high <p> → escape hatch: model-judgment review on the shared rubric
```

### The Haiku-proof format (`format_version: 2`)

Every `Action: Edit` step carries a literal ```anchor```/```replace``` pair — the exact existing
text and its replacement — so a cheap build model applies it as a find-and-replace instead of
guessing. The anchor must be **unique** in the target file; decisions are **pre-resolved** (no
"TBD" / "decide later"); steps are atomic. A judgment step that can't be anchored is tagged
`[target: capable]` and counted — so a Haiku-targeted plan that isn't fully cold-executable says so.
On Claude Code the format lives in the `/plan` skill; on Cursor it lives in the agent-requested
`mirdan-planning.mdc` rule that native Plan Mode reads.

### What `/plan-verify` catches (deterministic, local, no LLM)

- `phantom_files` — a step's `**File:**` points at a path that doesn't exist
- `dependency_errors` — `**Depends On:**` refs to missing steps, or cycles
- `vague_cross_references` — "as discussed", "see above" — unresolvable by a reader
- `missing_grounding` — a step missing File / Action / Details / Verify / Grounding
- `missing_edit_anchors` — an Edit step without an ```anchor```/```replace``` pair
- `anchor_uniqueness_errors` — an anchor not found, or matching more than one place in the file
- `atomicity_violations` — more than one file/action per step, or compound "and then" steps
- `unresolved_decisions` — "TBD" / either-or / "decide later" left in the plan
- `lld_gaps` (advisory) — an `[EXISTING]` interface without a citation, or a `[NEW]` one created by no step

The **Low-Level Design** section is where engineering substance lives — interfaces & signatures
(each tagged `[NEW]`/`[EXISTING]` with a `file:line` citation), an error taxonomy, and the design
decisions surfaced by `enhance_prompt`. Subsections are applicability-gated — include one only if it
applies, no "fill every heading" filler.

### MCP tool

| Tool | Returns |
|---|---|
| `verify_plan(plan_path)` | `{verified, coverage_score, format_version, phantom_files, dependency_errors, vague_cross_references, missing_grounding, missing_edit_anchors, anchor_uniqueness_errors, atomicity_violations, unresolved_decisions, capable_steps, lld_gaps, summary}` |

---

## Deterministic + content, not a local model (2.3.0)

The local-LLM "intelligence layer" (Gemma via Ollama / llama-cpp) was **removed in 2.3.0**.
mirdan is now **deterministic checks + opinionated content** — fast, dependency-light, and fully
local (no model downloads, no external calls). The *intelligence* comes from your coding
assistant's own models; mirdan supplies what they lack and **rides their native mechanisms**
(Claude Code 2.x and Cursor 2.x plan modes, subagents, command hooks, rules/skills) instead of
re-implementing them: a no-LLM plan verifier, the Haiku-proof plan format, the plan-review rubric,
the AI-slop ruleset, and curated quality standards.

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

# Optional extras:
uv tool install 'mirdan[ast]'            # + tree-sitter for TS/JS AST analysis

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

This generates everything — MCP server config, command-type quality hooks, rules files, skills, and agent definitions. After init, your IDE automatically:

1. Validates each edited file against security + quality rules (PostToolUse hook, zero model tokens)
2. Runs a final quality gate before the turn/task completes (Stop / TaskCompleted hooks)
3. Surfaces the rules, skills (`/plan`, `/code`), and opt-in `enhance_prompt` for non-trivial work

### Use From the Command Line

```bash
mirdan validate --file src/auth.py     # Validate a file
mirdan validate --staged               # Validate git staged changes
mirdan fix --file src/auth.py          # Auto-fix violations (pattern-based)
mirdan check                           # Run lint + typecheck + test
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
│  2. AI generates code                            │
│  3. PostToolUse hook validates the edited file   │
│  4. AI fixes the flagged issues                  │
│  5. Stop hook runs the quality gate → complete   │
└──────────────────────────────────────────────────┘
         │                        ▲
         ▼                        │
┌──────────────────────────────────────────────────┐
│  Mirdan MCP Server (deterministic + content)     │
│                                                  │
│  MCP Tools:                                       │
│  enhance_prompt         → Quality requirements   │
│  validate_code_quality  → 64 deterministic rules │
│  validate_quick         → Fast security checks   │
│  get_quality_standards  → Language/framework ref  │
│  get_quality_trends     → Historical analysis    │
│  scan_dependencies      → CVE detection (OSV)    │
│  scan_conventions       → Convention discovery   │
│  verify_plan            → No-LLM plan verifier   │
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

Generates `.mcp.json`, command-type hooks, rules, 4 skills (`/code`, `/plan`, `/plan-review`, `/plan-verify`), and 3 agents (quality-gate, security-audit, plan-reviewer).

Hooks are **command-type** — deterministic shell checks that run outside the model context (zero model tokens) and can block on failure:

| Hook | Runs |
|------|------|
| PostToolUse | Validate the just-edited file (`mirdan validate --quick --scope security`) |
| Stop | Quality gate over the staged/changed set before the turn completes |
| TaskCompleted | Final validation gate on task completion |

Quality guidance (AI/SEC rules, the planning format) lives in `.claude/rules/` and the skills, not in token-spending prompt hooks. `enhance_prompt` is opt-in — recommended before security-sensitive, multi-file, or new-library work; `validate_code_quality` after writing stays the mandatory gate.

### Cursor

```bash
mirdan init --cursor
```

Generates a complete Cursor 2.x integration:

- **Rules** — `.cursor/rules/*.mdc` (always-on, security, planning, plan-review, plan-verify, agent, language-specific) — the `mirdan-planning.mdc` rule carries the Haiku-proof plan format that native Plan Mode reads
- **Hooks** — `.cursor/hooks.json` with zero-token command-type hooks (validate-on-edit, shell-guard, staged-validate-on-stop), `.cursor/hooks/*.sh` scripts
- **Subagents** — `.cursor/agents/*.md` (quality-validator, security-scanner, plan-reviewer)
- **Skills** — `.cursor/skills/*/SKILL.md` following the [Agent Skills Standard](https://agentskills.io) (code, plan-review)
- **Commands** — `.cursor/commands/*.md` slash commands (`/code`, `/plan`, `/plan-verify`, `/plan-review`, `/automations`)
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
  enabled_events: ["PostToolUse", "Stop"]
  quick_validate_timeout: 5000
  auto_fix_suggestions: true
```

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
| `mirdan check` | Run lint + typecheck + test |
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
| `command not found: mirdan` | `uv tool install mirdan` |
| Server starts but no tools appear | Restart your IDE after config changes |
| Python version error | Ensure Python 3.11+ is installed |
| Hook not firing | Check hook stringency level — MINIMAL only fires on 2 events |
| Tool budget limiting tools | Set `MIRDAN_TOOL_BUDGET=5` or remove the env var |

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
