# Mirdan

AI Code Quality Orchestrator - Automatically transforms developer prompts into high-quality, structured requests that maximize AI coding assistant capabilities.

## The Problem

AI coding assistants produce "slop" not because the models are incapable, but because developers provide prompts that lack context, structure, and quality constraints. Research shows properly structured prompts achieve 15-74% better results.

## The Solution

Mirdan is an MCP server that intercepts prompts, automatically enhances them with quality requirements, codebase context, and architectural patterns, then intelligently orchestrates other available MCPs to ground the AI in reality.

## Features

- **Intent Analysis**: Classifies task type (generation, refactor, debug, review, test)
- **Quality Injection**: Applies language-specific coding standards and security requirements
- **Architecture Validation**: AST-based detection of function complexity, deep nesting, and import hygiene
- **Plan Validation**: Validates implementation plans for grounding, completeness, and cheap-model readiness
- **Prompt Composition**: Structures prompts using proven frameworks (Role/Goal/Constraints)
- **MCP Orchestration**: Recommends which tools to use for context gathering
- **Verification Checklists**: Generates task-specific verification steps

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
# Clone the repository
git clone https://github.com/S-Corkum/mirdan.git
cd mirdan

# Install dependencies
uv sync

# Run from source
uv run mirdan
```

## Quick Start

```bash
# 1. Add mirdan to Claude Code
claude mcp add mirdan -- uvx mirdan

# 2. Verify connection
# In Claude Code, run: /mcp
# Mirdan should appear in the connected servers list
```

**Next steps:** See the [Claude Code Integration](#claude-code-integration) section below for automatic orchestration setup.

For other MCP clients (Cursor, Claude Desktop), see [MCP Configuration Reference](#mcp-configuration-reference).

---

## Claude Code Integration

Claude Code provides multiple integration points for maximizing mirdan's effectiveness. This section covers all available methods from simple to advanced.

### Automatic Orchestration

Mirdan works best when it automatically enhances every coding task. Claude Code offers several integration levels:

| Level | Method | Effort | Enforcement |
|-------|--------|--------|-------------|
| **Basic** | CLAUDE.md | Copy-paste | Soft (instructions) |
| **Standard** | CLAUDE.md + Slash Commands | Copy-paste | Medium (explicit trigger) |
| **Advanced** | CLAUDE.md + Hooks | Configuration | Hard (automatic checks) |
| **Enterprise** | Managed Settings + Hooks | IT deployment | Mandatory |

---

### Level 1: CLAUDE.md Instructions (Recommended Start)

Add these instructions to your project's `CLAUDE.md` file. Claude will automatically follow them for all coding tasks.

**File location:** `./CLAUDE.md` (project) or `~/.claude/CLAUDE.md` (global)

```markdown
## Mirdan Code Quality Orchestration

When performing ANY coding task (writing, editing, debugging, refactoring code), follow this workflow:

### 1. Entry Point (REQUIRED)
Before writing any code, call `mcp__mirdan__enhance_prompt` with the task description.

Use the response to guide your work:
- `detected_frameworks` → query context7 for documentation if unfamiliar
- `touches_security` → use stricter validation in step 3
- `quality_requirements` → follow these during implementation
- `tool_recommendations` → use suggested MCPs for context gathering

### 2. Implementation
Write code following the quality_requirements from step 1.

### 3. Exit Gate (REQUIRED)
Before marking any coding task complete, call `mcp__mirdan__validate_code_quality` with your code.
- Set `check_security=true` if `touches_security` was true in step 1
- If validation fails, fix all violations and re-validate
- Code is NOT complete until validation passes

### 4. Verification
Call `mcp__mirdan__get_verification_checklist` for the task type and execute each item.
```

---

### Level 2: Slash Commands (Explicit Control)

Create custom slash commands for different workflows. Slash commands provide explicit triggers with full context.

#### `/code` - General Coding Tasks

**File:** `.claude/commands/code.md`

```markdown
---
description: Execute coding task with mirdan quality orchestration
allowed-tools: Read, Edit, Write, Bash(*), Grep, Glob
---

Execute this coding task with full mirdan orchestration:

$ARGUMENTS

## Workflow

1. **Entry Gate**: Call `mcp__mirdan__enhance_prompt` with the task above
2. **Context Gathering**:
   - Use `detected_frameworks` to query context7 for documentation
   - Use `tool_recommendations` for additional MCP calls
3. **Implementation**: Follow `quality_requirements` during coding
4. **Exit Gate**: Call `mcp__mirdan__validate_code_quality` on completed code
   - Set `check_security=true` if `touches_security` was true
   - Fix violations and re-validate until passed
5. **Verification**: Call `mcp__mirdan__get_verification_checklist` and complete each item

Code is NOT complete until validation passes and checklist is done.
```

**Usage:**
```bash
/code implement user authentication with JWT tokens
```

#### `/debug` - Debugging Tasks

**File:** `.claude/commands/debug.md`

```markdown
---
description: Debug issue with mirdan quality gates
allowed-tools: Read, Edit, Write, Bash(*), Grep, Glob
---

Debug this issue with mirdan orchestration:

$ARGUMENTS

## Workflow

1. **Classify**: Call `mcp__mirdan__analyze_intent` to understand security implications
2. **Investigate**: Analyze the issue thoroughly
3. **Fix**: Implement the fix
4. **Validate**: Call `mcp__mirdan__validate_code_quality` with `check_security=true`
   (bugs often have security implications)
5. **Verify**: Call `mcp__mirdan__get_verification_checklist(task_type="debug")` and complete each item

Fix is NOT complete until validation passes.
```

#### `/review` - Code Review

**File:** `.claude/commands/review.md`

```markdown
---
description: Review code with mirdan quality standards
allowed-tools: Read, Grep, Glob
---

Review this code with mirdan quality standards:

$ARGUMENTS

## Workflow

1. **Standards**: Call `mcp__mirdan__get_quality_standards` for the detected language
2. **Validate**: Call `mcp__mirdan__validate_code_quality` on the code
3. **Checklist**: Call `mcp__mirdan__get_verification_checklist(task_type="review")`
4. **Report**: Provide findings organized by:
   - Security issues (critical)
   - Quality violations
   - Improvement suggestions
```

---

### Level 3: Hooks (Automatic Enforcement)

Hooks provide automatic enforcement without requiring explicit commands. They run before or after specific tool calls.

#### Pre-Implementation Reminder

Reminds to use mirdan before writing code. Add to `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "prompt",
            "prompt": "Before modifying code, ensure you've called mcp__mirdan__enhance_prompt for the current task. Have you done this?",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

#### Post-Implementation Validation Gate

Automatically prompts for validation after code changes:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "prompt",
            "prompt": "Code was just modified. Run mcp__mirdan__validate_code_quality on the changes before proceeding. If security-related code, use check_security=true.",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

#### Combined Hooks Configuration

**File:** `.claude/settings.json`

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "echo '🔷 Mirdan code quality orchestration active'",
            "timeout": 5
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "prompt",
            "prompt": "Ensure mcp__mirdan__enhance_prompt was called for this task before writing code.",
            "timeout": 30
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "prompt",
            "prompt": "Code modified. Call mcp__mirdan__validate_code_quality before proceeding.",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

---

### Level 4: Project Rules (Path-Specific Enforcement)

Use `.claude/rules/` for path-specific quality requirements.

#### Security-Critical Paths

**File:** `.claude/rules/security.md`

```markdown
---
paths: ["**/auth/**", "**/security/**", "**/crypto/**", "**/*token*", "**/*session*"]
---

# Security-Critical Code Rules

This file is in a security-sensitive path. STRICT validation required.

## Mandatory Workflow

1. Call `mcp__mirdan__enhance_prompt` - task WILL be flagged as `touches_security=true`
2. Call `mcp__mirdan__get_quality_standards` with security focus
3. Implement with security-first mindset
4. Call `mcp__mirdan__validate_code_quality` with:
   - `check_security=true` (REQUIRED)
   - Address ALL security findings before completion
5. Complete full verification checklist

NO EXCEPTIONS. Security code is not complete until validation passes with zero security findings.
```

#### API Code Rules

**File:** `.claude/rules/api.md`

```markdown
---
paths: ["**/api/**", "**/routes/**", "**/endpoints/**", "**/handlers/**"]
---

# API Code Rules

API code requires input validation and error handling verification.

## Workflow

1. `mcp__mirdan__enhance_prompt` - include "API endpoint" in task description
2. `mcp__mirdan__get_quality_standards` for the language
3. Implement with:
   - Input validation for all external data
   - Proper error responses
   - Authentication/authorization checks
4. `mcp__mirdan__validate_code_quality` with `check_security=true`
5. Complete verification checklist
```

---

### Project Settings

Configure mirdan-related settings in `.claude/settings.json`:

```json
{
  "permissions": {
    "allow": [
      "mcp__mirdan__enhance_prompt",
      "mcp__mirdan__validate_code_quality",
      "mcp__mirdan__validate_quick",
      "mcp__mirdan__get_quality_standards",
      "mcp__mirdan__get_quality_trends"
    ]
  },
  "enableAllProjectMcpServers": true,
  "enabledMcpjsonServers": ["mirdan"]
}
```

---

### Enterprise Deployment

For organization-wide mirdan enforcement, IT can deploy managed configuration.

#### Managed MCP Configuration

**File (macOS):** `/Library/Application Support/ClaudeCode/managed-mcp.json`
**File (Linux):** `/etc/claude-code/managed-mcp.json`
**File (Windows):** `C:\Program Files\ClaudeCode\managed-mcp.json`

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

#### Managed Settings (Enforcement)

**File:** Same paths as above, but `managed-settings.json`

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "prompt",
            "prompt": "Company policy: Use mcp__mirdan__enhance_prompt before coding and mcp__mirdan__validate_code_quality after.",
            "timeout": 30
          }
        ]
      }
    ]
  },
  "allowManagedHooksOnly": true
}
```

---

### Which Approach Should I Use?

| Scenario | Recommended Setup |
|----------|-------------------|
| **Individual developer, trying mirdan** | Level 1 (CLAUDE.md only) |
| **Individual developer, daily use** | Level 1 + Level 2 (CLAUDE.md + slash commands) |
| **Team project** | Level 1 + Level 3 (CLAUDE.md + hooks) |
| **Security-sensitive project** | All levels including path-specific rules |
| **Enterprise/regulated environment** | Level 4 (managed settings + hooks) |

**Recommended progression:**
1. Start with CLAUDE.md (copy-paste, immediate benefit)
2. Add `/code` slash command for explicit control
3. Add hooks when you want automatic enforcement
4. Add path-specific rules for security-critical code

---

### Cursor: Full Integration (Cursor 1.7+)

Cursor's [Project Rules](https://cursor.com/docs/context/rules), [Hooks](https://cursor.com/docs/agent/hooks), and [AGENTS.md](https://cursor.com/docs/context/agents-md) provide a complete quality orchestration surface. mirdan leverages all three for **invisible quality enforcement** — you just code normally, mirdan handles the rest.

#### Automatic Setup (Recommended)

```bash
mirdan init --cursor
```

This generates everything needed for automatic quality enforcement:

| Generated | Path | Purpose |
|-----------|------|---------|
| Hooks | `.cursor/hooks.json` | Prompt-based hooks for afterFileEdit, preToolUse, stop, beforeSubmitPrompt |
| MCP config | `.cursor/mcp.json` | Registers mirdan as MCP server with tool budget |
| Project rules | `.cursor/rules/*.mdc` | Language-specific quality standards (code quality, security, debug) |
| AGENTS.md | `AGENTS.md` | Quality checkpoints, AI/security rules, quality thresholds for agents |
| BUGBOT.md | `BUGBOT.md` | Structured PR review rules with regex pattern matching |

After init, quality gates fire automatically with zero manual effort.

#### What Gets Generated

**`.cursor/hooks.json`** — Prompt-type hooks that fire automatically:

| Event | What It Does |
|-------|-------------|
| `afterFileEdit` | Calls `validate_code_quality` on changed code, fixes any errors |
| `preToolUse` | Security review before Write/Edit operations (SQL injection, secrets, command injection) |
| `stop` | Final quality gate — verifies all changed files were validated before task completion |
| `beforeSubmitPrompt` | Suggests calling `enhance_prompt` for quality requirements |

Hook stringency levels control how many events are active:

| Level | Events | Best For |
|-------|--------|----------|
| `MINIMAL` | afterFileEdit, stop | Low-friction onboarding |
| `STANDARD` | + preToolUse | Daily development |
| `COMPREHENSIVE` | + beforeSubmitPrompt | Teams, production projects |

**`.cursor/mcp.json`** — MCP server registration with tool budget:

```json
{
  "mcpServers": {
    "mirdan": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mirdan", "serve"],
      "env": {"MIRDAN_TOOL_BUDGET": "3"}
    }
  }
}
```

The `MIRDAN_TOOL_BUDGET` env var limits exposed tools by priority (useful for Cursor's tool slot limits):

| Budget | Tools Exposed |
|--------|--------------|
| 2 | validate_code_quality, validate_quick |
| 3 | + enhance_prompt |
| 4 | + get_quality_standards |
| 5 | All tools (default when unset) |

**`AGENTS.md`** — Enhanced with quality enforcement sections:

- **Mandatory Quality Checkpoints**: Before writing code, after every file edit, before PR/completion
- **AI Quality Rules (AI001–AI008)**: Inline rule descriptions with severity levels
- **Security Standards (SEC001–SEC010)**: Critical and important security rules
- **Quality Thresholds**: Code cannot be marked complete if quality score < 0.7

**`BUGBOT.md`** — Structured PR review rules for Cursor's BugBot:

- **Blocking Bugs** (severity: critical): AI001 placeholders, AI008 injection, SEC001–SEC003 hardcoded secrets/SQL injection/command injection with regex patterns
- **Request Changes** (severity: warning): AI003 over-abstraction, AI007 weak crypto, SEC006–SEC010
- **Best Practice** (severity: info): AI004–AI006 duplication/error handling, documentation

#### Granular Project Rules

For fine-grained control, `mirdan init --cursor` also generates focused rules in `.cursor/rules/`:

**`.cursor/rules/mirdan-code-quality/RULE.md`** — Auto-attaches on code files:

```markdown
---
description: Code quality orchestration for all code modifications
globs: ["*.py", "*.ts", "*.tsx", "*.js", "*.jsx", "*.go", "*.rs", "*.java", "*.rb"]
alwaysApply: false
---
# Mirdan Code Quality Gate

When modifying code files:

## Entry Point
Call `mirdan.enhance_prompt` with task description. Use response for:
- `detected_frameworks` → query documentation if unfamiliar
- `touches_security` → enables strict validation
- `quality_requirements` → constraints during implementation

## Exit Gate (REQUIRED)
Before completion:
1. Call `mirdan.validate_code_quality` on your code
2. Set `check_security=true` if touches_security was true
3. Fix all violations and re-validate until passed
```

**`.cursor/rules/mirdan-security/RULE.md`** — Auto-attaches on auth/security paths:

```markdown
---
description: Strict security validation for authentication and authorization code
globs: ["**/auth/**", "**/security/**", "**/crypto/**", "**/*token*", "**/*session*"]
alwaysApply: false
---
# Mirdan Security Gate

Code in security-sensitive paths requires STRICT validation:

1. `mirdan.validate_code_quality` with `check_security=true` and `severity_threshold="info"`
2. Resolve ALL security violations before completion
```

#### How Rules & Hooks Work Together

| Your Action | What Fires | Why |
|-------------|-----------|-----|
| Edit `api/routes.py` | `afterFileEdit` hook + mirdan-code-quality rule | Hook validates; rule provides context |
| Edit `src/auth/login.ts` | `afterFileEdit` hook + code-quality + security rules | Hook + both rules activate |
| Say "debug this error" | mirdan-debug rule | Description matches intent |
| Complete a task | `stop` hook | Final quality gate verifies all files validated |
| Submit a prompt | `beforeSubmitPrompt` hook | Suggests enhance_prompt for quality context |

#### Cursor Feature Integration

**Debug Mode**: mirdan augments Cursor's hypothesis-driven debugging. `enhance_prompt` classifies the bug, `validate_code_quality` ensures fixes don't introduce new vulnerabilities.

**Plan Mode**: When using Plan Mode, `enhance_prompt` at planning stage surfaces security considerations. When delegating steps to parallel agents, each follows mirdan-code-quality rules.

**Background Agents**: mirdan hooks and AGENTS.md provide guardrails for autonomous agents, ensuring quality even without human oversight.

**BugBot PR Reviews**: BUGBOT.md gives BugBot structured detection rules with regex patterns, catching placeholder code, injection vulnerabilities, and hardcoded secrets automatically.

### Automatic Setup with `mirdan init`

The fastest way to integrate mirdan with your IDE:

```bash
# Cursor (hooks + rules + AGENTS.md + BUGBOT.md + MCP config)
mirdan init --cursor

# Claude Code (hooks + rules + skills + agents + MCP config)
mirdan init --claude-code

# Both platforms at once
mirdan init --all
```

#### Cursor Setup (`mirdan init --cursor`)

| Generated | Path | Purpose |
|-----------|------|---------|
| Hooks | `.cursor/hooks.json` | Prompt hooks: afterFileEdit, preToolUse, stop, beforeSubmitPrompt |
| MCP config | `.cursor/mcp.json` | Registers mirdan MCP server with tool budget |
| Project rules | `.cursor/rules/*.mdc` | Language-specific quality standards |
| AGENTS.md | `AGENTS.md` | Quality checkpoints + inline AI/security rules |
| BUGBOT.md | `BUGBOT.md` | Structured PR review rules with regex patterns |

#### Claude Code Setup (`mirdan init --claude-code`)

| Generated | Path | Purpose |
|-----------|------|---------|
| MCP config | `.mcp.json` | Registers mirdan as MCP server |
| Hooks | `.claude/hooks.json` | Auto-validates edits (PostToolUse) and final output (Stop) |
| Quality rules | `.claude/rules/mirdan-*.md` | Language-specific quality standards |
| Skills | `.claude/skills/{code,debug,review}/SKILL.md` | `/mirdan:code`, `/mirdan:debug`, `/mirdan:review` |
| Agent | `.claude/agents/quality-gate.md` | Background quality validation subagent |

#### Additional Flags

| Flag | Effect |
|------|--------|
| `--all` | Run both `--cursor` and `--claude-code` setup |
| `--quality-profile PROFILE` | Set quality profile (e.g., `enterprise`, `startup`) |

After init, quality gates fire automatically with zero manual effort.

#### Plugin Distribution

mirdan can also be exported as a standalone Claude Code plugin:

```bash
mirdan plugin export --output-dir ./mirdan-plugin
# Install with: claude --plugin-dir ./mirdan-plugin
```

### Available Tools (5)

#### enhance_prompt

Automatically enhance a coding prompt with quality requirements and tool recommendations.

#### get_quality_standards

Retrieve quality standards for a language/framework combination.

#### validate_code_quality

Validate generated code against quality standards. Checks for security issues, architecture patterns, AI-specific quality rules, and language-specific style violations.

#### validate_quick

Fast validation mode (<500ms) for hook integration. Runs only security-critical rules (SEC001-SEC013, AI001, AI008).

#### get_quality_trends

Get quality score trends across validation sessions.

### AI Quality Rules

mirdan includes AI-specific quality rules that catch issues unique to AI-generated code:

| Rule | Severity | Description |
|------|----------|-------------|
| **AI001** | error | Placeholder detection — catches `raise NotImplementedError`, `pass` with TODO/FIXME comments (skips `@abstractmethod`) |
| **AI002** | warning | Hallucinated import detection — flags imports not found in Python stdlib or project dependencies |
| **AI008** | error | Injection vulnerability — catches f-string SQL, `eval`/`exec`/`os.system`/`subprocess` with f-strings |

AI001 and AI008 run in both full and quick validation modes (security-critical).

## MCP Configuration Reference

Mirdan works with any MCP-compatible client. This section provides quick configuration for each client.

> **Note:** For Claude Code users, see the comprehensive [Claude Code Integration](#claude-code-integration) section above for advanced setup including hooks, slash commands, and enterprise deployment.

### Claude Code (Quick Reference)

**CLI setup (recommended):**
```bash
claude mcp add mirdan -- uvx mirdan
```

**Or manual configuration:**

| Scope | File |
|-------|------|
| Project (team-shared) | `.mcp.json` |
| User (personal) | `~/.claude.json` |

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

### Claude Desktop

**File locations:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

**Configuration:**
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

### Cursor

**File locations:**
- Global: `~/.cursor/mcp.json`
- Project: `.cursor/mcp.json`

**Configuration:**
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

**UI setup:** File → Preferences → Cursor Settings → MCP

### From Source (Development)

If running from a local clone instead of PyPI:

```json
{
  "mcpServers": {
    "mirdan": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/mirdan", "run", "mirdan"]
    }
  }
}
```

## Configuration

Create a `.mirdan/config.yaml` in your project:

```yaml
version: "1.0"

project:
  name: "MyApp"
  primary_language: "typescript"
  frameworks: ["next.js", "prisma"]

quality:
  security: "strict"
  architecture: "moderate"
```

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

Enable verbose output for troubleshooting:

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
| Server starts but no tools appear | Restart Claude Code after config changes |
| Python version error | Ensure Python 3.11+ is installed |

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

## License

MIT
