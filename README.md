# Mirdan

AI Code Quality Orchestrator - Automatically transforms developer prompts into high-quality, structured requests that maximize AI coding assistant capabilities.

## The Problem

AI coding assistants produce "slop" not because the models are incapable, but because developers provide prompts that lack context, structure, and quality constraints. Research shows properly structured prompts achieve 15-74% better results.

## The Solution

Mirdan is an MCP server that intercepts prompts, automatically enhances them with quality requirements, codebase context, and architectural patterns, then intelligently orchestrates other available MCPs to ground the AI in reality.

## Features

- **Intent Analysis**: Classifies task type (generation, refactor, debug, review, test)
- **Quality Injection**: Applies language-specific coding standards and security requirements
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

### 1. Configure Your MCP Client

Add mirdan to your Claude Code configuration (`.mcp.json` in project root or `~/.claude.json` for global):

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

**Or via CLI:**
```bash
claude mcp add mirdan -- uvx mirdan
```

### 2. Verify It's Working

In Claude Code, run `/mcp` to see connected servers. Mirdan should appear in the list.

### 3. Start Using

Mirdan tools are now available. See the next section for how to make mirdan automatically orchestrate your coding tasks.

## Automatic Orchestration

Mirdan works best when it automatically enhances every coding task. There are two ways to achieve this:

### Option 1: CLAUDE.md Instructions (Recommended)

Add these instructions to your project's `CLAUDE.md` file (or `~/.claude/CLAUDE.md` for global). Claude will automatically follow them for all coding tasks.

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

### Option 2: Custom Slash Command

Create a slash command that enforces the mirdan workflow. Create this file:

**`.claude/commands/code.md`**
```markdown
---
description: Execute coding task with mirdan quality orchestration
---

Execute this coding task with full mirdan orchestration:

$ARGUMENTS

Follow these steps in order:

1. Call mcp__mirdan__enhance_prompt with the task description above
2. Review the quality_requirements and tool_recommendations from the response
3. If detected_frameworks lists libraries you're unfamiliar with, query context7
4. Implement the solution following the quality_requirements
5. Call mcp__mirdan__validate_code_quality on your completed code
6. If validation fails, fix all violations and re-validate until it passes
7. Call mcp__mirdan__get_verification_checklist and complete each item
8. Only report completion after validation passes and checklist is done
```

**Usage:**
```bash
/code implement user authentication with JWT tokens
```

### Which Should I Use? (Claude Code)

| Approach | Best For |
|----------|----------|
| **CLAUDE.md** | Automatic orchestration for all coding tasks without extra typing |
| **Slash command** | Explicit control over when orchestration runs |

**Recommended:** Start with CLAUDE.md for automatic orchestration. Add the slash command if you want an explicit trigger for complex tasks.

---

### Cursor: Project Rules

Cursor uses [Project Rules](https://cursor.com/docs/context/rules) to provide persistent instructions. Create a rule that applies to all coding sessions.

**Option 1: Modern Rules (Recommended)**

Create `.cursor/rules/mirdan-orchestration.md`:

```markdown
---
description: Mirdan code quality orchestration - automatically enhance and validate all coding tasks
alwaysApply: true
---

## Mirdan Code Quality Orchestration

When performing ANY coding task (writing, editing, debugging, refactoring), follow this workflow:

### 1. Entry Point (REQUIRED)
Before writing code, use mirdan's `enhance_prompt` tool with the task description.

From the response, use:
- `detected_frameworks` → query documentation if unfamiliar
- `touches_security` → enable stricter validation later
- `quality_requirements` → follow during implementation

### 2. Implementation
Write code following the quality_requirements from step 1.

### 3. Exit Gate (REQUIRED)
Before completing, use mirdan's `validate_code_quality` tool with your code.
- If validation fails, fix violations and re-validate
- Do not mark complete until validation passes

### 4. Verification
Use mirdan's `get_verification_checklist` tool and complete each item.
```

**Option 2: Legacy .cursorrules**

Alternatively, create `.cursorrules` in your project root with the same content (without the frontmatter).

> **Note:** `.cursorrules` is deprecated. Cursor recommends migrating to `.cursor/rules/` for better flexibility.

### Which Should I Use? (Cursor)

| Approach | Best For |
|----------|----------|
| **Project Rules** (`alwaysApply: true`) | Automatic orchestration for all coding tasks |
| **Project Rules** (with `globs`) | Orchestration only for specific file types |

**Recommended:** Use Project Rules with `alwaysApply: true` for consistent orchestration across all coding tasks.

### Available Tools

#### enhance_prompt

Automatically enhance a coding prompt with quality requirements and tool recommendations.

#### analyze_intent

Analyze a prompt without enhancement to understand the detected intent.

#### get_quality_standards

Retrieve quality standards for a language/framework combination.

#### suggest_tools

Get recommendations for which MCP tools to use.

#### get_verification_checklist

Get a verification checklist for a specific task type (generation, refactor, debug, review, test).

#### validate_code_quality

Validate generated code against quality standards. Checks for security issues, architecture patterns, and language-specific style violations.

## MCP Integration

Mirdan works with any MCP-compatible client.

### Claude Code

**File locations:**
- Project scope: `.mcp.json` (in project root)
- User scope: `~/.claude.json`

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

**CLI setup:**
```bash
claude mcp add mirdan -- uvx mirdan
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
