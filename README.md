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

```bash
# Using uv
uv add mirdan

# Or with pip
pip install mirdan
```

## Quick Start

### As an MCP Server

Add to your Claude Code configuration:

```json
{
  "mcpServers": {
    "mirdan": {
      "command": "uv",
      "args": ["run", "mirdan"]
    }
  }
}
```

### Available Tools

#### enhance_prompt

Automatically enhance a coding prompt with quality requirements and tool recommendations.

#### analyze_intent

Analyze a prompt without enhancement to understand the detected intent.

#### get_quality_standards

Retrieve quality standards for a language/framework combination.

#### suggest_tools

Get recommendations for which MCP tools to use.

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
