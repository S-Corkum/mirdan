# Mirdan Quality Orchestration

Use the mirdan MCP server for code quality enforcement on every coding task.

## Entry Point

Before writing any code, call `mcp__mirdan__enhance_prompt` with the task description.
Use the returned quality requirements as constraints during implementation.

## Exit Gate

After writing code, call `mcp__mirdan__validate_code_quality` with:
- The written code
- `check_security=true` if enhance_prompt indicated `touches_security`
- The detected language

Fix all errors before proceeding. Note warnings and address if reasonable.

## Quick Validation

For rapid feedback during editing, use `mcp__mirdan__validate_quick` which runs
security-only checks in under 500ms. Use the full `validate_code_quality` for
final validation before commits.
