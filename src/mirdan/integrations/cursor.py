"""Generate Cursor IDE configuration files for mirdan integration.

Supports .cursor/rules/*.mdc, .cursor/AGENTS.md, .cursor/BUGBOT.md,
.cursor/hooks.json, and .cursor/mcp.json generation.
"""

from __future__ import annotations

import json
from enum import Enum
from importlib.resources import files
from pathlib import Path
from typing import Any

from mirdan.cli.detect import DetectedProject
from mirdan.core.quality_standards import QualityStandards

# ---------------------------------------------------------------------------
# Cursor Hook Stringency
# ---------------------------------------------------------------------------


class CursorHookStringency(Enum):
    """Hook stringency levels for Cursor hooks.json generation (command-type only)."""

    MINIMAL = "minimal"  # afterFileEdit, stop
    STANDARD = "standard"  # + beforeShellExecution
    COMPREHENSIVE = "comprehensive"  # same as standard — all zero-token command hooks


# Command-type events per stringency level. All hooks are zero-token shell commands;
# quality guidance lives in rules, not prompt-type lifecycle hooks.
CURSOR_STRINGENCY_EVENTS: dict[CursorHookStringency, list[str]] = {
    CursorHookStringency.MINIMAL: ["afterFileEdit", "stop"],
    CursorHookStringency.STANDARD: ["afterFileEdit", "beforeShellExecution", "stop"],
    CursorHookStringency.COMPREHENSIVE: ["afterFileEdit", "beforeShellExecution", "stop"],
}


# ---------------------------------------------------------------------------
# Cursor Hooks Generation
# ---------------------------------------------------------------------------


def generate_cursor_hooks(
    cursor_dir: Path,
    stringency: CursorHookStringency = CursorHookStringency.COMPREHENSIVE,
) -> Path | None:
    """Generate .cursor/hooks.json with zero-token command-type hooks.

    All hooks are command-type shell scripts/CLI that run outside the model
    context — no prompt-type lifecycle hooks (quality guidance lives in rules).

    Args:
        cursor_dir: The .cursor/ directory to write into.
        stringency: Hook stringency level controlling which command hooks are emitted.

    Returns:
        Path to created hooks.json, or None if file already exists.
    """
    cursor_dir.mkdir(parents=True, exist_ok=True)
    hooks_path = cursor_dir / "hooks.json"

    # Respect user customizations — skip if already exists
    if hooks_path.exists():
        return None

    # Generate command-type hook scripts first
    generate_cursor_hook_scripts(cursor_dir)

    events = CURSOR_STRINGENCY_EVENTS[stringency]
    hooks: dict[str, list[dict[str, str | int]]] = {}

    # Zero-token command hooks only — quality guidance lives in rules, not prompt hooks.
    if "afterFileEdit" in events:
        hooks["afterFileEdit"] = [
            {"type": "command", "command": ".cursor/hooks/mirdan-validate-file.sh", "timeout": 10}
        ]
    if "beforeShellExecution" in events:
        hooks["beforeShellExecution"] = [
            {
                "type": "command",
                "command": ".cursor/hooks/mirdan-shell-guard.sh",
                "timeout": 5,
                "failClosed": True,
            }
        ]
    if "stop" in events:
        hooks["stop"] = [
            {"type": "command", "command": "mirdan validate --staged --format text", "timeout": 60}
        ]

    config = {"version": 1, "hooks": hooks}

    with hooks_path.open("w") as f:
        json.dump(config, f, indent=2)

    return hooks_path


# ---------------------------------------------------------------------------
# Command-Type Hook Scripts (.cursor/hooks/*.sh)
# ---------------------------------------------------------------------------

_SHELL_GUARD_SCRIPT = """\
#!/usr/bin/env bash
# mirdan shell guard — block destructive commands before execution.
# Reads JSON from stdin (Cursor hooks protocol), checks the command
# field against a deny-list of dangerous patterns.
#
# Exit 0 = allow, Exit 2 = block.
set -euo pipefail

INPUT=$(cat)
CMD_EXTRACT='import sys,json; print(json.load(sys.stdin).get("command",""))'
COMMAND=$(echo "$INPUT" | python3 -c "$CMD_EXTRACT" 2>/dev/null || echo "")

# Deny patterns: truly destructive operations only.
# Standard git workflow (commit, push <branch>, pull, fetch) is allowed.
if echo "$COMMAND" | grep -qE 'rm\\s+-rf\\s+/[^.]'; then
    echo '{"permission":"deny","user_message":"Blocked by mirdan: rm -rf on root path"}'
    exit 2
fi

if echo "$COMMAND" | grep -qiE 'DROP\\s+(TABLE|DATABASE|SCHEMA)'; then
    echo '{"permission":"deny","user_message":"Blocked by mirdan: destructive SQL detected"}'
    exit 2
fi

if echo "$COMMAND" | grep -qE 'git\\s+push\\s+--force\\s+(origin\\s+)?(main|master)'; then
    echo '{"permission":"deny","user_message":"Blocked by mirdan: force push to main/master"}'
    exit 2
fi

if echo "$COMMAND" | grep -qE 'git\\s+reset\\s+--hard'; then
    MSG='Blocked by mirdan: git reset --hard (use --soft or stash)'
    echo "{\\"permission\\":\\"deny\\",\\"user_message\\":\\"$MSG\\"}"
    exit 2
fi

echo '{"permission":"allow"}'
exit 0
"""

_VALIDATE_FILE_SCRIPT = """\
#!/usr/bin/env bash
# mirdan validate-file — validate the just-edited file (Cursor afterFileEdit hook).
# Reads JSON from stdin (Cursor hooks protocol), extracts the file path, and runs
# mirdan's deterministic quick validation. Advisory: surfaces findings, never blocks.
set -euo pipefail

INPUT=$(cat)
PATH_EXTRACT='import sys,json; print(json.load(sys.stdin).get("file_path",""))'
FILE=$(echo "$INPUT" | python3 -c "$PATH_EXTRACT" 2>/dev/null || echo "")

[ -z "$FILE" ] && exit 0
[ -f "$FILE" ] || exit 0

mirdan validate --quick --scope security --file "$FILE" --format micro 2>&1 || true
exit 0
"""


def generate_cursor_hook_scripts(cursor_dir: Path) -> list[Path]:
    """Generate .cursor/hooks/*.sh command-type hook scripts.

    Creates executable shell scripts for the zero-token command-type hooks
    (deterministic, no model overhead).

    Existing scripts are preserved — this function is idempotent per file.

    Args:
        cursor_dir: The .cursor/ directory to write into.

    Returns:
        List of newly created script file paths.
    """
    hooks_dir = cursor_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    scripts = {
        "mirdan-shell-guard.sh": _SHELL_GUARD_SCRIPT,
        "mirdan-validate-file.sh": _VALIDATE_FILE_SCRIPT,
    }

    created: list[Path] = []
    for filename, content in scripts.items():
        dest = hooks_dir / filename
        if not dest.exists():
            dest.write_text(content)
            dest.chmod(0o755)
            created.append(dest)

    return created


# ---------------------------------------------------------------------------
# Rule Generation (existing)
# ---------------------------------------------------------------------------


def generate_cursor_rules(
    rules_dir: Path,
    detected: DetectedProject,
    standards: QualityStandards | None = None,
    languages: list[str] | None = None,
) -> list[Path]:
    """Generate .cursor/rules/*.mdc files with dynamic standards content.

    If a QualityStandards instance is provided, generates dynamic .mdc files
    from actual quality rules. Otherwise falls back to static templates.

    Args:
        rules_dir: The .cursor/rules/ directory to write into.
        detected: Detected project metadata.
        standards: Optional QualityStandards for dynamic generation.
        languages: Optional list of languages to generate rules for (workspace mode).

    Returns:
        List of generated file paths.
    """
    generated: list[Path] = []

    if standards:
        generated.extend(
            _generate_dynamic_rules(rules_dir, detected, standards, languages=languages)
        )
    else:
        generated.extend(_generate_static_rules(rules_dir, detected, languages=languages))

    return generated


def generate_cursor_agents(
    cursor_dir: Path,
    detected: DetectedProject,
    standards: QualityStandards | None = None,
) -> list[Path]:
    """Generate .cursor/AGENTS.md and .cursor/BUGBOT.md.

    Args:
        cursor_dir: The .cursor/ directory to write into.
        detected: Detected project metadata.
        standards: Optional QualityStandards for content generation.

    Returns:
        List of generated file paths.
    """
    generated: list[Path] = []
    cursor_dir.mkdir(parents=True, exist_ok=True)

    # AGENTS.md — quality gate agent instructions for Cursor background agents
    agents_content = _generate_agents_md(detected, standards)
    agents_path = cursor_dir / "AGENTS.md"
    agents_path.write_text(agents_content)
    generated.append(agents_path)

    # BUGBOT.md — security standards for BugBot PR review integration
    bugbot_content = _generate_bugbot_md(detected, standards)
    bugbot_path = cursor_dir / "BUGBOT.md"
    bugbot_path.write_text(bugbot_content)
    generated.append(bugbot_path)

    return generated


# ---------------------------------------------------------------------------
# Dynamic generation (from QualityStandards)
# ---------------------------------------------------------------------------


def _generate_dynamic_rules(
    rules_dir: Path,
    detected: DetectedProject,
    standards: QualityStandards,
    languages: list[str] | None = None,
) -> list[Path]:
    """Generate .mdc files from QualityStandards data."""
    generated: list[Path] = []

    # Always-on rule
    always_content = _build_always_mdc(standards)
    path = rules_dir / "mirdan-always.mdc"
    path.write_text(always_content)
    generated.append(path)

    # Language-specific rules — iterate all languages in workspace mode
    langs = languages or ([detected.primary_language] if detected.primary_language else [])
    generated_names: set[str] = set()
    for lang in langs:
        if lang == "python" and "mirdan-python.mdc" not in generated_names:
            content = _build_language_mdc("python", "**/*.py", standards)
            path = rules_dir / "mirdan-python.mdc"
            path.write_text(content)
            generated.append(path)
            generated_names.add("mirdan-python.mdc")
        if lang in ("typescript", "javascript") and "mirdan-typescript.mdc" not in generated_names:
            content = _build_language_mdc("typescript", "**/*.{ts,tsx,js,jsx}", standards)
            path = rules_dir / "mirdan-typescript.mdc"
            path.write_text(content)
            generated.append(path)
            generated_names.add("mirdan-typescript.mdc")
        if lang == "go" and "mirdan-go.mdc" not in generated_names:
            content = _build_language_mdc("go", "**/*.go", standards)
            path = rules_dir / "mirdan-go.mdc"
            path.write_text(content)
            generated.append(path)
            generated_names.add("mirdan-go.mdc")
        if lang == "rust" and "mirdan-rust.mdc" not in generated_names:
            content = _build_language_mdc("rust", "**/*.rs", standards)
            path = rules_dir / "mirdan-rust.mdc"
            path.write_text(content)
            generated.append(path)
            generated_names.add("mirdan-rust.mdc")

    # Security rule (always)
    security_content = _build_security_mdc(standards)
    path = rules_dir / "mirdan-security.mdc"
    path.write_text(security_content)
    generated.append(path)

    # Planning rule (always)
    planning_content = _build_planning_mdc()
    path = rules_dir / "mirdan-planning.mdc"
    path.write_text(planning_content)
    generated.append(path)

    # Plan review rule (always)
    plan_review_content = _build_plan_review_mdc()
    path = rules_dir / "mirdan-plan-review.mdc"
    path.write_text(plan_review_content)
    generated.append(path)

    # Plan verify rule (always)
    plan_verify_content = _build_plan_verify_mdc()
    path = rules_dir / "mirdan-plan-verify.mdc"
    path.write_text(plan_verify_content)
    generated.append(path)

    # Agent rule (always)
    agent_content = _build_agent_mdc()
    path = rules_dir / "mirdan-agent.mdc"
    path.write_text(agent_content)
    generated.append(path)

    return generated


def _build_always_mdc(standards: QualityStandards) -> str:
    """Build the always-on mirdan rule."""
    return """---
description: "mirdan quality standards — always active"
globs: "**/*"
alwaysApply: true
---

# mirdan Quality Standards

## AI-Specific Quality Rules (Always Active)
- **AI001**: No placeholder code (raise NotImplementedError, pass with TODO)
- **AI002**: No hallucinated imports (verify all imports exist in dependencies)
- **AI003**: No over-engineering (unnecessary abstractions, excessive generics)
- **AI004**: No duplicate code blocks (extract shared logic)
- **AI005**: Consistent error handling patterns within each file
- **AI006**: Prefer lightweight alternatives for simple operations
- **AI007**: No security theater (hash() on passwords, always-true validators)
- **AI008**: No injection vulnerabilities (no f-string SQL, eval, exec)

## Quality Workflow
1. Before writing code: consider quality requirements
2. After writing code: validate with mirdan
3. Fix all errors before committing
"""


def _build_language_mdc(language: str, globs: str, standards: QualityStandards) -> str:
    """Build a language-specific .mdc rule from standards."""
    rules = standards.get_all_standards(language=language, category="all")
    rules_text = ""
    if rules:
        for category, items in rules.items():
            if isinstance(items, list):
                rules_text += f"\n### {category.title()}\n"
                for item in items[:10]:  # Cap at 10 per category
                    if isinstance(item, dict):
                        desc = item.get("description", item.get("message", ""))
                        rules_text += f"- **{item.get('id', '')}**: {desc}\n"
                    elif isinstance(item, str):
                        rules_text += f"- {item}\n"

    return f"""---
description: "mirdan {language} quality standards"
globs: "{globs}"
---

# mirdan {language.title()} Standards
{
        rules_text
        if rules_text
        else f'''
## Code Quality
- Follow {language} best practices and idioms
- Use type annotations where supported
- Handle errors explicitly
- Keep functions focused and small
'''
    }
"""


def _build_security_mdc(standards: QualityStandards) -> str:
    """Build the security .mdc rule from standards."""
    return """---
description: "mirdan security standards"
globs: "**/*"
alwaysApply: true
---

# mirdan Security Standards

## Critical (Errors)
- **AI007**: No security theater (hash() on passwords, always-true validators, MD5 for auth)
- **AI008**: No injection via string interpolation (SQL, eval, exec, os.system, subprocess)
- **SEC001**: No hardcoded secrets or API keys
- **SEC002**: No SQL injection via string concatenation
- **SEC003**: No command injection via unsanitized input
- **SEC004**: No path traversal vulnerabilities
- **SEC005**: No insecure deserialization (pickle.loads on untrusted data)

## Important (Warnings)
- **SEC006**: Use HTTPS for all external requests
- **SEC007**: Validate and sanitize all user input
- **SEC008**: Use parameterized queries for database operations
- **SEC009**: Apply principle of least privilege
- **SEC010**: Log security events without exposing sensitive data
"""


def _build_planning_mdc() -> str:
    """Build the planning .mdc rule — loads from static template."""
    templates = _load_templates()
    try:
        return templates["mirdan-planning.mdc"]
    except KeyError:
        return "# mirdan Planning Enhancement\n\nTemplate not found.\n"


def _build_plan_review_mdc() -> str:
    """Build the plan-review .mdc rule — loads from static template."""
    templates = _load_templates()
    try:
        return templates["mirdan-plan-review.mdc"]
    except KeyError:
        return "# mirdan Plan Review\n\nTemplate not found.\n"


def _build_plan_verify_mdc() -> str:
    """Build the plan-verify .mdc rule — loads from static template."""
    templates = _load_templates()
    try:
        return templates["mirdan-plan-verify.mdc"]
    except KeyError:
        return "# mirdan Plan Verify\n\nTemplate not found.\n"


def _build_agent_mdc() -> str:
    """Build the agent .mdc rule — loads from static template."""
    templates = _load_templates()
    try:
        return templates["mirdan-agent.mdc"]
    except KeyError:
        return "# mirdan Agent Mode Standards\n\nTemplate not found.\n"


# ---------------------------------------------------------------------------
# Static generation (fallback: copies templates)
# ---------------------------------------------------------------------------


def _generate_static_rules(
    rules_dir: Path,
    detected: DetectedProject,
    languages: list[str] | None = None,
) -> list[Path]:
    """Generate .mdc files from static templates (legacy fallback)."""
    generated: list[Path] = []
    templates = _load_templates()

    if "mirdan-always.mdc" in templates:
        path = rules_dir / "mirdan-always.mdc"
        path.write_text(templates["mirdan-always.mdc"])
        generated.append(path)

    # Language-specific rules — iterate all languages in workspace mode
    langs = languages or ([detected.primary_language] if detected.primary_language else [])
    generated_names: set[str] = set()
    for lang in langs:
        if (
            lang == "python"
            and "mirdan-python.mdc" in templates
            and "mirdan-python.mdc" not in generated_names
        ):
            path = rules_dir / "mirdan-python.mdc"
            path.write_text(templates["mirdan-python.mdc"])
            generated.append(path)
            generated_names.add("mirdan-python.mdc")

        if (
            lang in ("typescript", "javascript")
            and "mirdan-typescript.mdc" in templates
            and "mirdan-typescript.mdc" not in generated_names
        ):
            path = rules_dir / "mirdan-typescript.mdc"
            path.write_text(templates["mirdan-typescript.mdc"])
            generated.append(path)
            generated_names.add("mirdan-typescript.mdc")

    if "mirdan-security.mdc" in templates:
        path = rules_dir / "mirdan-security.mdc"
        path.write_text(templates["mirdan-security.mdc"])
        generated.append(path)

    if "mirdan-planning.mdc" in templates:
        path = rules_dir / "mirdan-planning.mdc"
        path.write_text(templates["mirdan-planning.mdc"])
        generated.append(path)

    if "mirdan-plan-verify.mdc" in templates:
        path = rules_dir / "mirdan-plan-verify.mdc"
        path.write_text(templates["mirdan-plan-verify.mdc"])
        generated.append(path)

    if "mirdan-plan-review.mdc" in templates:
        path = rules_dir / "mirdan-plan-review.mdc"
        path.write_text(templates["mirdan-plan-review.mdc"])
        generated.append(path)

    if "mirdan-agent.mdc" in templates:
        path = rules_dir / "mirdan-agent.mdc"
        path.write_text(templates["mirdan-agent.mdc"])
        generated.append(path)

    return generated


# ---------------------------------------------------------------------------
# AGENTS.md / BUGBOT.md generation (enhanced for v0.4.0)
# ---------------------------------------------------------------------------


def _generate_agents_md(
    detected: DetectedProject,
    standards: QualityStandards | None,
) -> str:
    """Generate enhanced AGENTS.md for Cursor background agents.

    Delegates to AgentsMDGenerator for base content, then appends
    enhanced sections: quality checkpoints, AI/SEC rules inline,
    and quality thresholds.
    """
    from mirdan.integrations.agents_md import AgentsMDGenerator

    generator = AgentsMDGenerator(standards=standards)
    base_content = generator.generate(detected, platform="cursor")

    # Append enhanced sections for v0.4.0
    enhanced = (
        base_content
        + _AGENTS_QUALITY_CHECKPOINTS
        + _AGENTS_AI_RULES
        + _AGENTS_SEC_RULES
        + _AGENTS_THRESHOLDS
    )

    return enhanced


_AGENTS_QUALITY_CHECKPOINTS = """
## Quality Checkpoints

### Before Writing Code
- `enhance_prompt` is optional by default and recommended before security-sensitive,
  multi-file, or new-library work — call `mcp__mirdan__enhance_prompt` to get quality
  requirements and review `quality_requirements` / `touches_security` in the response

### After Every File Edit
- Call `mcp__mirdan__validate_code_quality` on all changed files
- Fix any violations with severity "error" immediately
- Note warnings for awareness

### Before PR / Completion
- Verify quality score >= 0.7 for all changed files
- Ensure no unresolved errors remain from validation
- Security-sensitive code must pass with `check_security=true`

### Periodic (Every 30 Minutes)
- Run `validate_code_quality` on all recently modified files
- Check for quality regressions introduced during the session

### Long-Running Sessions (>1 hour)
- Re-validate ALL changed files at 1-hour intervals, not just recent ones
- Track cumulative quality drift — if average score decreases, pause and fix
- Before creating any PR, run full validation sweep across the entire changeset
- For sessions >4 hours, generate a quality summary report at each checkpoint
"""

_AGENTS_AI_RULES = """
## AI Quality Rules (Inline Reference)

| Rule | Description | Severity |
|------|-------------|----------|
| AI001 | No placeholder code (`raise NotImplementedError`, `pass` with TODO) | Error |
| AI002 | No hallucinated imports (verify all imports exist in dependencies) | Warning |
| AI003 | No over-engineering (unnecessary abstractions for single-use code) | Warning |
| AI004 | No duplicate code blocks (extract shared logic) | Warning |
| AI005 | Consistent error handling patterns within each file | Warning |
| AI006 | Prefer lightweight alternatives for simple operations | Info |
| AI007 | No security theater (`hash()` on passwords, always-true validators) | Error |
| AI008 | No injection vulnerabilities (f-string SQL, eval, exec with user input) | Error |
"""

_AGENTS_SEC_RULES = """
## Security Standards (Inline Reference)

| Rule | Description | Severity |
|------|-------------|----------|
| SEC001 | No hardcoded secrets or API keys | Critical |
| SEC002 | No SQL injection via string concatenation | Critical |
| SEC003 | No command injection via unsanitized input | Critical |
| SEC004 | No path traversal vulnerabilities | Critical |
| SEC005 | No insecure deserialization (pickle.loads on untrusted data) | Critical |
| SEC006 | Use HTTPS for all external requests | Important |
| SEC007 | Validate and sanitize all user input | Important |
| SEC008 | Use parameterized queries for database operations | Important |
| SEC009 | Apply principle of least privilege | Important |
| SEC010 | Log security events without exposing sensitive data | Important |
| SEC011 | No Cypher injection — use parameterized graph queries | Important |
| SEC012 | No Gremlin injection — use parameterized traversals | Important |
| SEC013 | Use bcrypt or argon2 for password hashing, never MD5/SHA | Important |
| SEC014 | No vulnerable dependencies — upgrade packages with known CVEs | Important |
"""

_AGENTS_THRESHOLDS = """
## Quality Thresholds

- **Minimum passing score**: 0.7
- Do NOT mark a task complete if any changed file has quality score below 0.7
- If `validate_code_quality` returns errors, fix them before proceeding
- Security-critical files (auth, API endpoints, input handling) require score >= 0.8
"""


def _generate_bugbot_md(
    detected: DetectedProject,
    standards: QualityStandards | None,
) -> str:
    """Generate enhanced BUGBOT.md with structured detection rules and regex patterns."""
    return """# BugBot — mirdan Quality Standards

> Auto-generated by mirdan. Regenerate with `mirdan init --upgrade`.

## Blocking Bugs (Severity: Critical)

These patterns MUST block the PR. They indicate incomplete, insecure, or broken code.

### AI001 — Placeholder Code
```regex
NotImplementedError
pass\\s*#\\s*TODO
raise NotImplementedError
```

### AI008 — Injection Vulnerabilities
```regex
f["'].*\\{.*\\}.*["']
```
Flag when found in SQL query context, `eval()`, `exec()`, or `os.system()` calls.

### SEC001 — Hardcoded Secrets
```regex
(?:password|api_key|secret|token|auth)\\s*=\\s*["'][^"']+["']
```

### SEC002 — SQL Injection
```regex
f["']SELECT.*\\{
"SELECT.*"\\s*\\+
```

### SEC003 — Command Injection
```regex
os\\.system\\(
subprocess\\.(?:call|run)\\(.*shell\\s*=\\s*True
```
Flag when combined with user-controlled input or f-strings.

## Request Changes (Severity: Warning)

These patterns should trigger a change request in the review.

### AI003 — Over-Engineering
- Excessive abstraction layers for single-use code
- Generic type parameters on classes used once
- Factory patterns for single implementations

### AI007 — Security Theater
```regex
hash\\(
md5
```
Flag `hash()` when used for passwords. Flag `md5` when used for authentication.

### SEC006 — Insecure HTTP
```regex
http://(?!localhost|127\\.0\\.0\\.1)
```

### SEC007 — Missing Input Validation
- External data used without validation or sanitization
- User input passed directly to database queries

### SEC008 — Non-Parameterized Queries
```regex
cursor\\.execute\\(f["']
cursor\\.execute\\(.*%.*%
```

### SEC009 — Excessive Privileges
- Overly broad permissions or roles
- `chmod 777` or equivalent

### SEC010 — Sensitive Data in Logs
```regex
log(?:ger)?\\.(?:info|debug|warning|error)\\(.*(?:password|secret|token|key)
```

### SEC011 — Cypher Injection
```regex
f["']\\s*MATCH|f["']\\s*CREATE|f["']\\s*MERGE|\\.run\\(f["']
```

### SEC012 — Gremlin Injection
```regex
f["']\\s*g\\.V|f["']\\s*g\\.E|\\.submit\\(f["']
```

### SEC013 — Weak Password Hashing
```regex
hashlib\\.(?:md5|sha1|sha256)\\(.*password
md5\\(.*password
```

### SEC014 — Vulnerable Dependencies
```regex
# Advisory: Check dependencies against known CVE databases
# No single regex — use `pip audit`, `safety check`, or `npm audit`
```

## Best Practice (Severity: Info)

These patterns are advisory and should be mentioned as comments.

### AI004 — Code Duplication
- Three or more nearly identical code blocks that should be extracted

### AI005 — Inconsistent Error Handling
- Mixed patterns: some functions use exceptions, others return error codes

### AI006 — Heavy Imports for Simple Operations
- Using pandas for simple CSV reading
- Using requests for a single GET call when urllib suffices

### Documentation
- Public APIs missing docstrings or type annotations
- Complex functions (>15 lines) without inline comments explaining logic

## Bugbot Autofix Integration

When Bugbot Autofix spawns a Cloud Agent to fix detected issues:

### Quality Gate for Auto-Generated Fixes
- Autofix changes MUST pass `mcp__mirdan__validate_code_quality` with score >= 0.7
- Security-related fixes MUST pass with `check_security=true` and score >= 0.8
- Fixes that introduce new violations (AI001-AI008, SEC001-SEC014) should be rejected

### Merge Protocol
- Use `@cursor approve` to merge autofix changes that pass quality gate
- Use `@cursor reject` to reject fixes that fail validation or introduce regressions
- Review autofix diffs for over-engineering (AI003) before approving

### Autofix Priority
- Critical (auto-fix immediately): SEC001, SEC002, SEC003, AI008
- High (auto-fix with review): SEC004-SEC005, AI001, AI007
- Medium (suggest fix, require approval): AI003-AI006, SEC006-SEC014
"""


# ---------------------------------------------------------------------------
# Cursor Commands Generation (.cursor/commands/*.md)
# ---------------------------------------------------------------------------

_COMMAND_CODE = """\
# /code — Quality-Orchestrated Coding

Execute coding tasks with automatic quality enforcement via mirdan.

## Workflow

1. (Optional, recommended for security-sensitive / multi-file / new-library work) Call
   `mcp__mirdan__enhance_prompt` with the task for quality requirements, detected language,
   and security sensitivity.

2. Call `mcp__enyal__enyal_recall` with `input: { query: "<task description>" }`
   to load project conventions, past decisions, and relevant patterns before
   writing any code.

3. Use `mcp__sequential-thinking__sequentialthinking` to plan the implementation
   approach: break down the task, identify components, dependencies, and edge
   cases before writing code.

4. Use `@Docs [library-name]` to look up current API documentation for any
   libraries involved in this task.

5. Follow the `quality_requirements` from enhance_prompt and conventions from
   enyal_recall as constraints during implementation.

6. After writing code, call `mcp__mirdan__validate_code_quality` on changed files.
   Set `check_security=true` if `touches_security` was flagged.

7. Call `mcp__enyal__enyal_remember` with
   `input: { content: "<knowledge>", content_type: "<type>", tags: [...] }`
   to store any new decisions, patterns, or conventions discovered.

**Note:** All enyal tools require parameters inside an `input` object.

8. Fix all errors before marking complete. Note warnings for review.
"""

_COMMAND_AUTOMATIONS = """\
# /automations — Cursor Automations Setup Guide

Guide for setting up Cursor Automations with mirdan quality enforcement.

## About Cursor Automations

Cursor Automations are always-on agents configured at cursor.com/automations.
They run in cloud sandboxes, triggered by events such as GitHub PRs, Slack
messages, or scheduled timers. They execute autonomously and report results
back via comments, PRs, or notifications.

## Recommended Automations with mirdan

### 1. PR Quality Gate
**Trigger:** GitHub PR opened or pushed
**Instructions:**
> Run `mcp__mirdan__validate_code_quality` with `check_security=true` on all
> changed files. Require quality score >= 0.7 for all files. Post a summary
> comment on the PR with per-file scores and any violations found. If any
> file fails, request changes. If all pass, approve the PR.

### 2. Scheduled Quality Audit
**Trigger:** Scheduled (daily or weekly)
**Instructions:**
> Run `mcp__mirdan__scan_conventions` across the entire codebase. Compare
> results with the previous audit. Report new violations, resolved violations,
> and quality trend direction. Post results to the configured notification
> channel.

### 3. Security Review on Sensitive Files
**Trigger:** GitHub PR with file filter (`**/auth*`, `**/api/**`, `**/middleware/**`)
**Instructions:**
> Run `mcp__mirdan__validate_code_quality` with `check_security=true` and
> `severity_threshold="warning"` on all matched files. Flag any SEC001-SEC005
> violations as blocking. Require score >= 0.8 for security-sensitive files.
> Post detailed findings as inline PR comments.

## Setup Steps

1. Go to cursor.com/automations
2. Click "Create New Automation"
3. Select the trigger type (GitHub PR, Scheduled, etc.)
4. Copy the instructions from the relevant template above
5. Under "MCP Servers", add mirdan (ensure it's configured in your project)
6. Under "Environment", enable the mirdan-quality environment
7. Save and activate the automation
"""

_CURSOR_COMMANDS: dict[str, str] = {
    "code.md": _COMMAND_CODE,
    "automations.md": _COMMAND_AUTOMATIONS,
}


def generate_cursor_commands(cursor_dir: Path, *, force: bool = False) -> list[Path]:
    """Generate .cursor/commands/*.md files for Cursor slash commands.

    Writes two groups of commands:

    1. **Always-available commands** from the in-file ``_CURSOR_COMMANDS`` dict:
       ``/code`` and ``/automations``.

    2. **Planning commands** from packaged templates at
       ``mirdan/integrations/templates/cursor_commands/*.md``:
       ``/plan`` (flat, with a low-level design), ``/plan-verify``, ``/plan-review``.

    Existing files are preserved — idempotent per file unless ``force=True``.

    Args:
        cursor_dir: The .cursor/ directory to write into.
        force: If True, overwrite existing files with latest content.

    Returns:
        List of newly created (or overwritten) command file paths.
    """
    commands_dir = cursor_dir / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []

    for filename, content in _CURSOR_COMMANDS.items():
        dest = commands_dir / filename
        if force or not dest.exists():
            dest.write_text(content)
            created.append(dest)

    # Planning commands from packaged templates (the canonical /plan source).
    pipeline_filenames = (
        "plan.md",
        "plan-verify.md",
        "plan-review.md",
    )
    try:
        pipeline_pkg = files("mirdan.integrations.templates.cursor_commands")
    except (ModuleNotFoundError, FileNotFoundError):
        return created

    for filename in pipeline_filenames:
        try:
            content = (pipeline_pkg / filename).read_text()
        except (FileNotFoundError, OSError):
            continue
        dest = commands_dir / filename
        if force or not dest.exists():
            dest.write_text(content)
            if dest not in created:
                created.append(dest)

    return created


# ---------------------------------------------------------------------------
# Cursor Subagents Generation (.cursor/agents/*.md)
# ---------------------------------------------------------------------------

_SUBAGENT_QUALITY_VALIDATOR = """\
---
name: mirdan-quality-validator
description: >-
  Validate code quality against mirdan standards in one pass — AI slop (AI001-008),
  security, architecture, test coverage, and style. Use after writing or editing code files.
model: fast
readonly: true
is_background: true
---

# mirdan Quality Validator

Validate recently changed code files against mirdan quality standards.

## Instructions

1. Identify recently changed or created files using semantic search or
   file search.

2. Read each file to understand its purpose and context.

3. Call `mcp__mirdan__validate_code_quality` for each file with:
   - `check_security=true`
   - `check_architecture=true`
   - `check_style=true`
   - `severity_threshold="info"`

4. Aggregate findings across all files.

5. Report results as a markdown summary, grouped by family (Security / Architecture /
   AI-slop / Test / Style):
   - Total files checked
   - Overall quality score
   - Errors (must fix) with file path, line number, and rule ID
   - Warnings (should fix) with file path, line number, and rule ID

## Async Execution Notes

This subagent runs in the background (async). The parent agent continues
working while validation runs. Results are returned via agent ID — the
parent can resume this subagent to retrieve findings.
"""

_SUBAGENT_SECURITY_SCANNER = """\
---
name: mirdan-security-scanner
description: >-
  Scan files for security vulnerabilities including injection, hardcoded
  secrets, and path traversal. Use to review files that handle
  authentication, user input, database queries, or file I/O.
model: inherit
readonly: true
is_background: false
---

# mirdan Security Scanner

Proactively scan security-sensitive files using mirdan validation.

## Trigger Patterns

Activate for files matching: `**/auth*`, `**/*login*`, `**/*password*`,
`**/api/**`, `**/middleware/**`, `**/*token*`, `**/*session*`, `**/*crypto*`,
or files containing SQL queries, eval/exec calls, or subprocess usage.

## Instructions

1. Use file search to identify files matching security patterns.

2. Read each file completely.

3. Call `mcp__mirdan__validate_code_quality` for each file with:
   - `check_security=true`
   - `severity_threshold="warning"`

4. For each violation, note:
   - Rule ID (SEC001-SEC014, AI007, AI008)
   - Exact line number
   - Fix recommendation

5. Report: files scanned, critical findings count, warnings count,
   and violations listed by rule ID with line numbers and recommendations.

## Subagent Coordination

This subagent runs in the foreground (blocking) because security findings
must be addressed before proceeding. It may spawn child subagents for
parallel file scanning when many files match trigger patterns.
"""

_SUBAGENT_PLAN_REVIEWER = """\
---
name: mirdan-plan-reviewer
description: >-
  Plan review subagent — verifies AI-generated implementation plans for
  grounding accuracy and cheap-model executability. Produces a structured
  7-dimension review report with PASS/REVISE/FAIL verdict.
model: sonnet
readonly: true
is_background: true
---

# mirdan Plan Reviewer

You are a plan review agent. Verify AI-generated implementation plans for
factual accuracy and executability by cheaper models (Haiku, Flash).

## Instructions

1. Read the plan from the provided path.

2. Call `mcp__mirdan__enhance_prompt` with plan text and
   `task_type="plan_validation"` for structural validation.

3. Extract all verifiable references (file paths, line numbers,
   function names, imports, step refs, directories).

4. For each reference, verify with tools (Read, Glob, Grep).
   Apply false-positive prevention:
   - NEW files → verify parent directory instead
   - Cumulative state: Step N creates → Step M>N can reference
   - Line drift from insertions → warning, not error
   Classify: VERIFIED / MISMATCH / NOT_FOUND / EXPECTED_NEW / UNVERIFIABLE

5. For any code snippets in the plan, call `mcp__mirdan__validate_code_quality`
   to check for AI quality rules (AI001-AI008).

6. Check dependency ordering (circular deps, creation before modification).

7. Semantic review: completeness, executability, safety, architecture. For every
   `Action: Edit`, confirm a unique ```anchor```/```replace``` pair (anchor findable
   exactly once in the target file); flag any unresolved decision (TBD / either-or).

8. Score 7 dimensions (Grounding 30%, Completeness 20%, Atomicity 15%,
   Clarity 10%, Dependency 10%, Executability 10%, Safety 5%). Atomicity 1.0 requires
   every non-capable Edit to carry a unique anchor/replace.

9. Output verdict: PASS (>=0.95 for haiku) / REVISE (>=0.5) / FAIL (<0.5). For a haiku
   target, any unresolved decision or `[target: capable]` step → at most REVISE.

## Report Format

Report each finding as:
- **[DIMENSION/CONFIDENCE]** Description — Fix: recommendation

Example:
- **[GROUNDING/HIGH]** File `src/auth.py` exists but line 45 is `import os`,
  not `def validate_token()` as claimed. Fix: Update to line 72.

## Async Execution Notes

This subagent runs in the background. It reads the plan, verifies references,
and produces a review report. It does not modify any files (readonly).
Results are returned as a structured report when the subagent completes.
"""

_CURSOR_SUBAGENTS: dict[str, str] = {
    "mirdan-quality-validator.md": _SUBAGENT_QUALITY_VALIDATOR,
    "mirdan-security-scanner.md": _SUBAGENT_SECURITY_SCANNER,
    "mirdan-plan-reviewer.md": _SUBAGENT_PLAN_REVIEWER,
}


def generate_cursor_subagents(cursor_dir: Path, *, force: bool = False) -> list[Path]:
    """Generate .cursor/agents/*.md files for Cursor subagent definitions.

    Creates markdown files with YAML frontmatter for each mirdan quality
    subagent. These are automatically invoked by Cursor's agent based on
    the description field, or explicitly via /subagent-name.

    Existing files are preserved — this function is idempotent per file.
    If force=True, overwrite existing files with latest mirdan content.

    Args:
        cursor_dir: The .cursor/ directory to write into.
        force: If True, overwrite existing files with latest content.

    Returns:
        List of newly created (or overwritten) subagent file paths.
    """
    agents_dir = cursor_dir / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for filename, content in _CURSOR_SUBAGENTS.items():
        dest = agents_dir / filename
        if force or not dest.exists():
            dest.write_text(content)
            created.append(dest)

    return created


# ---------------------------------------------------------------------------
# Cursor Skills Generation (.cursor/skills/*/SKILL.md)
# ---------------------------------------------------------------------------

_SKILL_CODE = """\
---
name: mirdan-code
description: >-
  Quality-orchestrated coding workflow. Use when implementing features,
  fixing bugs, or writing new code. Automatically validates code quality
  via mirdan MCP tools.
disable-model-invocation: false
---

# mirdan Code — Quality-Orchestrated Coding

Execute coding tasks with automatic quality enforcement via mirdan.

## When to Use

- Implementing new features
- Fixing bugs
- Writing new modules or functions
- Any code creation or modification task

## Workflow

1. Call `mcp__mirdan__enhance_prompt` with the task description to get:
   - Quality requirements for this specific task
   - Security sensitivity detection (`touches_security`)
   - Framework-specific guidance

2. Use `@Docs [library-name]` to look up current API documentation for
   any libraries involved in this task.

3. Follow the `quality_requirements` from enhance_prompt as constraints
   during implementation.

4. After writing code, call `mcp__mirdan__validate_code_quality` on
   changed files. Set `check_security=true` if `touches_security` was
   flagged.

5. Fix all errors before marking complete. Note warnings for review.
"""

_SKILL_PLAN_REVIEW = """\
---
name: mirdan-plan-review
description: >-
  Staff-engineer-grade plan review. Use after creating implementation plans
  in Plan Mode to verify accuracy and cheap-model executability before
  switching to Build.
disable-model-invocation: true
---

# mirdan Plan Review — Staff-Engineer Plan Verification

Review AI-generated implementation plans for factual accuracy and executability
by cheaper models (Haiku, Flash). Judge half of Judge/Planner separation.

## When to Use

Invoke explicitly with `/mirdan-plan-review` when you want to:
- Verify a plan before switching from Plan Mode to Build
- Review a plan created by another agent or session
- Validate that a plan is executable by a cheaper model

## Effort Calibration

- Small plans (1-5 steps): Verify ALL references
- Medium plans (6-15 steps): Verify file paths + line numbers, sample functions
- Large plans (16+ steps): Verify file paths for all, deep-verify 5 highest-risk
- Budget: Maximum 20 tool calls for grounding verification

## Workflow

1. **Read Plan** — Read the plan from the provided path or most recent plan file.

2. **Structural Validate** — Call `mcp__mirdan__enhance_prompt` with the plan
   text and `task_type="plan_validation"`. Record the PlanQualityScore.

3. **Extract References** — Scan the plan for verifiable claims:
   - File paths (backtick-wrapped with `/` and extension)
   - Line numbers (preceded by `line `, `:`, or `L`)
   - Function/class names (backtick `def name()` or `class Name`)
   - Import references (backtick `from X import Y`)
   - Step dependency refs ("Depends On: Step N")
   - Directory refs (paths ending in `/`)

4. **Grounding Verification** — For each reference, verify with tools.
   Apply false-positive prevention first:
   - "Write"/"Create" actions or "NEW:" prefix → EXPECTED_NEW, verify parent dir
   - Step N creates something, Step M>N references it → EXPECTED_NEW
   - Line insertions may shift later line numbers → warning, not error

   Classify: VERIFIED / MISMATCH / NOT_FOUND / EXPECTED_NEW / UNVERIFIABLE

5. **Dependency Analysis** — Check step graph ordering, circular deps,
   creation-before-modification, import-before-usage.

6. **Semantic Review** — Assess:
   - COMPLETENESS: Missing tests, exports, configs?
   - EXECUTABILITY: Can Haiku execute each step with only its context? Every
     `Action: Edit` has a ```anchor```/```replace``` whose anchor is findable exactly
     once in the file; no unresolved decisions (TBD / either-or).
   - SAFETY: Auth/input/DB changes have validation?

7. **Synthesize Report** — Score 7 dimensions, determine verdict.

## Scoring (weights)

| Dimension | Weight | 1.0 means |
|-----------|--------|-----------|
| Grounding | 30% | All refs verified or expected-new |
| Completeness | 20% | No gaps in steps |
| Atomicity | 15% | All steps single-action; every non-capable Edit has a unique anchor/replace |
| Clarity | 10% | Zero vague language |
| Dependency Order | 10% | Valid ordering |
| Executability | 10% | Self-contained steps |
| Safety | 5% | Security addressed |

## Verdict: PASS (>=0.95 haiku) / REVISE (>=0.5) / FAIL (<0.5)

For a **haiku** target, any unresolved decision or `[target: capable]` step → at most REVISE.

Report findings as: `[DIMENSION/CONFIDENCE] Description — Fix: recommendation`
"""

_CURSOR_SKILLS: dict[str, str] = {
    "mirdan-code": _SKILL_CODE,
    "mirdan-plan-review": _SKILL_PLAN_REVIEW,
}


def generate_cursor_skills(cursor_dir: Path, *, force: bool = False) -> list[Path]:
    """Generate .cursor/skills/*/SKILL.md files for Cursor skill definitions.

    Creates skill directories with SKILL.md manifests following the Agent
    Skills Standard (agentskills.io). Each skill provides a structured
    workflow for a mirdan quality task.

    Existing SKILL.md files are preserved — this function is idempotent.
    If force=True, overwrite existing files with latest mirdan content.

    Args:
        cursor_dir: The .cursor/ directory to write into.
        force: If True, overwrite existing files with latest content.

    Returns:
        List of newly created (or overwritten) SKILL.md file paths.
    """
    skills_dir = cursor_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for skill_name, content in _CURSOR_SKILLS.items():
        skill_dir = skills_dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        dest = skill_dir / "SKILL.md"
        if force or not dest.exists():
            dest.write_text(content)
            created.append(dest)

    return created


# ---------------------------------------------------------------------------
# Cursor Environment Config Generation (.cursor/environment.json)
# ---------------------------------------------------------------------------


def generate_cursor_environment(
    cursor_dir: Path,
    detected: DetectedProject,
) -> Path | None:
    """Generate .cursor/environment.json for Cursor Cloud Agent environments.

    Creates a minimal environment configuration ensuring mirdan is available
    in cloud agent environments. Skips if environment.json already exists.

    Args:
        cursor_dir: The .cursor/ directory to write into.
        detected: Detected project metadata.

    Returns:
        Path to the generated file, or None if already exists.
    """
    cursor_dir.mkdir(parents=True, exist_ok=True)
    env_path = cursor_dir / "environment.json"

    if env_path.exists():
        return None

    # Detect install command based on project tooling
    install_cmd = "pip install mirdan"
    if detected.primary_language == "python":
        # Check for uv/poetry indicators
        project_dir = cursor_dir.parent
        if (project_dir / "uv.lock").exists() or (project_dir / ".python-version").exists():
            install_cmd = "uv pip install mirdan"

    config = {
        "name": "mirdan-quality",
        "install": install_cmd,
        "terminals": [
            {
                "name": "mirdan",
                "command": "mirdan --version",
                "description": "Verify mirdan code quality tools are available",
            }
        ],
    }

    with env_path.open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    return env_path


# ---------------------------------------------------------------------------
# Cursor Sandbox Config Generation (.cursor/sandbox.json)
# ---------------------------------------------------------------------------


def generate_cursor_sandbox(
    cursor_dir: Path,
    detected: DetectedProject,
) -> Path | None:
    """Generate .cursor/sandbox.json with secure default access controls.

    Creates a sandbox configuration with deny-default network policy and
    language-specific package registry allowlists. Skips if sandbox.json
    already exists to respect user customizations.

    Args:
        cursor_dir: The .cursor/ directory to write into.
        detected: Detected project metadata for language-specific registries.

    Returns:
        Path to the generated file, or None if already exists.
    """
    cursor_dir.mkdir(parents=True, exist_ok=True)
    sandbox_path = cursor_dir / "sandbox.json"

    if sandbox_path.exists():
        return None

    # Base allow list — common registries and GitHub
    allow_list = [
        "pypi.org",
        "files.pythonhosted.org",
        "registry.npmjs.org",
        "registry.yarnpkg.com",
        "*.docker.io",
        "ghcr.io",
        "github.com",
        "api.github.com",
        "objects.githubusercontent.com",
    ]

    # Language-specific registry additions
    lang = (detected.primary_language or "").lower()
    if lang == "rust":
        allow_list.extend(["crates.io", "static.crates.io"])
    elif lang == "go":
        allow_list.extend(["proxy.golang.org", "sum.golang.org"])

    config = {
        "type": "workspace_readwrite",
        "networkPolicy": {
            "default": "deny",
            "allow": allow_list,
        },
    }

    with sandbox_path.open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    return sandbox_path


def _load_templates() -> dict[str, str]:
    """Load .mdc templates from the package templates directory."""
    templates: dict[str, str] = {}
    try:
        templates_pkg = files("mirdan.integrations.templates")
        for item in templates_pkg.iterdir():
            if item.name.endswith(".mdc"):
                templates[item.name] = item.read_text()
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass
    return templates


# ---------------------------------------------------------------------------
# Cursor MCP Config Generation
# ---------------------------------------------------------------------------


def generate_cursor_mcp_json(cursor_dir: Path) -> Path:
    """Generate .cursor/mcp.json with mirdan MCP server configuration.

    Args:
        cursor_dir: The .cursor/ directory to write into.

    Returns:
        Path to the generated mcp.json file.
    """
    from mirdan.integrations.claude_code import detect_mirdan_command

    cursor_dir.mkdir(parents=True, exist_ok=True)
    mcp_json_path = cursor_dir / "mcp.json"

    command, args = detect_mirdan_command()

    config: dict[str, Any] = {
        "mcpServers": {
            "mirdan": {
                "type": "stdio",
                "command": command,
                "args": args,
                "env": {"MIRDAN_TOOL_BUDGET": "3"},
            },
            "sequential-thinking": {
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
            },
        }
    }

    # If mcp.json already exists, merge rather than overwrite
    if mcp_json_path.exists():
        try:
            existing = json.loads(mcp_json_path.read_text())
            if "mcpServers" in existing:
                existing["mcpServers"]["mirdan"] = config["mcpServers"]["mirdan"]
                # Only add sequential-thinking if not already configured
                if "sequential-thinking" not in existing["mcpServers"]:
                    existing["mcpServers"]["sequential-thinking"] = config["mcpServers"][
                        "sequential-thinking"
                    ]
                config = existing
        except (json.JSONDecodeError, KeyError):
            pass

    with mcp_json_path.open("w") as f:
        json.dump(config, f, indent=2)

    return mcp_json_path


# ---------------------------------------------------------------------------
# Platform Adapter
# ---------------------------------------------------------------------------


class CursorAdapter:
    """Platform adapter for Cursor IDE integration.

    Delegates to existing public functions, providing a unified
    entry point for generating all Cursor configuration files.
    """

    def __init__(
        self,
        project_dir: Path,
        detected: DetectedProject,
        standards: QualityStandards | None = None,
        hook_stringency: CursorHookStringency = CursorHookStringency.COMPREHENSIVE,
        *,
        force_regenerate: bool = False,
    ) -> None:
        self.project_dir = project_dir
        self.detected = detected
        self.standards = standards
        self.hook_stringency = hook_stringency
        self.force_regenerate = force_regenerate

    def generate_hooks(self) -> list[Path]:
        """Generate .cursor/hooks.json."""
        cursor_dir = self.project_dir / ".cursor"
        result = generate_cursor_hooks(cursor_dir, self.hook_stringency)
        return [result] if result else []

    def generate_rules(self) -> list[Path]:
        """Generate .cursor/rules/*.mdc files."""
        rules_dir = self.project_dir / ".cursor" / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        return generate_cursor_rules(rules_dir, self.detected, self.standards)

    def generate_agents(self) -> list[Path]:
        """Generate .cursor/AGENTS.md and .cursor/BUGBOT.md."""
        cursor_dir = self.project_dir / ".cursor"
        return generate_cursor_agents(cursor_dir, self.detected, self.standards)

    def generate_mcp_config(self) -> Path | None:
        """Generate .cursor/mcp.json."""
        cursor_dir = self.project_dir / ".cursor"
        return generate_cursor_mcp_json(cursor_dir)

    def generate_commands(self) -> list[Path]:
        """Generate .cursor/commands/*.md files."""
        cursor_dir = self.project_dir / ".cursor"
        return generate_cursor_commands(cursor_dir, force=self.force_regenerate)

    def generate_subagents(self) -> list[Path]:
        """Generate .cursor/agents/*.md subagent definitions."""
        cursor_dir = self.project_dir / ".cursor"
        return generate_cursor_subagents(cursor_dir, force=self.force_regenerate)

    def generate_skills(self) -> list[Path]:
        """Generate .cursor/skills/*/SKILL.md skill definitions."""
        cursor_dir = self.project_dir / ".cursor"
        return generate_cursor_skills(cursor_dir, force=self.force_regenerate)

    def generate_environment(self) -> Path | None:
        """Generate .cursor/environment.json for cloud agents."""
        cursor_dir = self.project_dir / ".cursor"
        return generate_cursor_environment(cursor_dir, self.detected)

    def generate_sandbox(self) -> Path | None:
        """Generate .cursor/sandbox.json with secure defaults."""
        cursor_dir = self.project_dir / ".cursor"
        return generate_cursor_sandbox(cursor_dir, self.detected)

    def generate_all(self) -> list[Path]:
        """Call all generators, return all created paths."""
        paths: list[Path] = []
        paths.extend(self.generate_hooks())
        paths.extend(self.generate_rules())
        paths.extend(self.generate_agents())
        paths.extend(self.generate_commands())
        paths.extend(self.generate_subagents())
        paths.extend(self.generate_skills())
        env = self.generate_environment()
        if env:
            paths.append(env)
        sandbox = self.generate_sandbox()
        if sandbox:
            paths.append(sandbox)
        mcp = self.generate_mcp_config()
        if mcp:
            paths.append(mcp)
        return paths
