"""Generate Cursor IDE configuration files for mirdan integration.

Supports .cursor/rules/*.mdc, .cursor/AGENTS.md, .cursor/BUGBOT.md,
.cursor/hooks.json, and .cursor/mcp.json generation.
"""

from __future__ import annotations

import json
from enum import Enum
from importlib.resources import files
from pathlib import Path

from mirdan.cli.detect import DetectedProject
from mirdan.core.quality_standards import QualityStandards

# ---------------------------------------------------------------------------
# Cursor Hook Stringency
# ---------------------------------------------------------------------------


class CursorHookStringency(Enum):
    """Hook stringency levels for Cursor hooks.json generation."""

    MINIMAL = "minimal"  # 2 events: afterFileEdit, stop
    STANDARD = "standard"  # 5 events: + preToolUse, postToolUse, sessionStart
    COMPREHENSIVE = "comprehensive"  # All events (~16)


# Events for each stringency level
CURSOR_STRINGENCY_EVENTS: dict[CursorHookStringency, list[str]] = {
    CursorHookStringency.MINIMAL: ["afterFileEdit", "stop"],
    CursorHookStringency.STANDARD: [
        "afterFileEdit",
        "preToolUse",
        "postToolUse",
        "sessionStart",
        "stop",
    ],
    CursorHookStringency.COMPREHENSIVE: [
        "afterFileEdit",
        "preToolUse",
        "postToolUse",
        "postToolUseFailure",
        "stop",
        "sessionStart",
        "sessionEnd",
        "beforeSubmitPrompt",
        "subagentStart",
        "subagentStop",
        "beforeShellExecution",
        "afterShellExecution",
        "beforeMCPExecution",
        "afterMCPExecution",
        "preCompact",
        "afterAgentResponse",
    ],
}


# ---------------------------------------------------------------------------
# Cursor Hooks Generation
# ---------------------------------------------------------------------------


def generate_cursor_hooks(
    cursor_dir: Path,
    stringency: CursorHookStringency = CursorHookStringency.COMPREHENSIVE,
) -> Path | None:
    """Generate .cursor/hooks.json with prompt-type hooks.

    Produces Cursor 1.7+ hooks.json with prompt hooks that invoke
    mirdan MCP tools for quality enforcement.

    Args:
        cursor_dir: The .cursor/ directory to write into.
        stringency: Hook stringency level controlling event count.

    Returns:
        Path to created hooks.json, or None if file already exists.
    """
    cursor_dir.mkdir(parents=True, exist_ok=True)
    hooks_path = cursor_dir / "hooks.json"

    # Respect user customizations — skip if already exists
    if hooks_path.exists():
        return None

    events = CURSOR_STRINGENCY_EVENTS[stringency]
    hooks: dict[str, list[dict]] = {}

    for event in events:
        generator = _CURSOR_HOOK_GENERATORS.get(event)
        if generator:
            hooks[event] = generator()

    config = {"version": 1, "hooks": hooks}

    with hooks_path.open("w") as f:
        json.dump(config, f, indent=2)

    return hooks_path


def _hook_after_file_edit() -> list[dict]:
    """afterFileEdit: Validate changed code quality."""
    return [
        {
            "type": "prompt",
            "prompt": (
                "The file {file_path} was just edited. Call"
                " mcp__mirdan__validate_code_quality on the changed code to"
                " check for quality violations. Fix any errors found."
            ),
        }
    ]


def _hook_pre_tool_use() -> list[dict]:
    """preToolUse: Security-aware reminder before Write/Edit."""
    return [
        {
            "type": "prompt",
            "matcher": "Write|Edit",
            "prompt": (
                "Before writing/editing files, consider: security"
                " implications (SQL injection, command injection, hardcoded"
                " secrets), quality requirements from the current task"
                " context. If this edit touches auth/security code, be"
                " especially careful."
            ),
        }
    ]


def _hook_stop() -> list[dict]:
    """stop: Verification gate before task completion."""
    return [
        {
            "type": "prompt",
            "loop_limit": 3,
            "prompt": (
                "Before completing this task, verify:"
                " 1) All changed files were validated with"
                " mcp__mirdan__validate_code_quality."
                " 2) No unresolved errors remain."
                " 3) Security-sensitive code was reviewed."
                " If any files weren't validated, call"
                " validate_code_quality now."
            ),
        }
    ]


def _hook_before_submit_prompt() -> list[dict]:
    """beforeSubmitPrompt: Inject quality context."""
    return [
        {
            "type": "prompt",
            "prompt": (
                "Before processing this request, consider calling"
                " mcp__mirdan__enhance_prompt to understand quality"
                " requirements and get framework-specific guidance."
            ),
        }
    ]


def _hook_post_tool_use() -> list[dict]:
    """postToolUse: Validate after tool execution."""
    return [
        {
            "type": "prompt",
            "prompt": (
                "A tool was just used. If code was written or edited,"
                " call mcp__mirdan__validate_code_quality on the changed"
                " code. Fix any errors immediately."
            ),
        }
    ]


def _hook_post_tool_use_failure() -> list[dict]:
    """postToolUseFailure: Handle failed tool calls."""
    return [
        {
            "type": "prompt",
            "prompt": (
                "A tool call just failed. Analyze the error before"
                " retrying. Consider an alternative approach rather"
                " than repeating the same call."
            ),
        }
    ]


def _hook_session_start() -> list[dict]:
    """sessionStart: Initialize quality context for new session."""
    return [
        {
            "type": "prompt",
            "prompt": (
                "mirdan quality context is active. For coding tasks:"
                " 1) Call mcp__mirdan__enhance_prompt for quality requirements."
                " 2) After writing code, call mcp__mirdan__validate_code_quality."
                " 3) Fix all errors before marking complete."
            ),
        }
    ]


def _hook_session_end() -> list[dict]:
    """sessionEnd: Persist quality summary at session end."""
    return [
        {
            "type": "prompt",
            "prompt": (
                "Session ending. Verify all changed files were validated"
                " with mcp__mirdan__validate_code_quality and no"
                " unresolved errors remain."
            ),
        }
    ]


def _hook_subagent_start() -> list[dict]:
    """subagentStart: Pass quality context to subagents."""
    return [
        {
            "type": "prompt",
            "prompt": (
                "You are a subagent. mirdan quality standards are active."
                " Validate any code with mcp__mirdan__validate_code_quality"
                " before returning results."
            ),
        }
    ]


def _hook_subagent_stop() -> list[dict]:
    """subagentStop: Review subagent output quality."""
    return [
        {
            "type": "prompt",
            "prompt": (
                "Review the subagent's output. If code was written,"
                " ensure it was validated with"
                " mcp__mirdan__validate_code_quality."
            ),
        }
    ]


def _hook_before_shell_execution() -> list[dict]:
    """beforeShellExecution: Security check before shell commands."""
    return [
        {
            "type": "prompt",
            "prompt": (
                "A shell command is about to execute. Check for:"
                " 1) Command injection risks (unsanitized input)."
                " 2) Destructive operations (rm -rf, git reset --hard)."
                " 3) Overly broad permissions."
            ),
        }
    ]


def _hook_after_shell_execution() -> list[dict]:
    """afterShellExecution: Review shell command results."""
    return [
        {
            "type": "prompt",
            "prompt": (
                "A shell command completed. Review the output for"
                " errors or warnings that might affect code quality."
            ),
        }
    ]


def _hook_before_mcp_execution() -> list[dict]:
    """beforeMCPExecution: Context before MCP tool calls."""
    return [
        {
            "type": "prompt",
            "prompt": (
                "An MCP tool is about to be called. If calling"
                " mirdan tools, ensure correct parameters are set"
                " (check_security, language, session_id)."
            ),
        }
    ]


def _hook_after_mcp_execution() -> list[dict]:
    """afterMCPExecution: Review MCP tool results."""
    return [
        {
            "type": "prompt",
            "prompt": (
                "An MCP tool call completed. If mirdan validation"
                " returned errors, fix them before continuing."
            ),
        }
    ]


def _hook_pre_compact() -> list[dict]:
    """preCompact: Preserve quality state before compaction."""
    return [
        {
            "type": "prompt",
            "prompt": (
                "Context is being compacted. Preserve mirdan quality"
                " state: session_id, task_type, language, security"
                " sensitivity, last validation score, and open"
                " violations. Restore by calling enhance_prompt"
                " after compaction."
            ),
        }
    ]


def _hook_after_agent_response() -> list[dict]:
    """afterAgentResponse: Quality check on agent responses."""
    return [
        {
            "type": "prompt",
            "prompt": (
                "Agent response generated. If code was included,"
                " ensure it was validated with"
                " mcp__mirdan__validate_code_quality before presenting."
            ),
        }
    ]


_CURSOR_HOOK_GENERATORS: dict[str, object] = {
    "afterFileEdit": _hook_after_file_edit,
    "preToolUse": _hook_pre_tool_use,
    "postToolUse": _hook_post_tool_use,
    "postToolUseFailure": _hook_post_tool_use_failure,
    "stop": _hook_stop,
    "sessionStart": _hook_session_start,
    "sessionEnd": _hook_session_end,
    "beforeSubmitPrompt": _hook_before_submit_prompt,
    "subagentStart": _hook_subagent_start,
    "subagentStop": _hook_subagent_stop,
    "beforeShellExecution": _hook_before_shell_execution,
    "afterShellExecution": _hook_after_shell_execution,
    "beforeMCPExecution": _hook_before_mcp_execution,
    "afterMCPExecution": _hook_after_mcp_execution,
    "preCompact": _hook_pre_compact,
    "afterAgentResponse": _hook_after_agent_response,
}


# ---------------------------------------------------------------------------
# Rule Generation (existing)
# ---------------------------------------------------------------------------


def generate_cursor_rules(
    rules_dir: Path,
    detected: DetectedProject,
    standards: QualityStandards | None = None,
) -> list[Path]:
    """Generate .cursor/rules/*.mdc files with dynamic standards content.

    If a QualityStandards instance is provided, generates dynamic .mdc files
    from actual quality rules. Otherwise falls back to static templates.

    Args:
        rules_dir: The .cursor/rules/ directory to write into.
        detected: Detected project metadata.
        standards: Optional QualityStandards for dynamic generation.

    Returns:
        List of generated file paths.
    """
    generated: list[Path] = []

    if standards:
        generated.extend(_generate_dynamic_rules(rules_dir, detected, standards))
    else:
        generated.extend(_generate_static_rules(rules_dir, detected))

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
) -> list[Path]:
    """Generate .mdc files from QualityStandards data."""
    generated: list[Path] = []

    # Always-on rule
    always_content = _build_always_mdc(standards)
    path = rules_dir / "mirdan-always.mdc"
    path.write_text(always_content)
    generated.append(path)

    # Language-specific rules
    lang = detected.primary_language
    if lang == "python":
        content = _build_language_mdc("python", "**/*.py", standards)
        path = rules_dir / "mirdan-python.mdc"
        path.write_text(content)
        generated.append(path)
    elif lang in ("typescript", "javascript"):
        content = _build_language_mdc("typescript", "**/*.{ts,tsx,js,jsx}", standards)
        path = rules_dir / "mirdan-typescript.mdc"
        path.write_text(content)
        generated.append(path)
    elif lang == "go":
        content = _build_language_mdc("go", "**/*.go", standards)
        path = rules_dir / "mirdan-go.mdc"
        path.write_text(content)
        generated.append(path)
    elif lang == "rust":
        content = _build_language_mdc("rust", "**/*.rs", standards)
        path = rules_dir / "mirdan-rust.mdc"
        path.write_text(content)
        generated.append(path)

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
{rules_text if rules_text else f'''
## Code Quality
- Follow {language} best practices and idioms
- Use type annotations where supported
- Handle errors explicitly
- Keep functions focused and small
'''}
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
    """Build the planning .mdc rule."""
    return """---
description: "mirdan planning standards"
globs: "**/*.md"
---

# mirdan Planning Standards

When creating implementation plans:
- Every step must reference verified files (Read them first)
- Include exact line numbers and function names
- Specify verification method for each step
- No vague language ("should", "probably", "maybe")
- Each step must be atomic (single action)
"""


# ---------------------------------------------------------------------------
# Static generation (fallback: copies templates)
# ---------------------------------------------------------------------------


def _generate_static_rules(rules_dir: Path, detected: DetectedProject) -> list[Path]:
    """Generate .mdc files from static templates (legacy fallback)."""
    generated: list[Path] = []
    templates = _load_templates()

    if "mirdan-always.mdc" in templates:
        path = rules_dir / "mirdan-always.mdc"
        path.write_text(templates["mirdan-always.mdc"])
        generated.append(path)

    lang = detected.primary_language
    if lang == "python" and "mirdan-python.mdc" in templates:
        path = rules_dir / "mirdan-python.mdc"
        path.write_text(templates["mirdan-python.mdc"])
        generated.append(path)

    if lang in ("typescript", "javascript") and "mirdan-typescript.mdc" in templates:
        path = rules_dir / "mirdan-typescript.mdc"
        path.write_text(templates["mirdan-typescript.mdc"])
        generated.append(path)

    if "mirdan-security.mdc" in templates:
        path = rules_dir / "mirdan-security.mdc"
        path.write_text(templates["mirdan-security.mdc"])
        generated.append(path)

    if "mirdan-planning.mdc" in templates:
        path = rules_dir / "mirdan-planning.mdc"
        path.write_text(templates["mirdan-planning.mdc"])
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
## Mandatory Quality Checkpoints

### Before Writing Code
- Call `mcp__mirdan__enhance_prompt` to get quality requirements for the task
- Review the `quality_requirements` and `touches_security` fields in the response

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
"""


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

    config: dict = {
        "mcpServers": {
            "mirdan": {
                "type": "stdio",
                "command": command,
                "args": args,
                "env": {"MIRDAN_TOOL_BUDGET": "3"},
            }
        }
    }

    # If mcp.json already exists, merge rather than overwrite
    if mcp_json_path.exists():
        try:
            existing = json.loads(mcp_json_path.read_text())
            if "mcpServers" in existing:
                existing["mcpServers"]["mirdan"] = config["mcpServers"]["mirdan"]
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
    ) -> None:
        self.project_dir = project_dir
        self.detected = detected
        self.standards = standards
        self.hook_stringency = hook_stringency

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

    def generate_all(self) -> list[Path]:
        """Call all generators, return all created paths."""
        paths: list[Path] = []
        paths.extend(self.generate_hooks())
        paths.extend(self.generate_rules())
        paths.extend(self.generate_agents())
        mcp = self.generate_mcp_config()
        if mcp:
            paths.append(mcp)
        return paths
