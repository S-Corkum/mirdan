"""Generate Cursor IDE configuration files for mirdan integration.

Supports .cursor/rules/*.mdc, .cursor/AGENTS.md, .cursor/BUGBOT.md,
.cursor/hooks.json, and .cursor/mcp.json generation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
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
    """Hook stringency levels for Cursor hooks.json generation."""

    MINIMAL = "minimal"  # 2 events: afterFileEdit, stop
    STANDARD = "standard"  # 4 events: + postToolUse, sessionStart
    COMPREHENSIVE = "comprehensive"  # 7 events: all enforcement hooks


# Events for each stringency level
CURSOR_STRINGENCY_EVENTS: dict[CursorHookStringency, list[str]] = {
    CursorHookStringency.MINIMAL: ["afterFileEdit", "stop"],
    CursorHookStringency.STANDARD: [
        "afterFileEdit",
        "postToolUse",
        "sessionStart",
        "stop",
    ],
    CursorHookStringency.COMPREHENSIVE: [
        "afterFileEdit",
        "postToolUseFailure",
        "stop",
        "sessionStart",
        "beforeShellExecution",
        "subagentStart",
        "preCompact",
    ],
}


# ---------------------------------------------------------------------------
# Cursor Hooks Generation
# ---------------------------------------------------------------------------


def generate_cursor_hooks(
    cursor_dir: Path,
    stringency: CursorHookStringency = CursorHookStringency.COMPREHENSIVE,
) -> Path | None:
    """Generate .cursor/hooks.json with prompt-type and command-type hooks.

    Produces Cursor 1.7+ hooks.json with:
    - Prompt hooks for context-dependent quality enforcement (LLM-evaluated)
    - Command hooks for fast, deterministic checks (shell scripts)

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

    # Generate command-type hook scripts first
    generate_cursor_hook_scripts(cursor_dir)

    events = CURSOR_STRINGENCY_EVENTS[stringency]
    hooks: dict[str, list[dict[str, str | int]]] = {}

    for event in events:
        generator = _CURSOR_HOOK_GENERATORS.get(event)
        if generator:
            hooks[event] = generator()

    # Append command-type hooks for events that benefit from fast checks
    _append_command_hooks(hooks, events, cursor_dir)

    config = {"version": 1, "hooks": hooks}

    with hooks_path.open("w") as f:
        json.dump(config, f, indent=2)

    return hooks_path


def _append_command_hooks(
    hooks: dict[str, list[dict[str, str | int]]],
    events: list[str],
    cursor_dir: Path,
) -> None:
    """Append command-type hooks to events that benefit from fast checks.

    Adds deterministic shell-script hooks alongside existing prompt hooks.
    Command hooks fire first (faster), providing a first line of defense.

    Args:
        hooks: Existing hooks dict to mutate.
        events: Active events for this stringency level.
        cursor_dir: The .cursor/ directory (for script paths).
    """
    shell_guard = cursor_dir / "hooks" / "mirdan-shell-guard.sh"
    stop_gate = cursor_dir / "hooks" / "mirdan-stop-gate.sh"

    if "beforeShellExecution" in events and shell_guard.exists():
        hooks.setdefault("beforeShellExecution", []).insert(
            0,
            {
                "type": "command",
                "command": ".cursor/hooks/mirdan-shell-guard.sh",
                "timeout": 5,
                "failClosed": True,
            },
        )

    if "stop" in events and stop_gate.exists():
        hooks.setdefault("stop", []).insert(
            0,
            {
                "type": "command",
                "command": ".cursor/hooks/mirdan-stop-gate.sh",
                "timeout": 15,
                "loop_limit": 3,
            },
        )


def _hook_after_file_edit() -> list[dict[str, str | int]]:
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


def _hook_stop() -> list[dict[str, str | int]]:
    """stop: Verification gate before task completion."""
    return [
        {
            "type": "prompt",
            "loop_limit": 2,
            "prompt": (
                "If code was written or edited during this task,"
                " verify that mcp__mirdan__validate_code_quality was"
                " called on all changed files and no unresolved errors"
                " remain. If validation was not run on changed code,"
                " call it now. If no code was changed, this check"
                " passes automatically."
            ),
        }
    ]


def _hook_before_submit_prompt() -> list[dict[str, str | int]]:
    """beforeSubmitPrompt: Lightweight quality reminder (no blocking MCP calls).

    IMPORTANT: This hook's prompt must NOT instruct the evaluator to verify
    file existence or check attachments — the fast eval model cannot access
    the filesystem and will fail with false positives when attachments are
    present in the hook input data.
    """
    return [
        {
            "type": "prompt",
            "prompt": (
                "mirdan quality standards are active. Use @Docs for"
                " external APIs. Quality validation runs automatically"
                " after file edits. Always allow the prompt to proceed."
            ),
        }
    ]


def _hook_post_tool_use() -> list[dict[str, str | int]]:
    """postToolUse: Review non-edit tool results.

    Note: file-edit validation is handled by afterFileEdit to avoid
    duplicate mirdan calls.  This hook covers shell commands, MCP
    calls, and other non-edit tool outcomes.
    """
    return [
        {
            "type": "prompt",
            "prompt": (
                "A tool just completed. If the output indicates errors"
                " or warnings, review them before continuing. File-edit"
                " quality checks are handled separately."
            ),
        }
    ]


def _hook_post_tool_use_failure() -> list[dict[str, str | int]]:
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


def _hook_session_start() -> list[dict[str, str | int]]:
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


def _hook_session_end() -> list[dict[str, str | int]]:
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


def _hook_subagent_start() -> list[dict[str, str | int]]:
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


def _hook_subagent_stop() -> list[dict[str, str | int]]:
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


def _hook_before_shell_execution() -> list[dict[str, str | int]]:
    """beforeShellExecution: Security check before shell commands.

    NOTE: The deterministic command hook (mirdan-shell-guard.sh) handles
    blocking truly dangerous patterns (rm -rf /, DROP TABLE, force-push
    to main/master, git reset --hard). This prompt hook is kept minimal
    to avoid the fast eval model being overly cautious about safe
    operations like ``git commit``, ``git push origin <branch>``, etc.
    """
    return [
        {
            "type": "prompt",
            "prompt": (
                "A shell command is about to execute."
                " Allow all standard development commands (git, npm, pip,"
                " make, cargo, uv, pytest, etc.) without blocking."
                " Only flag commands that contain hardcoded secrets or"
                " credentials visible in the command string."
            ),
        }
    ]


def _hook_after_shell_execution() -> list[dict[str, str | int]]:
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


def _hook_before_mcp_execution() -> list[dict[str, str | int]]:
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


def _hook_after_mcp_execution() -> list[dict[str, str | int]]:
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


def _hook_pre_compact() -> list[dict[str, str | int]]:
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


def _hook_after_agent_response() -> list[dict[str, str | int]]:
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

_STOP_GATE_SCRIPT = """\
#!/usr/bin/env bash
# mirdan stop gate — remind about quality validation at task completion.
# Reads JSON from stdin (Cursor hooks protocol). If there are uncommitted
# changes, outputs a followup_message suggesting /mirdan-gate.
#
# Always exits 0 (advisory, never blocks).
set -euo pipefail

# Check for uncommitted changes
CHANGED=$(git diff --name-only 2>/dev/null | head -20 || echo "")
STAGED=$(git diff --cached --name-only 2>/dev/null | head -20 || echo "")
ALL_CHANGED=$(echo -e "${CHANGED}\\n${STAGED}" | sort -u | sed '/^$/d')

if [ -z "$ALL_CHANGED" ]; then
    exit 0
fi

COUNT=$(echo "$ALL_CHANGED" | wc -l | tr -d ' ')
MSG="Quality gate: ${COUNT} file(s) changed. Run /mirdan-gate."
echo "{\\"followup_message\\":\\"$MSG\\"}"
exit 0
"""


def generate_cursor_hook_scripts(cursor_dir: Path) -> list[Path]:
    """Generate .cursor/hooks/*.sh command-type hook scripts.

    Creates executable shell scripts for fast, deterministic hook checks.
    These complement the prompt-type hooks with zero LLM overhead.

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
        "mirdan-stop-gate.sh": _STOP_GATE_SCRIPT,
    }

    created: list[Path] = []
    for filename, content in scripts.items():
        dest = hooks_dir / filename
        if not dest.exists():
            dest.write_text(content)
            dest.chmod(0o755)
            created.append(dest)

    return created


_CURSOR_HOOK_GENERATORS: dict[str, Callable[[], list[dict[str, str | int]]]] = {
    "afterFileEdit": _hook_after_file_edit,
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

    # Debug rule (always)
    debug_content = _build_debug_mdc()
    path = rules_dir / "mirdan-debug.mdc"
    path.write_text(debug_content)
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


def _build_debug_mdc() -> str:
    """Build the debug .mdc rule — loads from static template."""
    templates = _load_templates()
    try:
        return templates["mirdan-debug.mdc"]
    except KeyError:
        return "# mirdan Debug Mode Standards\n\nTemplate not found.\n"


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

    if "mirdan-debug.mdc" in templates:
        path = rules_dir / "mirdan-debug.mdc"
        path.write_text(templates["mirdan-debug.mdc"])
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

1. Call `mcp__mirdan__enhance_prompt` with the task to get quality requirements,
   detected language, and security sensitivity.

2. Call `mcp__enyal__enyal_recall` with the task description to load project
   conventions, past decisions, and relevant patterns before writing any code.

3. Use `mcp__sequential-thinking__sequentialthinking` to plan the implementation
   approach: break down the task, identify components, dependencies, and edge
   cases before writing code.

4. Use `@Docs [library-name]` to look up current API documentation for any
   libraries involved in this task.

5. Follow the `quality_requirements` from enhance_prompt and conventions from
   enyal_recall as constraints during implementation.

6. After writing code, call `mcp__mirdan__validate_code_quality` on changed files.
   Set `check_security=true` if `touches_security` was flagged.

7. Call `mcp__enyal__enyal_remember` to store any new decisions, patterns, or
   conventions discovered during implementation.

8. Fix all errors before marking complete. Note warnings for review.
"""

_COMMAND_DEBUG = """\
# /debug — Quality-Aware Debugging

Debug issues with mirdan quality analysis to prevent introducing new problems.

## Workflow

1. Call `mcp__mirdan__enhance_prompt` with the bug description to get context.

2. Call `mcp__enyal__enyal_recall` with the bug description to check if a similar
   issue was previously solved.

3. Use `mcp__sequential-thinking__sequentialthinking` to form structured
   hypotheses about root cause, then plan systematic verification for each.

4. Use `@Docs [library-name]` to verify correct API behavior — many bugs are
   incorrect assumptions about library APIs.

5. Read relevant code and trace the actual error path to verify hypotheses.

6. Apply the minimal fix targeting the root cause, not just symptoms.

7. Call `mcp__mirdan__validate_code_quality` on modified files to confirm the fix
   does not introduce new violations.

8. Call `mcp__enyal__enyal_remember` to store the fix pattern for future reference.
"""

_COMMAND_REVIEW = """\
# /review — Code Review with Quality Standards

Review code against mirdan quality standards.

## Workflow

1. Call `mcp__mirdan__get_quality_standards` for the language/framework to establish
   review criteria.

2. Call `mcp__enyal__enyal_recall` with the file path to get project code conventions.
   Use the `file_path` parameter for scope-weighted results.

3. Read each changed file. Check for:
   - AI quality rules: AI001 (placeholders), AI002 (hallucinated imports),
     AI007 (security theater), AI008 (injection vulnerabilities)
   - Security rules: SEC001-SEC014
   - Architecture rules: ARCH001-ARCH005
   - Deviations from project conventions loaded from enyal

4. Call `mcp__mirdan__validate_code_quality` with `check_security=true` on
   security-sensitive files.

5. Report findings grouped by severity: errors (must fix), warnings (should fix).
"""

_COMMAND_PLAN = """\
# /plan — Enhanced Implementation Planning

Create plans enhanced with structured analysis and grounded facts.

## Workflow

1. **Context** — Call `mcp__enyal__enyal_recall("architecture conventions decisions")`
   to load project context. Then call `mcp__enyal__enyal_traverse` with the planned
   area to discover related decisions and dependencies.

2. **Analyze** — Use `mcp__sequential-thinking__sequentialthinking` to think
   through the task deeply before generating steps. Cover scope, phases,
   dependencies, edge cases, and completeness. Start with `totalThoughts: 8`,
   adjust as needed.

3. **Ground** — Read all files that will be modified — verify they exist, note
   current structure and relevant line numbers. Use `@Docs [library-name]`
   for every external API involved — never plan around assumed APIs.

4. **Plan** — Output a clear task list using Cursor's native plan format. Use
   file paths in backticks for clickable links. Each step: one atomic action.

   For tasks with independent work streams, group TODOs into parallel streams:

   ```
   ## Stream A: Data Layer [mirdan-implementer]
   - [ ] Add model class in `src/models.py`
   - [ ] Add migration in `src/migrations/`

   ## Stream B: API Layer [mirdan-implementer]
   (Depends on: Stream A)
   - [ ] Add endpoint in `src/api.py`
   - [ ] Add input validation

   ## Stream C: Tests [mirdan-test-writer]
   (Depends on: Stream A, Stream B)
   - [ ] Add model tests
   - [ ] Add API endpoint tests
   ```

5. **Constrain** — No vague language ("should", "probably", "maybe"). Verified
   paths only. Include tests, imports, and config changes. Architecture decisions
   from enyal must be respected.

## Multi-Agent Execution

When the plan has parallel streams, send each stream's TODOs to a separate agent:
1. Select Stream A TODOs → send to agent (uses `mirdan-implementer`)
2. Select Stream B TODOs → send to agent (uses `mirdan-implementer`)
3. Select Stream C TODOs → send to agent (uses `mirdan-test-writer`)

Independent streams run in parallel. Dependent streams wait for their
dependencies to complete. Each agent spawns readonly validators as subagents.

## During Build Execution

Execute each step directly. Skip enhance_prompt and sequential-thinking.
Quality validation runs automatically after each file edit.
"""

_COMMAND_QUALITY = """\
# /quality — On-Demand Quality Validation

Run mirdan quality validation on demand.

## Workflow

1. Identify files to validate — changed files, staged files, or specific path.

2. Call `mcp__enyal__enyal_recall("quality conventions")` to load project quality
   standards for comparison.

3. Call `mcp__mirdan__validate_code_quality` on each file:
   - Use `check_security=true` for auth, input handling, database, or API code
   - Use `severity_threshold="info"` for comprehensive results

4. Call `mcp__mirdan__get_quality_trends` to check session-wide quality trends.

5. Report findings:
   - Errors (must fix before completing)
   - Warnings (should fix, note if deferring)
   - Overall quality score
"""

_COMMAND_SCAN = """\
# /scan — Convention Scanner

Scan the codebase for quality violations and convention patterns.

## Workflow

1. Call `mcp__enyal__enyal_recall("conventions patterns")` to load known project
   conventions for comparison.

2. Identify files to scan — all source files, or scope by argument (path or language).

3. Call `mcp__mirdan__scan_conventions` to discover patterns and violations.

4. Call `mcp__mirdan__get_quality_standards` for the project language to verify
   findings against defined rules.

5. Report:
   - Violations by rule ID and count
   - New patterns discovered
   - Conventions that diverge from standards or stored knowledge

6. If new high-confidence conventions are discovered, call `mcp__enyal__enyal_remember`
   to persist them for future sessions.
"""

_COMMAND_GATE = """\
# /gate — Quality Gate

Run the full quality gate before committing or completing a task.

## Workflow

1. Find all changed files: uncommitted changes and staged files.

2. Call `mcp__mirdan__validate_code_quality` on each changed file.
   Use `check_security=true` for any file touching auth, input, SQL, or APIs.

3. Check that every file passes with quality score >= 0.7.
   Security-critical files require score >= 0.8.

4. Fix all errors. For warnings, note rule IDs and justification if deferring.

5. Re-validate after fixes to confirm PASS status.

6. If validation produced `knowledge_entries`, call `mcp__enyal__enyal_remember`
   to store them with the suggested tags and scope.

7. Only mark the task complete after all files pass the quality gate.
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
    "debug.md": _COMMAND_DEBUG,
    "review.md": _COMMAND_REVIEW,
    "plan.md": _COMMAND_PLAN,
    "quality.md": _COMMAND_QUALITY,
    "scan.md": _COMMAND_SCAN,
    "gate.md": _COMMAND_GATE,
    "automations.md": _COMMAND_AUTOMATIONS,
}


def generate_cursor_commands(cursor_dir: Path, *, force: bool = False) -> list[Path]:
    """Generate .cursor/commands/*.md files for Cursor slash commands.

    Creates plain Markdown command files (no frontmatter) for each mirdan
    workflow. These are injected as prompt context when the user types
    /code, /debug, /review, /plan, /quality, /scan, or /gate in Cursor.

    Existing files are preserved — this function is idempotent per file.
    If force=True, overwrite existing files with latest mirdan content.

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

    return created


# ---------------------------------------------------------------------------
# Cursor Subagents Generation (.cursor/agents/*.md)
# ---------------------------------------------------------------------------

_SUBAGENT_QUALITY_VALIDATOR = """\
---
name: mirdan-quality-validator
description: >-
  Validate code quality against mirdan standards.
  Use after writing or editing code files to check for AI slop,
  security issues, and style violations.
model: fast
readonly: true
background: true
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

5. Report results as a markdown summary:
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
background: false
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

_SUBAGENT_TEST_AUDITOR = """\
---
name: mirdan-test-auditor
description: >-
  Audit test files for meaningful coverage and correctness.
  Use when reviewing existing test files, not for creating new ones.
model: inherit
readonly: true
background: false
---

# mirdan Test Auditor

Audit test quality for meaningful coverage and correctness.

## Focus Areas

- Meaningful assertions (not just "no error" or `assert True`)
- Test isolation (no interdependencies or shared mutable state)
- Edge cases (boundary conditions, error paths)
- Fixture quality (minimal, focused)
- Naming (test names describe behavior)

## Instructions

1. Use file search to find test files (`**/test_*.py`, `**/*.test.ts`,
   `**/*.spec.ts`).

2. Read each test file.

3. Call `mcp__mirdan__validate_code_quality` for each file with:
   - `severity_threshold="info"`

4. Additionally check for:
   - Tests with no assertions
   - Tests that only assert `True` or check no exception
   - Tests coupled to implementation details
   - Missing error path coverage
   - External service dependencies without mocking
   - Overly broad exception handling in tests

5. Report: test files audited, quality issues with file and test name,
   missing coverage areas, and overall test quality score.

## Subagent Coordination

This subagent runs in the foreground. For large test suites, it may spawn
child subagents to parallelize auditing across test directories.
"""

_SUBAGENT_SLOP_DETECTOR = """\
---
name: mirdan-slop-detector
description: >-
  Detect AI-generated code quality issues: placeholder code, hallucinated
  imports, invented APIs, dead code, and copy-paste artifacts.
  Use proactively on any AI-generated or recently edited code.
model: fast
readonly: true
background: true
---

# mirdan AI Slop Detector

Detect AI-generated code quality issues proactively.

## AI Quality Rules

| Rule | Severity | Description |
|------|----------|-------------|
| AI001 | error | Placeholder code — `NotImplementedError`, bare `pass`, `TODO` |
| AI002 | error | Hallucinated imports — imports that don't resolve |
| AI003 | warning | Invented APIs — function calls with wrong signatures |
| AI004 | warning | Dead code — unused functions, variables, imports |
| AI005 | info | Copy-paste artifacts — duplicate code blocks |
| AI006 | warning | Inconsistent naming — doesn't match codebase conventions |
| AI007 | error | Unvalidated input — missing validation at boundaries |
| AI008 | error | String injection — f-strings in SQL/eval/exec/shell |

## Instructions

1. Identify recently modified code files.

2. Read each file.

3. Call `mcp__mirdan__validate_code_quality` for each file with:
   - `check_security=true`
   - `severity_threshold="info"`

4. Additionally check for:
   - Excessive try/except blocks swallowing errors
   - Unnecessary abstractions for one-time operations
   - Over-engineered solutions beyond what was requested

5. Report: files checked, AI-specific issues by rule ID with line numbers,
   errors (must fix), warnings (should fix), and observations.

## Async Execution Notes

This subagent runs in the background (async). The parent agent continues
working while validation runs. Results are returned via agent ID — the
parent can resume this subagent to retrieve findings.
"""

_SUBAGENT_ARCHITECTURE_REVIEWER = """\
---
name: mirdan-architecture-reviewer
description: >-
  Review code architecture for structural quality issues including
  function length, file length, nesting depth, god classes, and SOLID
  violations. Use to review architecture after refactors or new modules.
model: inherit
readonly: true
background: false
---

# mirdan Architecture Reviewer

Review code architecture for structural quality issues.

## Focus Areas

- Function length exceeding 30 lines (ARCH001)
- File length exceeding 300 non-empty lines (ARCH002)
- Nesting depth deeper than 4 levels (ARCH004)
- God classes with more than 10 methods (ARCH005)
- SOLID violations: single responsibility, interface segregation
- Cross-file patterns: consistent error handling, naming, imports

## Instructions

1. Identify recently changed or created files.

2. Read each file completely.

3. Call `mcp__mirdan__validate_code_quality` for each file with:
   - `check_architecture=true`
   - `severity_threshold="warning"`

4. Analyze cross-file patterns:
   - Are naming conventions consistent across files?
   - Is error handling consistent?
   - Are import patterns consistent?

5. Report: files analyzed, structural issues with line numbers,
   architecture violations, cross-file observations, and
   refactoring recommendations.

## Subagent Coordination

This subagent runs in the foreground. For multi-module projects, it may
spawn child subagents to review each module's architecture in parallel.
"""

_SUBAGENT_IMPLEMENTER = """\
---
name: mirdan-implementer
description: >-
  Execute implementation TODO groups from plan streams. Use when the user
  sends a Stream of TODOs from a mirdan parallel plan for code implementation.
model: inherit
readonly: false
background: false
---

# mirdan Implementer

Execute TODO groups from plan streams — write code following mirdan quality standards.

## Instructions

1. Call `mcp__mirdan__enhance_prompt` with a summary of the TODO group to
   establish quality context and detect security sensitivity.

2. Call `mcp__enyal__enyal_recall` with relevant queries to load project
   conventions and patterns for the area being implemented.

3. For each TODO in the stream:
   - Read the target file before modifying it
   - Implement the change following the plan's exact details
   - Call `mcp__mirdan__validate_code_quality` after each file edit
   - If `touches_security` was flagged, use `check_security=true`

4. After all TODOs are complete, store any new decisions or patterns via
   `mcp__enyal__enyal_remember` for future reference.

## Subagent Coordination

This subagent runs in the foreground — it receives TODO groups explicitly
from the user and must complete before reporting back. It may spawn readonly
subagents for parallel validation:
- `mirdan-quality-validator` (background) for async quality checks
- `mirdan-slop-detector` (background) for async AI slop detection
"""

_SUBAGENT_TEST_WRITER = """\
---
name: mirdan-test-writer
description: >-
  Write tests for implementation TODO groups from plan streams. Use when the
  user sends a test-writing Stream of TODOs from a mirdan parallel plan.
model: inherit
readonly: false
background: false
---

# mirdan Test Writer

Write tests for implementation streams — ensure coverage and correctness.

## Instructions

1. Read the implementation code being tested to understand its API surface,
   edge cases, and error paths.

2. Call `mcp__enyal__enyal_recall("testing conventions")` to load project
   test patterns (fixtures, naming, structure).

3. For each test TODO in the stream:
   - Write tests following the project's existing test patterns
   - Include positive cases, error cases, and boundary conditions
   - Call `mcp__mirdan__validate_code_quality` on each test file

4. Run the test suite to confirm all new tests pass:
   - Fix any failures before marking the TODO complete

## Subagent Coordination

This subagent runs in the foreground — it receives test TODO groups explicitly
from the user and must complete before reporting back. It may spawn readonly
subagents for test quality review:
- `mirdan-test-auditor` (foreground) for test quality auditing
"""

_CURSOR_SUBAGENTS: dict[str, str] = {
    "mirdan-quality-validator.md": _SUBAGENT_QUALITY_VALIDATOR,
    "mirdan-security-scanner.md": _SUBAGENT_SECURITY_SCANNER,
    "mirdan-test-auditor.md": _SUBAGENT_TEST_AUDITOR,
    "mirdan-slop-detector.md": _SUBAGENT_SLOP_DETECTOR,
    "mirdan-architecture-reviewer.md": _SUBAGENT_ARCHITECTURE_REVIEWER,
    "mirdan-implementer.md": _SUBAGENT_IMPLEMENTER,
    "mirdan-test-writer.md": _SUBAGENT_TEST_WRITER,
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

_SKILL_DEBUG = """\
---
name: mirdan-debug
description: >-
  Quality-aware debugging workflow. Use when diagnosing bugs, tracing
  errors, or investigating unexpected behavior. Validates fixes against
  mirdan quality standards.
disable-model-invocation: false
---

# mirdan Debug — Quality-Aware Debugging

Debug issues with mirdan quality analysis to prevent introducing new problems.

## When to Use

- Diagnosing bugs or unexpected behavior
- Tracing error paths
- Investigating test failures
- Performance issues

## Workflow

1. Call `mcp__mirdan__enhance_prompt` with the bug description to get
   context and detect frameworks.

2. Use `@Docs [library-name]` to verify correct API behavior — many
   bugs are incorrect assumptions about library APIs.

3. Read relevant code and trace the actual error path before forming
   hypotheses.

4. Apply the minimal fix targeting the root cause, not just symptoms.

5. Call `mcp__mirdan__validate_code_quality` on modified files to
   confirm the fix does not introduce new violations.

6. Use Cursor's runtime instrumentation (v2.6) to capture actual execution
   paths and variable states when the root cause is not obvious from static
   code reading alone.
"""

_SKILL_REVIEW = """\
---
name: mirdan-review
description: >-
  Code review with mirdan quality standards enforcement. Use when
  reviewing PRs, code changes, or preparing code for review.
disable-model-invocation: false
---

# mirdan Review — Code Review with Quality Standards

Review code against mirdan quality standards.

## When to Use

- Reviewing pull requests
- Preparing code for review
- Auditing existing code quality
- Post-implementation quality check

## Workflow

1. Call `mcp__mirdan__get_quality_standards` for the language/framework
   to establish review criteria.

2. Read each changed file. Check for:
   - AI quality rules: AI001 (placeholders), AI002 (hallucinated imports),
     AI007 (security theater), AI008 (injection vulnerabilities)
   - Security rules: SEC001-SEC014
   - Architecture rules: ARCH001-ARCH005

3. Call `mcp__mirdan__validate_code_quality` with `check_security=true`
   on security-sensitive files.

4. Report findings grouped by severity: errors (must fix), warnings
   (should fix).
"""

_SKILL_PLAN = """\
---
name: mirdan-plan
description: >-
  Quality-gated implementation planning. Use when creating implementation
  plans, architecture designs, or task breakdowns.
disable-model-invocation: false
---

# mirdan Plan — Quality-Gated Implementation Planning

Create implementation plans with anti-hallucination standards.

## When to Use

- Creating implementation plans
- Architecture design sessions
- Task breakdown and estimation
- Migration or refactor planning

## Workflow

1. Call `mcp__mirdan__enhance_prompt` with the planning task to detect
   frameworks and quality requirements.

2. Use `@Docs [library-name]` to look up current API documentation for
   each library involved — never plan around assumed APIs.

3. Read all files that will be modified before writing plan steps. Cite
   exact line numbers and function names.

4. Each plan step must include:
   - Exact file path (verified to exist)
   - Specific action with details
   - Verification method
   - Grounding (which tool confirmed the facts)

5. For tasks with independent work streams, group TODOs into parallel
   streams with agent annotations:
   `## Stream A: Data Layer [mirdan-implementer]`
   `## Stream C: Tests [mirdan-test-writer]`
   Use `(Depends on: Stream X)` to declare inter-stream dependencies.

6. No vague language: "should", "probably", "maybe" are not allowed in
   plan steps.
"""

_SKILL_QUALITY = """\
---
name: mirdan-quality
description: >-
  On-demand code quality validation. Use to validate specific files or
  recent changes against mirdan quality standards.
disable-model-invocation: true
---

# mirdan Quality — On-Demand Quality Validation

Run mirdan quality validation on demand.

## When to Use

Invoke explicitly with `/mirdan-quality` when you want to:
- Validate specific files
- Check quality of staged changes
- Get a quality score for recent work

## Workflow

1. Identify files to validate — changed files, staged files, or
   specific path.

2. Call `mcp__mirdan__validate_code_quality` on each file:
   - Use `check_security=true` for auth, input handling, database, or
     API code
   - Use `severity_threshold="info"` for comprehensive results

3. Call `mcp__mirdan__get_quality_trends` to check session-wide quality
   trends.

4. Report findings:
   - Errors (must fix before completing)
   - Warnings (should fix, note if deferring)
   - Overall quality score
"""

_SKILL_SCAN = """\
---
name: mirdan-scan
description: >-
  Convention and pattern scanner. Use to scan codebase for coding
  conventions, violations, and AI quality issues.
disable-model-invocation: true
---

# mirdan Scan — Convention Scanner

Scan the codebase for quality violations and convention patterns.

## When to Use

Invoke explicitly with `/mirdan-scan` when you want to:
- Discover codebase conventions
- Find pattern violations
- Audit AI code quality across the project

## Workflow

1. Identify files to scan — all source files, or scope by argument
   (path or language).

2. Call `mcp__mirdan__scan_conventions` to discover patterns and
   violations.

3. Call `mcp__mirdan__get_quality_standards` for the project language
   to verify findings against defined rules.

4. Report:
   - Violations by rule ID and count
   - New patterns discovered
   - Conventions that diverge from standards

5. If new high-confidence conventions are discovered, store them for
   future reference.
"""

_SKILL_GATE = """\
---
name: mirdan-gate
description: >-
  Quality gate check before commit or task completion. Use to validate
  all changed files pass mirdan quality standards.
disable-model-invocation: true
---

# mirdan Gate — Quality Gate

Run the full quality gate before committing or completing a task.

## When to Use

Invoke explicitly with `/mirdan-gate` when you want to:
- Validate all changes before committing
- Run a final quality check before marking a task complete
- Ensure security-critical files meet the 0.8 threshold

## Workflow

1. Find all changed files: uncommitted changes and staged files.

2. Call `mcp__mirdan__validate_code_quality` on each changed file.
   Use `check_security=true` for any file touching auth, input, SQL,
   or APIs.

3. Check that every file passes with quality score >= 0.7.
   Security-critical files require score >= 0.8.

4. Fix all errors. For warnings, note rule IDs and justification if
   deferring.

5. Re-validate after fixes to confirm PASS status.

6. Only mark the task complete after all files pass the quality gate.
"""

_CURSOR_SKILLS: dict[str, str] = {
    "mirdan-code": _SKILL_CODE,
    "mirdan-debug": _SKILL_DEBUG,
    "mirdan-review": _SKILL_REVIEW,
    "mirdan-plan": _SKILL_PLAN,
    "mirdan-quality": _SKILL_QUALITY,
    "mirdan-scan": _SKILL_SCAN,
    "mirdan-gate": _SKILL_GATE,
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
