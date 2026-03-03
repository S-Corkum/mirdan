"""Hook template generator for Claude Code hook lifecycle.

Generates hooks.json configurations covering the full Claude Code
hook event lifecycle: UserPromptSubmit, PreToolUse, PostToolUse, Stop,
SessionStart, SessionStop, SubagentStart, SubagentStop, PreCompact,
Notification.

Supports three stringency levels:
- minimal: PostToolUse + Stop (2 hooks)
- standard: UserPromptSubmit + PreToolUse + PostToolUse + Stop + SubagentStart (5 hooks)
- comprehensive: All 7 enforcement hooks (full lifecycle)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class HookStringency(Enum):
    """Hook stringency levels controlling how many hooks are generated."""

    MINIMAL = "minimal"  # 2 hooks: PostToolUse, Stop
    STANDARD = "standard"  # 5 hooks: UserPromptSubmit, PreToolUse, PostToolUse, Stop, SubagentStart
    COMPREHENSIVE = "comprehensive"  # 7 hooks: full enforcement lifecycle


# Events for each stringency level
STRINGENCY_EVENTS: dict[HookStringency, list[str]] = {
    HookStringency.MINIMAL: ["PostToolUse", "Stop"],
    HookStringency.STANDARD: [
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "Stop",
        "SubagentStart",
    ],
    HookStringency.COMPREHENSIVE: [
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "Stop",
        "PreCompact",
        "SubagentStart",
    ],
}


@dataclass
class HookConfig:
    """Configuration for hook generation behavior."""

    enabled_events: list[str] = field(
        default_factory=lambda: [
            "PreToolUse",
            "PostToolUse",
            "Stop",
        ]
    )
    quick_validate_timeout: int = 5000  # ms
    auto_fix_suggestions: bool = True
    compaction_resilience: bool = False
    multi_agent_awareness: bool = False

    # Advanced events (opt-in)
    session_hooks: bool = False
    subagent_hooks: bool = False
    notification_hooks: bool = False


# All supported Claude Code hook events
ALL_HOOK_EVENTS = [
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "Stop",
    "SessionStart",
    "SessionStop",
    "SubagentStart",
    "SubagentStop",
    "PreCompact",
    "Notification",
]


class HookTemplateGenerator:
    """Generates hooks.json configurations for Claude Code.

    Each hook event has a dedicated method that returns the hook
    definition. The generate() method assembles all enabled hooks
    into a complete hooks.json structure.
    """

    def __init__(
        self,
        config: HookConfig | None = None,
        mirdan_command: str = "mirdan",
    ) -> None:
        self._config = config or HookConfig()
        self._mirdan_cmd = mirdan_command

    def generate(self) -> dict[str, Any]:
        """Generate complete hooks.json configuration.

        Returns:
            hooks.json-compatible dict with all enabled hook events.
        """
        hooks: dict[str, list[dict[str, Any]]] = {}

        for event in self._effective_events():
            generator = self._event_generators().get(event)
            if generator:
                hook_def = generator()
                if hook_def:
                    hooks[event] = hook_def

        return {"hooks": hooks}

    def generate_claude_code_hooks(
        self,
        stringency: HookStringency = HookStringency.COMPREHENSIVE,
    ) -> dict[str, Any]:
        """Generate hooks.json for Claude Code with prompt-type hooks.

        Uses prompt-type hooks (not command-type) for Claude Code plugin
        compatibility. The stringency level controls how many hooks are
        generated.

        Args:
            stringency: Hook stringency level (minimal, standard, comprehensive).

        Returns:
            hooks.json-compatible dict.
        """
        hooks: dict[str, list[dict[str, Any]]] = {}
        events = STRINGENCY_EVENTS[stringency]

        for event in events:
            generator = self._event_generators().get(event)
            if generator:
                hook_def = generator()
                if hook_def:
                    hooks[event] = hook_def

        return {"hooks": hooks}

    def generate_and_write(self, hooks_path: Path) -> Path:
        """Generate and write hooks.json to disk.

        Args:
            hooks_path: Path to write hooks.json.

        Returns:
            Path to the written file.
        """
        hooks_path.parent.mkdir(parents=True, exist_ok=True)
        config = self.generate()
        with hooks_path.open("w") as f:
            json.dump(config, f, indent=2)
        return hooks_path

    def _effective_events(self) -> list[str]:
        """Get the effective list of enabled events."""
        events = list(self._config.enabled_events)

        # Add opt-in advanced events
        if self._config.session_hooks:
            for ev in ("SessionStart", "SessionStop"):
                if ev not in events:
                    events.append(ev)

        if self._config.subagent_hooks:
            for ev in ("SubagentStart", "SubagentStop"):
                if ev not in events:
                    events.append(ev)

        if self._config.compaction_resilience and "PreCompact" not in events:
            events.append("PreCompact")

        if self._config.notification_hooks and "Notification" not in events:
            events.append("Notification")

        return events

    def _event_generators(self) -> dict[str, Any]:
        """Map event names to their generator methods."""
        return {
            "UserPromptSubmit": self._user_prompt_submit,
            "PreToolUse": self._pre_tool_use,
            "PostToolUse": self._post_tool_use,
            "Stop": self._stop,
            "SessionStart": self._session_start,
            "SessionStop": self._session_stop,
            "SubagentStart": self._subagent_start,
            "SubagentStop": self._subagent_stop,
            "PreCompact": self._pre_compact,
            "Notification": self._notification,
        }

    # ------------------------------------------------------------------
    # Hook event generators
    # ------------------------------------------------------------------

    def _user_prompt_submit(self) -> list[dict[str, Any]]:
        """UserPromptSubmit: Inject quality context before processing."""
        return [
            {
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "mirdan quality context is active. For coding tasks:"
                            " 1) Call mcp__mirdan__enhance_prompt first for quality"
                            " requirements and session context."
                            " 2) Call mcp__mirdan__get_quality_standards for the"
                            " detected language."
                            " 3) Follow the quality_requirements from enhance_prompt"
                            " output as constraints during implementation."
                        ),
                    }
                ],
            }
        ]

    def _pre_tool_use(self) -> list[dict[str, Any]]:
        """PreToolUse: Security-aware reminder before Write/Edit."""
        return [
            {
                "matcher": "Write|Edit|MultiEdit",
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "Before writing code: ensure you have called"
                            " mcp__mirdan__enhance_prompt for this task's quality"
                            " requirements. If writing security-sensitive code"
                            " (auth, input handling, SQL, API endpoints), note that"
                            " mcp__mirdan__validate_code_quality must be called"
                            " with check_security=true after this edit."
                        ),
                    }
                ],
            }
        ]

    def _post_tool_use(self) -> list[dict[str, Any]]:
        """PostToolUse: Validation after Write/Edit.

        Uses prompt-type hooks to invoke mirdan MCP tools for validation.
        """
        return [
            {
                "matcher": "Write|Edit|MultiEdit",
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "Code was just written or edited. Call"
                            " mcp__mirdan__validate_code_quality on the changed"
                            " code with max_tokens=500 and model_tier=haiku."
                            " If the code touches auth, input handling, SQL, or"
                            " API endpoints, set check_security=true. Fix all"
                            " errors (severity=error) immediately before"
                            " continuing. Note warnings for awareness but"
                            " continue working."
                        ),
                    }
                ],
            }
        ]

    def _stop(self) -> list[dict[str, Any]]:
        """Stop: Verification gate before task completion."""
        return [
            {
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "Before completing this task, verify quality was"
                            " enforced: 1) Was mcp__mirdan__enhance_prompt"
                            " called at the start? If not, call it now."
                            " 2) Was mcp__mirdan__validate_code_quality called"
                            " on all changed files? If not, validate now."
                            " 3) Are there any unresolved errors from"
                            " validation? If so, fix them. Do NOT complete the"
                            " task if there are unresolved validation errors."
                        ),
                    }
                ],
            }
        ]

    def _session_start(self) -> list[dict[str, Any]]:
        """SessionStart: Inject quality context at session beginning."""
        return [
            {
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "mirdan quality context is active. For any coding task:"
                            " 1) Call mcp__mirdan__enhance_prompt first for requirements."
                            " 2) After writing code, call mcp__mirdan__validate_code_quality."
                            " 3) Fix all errors before considering the task complete."
                        ),
                    }
                ],
            }
        ]

    def _session_stop(self) -> list[dict[str, Any]]:
        """SessionStop: Persist quality report at session end."""
        return [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{self._mirdan_cmd} report --session --format json",
                        "timeout": 10000,
                    }
                ],
            }
        ]

    def _subagent_start(self) -> list[dict[str, Any]]:
        """SubagentStart: Pass quality context to subagents."""
        return [
            {
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "You are a subagent. mirdan quality standards are active."
                            " Validate any code you write with"
                            " mcp__mirdan__validate_code_quality before returning results."
                        ),
                    }
                ],
            }
        ]

    def _subagent_stop(self) -> list[dict[str, Any]]:
        """SubagentStop: Validate subagent output."""
        return [
            {
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "Review the subagent's output for quality."
                            " If code was written, ensure it was validated"
                            " with mcp__mirdan__validate_code_quality."
                        ),
                    }
                ],
            }
        ]

    def _pre_compact(self) -> list[dict[str, Any]]:
        """PreCompact: Serialize quality state before context compaction."""
        return [
            {
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "Context is being compacted. Preserve this mirdan state:"
                            " - Active session and its quality requirements"
                            " - Any unresolved violations from the last validation"
                            " - The current task type and security sensitivity"
                            " After compaction, restore state by calling"
                            " mcp__mirdan__enhance_prompt with the preserved context."
                        ),
                    }
                ],
            }
        ]

    def _notification(self) -> list[dict[str, Any]]:
        """Notification: Quality alerts for significant events."""
        return [
            {
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "mirdan quality alert: Check if any recent code changes"
                            " introduced quality regressions. Run"
                            " mcp__mirdan__validate_code_quality on modified files."
                        ),
                    }
                ],
            }
        ]
