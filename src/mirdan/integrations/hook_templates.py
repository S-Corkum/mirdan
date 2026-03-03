"""Hook template generator for Claude Code hook lifecycle.

Generates hooks.json configurations covering the full Claude Code
hook event lifecycle: PreToolUse, PostToolUse, Stop, SessionStart,
SessionStop, SubagentStart, SubagentStop, PreCompact, Notification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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

        event_generators = {
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

        for event in self._effective_events():
            generator = event_generators.get(event)
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

    # ------------------------------------------------------------------
    # Hook event generators
    # ------------------------------------------------------------------

    def _pre_tool_use(self) -> list[dict[str, Any]]:
        """PreToolUse: Smart reminder before Write/Edit to ensure quality."""
        return [
            {
                "matcher": "Write|Edit",
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "Before writing code, ensure you have called"
                            " mcp__mirdan__enhance_prompt for quality requirements."
                            " If not yet done, call it first."
                        ),
                    }
                ],
            }
        ]

    def _post_tool_use(self) -> list[dict[str, Any]]:
        """PostToolUse: Quick validation + auto-fix after Write/Edit."""
        hooks_list: list[dict[str, Any]] = [
            {
                "type": "command",
                "command": (
                    f'{self._mirdan_cmd} validate --quick'
                    ' --file "$TOOL_INPUT_FILE_PATH"'
                    ' --format micro'
                ),
                "timeout": self._config.quick_validate_timeout,
            }
        ]

        if self._config.auto_fix_suggestions:
            hooks_list.append({
                "type": "prompt",
                "prompt": (
                    "If the validation above found violations, consider running"
                    " mcp__mirdan__validate_code_quality with the full code for"
                    " detailed fixes. Fix all errors before proceeding."
                ),
            })

        return [
            {
                "matcher": "Write|Edit",
                "hooks": hooks_list,
            }
        ]

    def _stop(self) -> list[dict[str, Any]]:
        """Stop: Full validation before task completion."""
        return [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{self._mirdan_cmd} validate --staged --format text",
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
