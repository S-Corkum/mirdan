"""Hook template generator for Claude Code hook lifecycle.

Generates hooks.json configurations covering the full Claude Code
hook event lifecycle with 17 supported events across 4 hook types
(prompt, command, agent, http).

Supports three stringency levels:
- minimal: PostToolUse + Stop (2 hooks)
- standard: UserPromptSubmit + PostToolUse + Stop + SubagentStart (4 hooks)
- comprehensive: All enforcement hooks including new March 2026 events
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
    STANDARD = "standard"  # 5 hooks: core enforcement
    COMPREHENSIVE = "comprehensive"  # Full enforcement lifecycle


# Events for each stringency level
STRINGENCY_EVENTS: dict[HookStringency, list[str]] = {
    HookStringency.MINIMAL: ["PostToolUse", "Stop"],
    HookStringency.STANDARD: [
        "UserPromptSubmit",
        "PostToolUse",
        "Stop",
        "SubagentStart",
    ],
    HookStringency.COMPREHENSIVE: [
        "UserPromptSubmit",
        "PostToolUse",
        "PostToolUseFailure",
        "Stop",
        "SessionStart",
        "PreCompact",
        "SubagentStart",
        "SubagentStop",
        "TaskCompleted",
        "TeammateIdle",
        "PermissionRequest",
        "ConfigChange",
        "WorktreeCreate",
        "WorktreeRemove",
    ],
}


@dataclass
class HookConfig:
    """Configuration for hook generation behavior."""

    enabled_events: list[str] = field(
        default_factory=lambda: [
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


# All supported Claude Code hook events (17 total, March 2026)
ALL_HOOK_EVENTS = [
    "UserPromptSubmit",
    "PostToolUse",
    "PostToolUseFailure",
    "Stop",
    "SessionStart",
    "SessionStop",
    "SubagentStart",
    "SubagentStop",
    "PreCompact",
    "Notification",
    "PermissionRequest",
    "TaskCompleted",
    "TeammateIdle",
    "ConfigChange",
    "WorktreeCreate",
    "WorktreeRemove",
]


class HookTemplateGenerator:
    """Generates hooks.json configurations for Claude Code.

    Each hook event has a dedicated method that returns the hook
    definition. The generate() method assembles all enabled hooks
    into a complete hooks.json structure.

    Supports multiple hook types:
    - prompt: Injects text into the LLM context
    - command: Runs a shell command
    - agent: Spawns a background agent
    """

    def __init__(
        self,
        config: HookConfig | None = None,
        mirdan_command: str = "mirdan",
        hook_script_path: str | None = None,
    ) -> None:
        self._config = config or HookConfig()
        self._mirdan_cmd = mirdan_command
        # Absolute path to the companion validate-file.sh script. When None,
        # falls back to the relative ``.claude/hooks/validate-file.sh`` —
        # which only resolves correctly when the shell cwd is the project
        # root. Init should always supply the absolute path.
        self._hook_script_path = hook_script_path or ".claude/hooks/validate-file.sh"
        self._current_stringency: HookStringency | None = None

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
        llm_enabled: bool = False,
    ) -> dict[str, Any]:
        """Generate hooks.json for Claude Code.

        Uses a mix of hook types (prompt, command, agent) depending
        on the event. The stringency level controls how many hooks
        are generated.

        When llm_enabled is True, UserPromptSubmit and Stop hooks use
        command-type hooks that call the sidecar/CLI for local LLM
        triage and check runner, injecting results into Claude's context.

        Args:
            stringency: Hook stringency level (minimal, standard, comprehensive).
            llm_enabled: Whether local LLM features are enabled.

        Returns:
            hooks.json-compatible dict.
        """
        hooks: dict[str, list[dict[str, Any]]] = {}
        events = STRINGENCY_EVENTS[stringency]
        self._current_stringency = stringency

        for event in events:
            # LLM-enabled overrides for specific events
            if llm_enabled and event == "UserPromptSubmit":
                hooks[event] = self._user_prompt_submit_llm()
                continue
            if llm_enabled and event == "Stop":
                hooks[event] = self._stop_llm()
                continue

            generator = self._event_generators().get(event)
            if generator:
                hook_def = generator()
                if hook_def:
                    hooks[event] = hook_def

        self._current_stringency = None
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

        if self._config.multi_agent_awareness:
            for ev in ("TeammateIdle", "TaskCompleted"):
                if ev not in events:
                    events.append(ev)

        if self._config.notification_hooks and "Notification" not in events:
            events.append("Notification")

        return events

    def _event_generators(self) -> dict[str, Any]:
        """Map event names to their generator methods."""
        return {
            "UserPromptSubmit": self._user_prompt_submit,
            "PostToolUse": self._post_tool_use,
            "PostToolUseFailure": self._post_tool_use_failure,
            "Stop": self._stop,
            "SessionStart": self._session_start,
            "SessionStop": self._session_stop,
            "SubagentStart": self._subagent_start,
            "SubagentStop": self._subagent_stop,
            "PreCompact": self._pre_compact,
            "Notification": self._notification,
            "PermissionRequest": self._permission_request,
            "TaskCompleted": self._task_completed,
            "TeammateIdle": self._teammate_idle,
            "ConfigChange": self._config_change,
            "WorktreeCreate": self._worktree_create,
            "WorktreeRemove": self._worktree_remove,
        }

    # ------------------------------------------------------------------
    # Hook event generators
    # ------------------------------------------------------------------

    def _user_prompt_submit(self) -> list[dict[str, Any]]:
        """UserPromptSubmit: No-op placeholder.

        UserPromptSubmit is a blocking event — any prompt-type hook here
        is LLM-evaluated as a gate, and ends up blocking turns whenever
        the evaluator reads the injected text as an unsatisfied
        condition. Guidance belongs in rule files (``.claude/rules/``)
        or in a command-type hook that injects structured context, not
        here. Emit an empty list so the event is simply not registered.
        """
        return []

    def _post_tool_use(self) -> list[dict[str, Any]]:
        """PostToolUse: Validation after Write/Edit.

        **Command-type only.** Claude Code's hook harness runs every
        prompt-type hook through an LLM evaluator that treats the prompt
        as a gating condition — when the condition can't be cleanly
        satisfied (e.g. the edited file isn't a dependency manifest,
        validation produced no stdout, the condition references output
        that didn't arrive), the evaluator reports "stopped continuation"
        and the turn is blocked. This affects every event, not just
        blocking ones. Guidance ("call X for deeper analysis", "scan
        deps on manifest edits") belongs in ``.claude/rules/*.md`` rule
        files, which are injected as context without gate semantics.
        """
        return [
            {
                "matcher": "Write|Edit|MultiEdit",
                "hooks": [
                    {
                        "type": "command",
                        "command": self._hook_script_path,
                        "timeout": self._config.quick_validate_timeout,
                    },
                ],
            },
        ]

    def _post_tool_use_failure(self) -> list[dict[str, Any]]:
        """PostToolUseFailure: Not registered — prompt-only event.

        Guidance on handling failed tool calls is injected via
        ``.claude/rules/*.md`` files rather than prompt-type hooks,
        which are evaluated as blocking gates.
        """
        return []

    def _stop(self) -> list[dict[str, Any]]:
        """Stop: Quality gate before task completion.

        Uses **command-type only**. The subprocess stdout is injected
        as context for the assistant; any prompt-type hook here would
        be evaluated as a blocking gate regardless of wording, which
        locks users out when the gate fails for infra reasons or when
        the assistant's prompt text is read as an unsatisfied condition.
        Non-zero exit code from the command is the real gate.
        """
        return [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{self._mirdan_cmd} gate --format text",
                        "timeout": 30000,
                    },
                ],
            }
        ]

    def _session_start(self) -> list[dict[str, Any]]:
        """SessionStart: Not registered — prompt-only event.

        Session-level quality context lives in ``.claude/rules/*.md``
        (alwaysApply) rather than in a prompt-type hook that the
        evaluator treats as a blocking gate.
        """
        return []

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
        """SubagentStart: Not registered — prompt-only event.

        Subagent guidance lives in ``.claude/rules/*.md`` files.
        """
        return []

    def _subagent_stop(self) -> list[dict[str, Any]]:
        """SubagentStop: Not registered — prompt-only event."""
        return []

    def _pre_compact(self) -> list[dict[str, Any]]:
        """PreCompact: Not registered — prompt-only event.

        Compaction-resilience state preservation is guidance, not a
        gate. Moved to rule files.
        """
        return []

    def _notification(self) -> list[dict[str, Any]]:
        """Notification: Not registered — prompt-only event."""
        return []

    def _permission_request(self) -> list[dict[str, Any]]:
        """PermissionRequest: Not registered — prompt-only event.

        A prompt-type hook here would be evaluated as a blocking
        permission gate, which is exactly the wrong shape — the user
        already sees the permission prompt and can answer it.
        """
        return []

    def _task_completed(self) -> list[dict[str, Any]]:
        """TaskCompleted: Command-only session quality report.

        The ``mirdan report --session`` output is injected as context.
        No prompt-type hook is registered — it would be evaluated as a
        gate and block turn completion on noisy output.
        """
        return [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": (f"{self._mirdan_cmd} report --session --format json"),
                        "timeout": 30000,
                    },
                ],
            }
        ]

    def _teammate_idle(self) -> list[dict[str, Any]]:
        """TeammateIdle: Not registered — prompt-only event."""
        return []

    def _config_change(self) -> list[dict[str, Any]]:
        """ConfigChange: Not registered — prompt-only event."""
        return []

    def _worktree_create(self) -> list[dict[str, Any]]:
        """WorktreeCreate: Not registered — prompt-only event."""
        return []

    def _worktree_remove(self) -> list[dict[str, Any]]:
        """WorktreeRemove: Not registered — prompt-only event."""
        return []

    # ------------------------------------------------------------------
    # LLM-enhanced hook generators (Claude Code only)
    # ------------------------------------------------------------------

    def _user_prompt_submit_llm(self) -> list[dict[str, Any]]:
        """UserPromptSubmit with local LLM triage via sidecar.

        **Command-type only.** The command's stdout (triage JSON) is
        injected as context that the assistant reads naturally. No
        prompt-type hook is registered, because on a blocking event
        any prompt text is treated as a condition by the hook-evaluator
        and ends up blocking turns when the condition can't be
        evaluated against actually-injected context. The triage result
        speaks for itself; guidance on how to use it belongs in rule
        files, not in an inline gate.
        """
        return [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": _TRIAGE_HOOK_SCRIPT,
                        "timeout": 15000,
                    },
                ],
            }
        ]

    def _stop_llm(self) -> list[dict[str, Any]]:
        """Stop with local LLM check runner via sidecar.

        **Command-type only.** Runs lint + typecheck + test locally and
        emits JSON as stdout. The JSON is injected into the assistant's
        context; real gating is the command's exit code, not any
        prompt-hook wording. A prompt-type hook here is always an
        LLM-evaluated gate that blocks whenever the evaluator decides
        the condition isn't satisfied — which happens for infra
        failures, empty output, and even correctly-worded "non-blocking
        advisory" text. Keeping this path command-only prevents the
        class of lockups we've seen.
        """
        return [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{self._mirdan_cmd} check --smart",
                        "timeout": 300000,
                    },
                ],
            }
        ]


# Inline shell script for triage hook — tries sidecar first, falls back to CLI.
_TRIAGE_HOOK_SCRIPT = (
    "bash -c '"
    'PORT_FILE=".mirdan/sidecar.port";'
    ' if [ -f "$PORT_FILE" ]; then'
    ' curl -s --max-time 10 "http://localhost:$(cat $PORT_FILE)/triage" --data-binary @-;'
    " else"
    " mirdan triage --stdin 2>/dev/null;"
    " fi'"
)
