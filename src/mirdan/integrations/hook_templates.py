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
    ) -> None:
        self._config = config or HookConfig()
        self._mirdan_cmd = mirdan_command
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

    def _post_tool_use(self) -> list[dict[str, Any]]:
        """PostToolUse: Validation after Write/Edit.

        Uses both command-type (fast CLI validation) and prompt-type
        (LLM-driven fix guidance) hooks for comprehensive coverage.
        Includes dependency manifest change detection at STANDARD+.
        """
        hooks: list[dict[str, Any]] = [
            {
                "matcher": "Write|Edit|MultiEdit",
                "hooks": [
                    {
                        "type": "command",
                        "command": (
                            f"{self._mirdan_cmd} validate --quick --scope essential"
                            " --file $TOOL_INPUT_FILE_PATH --format micro"
                        ),
                        "timeout": self._config.quick_validate_timeout,
                    },
                    {
                        "type": "prompt",
                        "prompt": (
                            "Review validation output above. Fix errors"
                            " immediately. If you intentionally skip a"
                            " violation, note the rule ID — frequent"
                            " overrides inform severity tuning."
                            " If the code touches auth, input handling,"
                            " SQL, or API endpoints, call"
                            " mcp__mirdan__validate_code_quality with"
                            " check_security=true for deeper analysis."
                        ),
                    },
                ],
            },
        ]

        # Dependency manifest change detection (STANDARD+ stringency)
        is_standard_plus = self._current_stringency in (
            HookStringency.STANDARD,
            HookStringency.COMPREHENSIVE,
        )
        if is_standard_plus:
            hooks.append(
                {
                    "matcher": "Write|Edit",
                    "hooks": [
                        {
                            "type": "prompt",
                            "prompt": (
                                "If the file just modified is a dependency manifest"
                                " (package.json, pyproject.toml, Cargo.toml, go.mod,"
                                " requirements.txt, pom.xml), call"
                                " mcp__mirdan__scan_dependencies to check for"
                                " vulnerabilities in the updated dependencies."
                            ),
                        }
                    ],
                }
            )

        return hooks

    def _post_tool_use_failure(self) -> list[dict[str, Any]]:
        """PostToolUseFailure: Log failed tool calls and suggest alternatives."""
        return [
            {
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "A tool call just failed. Before retrying:"
                            " 1) Analyze the error — is it a permissions issue,"
                            " invalid input, or a real bug?"
                            " 2) If a Write/Edit failed, check if the file path"
                            " is correct and the content is valid."
                            " 3) Consider an alternative approach rather than"
                            " retrying the exact same call."
                            " 4) If validation tools failed, proceed with caution"
                            " and validate manually before completing."
                        ),
                    }
                ],
            }
        ]

    def _stop(self) -> list[dict[str, Any]]:
        """Stop: Quality gate before task completion.

        Uses command-type (mirdan gate) for automated pass/fail
        and prompt-type for LLM-driven remediation if gate fails.
        """
        return [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{self._mirdan_cmd} gate --format text",
                        "timeout": 30000,
                    },
                    {
                        "type": "prompt",
                        "prompt": (
                            "Review the quality gate output above. If FAIL:"
                            " fix all errors, then re-run validation. If PASS:"
                            " the task can be completed. Do NOT complete the"
                            " task if the quality gate failed."
                        ),
                    },
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
        """SubagentStart: Register agent and pass quality context."""
        return [
            {
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "You are a subagent. mirdan quality standards are active."
                            " Validate any code you write with"
                            " mcp__mirdan__validate_code_quality before returning results."
                            " Multi-agent coordination is active — claim files before"
                            " validating to avoid duplicate work with other agents."
                            " Use mcp__mirdan__get_quality_trends to check session state."
                        ),
                    }
                ],
            }
        ]

    def _subagent_stop(self) -> list[dict[str, Any]]:
        """SubagentStop: Release claims and validate subagent output."""
        return [
            {
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "Review the subagent's output for quality."
                            " If code was written, ensure it was validated"
                            " with mcp__mirdan__validate_code_quality."
                            " Release any file claims held by this agent"
                            " and aggregate results for the coordination summary."
                        ),
                    }
                ],
            }
        ]

    def _pre_compact(self) -> list[dict[str, Any]]:
        """PreCompact: Serialize quality state before context compaction.

        Includes structured format for critical quality state so the LLM
        can preserve session scores and unresolved violations across
        compaction boundaries.
        """
        return [
            {
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "Context is being compacted. CRITICAL: Preserve"
                            " mirdan quality state in your compacted context"
                            " using this exact format:\n"
                            "## mirdan Compacted State (Restore After Compaction)\n"
                            "- Session: <session_id from enhance_prompt>\n"
                            "- Task: <task_type>\n"
                            "- Language: <detected language>\n"
                            "- Security: sensitive|normal\n"
                            "- Last score: <most recent validation score>\n"
                            "- Open violations: <count of unresolved errors>\n"
                            "- Frameworks: <comma-separated list>\n"
                            "- Validated files: <list of files already validated>\n\n"
                            "After compaction, restore state by calling"
                            " mcp__mirdan__enhance_prompt with the preserved"
                            " context to re-establish the quality session."
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

    def _permission_request(self) -> list[dict[str, Any]]:
        """PermissionRequest: Security audit before granting permissions."""
        return [
            {
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "A permission is being requested. Before granting:"
                            " 1) Verify the action is necessary for the current task."
                            " 2) Check if the action could modify security-sensitive"
                            " files (auth, secrets, configs)."
                            " 3) Prefer minimal permissions — grant only what's needed."
                            " 4) If the action involves destructive operations"
                            " (delete, overwrite), confirm intent."
                        ),
                    }
                ],
            }
        ]

    def _task_completed(self) -> list[dict[str, Any]]:
        """TaskCompleted: Generate session quality report when task finishes.

        Uses command-type to run mirdan report --session and
        prompt-type to surface unresolved issues.
        """
        return [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": (f"{self._mirdan_cmd} report --session --format json"),
                        "timeout": 30000,
                    },
                    {
                        "type": "prompt",
                        "prompt": (
                            "Review the session quality report above."
                            " If there are unresolved errors, fix them"
                            " before considering the task complete."
                        ),
                    },
                ],
            }
        ]

    def _teammate_idle(self) -> list[dict[str, Any]]:
        """TeammateIdle: Assign background quality validation to idle agents."""
        return [
            {
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "An agent teammate is idle. Use multi-agent"
                            " quality coordination to assign work:"
                            " 1) Check for unassigned files that need validation."
                            " 2) Claim an unvalidated file and run"
                            " mcp__mirdan__validate_code_quality on it."
                            " 3) Release the file claim with validation results."
                            " 4) Check for quality regressions via"
                            " mcp__mirdan__get_quality_trends."
                            " 5) Report any new violations found."
                        ),
                    }
                ],
            }
        ]

    def _config_change(self) -> list[dict[str, Any]]:
        """ConfigChange: Re-validate after config changes."""
        return [
            {
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "Configuration was just changed. If .mirdan/config.yaml"
                            " or quality-related settings were modified, call"
                            " mcp__mirdan__enhance_prompt to refresh quality context"
                            " with the new configuration."
                        ),
                    }
                ],
            }
        ]

    def _worktree_create(self) -> list[dict[str, Any]]:
        """WorktreeCreate: Initialize quality context for new worktree."""
        return [
            {
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "A new git worktree was created. Initialize mirdan"
                            " quality context for this worktree:"
                            " 1) Call mcp__mirdan__enhance_prompt to establish"
                            " quality requirements for the worktree's task."
                            " 2) Note the worktree path for file validation."
                        ),
                    }
                ],
            }
        ]

    def _worktree_remove(self) -> list[dict[str, Any]]:
        """WorktreeRemove: Save quality summary for completed worktree."""
        return [
            {
                "hooks": [
                    {
                        "type": "prompt",
                        "prompt": (
                            "A git worktree is being removed. Before cleanup:"
                            " 1) Verify all code in the worktree was validated."
                            " 2) Note any unresolved quality issues for the"
                            " main branch."
                        ),
                    }
                ],
            }
        ]

    # ------------------------------------------------------------------
    # LLM-enhanced hook generators (Claude Code only)
    # ------------------------------------------------------------------

    def _user_prompt_submit_llm(self) -> list[dict[str, Any]]:
        """UserPromptSubmit with local LLM triage via sidecar.

        The command hook outputs triage results which Claude Code injects
        into the model's context before processing. This allows the paid
        model to skip unnecessary work for LOCAL_ONLY tasks.
        """
        return [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": _TRIAGE_HOOK_SCRIPT,
                        "timeout": 15000,
                    },
                    {
                        "type": "prompt",
                        "prompt": (
                            "The triage output above classifies this task."
                            " If classification is 'local_only', handle it"
                            " with minimal token usage — no deep exploration."
                            " If 'local_assist' or 'paid_minimal', call"
                            " mcp__mirdan__enhance_prompt for focused guidance."
                            " If 'paid_required', proceed normally."
                        ),
                    },
                ],
            }
        ]

    def _stop_llm(self) -> list[dict[str, Any]]:
        """Stop with local LLM check runner via sidecar.

        Runs lint + typecheck + test locally, LLM parses output.
        Results are injected so Claude only fixes complex issues.
        """
        return [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{self._mirdan_cmd} check --smart",
                        "timeout": 60000,
                    },
                    {
                        "type": "prompt",
                        "prompt": (
                            "Review the check results above. Items marked"
                            " 'auto_fixed' are already resolved. Focus on"
                            " 'needs_attention' items only. If all_pass is"
                            " true, the task can be completed."
                        ),
                    },
                ],
            }
        ]


# Inline shell script for triage hook — tries sidecar first, falls back to CLI.
_TRIAGE_HOOK_SCRIPT = (
    'bash -c \''
    'PORT_FILE=".mirdan/sidecar.port";'
    ' if [ -f "$PORT_FILE" ]; then'
    ' curl -s --max-time 10 "http://localhost:$(cat $PORT_FILE)/triage" --data-binary @-;'
    " else"
    " mirdan triage --stdin 2>/dev/null;"
    " fi'"
)
