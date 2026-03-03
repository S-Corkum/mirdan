"""IDE and environment detection from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class IDEType(Enum):
    """Known IDE/agent environments."""

    CLAUDE_CODE = "claude_code"
    CURSOR = "cursor"
    VSCODE = "vscode"
    WINDSURF = "windsurf"
    UNKNOWN = "unknown"


@dataclass
class EnvironmentInfo:
    """Detected environment information."""

    ide: IDEType = IDEType.UNKNOWN
    ide_name: str = "unknown"
    is_agent_context: bool = False
    context_budget: int | None = None  # Remaining context tokens, if detectable
    env_hints: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str | bool | int | None]:
        """Convert to dictionary for API response."""
        result: dict[str, str | bool | int | None] = {
            "ide": self.ide.value,
            "ide_name": self.ide_name,
            "is_agent_context": self.is_agent_context,
        }
        if self.context_budget is not None:
            result["context_budget"] = self.context_budget
        return result


# Environment variable mapping for IDE detection.
# Order matters: more specific checks first.
_IDE_DETECTORS: list[tuple[str, IDEType, str]] = [
    ("CLAUDE_CODE_RUNNING", IDEType.CLAUDE_CODE, "Claude Code"),
    ("CLAUDE_CODE_VERSION", IDEType.CLAUDE_CODE, "Claude Code"),
    ("CURSOR_TRACE_ID", IDEType.CURSOR, "Cursor"),
    ("CURSOR_SESSION_ID", IDEType.CURSOR, "Cursor"),
    ("WINDSURF_SESSION_ID", IDEType.WINDSURF, "Windsurf"),
    ("VSCODE_PID", IDEType.VSCODE, "VS Code"),
    ("VSCODE_IPC_HOOK", IDEType.VSCODE, "VS Code"),
    ("TERM_PROGRAM", IDEType.VSCODE, "VS Code"),  # Often "vscode"
]


def detect_environment() -> EnvironmentInfo:
    """Detect the current IDE/agent environment from env vars.

    Checks environment variables to determine which IDE or agent
    platform is running Mirdan. This enables environment-specific
    output formatting and feature enablement.

    Returns:
        EnvironmentInfo with detected IDE and context flags.
    """
    env_hints: dict[str, str] = {}

    for env_var, ide_type, ide_name in _IDE_DETECTORS:
        value = os.environ.get(env_var, "")
        if not value:
            continue

        # Special case: TERM_PROGRAM must contain "vscode"
        if env_var == "TERM_PROGRAM" and "vscode" not in value.lower():
            continue

        env_hints[env_var] = value
        is_agent = ide_type in (IDEType.CLAUDE_CODE, IDEType.CURSOR, IDEType.WINDSURF)
        context_budget = _detect_context_budget()

        return EnvironmentInfo(
            ide=ide_type,
            ide_name=ide_name,
            is_agent_context=is_agent,
            context_budget=context_budget,
            env_hints=env_hints,
        )

    return EnvironmentInfo(
        context_budget=_detect_context_budget(),
        env_hints=env_hints,
    )


def _detect_context_budget() -> int | None:
    """Detect remaining context budget from environment variables.

    Checks for context budget hints set by IDE integrations or hook
    environments. Returns None if no budget information is available.

    Returns:
        Estimated remaining context tokens, or None.
    """
    # Claude Code sets these in hook environments
    for env_var in (
        "MIRDAN_CONTEXT_BUDGET",
        "CLAUDE_CONTEXT_REMAINING",
        "CONTEXT_BUDGET",
    ):
        value = os.environ.get(env_var, "")
        if value:
            try:
                budget = int(value)
                if budget > 0:
                    return budget
            except ValueError:
                continue
    return None


def is_claude_code() -> bool:
    """Quick check if running inside Claude Code."""
    return bool(os.environ.get("CLAUDE_CODE_RUNNING") or os.environ.get("CLAUDE_CODE_VERSION"))


def is_cursor() -> bool:
    """Quick check if running inside Cursor."""
    return bool(os.environ.get("CURSOR_TRACE_ID") or os.environ.get("CURSOR_SESSION_ID"))
