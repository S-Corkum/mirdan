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
    env_hints: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str | bool]:
        """Convert to dictionary for API response."""
        return {
            "ide": self.ide.value,
            "ide_name": self.ide_name,
            "is_agent_context": self.is_agent_context,
        }


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

        return EnvironmentInfo(
            ide=ide_type,
            ide_name=ide_name,
            is_agent_context=is_agent,
            env_hints=env_hints,
        )

    return EnvironmentInfo(env_hints=env_hints)


def is_claude_code() -> bool:
    """Quick check if running inside Claude Code."""
    return bool(os.environ.get("CLAUDE_CODE_RUNNING") or os.environ.get("CLAUDE_CODE_VERSION"))


def is_cursor() -> bool:
    """Quick check if running inside Cursor."""
    return bool(os.environ.get("CURSOR_TRACE_ID") or os.environ.get("CURSOR_SESSION_ID"))
