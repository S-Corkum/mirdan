"""Coordinates file access across concurrent agent sessions.

Maintains an in-memory registry of file claims and detects conflicts
when multiple sessions operate on overlapping files. All data is
session-scoped and expires with sessions.

Thread safety: FastMCP serializes all tool calls on a single event
loop, so no locks are needed.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from mirdan.models import ConflictWarning, FileClaim

if TYPE_CHECKING:
    from mirdan.config import CoordinationConfig


class AgentCoordinator:
    """Coordinates file access across concurrent agent sessions.

    Maintains an in-memory registry of file claims and detects conflicts
    when multiple sessions operate on overlapping files. All data is
    session-scoped and expires with sessions.
    """

    def __init__(self, config: CoordinationConfig) -> None:
        self._config = config
        self._claims: dict[str, list[FileClaim]] = {}  # file_path → claims

    @property
    def is_enabled(self) -> bool:
        """Whether coordination is enabled (public API to avoid private attribute access)."""
        return self._config.enabled

    def claim_files(
        self,
        session_id: str,
        file_paths: list[str],
        claim_type: str,
        agent_label: str = "",
    ) -> list[ConflictWarning]:
        """Register file claims and return warnings for detected conflicts.

        Args:
            session_id: The claiming session's ID.
            file_paths: List of file paths to claim.
            claim_type: "read" or "write".
            agent_label: Optional human-readable label for the agent.

        Returns:
            List of conflict warnings (empty if no conflicts detected).
        """
        if not self._config.enabled:
            return []

        warnings: list[ConflictWarning] = []
        now = time.monotonic()

        for file_path in file_paths:
            existing = self._claims.get(file_path, [])

            # Check for write-write overlap
            if claim_type == "write" and self._config.warn_on_write_overlap:
                warnings.extend(
                    ConflictWarning(
                        type="write_overlap",
                        message=(
                            f"File '{file_path}' is already claimed for writing "
                            f"by session {claim.session_id}"
                            + (f" ({claim.agent_label})" if claim.agent_label else "")
                        ),
                        conflicting_sessions=[claim.session_id],
                        file_path=file_path,
                        severity="warning",
                    )
                    for claim in existing
                    if claim.session_id != session_id and claim.claim_type == "write"
                )

            # Check for stale read (another session has a read, we're writing)
            if claim_type == "write" and self._config.warn_on_stale_read:
                warnings.extend(
                    ConflictWarning(
                        type="stale_read",
                        message=(
                            f"File '{file_path}' has a read claim from "
                            f"session {claim.session_id}"
                            + (f" ({claim.agent_label})" if claim.agent_label else "")
                            + " — their cached view may become stale"
                        ),
                        conflicting_sessions=[claim.session_id],
                        file_path=file_path,
                        severity="info",
                    )
                    for claim in existing
                    if claim.session_id != session_id and claim.claim_type == "read"
                )

            # Register the new claim (avoid duplicate claims from same session)
            already_claimed = any(
                c.session_id == session_id and c.claim_type == claim_type
                for c in existing
            )
            if not already_claimed:
                new_claim = FileClaim(
                    session_id=session_id,
                    file_path=file_path,
                    claim_type=claim_type,
                    timestamp=now,
                    agent_label=agent_label,
                )
                if file_path not in self._claims:
                    self._claims[file_path] = []
                self._claims[file_path].append(new_claim)

        return warnings

    def release_session(self, session_id: str) -> None:
        """Release all claims for a session.

        Args:
            session_id: The session whose claims should be released.
        """
        empty_paths: list[str] = []
        for file_path, claims in self._claims.items():
            self._claims[file_path] = [c for c in claims if c.session_id != session_id]
            if not self._claims[file_path]:
                empty_paths.append(file_path)
        for path in empty_paths:
            del self._claims[path]

    def check_conflicts(self, session_id: str, file_path: str) -> list[ConflictWarning]:
        """Check for conflicts affecting a specific file and session.

        Used by validate_code_quality to detect stale-read conflicts
        where another session has modified a file this session read.

        Args:
            session_id: The checking session's ID.
            file_path: The file to check for conflicts.

        Returns:
            List of conflict warnings.
        """
        if not self._config.enabled:
            return []

        warnings: list[ConflictWarning] = []
        existing = self._claims.get(file_path, [])

        for claim in existing:
            if claim.session_id == session_id:
                continue
            if claim.claim_type == "write" and self._config.warn_on_stale_read:
                warnings.append(
                    ConflictWarning(
                        type="stale_read",
                        message=(
                            f"File '{file_path}' was modified by session {claim.session_id}"
                            + (f" ({claim.agent_label})" if claim.agent_label else "")
                            + " — your validation may be against stale code"
                        ),
                        conflicting_sessions=[claim.session_id],
                        file_path=file_path,
                        severity="warning",
                    )
                )

        return warnings

    def get_active_claims(self) -> dict[str, list[FileClaim]]:
        """Get all active claims (for debugging/visibility).

        Returns:
            Dictionary mapping file paths to their claims.
        """
        return dict(self._claims)

    def cleanup_stale(self, active_session_ids: set[str]) -> int:
        """Remove claims for expired sessions.

        Args:
            active_session_ids: Set of currently active session IDs.

        Returns:
            Number of claims removed.
        """
        removed = 0
        empty_paths: list[str] = []
        for file_path, claims in self._claims.items():
            before = len(claims)
            self._claims[file_path] = [
                c for c in claims if c.session_id in active_session_ids
            ]
            removed += before - len(self._claims[file_path])
            if not self._claims[file_path]:
                empty_paths.append(file_path)
        for path in empty_paths:
            del self._claims[path]
        return removed
