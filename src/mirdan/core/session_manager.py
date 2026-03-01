"""In-memory session manager with TTL expiry for multi-turn orchestration."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from mirdan.models import Intent, SessionContext, TaskType

if TYPE_CHECKING:
    from mirdan.config import SessionConfig


class SessionManager:
    """Manages in-memory sessions with automatic TTL expiry.

    Sessions allow enhance_prompt to create state that validate_code_quality
    and other tools can reference, avoiding redundant parameter passing.
    """

    def __init__(self, config: SessionConfig | None = None) -> None:
        from mirdan.config import SessionConfig

        self._config = config or SessionConfig()
        self._sessions: dict[str, SessionContext] = {}

    def create_from_intent(self, intent: Intent) -> SessionContext:
        """Create a new session from an analyzed intent.

        Args:
            intent: The analyzed intent from enhance_prompt.

        Returns:
            A new SessionContext with a unique ID.
        """
        self._evict_expired()
        self._enforce_max_sessions()

        now = time.monotonic()
        session = SessionContext(
            session_id=uuid.uuid4().hex[:12],
            task_type=intent.task_type,
            detected_language=intent.primary_language,
            frameworks=list(intent.frameworks),
            touches_security=intent.touches_security,
            touches_rag=intent.touches_rag,
            touches_knowledge_graph=intent.touches_knowledge_graph,
            created_at=now,
            last_accessed=now,
        )
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> SessionContext | None:
        """Retrieve a session by ID, returning None if expired or missing.

        Args:
            session_id: The session identifier.

        Returns:
            The session context, or None if not found/expired.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None

        if self._is_expired(session):
            del self._sessions[session_id]
            return None

        session.last_accessed = time.monotonic()
        return session

    def remove(self, session_id: str) -> bool:
        """Remove a session explicitly.

        Args:
            session_id: The session identifier.

        Returns:
            True if the session was removed, False if not found.
        """
        return self._sessions.pop(session_id, None) is not None

    @property
    def active_count(self) -> int:
        """Return the number of non-expired sessions."""
        self._evict_expired()
        return len(self._sessions)

    def _is_expired(self, session: SessionContext) -> bool:
        """Check if a session has exceeded its TTL."""
        elapsed = time.monotonic() - session.last_accessed
        return elapsed > self._config.ttl_seconds

    def _evict_expired(self) -> None:
        """Remove all expired sessions."""
        expired = [sid for sid, s in self._sessions.items() if self._is_expired(s)]
        for sid in expired:
            del self._sessions[sid]

    def _enforce_max_sessions(self) -> None:
        """Evict oldest sessions if at capacity."""
        while len(self._sessions) >= self._config.max_sessions:
            # Remove the session with the oldest last_accessed time
            oldest_id = min(self._sessions, key=lambda sid: self._sessions[sid].last_accessed)
            del self._sessions[oldest_id]

    def apply_session_defaults(
        self,
        session_id: str,
        *,
        language: str = "auto",
        check_security: bool = True,
    ) -> tuple[str, bool]:
        """Apply session defaults to validation parameters.

        If a valid session exists and parameters are at their defaults,
        inherit from the session context.

        Args:
            session_id: The session identifier (empty string = no session).
            language: The language parameter (will be overridden if "auto").
            check_security: The security check flag.

        Returns:
            Tuple of (resolved_language, resolved_check_security).
        """
        if not session_id:
            return language, check_security

        session = self.get(session_id)
        if session is None:
            return language, check_security

        resolved_language = language
        if language == "auto" and session.detected_language:
            resolved_language = session.detected_language

        resolved_security = check_security
        if session.touches_security:
            resolved_security = True

        return resolved_language, resolved_security

    def create_empty(self) -> SessionContext:
        """Create a minimal session without intent (for testing or fallback).

        Returns:
            A new SessionContext with defaults.
        """
        self._evict_expired()
        self._enforce_max_sessions()

        now = time.monotonic()
        session = SessionContext(
            session_id=uuid.uuid4().hex[:12],
            task_type=TaskType.UNKNOWN,
            created_at=now,
            last_accessed=now,
        )
        self._sessions[session.session_id] = session
        return session
