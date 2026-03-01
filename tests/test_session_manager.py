"""Tests for the SessionManager component."""

import time

from mirdan.config import SessionConfig
from mirdan.core.session_manager import SessionManager
from mirdan.models import Intent, SessionContext, TaskType


class TestSessionCreation:
    """Tests for creating sessions from intents."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.manager = SessionManager(SessionConfig(ttl_seconds=60, max_sessions=10))

    def test_create_from_intent_returns_session(self) -> None:
        """Should create a session from an intent."""
        intent = Intent(
            original_prompt="Create a Python function",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=["fastapi"],
            touches_security=True,
        )
        session = self.manager.create_from_intent(intent)

        assert isinstance(session, SessionContext)
        assert len(session.session_id) == 12
        assert session.task_type == TaskType.GENERATION
        assert session.detected_language == "python"
        assert session.frameworks == ["fastapi"]
        assert session.touches_security is True

    def test_create_from_intent_unique_ids(self) -> None:
        """Each session should have a unique ID."""
        intent = Intent(original_prompt="test", task_type=TaskType.UNKNOWN)
        s1 = self.manager.create_from_intent(intent)
        s2 = self.manager.create_from_intent(intent)

        assert s1.session_id != s2.session_id

    def test_create_from_intent_copies_all_flags(self) -> None:
        """Session should copy all relevant intent fields."""
        intent = Intent(
            original_prompt="Build a RAG knowledge graph",
            task_type=TaskType.GENERATION,
            primary_language="python",
            touches_security=False,
            touches_rag=True,
            touches_knowledge_graph=True,
        )
        session = self.manager.create_from_intent(intent)

        assert session.touches_rag is True
        assert session.touches_knowledge_graph is True
        assert session.touches_security is False

    def test_create_empty_session(self) -> None:
        """Should create a minimal empty session."""
        session = self.manager.create_empty()

        assert isinstance(session, SessionContext)
        assert session.task_type == TaskType.UNKNOWN
        assert session.detected_language is None

    def test_session_to_dict(self) -> None:
        """SessionContext.to_dict() should produce valid output."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.DEBUG,
            primary_language="typescript",
            touches_security=True,
        )
        session = self.manager.create_from_intent(intent)
        d = session.to_dict()

        assert d["session_id"] == session.session_id
        assert d["task_type"] == "debug"
        assert d["detected_language"] == "typescript"
        assert d["touches_security"] is True


class TestSessionLookup:
    """Tests for session retrieval."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.manager = SessionManager(SessionConfig(ttl_seconds=60, max_sessions=10))

    def test_get_existing_session(self) -> None:
        """Should retrieve an existing session by ID."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        session = self.manager.create_from_intent(intent)

        retrieved = self.manager.get(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_get_missing_session_returns_none(self) -> None:
        """Should return None for a nonexistent session ID."""
        assert self.manager.get("nonexistent") is None

    def test_get_updates_last_accessed(self) -> None:
        """Retrieving a session should update its last_accessed time."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        session = self.manager.create_from_intent(intent)
        original_time = session.last_accessed

        # Small delay to ensure monotonic time advances
        time.sleep(0.01)
        retrieved = self.manager.get(session.session_id)
        assert retrieved is not None
        assert retrieved.last_accessed >= original_time


class TestSessionExpiry:
    """Tests for TTL-based session expiry."""

    def test_expired_session_returns_none(self) -> None:
        """Should return None for an expired session."""
        # TTL of 0 means immediate expiry
        manager = SessionManager(SessionConfig(ttl_seconds=0, max_sessions=10))
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        session = manager.create_from_intent(intent)

        # Force expiry by waiting a tiny bit
        time.sleep(0.01)
        assert manager.get(session.session_id) is None

    def test_active_count_excludes_expired(self) -> None:
        """active_count should not include expired sessions."""
        manager = SessionManager(SessionConfig(ttl_seconds=0, max_sessions=10))
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        manager.create_from_intent(intent)

        time.sleep(0.01)
        assert manager.active_count == 0

    def test_non_expired_session_accessible(self) -> None:
        """Session within TTL should be accessible."""
        manager = SessionManager(SessionConfig(ttl_seconds=3600, max_sessions=10))
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        session = manager.create_from_intent(intent)

        assert manager.get(session.session_id) is not None
        assert manager.active_count == 1


class TestSessionCapacity:
    """Tests for max session enforcement."""

    def test_max_sessions_evicts_oldest(self) -> None:
        """Should evict oldest session when at capacity."""
        manager = SessionManager(SessionConfig(ttl_seconds=3600, max_sessions=3))
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)

        sessions = []
        for _ in range(3):
            s = manager.create_from_intent(intent)
            sessions.append(s)
            time.sleep(0.01)  # Ensure distinct timestamps

        # Adding a 4th should evict the oldest
        s4 = manager.create_from_intent(intent)
        assert manager.active_count == 3

        # First session should be evicted
        assert manager.get(sessions[0].session_id) is None
        # Latest should exist
        assert manager.get(s4.session_id) is not None

    def test_remove_session(self) -> None:
        """Should explicitly remove a session."""
        manager = SessionManager(SessionConfig(ttl_seconds=3600, max_sessions=10))
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        session = manager.create_from_intent(intent)

        assert manager.remove(session.session_id) is True
        assert manager.get(session.session_id) is None

    def test_remove_nonexistent_returns_false(self) -> None:
        """Should return False when removing nonexistent session."""
        manager = SessionManager(SessionConfig(ttl_seconds=3600, max_sessions=10))
        assert manager.remove("nonexistent") is False


class TestSessionDefaults:
    """Tests for apply_session_defaults method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.manager = SessionManager(SessionConfig(ttl_seconds=3600, max_sessions=10))

    def test_no_session_returns_original(self) -> None:
        """With empty session_id, should return original params."""
        lang, sec = self.manager.apply_session_defaults("", language="auto", check_security=True)
        assert lang == "auto"
        assert sec is True

    def test_invalid_session_returns_original(self) -> None:
        """With invalid session_id, should return original params."""
        lang, sec = self.manager.apply_session_defaults(
            "nonexistent", language="auto", check_security=False
        )
        assert lang == "auto"
        assert sec is False

    def test_session_overrides_auto_language(self) -> None:
        """Should override 'auto' language with session's detected language."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
        )
        session = self.manager.create_from_intent(intent)

        lang, _ = self.manager.apply_session_defaults(
            session.session_id, language="auto", check_security=True
        )
        assert lang == "python"

    def test_session_does_not_override_explicit_language(self) -> None:
        """Should not override explicitly provided language."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
        )
        session = self.manager.create_from_intent(intent)

        lang, _ = self.manager.apply_session_defaults(
            session.session_id, language="typescript", check_security=True
        )
        assert lang == "typescript"

    def test_session_enables_security_when_flagged(self) -> None:
        """Should enable security check when session has touches_security."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            touches_security=True,
        )
        session = self.manager.create_from_intent(intent)

        _, sec = self.manager.apply_session_defaults(
            session.session_id, language="auto", check_security=False
        )
        assert sec is True

    def test_session_preserves_security_when_not_flagged(self) -> None:
        """Should preserve original check_security when session doesn't flag it."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            touches_security=False,
        )
        session = self.manager.create_from_intent(intent)

        _, sec = self.manager.apply_session_defaults(
            session.session_id, language="auto", check_security=False
        )
        assert sec is False


class TestDefaultConfig:
    """Tests for SessionManager with default config."""

    def test_default_config_creation(self) -> None:
        """Should work with default config."""
        manager = SessionManager()
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        session = manager.create_from_intent(intent)
        assert session is not None
        assert manager.active_count == 1
