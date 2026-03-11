"""Tests for AgentCoordinator — multi-agent file claim tracking."""

from __future__ import annotations

import pytest

from mirdan.config import CoordinationConfig
from mirdan.core.agent_coordinator import AgentCoordinator


@pytest.fixture()
def coordinator() -> AgentCoordinator:
    """Standard enabled coordinator."""
    return AgentCoordinator(CoordinationConfig(enabled=True))


@pytest.fixture()
def disabled_coordinator() -> AgentCoordinator:
    """Disabled coordinator."""
    return AgentCoordinator(CoordinationConfig(enabled=False))


class TestClaimFiles:
    def test_claim_single_file_no_conflict(self, coordinator: AgentCoordinator) -> None:
        warnings = coordinator.claim_files("s1", ["file.py"], "write")
        assert len(warnings) == 0
        claims = coordinator.get_active_claims()
        assert "file.py" in claims
        assert len(claims["file.py"]) == 1

    def test_claim_write_overlap_warns(self, coordinator: AgentCoordinator) -> None:
        coordinator.claim_files("s1", ["file.py"], "write")
        warnings = coordinator.claim_files("s2", ["file.py"], "write")
        assert len(warnings) == 1
        assert warnings[0].type == "write_overlap"
        assert "s1" in warnings[0].conflicting_sessions

    def test_claim_read_no_overlap(self, coordinator: AgentCoordinator) -> None:
        coordinator.claim_files("s1", ["file.py"], "read")
        warnings = coordinator.claim_files("s2", ["file.py"], "read")
        assert len(warnings) == 0

    def test_claim_write_then_read_warns_stale(self, coordinator: AgentCoordinator) -> None:
        coordinator.claim_files("s1", ["file.py"], "read")
        warnings = coordinator.claim_files("s2", ["file.py"], "write")
        assert any(w.type == "stale_read" for w in warnings)

    def test_same_session_no_self_conflict(self, coordinator: AgentCoordinator) -> None:
        coordinator.claim_files("s1", ["file.py"], "write")
        warnings = coordinator.claim_files("s1", ["file.py"], "write")
        assert len(warnings) == 0

    def test_multiple_files(self, coordinator: AgentCoordinator) -> None:
        warnings = coordinator.claim_files("s1", ["a.py", "b.py", "c.py"], "write")
        assert len(warnings) == 0
        claims = coordinator.get_active_claims()
        assert len(claims) == 3

    def test_disabled_config_no_claims(self, disabled_coordinator: AgentCoordinator) -> None:
        warnings = disabled_coordinator.claim_files("s1", ["file.py"], "write")
        assert len(warnings) == 0
        claims = disabled_coordinator.get_active_claims()
        assert len(claims) == 0


class TestReleaseSession:
    def test_release_clears_claims(self, coordinator: AgentCoordinator) -> None:
        coordinator.claim_files("s1", ["a.py", "b.py"], "write")
        coordinator.release_session("s1")
        claims = coordinator.get_active_claims()
        assert len(claims) == 0

    def test_release_preserves_other_sessions(self, coordinator: AgentCoordinator) -> None:
        coordinator.claim_files("s1", ["a.py"], "write")
        coordinator.claim_files("s2", ["b.py"], "write")
        coordinator.release_session("s1")
        claims = coordinator.get_active_claims()
        assert "a.py" not in claims
        assert "b.py" in claims

    def test_release_nonexistent_session(self, coordinator: AgentCoordinator) -> None:
        # Should not raise
        coordinator.release_session("nonexistent")


class TestCheckConflicts:
    def test_returns_warnings_on_conflict(self, coordinator: AgentCoordinator) -> None:
        coordinator.claim_files("s1", ["file.py"], "write")
        warnings = coordinator.check_conflicts("s2", "file.py")
        assert len(warnings) == 1
        assert warnings[0].type == "stale_read"

    def test_no_conflict_same_session(self, coordinator: AgentCoordinator) -> None:
        coordinator.claim_files("s1", ["file.py"], "write")
        warnings = coordinator.check_conflicts("s1", "file.py")
        assert len(warnings) == 0

    def test_no_conflict_unclaimed_file(self, coordinator: AgentCoordinator) -> None:
        warnings = coordinator.check_conflicts("s1", "unclaimed.py")
        assert len(warnings) == 0

    def test_disabled_returns_empty(self, disabled_coordinator: AgentCoordinator) -> None:
        warnings = disabled_coordinator.check_conflicts("s1", "file.py")
        assert len(warnings) == 0


class TestCleanupStale:
    def test_removes_dead_sessions(self, coordinator: AgentCoordinator) -> None:
        coordinator.claim_files("s1", ["a.py"], "write")
        coordinator.claim_files("s2", ["b.py"], "write")
        removed = coordinator.cleanup_stale({"s1"})
        assert removed == 1
        claims = coordinator.get_active_claims()
        assert "a.py" in claims
        assert "b.py" not in claims

    def test_keeps_active_sessions(self, coordinator: AgentCoordinator) -> None:
        coordinator.claim_files("s1", ["a.py"], "write")
        removed = coordinator.cleanup_stale({"s1"})
        assert removed == 0


class TestConfigFlags:
    def test_warn_on_write_overlap_false_suppresses(self) -> None:
        config = CoordinationConfig(enabled=True, warn_on_write_overlap=False)
        coord = AgentCoordinator(config)
        coord.claim_files("s1", ["file.py"], "write")
        warnings = coord.claim_files("s2", ["file.py"], "write")
        overlap_warnings = [w for w in warnings if w.type == "write_overlap"]
        assert len(overlap_warnings) == 0

    def test_warn_on_stale_read_false_suppresses(self) -> None:
        config = CoordinationConfig(enabled=True, warn_on_stale_read=False)
        coord = AgentCoordinator(config)
        coord.claim_files("s1", ["file.py"], "read")
        warnings = coord.claim_files("s2", ["file.py"], "write")
        stale_warnings = [w for w in warnings if w.type == "stale_read"]
        assert len(stale_warnings) == 0


class TestAgentLabel:
    def test_label_in_warning_message(self, coordinator: AgentCoordinator) -> None:
        coordinator.claim_files("s1", ["file.py"], "write", agent_label="Agent-A")
        warnings = coordinator.claim_files("s2", ["file.py"], "write")
        assert len(warnings) == 1
        assert "Agent-A" in warnings[0].message
