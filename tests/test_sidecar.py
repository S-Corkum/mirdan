"""Tests for HTTP Sidecar endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.routing import Route

from mirdan.llm.sidecar import Sidecar
from mirdan.models import HealthState, LLMHealth


def _make_sidecar(
    triage_engine: Any = None,
    check_runner: Any = None,
) -> Sidecar:
    """Create a Sidecar with a mock manager."""
    manager = AsyncMock()
    manager.triage_engine = triage_engine
    manager.check_runner = check_runner
    manager.health.return_value = LLMHealth(
        state=HealthState.AVAILABLE,
        models_loaded=["gemma4-e2b-q4"],
        hardware_profile="standard",
    )
    return Sidecar(manager)


def _make_test_app(sidecar: Sidecar) -> Starlette:
    """Build the starlette app that Sidecar would normally create."""
    return Starlette(
        routes=[
            Route("/triage", sidecar._handle_triage, methods=["POST"]),
            Route("/check", sidecar._handle_check, methods=["POST"]),
            Route("/health", sidecar._handle_health, methods=["GET"]),
        ]
    )


class TestSidecarHealth:
    """Tests for GET /health."""

    def test_returns_health_json(self) -> None:
        sidecar = _make_sidecar()
        app = _make_test_app(sidecar)
        client = TestClient(app)

        resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "available"
        assert "gemma4-e2b-q4" in data["models_loaded"]


class TestSidecarTriage:
    """Tests for POST /triage."""

    def test_returns_stub_when_no_engine(self) -> None:
        sidecar = _make_sidecar(triage_engine=None)
        app = _make_test_app(sidecar)
        client = TestClient(app)

        resp = client.post("/triage", json={"prompt": "fix the bug"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["classification"] == "paid_required"
        assert data["confidence"] == 0.0
        assert data["reasoning"] == "triage not configured"

    def test_calls_engine_when_wired(self) -> None:
        mock_engine = AsyncMock()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "classification": "local_only",
            "confidence": 0.9,
            "reasoning": "simple task",
        }
        mock_engine.classify.return_value = mock_result

        sidecar = _make_sidecar(triage_engine=mock_engine)
        app = _make_test_app(sidecar)
        client = TestClient(app)

        resp = client.post("/triage", json={"prompt": "fix unused import"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["classification"] == "local_only"

    def test_returns_stub_on_engine_error(self) -> None:
        mock_engine = AsyncMock()
        mock_engine.classify.side_effect = RuntimeError("engine crashed")

        sidecar = _make_sidecar(triage_engine=mock_engine)
        app = _make_test_app(sidecar)
        client = TestClient(app)

        resp = client.post("/triage", json={"prompt": "test"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["classification"] == "paid_required"


class TestSidecarCheck:
    """Tests for POST /check."""

    def test_returns_stub_when_no_runner(self) -> None:
        sidecar = _make_sidecar(check_runner=None)
        app = _make_test_app(sidecar)
        client = TestClient(app)

        resp = client.post("/check", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert data["all_pass"] is True
        assert data["summary"] == "check runner not configured"

    def test_calls_runner_when_wired(self) -> None:
        mock_runner = AsyncMock()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "all_pass": False,
            "summary": "3 lint errors",
        }
        mock_runner.run_all.return_value = mock_result

        sidecar = _make_sidecar(check_runner=mock_runner)
        app = _make_test_app(sidecar)
        client = TestClient(app)

        resp = client.post("/check", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert data["all_pass"] is False


class TestSidecarPortDiscovery:
    """Tests for port allocation and file writing."""

    def test_find_free_port_returns_int(self) -> None:
        port = Sidecar._find_free_port()
        assert isinstance(port, int)
        assert port > 0

    def test_find_free_port_returns_different_ports(self) -> None:
        ports = {Sidecar._find_free_port() for _ in range(5)}
        # Should get at least 2 different ports (OS may reuse, but very unlikely for all 5)
        assert len(ports) >= 2
