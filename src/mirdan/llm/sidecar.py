"""Lightweight HTTP sidecar for hook script integration."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mirdan.llm.manager import LLMManager

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


# Human-readable label map. The technical ``classification`` values stem from
# mirdan's token-optimizer framing (where "paid" = paid API spend) and confuse
# Claude Code subscribers who pay a flat fee. The ``meaning`` field is a
# UI-friendly alternative returned alongside the technical value.
_HUMAN_LABELS: dict[str, str] = {
    "local_only": "Handle locally — no cloud model needed.",
    "local_assist": "Local model assists; cloud model still needed for completion.",
    "paid_minimal": "Cloud model needed with minimal context.",
    "paid_required": "Escalate to cloud model — task too complex for local model.",
}


def _augment_with_human_label(payload: dict[str, Any]) -> None:
    """Attach a ``meaning`` field describing the classification in plain English.

    Mutates ``payload`` in place. Safe to call on any response dict — unknown
    classifications are left unchanged.
    """
    classification = str(payload.get("classification", "")).lower()
    if classification in _HUMAN_LABELS:
        payload.setdefault("meaning", _HUMAN_LABELS[classification])


class Sidecar:
    """HTTP server on localhost for hook scripts to call the warm LLM.

    Hook scripts (bash) use ``curl localhost:$PORT/triage`` to avoid
    cold-starting a CLI process. The sidecar reuses the already-loaded
    model from the MCP server, giving <5ms latency.

    Endpoints return stub responses until TriageEngine (Phase 3) and
    CheckRunner (Phase 4) are wired via LLMManager attributes.
    """

    def __init__(self, manager: LLMManager) -> None:
        self._manager = manager
        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._port: int | None = None
        # Lazy-initialized metrics collector — shared with the rest of mirdan so
        # ``mirdan llm metrics`` reflects hook-driven traffic, not just MCP calls.
        self._metrics: Any = None

    def _get_metrics(self) -> Any:
        """Return a shared TokenMetrics instance, creating it on first use."""
        if self._metrics is None:
            from mirdan.llm.metrics import TokenMetrics

            self._metrics = TokenMetrics()
        return self._metrics

    @property
    def port(self) -> int | None:
        """The port the sidecar is listening on, or None if not started."""
        return self._port

    async def start(self) -> None:
        """Start the sidecar on a random free port. Non-blocking."""
        app = Starlette(
            routes=[
                Route("/triage", self._handle_triage, methods=["POST"]),
                Route("/check", self._handle_check, methods=["POST"]),
                Route("/health", self._handle_health, methods=["GET"]),
            ]
        )

        # Bind to a random free port
        self._port = self._find_free_port()

        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self._port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._serve_task = asyncio.create_task(self._server.serve())

        # Write port file for hook scripts
        port_file = Path(".mirdan/sidecar.port")
        port_file.parent.mkdir(parents=True, exist_ok=True)
        port_file.write_text(str(self._port))
        logger.info("Sidecar started on port %d", self._port)

    async def stop(self) -> None:
        """Stop the sidecar server and remove the port file."""
        if self._server:
            self._server.should_exit = True
        if self._serve_task and not self._serve_task.done():
            self._serve_task.cancel()
            try:
                await self._serve_task
            except asyncio.CancelledError:
                pass

        port_file = Path(".mirdan/sidecar.port")
        if port_file.exists():
            try:
                port_file.unlink()
            except OSError:
                pass

        logger.info("Sidecar stopped")

    @staticmethod
    def _find_free_port() -> int:
        """Find a free port using stdlib socket."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port: int = sock.getsockname()[1]
            return port

    async def _handle_health(self, request: Request) -> JSONResponse:
        """Return LLM health state."""
        health = await self._manager.health()
        return JSONResponse(health.to_dict())

    async def _handle_triage(self, request: Request) -> JSONResponse:
        """Classify a task via TriageEngine, or return stub if not wired.

        Accepts both JSON (``{"prompt": "..."}`` from programmatic callers)
        and raw text (from hook scripts using ``curl --data-binary @-``).

        Records a metrics sample on successful classification so hook-driven
        traffic shows up in ``mirdan llm metrics``.
        """
        if self._manager.triage_engine:
            try:
                prompt = await self._read_prompt(request)
                if prompt:
                    result = await self._manager.triage_engine.classify(prompt)
                    if result:
                        payload = dict(result.to_dict())
                        _augment_with_human_label(payload)
                        self._record_triage_metric(payload)
                        return JSONResponse(payload)
            except Exception as exc:
                logger.error("Triage endpoint error: %s", exc)

        return JSONResponse(
            {
                "classification": "paid_required",
                "confidence": 0.0,
                "reasoning": "triage not configured",
            }
        )

    def _record_triage_metric(self, payload: dict[str, Any]) -> None:
        """Record a triage sample into the shared metrics store."""
        try:
            self._get_metrics().record_triage(
                classification=str(payload.get("classification", "")),
                confidence=float(payload.get("confidence", 0.0)),
            )
        except Exception as exc:  # metrics must never break the endpoint
            logger.debug("Metrics record failed: %s", exc)

    @staticmethod
    async def _read_prompt(request: Request) -> str:
        """Extract prompt text from a request body.

        Hook scripts send raw text via ``curl --data-binary @-``.
        Programmatic callers send JSON ``{"prompt": "..."}``.
        Handles both transparently.
        """
        body = await request.body()
        if not body:
            return ""

        # Try JSON first (programmatic callers)
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                return str(data.get("prompt", ""))
        except (json.JSONDecodeError, ValueError):
            pass

        # Raw text from hook script
        return body.decode("utf-8", errors="replace").strip()

    async def _handle_check(self, request: Request) -> JSONResponse:
        """Run checks via CheckRunner, or return stub if not wired."""
        if self._manager.check_runner:
            try:
                result = await self._manager.check_runner.run_all()
                return JSONResponse(result.to_dict())
            except Exception as exc:
                logger.error("Check endpoint error: %s", exc)

        return JSONResponse(
            {
                "all_pass": True,
                "summary": "check runner not configured",
            }
        )
