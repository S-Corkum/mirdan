"""Network-isolation proof: any non-localhost HTTP call from mirdan's hot paths
fails the test.

Swaps in a custom httpx transport that inspects every outgoing request and
raises if the host isn't localhost / 127.0.0.1. Runs mirdan's local-only tools
through it. If anything tries to reach the internet, the test fails loudly with
the attempted URL.

This is the runtime-side complement to test_no_external_apis.py's
schema-side rejection.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from mirdan.usecases.verify_plan import VerifyPlanUseCase


class ExternalCallDetectedError(AssertionError):
    """Raised when mirdan attempts to reach a non-localhost host."""


class LocalhostOnlyTransport(httpx.AsyncBaseTransport):
    """httpx transport that raises on any non-localhost request.

    Records every URL it sees so the test can assert what did/didn't happen.
    """

    def __init__(self) -> None:
        self.seen_urls: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.seen_urls.append(url)
        host = request.url.host or ""
        if host not in ("localhost", "127.0.0.1", "::1", ""):
            raise ExternalCallDetectedError(f"External call attempted: {url} (host={host!r})")
        return httpx.Response(200, content=b'{"ok": true}', request=request)


class LocalhostOnlySyncTransport(httpx.BaseTransport):
    def __init__(self) -> None:
        self.seen_urls: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.seen_urls.append(url)
        host = request.url.host or ""
        if host not in ("localhost", "127.0.0.1", "::1", ""):
            raise ExternalCallDetectedError(f"External call attempted: {url} (host={host!r})")
        return httpx.Response(200, content=b'{"ok": true}', request=request)


@pytest.fixture
def blocking_transport(monkeypatch):
    """Patch httpx.AsyncClient / Client to use a localhost-only transport.

    Any non-localhost HTTP attempt — sync or async — raises
    ExternalCallDetectedError before leaving the process.
    """
    async_transport = LocalhostOnlyTransport()
    sync_transport = LocalhostOnlySyncTransport()

    orig_async_init = httpx.AsyncClient.__init__
    orig_sync_init = httpx.Client.__init__

    def patched_async_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("transport", async_transport)
        return orig_async_init(self, *args, **kwargs)

    def patched_sync_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("transport", sync_transport)
        return orig_sync_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_async_init)
    monkeypatch.setattr(httpx.Client, "__init__", patched_sync_init)

    yield {"async": async_transport, "sync": sync_transport}


# ---------------------------------------------------------------------------
# The actual isolation proofs
# ---------------------------------------------------------------------------


_FLAT_PLAN = """# Plan

## Research Notes
### Files Verified
- ok

## Plan Steps

### Step 1: act
**File:** `README.md`
**Action:** Edit
**Details:** d
**Depends On:** —
**Verify:** v
**Grounding:** g
"""


@pytest.fixture
def plan_path(tmp_path: Path) -> Path:
    p = tmp_path / "p.md"
    p.write_text(_FLAT_PLAN)
    return p


class TestNetworkIsolation:
    """mirdan's plan verification must run with no non-localhost HTTP traffic."""

    def test_verify_plan_makes_no_external_calls(self, blocking_transport, plan_path):
        # verify_plan is purely mechanical — it must touch no network at all.
        VerifyPlanUseCase(project_root=plan_path.parent).execute(str(plan_path))
        assert blocking_transport["async"].seen_urls == []
        assert blocking_transport["sync"].seen_urls == []

    def test_external_host_would_be_blocked(self, blocking_transport):
        """Self-test: confirm the transport actually blocks external hosts.

        If this fails, the isolation test above is giving false assurance.
        """
        import httpx as _httpx

        with _httpx.Client() as client, pytest.raises(ExternalCallDetectedError):
            # Would attempt api.anthropic.com — must be blocked.
            client.get("https://api.anthropic.com/v1/messages")

    def test_localhost_would_be_allowed(self, blocking_transport):
        """Self-test: localhost traffic flows through (local LLM backend)."""
        import httpx as _httpx

        with _httpx.Client() as client:
            response = client.get("http://127.0.0.1:11434/api/tags")
            assert response.status_code == 200
