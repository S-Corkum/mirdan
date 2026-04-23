"""Network-isolation proof: any non-localhost HTTP call from mirdan's hot paths
fails the test.

Swaps in a custom httpx transport that inspects every outgoing request and
raises if the host isn't localhost / 127.0.0.1. Runs the pipeline's MCP
tools through it. If anything tries to reach the internet, the test fails
loudly with the attempted URL.

This is the runtime-side complement to test_no_external_apis.py's
schema-side rejection.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from mirdan.usecases.propose_subtask_diff import ProposeSubtaskDiffUseCase
from mirdan.usecases.validate_brief import ValidateBriefUseCase
from mirdan.usecases.verify_plan_against_brief import VerifyPlanAgainstBriefUseCase


class ExternalCallDetectedError(AssertionError):
    """Raised when mirdan attempts to reach a non-localhost host."""


class LocalhostOnlyTransport(httpx.AsyncBaseTransport):
    """httpx transport that raises on any non-localhost request.

    Records every URL it sees so the test can assert what did/didn't happen.
    """

    def __init__(self) -> None:
        self.seen_urls: list[str] = []

    async def handle_async_request(
        self, request: httpx.Request
    ) -> httpx.Response:
        url = str(request.url)
        self.seen_urls.append(url)
        host = request.url.host or ""
        if host not in ("localhost", "127.0.0.1", "::1", ""):
            raise ExternalCallDetectedError(
                f"External call attempted: {url} (host={host!r})"
            )
        # Return a stub 200 so code that tests for failure still works.
        return httpx.Response(
            200,
            content=b'{"ok": true}',
            request=request,
        )


class LocalhostOnlySyncTransport(httpx.BaseTransport):
    def __init__(self) -> None:
        self.seen_urls: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.seen_urls.append(url)
        host = request.url.host or ""
        if host not in ("localhost", "127.0.0.1", "::1", ""):
            raise ExternalCallDetectedError(
                f"External call attempted: {url} (host={host!r})"
            )
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


_MIN_BRIEF = """# Brief

## Outcome
x

## Users & Scenarios
y

## Business Acceptance Criteria
- [ ] a
- [ ] b
- [ ] c

## Constraints
- x

## Out of Scope
- y
"""


_MIN_PLAN = """---
brief: /tmp/b.md
---

# Plan

## Research Notes
ok

## Epic Layer
e

## Story Layer

### Story 1 — t
- **As** u
- **I want** x
- **So that** y

**Acceptance Criteria:**
- [ ] a

#### Subtasks
##### 1.1 — act
**File:** x
**Action:** Edit
**Details:** d
**Depends on:** —
**Verify:** v
**Grounding:** g
"""


@pytest.fixture
def brief_path(tmp_path: Path) -> Path:
    p = tmp_path / "b.md"
    p.write_text(_MIN_BRIEF)
    return p


@pytest.fixture
def plan_path(tmp_path: Path) -> Path:
    p = tmp_path / "p.md"
    p.write_text(_MIN_PLAN)
    return p


class TestNetworkIsolation:
    """Each hot-path usecase must run with no non-localhost HTTP traffic."""

    @pytest.mark.asyncio
    async def test_validate_brief_makes_no_external_calls(
        self, blocking_transport, brief_path
    ):
        await ValidateBriefUseCase().execute(str(brief_path))
        # If it attempted an external host, LocalhostOnlyTransport would have
        # raised. We additionally assert no URLs were seen at all — the
        # validator shouldn't talk to anything.
        assert blocking_transport["async"].seen_urls == []
        assert blocking_transport["sync"].seen_urls == []

    @pytest.mark.asyncio
    async def test_verify_plan_mechanical_makes_no_external_calls(
        self, blocking_transport, brief_path, plan_path
    ):
        # llm_manager=None → mechanical path only, no LLM backend touched.
        await VerifyPlanAgainstBriefUseCase(llm_manager=None).execute(
            str(plan_path), str(brief_path)
        )
        assert blocking_transport["async"].seen_urls == []
        assert blocking_transport["sync"].seen_urls == []

    @pytest.mark.asyncio
    async def test_propose_subtask_diff_no_llm_makes_no_external_calls(
        self, blocking_transport
    ):
        # Fail-closed path: LLM unavailable → synchronous halt, no HTTP.
        result = await ProposeSubtaskDiffUseCase(llm_manager=None).execute(
            subtask_yaml="File: x.py\nAction: Edit\n",
            file_context={"x.py": "# content\n"},
        )
        assert result["halted"] is True
        assert blocking_transport["async"].seen_urls == []
        assert blocking_transport["sync"].seen_urls == []

    def test_external_host_would_be_blocked(self, blocking_transport):
        """Self-test: confirm the transport actually blocks external hosts.

        If this fails, the isolation tests above are giving false assurance.
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
