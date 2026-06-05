"""Enforcement tests: mirdan never calls external LLM APIs (local-LLM-only).

Two layers of enforcement:
1. Static — no module under ``src/mirdan`` references an external LLM API host
   (the local backends talk only to localhost / in-process llama.cpp).
2. Schema — mirdan's Pydantic config rejects ``*_api_key`` fields so users can't
   accidentally wire an external provider.

These are the architectural teeth behind mirdan's "local LLM only" contract.
They must pass for every release.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mirdan.config import ConfigError, MirdanConfig

# External LLM API hosts mirdan must never reach. The local backends use
# localhost (ollama) or run in-process (llama.cpp), so none of these may appear
# anywhere in the shipped source.
_EXTERNAL_LLM_HOSTS = [
    "api.anthropic.com",
    "api.openai.com",
    "openai.azure.com",
    "generativelanguage.googleapis.com",
    "api.cohere.ai",
    "api.cohere.com",
]

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "mirdan"


# ---------------------------------------------------------------------------
# Static — no external LLM API host appears in the source
# ---------------------------------------------------------------------------


class TestNoExternalLLMAPIs:
    """No shipped module may reference an external LLM API host."""

    def test_no_external_llm_api_hosts_in_source(self) -> None:
        offenders: list[str] = []
        for py in _SRC_ROOT.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            offenders.extend(
                f"{py.relative_to(_SRC_ROOT)}: {host}"
                for host in _EXTERNAL_LLM_HOSTS
                if host in text
            )
        assert not offenders, (
            f"External LLM API host(s) referenced in source — mirdan is local-only: {offenders}"
        )

    def test_httpx_clients_target_localhost(self) -> None:
        """The only httpx client in the LLM layer (ollama) defaults to localhost."""
        import inspect

        from mirdan.llm.ollama import OllamaBackend

        # The default base URL must be a loopback address.
        src = inspect.getsource(OllamaBackend.__init__)
        assert "localhost" in src or "127.0.0.1" in src, (
            "OllamaBackend default base_url is not loopback"
        )


# ---------------------------------------------------------------------------
# Schema — config rejects API-key fields
# ---------------------------------------------------------------------------


class TestConfigRejectsAPIKeys:
    """Mirdan config must not accept API-key fields for external providers."""

    @pytest.mark.parametrize(
        "api_key_field",
        ["anthropic_api_key", "openai_api_key", "google_api_key"],
    )
    def test_mirdan_config_rejects_api_key_fields_in_yaml(
        self, api_key_field: str, tmp_path: Path
    ) -> None:
        """MirdanConfig loaded from YAML must reject top-level *_api_key fields."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({api_key_field: "should-not-be-accepted"}))

        with pytest.raises((ConfigError, Exception)) as exc_info:
            MirdanConfig.load(cfg)
        msg = str(exc_info.value).lower()
        assert "extra" in msg or api_key_field in msg or "forbid" in msg
