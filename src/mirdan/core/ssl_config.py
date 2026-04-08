"""SSL and network configuration for corporate environments.

Handles SSL certificate configuration for users in corporate networks
with SSL inspection (Netskope, Zscaler, BlueCoat) that inject enterprise
CA certificates into the SSL chain.

When enyal is co-installed, reuses its battle-tested SSL configuration.
Otherwise provides a lightweight standalone equivalent.

Resolution order:
    1. Reuse enyal's SSL config if available (zero duplication)
    2. truststore package → use OS native trust store (recommended)
    3. MIRDAN_SSL_CERT_FILE → explicit CA bundle
    4. SSL_CERT_FILE / REQUESTS_CA_BUNDLE → standard env vars
    5. Default Python certifi bundle

Environment Variables:
    MIRDAN_SSL_CERT_FILE: Path to corporate CA certificate bundle.
    MIRDAN_HF_ENDPOINT: Custom HuggingFace Hub URL (e.g., Artifactory proxy).
    MIRDAN_HF_TOKEN: Auth token for HuggingFace Hub or Artifactory proxy.
    MIRDAN_OFFLINE_MODE: Skip all network downloads (default: "false").
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_configured = False


def configure_ssl() -> None:
    """Configure SSL for corporate environments. Safe to call multiple times.

    Tries to reuse enyal's SSL config if available (both tools run on the
    same machine, so enyal's corporate network config applies to mirdan too).
    Falls back to standalone configuration.
    """
    global _configured
    if _configured:
        return
    _configured = True

    # Try reusing enyal's comprehensive SSL config first
    if _try_enyal_ssl():
        return

    # Standalone: configure from mirdan's own env vars
    _configure_standalone()


def _try_enyal_ssl() -> bool:
    """Try to reuse enyal's SSL configuration if co-installed.

    Returns True if enyal handled it, False to fall back to standalone.
    """
    try:
        from enyal.core.ssl_config import (  # type: ignore[import-not-found]
            configure_ssl_environment,
            get_ssl_config,
        )

        config = get_ssl_config()
        configure_ssl_environment(config)
        logger.info("Using enyal's SSL configuration")
        return True
    except ImportError:
        return False
    except Exception:
        logger.debug("enyal SSL config failed, falling back to standalone")
        return False


def _configure_standalone() -> None:
    """Standalone SSL configuration without enyal.

    Handles:
    1. truststore injection (OS native trust store — best for Netskope)
    2. Custom HuggingFace endpoint (Artifactory proxy)
    3. Custom CA cert bundle
    4. Offline mode
    """
    # 1. Try truststore — single best solution for corporate SSL inspection.
    # It monkey-patches Python's ssl module to use the OS trust store,
    # which includes any CA certs installed by Netskope/Zscaler.
    try:
        import truststore  # type: ignore[import-not-found,unused-ignore]

        truststore.inject_into_ssl()
        logger.info("Using OS native trust store via truststore")
    except ImportError:
        logger.debug("truststore not installed — using default cert handling")
    except Exception:
        logger.debug("truststore injection failed")

    # 2. Custom CA cert bundle
    cert_file = (
        os.environ.get("MIRDAN_SSL_CERT_FILE")
        or os.environ.get("ENYAL_SSL_CERT_FILE")
        or os.environ.get("SSL_CERT_FILE")
    )
    if cert_file and Path(cert_file).is_file():
        os.environ["SSL_CERT_FILE"] = cert_file
        os.environ["REQUESTS_CA_BUNDLE"] = cert_file
        logger.info("Using CA bundle: %s", cert_file)

    # 3. Custom HuggingFace endpoint (Artifactory proxy)
    hf_endpoint = (
        os.environ.get("MIRDAN_HF_ENDPOINT")
        or os.environ.get("ENYAL_HF_ENDPOINT")
        or os.environ.get("HF_ENDPOINT")
    )
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint.rstrip("/")
        logger.info("Using custom HuggingFace endpoint: %s", hf_endpoint)

    # 4. Offline mode
    if _is_offline_mode():
        logger.info("Offline mode — no network downloads")


def is_corporate_network() -> bool:
    """Detect if running in a corporate network environment.

    Checks for indicators that suggest Netskope/Zscaler/corporate proxy.
    """
    indicators = [
        os.environ.get("MIRDAN_SSL_CERT_FILE"),
        os.environ.get("ENYAL_SSL_CERT_FILE"),
        os.environ.get("MIRDAN_HF_ENDPOINT"),
        os.environ.get("ENYAL_HF_ENDPOINT"),
        os.environ.get("HTTPS_PROXY"),
        os.environ.get("https_proxy"),
    ]
    if any(indicators):
        return True

    # Check for Netskope on macOS
    return Path("/Library/Application Support/Netskope").exists()


def _is_offline_mode() -> bool:
    """Check if offline mode is enabled."""
    val = os.environ.get("MIRDAN_OFFLINE_MODE", "false").lower()
    return val in ("true", "1", "yes")


def get_hf_endpoint() -> str | None:
    """Get the configured HuggingFace endpoint (Artifactory proxy or default)."""
    return (
        os.environ.get("MIRDAN_HF_ENDPOINT")
        or os.environ.get("ENYAL_HF_ENDPOINT")
        or os.environ.get("HF_ENDPOINT")
    )


def get_hf_token() -> str | None:
    """Get the HuggingFace / Artifactory auth token.

    Resolution order:
        1. MIRDAN_HF_TOKEN — mirdan-specific token
        2. ENYAL_HF_TOKEN — enyal-specific token (shared machine)
        3. HF_TOKEN — standard HuggingFace token
    """
    return (
        os.environ.get("MIRDAN_HF_TOKEN")
        or os.environ.get("ENYAL_HF_TOKEN")
        or os.environ.get("HF_TOKEN")
    )
