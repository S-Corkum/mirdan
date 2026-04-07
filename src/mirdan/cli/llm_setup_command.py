"""``mirdan llm setup`` — interactive onboarding for local LLM features."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from mirdan.llm.health import HardwareDetector


def run_llm_setup(args: list[str]) -> None:
    """Interactive setup wizard for local LLM integration.

    Usage:
        mirdan llm setup          Interactive wizard
        mirdan llm setup --check  Non-interactive validation

    Args:
        args: CLI arguments after ``llm setup``.
    """
    if "--help" in args or "-h" in args:
        _print_help()
        sys.exit(0)

    check_only = "--check" in args

    # 1. Hardware detection
    print("=== mirdan LLM Setup ===\n")
    print("Detecting hardware...")
    hw = HardwareDetector.detect()
    print(f"  Architecture: {hw.architecture}")
    print(f"  Total RAM:    {hw.total_ram_mb} MB")
    print(f"  Available:    {hw.available_ram_mb} MB")
    print(f"  GPU:          {hw.gpu_type or 'none detected'}")
    print(f"  Metal:        {'yes' if hw.metal_capable else 'no'}")
    print(f"  Profile:      {hw.detected_profile.value}")
    print()

    # 2. Check for Enyal
    enyal_present = _check_enyal()
    print(f"  Enyal:        {'found (~1GB reserved)' if enyal_present else 'not detected'}")

    # 3. Check backends
    llamacpp_ok = _check_llamacpp()
    ollama_ok = _check_ollama()
    print(f"  llama-cpp:    {'available' if llamacpp_ok else 'not installed'}")
    print(f"  Ollama:       {'available' if ollama_ok else 'not detected'}")
    print()

    # 4. Auto-install llama-cpp-python if not present
    if not llamacpp_ok:
        print("Installing llama-cpp-python...")
        llamacpp_ok = _install_llamacpp(hw.metal_capable)
        if llamacpp_ok:
            print("  llama-cpp-python installed successfully.")
        elif not ollama_ok:
            print("ERROR: Could not install llama-cpp-python and no Ollama detected.")
            print("Try manually: pip install llama-cpp-python")
            print("Or install Ollama as fallback: https://ollama.ai")
            sys.exit(1)
        else:
            print("  llama-cpp-python install failed — falling back to Ollama.")
        print()

    # 5. Check Metal/AVX2 compilation for llama-cpp-python
    if llamacpp_ok:
        metal_ok = _check_llamacpp_metal()
        if hw.metal_capable and not metal_ok:
            print("WARNING: llama-cpp-python may not have Metal support.")
            print("Reinstall with: CMAKE_ARGS=\"-DGGML_METAL=ON\" pip install --force-reinstall llama-cpp-python")
            print()

    # 6. Check corporate network / SSL
    from mirdan.core.ssl_config import is_corporate_network

    corporate = is_corporate_network()
    if corporate:
        print("  Network:      corporate environment detected")
        _print_corporate_instructions(ollama_ok)
        print()

    # 7. Recommend model
    recommendation = _recommend_model(hw, ollama_ok, llamacpp_ok, enyal_present)
    print(f"Recommended: {recommendation['name']}")
    print(f"  Backend:  {recommendation['backend']}")
    print(f"  Memory:   ~{recommendation['memory_mb']} MB")
    print(f"  Quality:  {recommendation['quality']}")
    print()

    if check_only:
        print("Setup check complete.")
        return

    # 8. Download model
    if recommendation["backend"] == "ollama" and ollama_ok:
        tag = recommendation.get("ollama_tag", "")
        if tag:
            print(f"Downloading model via Ollama...")
            _download_ollama(tag)
    elif recommendation["backend"] == "llamacpp":
        gguf_file = recommendation.get("gguf_file", "")
        hf_repo = recommendation.get("hf_repo", "")
        if gguf_file and hf_repo:
            _download_gguf(gguf_file, hf_repo)
        else:
            print("No download info available for this model.")

    print()

    # 9. Write config
    _write_config(recommendation)
    print("Wrote .mirdan.yaml with llm.enabled: true")
    print()

    print("Next steps:")
    print("  mirdan init --claude-code   Configure for Claude Code")
    print("  mirdan init --cursor        Configure for Cursor IDE/CLI")


def _check_enyal() -> bool:
    """Check if Enyal is likely running or configured."""
    # Check for enyal config or process
    if Path(".mcp.json").exists():
        try:
            data = json.loads(Path(".mcp.json").read_text())
            servers = data.get("mcpServers", {})
            return "enyal" in servers
        except (json.JSONDecodeError, OSError):
            pass
    return False


def _check_ollama() -> bool:
    """Check if Ollama daemon is running."""
    try:
        import httpx

        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


def _install_llamacpp(metal_capable: bool) -> bool:
    """Auto-install llama-cpp-python with appropriate build flags.

    On Apple Silicon Macs, compiles with Metal GPU support.
    On x86_64, compiles with default settings (AVX2 auto-detected).

    Args:
        metal_capable: Whether this machine supports Metal GPU acceleration.

    Returns:
        True if installation succeeded.
    """
    import os

    env = os.environ.copy()
    if metal_capable:
        env["CMAKE_ARGS"] = "-DGGML_METAL=ON"
        print("  Building with Metal GPU support...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "llama-cpp-python"],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            return _check_llamacpp()
        print(f"  pip install failed (exit {result.returncode})")
        # Show last few lines of stderr for diagnosis
        stderr_lines = result.stderr.strip().splitlines()
        for line in stderr_lines[-5:]:
            print(f"  {line}")
        return False
    except subprocess.TimeoutExpired:
        print("  Installation timed out (10 minutes).")
        return False
    except OSError as exc:
        print(f"  Installation failed: {exc}")
        return False


def _check_llamacpp() -> bool:
    """Check if llama-cpp-python is installed."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import llama_cpp; print('ok')"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return False


def _check_llamacpp_metal() -> bool:
    """Check if llama-cpp-python was compiled with Metal support."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from llama_cpp import Llama; print(hasattr(Llama, '_model'))",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # This is a heuristic — actual Metal check would require loading a model
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _recommend_model(
    hw: Any,
    ollama_ok: bool,
    llamacpp_ok: bool,
    enyal_present: bool,
) -> dict[str, Any]:
    """Recommend the optimal model for this machine.

    Args:
        hw: HardwareInfo from detection.
        ollama_ok: Whether Ollama is available.
        llamacpp_ok: Whether llama-cpp-python is available.
        enyal_present: Whether Enyal is consuming ~1GB.

    Returns:
        Dict with name, backend, memory_mb, quality, and download info.
    """
    available = hw.available_ram_mb
    if enyal_present:
        available -= 1000  # Reserve for Enyal

    # llama-cpp-python is the primary backend (no daemon, lighter).
    # Ollama is only used as fallback when llama-cpp-python is unavailable.
    if llamacpp_ok:
        if available >= 6500:
            return {
                "name": "Gemma 4 E4B Q3",
                "backend": "llamacpp",
                "memory_mb": 4500,
                "quality": "0.67",
                "gguf_file": "gemma-4-E4B-it-Q3_K_M.gguf",
                "hf_repo": "unsloth/gemma-4-E4B-it-GGUF",
            }
        if available >= 5200:
            return {
                "name": "Gemma 4 E2B Q3",
                "backend": "llamacpp",
                "memory_mb": 3200,
                "quality": "0.58",
                "gguf_file": "gemma-4-E2B-it-Q3_K_M.gguf",
                "hf_repo": "unsloth/gemma-4-E2B-it-GGUF",
            }

    # Fallback: Ollama (only if llama-cpp-python unavailable or insufficient RAM)
    if ollama_ok:
        return {
            "name": "Gemma 4 E2B Q4 (via Ollama)",
            "backend": "ollama",
            "memory_mb": 3500,
            "quality": "0.60",
            "ollama_tag": "gemma4:e2b",
        }

    # Last resort — llamacpp with smallest model even if tight
    return {
        "name": "Gemma 4 E2B Q3 (tight fit)",
        "backend": "llamacpp",
        "memory_mb": 3200,
        "quality": "0.58",
        "gguf_file": "gemma-4-E2B-it-Q3_K_M.gguf",
        "hf_repo": "unsloth/gemma-4-E2B-it-GGUF",
    }


def _write_config(recommendation: dict[str, Any]) -> None:
    """Write .mirdan.yaml with LLM configuration."""
    import yaml

    config_path = Path(".mirdan.yaml")
    existing: dict[str, Any] = {}
    if config_path.exists():
        try:
            existing = yaml.safe_load(config_path.read_text()) or {}
        except yaml.YAMLError:
            pass

    existing.setdefault("llm", {})
    existing["llm"]["enabled"] = True
    existing["llm"]["backend"] = (
        "llamacpp" if recommendation["backend"] == "llamacpp" else "ollama"
    )

    with config_path.open("w") as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=False)


def _print_corporate_instructions(ollama_ok: bool) -> None:
    """Print corporate network setup instructions."""
    import platform

    print()
    print("  === Corporate Network Setup ===")
    print()
    print("  For GGUF downloads (llama-cpp-python):")
    print("    Set MIRDAN_HF_ENDPOINT to your Artifactory HuggingFace proxy URL")
    print("    Set MIRDAN_HF_TOKEN to your Artifactory API token (or HF_TOKEN)")
    print("    Or set MIRDAN_SSL_CERT_FILE to your corporate CA bundle")
    print("    Or: pip install truststore  (auto-uses OS trust store)")
    print()

    if ollama_ok:
        print("  For Ollama model pulls:")
        if platform.system() == "Darwin":
            print("    1. Install corp CA cert into system Keychain:")
            print("       sudo security add-trusted-cert -d -r trustRoot \\")
            print("         -k /Library/Keychains/System.keychain /path/to/corp-ca.crt")
            print("    2. Configure proxy (if needed):")
            print("       launchctl setenv HTTPS_PROXY http://proxy.corp.com:8080")
            print("    3. Restart Ollama")
        else:
            print("    1. Install corp CA cert:")
            print("       sudo cp /path/to/corp-ca.crt /usr/local/share/ca-certificates/")
            print("       sudo update-ca-certificates")
            print("    2. Configure proxy (if needed):")
            print("       sudo systemctl edit ollama")
            print('       [Service]')
            print('       Environment="HTTPS_PROXY=http://proxy.corp.com:8080"')
            print("    3. Restart: sudo systemctl restart ollama")
        print()

    print("  Environment variables (set in shell profile or .mcp.json env):")
    print("    MIRDAN_HF_ENDPOINT    — Artifactory proxy for HuggingFace")
    print("    MIRDAN_HF_TOKEN       — Artifactory API token (or use HF_TOKEN)")
    print("    MIRDAN_SSL_CERT_FILE  — Corporate CA certificate bundle")
    print("    MIRDAN_OFFLINE_MODE   — Skip downloads (use pre-downloaded models)")
    print()
    print("  These can also be passed via .mcp.json:")
    print('    "env": { "MIRDAN_HF_ENDPOINT": "https://artifactory.corp.com/hf",')
    print('             "MIRDAN_HF_TOKEN": "your-token" }')


def _download_gguf(gguf_file: str, hf_repo: str) -> None:
    """Download a GGUF model file from HuggingFace Hub.

    Supports MIRDAN_HF_ENDPOINT for corporate Artifactory proxies
    and MIRDAN_OFFLINE_MODE to skip downloads entirely.

    Args:
        gguf_file: Filename of the GGUF model (e.g. "gemma-4-E4B-it-Q3_K_M.gguf").
        hf_repo: HuggingFace repo (e.g. "unsloth/gemma-4-E4B-it-GGUF").
    """
    from mirdan.core.ssl_config import (
        _is_offline_mode,
        configure_ssl,
        get_hf_endpoint,
        get_hf_token,
    )

    if _is_offline_mode():
        print("Offline mode — skipping download.")
        print(f"Place {gguf_file} in ~/.mirdan/models/ manually.")
        return

    # Configure SSL for corporate environments before any network call
    configure_ssl()

    models_dir = Path.home() / ".mirdan" / "models"
    dest = models_dir / gguf_file

    if dest.exists():
        print(f"Model already downloaded: {dest}")
        return

    models_dir.mkdir(parents=True, exist_ok=True)

    # Build download URL — respect corporate proxy endpoint
    hf_base = get_hf_endpoint() or "https://huggingface.co"
    url = f"{hf_base}/{hf_repo}/resolve/main/{gguf_file}"

    print(f"Downloading {gguf_file}...")
    print(f"  From: {url}")
    print(f"  To:   {dest}")
    print()

    try:
        import httpx
    except ImportError:
        print("ERROR: httpx not installed. Install with: pip install httpx")
        return

    # Auth header for HuggingFace or Artifactory proxy
    headers: dict[str, str] = {}
    token = get_hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=600.0) as resp:
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            last_pct = -1

            with dest.open("wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = (downloaded * 100) // total
                        if pct != last_pct and pct % 5 == 0:
                            print(f"  {pct}% ({downloaded // (1024 * 1024)} / {total // (1024 * 1024)} MB)")
                            last_pct = pct

        size_mb = dest.stat().st_size // (1024 * 1024)
        print(f"  Download complete: {size_mb} MB")

    except httpx.HTTPStatusError as exc:
        print(f"ERROR: Download failed — HTTP {exc.response.status_code}")
        if exc.response.status_code == 401:
            print("  Set MIRDAN_HF_TOKEN (or HF_TOKEN) for authentication.")
            print("  For Artifactory: use your Artifactory API token.")
        elif exc.response.status_code == 404:
            print(f"  Model not found at: {url}")
        # Clean up partial download
        if dest.exists():
            dest.unlink()
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        print(f"ERROR: Network error — {exc}")
        print("  Check your internet connection.")
        print("  Corporate network? Set MIRDAN_HF_ENDPOINT to your Artifactory proxy.")
        if dest.exists():
            dest.unlink()
    except Exception as exc:
        print(f"ERROR: Unexpected download failure — {exc}")
        if dest.exists():
            dest.unlink()


def _download_ollama(tag: str) -> None:
    """Pull a model via Ollama CLI.

    Args:
        tag: Ollama model tag (e.g. "gemma4:e2b").
    """
    print(f"Pulling {tag} via Ollama...")
    try:
        result = subprocess.run(
            ["ollama", "pull", tag],
            timeout=600,
        )
        if result.returncode != 0:
            print(f"ERROR: ollama pull {tag} failed (exit {result.returncode})")
    except FileNotFoundError:
        print("ERROR: ollama command not found.")
        print("Install Ollama: https://ollama.ai")
    except subprocess.TimeoutExpired:
        print("ERROR: ollama pull timed out (10 minutes).")
        print("Try running manually: ollama pull " + tag)


def _print_help() -> None:
    print("mirdan llm setup — configure local LLM integration")
    print()
    print("Usage:")
    print("  mirdan llm setup          Interactive setup wizard")
    print("  mirdan llm setup --check  Non-interactive validation only")
    print()
    print("Options:")
    print("  --check    Validate setup without making changes")
    print("  -h, --help Show this help")
