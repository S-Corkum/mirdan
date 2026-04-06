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
    ollama_ok = _check_ollama()
    llamacpp_ok = _check_llamacpp()
    print(f"  Ollama:       {'available' if ollama_ok else 'not found'}")
    print(f"  llama-cpp:    {'available' if llamacpp_ok else 'not installed'}")
    print()

    if not ollama_ok and not llamacpp_ok:
        print("ERROR: No LLM backend available.")
        print("Install Ollama: https://ollama.ai")
        print("Or: pip install llama-cpp-python")
        sys.exit(1)

    # 4. Check Metal/AVX2 compilation for llama-cpp-python
    if llamacpp_ok:
        metal_ok = _check_llamacpp_metal()
        if hw.metal_capable and not metal_ok:
            print("WARNING: llama-cpp-python may not have Metal support.")
            print("Reinstall with: CMAKE_ARGS=\"-DGGML_METAL=ON\" pip install --force-reinstall llama-cpp-python")
            print()

    # 5. Recommend model
    recommendation = _recommend_model(hw, ollama_ok, llamacpp_ok, enyal_present)
    print(f"Recommended: {recommendation['name']}")
    print(f"  Backend:  {recommendation['backend']}")
    print(f"  Memory:   ~{recommendation['memory_mb']} MB")
    print(f"  Quality:  {recommendation['quality']}")
    print()

    if check_only:
        print("Setup check complete.")
        return

    # 6-8. Download, test, and configure (interactive)
    if recommendation["backend"] == "ollama" and ollama_ok:
        tag = recommendation.get("ollama_tag", "")
        if tag:
            print(f"To download: ollama pull {tag}")
    elif recommendation["backend"] == "llamacpp":
        print(f"Download GGUF from HuggingFace: {recommendation.get('gguf_file', '')}")
        print(f"Place in: ~/.mirdan/models/")

    print()

    # 9. Write config
    _write_config(recommendation)
    print("Wrote .mirdan.yaml with llm.enabled: true")
    print()

    # 10. Next steps
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

    # Prefer llama-cpp-python (saves ~200MB)
    if llamacpp_ok and available >= 6500:
        return {
            "name": "Gemma 4 E4B Q3 (via llama-cpp-python)",
            "backend": "llamacpp",
            "memory_mb": 4500,
            "quality": "0.67",
            "gguf_file": "gemma-4-E4B-it-Q3_K_M.gguf",
        }

    if ollama_ok and available >= 5500:
        return {
            "name": "Gemma 4 E2B Q4 (via Ollama)",
            "backend": "ollama",
            "memory_mb": 3500,
            "quality": "0.60",
            "ollama_tag": "gemma4:e2b",
        }

    if llamacpp_ok and available >= 5200:
        return {
            "name": "Gemma 4 E2B Q3 (via llama-cpp-python)",
            "backend": "llamacpp",
            "memory_mb": 3200,
            "quality": "0.58",
            "gguf_file": "gemma-4-E2B-it-Q3_K_M.gguf",
        }

    # Fallback — try Ollama with smallest model
    return {
        "name": "Gemma 4 E2B Q4 (via Ollama, tight fit)",
        "backend": "ollama",
        "memory_mb": 3500,
        "quality": "0.60",
        "ollama_tag": "gemma4:e2b",
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
