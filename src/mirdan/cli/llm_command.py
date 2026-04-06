"""``mirdan llm status|warmup|metrics`` — LLM subsystem management commands."""

from __future__ import annotations

import json
import sys

from mirdan.config import MirdanConfig


def run_llm_status(args: list[str]) -> None:
    """Show LLM subsystem status: hardware, model, features.

    Usage:
        mirdan llm status [--json]
    """
    from mirdan.llm.health import HardwareDetector

    hw = HardwareDetector.detect()
    config = MirdanConfig.find_config()

    features = {
        "triage": config.llm.triage,
        "smart_validation": config.llm.smart_validation,
        "check_runner": config.llm.check_runner,
        "prompt_optimization": config.llm.prompt_optimization,
        "research_agent": config.llm.research_agent,
    }

    status = {
        "llm_enabled": config.llm.enabled,
        "backend": config.llm.backend,
        "hardware": hw.to_dict(),
        "features": features,
    }

    if "--json" in args:
        print(json.dumps(status, indent=2))
    else:
        print(f"LLM enabled:  {config.llm.enabled}")
        print(f"Backend:      {config.llm.backend}")
        print(f"Architecture: {hw.architecture}")
        print(f"RAM:          {hw.total_ram_mb} MB total, {hw.available_ram_mb} MB available")
        print(f"Profile:      {hw.detected_profile.value}")
        print(f"GPU:          {hw.gpu_type or 'none'}")
        print()
        print("Features:")
        for feat, enabled in features.items():
            print(f"  {feat}: {'enabled' if enabled else 'disabled'}")


def run_llm_warmup(args: list[str]) -> None:
    """Pre-load the configured model.

    Usage:
        mirdan llm warmup
    """
    print("Model warmup is handled automatically by the MCP server.")
    print("Start mirdan (MCP server) to trigger warmup.")


def run_llm_metrics(args: list[str]) -> None:
    """Show 30-day token savings dashboard.

    Usage:
        mirdan llm metrics [--json] [--days N]
    """
    from mirdan.llm.metrics import TokenMetrics

    days = 30
    for i, arg in enumerate(args):
        if arg == "--days" and i + 1 < len(args):
            try:
                days = int(args[i + 1])
            except ValueError:
                pass

    metrics = TokenMetrics()
    summary = metrics.get_summary(days=days)
    triage_dist = metrics.get_triage_distribution(days=days)

    if "--json" in args:
        output = {**summary, "triage_distribution": triage_dist}
        print(json.dumps(output, indent=2))
    else:
        print(f"=== mirdan LLM Metrics ({days} days) ===\n")
        print(f"LLM calls:           {summary['llm_calls']}")
        print(f"Local tokens used:   {summary['total_local_tokens']}")
        print(f"Est. paid saved:     {summary['estimated_paid_tokens_saved']}")
        print(f"Savings:             {summary['savings_percentage']}%")
        print(f"Triage count:        {summary['triage_count']}")
        if triage_dist:
            print("\nTriage distribution:")
            for cls, count in sorted(triage_dist.items()):
                print(f"  {cls}: {count}")
