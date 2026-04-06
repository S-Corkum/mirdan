"""``mirdan fine-tune status|export`` — training data management."""

from __future__ import annotations

import sys

from mirdan.config import MirdanConfig


def run_finetune(args: list[str]) -> None:
    """Manage fine-tuning training data.

    Usage:
        mirdan fine-tune status              Show sample counts
        mirdan fine-tune export [--format F] Export training data

    Args:
        args: CLI arguments after ``fine-tune``.
    """
    if not args or args[0] in ("--help", "-h"):
        _print_help()
        sys.exit(0)

    if args[0] == "status":
        _status()
    elif args[0] == "export":
        _export(args[1:])
    else:
        print(f"Unknown fine-tune subcommand: {args[0]}")
        _print_help()
        sys.exit(1)


def _status() -> None:
    """Show collected sample counts."""
    from mirdan.llm.training_collector import TrainingCollector

    collector = TrainingCollector()
    counts = collector.get_sample_counts()

    print("=== Fine-Tuning Data Status ===\n")
    total = 0
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count} samples")
        total += count
    print(f"\n  Total: {total} samples")


def _export(args: list[str]) -> None:
    """Export training data."""
    from mirdan.llm.training_collector import TrainingCollector

    fmt = "jsonl"
    for i, arg in enumerate(args):
        if arg == "--format" and i + 1 < len(args):
            fmt = args[i + 1]

    collector = TrainingCollector()
    data = collector.export(format=fmt)

    if data:
        print(data)
    else:
        print("No training data collected yet.")


def _print_help() -> None:
    print("mirdan fine-tune — manage fine-tuning training data")
    print()
    print("Usage:")
    print("  mirdan fine-tune status              Show sample counts")
    print("  mirdan fine-tune export [--format F]  Export training data (default: jsonl)")
    print()
    print("Options:")
    print("  -h, --help Show this help")
