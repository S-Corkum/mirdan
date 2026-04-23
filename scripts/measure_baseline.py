"""Baseline measurement for mirdan 2.1.0 brief-driven pipeline regression checks.

Measures two things across recent 2.0.x plans, matching brief Outcome targets:

1. **Plan-creation token cost:** estimated from plan file sizes (markdown token
   proxy) over the last N plans in ``docs/plans/``. Multiplies by current Opus
   pricing to produce a dollar estimate.

2. **Plan-execution token cost:** estimated from git-diff sizes on the 3 most
   recent commits that match plan slugs, as a rough proxy for execution work.

Output: ``baselines/2.0.x-baseline.json`` with ``plan_creation`` and
``plan_execution`` mean/median stats.

Used to validate 2.1.0 post-launch targets in brief Outcome:

- ≥70% frontier-token reduction per plan
- Plan execution at no cost regression vs. 2.0.x baseline
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Token cost estimates (USD per 1M tokens). Conservative; used for order-of-
# magnitude comparisons, not precise accounting.
_OPUS_INPUT_USD_PER_MTOK = 15.0
_OPUS_OUTPUT_USD_PER_MTOK = 75.0
_HAIKU_INPUT_USD_PER_MTOK = 1.0
_HAIKU_OUTPUT_USD_PER_MTOK = 5.0

# Rough char→token ratio for markdown (empirical; varies by tokenizer)
_CHARS_PER_TOKEN = 4


def _chars_to_tokens(chars: int) -> int:
    return chars // _CHARS_PER_TOKEN


def _plan_creation_tokens(plan_path: Path) -> int:
    """Rough proxy: plan file size → tokens (assumes plan is close to output size)."""
    return _chars_to_tokens(len(plan_path.read_text()))


def _plan_creation_usd(tokens: int) -> float:
    """Estimate USD assuming 70/30 input/output split on Opus."""
    input_tokens = int(tokens * 0.3)  # creator reads a lot before writing
    output_tokens = int(tokens * 0.7)
    return (
        input_tokens * _OPUS_INPUT_USD_PER_MTOK / 1_000_000
        + output_tokens * _OPUS_OUTPUT_USD_PER_MTOK / 1_000_000
    )


def _plan_execution_tokens(slug: str, repo_root: Path) -> int | None:
    """Rough proxy: cumulative diff size of commits touching ``slug`` files."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "log", "--oneline", "--all", "--grep", slug],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    commits = [line.split()[0] for line in result.stdout.strip().splitlines() if line]
    if not commits:
        return None
    total_chars = 0
    for sha in commits[:5]:
        diff = subprocess.run(
            ["git", "-C", str(repo_root), "show", "--stat", sha],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        total_chars += len(diff.stdout)
    return _chars_to_tokens(total_chars)


def _plan_execution_usd(tokens: int) -> float:
    """2.0.x execution used Haiku-class via Agent tool; 70/30 input/output split."""
    input_tokens = int(tokens * 0.3)
    output_tokens = int(tokens * 0.7)
    return (
        input_tokens * _HAIKU_INPUT_USD_PER_MTOK / 1_000_000
        + output_tokens * _HAIKU_OUTPUT_USD_PER_MTOK / 1_000_000
    )


def measure(repo_root: Path, plans_dir: Path, n_plans: int, n_exec: int) -> dict[str, Any]:
    plan_files = sorted(
        plans_dir.glob("*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:n_plans]

    creation_tokens = [_plan_creation_tokens(p) for p in plan_files]
    creation_usd = [_plan_creation_usd(t) for t in creation_tokens]

    exec_tokens: list[int] = []
    for p in plan_files[:n_exec]:
        slug = p.stem
        t = _plan_execution_tokens(slug, repo_root)
        if t is not None:
            exec_tokens.append(t)
    exec_usd = [_plan_execution_usd(t) for t in exec_tokens]

    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "source_plans": [
            str(p.relative_to(repo_root)) if str(p).startswith(str(repo_root)) else str(p)
            for p in plan_files
        ],
        "plan_creation": _stats(creation_tokens, creation_usd, "Opus"),
        "plan_execution": _stats(exec_tokens, exec_usd, "Haiku"),
        "notes": (
            "Baseline for mirdan 2.1.0 brief-driven pipeline. Token counts "
            "are char-to-token proxy estimates (~4 chars/token). Costs use "
            "Claude 4.x published pricing. Used for regression comparison "
            "post-2.1.0 launch."
        ),
    }


def _stats(tokens: list[int], usd: list[float], model: str) -> dict[str, Any]:
    if not tokens:
        return {
            "n": 0,
            "model": model,
            "note": "no samples available (insufficient plans or git history)",
        }
    return {
        "n": len(tokens),
        "model": model,
        "mean_tokens": int(statistics.mean(tokens)),
        "median_tokens": int(statistics.median(tokens)),
        "mean_usd": round(statistics.mean(usd), 4),
        "median_usd": round(statistics.median(usd), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("baselines/2.0.x-baseline.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--plans-dir",
        type=Path,
        default=Path("docs/plans"),
        help="Directory of plan markdown files",
    )
    parser.add_argument("--n-plans", type=int, default=10)
    parser.add_argument("--n-exec", type=int, default=3)
    args = parser.parse_args()

    repo_root = Path.cwd()
    if not args.plans_dir.exists():
        print(f"plans_dir does not exist: {args.plans_dir}", file=sys.stderr)
        return 1

    data = measure(repo_root, args.plans_dir, args.n_plans, args.n_exec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Baseline written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
