"""Token savings and LLM usage metrics — JSONL append-only storage."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TokenMetrics:
    """Tracks LLM calls, token savings, triage distributions, and optimization outcomes.

    Stores metrics as JSONL files under .mirdan/metrics/ for durability
    and easy export. Each method appends one line — no locks needed for
    single-writer append.
    """

    def __init__(self, storage_dir: Path | None = None) -> None:
        self._dir = storage_dir or Path(".mirdan/metrics")
        self._dir.mkdir(parents=True, exist_ok=True)

    def record_llm_call(
        self,
        role: str,
        tokens_in: int,
        tokens_out: int,
        feature: str,
        elapsed_ms: float,
        hardware_profile: str = "unknown",
    ) -> None:
        """Record a local LLM call.

        Args:
            role: Model role (fast/brain).
            tokens_in: Input tokens.
            tokens_out: Output tokens.
            feature: Which feature used the LLM (triage/validation/optimization/research).
            elapsed_ms: Wall-clock time in ms.
            hardware_profile: Hardware profile tier.
        """
        self._append("llm_calls.jsonl", {
            "ts": time.time(),
            "role": role,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "feature": feature,
            "elapsed_ms": round(elapsed_ms, 1),
            "hardware_profile": hardware_profile,
        })

    def record_token_savings(
        self,
        feature: str,
        estimated_paid_tokens: int,
        actual_local_tokens: int,
    ) -> None:
        """Record estimated token savings from local processing.

        Args:
            feature: Feature that saved tokens.
            estimated_paid_tokens: Tokens the paid model would have used.
            actual_local_tokens: Tokens the local model used instead.
        """
        self._append("savings.jsonl", {
            "ts": time.time(),
            "feature": feature,
            "estimated_paid": estimated_paid_tokens,
            "actual_local": actual_local_tokens,
            "saved": estimated_paid_tokens - actual_local_tokens,
        })

    def record_triage(
        self,
        classification: str,
        confidence: float,
        was_correct: bool | None = None,
    ) -> None:
        """Record a triage classification.

        Args:
            classification: The triage result.
            confidence: Model confidence.
            was_correct: Ground truth if known (for accuracy tracking).
        """
        entry: dict[str, Any] = {
            "ts": time.time(),
            "classification": classification,
            "confidence": confidence,
        }
        if was_correct is not None:
            entry["was_correct"] = was_correct
        self._append("triage.jsonl", entry)

    def record_optimization_outcome(
        self,
        session_id: str,
        target_model: str,
        first_try_passed: bool,
        validation_score: float,
    ) -> None:
        """Record prompt optimization outcome (first_try_passed as quality proxy).

        Args:
            session_id: Session identifier.
            target_model: Which paid model was targeted.
            first_try_passed: Whether validation passed on first attempt.
            validation_score: Quality score from validation.
        """
        self._append("optimization.jsonl", {
            "ts": time.time(),
            "session_id": session_id,
            "target_model": target_model,
            "first_try_passed": first_try_passed,
            "validation_score": round(validation_score, 3),
        })

    def record_sanity_cap_triggered(
        self,
        feature: str,
        original_fp_count: int,
        cap_applied: bool,
    ) -> None:
        """Record when a sanity cap was triggered.

        Args:
            feature: Feature that triggered the cap.
            original_fp_count: Number of false positives before cap.
            cap_applied: Whether the cap was actually applied.
        """
        self._append("sanity_caps.jsonl", {
            "ts": time.time(),
            "feature": feature,
            "original_fp_count": original_fp_count,
            "cap_applied": cap_applied,
        })

    def get_summary(self, days: int = 30) -> dict[str, Any]:
        """Get a summary of metrics over the given period.

        Args:
            days: Number of days to include.

        Returns:
            Summary dict with call counts, savings, and distributions.
        """
        cutoff = time.time() - (days * 86400)

        calls = self._read("llm_calls.jsonl", cutoff)
        savings = self._read("savings.jsonl", cutoff)
        triages = self._read("triage.jsonl", cutoff)

        total_saved = sum(e.get("saved", 0) for e in savings)
        total_local = sum(e.get("actual_local", 0) for e in savings)
        total_paid_est = sum(e.get("estimated_paid", 0) for e in savings)

        return {
            "period_days": days,
            "llm_calls": len(calls),
            "total_local_tokens": total_local,
            "estimated_paid_tokens_saved": total_saved,
            "estimated_total_paid_tokens": total_paid_est,
            "savings_percentage": (
                round(total_saved / total_paid_est * 100, 1) if total_paid_est > 0 else 0
            ),
            "triage_count": len(triages),
        }

    def get_triage_distribution(self, days: int = 30) -> dict[str, int]:
        """Get distribution of triage classifications.

        Args:
            days: Number of days to include.

        Returns:
            Dict mapping classification → count.
        """
        cutoff = time.time() - (days * 86400)
        triages = self._read("triage.jsonl", cutoff)
        dist: dict[str, int] = {}
        for t in triages:
            cls = t.get("classification", "unknown")
            dist[cls] = dist.get(cls, 0) + 1
        return dist

    def _append(self, filename: str, entry: dict[str, Any]) -> None:
        """Append a JSON line to a metrics file."""
        try:
            filepath = self._dir / filename
            with filepath.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            logger.warning("Failed to write metric to %s", filename)

    def _read(self, filename: str, cutoff: float = 0) -> list[dict[str, Any]]:
        """Read JSONL entries newer than cutoff."""
        filepath = self._dir / filename
        if not filepath.exists():
            return []
        entries: list[dict[str, Any]] = []
        try:
            for line in filepath.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    entry: dict[str, Any] = json.loads(line)
                    if entry.get("ts", 0) >= cutoff:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass
        return entries
