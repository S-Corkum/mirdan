"""Training data collector for future fine-tuning — JSONL storage."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from mirdan.config import LLMConfig

logger = logging.getLogger(__name__)


class TrainingCollector:
    """Collects training samples as JSONL for future fine-tuning.

    All methods are fire-and-forget: errors are logged, never raised.
    Privacy: optional code redaction when configured.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        storage_dir: Path | None = None,
    ) -> None:
        self._config = config or LLMConfig()
        self._dir = storage_dir or Path(".mirdan/training")
        self._dir.mkdir(parents=True, exist_ok=True)

    def record_triage_sample(
        self,
        prompt: str,
        intent_summary: str,
        classification: str,
        confidence: float,
    ) -> None:
        """Record a triage training sample.

        Args:
            prompt: Developer's original prompt.
            intent_summary: Summarized intent.
            classification: Triage classification result.
            confidence: Model confidence.
        """
        self._append(
            "triage_samples.jsonl",
            {
                "ts": time.time(),
                "prompt": prompt,
                "intent_summary": intent_summary,
                "classification": classification,
                "confidence": confidence,
            },
        )

    def record_validation_sample(
        self,
        code: str,
        violations: list[dict[str, Any]],
        llm_analysis: dict[str, Any] | None,
        user_accepted: bool | None = None,
    ) -> None:
        """Record a validation training sample.

        Args:
            code: Source code (may be redacted).
            violations: Detected violations.
            llm_analysis: LLM's analysis result.
            user_accepted: Whether the user accepted the analysis.
        """
        self._append(
            "validation_samples.jsonl",
            {
                "ts": time.time(),
                "code": self._redact_if_needed(code),
                "violations_count": len(violations),
                "llm_analysis": llm_analysis,
                "user_accepted": user_accepted,
            },
        )

    def record_optimization_sample(
        self,
        original_prompt: str,
        optimized_prompt: str,
        target_model: str,
        outcome: dict[str, Any] | None = None,
    ) -> None:
        """Record a prompt optimization training sample.

        Args:
            original_prompt: Original enhance_prompt input.
            optimized_prompt: BRAIN-generated optimized prompt.
            target_model: Target paid model.
            outcome: Validation outcome if available.
        """
        self._append(
            "optimization_samples.jsonl",
            {
                "ts": time.time(),
                "original": original_prompt,
                "optimized": optimized_prompt,
                "target_model": target_model,
                "outcome": outcome,
            },
        )

    def get_sample_counts(self) -> dict[str, int]:
        """Get counts of collected samples per type.

        Returns:
            Dict mapping sample type → count.
        """
        counts: dict[str, int] = {}
        for name in ("triage_samples", "validation_samples", "optimization_samples"):
            filepath = self._dir / f"{name}.jsonl"
            if filepath.exists():
                counts[name] = sum(1 for line in filepath.read_text().splitlines() if line.strip())
            else:
                counts[name] = 0
        return counts

    def export(self, output_format: str = "jsonl") -> str:
        """Export all training data.

        Args:
            output_format: Export format (only "jsonl" supported currently).

        Returns:
            Path to exported file or concatenated JSONL content.
        """
        lines: list[str] = []
        for filepath in sorted(self._dir.glob("*.jsonl")):
            for line in filepath.read_text().splitlines():
                if line.strip():
                    lines.append(line)
        return "\n".join(lines)

    def _redact_if_needed(self, code: str) -> str:
        """Optionally redact code for privacy."""
        # Placeholder for future redaction config
        return code

    def _append(self, filename: str, entry: dict[str, Any]) -> None:
        """Append a JSONL entry. Fire-and-forget — errors logged, not raised."""
        try:
            filepath = self._dir / filename
            with filepath.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            logger.warning("Failed to write training sample to %s", filename)
