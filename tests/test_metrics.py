"""Tests for TokenMetrics JSONL storage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mirdan.llm.metrics import TokenMetrics


class TestTokenMetricsRecord:
    """Tests for recording metrics."""

    def test_record_llm_call(self, tmp_path: Path) -> None:
        metrics = TokenMetrics(storage_dir=tmp_path)
        metrics.record_llm_call("fast", 100, 50, "triage", 150.0, "standard")

        filepath = tmp_path / "llm_calls.jsonl"
        assert filepath.exists()
        entry = json.loads(filepath.read_text().strip())
        assert entry["role"] == "fast"
        assert entry["tokens_in"] == 100
        assert entry["tokens_out"] == 50
        assert entry["feature"] == "triage"

    def test_record_token_savings(self, tmp_path: Path) -> None:
        metrics = TokenMetrics(storage_dir=tmp_path)
        metrics.record_token_savings("triage", 5000, 200)

        filepath = tmp_path / "savings.jsonl"
        entry = json.loads(filepath.read_text().strip())
        assert entry["estimated_paid"] == 5000
        assert entry["actual_local"] == 200
        assert entry["saved"] == 4800

    def test_record_triage(self, tmp_path: Path) -> None:
        metrics = TokenMetrics(storage_dir=tmp_path)
        metrics.record_triage("local_only", 0.95, was_correct=True)

        filepath = tmp_path / "triage.jsonl"
        entry = json.loads(filepath.read_text().strip())
        assert entry["classification"] == "local_only"
        assert entry["was_correct"] is True

    def test_record_triage_without_ground_truth(self, tmp_path: Path) -> None:
        metrics = TokenMetrics(storage_dir=tmp_path)
        metrics.record_triage("paid_required", 0.8)

        filepath = tmp_path / "triage.jsonl"
        entry = json.loads(filepath.read_text().strip())
        assert "was_correct" not in entry

    def test_record_optimization_outcome(self, tmp_path: Path) -> None:
        metrics = TokenMetrics(storage_dir=tmp_path)
        metrics.record_optimization_outcome("sess-1", "sonnet", True, 0.92)

        filepath = tmp_path / "optimization.jsonl"
        entry = json.loads(filepath.read_text().strip())
        assert entry["first_try_passed"] is True

    def test_record_sanity_cap(self, tmp_path: Path) -> None:
        metrics = TokenMetrics(storage_dir=tmp_path)
        metrics.record_sanity_cap_triggered("validation", 8, True)

        filepath = tmp_path / "sanity_caps.jsonl"
        entry = json.loads(filepath.read_text().strip())
        assert entry["cap_applied"] is True

    def test_multiple_records_append(self, tmp_path: Path) -> None:
        metrics = TokenMetrics(storage_dir=tmp_path)
        metrics.record_triage("local_only", 0.9)
        metrics.record_triage("paid_required", 0.8)

        filepath = tmp_path / "triage.jsonl"
        lines = filepath.read_text().strip().splitlines()
        assert len(lines) == 2


class TestTokenMetricsSummary:
    """Tests for get_summary and get_triage_distribution."""

    def test_get_summary(self, tmp_path: Path) -> None:
        metrics = TokenMetrics(storage_dir=tmp_path)
        metrics.record_llm_call("fast", 100, 50, "triage", 50.0)
        metrics.record_token_savings("triage", 5000, 200)
        metrics.record_triage("local_only", 0.95)

        summary = metrics.get_summary(days=30)
        assert summary["llm_calls"] == 1
        assert summary["estimated_paid_tokens_saved"] == 4800
        assert summary["triage_count"] == 1
        assert summary["savings_percentage"] > 0

    def test_get_summary_empty(self, tmp_path: Path) -> None:
        metrics = TokenMetrics(storage_dir=tmp_path)
        summary = metrics.get_summary()
        assert summary["llm_calls"] == 0
        assert summary["savings_percentage"] == 0

    def test_get_triage_distribution(self, tmp_path: Path) -> None:
        metrics = TokenMetrics(storage_dir=tmp_path)
        metrics.record_triage("local_only", 0.9)
        metrics.record_triage("local_only", 0.8)
        metrics.record_triage("paid_required", 0.7)

        dist = metrics.get_triage_distribution()
        assert dist["local_only"] == 2
        assert dist["paid_required"] == 1
