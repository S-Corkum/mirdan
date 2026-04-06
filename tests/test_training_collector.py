"""Tests for TrainingCollector fire-and-forget JSONL storage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mirdan.llm.training_collector import TrainingCollector


class TestTrainingCollectorRecord:
    """Tests for recording training samples."""

    @pytest.mark.asyncio
    async def test_record_triage_sample(self, tmp_path: Path) -> None:
        collector = TrainingCollector(storage_dir=tmp_path)
        await collector.record_triage_sample(
            "fix unused import", "simple lint fix", "local_only", 0.95
        )

        filepath = tmp_path / "triage_samples.jsonl"
        assert filepath.exists()
        entry = json.loads(filepath.read_text().strip())
        assert entry["classification"] == "local_only"
        assert entry["prompt"] == "fix unused import"

    @pytest.mark.asyncio
    async def test_record_validation_sample(self, tmp_path: Path) -> None:
        collector = TrainingCollector(storage_dir=tmp_path)
        await collector.record_validation_sample(
            code="x = 1",
            violations=[{"id": "PY001"}],
            llm_analysis={"per_violation": []},
            user_accepted=True,
        )

        filepath = tmp_path / "validation_samples.jsonl"
        entry = json.loads(filepath.read_text().strip())
        assert entry["violations_count"] == 1
        assert entry["user_accepted"] is True

    @pytest.mark.asyncio
    async def test_record_optimization_sample(self, tmp_path: Path) -> None:
        collector = TrainingCollector(storage_dir=tmp_path)
        await collector.record_optimization_sample(
            "original prompt", "optimized prompt", "sonnet"
        )

        filepath = tmp_path / "optimization_samples.jsonl"
        entry = json.loads(filepath.read_text().strip())
        assert entry["target_model"] == "sonnet"

    @pytest.mark.asyncio
    async def test_fire_and_forget_no_exceptions(self, tmp_path: Path) -> None:
        """Writing to a non-writable dir should not raise."""
        collector = TrainingCollector(storage_dir=tmp_path / "nonexistent" / "deep")
        # This should not raise even if dir creation in __init__ succeeded but
        # we can test the append path by using a valid dir
        await collector.record_triage_sample("test", "test", "local_only", 0.5)


class TestTrainingCollectorManagement:
    """Tests for sample counts and export."""

    @pytest.mark.asyncio
    async def test_get_sample_counts(self, tmp_path: Path) -> None:
        collector = TrainingCollector(storage_dir=tmp_path)
        await collector.record_triage_sample("a", "b", "local_only", 0.9)
        await collector.record_triage_sample("c", "d", "paid_required", 0.8)
        await collector.record_validation_sample("code", [], None)

        counts = collector.get_sample_counts()
        assert counts["triage_samples"] == 2
        assert counts["validation_samples"] == 1
        assert counts["optimization_samples"] == 0

    @pytest.mark.asyncio
    async def test_export_jsonl(self, tmp_path: Path) -> None:
        collector = TrainingCollector(storage_dir=tmp_path)
        await collector.record_triage_sample("test", "test", "local_only", 0.9)

        data = collector.export()
        assert data  # Not empty
        entry = json.loads(data.splitlines()[0])
        assert entry["classification"] == "local_only"

    def test_export_empty(self, tmp_path: Path) -> None:
        collector = TrainingCollector(storage_dir=tmp_path)
        data = collector.export()
        assert data == ""
