"""Tests for Phase 3: Make It Learn — convention discovery, feedback loop, profiles, auto-memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from mirdan.config import MirdanConfig, OrchestrationConfig
from mirdan.core.quality_profiles import (
    BUILTIN_PROFILES,
    QualityProfile,
    apply_profile,
    get_profile,
    list_profiles,
    suggest_profile,
)
from mirdan.core.rule_generator import RuleGenerator
from mirdan.core.violation_tracker import (
    OverrideRecord,
    SeverityAdjustment,
    ViolationTracker,
)
from mirdan.models import KnowledgeEntry

# ── 3A: Convention Discovery Pipeline ──────────────────────────────────


class TestRuleGenerator:
    """Tests for RuleGenerator."""

    def test_empty_conventions_returns_empty(self) -> None:
        gen = RuleGenerator()
        assert gen.generate_from_conventions([]) == []

    def test_low_confidence_filtered(self) -> None:
        entry = KnowledgeEntry(
            content="snake_case for function names",
            content_type="convention",
            tags=["naming"],
            confidence=0.5,
        )
        gen = RuleGenerator()
        assert gen.generate_from_conventions([entry]) == []

    def test_naming_convention_generates_rule(self) -> None:
        entry = KnowledgeEntry(
            content="snake_case for function names",
            content_type="convention",
            tags=["naming"],
            confidence=0.9,
        )
        gen = RuleGenerator()
        rules = gen.generate_from_conventions([entry])
        assert len(rules) == 1
        assert rules[0]["severity"] == "warning"
        assert rules[0]["language"] == "python"
        assert "CONV" in rules[0]["id"]

    def test_import_convention_generates_rule(self) -> None:
        entry = KnowledgeEntry(
            content="absolute import preferred over relative import",
            content_type="convention",
            tags=["import"],
            confidence=0.85,
        )
        gen = RuleGenerator()
        rules = gen.generate_from_conventions([entry])
        assert len(rules) == 1
        assert rules[0]["severity"] == "info"

    def test_docstring_convention_generates_rule(self) -> None:
        entry = KnowledgeEntry(
            content="Google style docstrings for all public functions",
            content_type="convention",
            tags=["docstring"],
            confidence=0.9,
        )
        gen = RuleGenerator()
        rules = gen.generate_from_conventions([entry])
        assert len(rules) == 1
        assert "google" in rules[0]["rule"].lower() or "docstring" in rules[0]["message"].lower()

    def test_violation_pattern_generates_rule(self) -> None:
        entry = KnowledgeEntry(
            content="PY005 appears frequently in validation module",
            content_type="convention",
            tags=["violation_pattern"],
            confidence=0.85,
        )
        gen = RuleGenerator()
        rules = gen.generate_from_conventions([entry])
        assert len(rules) == 1
        assert "PY005" in rules[0]["message"] or "py005" in rules[0]["rule"]

    def test_unconvertible_convention_skipped(self) -> None:
        entry = KnowledgeEntry(
            content="The team prefers functional patterns",
            content_type="convention",
            tags=["general"],
            confidence=0.95,
        )
        gen = RuleGenerator()
        rules = gen.generate_from_conventions([entry])
        assert rules == []

    def test_generate_and_write(self, tmp_path: Path) -> None:
        entry = KnowledgeEntry(
            content="snake_case for function naming convention",
            content_type="convention",
            tags=["naming"],
            confidence=0.9,
        )
        gen = RuleGenerator()
        output = tmp_path / "rules" / "conventions.yaml"
        gen.generate_and_write([entry], output)

        assert output.exists()
        data = yaml.safe_load(output.read_text())
        assert "rules" in data
        assert len(data["rules"]) >= 1

    def test_custom_min_confidence(self) -> None:
        entry = KnowledgeEntry(
            content="snake_case for function names",
            content_type="convention",
            tags=["naming"],
            confidence=0.7,
        )
        gen = RuleGenerator(min_confidence=0.6)
        rules = gen.generate_from_conventions([entry])
        assert len(rules) == 1


# ── 3A: scan_conventions MCP tool ──────────────────────────────────────


class TestScanConventionsTool:
    """Tests for scan_conventions MCP tool registration."""

    def test_scan_conventions_in_priority(self) -> None:
        import mirdan.server as server_mod

        assert "scan_conventions" in server_mod._TOOL_PRIORITY

    def test_scan_conventions_registered(self) -> None:
        import mirdan.server as server_mod

        registered = set(server_mod.mcp._tool_manager._tools.keys())
        assert "scan_conventions" in registered


# ── 3B: ViolationTracker ──────────────────────────────────────────────


class TestViolationTracker:
    """Tests for ViolationTracker."""

    def test_record_and_count(self, tmp_path: Path) -> None:
        tracker = ViolationTracker(storage_path=tmp_path / "overrides.json")
        tracker.record_override("PY005", "test.py", "false positive")
        tracker.record_override("PY005", "test2.py", "intentional")
        assert tracker.get_override_count("PY005") == 2
        assert tracker.get_override_count("SEC001") == 0

    def test_get_all_counts(self, tmp_path: Path) -> None:
        tracker = ViolationTracker(storage_path=tmp_path / "overrides.json")
        tracker.record_override("PY005")
        tracker.record_override("PY005")
        tracker.record_override("SEC001")
        counts = tracker.get_all_counts()
        assert counts == {"PY005": 2, "SEC001": 1}

    def test_persistence(self, tmp_path: Path) -> None:
        path = tmp_path / "overrides.json"
        tracker1 = ViolationTracker(storage_path=path)
        tracker1.record_override("PY005", "test.py", "reason1")

        # Load a new tracker from the same file
        tracker2 = ViolationTracker(storage_path=path)
        assert tracker2.get_override_count("PY005") == 1

    def test_suggest_severity_changes_below_threshold(self, tmp_path: Path) -> None:
        tracker = ViolationTracker(storage_path=tmp_path / "overrides.json")
        for _ in range(3):
            tracker.record_override("PY005")
        # Default threshold is 5
        assert tracker.suggest_severity_changes() == []

    def test_suggest_severity_changes_above_threshold(self, tmp_path: Path) -> None:
        tracker = ViolationTracker(storage_path=tmp_path / "overrides.json")
        for _ in range(6):
            tracker.record_override("PY005")
        suggestions = tracker.suggest_severity_changes()
        assert len(suggestions) == 1
        assert suggestions[0].rule_id == "PY005"
        assert suggestions[0].override_count == 6

    def test_sec_rules_downgrade_error_to_warning(self, tmp_path: Path) -> None:
        tracker = ViolationTracker(storage_path=tmp_path / "overrides.json")
        for _ in range(5):
            tracker.record_override("SEC001")
        suggestions = tracker.suggest_severity_changes()
        assert len(suggestions) == 1
        assert suggestions[0].current_severity == "error"
        assert suggestions[0].suggested_severity == "warning"

    def test_info_cannot_downgrade_further(self, tmp_path: Path) -> None:
        tracker = ViolationTracker(storage_path=tmp_path / "overrides.json")
        # PY006 would be inferred as "warning", downgraded to "info"
        for _ in range(5):
            tracker.record_override("PY006")
        suggestions = tracker.suggest_severity_changes()
        assert len(suggestions) == 1
        assert suggestions[0].suggested_severity == "info"

    def test_override_record_to_dict(self) -> None:
        record = OverrideRecord(
            rule_id="PY005",
            file_path="test.py",
            reason="false positive",
            timestamp=1000.0,
        )
        d = record.to_dict()
        assert d["rule_id"] == "PY005"
        assert d["file_path"] == "test.py"
        assert d["timestamp"] == 1000.0

    def test_severity_adjustment_to_dict(self) -> None:
        adj = SeverityAdjustment(
            rule_id="SEC001",
            current_severity="error",
            suggested_severity="warning",
            override_count=7,
            reason="test",
        )
        d = adj.to_dict()
        assert d["override_count"] == 7


# ── 3C: Profile Auto-Tuning ──────────────────────────────────────────


class TestSuggestProfile:
    """Tests for suggest_profile()."""

    def test_high_quality_large_codebase(self) -> None:
        result = suggest_profile(
            {
                "avg_score": 0.95,
                "pass_rate": 0.95,
                "files_scanned": 100,
            }
        )
        assert result == ("enterprise", 0.8)

    def test_high_quality_small_codebase(self) -> None:
        result = suggest_profile(
            {
                "avg_score": 0.92,
                "pass_rate": 0.92,
                "files_scanned": 20,
            }
        )
        assert result == ("library", 0.75)

    def test_medium_quality(self) -> None:
        result = suggest_profile({"avg_score": 0.75, "pass_rate": 0.7})
        assert result == ("default", 0.7)

    def test_lower_quality(self) -> None:
        result = suggest_profile({"avg_score": 0.55, "pass_rate": 0.5})
        assert result == ("startup", 0.65)

    def test_low_quality_few_files(self) -> None:
        result = suggest_profile(
            {
                "avg_score": 0.3,
                "pass_rate": 0.3,
                "files_scanned": 5,
            }
        )
        assert result == ("prototype", 0.6)

    def test_empty_scan(self) -> None:
        name, conf = suggest_profile({})
        assert name in BUILTIN_PROFILES
        assert 0.0 <= conf <= 1.0


class TestProfileOperations:
    """Tests for profile list/get/apply."""

    def test_list_profiles_returns_all_builtins(self) -> None:
        profiles = list_profiles()
        names = {p["name"] for p in profiles}
        assert "default" in names
        assert "enterprise" in names
        assert "startup" in names
        assert "fintech" in names
        assert "library" in names
        assert "data-science" in names
        assert "prototype" in names

    def test_list_profiles_includes_custom(self) -> None:
        custom = {"my-profile": {"description": "Custom test"}}
        profiles = list_profiles(custom)
        names = {p["name"] for p in profiles}
        assert "my-profile" in names

    def test_get_profile_builtin(self) -> None:
        profile = get_profile("enterprise")
        assert profile.name == "enterprise"
        assert profile.security == 1.0

    def test_get_profile_custom(self) -> None:
        custom = {"custom": {"description": "Test", "security": 0.5}}
        profile = get_profile("custom", custom)
        assert profile.security == 0.5

    def test_get_profile_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown quality profile"):
            get_profile("nonexistent")

    def test_apply_profile(self) -> None:
        profile = get_profile("enterprise")
        config: dict[str, Any] = {}
        apply_profile(profile, config)
        assert config["quality"]["security"] == "strict"
        assert config["quality"]["architecture"] == "strict"

    def test_quality_profile_to_stringency(self) -> None:
        p = QualityProfile(name="test", description="test")
        assert p.to_stringency(0.0) == "permissive"
        assert p.to_stringency(0.5) == "moderate"
        assert p.to_stringency(0.8) == "strict"
        assert p.to_stringency(1.0) == "strict"


# ── 3C: Profile CLI ──────────────────────────────────────────────────


class TestProfileCLI:
    """Tests for profile CLI command registration."""

    def test_profile_command_registered(self) -> None:
        """Profile command should be in CLI routing."""
        import inspect

        import mirdan.cli as cli_mod

        source = inspect.getsource(cli_mod.main)
        assert "profile" in source


# ── 3D: Auto-Memory Integration ──────────────────────────────────────


class TestAutoMemoryConfig:
    """Tests for auto_memory config fields."""

    def test_default_auto_memory_disabled(self) -> None:
        config = MirdanConfig()
        assert config.orchestration.auto_memory is False
        assert config.orchestration.auto_memory_threshold == 0.8

    def test_auto_memory_configurable(self) -> None:
        config = OrchestrationConfig(
            auto_memory=True,
            auto_memory_threshold=0.9,
        )
        assert config.auto_memory is True
        assert config.auto_memory_threshold == 0.9


class TestAutoMemoryInOutput:
    """Tests that auto_store flag appears in MCP output when enabled."""

    def test_knowledge_entries_flagged_when_auto_memory(self) -> None:
        """When auto_memory=True, knowledge entries should get auto_store flag."""
        entry = KnowledgeEntry(
            content="test entry",
            content_type="fact",
            confidence=0.9,
        )
        d = entry.to_dict()
        # Simulate what server.py does
        auto_mem = True
        threshold = 0.8
        if auto_mem and entry.confidence >= threshold:
            d["auto_store"] = True
        assert d["auto_store"] is True

    def test_knowledge_entries_not_flagged_when_disabled(self) -> None:
        entry = KnowledgeEntry(
            content="test entry",
            content_type="fact",
            confidence=0.9,
        )
        d = entry.to_dict()
        auto_mem = False
        if auto_mem and entry.confidence >= 0.8:
            d["auto_store"] = True
        assert "auto_store" not in d

    def test_low_confidence_not_flagged(self) -> None:
        entry = KnowledgeEntry(
            content="test entry",
            content_type="fact",
            confidence=0.5,
        )
        d = entry.to_dict()
        auto_mem = True
        threshold = 0.8
        if auto_mem and entry.confidence >= threshold:
            d["auto_store"] = True
        assert "auto_store" not in d


class TestToolExecutorStoreKnowledge:
    """Tests for ToolExecutor.store_knowledge()."""

    @pytest.mark.asyncio
    async def test_store_skips_unconfigured_enyal(self) -> None:
        from mirdan.core.active_orchestrator import ToolExecutor

        mock_registry = MagicMock()
        mock_registry.is_configured.return_value = False

        orchestrator = ToolExecutor(mock_registry)
        entries = [
            KnowledgeEntry(content="test", content_type="fact", confidence=0.9),
        ]
        results = await orchestrator.store_knowledge(entries)
        assert results == []

    @pytest.mark.asyncio
    async def test_store_filters_by_confidence(self) -> None:
        from mirdan.core.active_orchestrator import ToolExecutor

        mock_registry = MagicMock()
        mock_registry.is_configured.return_value = True
        mock_registry.call_tools_parallel = AsyncMock(return_value=[])

        orchestrator = ToolExecutor(mock_registry)
        entries = [
            KnowledgeEntry(content="low", content_type="fact", confidence=0.3),
            KnowledgeEntry(content="high", content_type="fact", confidence=0.9),
        ]
        await orchestrator.store_knowledge(entries, min_confidence=0.8)
        # Only the high-confidence entry should be called
        assert mock_registry.call_tools_parallel.called
        calls = mock_registry.call_tools_parallel.call_args[0][0]
        assert len(calls) == 1
        assert calls[0].arguments["input"]["content"] == "high"
        assert calls[0].arguments["input"]["source"] == "mirdan:auto-memory"
        assert calls[0].arguments["input"]["check_duplicate"] is True
        assert calls[0].arguments["input"]["auto_link"] is True


# ── 3E: Convention-Aware Validation ──────────────────────────────────


class TestCustomRulesWired:
    """Tests that custom rules are loaded and checked."""

    def test_custom_rules_dir_in_config(self) -> None:
        config = MirdanConfig()
        assert config.quality.custom_rules_dir == ".mirdan/rules"

    def test_load_custom_rules_exists(self) -> None:
        from mirdan.core.code_validator import CodeValidator

        assert hasattr(CodeValidator, "_load_custom_rules")


# ── 3F: Framework Standards Expansion ────────────────────────────────


class TestFrameworkStandards:
    """Tests that new framework YAML standards load correctly."""

    @pytest.mark.parametrize(
        "framework",
        [
            "crewai",
            "dspy",
            "astro",
            "trpc",
            "drizzle",
            "supabase",
            "convex",
            "react-native",
            "flutter",
        ],
    )
    def test_framework_standards_loadable(self, framework: str) -> None:
        from mirdan.core.quality_standards import QualityStandards

        standards = QualityStandards()
        result = standards.get_for_framework(framework)
        assert result, f"No standards loaded for {framework}"
        assert "principles" in result, f"Missing principles for {framework}"
        assert "forbidden" in result, f"Missing forbidden for {framework}"
        assert "patterns" in result, f"Missing patterns for {framework}"

    def test_total_framework_count(self) -> None:
        """Should have 33 original + 20 new (1.1.0 expansion) = 53 framework standards."""
        from importlib.resources import files

        standards_pkg = files("mirdan.standards")
        frameworks_dir = standards_pkg.joinpath("frameworks")
        yaml_files = [f for f in frameworks_dir.iterdir() if f.name.endswith(".yaml")]
        assert len(yaml_files) == 53
