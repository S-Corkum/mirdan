"""Tests for Phase 4: Make It Distributed — plugins, team profiles, SARIF, dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mirdan.core.quality_profiles import QualityProfile
from mirdan.core.team_profiles import (
    ComplianceRule,
    TeamProfile,
    list_team_profiles,
    load_team_profile,
    resolve_team_profile,
    save_team_profile,
)
from mirdan.integrations.mcp_apps import QualityDashboard
from mirdan.integrations.plugin_validator import PluginValidationResult, PluginValidator
from mirdan.integrations.sarif import SARIFExporter

# ── 4D: Team Quality Profiles ────────────────────────────────────────


class TestTeamProfile:
    """Tests for TeamProfile."""

    def test_create_basic(self) -> None:
        tp = TeamProfile(name="backend", description="Backend team")
        assert tp.name == "backend"
        assert tp.extends == "default"
        assert tp.security == 0.7

    def test_to_quality_profile(self) -> None:
        tp = TeamProfile(
            name="sec-team",
            description="Security team",
            team_name="security",
            security=1.0,
        )
        qp = tp.to_quality_profile()
        assert isinstance(qp, QualityProfile)
        assert qp.security == 1.0
        assert qp.metadata["team_name"] == "security"

    def test_to_dict_and_from_dict(self) -> None:
        tp = TeamProfile(
            name="test",
            description="Test profile",
            team_name="eng",
            org_name="acme",
            compliance=[
                ComplianceRule(name="sec-review", description="Security review required"),
            ],
            custom_thresholds={"security": 0.9},
        )
        d = tp.to_dict()
        restored = TeamProfile.from_dict(d)
        assert restored.name == "test"
        assert restored.team_name == "eng"
        assert len(restored.compliance) == 1
        assert restored.compliance[0].name == "sec-review"
        assert restored.custom_thresholds["security"] == 0.9

    def test_save_and_load(self, tmp_path: Path) -> None:
        tp = TeamProfile(
            name="my-team",
            description="My team profile",
            security=0.9,
        )
        save_team_profile(tp, tmp_path)
        loaded = load_team_profile("my-team", tmp_path)
        assert loaded is not None
        assert loaded.name == "my-team"
        assert loaded.security == 0.9

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        assert load_team_profile("nope", tmp_path) is None

    def test_list_team_profiles(self, tmp_path: Path) -> None:
        tp1 = TeamProfile(name="team-a", description="Team A")
        tp2 = TeamProfile(name="team-b", description="Team B")
        save_team_profile(tp1, tmp_path)
        save_team_profile(tp2, tmp_path)
        profiles = list_team_profiles(tmp_path)
        names = {p["name"] for p in profiles}
        assert names == {"team-a", "team-b"}

    def test_list_empty_dir(self, tmp_path: Path) -> None:
        assert list_team_profiles(tmp_path) == []

    def test_resolve_inherits_from_enterprise(self) -> None:
        tp = TeamProfile(
            name="sec-team",
            description="Security team",
            extends="enterprise",
        )
        resolved = resolve_team_profile(tp)
        # Enterprise has security=1.0, and the TeamProfile default is 0.7
        # Since 0.7 IS the default, it should inherit enterprise's value
        assert resolved.security == 1.0


class TestComplianceRule:
    def test_roundtrip(self) -> None:
        rule = ComplianceRule(
            name="test-rule",
            description="Test",
            required=True,
            min_score=0.8,
        )
        d = rule.to_dict()
        restored = ComplianceRule.from_dict(d)
        assert restored.name == "test-rule"
        assert restored.min_score == 0.8


# ── 4A: Enhanced Plugin + PluginValidator ────────────────────────────


class TestPluginValidator:
    """Tests for PluginValidator."""

    def test_valid_plugin(self, tmp_path: Path) -> None:
        # Create minimal valid plugin
        plugin_meta = tmp_path / ".claude-plugin"
        plugin_meta.mkdir()
        (plugin_meta / "plugin.json").write_text(
            json.dumps(
                {
                    "name": "mirdan",
                    "version": "0.4.0",
                    "description": "test",
                }
            )
        )
        (tmp_path / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"mirdan": {"type": "stdio", "command": "mirdan"}}})
        )

        validator = PluginValidator()
        result = validator.validate(tmp_path)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_missing_manifest(self, tmp_path: Path) -> None:
        validator = PluginValidator()
        result = validator.validate(tmp_path)
        assert result.valid is False
        assert any("plugin.json" in e for e in result.errors)

    def test_invalid_json_manifest(self, tmp_path: Path) -> None:
        plugin_meta = tmp_path / ".claude-plugin"
        plugin_meta.mkdir()
        (plugin_meta / "plugin.json").write_text("not json")

        validator = PluginValidator()
        result = validator.validate(tmp_path)
        assert result.valid is False

    def test_missing_required_fields(self, tmp_path: Path) -> None:
        plugin_meta = tmp_path / ".claude-plugin"
        plugin_meta.mkdir()
        (plugin_meta / "plugin.json").write_text(json.dumps({"name": "test"}))

        validator = PluginValidator()
        result = validator.validate(tmp_path)
        assert result.valid is False
        assert any("version" in e for e in result.errors)

    def test_detects_bad_mcp_prefix(self, tmp_path: Path) -> None:
        plugin_meta = tmp_path / ".claude-plugin"
        plugin_meta.mkdir()
        (plugin_meta / "plugin.json").write_text(
            json.dumps(
                {
                    "name": "test",
                    "version": "1.0",
                    "description": "test",
                }
            )
        )
        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        (rules_dir / "test.md").write_text("Use mcp_mirdan_validate for checking")

        validator = PluginValidator()
        result = validator.validate(tmp_path)
        assert result.valid is False
        assert any("mcp_mirdan_" in e for e in result.errors)

    def test_not_a_directory(self, tmp_path: Path) -> None:
        fake = tmp_path / "not-a-dir.txt"
        fake.write_text("nope")
        validator = PluginValidator()
        result = validator.validate(fake)
        assert result.valid is False

    def test_result_to_dict(self) -> None:
        result = PluginValidationResult(
            valid=True,
            errors=[],
            warnings=["test warning"],
            files_found=["plugin.json"],
        )
        d = result.to_dict()
        assert d["valid"] is True
        assert len(d["warnings"]) == 1


class TestEnhancedPluginManifest:
    """Tests that export_plugin produces enhanced manifest."""

    def test_manifest_has_categories(self, tmp_path: Path) -> None:
        from mirdan.integrations.claude_code import export_plugin

        export_plugin(tmp_path)
        manifest = json.loads((tmp_path / ".claude-plugin" / "plugin.json").read_text())
        assert "categories" in manifest
        assert "skills" in manifest
        assert "agents" in manifest
        assert "hooks" in manifest
        assert manifest["hooks"] is True
        assert "mcpServers" in manifest

    def test_export_includes_hooks_json(self, tmp_path: Path) -> None:
        from mirdan.integrations.claude_code import export_plugin

        export_plugin(tmp_path)
        hooks_path = tmp_path / "hooks.json"
        assert hooks_path.exists()
        data = json.loads(hooks_path.read_text())
        assert isinstance(data, dict)

    def test_export_includes_rules(self, tmp_path: Path) -> None:
        from mirdan.integrations.claude_code import export_plugin

        export_plugin(tmp_path)
        rules_dir = tmp_path / "rules"
        assert rules_dir.exists()
        md_files = list(rules_dir.glob("*.md"))
        assert len(md_files) > 0


# ── 4B: Cursor Plugin Export ──────────────────────────────────────────


class TestCursorPluginExporter:
    """Tests for CursorPluginExporter."""

    def test_export_creates_manifest(self, tmp_path: Path) -> None:
        from mirdan.integrations.cursor_plugin import CursorPluginExporter

        exporter = CursorPluginExporter()
        exporter.export(tmp_path)
        manifest = tmp_path / "manifest.json"
        assert manifest.exists()
        data = json.loads(manifest.read_text())
        assert data["platform"] == "cursor"

    def test_export_creates_mcp_json(self, tmp_path: Path) -> None:
        from mirdan.integrations.cursor_plugin import CursorPluginExporter

        exporter = CursorPluginExporter()
        exporter.export(tmp_path)
        mcp = tmp_path / "mcp.json"
        assert mcp.exists()
        data = json.loads(mcp.read_text())
        assert "mirdan" in data["mcpServers"]

    def test_export_creates_rules(self, tmp_path: Path) -> None:
        from mirdan.integrations.cursor_plugin import CursorPluginExporter

        exporter = CursorPluginExporter()
        exporter.export(tmp_path)
        rules_dir = tmp_path / "rules"
        assert rules_dir.exists()


class TestPluginCommandFlags:
    """Tests that plugin command supports --cursor and --all."""

    def test_plugin_command_source_has_cursor(self) -> None:
        import inspect

        from mirdan.cli.plugin_command import _export

        source = inspect.getsource(_export)
        assert "--cursor" in source
        assert "--all" in source


# ── 4E: SARIF Export ──────────────────────────────────────────────────


class TestSARIFExporter:
    """Tests for SARIFExporter."""

    def test_empty_result(self) -> None:
        exporter = SARIFExporter()
        sarif = exporter.export({"violations": [], "score": 1.0, "passed": True})
        assert sarif["version"] == "2.1.0"
        assert len(sarif["runs"]) == 1
        assert sarif["runs"][0]["results"] == []

    def test_with_violations(self) -> None:
        violations = [
            {
                "rule_id": "SEC001",
                "rule": "Hardcoded secret",
                "severity": "error",
                "message": "Found hardcoded API key",
                "suggestion": "Use environment variables",
                "line": 10,
                "file": "app.py",
            },
            {
                "rule_id": "PY005",
                "rule": "Missing type hint",
                "severity": "warning",
                "message": "Function lacks type hints",
                "line": 20,
                "file": "utils.py",
            },
        ]
        exporter = SARIFExporter()
        sarif = exporter.export(
            {
                "violations": violations,
                "score": 0.5,
                "passed": False,
            }
        )
        results = sarif["runs"][0]["results"]
        assert len(results) == 2
        assert results[0]["ruleId"] == "SEC001"
        assert results[0]["level"] == "error"
        assert results[1]["ruleId"] == "PY005"
        assert results[1]["level"] == "warning"

    def test_rules_extraction(self) -> None:
        violations = [
            {"rule_id": "SEC001", "rule": "test", "severity": "error"},
            {"rule_id": "SEC001", "rule": "test", "severity": "error"},
        ]
        exporter = SARIFExporter()
        sarif = exporter.export({"violations": violations})
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        # Should deduplicate
        assert len(rules) == 1

    def test_sarif_schema_present(self) -> None:
        exporter = SARIFExporter()
        sarif = exporter.export({"violations": []})
        assert "$schema" in sarif


# ── 4E: Enhanced GitHub CI/CD ────────────────────────────────────────


class TestGitHubCI:
    """Tests for enhanced GitHub CI functions."""

    def test_generate_quality_badge(self) -> None:
        from mirdan.integrations.github_ci import generate_quality_badge

        url = generate_quality_badge(0.95)
        assert "brightgreen" in url
        assert "95" in url

        url = generate_quality_badge(0.5)
        assert "yellow" in url

        url = generate_quality_badge(0.3)
        assert "red" in url

    def test_generate_pr_comment(self) -> None:
        from mirdan.integrations.github_ci import generate_pr_comment

        result: dict[str, Any] = {
            "score": 0.85,
            "passed": True,
            "violations": [
                {"severity": "warning", "rule_id": "PY001", "message": "test"},
            ],
        }
        comment = generate_pr_comment(result)
        assert "PASS" in comment
        assert "85.0%" in comment

    def test_generate_pr_comment_fail(self) -> None:
        from mirdan.integrations.github_ci import generate_pr_comment

        result: dict[str, Any] = {
            "score": 0.4,
            "passed": False,
            "violations": [
                {"severity": "error", "rule_id": "SEC001", "message": "bad"},
            ],
        }
        comment = generate_pr_comment(result)
        assert "FAIL" in comment
        assert "SEC001" in comment

    def test_generate_sarif_workflow(self, tmp_path: Path) -> None:
        from mirdan.integrations.github_ci import generate_sarif_workflow

        path = generate_sarif_workflow(tmp_path)
        assert path is not None
        assert path.exists()
        content = path.read_text()
        assert "sarif" in content.lower()

    def test_generate_pr_quality_check(self, tmp_path: Path) -> None:
        from mirdan.integrations.github_ci import generate_pr_quality_check

        path = generate_pr_quality_check(tmp_path)
        assert path is not None
        assert path.exists()


# ── 4C: MCP Apps Quality Dashboard ────────────────────────────────────


class TestQualityDashboard:
    """Tests for QualityDashboard data layer."""

    def test_score_timeline(self) -> None:
        dashboard = QualityDashboard()
        trends = {
            "snapshots": [
                {"date": "2026-03-01", "score": 0.8},
                {"date": "2026-03-02", "score": 0.85},
                {"date": "2026-03-03", "score": 0.9},
            ],
            "trend_direction": "improving",
        }
        result = dashboard.score_timeline(trends)
        assert result["type"] == "line"
        assert len(result["data"]["labels"]) == 3
        assert result["summary"]["trend"] == "improving"

    def test_violation_breakdown(self) -> None:
        dashboard = QualityDashboard()
        result_data: dict[str, Any] = {
            "violations": [
                {"category": "security"},
                {"category": "security"},
                {"category": "style"},
            ],
        }
        result = dashboard.violation_breakdown(result_data)
        assert result["type"] == "pie"
        assert result["summary"]["total"] == 3
        assert result["summary"]["top_category"] == "security"

    def test_session_overview(self) -> None:
        dashboard = QualityDashboard()
        session = {
            "validation_count": 5,
            "avg_score": 0.85,
            "files_validated": 10,
            "unresolved_errors": 2,
        }
        result = dashboard.session_overview(session)
        assert result["type"] == "stats"
        assert result["data"]["validation_count"] == 5

    def test_compliance_matrix(self) -> None:
        dashboard = QualityDashboard()
        profile = {
            "name": "enterprise",
            "security": 1.0,
            "architecture": 0.9,
            "testing": 0.9,
            "documentation": 0.8,
            "ai_slop_detection": 1.0,
            "performance": 0.7,
        }
        result = dashboard.compliance_matrix(profile)
        assert result["type"] == "grid"
        assert result["summary"]["dimensions"] == 6
        assert result["summary"]["profile_name"] == "enterprise"

    def test_empty_timeline(self) -> None:
        dashboard = QualityDashboard()
        result = dashboard.score_timeline({"snapshots": []})
        assert result["summary"]["count"] == 0

    def test_empty_violation_breakdown(self) -> None:
        dashboard = QualityDashboard()
        result = dashboard.violation_breakdown({"violations": []})
        assert result["summary"]["total"] == 0
        assert result["summary"]["top_category"] is None


# ── 4C: Dashboard format in get_quality_trends ────────────────────────


class TestDashboardFormat:
    """Tests that get_quality_trends supports format=dashboard."""

    def test_format_parameter_exists(self) -> None:
        import inspect

        import mirdan.server as server_mod

        # get_quality_trends is wrapped by FastMCP as a FunctionTool;
        # access the underlying function via .fn attribute.
        fn = server_mod.get_quality_trends
        underlying = getattr(fn, "fn", fn)
        sig = inspect.signature(underlying)  # type: ignore[arg-type]
        assert "format" in sig.parameters


# ── 4E: Export CLI ────────────────────────────────────────────────────


class TestExportCLI:
    """Tests for export command registration."""

    def test_export_command_registered(self) -> None:
        import inspect

        import mirdan.cli as cli_mod

        source = inspect.getsource(cli_mod.main)
        assert "export" in source
