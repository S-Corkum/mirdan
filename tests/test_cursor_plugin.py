"""Tests for Cursor plugin export."""

from __future__ import annotations

import json
from pathlib import Path

from mirdan.integrations.cursor_plugin import CursorPluginExporter


class TestCursorPluginExporter:
    """Tests for CursorPluginExporter."""

    def test_export_creates_plugin_json(self, tmp_path: Path) -> None:
        exporter = CursorPluginExporter()
        exporter.export(tmp_path)
        plugin_json = tmp_path / ".cursor-plugin" / "plugin.json"
        assert plugin_json.exists()

    def test_export_does_not_create_old_manifest(self, tmp_path: Path) -> None:
        exporter = CursorPluginExporter()
        exporter.export(tmp_path)
        assert not (tmp_path / "manifest.json").exists()

    def test_manifest_has_required_fields(self, tmp_path: Path) -> None:
        exporter = CursorPluginExporter()
        exporter.export(tmp_path)
        plugin_json = tmp_path / ".cursor-plugin" / "plugin.json"
        manifest = json.loads(plugin_json.read_text())
        assert manifest["name"] == "mirdan"
        assert "description" in manifest
        assert "version" in manifest
        assert isinstance(manifest["author"], dict)
        assert "name" in manifest["author"]
        assert manifest["license"] == "MIT"
        assert isinstance(manifest["keywords"], list)

    def test_manifest_has_component_paths(self, tmp_path: Path) -> None:
        exporter = CursorPluginExporter()
        exporter.export(tmp_path)
        plugin_json = tmp_path / ".cursor-plugin" / "plugin.json"
        manifest = json.loads(plugin_json.read_text())
        assert "rules" in manifest
        assert "agents" in manifest
        assert "skills" in manifest
        assert "commands" in manifest
        assert "hooks" in manifest
        assert "mcpServers" in manifest

    def test_cursor_plugin_directory_created(self, tmp_path: Path) -> None:
        exporter = CursorPluginExporter()
        exporter.export(tmp_path)
        assert (tmp_path / ".cursor-plugin").is_dir()

    def test_sandbox_json_generated(self, tmp_path: Path) -> None:
        exporter = CursorPluginExporter()
        exporter.export(tmp_path)
        assert (tmp_path / "sandbox.json").exists()

    def test_export_returns_output_dir(self, tmp_path: Path) -> None:
        exporter = CursorPluginExporter()
        result = exporter.export(tmp_path)
        assert result == tmp_path
