"""Tests for Cursor sandbox access control generation."""

from __future__ import annotations

import json
from pathlib import Path

from mirdan.cli.detect import DetectedProject
from mirdan.integrations.cursor import generate_cursor_sandbox


class TestGenerateCursorSandbox:
    """Tests for generate_cursor_sandbox()."""

    def test_generates_sandbox_json(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        detected = DetectedProject(primary_language="python")
        result = generate_cursor_sandbox(cursor_dir, detected)
        assert result is not None
        assert result.name == "sandbox.json"
        assert result.exists()

    def test_sandbox_has_deny_default_network(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        detected = DetectedProject(primary_language="python")
        result = generate_cursor_sandbox(cursor_dir, detected)
        assert result is not None
        config = json.loads(result.read_text())
        assert config["networkPolicy"]["default"] == "deny"

    def test_sandbox_allows_package_registries(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        detected = DetectedProject(primary_language="python")
        result = generate_cursor_sandbox(cursor_dir, detected)
        assert result is not None
        config = json.loads(result.read_text())
        allow = config["networkPolicy"]["allow"]
        assert "pypi.org" in allow
        assert "registry.npmjs.org" in allow
        assert "github.com" in allow

    def test_sandbox_adds_rust_registries(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        detected = DetectedProject(primary_language="rust")
        result = generate_cursor_sandbox(cursor_dir, detected)
        assert result is not None
        config = json.loads(result.read_text())
        allow = config["networkPolicy"]["allow"]
        assert "crates.io" in allow
        assert "static.crates.io" in allow

    def test_sandbox_adds_go_registries(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        detected = DetectedProject(primary_language="go")
        result = generate_cursor_sandbox(cursor_dir, detected)
        assert result is not None
        config = json.loads(result.read_text())
        allow = config["networkPolicy"]["allow"]
        assert "proxy.golang.org" in allow
        assert "sum.golang.org" in allow

    def test_sandbox_idempotent(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        detected = DetectedProject(primary_language="python")
        first = generate_cursor_sandbox(cursor_dir, detected)
        assert first is not None
        original_content = first.read_text()

        second = generate_cursor_sandbox(cursor_dir, detected)
        assert second is None
        assert first.read_text() == original_content

    def test_sandbox_type_workspace_readwrite(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        detected = DetectedProject(primary_language="python")
        result = generate_cursor_sandbox(cursor_dir, detected)
        assert result is not None
        config = json.loads(result.read_text())
        assert config["type"] == "workspace_readwrite"
