"""Tests for Cursor environment.json generation."""

from __future__ import annotations

import json
from pathlib import Path

from mirdan.cli.detect import DetectedProject
from mirdan.integrations.cursor import generate_cursor_environment


class TestGenerateCursorEnvironment:
    """Tests for generate_cursor_environment()."""

    def test_generates_environment_json(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        detected = DetectedProject(primary_language="python")
        result = generate_cursor_environment(cursor_dir, detected)
        assert result is not None
        assert result.name == "environment.json"
        assert result.exists()

    def test_environment_json_is_valid_json(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        detected = DetectedProject(primary_language="python")
        result = generate_cursor_environment(cursor_dir, detected)
        assert result is not None
        content = json.loads(result.read_text())
        assert isinstance(content, dict)

    def test_has_name_field(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        detected = DetectedProject(primary_language="python")
        result = generate_cursor_environment(cursor_dir, detected)
        assert result is not None
        content = json.loads(result.read_text())
        assert "name" in content
        assert content["name"] == "mirdan-quality"

    def test_has_install_field_with_mirdan(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        detected = DetectedProject(primary_language="python")
        result = generate_cursor_environment(cursor_dir, detected)
        assert result is not None
        content = json.loads(result.read_text())
        assert "install" in content
        assert "mirdan" in content["install"]

    def test_has_terminals_array(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        detected = DetectedProject(primary_language="python")
        result = generate_cursor_environment(cursor_dir, detected)
        assert result is not None
        content = json.loads(result.read_text())
        assert "terminals" in content
        assert isinstance(content["terminals"], list)
        assert len(content["terminals"]) >= 1

    def test_terminal_has_required_fields(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        detected = DetectedProject(primary_language="python")
        result = generate_cursor_environment(cursor_dir, detected)
        assert result is not None
        content = json.loads(result.read_text())
        terminal = content["terminals"][0]
        assert "command" in terminal  # required per schema

    def test_idempotent_returns_none_if_exists(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        detected = DetectedProject(primary_language="python")

        first = generate_cursor_environment(cursor_dir, detected)
        assert first is not None

        second = generate_cursor_environment(cursor_dir, detected)
        assert second is None

    def test_does_not_overwrite_existing(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir(parents=True)
        env_path = cursor_dir / "environment.json"
        env_path.write_text('{"custom": true}')

        detected = DetectedProject(primary_language="python")
        result = generate_cursor_environment(cursor_dir, detected)
        assert result is None
        assert json.loads(env_path.read_text()) == {"custom": True}

    def test_creates_cursor_directory(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        assert not cursor_dir.exists()
        detected = DetectedProject(primary_language="python")
        generate_cursor_environment(cursor_dir, detected)
        assert cursor_dir.is_dir()

    def test_uv_detection_uses_uv_pip(self, tmp_path: Path) -> None:
        """If uv.lock exists, install command should use uv pip."""
        project_dir = tmp_path
        (project_dir / "uv.lock").write_text("")
        cursor_dir = project_dir / ".cursor"
        detected = DetectedProject(primary_language="python")
        result = generate_cursor_environment(cursor_dir, detected)
        assert result is not None
        content = json.loads(result.read_text())
        assert "uv pip install mirdan" in content["install"]

    def test_default_install_uses_pip(self, tmp_path: Path) -> None:
        """Without uv indicators, install should use plain pip."""
        cursor_dir = tmp_path / ".cursor"
        detected = DetectedProject(primary_language="typescript")
        result = generate_cursor_environment(cursor_dir, detected)
        assert result is not None
        content = json.loads(result.read_text())
        assert content["install"] == "pip install mirdan"

    def test_no_unevaluated_properties(self, tmp_path: Path) -> None:
        """Schema is strict — only known fields should be present."""
        cursor_dir = tmp_path / ".cursor"
        detected = DetectedProject(primary_language="python")
        result = generate_cursor_environment(cursor_dir, detected)
        assert result is not None
        content = json.loads(result.read_text())
        allowed_keys = {
            "name",
            "user",
            "build",
            "snapshot",
            "agentCanUpdateSnapshot",
            "install",
            "start",
            "terminals",
            "ports",
            "repositoryDependencies",
        }
        assert set(content.keys()).issubset(allowed_keys)
