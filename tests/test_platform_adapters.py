"""Tests for platform adapter base class and concrete adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from mirdan.cli.detect import DetectedProject
from mirdan.integrations.base import PlatformAdapter


class TestPlatformAdapterBase:
    """Tests for the abstract PlatformAdapter."""

    def test_cannot_instantiate_abstract(self) -> None:
        """PlatformAdapter should not be directly instantiable."""
        detected = DetectedProject(primary_language="python")
        with pytest.raises(TypeError):
            PlatformAdapter(Path("/tmp"), detected)  # type: ignore[abstract]

    def test_concrete_subclass_works(self, tmp_path: Path) -> None:
        """A concrete subclass implementing all methods should work."""

        class DummyAdapter(PlatformAdapter):
            def generate_hooks(self) -> list[Path]:
                return []

            def generate_rules(self) -> list[Path]:
                return []

            def generate_agents(self) -> list[Path]:
                return []

            def generate_mcp_config(self) -> Path | None:
                return None

        detected = DetectedProject(primary_language="python")
        adapter = DummyAdapter(tmp_path, detected)
        assert adapter.project_dir == tmp_path
        assert adapter.detected is detected
        assert adapter.standards is None

    def test_generate_all_combines_results(self, tmp_path: Path) -> None:
        """generate_all() should combine all generator results."""

        class MockAdapter(PlatformAdapter):
            def generate_hooks(self) -> list[Path]:
                return [tmp_path / "hooks.json"]

            def generate_rules(self) -> list[Path]:
                return [tmp_path / "rule1", tmp_path / "rule2"]

            def generate_agents(self) -> list[Path]:
                return [tmp_path / "agents.md"]

            def generate_mcp_config(self) -> Path | None:
                return tmp_path / "mcp.json"

        detected = DetectedProject(primary_language="python")
        adapter = MockAdapter(tmp_path, detected)
        paths = adapter.generate_all()
        assert len(paths) == 5

    def test_generate_all_without_mcp_config(self, tmp_path: Path) -> None:
        """generate_all() should work when generate_mcp_config returns None."""

        class NoMCPAdapter(PlatformAdapter):
            def generate_hooks(self) -> list[Path]:
                return [tmp_path / "hooks"]

            def generate_rules(self) -> list[Path]:
                return []

            def generate_agents(self) -> list[Path]:
                return []

            def generate_mcp_config(self) -> Path | None:
                return None

        detected = DetectedProject(primary_language="python")
        paths = NoMCPAdapter(tmp_path, detected).generate_all()
        assert len(paths) == 1

    def test_standards_param_stored(self, tmp_path: Path) -> None:
        """Standards parameter should be stored on the adapter."""

        class DummyAdapter(PlatformAdapter):
            def generate_hooks(self) -> list[Path]:
                return []

            def generate_rules(self) -> list[Path]:
                return []

            def generate_agents(self) -> list[Path]:
                return []

            def generate_mcp_config(self) -> Path | None:
                return None

        detected = DetectedProject(primary_language="python")
        sentinel = object()
        adapter = DummyAdapter(tmp_path, detected, standards=sentinel)  # type: ignore[arg-type]
        assert adapter.standards is sentinel


class TestClaudeCodeAdapter:
    """Tests for ClaudeCodeAdapter."""

    def test_instantiation(self, tmp_path: Path) -> None:
        """Should instantiate with required params."""
        from mirdan.integrations.claude_code import ClaudeCodeAdapter

        detected = DetectedProject(primary_language="python")
        adapter = ClaudeCodeAdapter(tmp_path, detected)
        assert adapter.project_dir == tmp_path
        assert adapter.detected is detected

    def test_generate_all_creates_files(self, tmp_path: Path) -> None:
        """generate_all() should create hooks, rules, agents, skills, and MCP config."""
        from mirdan.integrations.claude_code import ClaudeCodeAdapter

        detected = DetectedProject(primary_language="python")
        adapter = ClaudeCodeAdapter(tmp_path, detected)
        paths = adapter.generate_all()

        # Should create at least some files (skills may fail without templates)
        assert len(paths) >= 1
        # MCP config should be created
        assert (tmp_path / ".mcp.json").exists()

    def test_has_generate_skills(self) -> None:
        """ClaudeCodeAdapter should have generate_skills method."""
        from mirdan.integrations.claude_code import ClaudeCodeAdapter

        assert hasattr(ClaudeCodeAdapter, "generate_skills")

    def test_generate_mcp_config(self, tmp_path: Path) -> None:
        """generate_mcp_config should create .mcp.json."""
        from mirdan.integrations.claude_code import ClaudeCodeAdapter

        detected = DetectedProject(primary_language="python")
        adapter = ClaudeCodeAdapter(tmp_path, detected)
        mcp_path = adapter.generate_mcp_config()
        assert mcp_path is not None
        assert mcp_path.exists()


class TestCursorAdapter:
    """Tests for CursorAdapter."""

    def test_instantiation(self, tmp_path: Path) -> None:
        """Should instantiate with required params."""
        from mirdan.integrations.cursor import CursorAdapter

        detected = DetectedProject(primary_language="python")
        adapter = CursorAdapter(tmp_path, detected)
        assert adapter.project_dir == tmp_path

    def test_generate_all_creates_expected_files(self, tmp_path: Path) -> None:
        """generate_all() should create hooks, rules, agents, bugbot, and mcp.json."""
        from mirdan.integrations.cursor import CursorAdapter

        detected = DetectedProject(primary_language="python")
        adapter = CursorAdapter(tmp_path, detected)
        paths = adapter.generate_all()

        names = [p.name for p in paths]
        # Should have hooks.json, at least one .mdc rule, AGENTS.md, BUGBOT.md, mcp.json
        assert "hooks.json" in names
        assert "AGENTS.md" in names
        assert "BUGBOT.md" in names
        assert "mcp.json" in names
        assert any(n.endswith(".mdc") for n in names)

    def test_generate_mcp_config(self, tmp_path: Path) -> None:
        """generate_mcp_config should create .cursor/mcp.json."""
        from mirdan.integrations.cursor import CursorAdapter

        detected = DetectedProject(primary_language="python")
        adapter = CursorAdapter(tmp_path, detected)
        mcp_path = adapter.generate_mcp_config()
        assert mcp_path is not None
        assert mcp_path.name == "mcp.json"
        assert mcp_path.exists()

    def test_accepts_detected_project(self, tmp_path: Path) -> None:
        """CursorAdapter should accept DetectedProject correctly."""
        from mirdan.integrations.cursor import CursorAdapter

        detected = DetectedProject(
            primary_language="typescript",
            project_name="my-app",
            frameworks=["react", "next.js"],
        )
        adapter = CursorAdapter(tmp_path, detected)
        assert adapter.detected.primary_language == "typescript"
        assert adapter.detected.project_name == "my-app"


class TestCursorMcpJsonSequentialThinking:
    """Tests for sequential-thinking MCP in generated mcp.json."""

    def test_fresh_config_includes_sequential_thinking(self, tmp_path: Path) -> None:
        """Fresh mcp.json should include sequential-thinking server."""
        import json

        from mirdan.integrations.cursor import generate_cursor_mcp_json

        cursor_dir = tmp_path / ".cursor"
        generate_cursor_mcp_json(cursor_dir)
        data = json.loads((cursor_dir / "mcp.json").read_text())
        assert "sequential-thinking" in data["mcpServers"]
        st_config = data["mcpServers"]["sequential-thinking"]
        assert st_config["command"] == "npx"
        assert "@modelcontextprotocol/server-sequential-thinking" in st_config["args"]

    def test_merge_preserves_existing_sequential_thinking(
        self, tmp_path: Path
    ) -> None:
        """Merge should NOT overwrite user's existing sequential-thinking config."""
        import json

        from mirdan.integrations.cursor import generate_cursor_mcp_json

        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir(parents=True)
        existing = {
            "mcpServers": {
                "sequential-thinking": {
                    "type": "stdio",
                    "command": "node",
                    "args": ["/custom/path/server.js"],
                }
            }
        }
        (cursor_dir / "mcp.json").write_text(json.dumps(existing))

        generate_cursor_mcp_json(cursor_dir)
        data = json.loads((cursor_dir / "mcp.json").read_text())
        # User's custom config should be preserved
        assert data["mcpServers"]["sequential-thinking"]["command"] == "node"
        assert "/custom/path/server.js" in data["mcpServers"]["sequential-thinking"]["args"]
        # mirdan should still be added
        assert "mirdan" in data["mcpServers"]

    def test_merge_adds_sequential_thinking_to_existing(self, tmp_path: Path) -> None:
        """Merge should add sequential-thinking when user has other MCPs but not ST."""
        import json

        from mirdan.integrations.cursor import generate_cursor_mcp_json

        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir(parents=True)
        existing = {
            "mcpServers": {
                "some-other-mcp": {"type": "stdio", "command": "other"}
            }
        }
        (cursor_dir / "mcp.json").write_text(json.dumps(existing))

        generate_cursor_mcp_json(cursor_dir)
        data = json.loads((cursor_dir / "mcp.json").read_text())
        assert "sequential-thinking" in data["mcpServers"]
        assert "some-other-mcp" in data["mcpServers"]
        assert "mirdan" in data["mcpServers"]
