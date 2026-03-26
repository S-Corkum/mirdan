"""Tests for plugin export and skill/agent generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mirdan.cli.detect import DetectedProject
from mirdan.integrations.claude_code import (
    export_plugin,
    generate_agents,
    generate_mcp_json,
    generate_skills,
)


@pytest.fixture()
def detected() -> DetectedProject:
    """Minimal detected project for testing."""
    return DetectedProject(
        project_type="python",
        project_name="test-project",
        primary_language="python",
        frameworks=["fastapi"],
    )


# ---------------------------------------------------------------------------
# generate_skills
# ---------------------------------------------------------------------------


class TestGenerateSkills:
    """Tests for skill file generation."""

    def test_creates_seven_skills(self, tmp_path: Path, detected: DetectedProject) -> None:
        paths = generate_skills(tmp_path, detected)
        assert len(paths) == 8

    def test_skill_names(self, tmp_path: Path, detected: DetectedProject) -> None:
        paths = generate_skills(tmp_path, detected)
        names = {p.parent.name for p in paths}
        assert names == {
            "code",
            "debug",
            "review",
            "plan",
            "plan-review",
            "quality",
            "scan",
            "gate",
        }

    def test_skills_are_valid_markdown(self, tmp_path: Path, detected: DetectedProject) -> None:
        paths = generate_skills(tmp_path, detected)
        for path in paths:
            content = path.read_text()
            assert content.startswith("---")
            assert "description:" in content

    def test_skills_have_frontmatter(self, tmp_path: Path, detected: DetectedProject) -> None:
        paths = generate_skills(tmp_path, detected)
        for path in paths:
            content = path.read_text()
            # Check YAML frontmatter delimiters
            parts = content.split("---")
            assert len(parts) >= 3  # empty + frontmatter + body

    def test_skills_reference_mirdan_tools(self, tmp_path: Path, detected: DetectedProject) -> None:
        paths = generate_skills(tmp_path, detected)
        for path in paths:
            content = path.read_text()
            assert "mcp__mirdan__" in content

    def test_skill_descriptions_under_1024_chars(
        self, tmp_path: Path, detected: DetectedProject
    ) -> None:
        paths = generate_skills(tmp_path, detected)
        for path in paths:
            content = path.read_text()
            # Extract description from frontmatter
            parts = content.split("---")
            frontmatter = parts[1]
            for line in frontmatter.strip().splitlines():
                if line.startswith("description:"):
                    desc = line[len("description:") :].strip()
                    assert len(desc) < 1024, f"Description too long in {path.name}"


# ---------------------------------------------------------------------------
# generate_agents
# ---------------------------------------------------------------------------


class TestGenerateAgents:
    """Tests for agent file generation."""

    def test_creates_all_agents(self, tmp_path: Path, detected: DetectedProject) -> None:
        paths = generate_agents(tmp_path, detected)
        assert len(paths) == 6
        names = {p.name for p in paths}
        assert "quality-gate.md" in names
        assert "security-audit.md" in names
        assert "test-quality.md" in names
        assert "architecture-reviewer.md" in names
        assert "convention-check.md" in names
        assert "plan-reviewer.md" in names

    def test_agent_has_frontmatter(self, tmp_path: Path, detected: DetectedProject) -> None:
        paths = generate_agents(tmp_path, detected)
        for path in paths:
            content = path.read_text()
            assert content.startswith("---"), f"{path.name} missing frontmatter"
            assert "name:" in content, f"{path.name} missing name"
            assert "model:" in content, f"{path.name} missing model"

    def test_no_agents_have_unsupported_fields(
        self, tmp_path: Path, detected: DetectedProject
    ) -> None:
        """memory:, background:, skills:, isolation:, mcpServers: are not valid Claude Code agent fields."""
        paths = generate_agents(tmp_path, detected)
        for path in paths:
            content = path.read_text()
            assert "memory:" not in content, f"{path.name} has unsupported memory: field"
            assert "background:" not in content, f"{path.name} has unsupported background: field"
            assert "isolation:" not in content, f"{path.name} has unsupported isolation: field"
            assert "mcpServers:" not in content, f"{path.name} has unsupported mcpServers: field"

    def test_quality_gate_references_mirdan_tools(
        self, tmp_path: Path, detected: DetectedProject
    ) -> None:
        paths = generate_agents(tmp_path, detected)
        qg = next(p for p in paths if p.name == "quality-gate.md")
        content = qg.read_text()
        assert "mcp__mirdan__validate_code_quality" in content


# ---------------------------------------------------------------------------
# generate_mcp_json
# ---------------------------------------------------------------------------


class TestGenerateMcpJson:
    """Tests for .mcp.json generation."""

    def test_creates_mcp_json(self, tmp_path: Path) -> None:
        path = generate_mcp_json(tmp_path)
        assert path.exists()
        assert path.name == ".mcp.json"

    def test_valid_json(self, tmp_path: Path) -> None:
        path = generate_mcp_json(tmp_path)
        data = json.loads(path.read_text())
        assert "mcpServers" in data
        assert "mirdan" in data["mcpServers"]

    def test_has_stdio_transport(self, tmp_path: Path) -> None:
        path = generate_mcp_json(tmp_path)
        data = json.loads(path.read_text())
        mirdan_config = data["mcpServers"]["mirdan"]
        assert mirdan_config["type"] == "stdio"
        assert "command" in mirdan_config

    def test_merges_with_existing(self, tmp_path: Path) -> None:
        """If .mcp.json exists with other servers, mirdan is added."""
        existing = {"mcpServers": {"other-server": {"type": "stdio", "command": "other"}}}
        mcp_json = tmp_path / ".mcp.json"
        mcp_json.write_text(json.dumps(existing))

        generate_mcp_json(tmp_path)
        data = json.loads(mcp_json.read_text())
        assert "mirdan" in data["mcpServers"]
        assert "other-server" in data["mcpServers"]


# ---------------------------------------------------------------------------
# export_plugin
# ---------------------------------------------------------------------------


class TestExportPlugin:
    """Tests for full plugin export."""

    def test_creates_plugin_structure(self, tmp_path: Path) -> None:
        output = tmp_path / "plugin-out"
        export_plugin(output)

        assert (output / ".claude-plugin" / "plugin.json").exists()
        assert (output / ".mcp.json").exists()
        assert (output / "README.md").exists()

    def test_plugin_json_has_required_fields(self, tmp_path: Path) -> None:
        output = tmp_path / "plugin-out"
        export_plugin(output)

        data = json.loads((output / ".claude-plugin" / "plugin.json").read_text())
        assert data["name"] == "mirdan"
        assert "version" in data
        assert "description" in data

    def test_plugin_has_skills(self, tmp_path: Path) -> None:
        output = tmp_path / "plugin-out"
        export_plugin(output)

        for skill in ("code", "debug", "review"):
            assert (output / "skills" / skill / "SKILL.md").exists()

    def test_plugin_has_agents(self, tmp_path: Path) -> None:
        output = tmp_path / "plugin-out"
        export_plugin(output)

        assert (output / "agents" / "quality-gate.md").exists()

    def test_plugin_mcp_json_valid(self, tmp_path: Path) -> None:
        output = tmp_path / "plugin-out"
        export_plugin(output)

        data = json.loads((output / ".mcp.json").read_text())
        assert data["mcpServers"]["mirdan"]["type"] == "stdio"


# ---------------------------------------------------------------------------
# _setup_claude_code integration (via init)
# ---------------------------------------------------------------------------


class TestSetupClaudeCodeIntegration:
    """Tests for the full _setup_claude_code flow."""

    def test_generates_all_categories(self, tmp_path: Path, detected: DetectedProject) -> None:
        """init --claude-code should generate rules, skills, agents, and hooks."""
        from mirdan.integrations.claude_code import generate_claude_code_config

        # Run all generation functions
        generate_mcp_json(tmp_path)
        generate_claude_code_config(tmp_path, detected)
        generate_skills(tmp_path, detected)
        generate_agents(tmp_path, detected)

        # Verify all categories exist
        assert (tmp_path / ".mcp.json").exists()
        assert (tmp_path / ".claude" / "hooks.json").exists()
        assert (tmp_path / ".claude" / "rules").is_dir()
        assert (tmp_path / ".claude" / "skills" / "code" / "SKILL.md").exists()
        assert (tmp_path / ".claude" / "agents" / "quality-gate.md").exists()

    def test_idempotent(self, tmp_path: Path, detected: DetectedProject) -> None:
        """Running init twice should not corrupt files."""
        from mirdan.integrations.claude_code import generate_claude_code_config

        # Run twice
        generate_mcp_json(tmp_path)
        generate_claude_code_config(tmp_path, detected)
        generate_skills(tmp_path, detected)
        generate_agents(tmp_path, detected)

        generate_mcp_json(tmp_path)
        generate_claude_code_config(tmp_path, detected)
        generate_skills(tmp_path, detected)
        generate_agents(tmp_path, detected)

        # All files should still be valid
        data = json.loads((tmp_path / ".mcp.json").read_text())
        assert "mirdan" in data["mcpServers"]

        # Skills should still be valid
        for skill in ("code", "debug", "review"):
            content = (tmp_path / ".claude" / "skills" / skill / "SKILL.md").read_text()
            assert content.startswith("---")
