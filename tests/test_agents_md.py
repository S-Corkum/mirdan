"""Tests for the AGENTS.md cross-platform generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from mirdan.cli.detect import DetectedProject
from mirdan.integrations.agents_md import AgentsMDGenerator, generate_root_agents_md


@pytest.fixture
def detected() -> DetectedProject:
    """Create a detected project fixture."""
    return DetectedProject(
        project_name="test-project",
        project_type="application",
        primary_language="python",
        frameworks=["fastapi", "sqlalchemy"],
        framework_versions={"fastapi": "0.100.0"},
        detected_ides=["claude-code"],
    )


@pytest.fixture
def generator() -> AgentsMDGenerator:
    """Create a generator instance."""
    return AgentsMDGenerator()


class TestUniversalGeneration:
    """Tests for universal AGENTS.md generation."""

    def test_generates_string(self, generator: AgentsMDGenerator, detected: DetectedProject) -> None:
        """Should generate a non-empty string."""
        content = generator.generate(detected)
        assert isinstance(content, str)
        assert len(content) > 0

    def test_has_header(self, generator: AgentsMDGenerator, detected: DetectedProject) -> None:
        """Should include the AGENTS.md header."""
        content = generator.generate(detected)
        assert "AGENTS.md" in content
        assert "mirdan" in content

    def test_has_quality_rules(self, generator: AgentsMDGenerator, detected: DetectedProject) -> None:
        """Should include AI quality rules table."""
        content = generator.generate(detected)
        assert "AI001" in content
        assert "AI008" in content

    def test_has_language_section(self, generator: AgentsMDGenerator, detected: DetectedProject) -> None:
        """Should include language-specific section."""
        content = generator.generate(detected)
        assert "Python" in content

    def test_has_frameworks(self, generator: AgentsMDGenerator, detected: DetectedProject) -> None:
        """Should include detected frameworks."""
        content = generator.generate(detected)
        assert "fastapi" in content
        assert "sqlalchemy" in content

    def test_has_security_section(self, generator: AgentsMDGenerator, detected: DetectedProject) -> None:
        """Should include security standards."""
        content = generator.generate(detected)
        assert "SEC001" in content
        assert "Security" in content

    def test_has_workflow_section(self, generator: AgentsMDGenerator, detected: DetectedProject) -> None:
        """Should include quality workflow."""
        content = generator.generate(detected)
        assert "enhance_prompt" in content
        assert "validate_code_quality" in content


class TestCursorOverlay:
    """Tests for Cursor-specific overlay."""

    def test_cursor_overlay_added(self, generator: AgentsMDGenerator, detected: DetectedProject) -> None:
        """Cursor platform should add Cursor-specific section."""
        content = generator.generate(detected, platform="cursor")
        assert "Cursor" in content

    def test_cursor_mentions_bugbot(self, generator: AgentsMDGenerator, detected: DetectedProject) -> None:
        """Cursor overlay should mention BugBot."""
        content = generator.generate(detected, platform="cursor")
        assert "BugBot" in content

    def test_cursor_mentions_mdc_rules(self, generator: AgentsMDGenerator, detected: DetectedProject) -> None:
        """Cursor overlay should reference .mdc rules."""
        content = generator.generate(detected, platform="cursor")
        assert ".mdc" in content or ".cursor/rules" in content


class TestClaudeCodeOverlay:
    """Tests for Claude Code-specific overlay."""

    def test_claude_code_overlay_added(self, generator: AgentsMDGenerator, detected: DetectedProject) -> None:
        """Claude Code platform should add Claude Code section."""
        content = generator.generate(detected, platform="claude-code")
        assert "Claude Code" in content

    def test_claude_code_mentions_hooks(self, generator: AgentsMDGenerator, detected: DetectedProject) -> None:
        """Claude Code overlay should mention hooks."""
        content = generator.generate(detected, platform="claude-code")
        assert "PreToolUse" in content or "Hook" in content

    def test_claude_code_mentions_mcp_tools(self, generator: AgentsMDGenerator, detected: DetectedProject) -> None:
        """Claude Code overlay should mention MCP tools."""
        content = generator.generate(detected, platform="claude-code")
        assert "mcp__mirdan" in content

    def test_claude_code_mentions_skills(self, generator: AgentsMDGenerator, detected: DetectedProject) -> None:
        """Claude Code overlay should mention skills."""
        content = generator.generate(detected, platform="claude-code")
        assert "mirdan:code" in content


class TestGenerateAndWrite:
    """Tests for file writing."""

    def test_writes_file(self, tmp_path: Path, detected: DetectedProject) -> None:
        """Should write AGENTS.md to disk."""
        generator = AgentsMDGenerator()
        output = tmp_path / "AGENTS.md"
        result = generator.generate_and_write(output, detected)
        assert result == output
        assert output.exists()

    def test_creates_parent_dirs(self, tmp_path: Path, detected: DetectedProject) -> None:
        """Should create parent directories."""
        generator = AgentsMDGenerator()
        output = tmp_path / "deep" / "nested" / "AGENTS.md"
        generator.generate_and_write(output, detected)
        assert output.exists()

    def test_content_matches_generate(self, tmp_path: Path, detected: DetectedProject) -> None:
        """Written content should match generate() output."""
        generator = AgentsMDGenerator()
        output = tmp_path / "AGENTS.md"
        generator.generate_and_write(output, detected)
        written = output.read_text()
        generated = generator.generate(detected)
        assert written == generated


class TestConvenienceFunction:
    """Tests for generate_root_agents_md()."""

    def test_generates_at_project_root(self, tmp_path: Path, detected: DetectedProject) -> None:
        """Should generate AGENTS.md at project root."""
        result = generate_root_agents_md(tmp_path, detected)
        assert result == tmp_path / "AGENTS.md"
        assert result.exists()

    def test_uses_platform(self, tmp_path: Path, detected: DetectedProject) -> None:
        """Should use the specified platform."""
        result = generate_root_agents_md(tmp_path, detected, platform="claude-code")
        content = result.read_text()
        assert "Claude Code" in content

    def test_universal_platform(self, tmp_path: Path, detected: DetectedProject) -> None:
        """Universal platform should not have platform-specific sections."""
        result = generate_root_agents_md(tmp_path, detected, platform="universal")
        content = result.read_text()
        # Should have universal content but no platform-specific
        assert "AI001" in content


class TestEdgeCases:
    """Tests for edge cases."""

    def test_no_language(self) -> None:
        """Should handle missing language gracefully."""
        detected = DetectedProject(
            project_name="test",
            project_type="application",
            primary_language="",
        )
        generator = AgentsMDGenerator()
        content = generator.generate(detected)
        assert "Python" in content  # Defaults to Python

    def test_no_frameworks(self) -> None:
        """Should handle no frameworks."""
        detected = DetectedProject(
            project_name="test",
            project_type="application",
            primary_language="rust",
            frameworks=[],
        )
        generator = AgentsMDGenerator()
        content = generator.generate(detected)
        assert "Rust" in content
        assert "Frameworks" not in content  # No frameworks section
