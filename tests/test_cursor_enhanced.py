"""Tests for enhanced AGENTS.md and BUGBOT.md generation (v0.4.0)."""

from __future__ import annotations

from pathlib import Path

from mirdan.cli.detect import DetectedProject
from mirdan.integrations.cursor import generate_cursor_agents


class TestEnhancedAgentsMd:
    """Tests for enhanced AGENTS.md content."""

    def _get_agents_content(self, tmp_path: Path) -> str:
        """Helper to generate AGENTS.md and return content."""
        cursor_dir = tmp_path / ".cursor"
        detected = DetectedProject(primary_language="python")
        generate_cursor_agents(cursor_dir, detected)
        return (cursor_dir / "AGENTS.md").read_text()

    def test_has_quality_checkpoints(self, tmp_path: Path) -> None:
        """Enhanced AGENTS.md should have Quality Checkpoints section."""
        content = self._get_agents_content(tmp_path)
        assert "Quality Checkpoints" in content

    def test_has_ai001_rule(self, tmp_path: Path) -> None:
        """Enhanced AGENTS.md should contain AI001 rule inline."""
        content = self._get_agents_content(tmp_path)
        assert "AI001" in content

    def test_has_ai008_rule(self, tmp_path: Path) -> None:
        """Enhanced AGENTS.md should contain AI008 rule inline."""
        content = self._get_agents_content(tmp_path)
        assert "AI008" in content

    def test_has_sec001_rule(self, tmp_path: Path) -> None:
        """Enhanced AGENTS.md should contain SEC001 security rule."""
        content = self._get_agents_content(tmp_path)
        assert "SEC001" in content

    def test_has_quality_threshold(self, tmp_path: Path) -> None:
        """Enhanced AGENTS.md should contain 0.7 quality threshold."""
        content = self._get_agents_content(tmp_path)
        assert "0.7" in content

    def test_mentions_enhance_prompt(self, tmp_path: Path) -> None:
        """Enhanced AGENTS.md should mention enhance_prompt tool."""
        content = self._get_agents_content(tmp_path)
        assert "enhance_prompt" in content

    def test_mentions_validate_code_quality(self, tmp_path: Path) -> None:
        """Enhanced AGENTS.md should mention validate_code_quality tool."""
        content = self._get_agents_content(tmp_path)
        assert "validate_code_quality" in content

    def test_has_periodic_checkpoint(self, tmp_path: Path) -> None:
        """Enhanced AGENTS.md should have periodic validation reminder."""
        content = self._get_agents_content(tmp_path)
        assert "30" in content  # "Every 30 Minutes"

    def test_has_security_standards_section(self, tmp_path: Path) -> None:
        """Enhanced AGENTS.md should have Security Standards section."""
        content = self._get_agents_content(tmp_path)
        assert "Security Standards" in content

    def test_has_all_sec_rules(self, tmp_path: Path) -> None:
        """Enhanced AGENTS.md should contain SEC001 through SEC010."""
        content = self._get_agents_content(tmp_path)
        for i in range(1, 11):
            assert f"SEC{i:03d}" in content, f"Missing SEC{i:03d}"


class TestEnhancedBugbotMd:
    """Tests for enhanced BUGBOT.md content."""

    def _get_bugbot_content(self, tmp_path: Path) -> str:
        """Helper to generate BUGBOT.md and return content."""
        cursor_dir = tmp_path / ".cursor"
        detected = DetectedProject(primary_language="python")
        generate_cursor_agents(cursor_dir, detected)
        return (cursor_dir / "BUGBOT.md").read_text()

    def test_has_blocking_section(self, tmp_path: Path) -> None:
        """Enhanced BUGBOT.md should have Blocking Bugs section."""
        content = self._get_bugbot_content(tmp_path)
        assert "Blocking" in content

    def test_has_request_changes_section(self, tmp_path: Path) -> None:
        """Enhanced BUGBOT.md should have Request Changes section."""
        content = self._get_bugbot_content(tmp_path)
        assert "Request Changes" in content

    def test_has_best_practice_section(self, tmp_path: Path) -> None:
        """Enhanced BUGBOT.md should have Best Practice section."""
        content = self._get_bugbot_content(tmp_path)
        assert "Best Practice" in content

    def test_has_regex_patterns(self, tmp_path: Path) -> None:
        """Enhanced BUGBOT.md should have regex patterns in code blocks."""
        content = self._get_bugbot_content(tmp_path)
        assert "```regex" in content

    def test_mentions_ai001(self, tmp_path: Path) -> None:
        """Enhanced BUGBOT.md should mention AI001."""
        content = self._get_bugbot_content(tmp_path)
        assert "AI001" in content

    def test_mentions_ai008(self, tmp_path: Path) -> None:
        """Enhanced BUGBOT.md should mention AI008."""
        content = self._get_bugbot_content(tmp_path)
        assert "AI008" in content

    def test_mentions_sec001(self, tmp_path: Path) -> None:
        """Enhanced BUGBOT.md should mention SEC001."""
        content = self._get_bugbot_content(tmp_path)
        assert "SEC001" in content

    def test_mentions_sec002(self, tmp_path: Path) -> None:
        """Enhanced BUGBOT.md should mention SEC002."""
        content = self._get_bugbot_content(tmp_path)
        assert "SEC002" in content

    def test_mentions_sec003(self, tmp_path: Path) -> None:
        """Enhanced BUGBOT.md should mention SEC003."""
        content = self._get_bugbot_content(tmp_path)
        assert "SEC003" in content

    def test_has_not_implemented_pattern(self, tmp_path: Path) -> None:
        """Enhanced BUGBOT.md should have NotImplementedError pattern."""
        content = self._get_bugbot_content(tmp_path)
        assert "NotImplementedError" in content

    def test_has_sql_injection_pattern(self, tmp_path: Path) -> None:
        """Enhanced BUGBOT.md should have SQL injection regex pattern."""
        content = self._get_bugbot_content(tmp_path)
        assert "SELECT" in content

    def test_has_command_injection_pattern(self, tmp_path: Path) -> None:
        """Enhanced BUGBOT.md should have command injection regex pattern."""
        content = self._get_bugbot_content(tmp_path)
        assert "os\\.system" in content or "os.system" in content
