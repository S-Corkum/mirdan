"""Tests for Cursor subagent definition generation."""

from __future__ import annotations

from pathlib import Path

import yaml

from mirdan.integrations.cursor import _CURSOR_SUBAGENTS, generate_cursor_subagents

_EXPECTED_SUBAGENTS = {
    "mirdan-quality-validator.md",
    "mirdan-security-scanner.md",
    "mirdan-test-auditor.md",
    "mirdan-slop-detector.md",
    "mirdan-architecture-reviewer.md",
}

_VALID_MODELS = {"fast", "inherit"}


class TestGenerateCursorSubagents:
    """Tests for generate_cursor_subagents()."""

    def test_generates_five_files(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        paths = generate_cursor_subagents(cursor_dir)
        assert len(paths) == 5

    def test_generates_expected_filenames(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_subagents(cursor_dir)
        agents_dir = cursor_dir / "agents"
        created = {p.name for p in agents_dir.iterdir()}
        assert created == _EXPECTED_SUBAGENTS

    def test_idempotent_does_not_overwrite_existing(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        first = generate_cursor_subagents(cursor_dir)
        assert len(first) == 5

        # Mutate one file to verify it is not overwritten
        target = cursor_dir / "agents" / "mirdan-quality-validator.md"
        target.write_text("# custom content")

        second = generate_cursor_subagents(cursor_dir)
        assert second == []
        assert target.read_text() == "# custom content"

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_subagents(cursor_dir)
        target = cursor_dir / "agents" / "mirdan-quality-validator.md"
        target.write_text("# custom content")

        result = generate_cursor_subagents(cursor_dir, force=True)
        assert len(result) > 0
        assert target.read_text() != "# custom content"

    def test_creates_agents_directory(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        assert not (cursor_dir / "agents").exists()
        generate_cursor_subagents(cursor_dir)
        assert (cursor_dir / "agents").is_dir()

    def test_returns_only_newly_created_paths(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        agents_dir = cursor_dir / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "mirdan-quality-validator.md").write_text("existing")

        paths = generate_cursor_subagents(cursor_dir)
        returned_names = {p.name for p in paths}
        assert "mirdan-quality-validator.md" not in returned_names
        assert len(paths) == 4


class TestCursorSubagentFrontmatter:
    """Tests for YAML frontmatter validity in subagent files."""

    def test_all_have_valid_frontmatter(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_subagents(cursor_dir)
        agents_dir = cursor_dir / "agents"

        for md_file in agents_dir.iterdir():
            content = md_file.read_text()
            assert content.startswith("---"), f"{md_file.name} must start with YAML frontmatter"
            # Extract frontmatter between --- markers
            parts = content.split("---", 2)
            assert len(parts) >= 3, f"{md_file.name} must have closing ---"
            fm = yaml.safe_load(parts[1])
            assert isinstance(fm, dict), f"{md_file.name} frontmatter must be a dict"

    def test_all_have_required_fields(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_subagents(cursor_dir)
        agents_dir = cursor_dir / "agents"

        for md_file in agents_dir.iterdir():
            content = md_file.read_text()
            parts = content.split("---", 2)
            fm = yaml.safe_load(parts[1])
            assert "name" in fm, f"{md_file.name} missing 'name'"
            assert "description" in fm, f"{md_file.name} missing 'description'"
            assert "model" in fm, f"{md_file.name} missing 'model'"
            assert "readonly" in fm, f"{md_file.name} missing 'readonly'"
            assert "background" in fm, f"{md_file.name} missing 'background'"

    def test_names_are_valid_format(self, tmp_path: Path) -> None:
        """Names must be lowercase with hyphens per Cursor spec."""
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_subagents(cursor_dir)
        agents_dir = cursor_dir / "agents"

        for md_file in agents_dir.iterdir():
            content = md_file.read_text()
            parts = content.split("---", 2)
            fm = yaml.safe_load(parts[1])
            name = fm["name"]
            assert name == name.lower(), f"{md_file.name}: name must be lowercase"
            assert " " not in name, f"{md_file.name}: name must not contain spaces"

    def test_models_are_valid_values(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_subagents(cursor_dir)
        agents_dir = cursor_dir / "agents"

        for md_file in agents_dir.iterdir():
            content = md_file.read_text()
            parts = content.split("---", 2)
            fm = yaml.safe_load(parts[1])
            assert fm["model"] in _VALID_MODELS, (
                f"{md_file.name}: model must be one of {_VALID_MODELS}"
            )

    def test_all_are_readonly(self, tmp_path: Path) -> None:
        """All mirdan subagents are read-only scanners."""
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_subagents(cursor_dir)
        agents_dir = cursor_dir / "agents"

        for md_file in agents_dir.iterdir():
            content = md_file.read_text()
            parts = content.split("---", 2)
            fm = yaml.safe_load(parts[1])
            assert fm["readonly"] is True, f"{md_file.name} must be readonly"


class TestCursorSubagentContent:
    """Tests for the body content of subagent files."""

    def test_all_reference_mirdan_validation(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_subagents(cursor_dir)
        agents_dir = cursor_dir / "agents"

        for md_file in agents_dir.iterdir():
            content = md_file.read_text()
            assert "validate_code_quality" in content, (
                f"{md_file.name} must reference mirdan validation tool"
            )

    def test_all_have_instructions_section(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_subagents(cursor_dir)
        agents_dir = cursor_dir / "agents"

        for md_file in agents_dir.iterdir():
            content = md_file.read_text()
            assert "## Instructions" in content, (
                f"{md_file.name} must have Instructions section"
            )

    def test_security_scanner_mentions_sec_rules(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_subagents(cursor_dir)
        content = (cursor_dir / "agents" / "mirdan-security-scanner.md").read_text()
        assert "SEC001" in content

    def test_slop_detector_mentions_ai_rules(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_subagents(cursor_dir)
        content = (cursor_dir / "agents" / "mirdan-slop-detector.md").read_text()
        assert "AI001" in content
        assert "AI008" in content


class TestCursorSubagentAsyncContent:
    """Tests for async/coordination sections in subagent content."""

    def test_background_subagents_have_async_notes(self, tmp_path: Path) -> None:
        """Background subagents should contain async execution notes."""
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_subagents(cursor_dir)
        agents_dir = cursor_dir / "agents"

        for md_file in agents_dir.iterdir():
            content = md_file.read_text()
            parts = content.split("---", 2)
            fm = yaml.safe_load(parts[1])
            if fm.get("background") is True:
                assert "Async Execution Notes" in content, (
                    f"{md_file.name} (background=true) must have Async Execution Notes"
                )

    def test_foreground_subagents_have_coordination_notes(self, tmp_path: Path) -> None:
        """Foreground subagents should contain subagent coordination notes."""
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_subagents(cursor_dir)
        agents_dir = cursor_dir / "agents"

        for md_file in agents_dir.iterdir():
            content = md_file.read_text()
            parts = content.split("---", 2)
            fm = yaml.safe_load(parts[1])
            if fm.get("background") is False:
                assert "Subagent Coordination" in content, (
                    f"{md_file.name} (background=false) must have Subagent Coordination"
                )


class TestCursorSubagentsDict:
    """Tests for the _CURSOR_SUBAGENTS constant."""

    def test_dict_has_five_entries(self) -> None:
        assert len(_CURSOR_SUBAGENTS) == 5

    def test_dict_keys_match_expected(self) -> None:
        assert set(_CURSOR_SUBAGENTS.keys()) == _EXPECTED_SUBAGENTS

    def test_all_values_are_non_empty_strings(self) -> None:
        for name, content in _CURSOR_SUBAGENTS.items():
            assert isinstance(content, str), f"{name} content must be a string"
            assert len(content.strip()) > 0, f"{name} content must not be empty"
