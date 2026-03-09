"""Tests for Cursor slash command file generation."""

from __future__ import annotations

from pathlib import Path

from mirdan.integrations.cursor import _CURSOR_COMMANDS, generate_cursor_commands

_EXPECTED_COMMANDS = {
    "code.md",
    "debug.md",
    "review.md",
    "plan.md",
    "quality.md",
    "scan.md",
    "gate.md",
}


class TestGenerateCursorCommands:
    """Tests for generate_cursor_commands()."""

    def test_generates_seven_files(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        paths = generate_cursor_commands(cursor_dir)
        assert len(paths) == 7

    def test_generates_expected_filenames(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_commands(cursor_dir)
        commands_dir = cursor_dir / "commands"
        created = {p.name for p in commands_dir.iterdir()}
        assert created == _EXPECTED_COMMANDS

    def test_files_are_plain_markdown_no_frontmatter(self, tmp_path: Path) -> None:
        """Command files must not start with YAML frontmatter (---)."""
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_commands(cursor_dir)
        commands_dir = cursor_dir / "commands"
        for md_file in commands_dir.iterdir():
            content = md_file.read_text()
            assert not content.startswith("---"), f"{md_file.name} must not have YAML frontmatter"

    def test_idempotent_does_not_overwrite_existing(self, tmp_path: Path) -> None:
        """Calling twice should not overwrite files created on first call."""
        cursor_dir = tmp_path / ".cursor"
        first = generate_cursor_commands(cursor_dir)
        assert len(first) == 7

        # Mutate one file to verify it is not overwritten
        target = cursor_dir / "commands" / "code.md"
        target.write_text("# custom content")

        second = generate_cursor_commands(cursor_dir)
        assert second == []  # no new files created
        assert target.read_text() == "# custom content"

    def test_creates_commands_directory(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        assert not (cursor_dir / "commands").exists()
        generate_cursor_commands(cursor_dir)
        assert (cursor_dir / "commands").is_dir()

    def test_returns_only_newly_created_paths(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        # Pre-create one file
        commands_dir = cursor_dir / "commands"
        commands_dir.mkdir(parents=True)
        (commands_dir / "code.md").write_text("existing")

        paths = generate_cursor_commands(cursor_dir)
        returned_names = {p.name for p in paths}
        assert "code.md" not in returned_names
        assert len(paths) == 6


class TestCursorCommandContent:
    """Tests for the content of each command file."""

    def test_code_command_mentions_enhance_prompt(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_commands(cursor_dir)
        content = (cursor_dir / "commands" / "code.md").read_text()
        assert "enhance_prompt" in content

    def test_debug_command_mentions_validate(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_commands(cursor_dir)
        content = (cursor_dir / "commands" / "debug.md").read_text()
        assert "validate" in content.lower()

    def test_plan_command_mentions_plan_mode(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_commands(cursor_dir)
        content = (cursor_dir / "commands" / "plan.md").read_text()
        assert "plan" in content.lower()

    def test_scan_command_mentions_scan_conventions(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_commands(cursor_dir)
        content = (cursor_dir / "commands" / "scan.md").read_text()
        assert "scan_conventions" in content

    def test_gate_command_mentions_gate(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_commands(cursor_dir)
        content = (cursor_dir / "commands" / "gate.md").read_text()
        assert "gate" in content.lower()

    def test_all_commands_have_heading(self, tmp_path: Path) -> None:
        """Every command file should start with a markdown heading."""
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_commands(cursor_dir)
        commands_dir = cursor_dir / "commands"
        for md_file in commands_dir.iterdir():
            content = md_file.read_text().strip()
            assert content.startswith("#"), f"{md_file.name} must start with a markdown heading"


class TestCursorCommandsDict:
    """Tests for the _CURSOR_COMMANDS constant."""

    def test_dict_has_seven_entries(self) -> None:
        assert len(_CURSOR_COMMANDS) == 7

    def test_dict_keys_match_expected(self) -> None:
        assert set(_CURSOR_COMMANDS.keys()) == _EXPECTED_COMMANDS

    def test_all_values_are_non_empty_strings(self) -> None:
        for name, content in _CURSOR_COMMANDS.items():
            assert isinstance(content, str), f"{name} content must be a string"
            assert len(content.strip()) > 0, f"{name} content must not be empty"
