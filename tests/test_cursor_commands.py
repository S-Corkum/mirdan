"""Tests for Cursor slash command file generation."""

from __future__ import annotations

from pathlib import Path

from mirdan.integrations.cursor import _CURSOR_COMMANDS, generate_cursor_commands

# Always-available commands from the in-file dict.
_DICT_COMMANDS = {
    "code.md",
    "automations.md",
}

# Planning commands from packaged templates.
_PIPELINE_COMMANDS = {
    "plan.md",
    "plan-verify.md",
    "plan-review.md",
}

_EXPECTED_COMMANDS = _DICT_COMMANDS | _PIPELINE_COMMANDS  # 5


class TestGenerateCursorCommands:
    """Tests for generate_cursor_commands()."""

    def test_generates_five_files(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        paths = generate_cursor_commands(cursor_dir)
        assert len(paths) == 5

    def test_generates_expected_filenames(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_commands(cursor_dir)
        commands_dir = cursor_dir / "commands"
        created = {p.name for p in commands_dir.iterdir()}
        assert created == _EXPECTED_COMMANDS

    def test_no_retired_or_brief_commands(self, tmp_path: Path) -> None:
        """No /debug /review /quality /scan /gate /brief /plan-execute files."""
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_commands(cursor_dir)
        created = {p.name for p in (cursor_dir / "commands").iterdir()}
        for gone in (
            "debug.md",
            "review.md",
            "quality.md",
            "scan.md",
            "gate.md",
            "brief.md",
            "plan-execute.md",
        ):
            assert gone not in created

    def test_files_are_plain_markdown_no_frontmatter(self, tmp_path: Path) -> None:
        """Command files must not start with YAML frontmatter (---)."""
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_commands(cursor_dir)
        commands_dir = cursor_dir / "commands"
        for md_file in commands_dir.iterdir():
            content = md_file.read_text()
            assert not content.startswith("---"), f"{md_file.name} must not have YAML frontmatter"

    def test_idempotent_does_not_overwrite_existing(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        first = generate_cursor_commands(cursor_dir)
        assert len(first) == 5

        target = cursor_dir / "commands" / "code.md"
        target.write_text("# custom content")

        second = generate_cursor_commands(cursor_dir)
        assert second == []
        assert target.read_text() == "# custom content"

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_commands(cursor_dir)
        target = cursor_dir / "commands" / "code.md"
        target.write_text("# custom content")

        result = generate_cursor_commands(cursor_dir, force=True)
        assert len(result) > 0
        assert target.read_text() != "# custom content"

    def test_creates_commands_directory(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        assert not (cursor_dir / "commands").exists()
        generate_cursor_commands(cursor_dir)
        assert (cursor_dir / "commands").is_dir()

    def test_returns_only_newly_created_paths(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        commands_dir = cursor_dir / "commands"
        commands_dir.mkdir(parents=True)
        (commands_dir / "code.md").write_text("existing")

        paths = generate_cursor_commands(cursor_dir)
        returned_names = {p.name for p in paths}
        assert "code.md" not in returned_names
        assert len(paths) == 4


class TestCursorCommandContent:
    """Tests for the content of each command file."""

    def test_code_command_mentions_enhance_prompt(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_commands(cursor_dir)
        content = (cursor_dir / "commands" / "code.md").read_text()
        assert "enhance_prompt" in content

    def test_plan_command_mentions_low_level_design(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_commands(cursor_dir)
        content = (cursor_dir / "commands" / "plan.md").read_text()
        assert "plan" in content.lower()
        assert "low-level design" in content.lower()

    def test_plan_verify_command_mentions_verify_plan(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_commands(cursor_dir)
        content = (cursor_dir / "commands" / "plan-verify.md").read_text()
        assert "verify_plan" in content

    def test_automations_command_mentions_cursor_automations(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_commands(cursor_dir)
        content = (cursor_dir / "commands" / "automations.md").read_text()
        assert "cursor.com/automations" in content
        assert "validate_code_quality" in content

    def test_all_commands_have_heading(self, tmp_path: Path) -> None:
        """Every command file should start with a markdown heading."""
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_commands(cursor_dir)
        commands_dir = cursor_dir / "commands"
        for md_file in commands_dir.iterdir():
            content = md_file.read_text().strip()
            assert content.startswith("#"), f"{md_file.name} must start with a markdown heading"


class TestCursorCommandsDict:
    """Tests for the _CURSOR_COMMANDS constant.

    The in-memory dict holds the always-available commands (/code, /automations).
    Planning commands (/plan, /plan-verify, /plan-review) live as packaged
    markdown templates and are merged in by generate_cursor_commands.
    """

    def test_dict_has_two_entries(self) -> None:
        assert len(_CURSOR_COMMANDS) == 2

    def test_dict_keys_match(self) -> None:
        assert set(_CURSOR_COMMANDS.keys()) == _DICT_COMMANDS

    def test_all_values_are_non_empty_strings(self) -> None:
        for name, content in _CURSOR_COMMANDS.items():
            assert isinstance(content, str), f"{name} content must be a string"
            assert len(content.strip()) > 0, f"{name} content must not be empty"
