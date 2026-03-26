"""Tests for Cursor skill definition generation."""

from __future__ import annotations

from pathlib import Path

import yaml

from mirdan.integrations.cursor import _CURSOR_SKILLS, generate_cursor_skills

_EXPECTED_SKILLS = {
    "mirdan-code",
    "mirdan-debug",
    "mirdan-review",
    "mirdan-plan",
    "mirdan-plan-review",
    "mirdan-quality",
    "mirdan-scan",
    "mirdan-gate",
}

# Skills that require explicit invocation (/mirdan-quality, etc.)
_MANUAL_SKILLS = {"mirdan-quality", "mirdan-scan", "mirdan-gate", "mirdan-plan-review"}

# Skills that auto-invoke based on context
_AUTO_SKILLS = {"mirdan-code", "mirdan-debug", "mirdan-review", "mirdan-plan"}


class TestGenerateCursorSkills:
    """Tests for generate_cursor_skills()."""

    def test_generates_eight_skill_directories(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        paths = generate_cursor_skills(cursor_dir)
        assert len(paths) == 8

    def test_generates_expected_skill_names(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_skills(cursor_dir)
        skills_dir = cursor_dir / "skills"
        created = {p.name for p in skills_dir.iterdir() if p.is_dir()}
        assert created == _EXPECTED_SKILLS

    def test_each_has_skill_md(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_skills(cursor_dir)
        skills_dir = cursor_dir / "skills"
        for skill_dir in skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_md = skill_dir / "SKILL.md"
                assert skill_md.exists(), f"{skill_dir.name} missing SKILL.md"

    def test_idempotent_does_not_overwrite_existing(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        first = generate_cursor_skills(cursor_dir)
        assert len(first) == 8

        # Mutate one file
        target = cursor_dir / "skills" / "mirdan-code" / "SKILL.md"
        target.write_text("# custom content")

        second = generate_cursor_skills(cursor_dir)
        assert second == []
        assert target.read_text() == "# custom content"

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_skills(cursor_dir)
        target = cursor_dir / "skills" / "mirdan-code" / "SKILL.md"
        target.write_text("# custom content")

        result = generate_cursor_skills(cursor_dir, force=True)
        assert len(result) > 0
        assert target.read_text() != "# custom content"

    def test_creates_skills_directory(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        assert not (cursor_dir / "skills").exists()
        generate_cursor_skills(cursor_dir)
        assert (cursor_dir / "skills").is_dir()

    def test_returns_only_newly_created_paths(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        skill_dir = cursor_dir / "skills" / "mirdan-code"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("existing")

        paths = generate_cursor_skills(cursor_dir)
        returned_names = {p.parent.name for p in paths}
        assert "mirdan-code" not in returned_names
        assert len(paths) == 7


class TestCursorSkillFrontmatter:
    """Tests for YAML frontmatter validity in SKILL.md files."""

    def test_all_have_valid_frontmatter(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_skills(cursor_dir)
        skills_dir = cursor_dir / "skills"

        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            content = (skill_dir / "SKILL.md").read_text()
            assert content.startswith("---"), f"{skill_dir.name} must start with frontmatter"
            parts = content.split("---", 2)
            assert len(parts) >= 3, f"{skill_dir.name} must have closing ---"
            fm = yaml.safe_load(parts[1])
            assert isinstance(fm, dict), f"{skill_dir.name} frontmatter must be a dict"

    def test_all_have_name_and_description(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_skills(cursor_dir)
        skills_dir = cursor_dir / "skills"

        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            content = (skill_dir / "SKILL.md").read_text()
            parts = content.split("---", 2)
            fm = yaml.safe_load(parts[1])
            assert "name" in fm, f"{skill_dir.name} missing 'name'"
            assert "description" in fm, f"{skill_dir.name} missing 'description'"

    def test_directory_names_match_skill_names(self, tmp_path: Path) -> None:
        """Per agentskills.io spec: directory name must match skill name."""
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_skills(cursor_dir)
        skills_dir = cursor_dir / "skills"

        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            content = (skill_dir / "SKILL.md").read_text()
            parts = content.split("---", 2)
            fm = yaml.safe_load(parts[1])
            assert fm["name"] == skill_dir.name, (
                f"Directory '{skill_dir.name}' must match name '{fm['name']}'"
            )

    def test_skill_names_are_valid_format(self, tmp_path: Path) -> None:
        """Names must be 1-64 chars, lowercase alphanumeric + hyphens."""
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_skills(cursor_dir)
        skills_dir = cursor_dir / "skills"

        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            content = (skill_dir / "SKILL.md").read_text()
            parts = content.split("---", 2)
            fm = yaml.safe_load(parts[1])
            name = fm["name"]
            assert 1 <= len(name) <= 64, f"{name}: must be 1-64 chars"
            assert name == name.lower(), f"{name}: must be lowercase"
            assert not name.startswith("-"), f"{name}: must not start with hyphen"
            assert not name.endswith("-"), f"{name}: must not end with hyphen"
            assert "--" not in name, f"{name}: must not have consecutive hyphens"

    def test_descriptions_within_limit(self, tmp_path: Path) -> None:
        """Descriptions must be 1-1024 chars per spec."""
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_skills(cursor_dir)
        skills_dir = cursor_dir / "skills"

        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            content = (skill_dir / "SKILL.md").read_text()
            parts = content.split("---", 2)
            fm = yaml.safe_load(parts[1])
            desc = fm["description"]
            assert 1 <= len(desc) <= 1024, f"{skill_dir.name}: description out of bounds"

    def test_manual_skills_disable_model_invocation(self, tmp_path: Path) -> None:
        """Quality, scan, and gate skills require explicit invocation."""
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_skills(cursor_dir)
        skills_dir = cursor_dir / "skills"

        for skill_name in _MANUAL_SKILLS:
            content = (skills_dir / skill_name / "SKILL.md").read_text()
            parts = content.split("---", 2)
            fm = yaml.safe_load(parts[1])
            assert fm.get("disable-model-invocation") is True, (
                f"{skill_name} must have disable-model-invocation: true"
            )

    def test_auto_skills_allow_model_invocation(self, tmp_path: Path) -> None:
        """Code, debug, review, plan skills auto-invoke based on context."""
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_skills(cursor_dir)
        skills_dir = cursor_dir / "skills"

        for skill_name in _AUTO_SKILLS:
            content = (skills_dir / skill_name / "SKILL.md").read_text()
            parts = content.split("---", 2)
            fm = yaml.safe_load(parts[1])
            assert fm.get("disable-model-invocation") is False, (
                f"{skill_name} must have disable-model-invocation: false"
            )


class TestCursorSkillContent:
    """Tests for the body content of SKILL.md files."""

    def test_all_have_when_to_use_section(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_skills(cursor_dir)
        skills_dir = cursor_dir / "skills"

        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            content = (skill_dir / "SKILL.md").read_text()
            assert "When to Use" in content or "## Workflow" in content, (
                f"{skill_dir.name} must have When to Use or Workflow section"
            )

    def test_all_have_workflow_section(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_skills(cursor_dir)
        skills_dir = cursor_dir / "skills"

        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            content = (skill_dir / "SKILL.md").read_text()
            assert "## Workflow" in content, f"{skill_dir.name} must have Workflow section"

    def test_code_skill_mentions_enhance_prompt(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_skills(cursor_dir)
        content = (cursor_dir / "skills" / "mirdan-code" / "SKILL.md").read_text()
        assert "enhance_prompt" in content

    def test_gate_skill_mentions_quality_threshold(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_skills(cursor_dir)
        content = (cursor_dir / "skills" / "mirdan-gate" / "SKILL.md").read_text()
        assert "0.7" in content
        assert "0.8" in content

    def test_scan_skill_mentions_scan_conventions(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        generate_cursor_skills(cursor_dir)
        content = (cursor_dir / "skills" / "mirdan-scan" / "SKILL.md").read_text()
        assert "scan_conventions" in content


class TestCursorSkillsDict:
    """Tests for the _CURSOR_SKILLS constant."""

    def test_dict_has_eight_entries(self) -> None:
        assert len(_CURSOR_SKILLS) == 8

    def test_dict_keys_match_expected(self) -> None:
        assert set(_CURSOR_SKILLS.keys()) == _EXPECTED_SKILLS

    def test_all_values_are_non_empty_strings(self) -> None:
        for name, content in _CURSOR_SKILLS.items():
            assert isinstance(content, str), f"{name} content must be a string"
            assert len(content.strip()) > 0, f"{name} content must not be empty"
