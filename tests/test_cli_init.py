"""Tests for CLI entry point, project detection, and init command."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from mirdan.cli import main
from mirdan.cli.detect import DetectedProject, detect_framework_version, detect_project
from mirdan.cli.init_command import _build_config, _create_rules_template, run_init
from mirdan.integrations.cursor import generate_cursor_rules


class TestCLIRouting:
    """Tests for CLI subcommand routing."""

    def test_help_flag(self, capsys: object) -> None:
        """--help should print usage and not error."""
        with patch.object(sys, "argv", ["mirdan", "--help"]):
            main()
        # If we get here, it didn't sys.exit(1)

    def test_version_flag(self, capsys: object) -> None:
        """--version should print version."""
        with patch.object(sys, "argv", ["mirdan", "--version"]):
            main()

    def test_unknown_command_exits(self) -> None:
        """Unknown subcommand should exit with code 1."""
        import pytest

        with patch.object(sys, "argv", ["mirdan", "unknown"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_bare_command_calls_serve(self) -> None:
        """Bare 'mirdan' should call _serve (which calls server main)."""
        with (
            patch.object(sys, "argv", ["mirdan"]),
            patch("mirdan.cli._serve") as mock_serve,
        ):
            main()
            mock_serve.assert_called_once()

    def test_serve_command_calls_serve(self) -> None:
        """'mirdan serve' should call _serve."""
        with (
            patch.object(sys, "argv", ["mirdan", "serve"]),
            patch("mirdan.cli._serve") as mock_serve,
        ):
            main()
            mock_serve.assert_called_once()

    def test_init_command_calls_init(self) -> None:
        """'mirdan init' should call _init."""
        with (
            patch.object(sys, "argv", ["mirdan", "init"]),
            patch("mirdan.cli._init") as mock_init,
        ):
            main()
            mock_init.assert_called_once_with([])


class TestProjectDetection:
    """Tests for project type auto-detection."""

    def test_detect_python_project(self, tmp_path: Path) -> None:
        """Should detect Python project from pyproject.toml."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "myapp"\n\n[project.dependencies]\nfastapi = ">=0.100.0"\n'
        )

        result = detect_project(tmp_path)
        assert result.project_type == "python"
        assert result.primary_language == "python"
        assert result.project_name == "myapp"
        assert "fastapi" in result.frameworks

    def test_detect_node_project(self, tmp_path: Path) -> None:
        """Should detect Node.js project from package.json."""
        pkg = tmp_path / "package.json"
        pkg.write_text(
            json.dumps(
                {
                    "name": "my-app",
                    "dependencies": {"react": "^18.2.0", "next": "^14.0.0"},
                }
            )
        )

        result = detect_project(tmp_path)
        assert result.project_type == "node"
        assert result.project_name == "my-app"
        assert "react" in result.frameworks
        assert "next.js" in result.frameworks

    def test_detect_typescript_language(self, tmp_path: Path) -> None:
        """Should detect TypeScript when tsconfig.json present."""
        pkg = tmp_path / "package.json"
        pkg.write_text(json.dumps({"name": "ts-app"}))
        tsconfig = tmp_path / "tsconfig.json"
        tsconfig.write_text("{}")

        result = detect_project(tmp_path)
        assert result.primary_language == "typescript"

    def test_detect_rust_project(self, tmp_path: Path) -> None:
        """Should detect Rust project from Cargo.toml."""
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[package]\nname = "my-crate"\nversion = "0.1.0"\n')

        result = detect_project(tmp_path)
        assert result.project_type == "rust"
        assert result.primary_language == "rust"
        assert result.project_name == "my-crate"

    def test_detect_go_project(self, tmp_path: Path) -> None:
        """Should detect Go project from go.mod."""
        go_mod = tmp_path / "go.mod"
        go_mod.write_text("module github.com/user/myapp\n\ngo 1.21\n")

        result = detect_project(tmp_path)
        assert result.project_type == "go"
        assert result.primary_language == "go"

    def test_detect_java_pom(self, tmp_path: Path) -> None:
        """Should detect Java project from pom.xml."""
        pom = tmp_path / "pom.xml"
        pom.write_text("<project></project>")

        result = detect_project(tmp_path)
        assert result.project_type == "java"

    def test_detect_java_gradle(self, tmp_path: Path) -> None:
        """Should detect Java project from build.gradle."""
        gradle = tmp_path / "build.gradle"
        gradle.write_text("apply plugin: 'java'")

        result = detect_project(tmp_path)
        assert result.project_type == "java"

    def test_detect_unknown_project(self, tmp_path: Path) -> None:
        """Should return unknown for empty directory."""
        result = detect_project(tmp_path)
        assert result.project_type == "unknown"

    def test_detect_ides(self, tmp_path: Path) -> None:
        """Should detect IDEs from directory presence."""
        (tmp_path / ".cursor").mkdir()
        (tmp_path / ".vscode").mkdir()

        result = detect_project(tmp_path)
        assert "cursor" in result.detected_ides
        assert "vscode" in result.detected_ides

    def test_detect_framework_versions(self, tmp_path: Path) -> None:
        """Should extract framework versions from package.json."""
        pkg = tmp_path / "package.json"
        pkg.write_text(
            json.dumps(
                {
                    "name": "app",
                    "dependencies": {"react": "^19.0.0"},
                }
            )
        )

        result = detect_project(tmp_path)
        assert result.framework_versions.get("react") == "19.0.0"

    def test_detect_framework_version_standalone(self, tmp_path: Path) -> None:
        """detect_framework_version should work standalone."""
        pkg = tmp_path / "package.json"
        pkg.write_text(
            json.dumps(
                {
                    "name": "app",
                    "dependencies": {"react": "^18.2.0"},
                }
            )
        )

        version = detect_framework_version("react", tmp_path)
        assert version == "18.2.0"

    def test_detect_framework_version_not_found(self, tmp_path: Path) -> None:
        """detect_framework_version returns None if not found."""
        version = detect_framework_version("react", tmp_path)
        assert version is None

    def test_python_framework_version(self, tmp_path: Path) -> None:
        """Should extract Python framework versions."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "app"\ndependencies = [\n  "fastapi>=0.115.0",\n]\n'
        )

        version = detect_framework_version("fastapi", tmp_path)
        assert version == "0.115.0"


class TestInitCommand:
    """Tests for mirdan init command."""

    def test_build_config(self) -> None:
        """Should build config from detected project."""
        detected = DetectedProject(
            project_type="python",
            project_name="myapp",
            primary_language="python",
            frameworks=["fastapi"],
        )
        config = _build_config(detected)
        assert config.project.name == "myapp"
        assert config.project.primary_language == "python"
        assert "fastapi" in config.project.frameworks

    def test_create_rules_template(self, tmp_path: Path) -> None:
        """Should create example.yaml in rules dir."""
        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        detected = DetectedProject(primary_language="python")
        _create_rules_template(rules_dir, detected)

        template = rules_dir / "example.yaml"
        assert template.exists()
        content = template.read_text()
        assert "CUSTOM001" in content

    def test_create_rules_template_idempotent(self, tmp_path: Path) -> None:
        """Should not overwrite existing template."""
        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        template = rules_dir / "example.yaml"
        template.write_text("existing content")

        detected = DetectedProject(primary_language="python")
        _create_rules_template(rules_dir, detected)

        assert template.read_text() == "existing content"

    def test_run_init_creates_config(self, tmp_path: Path) -> None:
        """run_init should create .mirdan/config.yaml."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "testapp"\n')

        with patch("builtins.input", return_value=""):
            run_init([str(tmp_path)])

        config_path = tmp_path / ".mirdan" / "config.yaml"
        assert config_path.exists()

    def test_run_init_creates_rules_dir(self, tmp_path: Path) -> None:
        """run_init should create .mirdan/rules/ directory."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "testapp"\n')

        with patch("builtins.input", return_value=""):
            run_init([str(tmp_path)])

        rules_dir = tmp_path / ".mirdan" / "rules"
        assert rules_dir.is_dir()

    def test_run_init_with_cursor(self, tmp_path: Path) -> None:
        """run_init should create .cursor/rules/ if Cursor detected."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "testapp"\n')
        (tmp_path / ".cursor").mkdir()

        with patch("builtins.input", return_value=""):
            run_init([str(tmp_path)])

        cursor_rules = tmp_path / ".cursor" / "rules"
        assert cursor_rules.is_dir()


class TestCursorIntegration:
    """Tests for Cursor .mdc rule generation."""

    def test_generate_always_rule(self, tmp_path: Path) -> None:
        """Should generate mirdan-always.mdc."""
        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        detected = DetectedProject(primary_language="python")

        generated = generate_cursor_rules(rules_dir, detected)
        names = [p.name for p in generated]
        assert "mirdan-always.mdc" in names

    def test_generate_python_rule(self, tmp_path: Path) -> None:
        """Should generate mirdan-python.mdc for Python projects."""
        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        detected = DetectedProject(primary_language="python")

        generated = generate_cursor_rules(rules_dir, detected)
        names = [p.name for p in generated]
        assert "mirdan-python.mdc" in names

    def test_generate_typescript_rule(self, tmp_path: Path) -> None:
        """Should generate mirdan-typescript.mdc for TS projects."""
        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        detected = DetectedProject(primary_language="typescript")

        generated = generate_cursor_rules(rules_dir, detected)
        names = [p.name for p in generated]
        assert "mirdan-typescript.mdc" in names

    def test_generate_security_rule(self, tmp_path: Path) -> None:
        """Should always generate mirdan-security.mdc."""
        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        detected = DetectedProject(primary_language="python")

        generated = generate_cursor_rules(rules_dir, detected)
        names = [p.name for p in generated]
        assert "mirdan-security.mdc" in names

    def test_generate_planning_rule(self, tmp_path: Path) -> None:
        """Should always generate mirdan-planning.mdc."""
        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        detected = DetectedProject(primary_language="python")

        generated = generate_cursor_rules(rules_dir, detected)
        names = [p.name for p in generated]
        assert "mirdan-planning.mdc" in names

    def test_mdc_has_frontmatter(self, tmp_path: Path) -> None:
        """Generated .mdc files should have valid YAML frontmatter."""
        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        detected = DetectedProject(primary_language="python")

        generated = generate_cursor_rules(rules_dir, detected)
        for path in generated:
            content = path.read_text()
            assert content.startswith("---"), f"{path.name} missing frontmatter"
            # Should have closing ---
            parts = content.split("---")
            assert len(parts) >= 3, f"{path.name} missing frontmatter closing"

    def test_no_typescript_for_python(self, tmp_path: Path) -> None:
        """Should not generate TypeScript rule for Python project."""
        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        detected = DetectedProject(primary_language="python")

        generated = generate_cursor_rules(rules_dir, detected)
        names = [p.name for p in generated]
        assert "mirdan-typescript.mdc" not in names


# ---------------------------------------------------------------------------
# v0.2.0: --upgrade flag, AGENTS.md generation, migration
# ---------------------------------------------------------------------------


class TestUpgradeFlag:
    """Tests for mirdan init --upgrade."""

    def test_upgrade_flag_parsed(self) -> None:
        """--upgrade should be parsed from CLI args."""
        with (
            patch.object(sys, "argv", ["mirdan", "init", "--upgrade"]),
            patch("mirdan.cli._init") as mock_init,
        ):
            main()
            mock_init.assert_called_once_with(["--upgrade"])

    def test_upgrade_requires_existing_config(self, tmp_path: Path) -> None:
        """--upgrade should exit if no .mirdan/config.yaml exists."""
        with pytest.raises(SystemExit) as exc_info:
            run_init(["--upgrade", str(tmp_path)])
        assert exc_info.value.code == 1

    def test_upgrade_preserves_existing_config(self, tmp_path: Path) -> None:
        """--upgrade should preserve existing config values."""
        # Create existing config
        config_dir = tmp_path / ".mirdan"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.yaml"
        config_path.write_text(
            "version: '1'\nproject:\n  name: myapp\n  primary_language: python\n"
            "quality:\n  security: strict\n"
        )
        # Create minimal project file so detect_project works
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "myapp"\n')

        run_init(["--upgrade", str(tmp_path)])

        # Config should still exist and contain project name
        assert config_path.exists()
        content = config_path.read_text()
        assert "myapp" in content

    def test_upgrade_generates_agents_md(self, tmp_path: Path) -> None:
        """--upgrade should generate root AGENTS.md."""
        config_dir = tmp_path / ".mirdan"
        config_dir.mkdir(parents=True)
        (config_dir / "config.yaml").write_text(
            "version: '1'\nproject:\n  name: testapp\n  primary_language: python\n"
            "quality:\n  security: strict\n"
        )
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "testapp"\n')

        run_init(["--upgrade", str(tmp_path)])

        agents_md = tmp_path / "AGENTS.md"
        assert agents_md.exists()


class TestAgentsMdGeneration:
    """Tests for AGENTS.md generation during init."""

    def test_init_generates_agents_md(self, tmp_path: Path) -> None:
        """Standard init should generate root AGENTS.md."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "testapp"\n')

        with patch("builtins.input", return_value=""):
            run_init([str(tmp_path)])

        agents_md = tmp_path / "AGENTS.md"
        assert agents_md.exists()

    def test_agents_md_has_quality_rules(self, tmp_path: Path) -> None:
        """Generated AGENTS.md should contain quality rules."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "testapp"\n')

        with patch("builtins.input", return_value=""):
            run_init([str(tmp_path)])

        content = (tmp_path / "AGENTS.md").read_text()
        assert "AI001" in content

    def test_agents_md_mentions_language(self, tmp_path: Path) -> None:
        """Generated AGENTS.md should mention the detected language."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "testapp"\n')

        with patch("builtins.input", return_value=""):
            run_init([str(tmp_path)])

        content = (tmp_path / "AGENTS.md").read_text()
        assert "Python" in content


class TestClaudeCodeInit:
    """Tests for Claude Code-specific init with v0.2.0 features."""

    def test_init_claude_code_generates_workflow_rule(self, tmp_path: Path) -> None:
        """--claude-code should generate mirdan-workflow.md."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "testapp"\n')

        with patch("builtins.input", return_value=""):
            run_init(["--claude-code", str(tmp_path)])

        workflow = tmp_path / ".claude" / "rules" / "mirdan-workflow.md"
        assert workflow.exists()
        content = workflow.read_text()
        assert "enhance_prompt" in content

    def test_init_claude_code_generates_hooks_with_all_events(self, tmp_path: Path) -> None:
        """--claude-code hooks.json should include advanced events."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "testapp"\n')

        with patch("builtins.input", return_value=""):
            run_init(["--claude-code", str(tmp_path)])

        hooks_path = tmp_path / ".claude" / "hooks.json"
        assert hooks_path.exists()
        data = json.loads(hooks_path.read_text())
        hooks = data["hooks"]
        # Should have all 9 events
        assert "SessionStart" in hooks
        assert "PreCompact" in hooks

    def test_fix_command_routed(self) -> None:
        """'mirdan fix' should route to the fix command."""
        with (
            patch.object(sys, "argv", ["mirdan", "fix", "test.py"]),
            patch("mirdan.cli._fix") as mock_fix,
        ):
            main()
            mock_fix.assert_called_once_with(["test.py"])
