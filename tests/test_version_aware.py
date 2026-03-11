"""Tests for version-aware quality standards."""

from __future__ import annotations

import json
from pathlib import Path

from mirdan.config import QualityConfig
from mirdan.core.manifest_parser import ManifestParser
from mirdan.core.quality_standards import QualityStandards


class TestManifestParserFrameworkVersion:
    """Tests for ManifestParser.get_framework_version()."""

    def test_python_pyproject_version(self, tmp_path: Path) -> None:
        """Should extract version from pyproject.toml dependencies."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "app"\ndependencies = [\n  "fastapi>=0.115.0",\n]\n'
        )

        parser = ManifestParser(project_dir=tmp_path)
        version = parser.get_framework_version("fastapi")
        assert version == "0.115.0"

    def test_python_pyproject_exact_version(self, tmp_path: Path) -> None:
        """Should extract exact (==) version."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "app"\ndependencies = [\n  "django==4.2.1",\n]\n')

        parser = ManifestParser(project_dir=tmp_path)
        version = parser.get_framework_version("django")
        assert version == "4.2.1"

    def test_node_package_json_version(self, tmp_path: Path) -> None:
        """Should extract version from package.json."""
        pkg = tmp_path / "package.json"
        pkg.write_text(
            json.dumps(
                {
                    "dependencies": {"react": "^19.0.0"},
                }
            )
        )

        parser = ManifestParser(project_dir=tmp_path)
        version = parser.get_framework_version("react")
        assert version == "19.0.0"

    def test_node_dev_dependency(self, tmp_path: Path) -> None:
        """Should find version in devDependencies."""
        pkg = tmp_path / "package.json"
        pkg.write_text(
            json.dumps(
                {
                    "devDependencies": {"typescript": "~5.3.0"},
                }
            )
        )

        parser = ManifestParser(project_dir=tmp_path)
        version = parser.get_framework_version("typescript")
        assert version == "5.3.0"

    def test_not_found_returns_none(self, tmp_path: Path) -> None:
        """Should return None when framework not found."""
        parser = ManifestParser(project_dir=tmp_path)
        version = parser.get_framework_version("react")
        assert version is None

    def test_invalid_json_returns_none(self, tmp_path: Path) -> None:
        """Should handle invalid package.json gracefully."""
        pkg = tmp_path / "package.json"
        pkg.write_text("not json")

        parser = ManifestParser(project_dir=tmp_path)
        version = parser.get_framework_version("react")
        assert version is None

    def test_version_with_multiple_specifiers(self, tmp_path: Path) -> None:
        """Should handle version with multiple specifiers (takes first)."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "app"\ndependencies = [\n  "pydantic>=2.0,<3.0",\n]\n'
        )

        parser = ManifestParser(project_dir=tmp_path)
        version = parser.get_framework_version("pydantic")
        assert version == "2.0"


class TestVersionAwareStandards:
    """Tests for version-aware framework standards loading."""

    def test_detect_framework_version_with_project_dir(self, tmp_path: Path) -> None:
        """Should detect version when project_dir is set."""
        pkg = tmp_path / "package.json"
        pkg.write_text(
            json.dumps(
                {
                    "dependencies": {"react": "^18.2.0"},
                }
            )
        )

        parser = ManifestParser(project_dir=tmp_path)
        standards = QualityStandards(project_dir=tmp_path, manifest_parser=parser)
        version = standards.detect_framework_version("react")
        assert version == "18.2.0"

    def test_detect_framework_version_without_manifest_parser(self) -> None:
        """Should return None when manifest_parser is not set."""
        standards = QualityStandards()
        version = standards.detect_framework_version("react")
        assert version is None

    def test_get_for_framework_falls_back_to_generic(self) -> None:
        """Should return generic standards when no versioned file exists."""
        standards = QualityStandards()
        # React standards exist in the built-in standards
        result = standards.get_for_framework("react")
        # Should return something (the generic react standards)
        assert isinstance(result, dict)

    def test_get_for_framework_with_version_no_versioned_file(self, tmp_path: Path) -> None:
        """Should fall back to generic when versioned file doesn't exist."""
        # Use vue@4.0.0 — no vue-4.yaml exists, so this tests the fallback path
        pkg = tmp_path / "package.json"
        pkg.write_text(
            json.dumps(
                {
                    "dependencies": {"vue": "^4.0.0"},
                }
            )
        )

        parser = ManifestParser(project_dir=tmp_path)
        standards = QualityStandards(project_dir=tmp_path, manifest_parser=parser)
        # vue-4.yaml doesn't exist, should fall back to vue.yaml
        result = standards.get_for_framework("vue")
        assert isinstance(result, dict)

    def test_react19_versioned_standards_loaded(self, tmp_path: Path) -> None:
        """Should merge react-19.yaml on top of react.yaml for React 19 projects."""
        pkg = tmp_path / "package.json"
        pkg.write_text(
            json.dumps(
                {
                    "dependencies": {"react": "^19.0.0"},
                }
            )
        )

        parser = ManifestParser(project_dir=tmp_path)
        standards = QualityStandards(project_dir=tmp_path, manifest_parser=parser)
        result = standards.get_for_framework("react")
        # react-19.yaml adds React 19-specific principles — merged result should be a non-empty dict
        assert isinstance(result, dict)
        assert "principles" in result
        # React 19 principles mention the compiler or use() hook or Server Actions
        principles_text = " ".join(result["principles"]).lower()
        assert any(
            keyword in principles_text
            for keyword in ("compiler", "use()", "server action", "useoptimistic", "useactionstate")
        )

    def test_nextjs15_versioned_standards_loaded(self, tmp_path: Path) -> None:
        """Should merge next.js-15.yaml on top of next.js standards for Next.js 15 projects."""
        # The npm package is "next" which maps to framework "next.js"
        pkg = tmp_path / "package.json"
        pkg.write_text(
            json.dumps(
                {
                    "dependencies": {"next": "^15.0.0"},
                }
            )
        )

        parser = ManifestParser(project_dir=tmp_path)
        standards = QualityStandards(project_dir=tmp_path, manifest_parser=parser)
        result = standards.get_for_framework("next.js")
        assert isinstance(result, dict)
        assert "principles" in result
        # Next.js 15 principles mention async params or after()
        principles_text = " ".join(result["principles"]).lower()
        assert any(
            keyword in principles_text
            for keyword in ("params", "await", "after()", "turbopack", "fetch()")
        )

    def test_project_dir_passed_to_standards(self, tmp_path: Path) -> None:
        """QualityStandards should accept and store project_dir."""
        config = QualityConfig()
        standards = QualityStandards(config=config, project_dir=tmp_path)
        assert standards._project_dir == tmp_path
