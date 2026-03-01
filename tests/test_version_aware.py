"""Tests for version-aware quality standards."""

from __future__ import annotations

import json
from pathlib import Path

from mirdan.config import QualityConfig
from mirdan.core.quality_standards import QualityStandards, _detect_version_from_manifests


class TestDetectVersionFromManifests:
    """Tests for _detect_version_from_manifests helper."""

    def test_python_pyproject_version(self, tmp_path: Path) -> None:
        """Should extract version from pyproject.toml dependencies."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "app"\ndependencies = [\n  "fastapi>=0.115.0",\n]\n'
        )

        version = _detect_version_from_manifests("fastapi", tmp_path)
        assert version == "0.115.0"

    def test_python_pyproject_exact_version(self, tmp_path: Path) -> None:
        """Should extract exact (==) version."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "app"\ndependencies = [\n  "django==4.2.1",\n]\n')

        version = _detect_version_from_manifests("django", tmp_path)
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

        version = _detect_version_from_manifests("react", tmp_path)
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

        version = _detect_version_from_manifests("typescript", tmp_path)
        assert version == "5.3.0"

    def test_not_found_returns_none(self, tmp_path: Path) -> None:
        """Should return None when framework not found."""
        version = _detect_version_from_manifests("react", tmp_path)
        assert version is None

    def test_invalid_json_returns_none(self, tmp_path: Path) -> None:
        """Should handle invalid package.json gracefully."""
        pkg = tmp_path / "package.json"
        pkg.write_text("not json")

        version = _detect_version_from_manifests("react", tmp_path)
        assert version is None

    def test_version_with_multiple_specifiers(self, tmp_path: Path) -> None:
        """Should handle version with multiple specifiers (takes first)."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "app"\ndependencies = [\n  "pydantic>=2.0,<3.0",\n]\n'
        )

        version = _detect_version_from_manifests("pydantic", tmp_path)
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

        standards = QualityStandards(project_dir=tmp_path)
        version = standards.detect_framework_version("react")
        assert version == "18.2.0"

    def test_detect_framework_version_without_project_dir(self) -> None:
        """Should return None when project_dir is not set."""
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
        pkg = tmp_path / "package.json"
        pkg.write_text(
            json.dumps(
                {
                    "dependencies": {"react": "^19.0.0"},
                }
            )
        )

        standards = QualityStandards(project_dir=tmp_path)
        # react-19.yaml doesn't exist, should fall back to react.yaml
        result = standards.get_for_framework("react")
        assert isinstance(result, dict)

    def test_project_dir_passed_to_standards(self, tmp_path: Path) -> None:
        """QualityStandards should accept and store project_dir."""
        config = QualityConfig()
        standards = QualityStandards(config=config, project_dir=tmp_path)
        assert standards._project_dir == tmp_path
