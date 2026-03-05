"""Tests for manifest_parser.py — dependency manifest parsing."""

from __future__ import annotations

import json

import pytest

from mirdan.core.manifest_parser import ManifestParser


class TestManifestParser:
    """Tests for ManifestParser."""

    def test_parse_pyproject_toml(self, tmp_path: pytest.TempPathFactory) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "myapp"\ndependencies = [\n'
            '  "fastmcp>=2.0.0",\n  "pyyaml>=6.0",\n]\n'
        )
        parser = ManifestParser(project_dir=tmp_path)
        packages = parser.parse()
        names = {p.name for p in packages}
        assert "fastmcp" in names
        assert "pyyaml" in names
        assert all(p.ecosystem == "PyPI" for p in packages)

    def test_parse_requirements_txt(self, tmp_path: pytest.TempPathFactory) -> None:
        reqs = tmp_path / "requirements.txt"
        reqs.write_text("requests==2.31.0\nflask>=3.0\n# comment\n")
        parser = ManifestParser(project_dir=tmp_path)
        packages = parser.parse()
        names = {p.name for p in packages}
        assert "requests" in names
        assert "flask" in names

    def test_parse_package_json(self, tmp_path: pytest.TempPathFactory) -> None:
        pkg = tmp_path / "package.json"
        pkg.write_text(json.dumps({
            "dependencies": {"react": "^18.2.0", "next": "~14.0.0"},
            "devDependencies": {"typescript": "^5.0.0"},
        }))
        parser = ManifestParser(project_dir=tmp_path)
        packages = parser.parse()
        names = {p.name for p in packages}
        assert "react" in names
        assert "next" in names
        assert "typescript" in names
        ts_pkg = next(p for p in packages if p.name == "typescript")
        assert ts_pkg.is_dev is True
        assert ts_pkg.ecosystem == "npm"

    def test_parse_cargo_toml(self, tmp_path: pytest.TempPathFactory) -> None:
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text(
            '[package]\nname = "myapp"\n\n[dependencies]\n'
            'serde = "1.0"\ntokio = { version = "1.35", features = ["full"] }\n'
        )
        parser = ManifestParser(project_dir=tmp_path)
        packages = parser.parse()
        names = {p.name for p in packages}
        assert "serde" in names
        assert "tokio" in names

    def test_parse_go_mod(self, tmp_path: pytest.TempPathFactory) -> None:
        go_mod = tmp_path / "go.mod"
        go_mod.write_text(
            "module example.com/myapp\n\ngo 1.21\n\nrequire (\n"
            "\tgithub.com/gin-gonic/gin v1.9.1\n"
            "\tgithub.com/go-sql-driver/mysql v1.7.0\n)\n"
        )
        parser = ManifestParser(project_dir=tmp_path)
        packages = parser.parse()
        names = {p.name for p in packages}
        assert "gin" in names
        assert "mysql" in names

    def test_no_manifests_returns_empty(self, tmp_path: pytest.TempPathFactory) -> None:
        parser = ManifestParser(project_dir=tmp_path)
        assert parser.parse() == []

    def test_malformed_manifest_doesnt_crash(self, tmp_path: pytest.TempPathFactory) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("this is not valid toml {{{{")
        parser = ManifestParser(project_dir=tmp_path)
        # Should not raise, just return empty
        packages = parser.parse()
        assert packages == []

    def test_cache_invalidation(self, tmp_path: pytest.TempPathFactory) -> None:
        reqs = tmp_path / "requirements.txt"
        reqs.write_text("flask==3.0.0\n")
        parser = ManifestParser(project_dir=tmp_path)
        packages1 = parser.parse()
        assert len(packages1) == 1

        # Modify file
        reqs.write_text("flask==3.0.0\nrequests==2.31.0\n")
        packages2 = parser.parse()
        assert len(packages2) == 2

    def test_get_version(self, tmp_path: pytest.TempPathFactory) -> None:
        reqs = tmp_path / "requirements.txt"
        reqs.write_text("flask==3.0.0\n")
        parser = ManifestParser(project_dir=tmp_path)
        parser.parse()
        assert parser.get_version("flask", "PyPI") == "3.0.0"
        assert parser.get_version("nonexistent", "PyPI") is None

    def test_multiple_manifests(self, tmp_path: pytest.TempPathFactory) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "myapp"\ndependencies = ["flask>=3.0"]\n'
        )
        pkg_json = tmp_path / "package.json"
        pkg_json.write_text(json.dumps({"dependencies": {"react": "^18.0"}}))
        parser = ManifestParser(project_dir=tmp_path)
        packages = parser.parse()
        ecosystems = {p.ecosystem for p in packages}
        assert "PyPI" in ecosystems
        assert "npm" in ecosystems
