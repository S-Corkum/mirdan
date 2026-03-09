"""Tests for workspace/monorepo detection in mirdan.cli.detect."""

from __future__ import annotations

import json
from pathlib import Path

from mirdan.cli.detect import (
    detect_workspace,
    resolve_file_to_project,
)


class TestDetectWorkspaceUvMembers:
    """Detect a uv workspace from pyproject.toml [tool.uv.workspace] members."""

    def test_detect_workspace_uv_members(self, tmp_path: Path) -> None:
        root_pyproject = tmp_path / "pyproject.toml"
        root_pyproject.write_text('[tool.uv.workspace]\nmembers = ["api", "web"]\n')

        # api sub-project: Python (pyproject.toml)
        api_dir = tmp_path / "api"
        api_dir.mkdir()
        (api_dir / "pyproject.toml").write_text('[project]\nname = "api"\n')

        # web sub-project: TypeScript (package.json + tsconfig.json)
        web_dir = tmp_path / "web"
        web_dir.mkdir()
        (web_dir / "package.json").write_text(json.dumps({"name": "web"}))
        (web_dir / "tsconfig.json").write_text("{}")

        ws = detect_workspace(tmp_path)

        assert ws is not None
        assert ws.workspace_type == "uv"
        assert len(ws.projects) == 2


class TestDetectWorkspaceNpm:
    """Detect an npm workspace from package.json workspaces key."""

    def test_detect_workspace_npm(self, tmp_path: Path) -> None:
        root_pkg = tmp_path / "package.json"
        root_pkg.write_text(
            json.dumps(
                {
                    "name": "monorepo",
                    "workspaces": ["packages/api", "packages/web"],
                }
            )
        )

        # packages/api  — node project
        api_dir = tmp_path / "packages" / "api"
        api_dir.mkdir(parents=True)
        (api_dir / "package.json").write_text(json.dumps({"name": "@mono/api"}))

        # packages/web  — node project
        web_dir = tmp_path / "packages" / "web"
        web_dir.mkdir(parents=True)
        (web_dir / "package.json").write_text(json.dumps({"name": "@mono/web"}))

        ws = detect_workspace(tmp_path)

        assert ws is not None
        assert ws.workspace_type == "npm"
        assert len(ws.projects) == 2
        project_names = {p.project_name for p in ws.projects}
        assert "@mono/api" in project_names
        assert "@mono/web" in project_names


class TestDetectWorkspaceCargo:
    """Detect a Cargo workspace from Cargo.toml [workspace] members."""

    def test_detect_workspace_cargo(self, tmp_path: Path) -> None:
        root_cargo = tmp_path / "Cargo.toml"
        root_cargo.write_text('[workspace]\nmembers = ["lib", "cli"]\n')

        # lib sub-project
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "Cargo.toml").write_text('[package]\nname = "mylib"\nversion = "0.1.0"\n')

        # cli sub-project
        cli_dir = tmp_path / "cli"
        cli_dir.mkdir()
        (cli_dir / "Cargo.toml").write_text('[package]\nname = "mycli"\nversion = "0.1.0"\n')

        ws = detect_workspace(tmp_path)

        assert ws is not None
        assert ws.workspace_type == "cargo"
        assert len(ws.projects) == 2
        types = {p.project_type for p in ws.projects}
        assert types == {"rust"}


class TestDetectWorkspaceAutoScan:
    """Fallback auto-detection when no root workspace manifest exists."""

    def test_detect_workspace_auto_scan(self, tmp_path: Path) -> None:
        # api sub-dir with pyproject.toml  → Python
        api_dir = tmp_path / "api"
        api_dir.mkdir()
        (api_dir / "pyproject.toml").write_text('[project]\nname = "api"\n')

        # infra sub-dir with go.mod  → Go
        infra_dir = tmp_path / "infra"
        infra_dir.mkdir()
        (infra_dir / "go.mod").write_text("module example.com/infra\n\ngo 1.22\n")

        ws = detect_workspace(tmp_path)

        assert ws is not None
        assert ws.workspace_type == "auto"
        assert len(ws.projects) == 2


class TestDetectWorkspaceSingleProjectReturnsNone:
    """A single sub-project should not be treated as a workspace."""

    def test_detect_workspace_single_project_returns_none(self, tmp_path: Path) -> None:
        only_dir = tmp_path / "only"
        only_dir.mkdir()
        (only_dir / "pyproject.toml").write_text('[project]\nname = "only"\n')

        ws = detect_workspace(tmp_path)

        assert ws is None


class TestDetectWorkspaceNoManifestsReturnsNone:
    """Empty sub-dirs without manifests should not be treated as a workspace."""

    def test_detect_workspace_no_manifests_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "empty_a").mkdir()
        (tmp_path / "empty_b").mkdir()

        ws = detect_workspace(tmp_path)

        assert ws is None


class TestResolveFileToProject:
    """resolve_file_to_project maps a file path to its containing sub-project."""

    def test_resolve_file_to_project(self, tmp_path: Path) -> None:
        # Set up a uv workspace with two members
        (tmp_path / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["api", "web"]\n')

        api_dir = tmp_path / "api"
        api_dir.mkdir()
        (api_dir / "pyproject.toml").write_text('[project]\nname = "api"\n')

        web_dir = tmp_path / "web"
        web_dir.mkdir()
        (web_dir / "pyproject.toml").write_text('[project]\nname = "web"\n')

        ws = detect_workspace(tmp_path)
        assert ws is not None

        matched = resolve_file_to_project(ws, "api/routes.py")
        assert matched is not None
        assert matched.project_name == "api"


class TestResolveFileLongestMatch:
    """With nested projects, the longest sub_path prefix should win."""

    def test_resolve_file_longest_match(self, tmp_path: Path) -> None:
        # Create two auto-detected projects at different nesting depths
        services_dir = tmp_path / "services"
        services_dir.mkdir()
        (services_dir / "pyproject.toml").write_text('[project]\nname = "services"\n')

        api_dir = services_dir / "api"
        api_dir.mkdir()
        (api_dir / "pyproject.toml").write_text('[project]\nname = "services-api"\n')

        # We need at least 2 projects for detect_workspace to return a result,
        # so add a second top-level project.
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        (other_dir / "pyproject.toml").write_text('[project]\nname = "other"\n')

        ws = detect_workspace(tmp_path)
        assert ws is not None

        # Manually add the nested project so both are present,
        # since auto-scan only looks at immediate children.
        from mirdan.cli.detect import DetectedProject

        nested = DetectedProject(
            project_type="python",
            project_name="services-api",
            primary_language="python",
            manifest_path=str(api_dir / "pyproject.toml"),
            sub_path="services/api",
        )
        # Only add if not already present
        if not any(p.sub_path == "services/api" for p in ws.projects):
            ws.projects.append(nested)

        matched = resolve_file_to_project(ws, "services/api/main.py")
        assert matched is not None
        assert matched.project_name == "services-api"

        # A file directly under services/ should match the services project
        matched_parent = resolve_file_to_project(ws, "services/util.py")
        assert matched_parent is not None
        assert matched_parent.project_name == "services"


class TestDetectWorkspaceAggregatesLanguages:
    """all_languages and all_frameworks are correctly aggregated and dedduped."""

    def test_detect_workspace_aggregates_languages(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[tool.uv.workspace]\nmembers = ["svc1", "svc2", "web"]\n'
        )

        # Two Python projects
        for name in ("svc1", "svc2"):
            d = tmp_path / name
            d.mkdir()
            (d / "pyproject.toml").write_text(
                f'[project]\nname = "{name}"\ndependencies = ["fastapi>=0.100"]\n'
            )

        # One TypeScript project
        web_dir = tmp_path / "web"
        web_dir.mkdir()
        (web_dir / "package.json").write_text(
            json.dumps({"name": "web", "dependencies": {"react": "^18.0.0"}})
        )
        (web_dir / "tsconfig.json").write_text("{}")

        ws = detect_workspace(tmp_path)
        assert ws is not None

        # Languages should be deduplicated
        assert "python" in ws.all_languages
        assert "typescript" in ws.all_languages
        assert ws.all_languages.count("python") == 1

        # Frameworks should be deduplicated
        assert "fastapi" in ws.all_frameworks
        assert "react" in ws.all_frameworks
        assert ws.all_frameworks.count("fastapi") == 1


class TestDetectWorkspaceSetsSubPath:
    """Each project in a workspace gets the correct sub_path."""

    def test_detect_workspace_sets_sub_path(self, tmp_path: Path) -> None:
        root_pkg = tmp_path / "package.json"
        root_pkg.write_text(
            json.dumps(
                {
                    "name": "mono",
                    "workspaces": ["packages/core", "packages/ui"],
                }
            )
        )

        core_dir = tmp_path / "packages" / "core"
        core_dir.mkdir(parents=True)
        (core_dir / "package.json").write_text(json.dumps({"name": "@mono/core"}))

        ui_dir = tmp_path / "packages" / "ui"
        ui_dir.mkdir(parents=True)
        (ui_dir / "package.json").write_text(json.dumps({"name": "@mono/ui"}))

        ws = detect_workspace(tmp_path)
        assert ws is not None

        sub_paths = {p.project_name: p.sub_path for p in ws.projects}
        assert sub_paths["@mono/core"] == "packages/core"
        assert sub_paths["@mono/ui"] == "packages/ui"
