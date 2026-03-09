"""Tests for workspace configuration models."""

from pathlib import Path

import yaml

from mirdan.config import MirdanConfig, SubProjectConfig, WorkspaceConfig


class TestWorkspaceConfigYamlRoundtrip:
    """Tests that workspace config survives YAML serialization."""

    def test_workspace_config_yaml_roundtrip(self, tmp_path: Path) -> None:
        """Create a MirdanConfig with workspace enabled and 2 sub-projects,
        save to YAML, reload, and verify all fields preserved."""
        config = MirdanConfig(
            workspace=WorkspaceConfig(
                enabled=True,
                workspace_type="monorepo",
                projects=[
                    SubProjectConfig(
                        path="api/",
                        name="backend",
                        primary_language="python",
                        frameworks=["fastapi", "pydantic"],
                    ),
                    SubProjectConfig(
                        path="web/",
                        name="frontend",
                        primary_language="typescript",
                        frameworks=["react", "vite"],
                    ),
                ],
            ),
        )

        config_path = tmp_path / ".mirdan" / "config.yaml"
        config.save(config_path)

        loaded = MirdanConfig.load(config_path)

        assert loaded.workspace.enabled is True
        assert loaded.workspace.workspace_type == "monorepo"
        assert len(loaded.workspace.projects) == 2

        api_proj = loaded.workspace.projects[0]
        assert api_proj.path == "api/"
        assert api_proj.name == "backend"
        assert api_proj.primary_language == "python"
        assert api_proj.frameworks == ["fastapi", "pydantic"]

        web_proj = loaded.workspace.projects[1]
        assert web_proj.path == "web/"
        assert web_proj.name == "frontend"
        assert web_proj.primary_language == "typescript"
        assert web_proj.frameworks == ["react", "vite"]


class TestV1ConfigLoadsWithoutWorkspace:
    """Tests backward compatibility for v1.0 configs without workspace."""

    def test_v1_config_loads_without_workspace(self, tmp_path: Path) -> None:
        """A v1.0 config YAML without workspace section should load with
        workspace.enabled=False and is_workspace=False."""
        v1_data = {
            "version": "1.0",
            "project": {
                "name": "legacy-app",
                "type": "application",
                "primary_language": "python",
            },
        }

        config_path = tmp_path / "config.yaml"
        with config_path.open("w") as f:
            yaml.dump(v1_data, f)

        loaded = MirdanConfig.load(config_path)

        assert loaded.workspace.enabled is False
        assert loaded.is_workspace is False
        assert loaded.project.name == "legacy-app"


class TestIsWorkspaceProperty:
    """Tests for the is_workspace computed property."""

    def test_enabled_with_projects_returns_true(self) -> None:
        """is_workspace returns True when enabled=True AND projects is non-empty."""
        config = MirdanConfig(
            workspace=WorkspaceConfig(
                enabled=True,
                projects=[
                    SubProjectConfig(path="svc/", name="service"),
                ],
            ),
        )
        assert config.is_workspace is True

    def test_enabled_without_projects_returns_false(self) -> None:
        """is_workspace returns False when enabled=True but projects is empty."""
        config = MirdanConfig(
            workspace=WorkspaceConfig(enabled=True, projects=[]),
        )
        assert config.is_workspace is False

    def test_disabled_with_projects_returns_false(self) -> None:
        """is_workspace returns False when enabled=False even with projects."""
        config = MirdanConfig(
            workspace=WorkspaceConfig(
                enabled=False,
                projects=[
                    SubProjectConfig(path="svc/", name="service"),
                ],
            ),
        )
        assert config.is_workspace is False

    def test_disabled_without_projects_returns_false(self) -> None:
        """is_workspace returns False when both disabled and no projects."""
        config = MirdanConfig(
            workspace=WorkspaceConfig(enabled=False, projects=[]),
        )
        assert config.is_workspace is False


class TestResolveProjectForPath:
    """Tests for resolve_project_for_path method."""

    def test_resolve_project_for_path(self) -> None:
        """Resolve file paths to the correct sub-project config."""
        config = MirdanConfig(
            workspace=WorkspaceConfig(
                enabled=True,
                projects=[
                    SubProjectConfig(path="api/", name="backend", primary_language="python"),
                    SubProjectConfig(path="web/", name="frontend", primary_language="typescript"),
                ],
            ),
        )

        api_result = config.resolve_project_for_path("api/routes.py")
        assert api_result is not None
        assert api_result.name == "backend"
        assert api_result.primary_language == "python"

        web_result = config.resolve_project_for_path("web/App.tsx")
        assert web_result is not None
        assert web_result.name == "frontend"
        assert web_result.primary_language == "typescript"

    def test_resolve_project_for_path_no_match(self) -> None:
        """Unknown paths return None."""
        config = MirdanConfig(
            workspace=WorkspaceConfig(
                enabled=True,
                projects=[
                    SubProjectConfig(path="api/", name="backend"),
                    SubProjectConfig(path="web/", name="frontend"),
                ],
            ),
        )

        result = config.resolve_project_for_path("docs/README.md")
        assert result is None

    def test_resolve_project_for_path_not_workspace(self) -> None:
        """Non-workspace configs always return None."""
        config = MirdanConfig(
            workspace=WorkspaceConfig(enabled=False, projects=[]),
        )

        result = config.resolve_project_for_path("api/routes.py")
        assert result is None

    def test_resolve_project_for_path_longest_match(self) -> None:
        """With nested projects, longest prefix match wins."""
        config = MirdanConfig(
            workspace=WorkspaceConfig(
                enabled=True,
                projects=[
                    SubProjectConfig(path="services/", name="all-services"),
                    SubProjectConfig(path="services/api/", name="api-service"),
                ],
            ),
        )

        # A file under services/api/ should match the more specific project
        result = config.resolve_project_for_path("services/api/handler.py")
        assert result is not None
        assert result.name == "api-service"

        # A file directly under services/ (not api/) matches the broader project
        result = config.resolve_project_for_path("services/worker.py")
        assert result is not None
        assert result.name == "all-services"


class TestFindConfigWithPath:
    """Tests for find_config_with_path class method."""

    def test_find_config_with_path(self, tmp_path: Path) -> None:
        """find_config_with_path returns both the config and the directory."""
        mirdan_dir = tmp_path / ".mirdan"
        mirdan_dir.mkdir()
        config_file = mirdan_dir / "config.yaml"

        data = {
            "version": "1.0",
            "project": {"name": "test-project"},
            "workspace": {
                "enabled": True,
                "projects": [
                    {"path": "src/", "name": "source"},
                ],
            },
        }
        with config_file.open("w") as f:
            yaml.dump(data, f)

        # Create a subdirectory to search from
        sub_dir = tmp_path / "src" / "deep" / "nested"
        sub_dir.mkdir(parents=True)

        config, found_dir = MirdanConfig.find_config_with_path(sub_dir)

        assert found_dir == tmp_path
        assert config.project.name == "test-project"
        assert config.workspace.enabled is True
        assert len(config.workspace.projects) == 1

    def test_find_config_with_path_not_found(self, tmp_path: Path) -> None:
        """When no config exists, returns (default_config, None)."""
        # Create an isolated directory with no .mirdan config anywhere
        isolated = tmp_path / "empty" / "project"
        isolated.mkdir(parents=True)

        config, found_dir = MirdanConfig.find_config_with_path(isolated)

        assert found_dir is None
        # Should be a default config
        assert config.workspace.enabled is False
        assert config.is_workspace is False
        assert config.project.name == ""
