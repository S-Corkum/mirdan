"""Tests for workspace-aware project resolution."""

from pathlib import Path

from mirdan.config import MirdanConfig, SubProjectConfig, WorkspaceConfig
from mirdan.core.workspace_resolver import ResolvedProject, WorkspaceResolver


def _make_config(projects: list[SubProjectConfig]) -> MirdanConfig:
    """Build a MirdanConfig with workspace enabled and the given sub-projects."""
    return MirdanConfig(
        workspace=WorkspaceConfig(
            enabled=True,
            projects=projects,
        ),
    )


class TestWorkspaceResolver:
    """Tests for WorkspaceResolver file-to-project resolution."""

    def test_resolve_python_file(self, tmp_path: Path) -> None:
        """Resolving a file under the api/ prefix returns a python project."""
        config = _make_config(
            [
                SubProjectConfig(path="api/", name="api", primary_language="python"),
                SubProjectConfig(path="web/", name="web", primary_language="typescript"),
            ]
        )
        resolver = WorkspaceResolver(config, tmp_path)
        result = resolver.resolve("api/routes.py")

        assert result is not None
        assert isinstance(result, ResolvedProject)
        assert result.language == "python"
        assert result.name == "api"

    def test_resolve_typescript_file(self, tmp_path: Path) -> None:
        """Resolving a file under the web/ prefix returns a typescript project."""
        config = _make_config(
            [
                SubProjectConfig(path="api/", name="api", primary_language="python"),
                SubProjectConfig(path="web/", name="web", primary_language="typescript"),
            ]
        )
        resolver = WorkspaceResolver(config, tmp_path)
        result = resolver.resolve("web/App.tsx")

        assert result is not None
        assert result.language == "typescript"
        assert result.name == "web"

    def test_resolve_unknown_file_returns_none(self, tmp_path: Path) -> None:
        """Files outside all sub-project prefixes resolve to None."""
        config = _make_config(
            [
                SubProjectConfig(path="api/", name="api", primary_language="python"),
                SubProjectConfig(path="web/", name="web", primary_language="typescript"),
            ]
        )
        resolver = WorkspaceResolver(config, tmp_path)

        assert resolver.resolve("README.md") is None
        assert resolver.resolve("unknown/file.py") is None

    def test_all_languages_deduplicated(self, tmp_path: Path) -> None:
        """all_languages() deduplicates while preserving insertion order."""
        config = _make_config(
            [
                SubProjectConfig(path="svc-a/", name="svc-a", primary_language="python"),
                SubProjectConfig(path="svc-b/", name="svc-b", primary_language="python"),
                SubProjectConfig(path="web/", name="web", primary_language="typescript"),
            ]
        )
        resolver = WorkspaceResolver(config, tmp_path)
        languages = resolver.all_languages()

        assert "python" in languages
        assert "typescript" in languages
        assert len(languages) == 2

    def test_all_project_dirs(self, tmp_path: Path) -> None:
        """all_project_dirs() returns absolute paths for every sub-project."""
        config = _make_config(
            [
                SubProjectConfig(path="api/", name="api", primary_language="python"),
                SubProjectConfig(path="web/", name="web", primary_language="typescript"),
            ]
        )
        resolver = WorkspaceResolver(config, tmp_path)
        dirs = resolver.all_project_dirs()

        assert len(dirs) == 2
        assert all(isinstance(d, Path) for d in dirs)
        assert all(d.is_absolute() for d in dirs)
        assert tmp_path / "api/" in dirs
        assert tmp_path / "web/" in dirs

    def test_resolve_longest_prefix_match(self, tmp_path: Path) -> None:
        """When prefixes are nested, the longest matching prefix wins."""
        config = _make_config(
            [
                SubProjectConfig(
                    path="services/",
                    name="services",
                    primary_language="python",
                ),
                SubProjectConfig(
                    path="services/api/",
                    name="services-api",
                    primary_language="typescript",
                ),
            ]
        )
        resolver = WorkspaceResolver(config, tmp_path)
        result = resolver.resolve("services/api/routes.py")

        assert result is not None
        assert result.name == "services-api"
        assert result.language == "typescript"

    def test_resolve_empty_path_skipped(self, tmp_path: Path) -> None:
        """A sub-project with an empty path should not match any file."""
        config = _make_config(
            [
                SubProjectConfig(path="", name="root", primary_language="python"),
                SubProjectConfig(path="api/", name="api", primary_language="typescript"),
            ]
        )
        resolver = WorkspaceResolver(config, tmp_path)

        # The empty-path project must never match
        assert resolver.resolve("some/random/file.py") is None
        # The real prefix still works
        result = resolver.resolve("api/handler.ts")
        assert result is not None
        assert result.name == "api"
