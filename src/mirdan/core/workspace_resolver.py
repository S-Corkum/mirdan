"""Workspace-aware project resolution for multi-project repositories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mirdan.config import MirdanConfig


@dataclass
class ResolvedProject:
    """Result of resolving a file to its containing sub-project."""

    name: str
    path: str  # relative path from workspace root
    project_dir: Path  # absolute path to sub-project directory
    language: str
    frameworks: list[str] = field(default_factory=list)


class WorkspaceResolver:
    """Resolves files to their containing sub-project in a workspace.

    Uses longest-prefix matching on sub-project paths for correct
    resolution of nested projects.
    """

    def __init__(self, config: MirdanConfig, workspace_root: Path) -> None:
        self._workspace_root = workspace_root
        self._projects: list[tuple[str, ResolvedProject]] = []

        # Build lookup sorted by path length descending (longest prefix first)
        for sub in config.workspace.projects:
            resolved = ResolvedProject(
                name=sub.name,
                path=sub.path,
                project_dir=workspace_root / sub.path,
                language=sub.primary_language,
                frameworks=list(sub.frameworks),
            )
            self._projects.append((sub.path, resolved))

        self._projects.sort(key=lambda x: len(x[0]), reverse=True)

    def resolve(self, file_path: str | Path) -> ResolvedProject | None:
        """Match file_path against sub-project paths (longest prefix wins).

        Args:
            file_path: File path relative to workspace root.

        Returns:
            ResolvedProject if matched, None otherwise.
        """
        path_str = str(file_path)
        for prefix, project in self._projects:
            if not prefix:
                continue
            normalized = prefix if prefix.endswith("/") else prefix + "/"
            if path_str.startswith(normalized):
                return project
        return None

    def all_languages(self) -> list[str]:
        """Deduplicated languages across all projects."""
        seen: list[str] = []
        for _prefix, proj in self._projects:
            if proj.language and proj.language not in seen:
                seen.append(proj.language)
        return seen

    def all_project_dirs(self) -> list[Path]:
        """All absolute sub-project directories."""
        return [proj.project_dir for _prefix, proj in self._projects]
