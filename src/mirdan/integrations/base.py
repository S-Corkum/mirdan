"""Platform adapter base class for IDE integrations.

Defines the abstract interface that all platform adapters (Claude Code,
Cursor, etc.) must implement. Each adapter generates platform-specific
configuration files: hooks, rules, agents, and MCP server config.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mirdan.cli.detect import DetectedProject
    from mirdan.core.quality_standards import QualityStandards


class PlatformAdapter(abc.ABC):
    """Abstract base class for platform-specific configuration generators.

    Subclasses implement the four generation methods to produce
    platform-appropriate configuration files.

    Args:
        project_dir: The project root directory.
        detected: Detected project metadata.
        standards: Optional QualityStandards for content generation.
    """

    def __init__(
        self,
        project_dir: Path,
        detected: DetectedProject,
        standards: QualityStandards | None = None,
    ) -> None:
        self.project_dir = project_dir
        self.detected = detected
        self.standards = standards

    @abc.abstractmethod
    def generate_hooks(self) -> list[Path]:
        """Generate hook configuration files.

        Returns:
            List of generated hook file paths.
        """

    @abc.abstractmethod
    def generate_rules(self) -> list[Path]:
        """Generate rule/standard files.

        Returns:
            List of generated rule file paths.
        """

    @abc.abstractmethod
    def generate_agents(self) -> list[Path]:
        """Generate agent configuration files.

        Returns:
            List of generated agent file paths.
        """

    @abc.abstractmethod
    def generate_mcp_config(self) -> Path | None:
        """Generate MCP server configuration.

        Returns:
            Path to the generated config file, or None if not applicable.
        """

    def generate_all(self) -> list[Path]:
        """Call all generators, return all created paths."""
        paths: list[Path] = []
        paths.extend(self.generate_hooks())
        paths.extend(self.generate_rules())
        paths.extend(self.generate_agents())
        mcp = self.generate_mcp_config()
        if mcp:
            paths.append(mcp)
        return paths
