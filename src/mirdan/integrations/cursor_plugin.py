"""Export mirdan as a Cursor IDE plugin.

Generates a complete Cursor plugin directory with rules, agents,
hooks, mcp.json, sandbox, and plugin.json manifest for distribution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from mirdan.cli.detect import DetectedProject

logger = logging.getLogger(__name__)


class CursorPluginExporter:
    """Exports mirdan configuration as a Cursor plugin."""

    def export(
        self,
        output_dir: Path,
        detected: DetectedProject | None = None,
    ) -> Path:
        """Generate a complete Cursor plugin directory.

        Creates rules (.mdc), AGENTS.md, BUGBOT.md, hooks, mcp.json,
        sandbox.json, and a .cursor-plugin/plugin.json manifest.

        Args:
            output_dir: Directory to write the plugin to.
            detected: Optional detected project info for language-specific rules.

        Returns:
            Path to the exported plugin directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if detected is None:
            detected = DetectedProject()

        self._write_manifest(output_dir)
        self._write_mcp_json(output_dir)
        self._write_rules(output_dir, detected)
        self._write_agents(output_dir, detected)
        self._write_hooks(output_dir)
        self._write_sandbox(output_dir, detected)
        self._write_commands(output_dir)
        self._write_subagents(output_dir)
        self._write_skills(output_dir)
        self._write_environment(output_dir, detected)

        return output_dir

    def _write_manifest(self, output_dir: Path) -> None:
        from mirdan.integrations.claude_code import _get_version

        plugin_dir = output_dir / ".cursor-plugin"
        plugin_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "name": "mirdan",
            "description": (
                "AI Code Quality Runtime — automatic quality enforcement"
                " with AI-specific slop detection"
            ),
            "version": _get_version(),
            "author": {"name": "Sean Corkum"},
            "license": "MIT",
            "keywords": ["code-quality", "security", "ai-safety"],
            "rules": "rules/",
            "agents": "agents/",
            "skills": "skills/",
            "commands": "commands/",
            "hooks": "hooks.json",
            "mcpServers": "mcp.json",
        }
        manifest_path = plugin_dir / "plugin.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)

    def _write_mcp_json(self, output_dir: Path) -> None:
        from mirdan.integrations.claude_code import detect_mirdan_command

        command, args = detect_mirdan_command()
        mcp_config = {
            "mcpServers": {
                "mirdan": {
                    "type": "stdio",
                    "command": command,
                    "args": args,
                }
            }
        }
        with (output_dir / "mcp.json").open("w") as f:
            json.dump(mcp_config, f, indent=2)

    def _write_rules(self, output_dir: Path, detected: DetectedProject) -> None:
        from mirdan.integrations.cursor import generate_cursor_rules

        rules_dir = output_dir / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        generate_cursor_rules(rules_dir, detected)

    def _write_agents(self, output_dir: Path, detected: DetectedProject) -> None:
        from mirdan.integrations.cursor import generate_cursor_agents

        agents_dir = output_dir
        generate_cursor_agents(agents_dir, detected)

    def _write_hooks(self, output_dir: Path) -> None:
        from mirdan.integrations.cursor import (
            CursorHookStringency,
            generate_cursor_hooks,
        )

        generate_cursor_hooks(output_dir, CursorHookStringency.COMPREHENSIVE)

    def _write_sandbox(self, output_dir: Path, detected: DetectedProject) -> None:
        from mirdan.integrations.cursor import generate_cursor_sandbox

        generate_cursor_sandbox(output_dir, detected)

    def _write_commands(self, output_dir: Path) -> None:
        from mirdan.integrations.cursor import generate_cursor_commands

        generate_cursor_commands(output_dir)

    def _write_subagents(self, output_dir: Path) -> None:
        from mirdan.integrations.cursor import generate_cursor_subagents

        generate_cursor_subagents(output_dir)

    def _write_skills(self, output_dir: Path) -> None:
        from mirdan.integrations.cursor import generate_cursor_skills

        generate_cursor_skills(output_dir)

    def _write_environment(self, output_dir: Path, detected: DetectedProject) -> None:
        from mirdan.integrations.cursor import generate_cursor_environment

        generate_cursor_environment(output_dir, detected)
