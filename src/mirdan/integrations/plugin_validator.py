"""Validate mirdan plugin directory structure.

Checks that a plugin export has all required files, valid manifest,
and correct MCP tool name references.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Expected MCP tool name prefix (double underscore)
_MCP_PREFIX = "mcp__mirdan__"
# Invalid single-underscore prefix
_BAD_PREFIX = "mcp_mirdan_"


@dataclass
class PluginValidationResult:
    """Result of plugin directory validation."""

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files_found: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "files_found": self.files_found,
        }


class PluginValidator:
    """Validates a mirdan plugin directory structure."""

    def validate(self, plugin_dir: Path) -> PluginValidationResult:
        """Validate a plugin directory.

        Checks:
        - Plugin manifest exists and is valid JSON
        - Required fields in manifest
        - MCP config exists
        - Referenced skills/agents exist
        - No incorrect MCP tool name prefixes

        Args:
            plugin_dir: Path to the plugin directory.

        Returns:
            PluginValidationResult with errors and warnings.
        """
        result = PluginValidationResult()

        if not plugin_dir.is_dir():
            result.valid = False
            result.errors.append(f"Not a directory: {plugin_dir}")
            return result

        self._check_manifest(plugin_dir, result)
        self._check_mcp_config(plugin_dir, result)
        self._check_skills(plugin_dir, result)
        self._check_agents(plugin_dir, result)
        self._check_mcp_tool_names(plugin_dir, result)
        self._check_hooks(plugin_dir, result)

        return result

    def _check_manifest(self, plugin_dir: Path, result: PluginValidationResult) -> None:
        manifest = plugin_dir / ".claude-plugin" / "plugin.json"
        if not manifest.exists():
            result.valid = False
            result.errors.append("Missing .claude-plugin/plugin.json")
            return

        result.files_found.append(".claude-plugin/plugin.json")

        try:
            data = json.loads(manifest.read_text())
        except json.JSONDecodeError as e:
            result.valid = False
            result.errors.append(f"Invalid JSON in plugin.json: {e}")
            return

        required = ["name", "version", "description"]
        for field_name in required:
            if field_name not in data:
                result.valid = False
                result.errors.append(f"Missing required field '{field_name}' in plugin.json")

    def _check_mcp_config(self, plugin_dir: Path, result: PluginValidationResult) -> None:
        mcp_json = plugin_dir / ".mcp.json"
        if not mcp_json.exists():
            result.warnings.append("Missing .mcp.json (MCP config)")
            return

        result.files_found.append(".mcp.json")

        try:
            data = json.loads(mcp_json.read_text())
        except json.JSONDecodeError:
            result.errors.append("Invalid JSON in .mcp.json")
            result.valid = False
            return

        if "mcpServers" not in data:
            result.warnings.append("No mcpServers in .mcp.json")
        elif "mirdan" not in data.get("mcpServers", {}):
            result.warnings.append("'mirdan' not in mcpServers")

    def _check_skills(self, plugin_dir: Path, result: PluginValidationResult) -> None:
        skills_dir = plugin_dir / "skills"
        if not skills_dir.exists():
            result.warnings.append("No skills/ directory")
            return

        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                rel = f"skills/{skill_dir.name}/SKILL.md"
                result.files_found.append(rel)
            else:
                result.warnings.append(f"Skill '{skill_dir.name}' missing SKILL.md")

    def _check_agents(self, plugin_dir: Path, result: PluginValidationResult) -> None:
        agents_dir = plugin_dir / "agents"
        if not agents_dir.exists():
            result.warnings.append("No agents/ directory")
            return

        for agent_file in sorted(agents_dir.glob("*.md")):
            result.files_found.append(f"agents/{agent_file.name}")

    def _check_mcp_tool_names(self, plugin_dir: Path, result: PluginValidationResult) -> None:
        """Check for incorrect MCP tool name prefixes."""
        for md_file in plugin_dir.rglob("*.md"):
            try:
                content = md_file.read_text()
            except OSError:
                continue

            if _BAD_PREFIX in content:
                rel = md_file.relative_to(plugin_dir)
                result.valid = False
                result.errors.append(
                    f"Incorrect MCP prefix '{_BAD_PREFIX}' in {rel} (should be '{_MCP_PREFIX}')"
                )

    def _check_hooks(self, plugin_dir: Path, result: PluginValidationResult) -> None:
        hooks_json = plugin_dir / "hooks.json"
        if hooks_json.exists():
            result.files_found.append("hooks.json")
            try:
                json.loads(hooks_json.read_text())
            except json.JSONDecodeError:
                result.errors.append("Invalid JSON in hooks.json")
                result.valid = False
