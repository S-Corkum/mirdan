"""Configuration system for Mirdan."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class QualityConfig(BaseModel):
    """Quality enforcement configuration."""

    security: str = Field(default="strict", pattern="^(strict|moderate|permissive)$")
    architecture: str = Field(default="moderate", pattern="^(strict|moderate|permissive)$")
    documentation: str = Field(default="moderate", pattern="^(strict|moderate|permissive)$")
    testing: str = Field(default="strict", pattern="^(strict|moderate|permissive)$")
    framework: str = Field(default="moderate", pattern="^(strict|moderate|permissive)$")


class MCPClientConfig(BaseModel):
    """Configuration for connecting to an external MCP server."""

    type: str = Field(description="Transport type: 'stdio' or 'http'")
    command: str | None = Field(default=None, description="Command for stdio transport")
    args: list[str] = Field(default_factory=list, description="Arguments for stdio command")
    url: str | None = Field(default=None, description="URL for http transport")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    cwd: str | None = Field(default=None, description="Working directory for stdio")
    timeout: float = Field(default=30.0, description="Connection timeout in seconds")


class OrchestrationConfig(BaseModel):
    """MCP orchestration preferences."""

    prefer_mcps: list[str] = Field(default_factory=lambda: ["context7", "filesystem"])
    auto_invoke: list[dict[str, Any]] = Field(default_factory=list)
    mcp_clients: dict[str, MCPClientConfig] = Field(
        default_factory=dict,
        description="Configuration for MCP clients mirdan can connect to",
    )
    gather_timeout: float = Field(default=10.0, description="Total timeout for context gathering")
    gatherer_timeout: float = Field(default=3.0, description="Timeout per gatherer")


class EnhancementConfig(BaseModel):
    """Enhancement behavior configuration."""

    mode: str = Field(default="auto", pattern="^(auto|confirm|manual)$")
    verbosity: str = Field(default="balanced", pattern="^(minimal|balanced|comprehensive)$")
    include_verification: bool = True
    include_tool_hints: bool = True


class ProjectConfig(BaseModel):
    """Project-level configuration."""

    name: str = ""
    type: str = "application"
    primary_language: str = ""
    frameworks: list[str] = Field(default_factory=list)


class MirdanConfig(BaseModel):
    """Main Mirdan configuration."""

    version: str = "1.0"
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    enhancement: EnhancementConfig = Field(default_factory=EnhancementConfig)
    rules: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Path) -> "MirdanConfig":
        """Load configuration from a YAML file."""
        if not config_path.exists():
            return cls()

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    @classmethod
    def find_config(cls, start_path: Path | None = None) -> "MirdanConfig":
        """Find and load configuration, searching up the directory tree."""
        if start_path is None:
            start_path = Path.cwd()

        current = start_path
        while current != current.parent:
            config_file = current / ".mirdan" / "config.yaml"
            if config_file.exists():
                return cls.load(config_file)

            # Also check for config.yaml directly
            config_file = current / ".mirdan.yaml"
            if config_file.exists():
                return cls.load(config_file)

            current = current.parent

        return cls()

    def save(self, config_path: Path) -> None:
        """Save configuration to a YAML file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


def get_default_config() -> MirdanConfig:
    """Get the default configuration."""
    return MirdanConfig()
