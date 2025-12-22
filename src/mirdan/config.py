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


class PlanningConfig(BaseModel):
    """Planning behavior configuration for cheap model handoff."""

    # Target model that will implement the plan
    target_model: str = Field(
        default="haiku",
        pattern="^(haiku|flash|cheap|capable)$",
        description="Model that will implement the plan",
    )

    # Quality thresholds (stricter for cheaper models)
    min_grounding_score: float = Field(default=0.9)
    min_completeness_score: float = Field(default=0.9)
    min_clarity_score: float = Field(default=0.95)

    # Required sections
    require_research_notes: bool = True
    require_step_grounding: bool = True
    require_verification_per_step: bool = True

    # Anti-slop enforcement
    reject_vague_language: bool = True
    max_words_per_step_detail: int = Field(default=100, description="Force atomic steps")


class ThresholdsConfig(BaseModel):
    """Centralized threshold values for various components.

    This consolidates magic numbers that were previously scattered
    throughout the codebase.
    """

    # Entity extraction thresholds
    entity_base_confidence: float = Field(default=0.7, description="Base confidence for entities")
    entity_confidence_boost: float = Field(
        default=0.15, description="Confidence boost for high-value patterns"
    )

    # Context7 gatherer
    max_doc_length: int = Field(default=2000, description="Max chars for documentation excerpts")

    # Code validator severity weights
    severity_error_weight: float = Field(default=0.25, description="Score penalty per error")
    severity_warning_weight: float = Field(default=0.08, description="Score penalty per warning")
    severity_info_weight: float = Field(default=0.02, description="Score penalty per info")

    # Language detection confidence thresholds
    lang_high_confidence_score: int = Field(default=8, description="Min score for high confidence")
    lang_high_confidence_margin: int = Field(
        default=3, description="Min margin for high confidence"
    )
    lang_medium_confidence_score: int = Field(
        default=4, description="Min score for medium confidence"
    )

    # Plan validator penalties
    plan_clarity_penalty: float = Field(
        default=0.1, description="Penalty per vague language instance"
    )
    plan_completeness_penalty: float = Field(
        default=0.25, description="Penalty per missing section"
    )
    plan_grounding_penalty: float = Field(default=0.1, description="Penalty per step issue")


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
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    rules: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Path) -> "MirdanConfig":
        """Load configuration from a YAML file."""
        if not config_path.exists():
            return cls()

        with config_path.open() as f:
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
        with config_path.open("w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


def get_default_config() -> MirdanConfig:
    """Get the default configuration."""
    return MirdanConfig()
