"""Data models for Mirdan."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskType(Enum):
    """Classification of developer task types."""

    GENERATION = "generation"
    REFACTOR = "refactor"
    DEBUG = "debug"
    REVIEW = "review"
    DOCUMENTATION = "documentation"
    TEST = "test"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    """Analyzed intent from a developer prompt."""

    original_prompt: str
    task_type: TaskType
    primary_language: str | None = None
    frameworks: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    touches_security: bool = False
    uses_external_framework: bool = False
    ambiguity_score: float = 0.0  # 0 = clear, 1 = very ambiguous


@dataclass
class ContextBundle:
    """Gathered context from various sources."""

    tech_stack: dict[str, Any] = field(default_factory=dict)
    existing_patterns: list[str] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)
    documentation_hints: list[str] = field(default_factory=list)

    def summarize_patterns(self) -> str:
        """Summarize detected patterns."""
        if not self.existing_patterns:
            return "No existing patterns detected for this task type."
        return "\n".join(f"- {p}" for p in self.existing_patterns[:5])


@dataclass
class ToolRecommendation:
    """A recommendation to use a specific MCP tool."""

    mcp: str
    action: str
    priority: str = "medium"
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mcp": self.mcp,
            "action": self.action,
            "priority": self.priority,
            "reason": self.reason,
        }


@dataclass
class MCPToolInfo:
    """Information about an MCP tool capability."""

    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None


@dataclass
class MCPResourceInfo:
    """Information about an MCP resource capability."""

    uri: str
    name: str | None = None
    description: str | None = None
    mime_type: str | None = None


@dataclass
class MCPResourceTemplateInfo:
    """Information about an MCP resource template capability."""

    uri_template: str
    name: str | None = None
    description: str | None = None


@dataclass
class MCPPromptInfo:
    """Information about an MCP prompt capability."""

    name: str
    description: str | None = None


@dataclass
class MCPCapabilities:
    """Discovered capabilities of an MCP server."""

    tools: list[MCPToolInfo] = field(default_factory=list)
    resources: list[MCPResourceInfo] = field(default_factory=list)
    resource_templates: list[MCPResourceTemplateInfo] = field(default_factory=list)
    prompts: list[MCPPromptInfo] = field(default_factory=list)
    discovered_at: str | None = None  # ISO timestamp

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool with the given name exists."""
        return any(t.name == tool_name for t in self.tools)

    def get_tool(self, tool_name: str) -> MCPToolInfo | None:
        """Get tool info by name."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None


@dataclass
class EnhancedPrompt:
    """The final enhanced prompt output."""

    enhanced_text: str
    intent: Intent
    tool_recommendations: list[ToolRecommendation]
    quality_requirements: list[str]
    verification_steps: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "enhanced_prompt": self.enhanced_text,
            "detected_task_type": self.intent.task_type.value,
            "detected_language": self.intent.primary_language,
            "detected_frameworks": self.intent.frameworks,
            "touches_security": self.intent.touches_security,
            "ambiguity_score": self.intent.ambiguity_score,
            "quality_requirements": self.quality_requirements,
            "verification_steps": self.verification_steps,
            "tool_recommendations": [r.to_dict() for r in self.tool_recommendations],
        }
