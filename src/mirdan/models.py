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
    PLANNING = "planning"
    UNKNOWN = "unknown"


class EntityType(Enum):
    """Types of entities that can be extracted from prompts."""

    FILE_PATH = "file_path"
    FUNCTION_NAME = "function_name"
    API_REFERENCE = "api_reference"


@dataclass
class Intent:
    """Analyzed intent from a developer prompt."""

    original_prompt: str
    task_type: TaskType
    primary_language: str | None = None
    frameworks: list[str] = field(default_factory=list)
    entities: list["ExtractedEntity"] = field(default_factory=list)
    touches_security: bool = False
    touches_rag: bool = False
    uses_external_framework: bool = False
    ambiguity_score: float = 0.0  # 0 = clear, 1 = very ambiguous
    clarifying_questions: list[str] = field(default_factory=list)


@dataclass
class ExtractedEntity:
    """An entity extracted from a developer prompt."""

    type: EntityType
    value: str
    raw_match: str = ""
    context: str = ""  # Surrounding text for disambiguation
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "type": self.type.value,
            "value": self.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class PlanQualityScore:
    """Quality assessment of a plan for cheap model implementation."""

    overall_score: float  # 0.0-1.0
    grounding_score: float  # Are all facts tool-verified?
    completeness_score: float  # Are there gaps?
    atomicity_score: float  # Is each step single-action?
    clarity_score: float  # Is language unambiguous?
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    ready_for_cheap_model: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "overall_score": self.overall_score,
            "grounding_score": self.grounding_score,
            "completeness_score": self.completeness_score,
            "atomicity_score": self.atomicity_score,
            "clarity_score": self.clarity_score,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "ready_for_cheap_model": self.ready_for_cheap_model,
        }


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
class MCPToolCall:
    """Request to call a tool on an MCP server."""

    mcp_name: str  # e.g., "context7", "enyal"
    tool_name: str  # e.g., "resolve-library-id"
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPToolResult:
    """Result from an MCP tool call."""

    mcp_name: str
    tool_name: str
    success: bool
    data: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0


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
            "task_type": self.intent.task_type.value,
            "language": self.intent.primary_language,
            "frameworks": self.intent.frameworks,
            "extracted_entities": [e.to_dict() for e in self.intent.entities],
            "touches_security": self.intent.touches_security,
            "touches_rag": self.intent.touches_rag,
            "ambiguity_score": self.intent.ambiguity_score,
            "clarifying_questions": self.intent.clarifying_questions,
            "quality_requirements": self.quality_requirements,
            "verification_steps": self.verification_steps,
            "tool_recommendations": [r.to_dict() for r in self.tool_recommendations],
        }


@dataclass
class Violation:
    """A code quality violation detected during validation."""

    id: str  # e.g., "PY001"
    rule: str  # e.g., "no-bare-except"
    category: str  # "security" | "architecture" | "style"
    severity: str  # "error" | "warning" | "info"
    message: str  # Human-readable description
    line: int | None = None
    column: int | None = None
    code_snippet: str = ""
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "rule": self.rule,
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "code_snippet": self.code_snippet,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of code quality validation."""

    passed: bool
    score: float  # 0.0-1.0 quality score
    language_detected: str
    violations: list[Violation] = field(default_factory=list)
    standards_checked: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self, severity_threshold: str = "warning") -> dict[str, Any]:
        """Convert to API response format with severity filtering."""
        threshold_order = ["error", "warning", "info"]
        try:
            threshold_idx = threshold_order.index(severity_threshold)
        except ValueError:
            threshold_idx = 1  # Default to warning

        filtered = [
            v for v in self.violations if threshold_order.index(v.severity) <= threshold_idx
        ]

        return {
            "passed": self.passed,
            "score": self.score,
            "language_detected": self.language_detected,
            "violations_count": {
                "error": sum(1 for v in filtered if v.severity == "error"),
                "warning": sum(1 for v in filtered if v.severity == "warning"),
                "info": sum(1 for v in filtered if v.severity == "info"),
            },
            "violations": [v.to_dict() for v in filtered],
            "summary": self._generate_summary(filtered),
            "standards_checked": self.standards_checked,
            "limitations": self.limitations,
        }

    def _generate_summary(self, violations: list[Violation]) -> str:
        """Generate human-readable summary."""
        if self.passed and not violations:
            return f"Code passes all {', '.join(self.standards_checked)} checks"

        if self.passed:
            warning_count = sum(1 for v in violations if v.severity == "warning")
            info_count = sum(1 for v in violations if v.severity == "info")
            parts = []
            if warning_count:
                parts.append(f"{warning_count} warning(s)")
            if info_count:
                parts.append(f"{info_count} info notice(s)")
            return f"Code passes with {' and '.join(parts)}"

        error_count = sum(1 for v in violations if v.severity == "error")
        return f"Code has {error_count} error(s) that should be fixed"
