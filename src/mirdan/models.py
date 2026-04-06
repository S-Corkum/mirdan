"""Data models for Mirdan."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import Any


class OutputFormat(Enum):
    """Output compression format levels."""

    FULL = "full"
    COMPACT = "compact"
    MINIMAL = "minimal"
    MICRO = "micro"


class ModelTier(Enum):
    """Model tier for output optimization."""

    AUTO = "auto"
    OPUS = "opus"
    SONNET = "sonnet"
    HAIKU = "haiku"


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


class CeremonyLevel(IntEnum):
    """Guidance depth level for enhance_prompt. Higher = more ceremony.

    MICRO: Trivial changes (typo fix, version bump). Returns intent analysis only.
    LIGHT: Simple single-file tasks. Critical tool recs only, minimal context.
    STANDARD: Normal development tasks. Full quality sandwich.
    THOROUGH: Complex multi-framework or security-sensitive tasks. Deep analysis.
    """

    MICRO = 0
    LIGHT = 1
    STANDARD = 2
    THOROUGH = 3


@dataclass(frozen=True)
class CeremonyPolicy:
    """Frozen policy mapping for a ceremony level.

    Maps each CeremonyLevel to concrete parameter values that control
    enhance_prompt behavior. Immutable to prevent runtime mutation.
    """

    level: CeremonyLevel
    enhancement_mode: str  # "analyze_only" | "auto" | "auto" | "auto"
    context_level: str  # "none" | "minimal" | "auto" | "comprehensive"
    recommended_validation: str  # "skip" | "quick" | "standard" | "full"
    filter_tool_recs: bool  # True for LIGHT (critical-only), False otherwise


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
    framework_versions: dict[str, str] = field(default_factory=dict)
    entities: list[ExtractedEntity] = field(default_factory=list)
    touches_security: bool = False
    touches_rag: bool = False
    touches_knowledge_graph: bool = False
    uses_external_framework: bool = False
    ambiguity_score: float = 0.0  # 0 = clear, 1 = very ambiguous
    clarifying_questions: list[str] = field(default_factory=list)
    task_types: list[TaskType] = field(default_factory=list)  # All detected types, primary first


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
class SessionContext:
    """Session state for multi-turn quality orchestration."""

    session_id: str
    task_type: TaskType = TaskType.UNKNOWN
    detected_language: str | None = None
    frameworks: list[str] = field(default_factory=list)
    touches_security: bool = False
    touches_rag: bool = False
    touches_knowledge_graph: bool = False
    created_at: float = 0.0
    last_accessed: float = 0.0

    # Session-wide quality tracking
    validation_count: int = 0
    cumulative_score: float = 0.0
    unresolved_errors: int = 0
    files_validated: list[str] = field(default_factory=list)
    last_validated_at: float = 0.0

    # Tool recommendations from enhance_prompt (for compliance tracking)
    tool_recommendations: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        result: dict[str, Any] = {
            "session_id": self.session_id,
            "task_type": self.task_type.value,
            "detected_language": self.detected_language,
            "frameworks": self.frameworks,
            "touches_security": self.touches_security,
        }
        if self.validation_count > 0:
            result["session_quality"] = {
                "validation_count": self.validation_count,
                "avg_score": round(self.cumulative_score / self.validation_count, 3),
                "unresolved_errors": self.unresolved_errors,
                "files_validated": len(self.files_validated),
            }
        return result


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
        return asdict(self)


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
            "task_types": (
                [t.value for t in self.intent.task_types]
                if self.intent.task_types
                else [self.intent.task_type.value]
            ),
            "language": self.intent.primary_language,
            "frameworks": self.intent.frameworks,
            "extracted_entities": [e.to_dict() for e in self.intent.entities],
            "touches_security": self.intent.touches_security,
            "touches_rag": self.intent.touches_rag,
            "touches_knowledge_graph": self.intent.touches_knowledge_graph,
            "ambiguity_score": self.intent.ambiguity_score,
            "clarifying_questions": self.intent.clarifying_questions,
            "quality_requirements": self.quality_requirements,
            "verification_steps": self.verification_steps,
            "tool_recommendations": [r.to_dict() for r in self.tool_recommendations],
        }


@dataclass
class ComparisonEntry:
    """A single implementation's validation result within a comparison."""

    label: str
    score: float
    passed: bool
    violation_counts: dict[str, int] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ComparisonResult:
    """Side-by-side comparison of multiple implementations."""

    entries: list[ComparisonEntry] = field(default_factory=list)
    winner: str = ""
    language_detected: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "entries": [e.to_dict() for e in self.entries],
            "winner": self.winner,
            "language_detected": self.language_detected,
            "count": len(self.entries),
        }


@dataclass
class QualitySnapshot:
    """A point-in-time quality measurement."""

    timestamp: str  # ISO 8601
    project_path: str
    language: str
    score: float
    passed: bool
    violation_counts: dict[str, int] = field(default_factory=dict)
    standards_checked: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class QualityTrend:
    """Quality trend data over a time period."""

    project_path: str
    days: int
    snapshot_count: int
    avg_score: float
    min_score: float
    max_score: float
    pass_rate: float  # Fraction of snapshots that passed
    score_trend: str  # "improving", "declining", "stable"
    snapshots: list[QualitySnapshot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "project_path": self.project_path,
            "days": self.days,
            "snapshot_count": self.snapshot_count,
            "avg_score": round(self.avg_score, 3),
            "min_score": round(self.min_score, 3),
            "max_score": round(self.max_score, 3),
            "pass_rate": round(self.pass_rate, 3),
            "score_trend": self.score_trend,
            "snapshots": [s.to_dict() for s in self.snapshots],
        }


@dataclass
class KnowledgeEntry:
    """A storable insight extracted from validation, ready for enyal_remember.

    Maps directly to enyal_remember parameters for frictionless storage
    by the AI agent.
    """

    content: str  # What to store
    content_type: str  # fact|convention|pattern|decision
    tags: list[str] = field(default_factory=list)
    scope: str = "project"  # global|project|file
    scope_path: str = ""  # For file-scoped entries
    confidence: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching enyal_remember parameters."""
        result: dict[str, Any] = {
            "content": self.content,
            "content_type": self.content_type,
            "tags": self.tags,
            "scope": self.scope,
            "confidence": self.confidence,
        }
        if self.scope_path:
            result["scope_path"] = self.scope_path
        return result


@dataclass
class CompactState:
    """Minimal state for compaction resilience.

    When the context window is compacted, this captures the essential
    quality state so mirdan can resume without full context. Used by
    the self-managing integration's PreCompact hook.
    """

    session_id: str = ""
    task_type: str = ""
    language: str = ""
    touches_security: bool = False
    last_score: float | None = None
    open_violations: int = 0
    frameworks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompactState:
        """Create from dictionary."""
        return cls(
            session_id=data.get("session_id", ""),
            task_type=data.get("task_type", ""),
            language=data.get("language", ""),
            touches_security=data.get("touches_security", False),
            last_score=data.get("last_score"),
            open_violations=data.get("open_violations", 0),
            frameworks=data.get("frameworks", []),
        )


# --- LLM Intelligence Layer models ---


class ModelRole(Enum):
    """Role a local LLM model fills."""

    FAST = "fast"
    BRAIN = "brain"


class HardwareProfile(Enum):
    """Hardware capability tier."""

    STANDARD = "standard"  # 16GB — most developers
    ENHANCED = "enhanced"  # 32GB
    FULL = "full"  # 64GB+ Apple Silicon


class HealthState(Enum):
    """LLM subsystem health state."""

    STARTING = "starting"
    WARMING_UP = "warming_up"
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class TaskClassification(Enum):
    """Triage classification for a coding task."""

    LOCAL_ONLY = "local_only"
    LOCAL_ASSIST = "local_assist"
    PAID_MINIMAL = "paid_minimal"
    PAID_REQUIRED = "paid_required"


@dataclass
class ModelInfo:
    """Information about a locally available LLM."""

    name: str
    role: ModelRole
    active_memory_mb: int
    quality_score: float
    model_family: str = "unknown"
    supports_tools: bool = False
    supports_structured_output: bool = False
    supports_thinking: bool = False
    loaded: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "name": self.name,
            "role": self.role.value,
            "active_memory_mb": self.active_memory_mb,
            "quality_score": self.quality_score,
            "model_family": self.model_family,
            "loaded": self.loaded,
        }


@dataclass
class LLMResponse:
    """Response from a local LLM call."""

    content: str
    model: str
    role: ModelRole
    elapsed_ms: float
    tokens_used: int
    structured_data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        result: dict[str, Any] = {
            "model": self.model,
            "role": self.role.value,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "tokens_used": self.tokens_used,
        }
        if self.structured_data:
            result["structured_data"] = self.structured_data
        return result


@dataclass
class HardwareInfo:
    """Detected hardware capabilities."""

    architecture: str
    total_ram_mb: int
    available_ram_mb: int
    gpu_type: str | None = None
    metal_capable: bool = False
    detected_profile: HardwareProfile = HardwareProfile.STANDARD

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "architecture": self.architecture,
            "total_ram_mb": self.total_ram_mb,
            "available_ram_mb": self.available_ram_mb,
            "gpu_type": self.gpu_type,
            "metal_capable": self.metal_capable,
            "profile": self.detected_profile.value,
        }


@dataclass
class LLMHealth:
    """Health status of the LLM subsystem."""

    state: HealthState
    models_loaded: list[str] = field(default_factory=list)
    hardware_profile: str = "unknown"
    detected_ide: str = "unknown"
    effective_timeout: float = 20.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        result: dict[str, Any] = {
            "state": self.state.value,
            "models_loaded": self.models_loaded,
            "hardware_profile": self.hardware_profile,
            "detected_ide": self.detected_ide,
            "effective_timeout": self.effective_timeout,
        }
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class TriageResult:
    """Result of task triage classification."""

    classification: TaskClassification
    confidence: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "classification": self.classification.value,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
        }


@dataclass
class SubprocessResult:
    """Output from a subprocess execution."""

    command: str
    returncode: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass
class CheckResult:
    """Results from lint + typecheck + test run."""

    lint: SubprocessResult
    typecheck: SubprocessResult
    test: SubprocessResult
    all_pass: bool
    auto_fixed: list[str] = field(default_factory=list)
    needs_attention: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "lint": self.lint.to_dict(),
            "typecheck": self.typecheck.to_dict(),
            "test": self.test.to_dict(),
            "all_pass": self.all_pass,
            "auto_fixed": self.auto_fixed,
            "needs_attention": self.needs_attention,
            "summary": self.summary,
        }


@dataclass
class SmartValidationResult:
    """LLM-enriched validation analysis."""

    per_violation: list[dict[str, Any]]  # violation_id, assessment, FP reason, fix
    root_causes: list[dict[str, Any]]  # cause, violation_ids, unified_fix
    was_sanity_capped: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "per_violation": self.per_violation,
            "root_causes": self.root_causes,
            "was_sanity_capped": self.was_sanity_capped,
        }


@dataclass
class OptimizedPrompt:
    """Result from prompt optimization."""

    text: str
    context_pruned: int
    target_model: str
    optimization_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "text": self.text,
            "context_pruned": self.context_pruned,
            "target_model": self.target_model,
            "optimization_notes": self.optimization_notes,
        }


@dataclass
class ResearchResult:
    """Result from research agent."""

    synthesis: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_made: int = 0
    tokens_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "synthesis": self.synthesis,
            "sources": self.sources,
            "tool_calls_made": self.tool_calls_made,
            "tokens_used": self.tokens_used,
        }


@dataclass
class FileClaim:
    """A file ownership claim by an agent session."""

    session_id: str
    file_path: str
    claim_type: str  # "read" | "write"
    timestamp: float
    agent_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "file_path": self.file_path,
            "claim_type": self.claim_type,
            "agent_label": self.agent_label,
        }


@dataclass
class ConflictWarning:
    """A warning about potential multi-agent conflicts."""

    type: str  # "write_overlap" | "stale_read"
    message: str
    conflicting_sessions: list[str]
    file_path: str
    severity: str = "warning"  # "warning" | "info"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "message": self.message,
            "conflicting_sessions": self.conflicting_sessions,
            "file_path": self.file_path,
            "severity": self.severity,
        }


@dataclass
class TidySuggestion:
    """A preparatory refactoring suggestion (Tidy First pattern)."""

    type: str  # "extract_method" | "simplify_conditional" | "reduce_nesting" | "split_file"
    file_path: str
    line: int | None = None
    description: str = ""
    effort: str = "small"  # "trivial" | "small" | "medium"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "file_path": self.file_path,
            "description": self.description,
            "effort": self.effort,
        }
        if self.line is not None:
            result["line"] = self.line
        if self.reason:
            result["reason"] = self.reason
        return result


@dataclass
class TidyFirstAnalysis:
    """Result of Tidy First analysis on target files."""

    suggestions: list[TidySuggestion] = field(default_factory=list)
    target_files: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggestions": [s.to_dict() for s in self.suggestions],
            "target_files": self.target_files,
            "skipped_files": self.skipped_files,
        }


@dataclass
class DecisionApproach:
    """A single approach option within a decision domain."""

    name: str
    when_best: str
    when_avoid: str
    complexity: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "when_best": self.when_best,
            "when_avoid": self.when_avoid,
            "complexity": self.complexity,
        }


@dataclass
class DecisionGuidance:
    """Trade-off analysis for a decision domain."""

    domain: str
    approaches: list[DecisionApproach] = field(default_factory=list)
    senior_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "approaches": [a.to_dict() for a in self.approaches],
            "senior_questions": self.senior_questions,
        }


@dataclass
class GuardrailAnalysis:
    """Domain-aware pre-flight guardrails."""

    domain: str
    guardrails: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "guardrails": self.guardrails,
        }


@dataclass
class ConfidenceAssessment:
    """Calibrated confidence level for validation results."""

    level: str  # "high" | "medium" | "low"
    reason: str
    attention_focus: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "reason": self.reason,
            "attention_focus": self.attention_focus,
        }


@dataclass
class ArchLayer:
    """A single architectural layer with import constraints."""

    name: str
    patterns: list[str] = field(default_factory=list)
    allowed_imports: list[str] = field(default_factory=list)
    forbidden_imports: list[str] = field(default_factory=list)


@dataclass
class ArchDriftResult:
    """Result of architecture drift detection."""

    violations: list[Violation] = field(default_factory=list)
    file_layer: str = ""
    context_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "violations": [v.to_dict() for v in self.violations],
            "file_layer": self.file_layer,
            "context_warnings": self.context_warnings,
        }


@dataclass
class SemanticCheck:
    """A semantic review question for the calling LLM to investigate."""

    concern: str  # e.g. sql, auth, crypto, file_io, network
    question: str  # Specific, actionable question with line numbers
    severity: str  # critical, warning, info
    related_violation: str = ""  # Rule ID that triggered this (e.g., "SEC004")
    focus_lines: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "concern": self.concern,
            "question": self.question,
            "severity": self.severity,
        }
        if self.related_violation:
            result["related_violation"] = self.related_violation
        if self.focus_lines:
            result["focus_lines"] = self.focus_lines
        return result


@dataclass
class AnalysisProtocol:
    """Structured protocol for the LLM to self-execute deep analysis."""

    type: str  # security_flow_analysis, auth_completeness, data_handling
    focus_areas: list[dict[str, Any]] = field(default_factory=list)
    response_format: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PackageInfo:
    """A dependency package parsed from a manifest."""

    name: str
    version: str
    ecosystem: str  # PyPI, npm, crates.io, Go, Maven
    source: str  # File it was parsed from
    is_dev: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "ecosystem": self.ecosystem,
            "source": self.source,
            "is_dev": self.is_dev,
        }


@dataclass
class VulnFinding:
    """A vulnerability found in a dependency."""

    package: str
    version: str
    ecosystem: str
    vuln_id: str  # CVE or OSV ID
    severity: str  # critical, high, medium, low
    summary: str
    fixed_version: str = ""
    advisory_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "package": self.package,
            "version": self.version,
            "ecosystem": self.ecosystem,
            "vuln_id": self.vuln_id,
            "severity": self.severity,
            "summary": self.summary,
        }
        if self.fixed_version:
            result["fixed_version"] = self.fixed_version
        if self.advisory_url:
            result["advisory_url"] = self.advisory_url
        return result


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
    fix_code: str = ""
    fix_description: str = ""
    explanation: str = ""
    related_violations: list[str] = field(default_factory=list)
    historical_frequency: int = 0
    verifiable: bool = True  # False for pattern-based heuristics (AI001-AI008)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
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
        if self.fix_code:
            result["fix_code"] = self.fix_code
            result["fix_description"] = self.fix_description
        if self.explanation:
            result["explanation"] = self.explanation
        if self.related_violations:
            result["related_violations"] = self.related_violations
        if self.historical_frequency > 0:
            result["historical_frequency"] = self.historical_frequency
        if not self.verifiable:
            result["verifiable"] = False
        return result


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
