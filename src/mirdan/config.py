"""Configuration system for Mirdan."""

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field, field_validator


class QualityConfig(BaseModel):
    """Quality enforcement configuration."""

    security: str = Field(default="strict", pattern="^(strict|moderate|permissive)$")
    architecture: str = Field(default="moderate", pattern="^(strict|moderate|permissive)$")
    documentation: str = Field(default="moderate", pattern="^(strict|moderate|permissive)$")
    testing: str = Field(default="strict", pattern="^(strict|moderate|permissive)$")
    framework: str = Field(default="moderate", pattern="^(strict|moderate|permissive)$")
    language: str = Field(default="moderate", pattern="^(strict|moderate|permissive)$")
    custom_rules_dir: str = Field(
        default=".mirdan/rules", description="Directory for custom validation rules"
    )


class SemanticConfig(BaseModel):
    """Semantic validation configuration."""

    enabled: bool = Field(default=True, description="Enable semantic analysis")
    analysis_protocol: str = Field(
        default="security",
        pattern="^(none|security|comprehensive)$",
        description="When to generate structured analysis protocols",
    )
    deep_analysis: bool = Field(
        default=True,
        description="Enable deep analysis patterns "
        "(concurrency, boundary, error_propagation, state_machine)",
    )


class DependencyConfig(BaseModel):
    """Dependency vulnerability scanning configuration."""

    enabled: bool = Field(default=True, description="Enable dependency scanning")
    osv_cache_ttl: int = Field(default=86400, description="OSV cache TTL in seconds")
    scan_on_gate: bool = Field(default=True, description="Include dep scan in mirdan gate")
    fail_on_severity: str = Field(
        default="high",
        pattern="^(critical|high|medium|low|none)$",
        description="Minimum severity to fail quality gate",
    )


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
    auto_memory: bool = Field(
        default=False,
        description="Automatically store high-confidence knowledge entries via enyal",
    )
    auto_memory_threshold: float = Field(
        default=0.8,
        description="Minimum confidence for auto-storing knowledge entries (0.0-1.0)",
    )


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


class SessionConfig(BaseModel):
    """Session management configuration."""

    ttl_seconds: int = Field(default=1800, description="Session time-to-live in seconds")
    max_sessions: int = Field(default=100, description="Maximum concurrent sessions")


class TokenConfig(BaseModel):
    """Token budget and output formatting configuration."""

    default_max_tokens: int = Field(default=0, description="Default max tokens (0=unlimited)")
    compact_threshold: int = Field(
        default=4000, description="Token budget below which compact format is used"
    )
    minimal_threshold: int = Field(
        default=1000, description="Token budget below which minimal format is used"
    )
    micro_threshold: int = Field(
        default=200, description="Token budget below which micro format is used (hooks)"
    )


class FileThresholdOverride(BaseModel):
    """Threshold overrides for files matching a glob pattern.

    Uses fnmatch semantics: ``*`` matches everything (including ``/``),
    ``?`` matches single char, ``[seq]`` matches chars in seq.

    Examples:
        - ``tests/**`` — all test files
        - ``**/auth/**`` — auth directories at any depth
        - ``src/generated/*`` — generated files in src/generated/
    """

    pattern: str = Field(description="Glob pattern for file path matching")
    severity_error_weight: float | None = None
    severity_warning_weight: float | None = None
    severity_info_weight: float | None = None
    arch_max_function_length: int | None = None
    arch_max_file_length: int | None = None
    arch_max_nesting_depth: int | None = None
    arch_max_class_methods: int | None = None


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

    # Architecture validation thresholds
    arch_max_function_length: int = Field(default=30, description="Max lines per function")
    arch_max_file_length: int = Field(default=300, description="Max non-empty lines per file")
    arch_max_nesting_depth: int = Field(default=4, description="Max nesting depth")
    arch_max_class_methods: int = Field(default=10, description="Max methods per class")

    # Per-file threshold overrides (first matching pattern wins)
    file_overrides: list[FileThresholdOverride] = Field(
        default_factory=list,
        description="Per-file threshold overrides. First matching pattern wins.",
    )

    def resolve_for_file(self, file_path: str) -> "ThresholdsConfig":
        """Return a copy with the first matching file override applied.

        Uses fnmatch.fnmatch() for pattern matching. First match wins.
        Returns self if no overrides or no match.

        Args:
            file_path: Relative file path to match against patterns.

        Returns:
            ThresholdsConfig with override values applied (or self if no match).
        """
        import fnmatch

        for override in self.file_overrides:
            if fnmatch.fnmatch(file_path, override.pattern):
                overrides = {
                    k: v
                    for k, v in override.model_dump().items()
                    if k != "pattern" and v is not None
                }
                if overrides:
                    return self.model_copy(update=overrides)
                return self
        return self


class PlatformProfile(BaseModel):
    """Platform profile for IDE-specific configuration."""

    name: str = Field(default="generic", description="Platform name (cursor, claude-code, generic)")
    context_level: str = Field(
        default="auto",
        description="Context gathering level",
    )
    tool_budget_aware: bool = Field(
        default=False,
        description="Respect tool budget limits",
    )


class HookConfig(BaseModel):
    """Configuration for Claude Code hook generation."""

    enabled_events: list[str] = Field(
        default_factory=lambda: ["PreToolUse", "PostToolUse", "Stop"],
        description="Hook events to generate",
    )
    quick_validate_timeout: int = Field(
        default=5000, description="Timeout for quick validation hooks (ms)"
    )
    auto_fix_suggestions: bool = Field(
        default=True, description="Include auto-fix suggestion prompts in PostToolUse"
    )
    compaction_resilience: bool = Field(
        default=False, description="Enable PreCompact hook for state persistence"
    )
    multi_agent_awareness: bool = Field(
        default=False, description="Enable SubagentStart/SubagentStop hooks"
    )
    session_hooks: bool = Field(default=False, description="Enable SessionStart/SessionStop hooks")
    notification_hooks: bool = Field(
        default=False, description="Enable Notification hooks for quality alerts"
    )


class LinterOrcConfig(BaseModel):
    """Configuration for external linter orchestration."""

    enabled_linters: list[str] = Field(
        default_factory=list,
        description="Explicit list of linters to run (empty = auto-detect)",
    )
    ruff_args: list[str] = Field(default_factory=list, description="Extra args for ruff")
    eslint_args: list[str] = Field(default_factory=list, description="Extra args for ESLint")
    mypy_args: list[str] = Field(default_factory=list, description="Extra args for mypy")
    auto_detect: bool = Field(default=True, description="Auto-detect available linters")
    timeout: float = Field(default=30.0, description="Timeout per linter invocation (seconds)")


class SubProjectConfig(BaseModel):
    """Configuration for a sub-project within a workspace."""

    path: str = ""
    name: str = ""
    primary_language: str = ""
    frameworks: list[str] = Field(default_factory=list)


class WorkspaceConfig(BaseModel):
    """Workspace/monorepo configuration."""

    enabled: bool = False
    workspace_type: str = "auto"
    projects: list[SubProjectConfig] = Field(default_factory=list)


class ProjectConfig(BaseModel):
    """Project-level configuration."""

    name: str = ""
    type: str = "application"
    primary_language: str = ""
    frameworks: list[str] = Field(default_factory=list)
    github_owner: str = ""
    github_repo: str = ""


class CeremonyConfig(BaseModel):
    """Adaptive ceremony configuration.

    Controls how enhance_prompt scales guidance depth based on task complexity.
    """

    enabled: bool = Field(
        default=True,
        description="Master switch for adaptive ceremony. When False, always uses STANDARD.",
    )
    default_level: str = Field(
        default="auto",
        pattern="^(auto|micro|light|standard|thorough)$",
        description="Default ceremony level. 'auto' uses scoring algorithm.",
    )
    min_level: str = Field(
        default="micro",
        pattern="^(micro|light|standard|thorough)$",
        description="Floor level. Enterprise teams can set 'standard' to prevent MICRO/LIGHT.",
    )
    security_escalation: bool = Field(
        default=True,
        description="Always escalate security-touching tasks to at least STANDARD.",
    )
    ambiguity_escalation: bool = Field(
        default=True,
        description="Escalate high-ambiguity tasks to at least STANDARD.",
    )
    ambiguity_threshold: float = Field(
        default=0.6,
        description="Ambiguity score threshold for escalation (0.0-1.0).",
    )


class TidyFirstConfig(BaseModel):
    """Tidy First refactoring intelligence configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable Tidy First analysis in enhance_prompt.",
    )
    max_suggestions: int = Field(
        default=3,
        description="Maximum tidy suggestions per enhance_prompt call.",
    )
    max_file_size_kb: int = Field(
        default=50,
        description="Skip files larger than this (KB). Prevents slow analysis on generated code.",
    )
    min_function_length: int = Field(
        default=25,
        description="Minimum function body lines to suggest extract_method.",
    )
    min_nesting_depth: int = Field(
        default=3,
        description="Minimum nesting depth to suggest simplify_conditional.",
    )


class CoordinationConfig(BaseModel):
    """Multi-agent coordination configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable multi-agent file claim tracking and conflict detection.",
    )
    warn_on_write_overlap: bool = Field(
        default=True,
        description="Warn when multiple sessions claim write access to the same file.",
    )
    warn_on_stale_read: bool = Field(
        default=True,
        description="Warn when a file is modified by one session while another has a read claim.",
    )


class DecisionConfig(BaseModel):
    """Decision Intelligence Engine configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable decision trade-off analysis in enhance_prompt.",
    )
    max_decisions: int = Field(
        default=1,
        description="Maximum decision domains to surface per prompt.",
    )


class GuardrailConfig(BaseModel):
    """Cognitive guardrails configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable domain-aware pre-flight guardrails.",
    )
    max_guardrails: int = Field(
        default=3,
        description="Maximum guardrail items per prompt.",
    )


class ArchitectureConfig(BaseModel):
    """Architecture drift detection configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable architecture model validation when .mirdan/architecture.yaml exists.",
    )
    warn_in_prompt: bool = Field(
        default=True,
        description="Surface architectural context in enhance_prompt.",
    )


class CheckRunnerConfig(BaseModel):
    """Configuration for external tool orchestration (lint, typecheck, test)."""

    lint_command: str = Field(default="ruff check")
    typecheck_command: str = Field(default="mypy")
    test_command: str = Field(default="pytest -x --tb=short")
    test_timeout: int = Field(default=30, description="Max seconds for test execution")
    auto_fix_lint: bool = Field(default=True, description="Run lint --fix for auto-fixable issues")


class LLMConfig(BaseModel):
    """Local LLM configuration for the intelligence layer."""

    enabled: bool = Field(default=False, description="Enable local LLM features")
    backend: str = Field(default="auto", pattern="^(auto|ollama|llamacpp)$")
    ollama_url: str = Field(default="http://localhost:11434")
    gguf_dir: str = Field(default="~/.mirdan/models")  # Consumers MUST call Path(gguf_dir).expanduser()
    model_keep_alive: str = Field(default="5m", description="Unload model after idle period")
    n_ctx: int = Field(default=4096, description="Context window for inference")
    n_threads: int | None = Field(default=None, description="CPU threads (None=auto)")
    # Feature toggles
    triage: bool = Field(default=True)
    smart_validation: bool = Field(default=True)
    check_runner: bool = Field(default=True)
    prompt_optimization: bool = Field(default=True)
    research_agent: bool = Field(default=False)
    # Safety
    max_false_positive_ratio: float = Field(default=0.4)
    validate_llm_fixes: bool = Field(default=True)
    # Training data collection
    collect_training_data: bool = Field(default=False, description="Collect training samples for fine-tuning")
    # Nested config
    checks: CheckRunnerConfig = Field(default_factory=CheckRunnerConfig)

    @field_validator("ollama_url")
    @classmethod
    def validate_ollama_url_is_localhost(cls, v: str) -> str:
        """Enforce that ollama_url points to localhost only.

        The local intelligence layer must never connect to remote machines.
        This prevents SSRF and source code exfiltration via malicious
        project config files.
        """
        allowed_hosts = {"localhost", "127.0.0.1", "::1", "[::1]"}
        parsed = urlparse(v)
        hostname = (parsed.hostname or "").lower()
        if hostname not in allowed_hosts:
            raise ValueError(
                f"ollama_url must point to localhost (got host: {hostname!r}). "
                f"Allowed hosts: {', '.join(sorted(allowed_hosts))}. "
                "mirdan's local intelligence layer only connects to local services."
            )
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"ollama_url must use http or https (got: {parsed.scheme!r})."
            )
        return v


class MirdanConfig(BaseModel):
    """Main Mirdan configuration."""

    version: str = "1.0"
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    quality_profile: str = Field(
        default="default",
        description="Named quality profile (e.g. default, enterprise)",
    )
    custom_profiles: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Custom quality profile definitions",
    )
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    enhancement: EnhancementConfig = Field(default_factory=EnhancementConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    tokens: TokenConfig = Field(default_factory=TokenConfig)
    linters: LinterOrcConfig = Field(default_factory=LinterOrcConfig)
    hooks: HookConfig = Field(default_factory=HookConfig)
    platform: PlatformProfile = Field(default_factory=PlatformProfile)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    dependencies: DependencyConfig = Field(default_factory=DependencyConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    ceremony: CeremonyConfig = Field(default_factory=CeremonyConfig)
    coordination: CoordinationConfig = Field(default_factory=CoordinationConfig)
    tidy_first: TidyFirstConfig = Field(default_factory=TidyFirstConfig)
    decisions: DecisionConfig = Field(default_factory=DecisionConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    architecture: ArchitectureConfig = Field(default_factory=ArchitectureConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rules: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_workspace(self) -> bool:
        """Whether this config represents a workspace with multiple projects."""
        return self.workspace.enabled and len(self.workspace.projects) > 0

    def resolve_project_for_path(self, file_path: str | Path) -> SubProjectConfig | None:
        """Resolve a file path to its containing sub-project config.

        Uses longest path prefix match.

        Args:
            file_path: File path (relative to workspace root).

        Returns:
            The matching SubProjectConfig, or None if no match.
        """
        if not self.is_workspace:
            return None

        path_str = str(file_path)
        best_match: SubProjectConfig | None = None
        best_len = 0
        for proj in self.workspace.projects:
            prefix = proj.path
            if not prefix:
                continue
            normalized = prefix if prefix.endswith("/") else prefix + "/"
            if path_str.startswith(normalized) and len(prefix) > best_len:
                best_match = proj
                best_len = len(prefix)
        return best_match

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
        config, _path = cls.find_config_with_path(start_path)
        return config

    @classmethod
    def find_config_with_path(
        cls, start_path: Path | None = None
    ) -> tuple["MirdanConfig", Path | None]:
        """Find and load configuration, returning both config and its directory.

        Args:
            start_path: Directory to start searching from. Defaults to cwd.

        Returns:
            Tuple of (config, config_directory) where config_directory is
            the directory containing .mirdan/config.yaml, or None if not found.
        """
        if start_path is None:
            start_path = Path.cwd()

        current = start_path
        while current != current.parent:
            config_file = current / ".mirdan" / "config.yaml"
            if config_file.exists():
                return cls.load(config_file), current

            config_file = current / ".mirdan.yaml"
            if config_file.exists():
                return cls.load(config_file), current

            current = current.parent

        return cls(), None

    def save(self, config_path: Path) -> None:
        """Save configuration to a YAML file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


def get_default_config() -> MirdanConfig:
    """Get the default configuration."""
    return MirdanConfig()
