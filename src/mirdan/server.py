"""Mirdan MCP Server - AI Code Quality Orchestrator."""

import contextlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from mirdan.config import MirdanConfig
from mirdan.core.code_validator import CodeValidator
from mirdan.core.context_aggregator import ContextAggregator
from mirdan.core.diff_parser import parse_unified_diff
from mirdan.core.environment_detector import detect_environment
from mirdan.core.intent_analyzer import IntentAnalyzer
from mirdan.core.knowledge_producer import KnowledgeProducer
from mirdan.core.linter_orchestrator import create_linter_runner, merge_linter_violations
from mirdan.core.linter_runner import LinterRunner
from mirdan.core.orchestrator import MCPOrchestrator
from mirdan.core.output_formatter import OutputFormatter
from mirdan.core.plan_validator import PlanValidator
from mirdan.core.prompt_composer import PromptComposer
from mirdan.core.quality_persistence import QualityPersistence
from mirdan.core.quality_standards import QualityStandards
from mirdan.core.session_manager import SessionManager
from mirdan.models import ComparisonEntry, ComparisonResult, Intent, ModelTier, TaskType

logger = logging.getLogger(__name__)

# Input size limits to prevent abuse and resource exhaustion
_MAX_PROMPT_LENGTH = 50_000  # ~12k tokens
_MAX_CODE_LENGTH = 500_000  # ~125k tokens
_MAX_PLAN_LENGTH = 200_000  # ~50k tokens


def _check_input_size(value: str, name: str, max_length: int) -> dict[str, Any] | None:
    """Return an error dict if value exceeds max_length, else None."""
    if len(value) > max_length:
        return {
            "error": f"{name} exceeds maximum length ({len(value):,} > {max_length:,} characters)",
            "max_length": max_length,
            "actual_length": len(value),
        }
    return None


@dataclass
class _Components:
    """Holds all initialized Mirdan components."""

    intent_analyzer: IntentAnalyzer
    quality_standards: QualityStandards
    prompt_composer: PromptComposer
    mcp_orchestrator: MCPOrchestrator
    context_aggregator: ContextAggregator
    code_validator: CodeValidator
    plan_validator: PlanValidator
    session_manager: SessionManager
    output_formatter: OutputFormatter
    quality_persistence: QualityPersistence
    knowledge_producer: KnowledgeProducer
    linter_runner: LinterRunner
    config: MirdanConfig


_components: _Components | None = None


def _get_components() -> _Components:
    """Get or create the singleton component set."""
    global _components
    if _components is not None:
        return _components

    config = MirdanConfig.find_config()
    quality_standards = QualityStandards(config=config.quality)

    # Detect project directory for AI002 import verification
    project_dir = Path.cwd()

    _components = _Components(
        intent_analyzer=IntentAnalyzer(config.project),
        quality_standards=quality_standards,
        prompt_composer=PromptComposer(quality_standards, config=config.enhancement),
        mcp_orchestrator=MCPOrchestrator(config.orchestration),
        context_aggregator=ContextAggregator(config),
        code_validator=CodeValidator(
            quality_standards,
            config=config.quality,
            thresholds=config.thresholds,
            project_dir=project_dir,
        ),
        plan_validator=PlanValidator(config.planning, thresholds=config.thresholds),
        session_manager=SessionManager(config.session),
        output_formatter=OutputFormatter(
            compact_threshold=config.tokens.compact_threshold,
            minimal_threshold=config.tokens.minimal_threshold,
        ),
        quality_persistence=QualityPersistence(),
        knowledge_producer=KnowledgeProducer(),
        linter_runner=create_linter_runner(config),
        config=config,
    )
    return _components


@asynccontextmanager
async def _lifespan(app: FastMCP[Any]) -> AsyncIterator[None]:
    """Manage server lifecycle: startup initialization and shutdown cleanup."""
    # Startup: eagerly initialize components
    _get_components()
    yield
    # Shutdown: cleanup MCP client connections
    if _components is not None:
        await _components.context_aggregator.close()


# Initialize the MCP server with lifespan
mcp = FastMCP("Mirdan", instructions="AI Code Quality Orchestrator", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Core Tool 1: enhance_prompt
# ---------------------------------------------------------------------------


@mcp.tool()
async def enhance_prompt(
    prompt: str,
    task_type: str = "auto",
    context_level: str = "auto",
    max_tokens: int = 0,
    model_tier: str = "auto",
) -> dict[str, Any]:
    """
    Automatically enhance a coding prompt with quality requirements,
    codebase context, and tool recommendations.

    Args:
        prompt: The original developer prompt
        task_type: Override auto-detection (generation|refactor|debug|review|test|planning|auto)
                   Use "analyze_only" to return just intent analysis without enhancement.
                   Use "plan_validation" to validate a plan for implementation quality.
        context_level: How much context to gather (minimal|auto|comprehensive|none)
        max_tokens: Maximum token budget for the response (0=unlimited). When set, output
                    is automatically compressed: <=1000 tokens produces minimal output,
                    <=4000 produces compact output.
        model_tier: Target model tier for output optimization (auto|opus|sonnet|haiku).
                    Haiku/Sonnet receive more compressed output.

    Returns:
        Enhanced prompt with quality requirements and tool recommendations
    """
    # Validate input size
    max_length = _MAX_PLAN_LENGTH if task_type == "plan_validation" else _MAX_PROMPT_LENGTH
    if error := _check_input_size(prompt, "prompt", max_length):
        return error

    c = _get_components()

    # --- Mode: analyze_only (replaces standalone analyze_intent tool) ---
    if task_type == "analyze_only":
        intent = c.intent_analyzer.analyze(prompt)
        ambiguity_level = (
            "low"
            if intent.ambiguity_score < 0.3
            else "medium"
            if intent.ambiguity_score < 0.6
            else "high"
        )
        return {
            "task_type": intent.task_type.value,
            "language": intent.primary_language,
            "frameworks": intent.frameworks,
            "touches_security": intent.touches_security,
            "touches_rag": intent.touches_rag,
            "touches_knowledge_graph": intent.touches_knowledge_graph,
            "uses_external_framework": intent.uses_external_framework,
            "ambiguity_score": intent.ambiguity_score,
            "ambiguity_level": ambiguity_level,
            "extracted_entities": [e.to_dict() for e in intent.entities],
            "clarifying_questions": intent.clarifying_questions,
        }

    # --- Mode: plan_validation (replaces standalone validate_plan_quality tool) ---
    if task_type == "plan_validation":
        result = c.plan_validator.validate(prompt, "haiku")
        return result.to_dict()

    # --- Standard enhancement flow ---
    # Analyze intent
    intent = c.intent_analyzer.analyze(prompt)

    # Override task type if specified
    if task_type != "auto":
        with contextlib.suppress(ValueError):
            intent.task_type = TaskType(task_type)

    # Create session from intent
    session = c.session_manager.create_from_intent(intent)

    # Get tool recommendations
    tool_recommendations = c.mcp_orchestrator.suggest_tools(intent)

    # Gather context from configured MCPs (skip if "none")
    if context_level == "none":
        from mirdan.models import ContextBundle

        context = ContextBundle()
    else:
        context = await c.context_aggregator.gather_all(intent, context_level)

    # Compose enhanced prompt
    enhanced = c.prompt_composer.compose(intent, context, tool_recommendations)

    result_dict = enhanced.to_dict()
    result_dict["session_id"] = session.session_id

    # Add knowledge entries from intent analysis
    knowledge_entries = c.knowledge_producer.extract_from_intent(
        task_type=intent.task_type.value,
        language=intent.primary_language,
        frameworks=intent.frameworks,
    )
    if knowledge_entries:
        result_dict["knowledge_entries"] = [e.to_dict() for e in knowledge_entries]

    # Detect environment for context
    env_info = detect_environment()
    result_dict["environment"] = env_info.to_dict()

    # Apply token-budget-aware formatting
    tier = _parse_model_tier(model_tier)
    result_dict = c.output_formatter.format_enhanced_prompt(
        result_dict, max_tokens=max_tokens, model_tier=tier
    )

    return result_dict


# ---------------------------------------------------------------------------
# Core Tool 2: validate_code_quality
# ---------------------------------------------------------------------------


@mcp.tool()
async def validate_code_quality(
    code: str,
    language: str = "auto",
    check_security: bool = True,
    check_architecture: bool = True,
    check_style: bool = True,
    severity_threshold: str = "warning",
    session_id: str = "",
    max_tokens: int = 0,
    model_tier: str = "auto",
    input_type: str = "code",
    compare: bool = False,
    file_path: str = "",
) -> dict[str, Any]:
    """
    Validate generated code against quality standards.

    Args:
        code: The code to validate
        language: Programming language (python|typescript|javascript|rust|go|auto)
        check_security: Validate against security standards
        check_architecture: Validate against architecture standards
        check_style: Validate against language-specific style standards
        severity_threshold: Minimum severity to include in results (error|warning|info)
        session_id: Session ID from enhance_prompt to auto-inherit language and security settings
        max_tokens: Maximum token budget for the response (0=unlimited)
        model_tier: Target model tier for output optimization (auto|opus|sonnet|haiku)
        input_type: Input type - "code" for raw code (default), "diff" for unified diff
        compare: If True, treat `code` as JSON array of implementations to compare
        file_path: Optional file path for external linter analysis. When provided,
                   runs ruff/eslint/mypy on the file and merges results.

    Returns:
        Validation results with pass/fail, score, violations, and summary
    """
    # --- Mode: compare (replaces standalone compare_approaches tool) ---
    if compare:
        return await _handle_compare(code, language)

    # --- Mode: diff (replaces standalone validate_diff tool) ---
    if input_type == "diff":
        return await _handle_diff(
            code, language, check_security, session_id, max_tokens, model_tier
        )

    # --- Standard code validation ---
    # Validate input size
    if error := _check_input_size(code, "code", _MAX_CODE_LENGTH):
        return error

    c = _get_components()

    # Apply session defaults if available
    resolved_language, resolved_security = c.session_manager.apply_session_defaults(
        session_id, language=language, check_security=check_security
    )

    result = c.code_validator.validate(
        code=code,
        language=resolved_language,
        check_security=resolved_security,
        check_architecture=check_architecture,
        check_style=check_style,
    )

    # Run external linters if file_path provided
    if file_path:
        fp = Path(file_path)
        if fp.exists():
            linter_violations = await c.linter_runner.run(fp, result.language_detected)
            if linter_violations:
                result = merge_linter_violations(result, linter_violations, c.config.thresholds)

    output = result.to_dict(severity_threshold=severity_threshold)

    # Add knowledge entries for enyal storage
    knowledge_entries = c.knowledge_producer.extract_from_validation(result)
    if knowledge_entries:
        output["knowledge_entries"] = [e.to_dict() for e in knowledge_entries]

    # Add verification checklist to output (absorbs get_verification_checklist)
    intent = Intent(
        original_prompt="",
        task_type=TaskType.UNKNOWN,
        touches_security=resolved_security,
    )
    # Detect task type from session if available
    if session_id:
        session = c.session_manager.get(session_id)
        if session:
            intent.task_type = session.task_type
    output["checklist"] = c.prompt_composer.generate_verification_steps(intent)

    # Apply token-budget-aware formatting
    tier = _parse_model_tier(model_tier)
    output = c.output_formatter.format_validation_result(
        output, max_tokens=max_tokens, model_tier=tier
    )

    return output


async def _handle_diff(
    diff: str,
    language: str,
    check_security: bool,
    session_id: str,
    max_tokens: int,
    model_tier: str,
) -> dict[str, Any]:
    """Handle diff validation (replaces standalone validate_diff tool)."""
    if error := _check_input_size(diff, "diff", _MAX_CODE_LENGTH):
        return error

    c = _get_components()

    # Parse the diff
    parsed = parse_unified_diff(diff)
    added_code = parsed.get_added_code()

    if not added_code.strip():
        return {
            "passed": True,
            "score": 1.0,
            "files_changed": parsed.files_changed,
            "summary": "No added code found in diff",
        }

    # Apply session defaults if available
    resolved_language, resolved_security = c.session_manager.apply_session_defaults(
        session_id, language=language, check_security=check_security
    )

    result = c.code_validator.validate(
        code=added_code,
        language=resolved_language,
        check_security=resolved_security,
        check_architecture=False,  # Architecture checks need full file context
        check_style=True,
    )

    output = result.to_dict(severity_threshold="warning")
    output["files_changed"] = parsed.files_changed
    output["lines_added"] = sum(len(h.added_lines) for h in parsed.hunks)

    # Apply token-budget-aware formatting
    tier = _parse_model_tier(model_tier)
    output = c.output_formatter.format_validation_result(
        output, max_tokens=max_tokens, model_tier=tier
    )

    return output


async def _handle_compare(
    code: str,
    language: str,
) -> dict[str, Any]:
    """Handle multi-implementation comparison (replaces standalone compare_approaches tool)."""
    try:
        implementations = json.loads(code)
    except (json.JSONDecodeError, TypeError):
        return {"error": "When compare=True, code must be a JSON array of implementation strings"}

    if not isinstance(implementations, list):
        return {"error": "When compare=True, code must be a JSON array of implementation strings"}

    if len(implementations) < 2:
        return {"error": "At least 2 implementations are required for comparison"}
    if len(implementations) > 10:
        return {"error": "Maximum 10 implementations can be compared at once"}

    for i, impl in enumerate(implementations):
        if not isinstance(impl, str):
            return {"error": f"implementation[{i}] must be a string"}
        if error := _check_input_size(impl, f"implementation[{i}]", _MAX_CODE_LENGTH):
            return error

    c = _get_components()

    labels = [f"Implementation {i + 1}" for i in range(len(implementations))]

    entries: list[ComparisonEntry] = []
    for impl, label in zip(implementations, labels, strict=True):
        result = c.code_validator.validate(
            code=impl,
            language=language,
            check_security=True,
            check_architecture=True,
            check_style=True,
        )
        output = result.to_dict(severity_threshold="warning")
        entries.append(
            ComparisonEntry(
                label=label,
                score=result.score,
                passed=result.passed,
                violation_counts=output.get("violations_count", {}),
                summary=output.get("summary", ""),
            )
        )

    # Determine winner (highest score, then fewest errors)
    best = max(entries, key=lambda e: (e.score, -e.violation_counts.get("error", 0)))

    comparison = ComparisonResult(
        entries=entries,
        winner=best.label,
        language_detected=entries[0].summary.split()[0] if entries else language,
    )

    return comparison.to_dict()


# ---------------------------------------------------------------------------
# Core Tool 2b: validate_quick
# ---------------------------------------------------------------------------


@mcp.tool()
async def validate_quick(
    code: str,
    language: str = "auto",
    max_tokens: int = 0,
    model_tier: str = "auto",
) -> dict[str, Any]:
    """Fast security-only validation for hooks and real-time feedback (<500ms target).

    Runs only security rules — skips style, architecture, framework, and custom checks.
    Ideal for PostToolUse hooks where speed matters more than comprehensive analysis.

    Args:
        code: The code to validate
        language: Programming language (python|typescript|javascript|rust|go|auto)
        max_tokens: Maximum token budget for the response (0=unlimited)
        model_tier: Target model tier for output optimization (auto|opus|sonnet|haiku)

    Returns:
        Validation results with pass/fail, score, and security-only violations
    """
    if error := _check_input_size(code, "code", _MAX_CODE_LENGTH):
        return error

    c = _get_components()

    result = c.code_validator.validate_quick(code=code, language=language)

    output = result.to_dict(severity_threshold="warning")

    # Apply token-budget-aware formatting
    tier = _parse_model_tier(model_tier)
    output = c.output_formatter.format_validation_result(
        output, max_tokens=max_tokens, model_tier=tier
    )

    return output


# ---------------------------------------------------------------------------
# Core Tool 3: get_quality_standards
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_quality_standards(
    language: str,
    framework: str = "",
    category: str = "all",
) -> dict[str, Any]:
    """
    Retrieve quality standards for a language/framework combination.

    Args:
        language: Programming language (typescript, python, etc.)
        framework: Optional framework (react, fastapi, etc.)
        category: Filter to specific category (security|architecture|style|all)

    Returns:
        Quality standards for the specified language/framework
    """
    c = _get_components()
    return c.quality_standards.get_all_standards(
        language=language, framework=framework, category=category
    )


# ---------------------------------------------------------------------------
# Core Tool 4: get_quality_trends
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_quality_trends(
    project_path: str = "",
    days: int = 30,
) -> dict[str, Any]:
    """
    Get quality score trends over time from stored validation history.

    Reads snapshots from `.mirdan/history/` and calculates aggregate
    statistics including average score, pass rate, and trend direction.

    Args:
        project_path: Optional project path filter
        days: Number of days of history to analyze (default: 30)

    Returns:
        Quality trend data with scores, pass rate, and trend direction
    """
    if days < 1:
        return {"error": "days must be at least 1"}
    if days > 365:
        return {"error": "days cannot exceed 365"}

    c = _get_components()
    trend = c.quality_persistence.get_trends(
        days=days,
        project_path=project_path or None,
    )
    return trend.to_dict()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _parse_model_tier(tier: str) -> ModelTier:
    """Parse a model tier string into the enum, defaulting to AUTO."""
    try:
        return ModelTier(tier.lower())
    except ValueError:
        return ModelTier.AUTO


def main() -> None:
    """Run the Mirdan MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
