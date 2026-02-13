"""Mirdan MCP Server - AI Code Quality Orchestrator."""

import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastmcp import FastMCP

from mirdan.config import MirdanConfig
from mirdan.core.code_validator import CodeValidator
from mirdan.core.context_aggregator import ContextAggregator
from mirdan.core.intent_analyzer import IntentAnalyzer
from mirdan.core.orchestrator import MCPOrchestrator
from mirdan.core.plan_validator import PlanValidator
from mirdan.core.prompt_composer import PromptComposer
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import Intent, TaskType

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
    config: MirdanConfig


_components: _Components | None = None


def _get_components() -> _Components:
    """Get or create the singleton component set."""
    global _components  # noqa: PLW0603
    if _components is not None:
        return _components

    config = MirdanConfig.find_config()
    quality_standards = QualityStandards(config=config.quality)

    _components = _Components(
        intent_analyzer=IntentAnalyzer(config.project),
        quality_standards=quality_standards,
        prompt_composer=PromptComposer(quality_standards, config=config.enhancement),
        mcp_orchestrator=MCPOrchestrator(config.orchestration),
        context_aggregator=ContextAggregator(config),
        code_validator=CodeValidator(
            quality_standards, config=config.quality, thresholds=config.thresholds
        ),
        plan_validator=PlanValidator(config.planning, thresholds=config.thresholds),
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


@mcp.tool()
async def enhance_prompt(
    prompt: str,
    task_type: str = "auto",
    context_level: str = "auto",
) -> dict[str, Any]:
    """
    Automatically enhance a coding prompt with quality requirements,
    codebase context, and tool recommendations.

    Args:
        prompt: The original developer prompt
        task_type: Override auto-detection (generation|refactor|debug|review|test|planning|auto)
        context_level: How much context to gather (minimal|auto|comprehensive)

    Returns:
        Enhanced prompt with quality requirements and tool recommendations
    """
    # Validate input size
    if error := _check_input_size(prompt, "prompt", _MAX_PROMPT_LENGTH):
        return error

    c = _get_components()

    # Analyze intent
    intent = c.intent_analyzer.analyze(prompt)

    # Override task type if specified
    if task_type != "auto":
        with contextlib.suppress(ValueError):
            intent.task_type = TaskType(task_type)

    # Get tool recommendations
    tool_recommendations = c.mcp_orchestrator.suggest_tools(intent)

    # Gather context from configured MCPs
    context = await c.context_aggregator.gather_all(intent, context_level)

    # Compose enhanced prompt
    enhanced = c.prompt_composer.compose(intent, context, tool_recommendations)

    return enhanced.to_dict()


@mcp.tool()
async def analyze_intent(prompt: str) -> dict[str, Any]:
    """
    Analyze a prompt without enhancement, returning the detected intent,
    entities, and recommended approach.

    Args:
        prompt: The developer prompt to analyze

    Returns:
        Structured intent analysis
    """
    # Validate input size
    if error := _check_input_size(prompt, "prompt", _MAX_PROMPT_LENGTH):
        return error

    c = _get_components()
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


@mcp.tool()
async def suggest_tools(
    intent_description: str,
    available_mcps: str = "",
    discover_capabilities: bool = False,
) -> dict[str, Any]:
    """
    Suggest which MCP tools should be used for a given intent.

    Args:
        intent_description: Description of what you're trying to do
        available_mcps: Comma-separated list of available MCPs (optional)
        discover_capabilities: If True, query actual MCP capabilities for recommended MCPs

    Returns:
        Tool recommendations with priorities and reasons
    """
    # Validate input size
    if error := _check_input_size(intent_description, "intent_description", _MAX_PROMPT_LENGTH):
        return error

    c = _get_components()

    # Parse available MCPs
    mcps = [m.strip() for m in available_mcps.split(",")] if available_mcps else None

    # Analyze the intent
    intent = c.intent_analyzer.analyze(intent_description)

    # Get recommendations
    recommendations = c.mcp_orchestrator.suggest_tools(intent, mcps)

    # Optionally discover actual MCP capabilities
    discovered_mcps: dict[str, list[str]] = {}
    if discover_capabilities:
        for rec in recommendations:
            mcp_name = rec.mcp
            if mcp_name and c.context_aggregator.is_mcp_configured(mcp_name):
                capabilities = await c.context_aggregator.discover_mcp_capabilities(mcp_name)
                if capabilities:
                    discovered_mcps[mcp_name] = [t.name for t in capabilities.tools[:10]]

    return {
        "recommendations": [r.to_dict() for r in recommendations],
        "detected_intent": intent.task_type.value,
        "discovered_tools": discovered_mcps,
    }


@mcp.tool()
async def get_verification_checklist(
    task_type: str,
    touches_security: bool = False,
) -> dict[str, Any]:
    """
    Get a verification checklist for a specific task type.

    Args:
        task_type: Type of task (generation|refactor|debug|review|test)
        touches_security: Whether the task involves security-sensitive code

    Returns:
        Verification checklist appropriate for the task
    """
    # Create a minimal intent for checklist generation
    try:
        task = TaskType(task_type)
    except ValueError:
        task = TaskType.UNKNOWN

    intent = Intent(
        original_prompt="",
        task_type=task,
        touches_security=touches_security,
    )

    c = _get_components()
    verification_steps = c.prompt_composer.generate_verification_steps(intent)

    return {
        "task_type": task.value,
        "touches_security": touches_security,
        "checklist": verification_steps,
    }


@mcp.tool()
async def validate_code_quality(
    code: str,
    language: str = "auto",
    check_security: bool = True,
    check_architecture: bool = True,
    check_style: bool = True,
    severity_threshold: str = "warning",
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

    Returns:
        Validation results with pass/fail, score, violations, and summary
    """
    # Validate input size
    if error := _check_input_size(code, "code", _MAX_CODE_LENGTH):
        return error

    c = _get_components()
    result = c.code_validator.validate(
        code=code,
        language=language,
        check_security=check_security,
        check_architecture=check_architecture,
        check_style=check_style,
    )

    return result.to_dict(severity_threshold=severity_threshold)


@mcp.tool()
async def validate_plan_quality(
    plan: str,
    target_model: str = "haiku",
) -> dict[str, Any]:
    """
    Validate a plan for implementation by a less capable model.
    Returns a quality score and list of issues that need fixing.

    Args:
        plan: The plan text to validate
        target_model: Model that will implement (haiku|flash|cheap|capable)
                     Cheaper models require stricter plan quality.

    Returns:
        Quality scores, issues list, and ready_for_cheap_model flag
    """
    # Validate input size
    if error := _check_input_size(plan, "plan", _MAX_PLAN_LENGTH):
        return error

    c = _get_components()
    result = c.plan_validator.validate(plan, target_model)
    return result.to_dict()


def main() -> None:
    """Run the Mirdan MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
