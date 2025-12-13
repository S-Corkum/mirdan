"""Mirdan MCP Server - AI Code Quality Orchestrator."""

from typing import Any

from fastmcp import FastMCP

from mirdan.core.intent_analyzer import IntentAnalyzer
from mirdan.core.orchestrator import MCPOrchestrator
from mirdan.core.prompt_composer import PromptComposer
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import ContextBundle, TaskType

# Initialize the MCP server
mcp = FastMCP("Mirdan", description="AI Code Quality Orchestrator")

# Initialize components
intent_analyzer = IntentAnalyzer()
quality_standards = QualityStandards()
prompt_composer = PromptComposer(quality_standards)
mcp_orchestrator = MCPOrchestrator()


@mcp.tool()
def enhance_prompt(
    prompt: str,
    task_type: str = "auto",
    context_level: str = "auto",
) -> dict[str, Any]:
    """
    Automatically enhance a coding prompt with quality requirements,
    codebase context, and tool recommendations.

    Args:
        prompt: The original developer prompt
        task_type: Override auto-detection (generation|refactor|debug|review|test|auto)
        context_level: How much context to gather (minimal|auto|comprehensive)

    Returns:
        Enhanced prompt with quality requirements and tool recommendations
    """
    # Analyze intent
    intent = intent_analyzer.analyze(prompt)

    # Override task type if specified
    if task_type != "auto":
        try:
            intent.task_type = TaskType(task_type)
        except ValueError:
            pass  # Keep auto-detected type

    # Get tool recommendations
    tool_recommendations = mcp_orchestrator.suggest_tools(intent)

    # Create context bundle (in real implementation, this would query MCPs)
    context = ContextBundle()

    # Compose enhanced prompt
    enhanced = prompt_composer.compose(intent, context, tool_recommendations)

    return enhanced.to_dict()


@mcp.tool()
def analyze_intent(prompt: str) -> dict[str, Any]:
    """
    Analyze a prompt without enhancement, returning the detected intent,
    entities, and recommended approach.

    Args:
        prompt: The developer prompt to analyze

    Returns:
        Structured intent analysis
    """
    intent = intent_analyzer.analyze(prompt)

    ambiguity_level = (
        "low"
        if intent.ambiguity_score < 0.3
        else "medium" if intent.ambiguity_score < 0.6 else "high"
    )

    return {
        "task_type": intent.task_type.value,
        "primary_language": intent.primary_language,
        "frameworks": intent.frameworks,
        "touches_security": intent.touches_security,
        "uses_external_framework": intent.uses_external_framework,
        "ambiguity_score": intent.ambiguity_score,
        "ambiguity_level": ambiguity_level,
    }


@mcp.tool()
def get_quality_standards(
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
    return quality_standards.get_all_standards(language=language, category=category)


@mcp.tool()
def suggest_tools(
    intent_description: str,
    available_mcps: str = "",
) -> dict[str, Any]:
    """
    Suggest which MCP tools should be used for a given intent.

    Args:
        intent_description: Description of what you're trying to do
        available_mcps: Comma-separated list of available MCPs (optional)

    Returns:
        Tool recommendations with priorities and reasons
    """
    # Parse available MCPs
    mcps = [m.strip() for m in available_mcps.split(",")] if available_mcps else None

    # Analyze the intent
    intent = intent_analyzer.analyze(intent_description)

    # Get recommendations
    recommendations = mcp_orchestrator.suggest_tools(intent, mcps)

    return {
        "recommendations": [r.to_dict() for r in recommendations],
        "detected_intent": intent.task_type.value,
    }


@mcp.tool()
def get_verification_checklist(
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
    from mirdan.models import Intent

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

    verification_steps = prompt_composer._generate_verification_steps(intent)

    return {
        "task_type": task.value,
        "touches_security": touches_security,
        "checklist": verification_steps,
    }


def main() -> None:
    """Run the Mirdan MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
