"""Prompt optimization templates for BRAIN model (31B, FULL profile only).

Optimized for Gemma 4 31B with thinking enabled:
- Thinking mode: ENABLED (meta-prompting needs planning)
- Temperature: 0.3 (creative but structured)
- Structured output: format param for pruning, free-form text for optimization
"""

from __future__ import annotations

from typing import Any

# Sampling parameters for prompt optimization
OPTIMIZATION_SAMPLING: dict[str, Any] = {
    "temperature": 0.3,
    "top_p": 0.95,
    "thinking": True,
}

# Sampling for context pruning (structured output)
PRUNING_SAMPLING: dict[str, Any] = {
    "temperature": 0.1,
    "thinking": True,
}

# Per-model optimization strategies
TARGET_MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "opus": {
        "name": "Claude Opus",
        "style": "concise",
        "description": "Concise, constraints-focused. Use XML sections. 3K-8K token budget.",
        "max_context_tokens": 8000,
        "instruction": (
            "Craft a concise, constraint-rich prompt. Opus excels at complex reasoning "
            "with minimal instruction. Lead with constraints, use XML sections for structure. "
            "Omit obvious context — Opus infers well."
        ),
    },
    "sonnet": {
        "name": "Claude Sonnet",
        "style": "structured",
        "description": "Structured with examples, clear success criteria. 4K-12K token budget.",
        "max_context_tokens": 12000,
        "instruction": (
            "Craft a well-structured prompt with examples and clear success criteria. "
            "Sonnet benefits from explicit structure and 1-2 examples of expected output. "
            "Include verification steps."
        ),
    },
    "haiku": {
        "name": "Claude Haiku",
        "style": "step-by-step",
        "description": "Step-by-step, all context inline, explicit format. 2K-6K token budget.",
        "max_context_tokens": 6000,
        "instruction": (
            "Craft a step-by-step prompt with all necessary context inline. "
            "Haiku needs explicit instructions — do not assume inference. "
            "Number steps, specify exact output format, include all constraints."
        ),
    },
}

CONTEXT_PRUNING_PROMPT = """\
<|think|>
You are an expert at context curation for AI coding assistants.

Score each context item by relevance to the task (0.0-1.0).
Items scoring below 0.3 should be pruned.
{ide_instruction}

Task: {task_description}
Target model: {target_model}

Context items:
{context_items_json}

Respond with ONLY a JSON object:
{{"kept": [{{"item": "...", "score": 0.0, "reason": "..."}}], "pruned": [{{"item": "...", "score": 0.0, "reason": "..."}}]}}"""

CONTEXT_PRUNING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["kept", "pruned"],
    "properties": {
        "kept": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "score": {"type": "number"},
                    "reason": {"type": "string"},
                },
            },
        },
        "pruned": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "score": {"type": "number"},
                    "reason": {"type": "string"},
                },
            },
        },
    },
}

PROMPT_OPTIMIZATION_PROMPT = """\
<|think|>
You are a Staff Prompt Engineer crafting an optimal prompt for {target_model_name}.

{model_instruction}

Task intent:
{task_description}

Available context (already pruned for relevance):
{pruned_context}

Tool recommendations:
{tool_recommendations}

Quality requirements:
{quality_requirements}

Craft the optimized prompt now. Output ONLY the prompt text, no JSON wrapper."""


def build_pruning_prompt(
    task_description: str,
    context_items: list[str],
    target_model: str,
    is_cursor: bool = False,
) -> str:
    """Build context pruning prompt.

    Args:
        task_description: What the developer wants to do.
        context_items: List of context strings to evaluate.
        target_model: Target model name (opus/sonnet/haiku).
        is_cursor: Whether Cursor IDE is detected (more aggressive pruning).

    Returns:
        Complete pruning prompt.
    """
    import json

    ide_instruction = ""
    if is_cursor:
        ide_instruction = (
            "IMPORTANT: Cursor IDE detected — be MORE aggressive with pruning. "
            "The model has no visibility into context budget, so minimize context."
        )

    return CONTEXT_PRUNING_PROMPT.format(
        task_description=task_description,
        context_items_json=json.dumps(context_items, indent=2),
        target_model=target_model,
        ide_instruction=ide_instruction,
    )


def build_optimization_prompt(
    task_description: str,
    pruned_context: str,
    tool_recommendations: str,
    quality_requirements: str,
    target_model: str = "sonnet",
) -> str:
    """Build the prompt optimization prompt for the BRAIN model.

    Args:
        task_description: Developer's task.
        pruned_context: Already-pruned context.
        tool_recommendations: Tool recommendations as text.
        quality_requirements: Quality requirements as text.
        target_model: Target model (opus/sonnet/haiku).

    Returns:
        Complete optimization prompt.
    """
    profile = TARGET_MODEL_PROFILES.get(target_model, TARGET_MODEL_PROFILES["sonnet"])
    return PROMPT_OPTIMIZATION_PROMPT.format(
        target_model_name=profile["name"],
        model_instruction=profile["instruction"],
        task_description=task_description,
        pruned_context=pruned_context,
        tool_recommendations=tool_recommendations,
        quality_requirements=quality_requirements,
    )
