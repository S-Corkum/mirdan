"""Research agent prompts for BRAIN model agentic tool selection.

Optimized for Gemma 4 31B with thinking enabled:
- Thinking mode: ENABLED (tool selection needs planning)
- Temperature: 0.2 (mostly deterministic tool selection)
- Function calling: native tool_call tokens if supported, JSON fallback otherwise
"""

from __future__ import annotations

from typing import Any

# Sampling parameters for tool selection
TOOL_SELECTION_SAMPLING: dict[str, Any] = {
    "temperature": 0.2,
    "top_p": 0.95,
    "thinking": True,
}

# Sampling for synthesis
SYNTHESIS_SAMPLING: dict[str, Any] = {
    "temperature": 0.1,
    "thinking": True,
}

# Maximum iterations for the agentic loop
MAX_ITERATIONS = 5

# Maximum local tokens to spend on research
MAX_TOKEN_BUDGET = 10000

RESEARCH_SYSTEM_PROMPT = """\
<|think|>
You are a research agent that gathers context for a coding task by calling MCP tools.
Select the most useful tool to call next based on the task and what you've learned so far.
If you have enough information, respond with null to indicate research is complete.

Available tools:
{tool_descriptions}

Task: {task_description}

Previous results:
{previous_results}

Select the next tool to call. Respond with ONLY a JSON object:
{{"tool": {{"mcp": "...", "name": "...", "arguments": {{...}}}}}}"
Or if research is complete:
{{"tool": null}}"""

TOOL_SELECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["tool"],
    "properties": {
        "tool": {
            "type": ["object", "null"],
            "properties": {
                "mcp": {"type": "string"},
                "name": {"type": "string"},
                "arguments": {"type": "object"},
            },
        }
    },
}

SYNTHESIS_PROMPT = """\
<|think|>
Synthesize the following research results into a concise context summary for a coding task.
Focus on actionable information: patterns to follow, APIs to use, constraints to respect.

Task: {task_description}

Research results:
{results_text}

Respond with a concise synthesis (200-500 words). No JSON, just text."""

FEW_SHOT_EXAMPLES = [
    {
        "task": "Add FastAPI endpoint for user registration",
        "selection": '{"tool": {"mcp": "context7", "name": "get-library-docs", "arguments": {"libraryId": "fastapi", "topic": "routing"}}}',
    },
    {
        "task": "Fix the database migration script",
        "selection": '{"tool": {"mcp": "enyal", "name": "enyal_recall", "arguments": {"query": "database migration convention"}}}',
    },
    {
        "task": "Simple task with enough context",
        "selection": '{"tool": null}',
    },
]


def build_tool_selection_prompt(
    task_description: str,
    tool_descriptions: list[dict[str, str]],
    previous_results: list[dict[str, Any]],
) -> str:
    """Build the tool selection prompt for the agentic loop.

    Args:
        task_description: Developer's task.
        tool_descriptions: Available tool descriptions.
        previous_results: Results from previous iterations.

    Returns:
        Complete tool selection prompt.
    """
    import json

    tools_text = json.dumps(tool_descriptions, indent=2)
    results_text = (
        json.dumps(previous_results, indent=2) if previous_results else "None yet."
    )

    return RESEARCH_SYSTEM_PROMPT.format(
        tool_descriptions=tools_text,
        task_description=task_description,
        previous_results=results_text,
    )


def build_synthesis_prompt(
    task_description: str,
    results: list[dict[str, Any]],
) -> str:
    """Build the synthesis prompt for combining research results.

    Args:
        task_description: Developer's task.
        results: All research results to synthesize.

    Returns:
        Complete synthesis prompt.
    """
    import json

    return SYNTHESIS_PROMPT.format(
        task_description=task_description,
        results_text=json.dumps(results, indent=2),
    )
