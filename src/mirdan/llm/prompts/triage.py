"""Triage classification prompts for local LLM.

Optimized for Gemma 4 FAST model:
- Thinking mode: DISABLED (classification, not reasoning)
- Temperature: 0 (deterministic)
- top_k: 1 (greedy decoding)
- Structured output: prompt-based JSON (Ollama format bug workaround)
"""

from __future__ import annotations

from typing import Any

# Sampling parameters for triage — passed to LLMManager.generate_structured()
TRIAGE_SAMPLING: dict[str, Any] = {
    "temperature": 0,
    "top_k": 1,
    "thinking": False,
}

TRIAGE_SYSTEM_PROMPT = """\
You are a task classifier for AI-assisted coding. Classify the developer's task into exactly one category. Respond with ONLY a JSON object, no other text.

Categories:
- local_only: Trivial tasks a linter or formatter can handle (unused imports, formatting, simple renames)
- local_assist: Simple tasks needing minimal AI help (docstrings, basic refactors, type annotations)
- paid_minimal: Standard implementation tasks with clear patterns (CRUD endpoints, unit tests, simple features)
- paid_required: Complex tasks needing deep reasoning (architecture, security, multi-file refactors, novel algorithms)

Respond with ONLY this JSON:
{"classification": "<category>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}"""

TRIAGE_FEW_SHOT = [
    {
        "user": "fix the unused import on line 5",
        "response": '{"classification": "local_only", "confidence": 0.95, "reasoning": "Single unused import removal"}',
    },
    {
        "user": "format this file with black",
        "response": '{"classification": "local_only", "confidence": 0.92, "reasoning": "Code formatting is a tool task"}',
    },
    {
        "user": "rename variable 'x' to 'count' in process_data()",
        "response": '{"classification": "local_only", "confidence": 0.90, "reasoning": "Simple single-variable rename"}',
    },
    {
        "user": "add docstrings to the User class",
        "response": '{"classification": "local_assist", "confidence": 0.85, "reasoning": "Mechanical documentation task"}',
    },
    {
        "user": "add type annotations to the utils module",
        "response": '{"classification": "local_assist", "confidence": 0.82, "reasoning": "Systematic type annotation without logic changes"}',
    },
    {
        "user": "add a GET endpoint for /users",
        "response": '{"classification": "paid_minimal", "confidence": 0.80, "reasoning": "Standard CRUD endpoint with clear pattern"}',
    },
    {
        "user": "write unit tests for the OrderService class",
        "response": '{"classification": "paid_minimal", "confidence": 0.78, "reasoning": "Standard test writing with known patterns"}',
    },
    {
        "user": "implement JWT authentication for the API",
        "response": '{"classification": "paid_required", "confidence": 0.90, "reasoning": "Security-sensitive multi-file feature"}',
    },
    {
        "user": "refactor the payment module to use the Strategy pattern",
        "response": '{"classification": "paid_required", "confidence": 0.88, "reasoning": "Architecture-level multi-file refactor"}',
    },
    {
        "user": "fix the auth bug",
        "response": '{"classification": "paid_required", "confidence": 0.55, "reasoning": "Ambiguous scope, could be simple or complex"}',
    },
]

# JSON schema for llama-cpp-python grammar constraint.
# Works reliably even without thinking mode.
TRIAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["classification", "confidence", "reasoning"],
    "properties": {
        "classification": {
            "type": "string",
            "enum": ["local_only", "local_assist", "paid_minimal", "paid_required"],
        },
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
}


def build_triage_prompt(user_prompt: str) -> str:
    """Build the full triage prompt with system context and few-shot examples.

    Args:
        user_prompt: The developer's original task description.

    Returns:
        Complete prompt string for the local LLM.
    """
    parts = [TRIAGE_SYSTEM_PROMPT, ""]

    for example in TRIAGE_FEW_SHOT:
        parts.append(f"User: \"{example['user']}\"")
        parts.append(example["response"])
        parts.append("")

    parts.append(f'User: "{user_prompt}"')

    return "\n".join(parts)
