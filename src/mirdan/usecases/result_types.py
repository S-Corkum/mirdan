"""TypedDict definitions for use-case return shapes.

Provides compile-time type safety for the dict-based return values
of EnhancePromptUseCase and ValidateCodeUseCase. Each TypedDict
represents the FULL (uncompressed) output shape. Output formatters
may strip keys for COMPACT/MINIMAL/MICRO formats.

These types exist for static analysis only — they impose no runtime
overhead due to ``from __future__ import annotations``.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class EnhancePromptResult(TypedDict):
    """Standard return shape for EnhancePromptUseCase.execute().

    Represents the FULL format output. COMPACT/MINIMAL/MICRO formats
    are subsets of these keys (output_formatter strips, never adds).
    """

    # Base keys from EnhancedPrompt.to_dict() — always present
    enhanced_prompt: str
    task_type: str
    task_types: list[str]
    language: str | None
    frameworks: list[str]
    extracted_entities: list[dict[str, Any]]
    touches_security: bool
    touches_rag: bool
    touches_knowledge_graph: bool
    ambiguity_score: float
    clarifying_questions: list[str]
    quality_requirements: list[str]
    verification_steps: list[str]
    tool_recommendations: list[dict[str, Any]]

    # Always added after to_dict()
    session_id: str
    environment: dict[str, Any]
    ceremony_level: str
    timing_ms: dict[str, float]

    # Conditional keys
    recommended_validation: NotRequired[str]
    ceremony_reason: NotRequired[str]
    knowledge_entries: NotRequired[list[dict[str, Any]]]
    coordination: NotRequired[dict[str, Any]]
    tidy_suggestions: NotRequired[dict[str, Any]]
    decision_guidance: NotRequired[list[dict[str, Any]]]
    cognitive_guardrails: NotRequired[list[dict[str, Any]]]
    architecture_context: NotRequired[Any]
    triage: NotRequired[dict[str, Any]]
    smart_analysis: NotRequired[dict[str, Any]]


class ValidateCodeResult(TypedDict):
    """Standard return shape for ValidateCodeUseCase.execute().

    Represents the FULL format output. COMPACT/MINIMAL/MICRO formats
    are subsets of these keys (output_formatter strips, never adds).
    """

    # Base keys from ValidationResult.to_dict() — always present
    passed: bool
    score: float
    language_detected: str
    violations_count: dict[str, int]
    violations: list[dict[str, Any]]
    summary: str
    standards_checked: list[str]
    limitations: list[str]

    # Always added after to_dict()
    checklist: list[str]
    timing_ms: dict[str, float]

    # Conditional keys
    smart_analysis: NotRequired[dict[str, Any]]
    quality_drift: NotRequired[dict[str, float]]
    security_regression: NotRequired[dict[str, Any]]
    semantic_checks: NotRequired[list[dict[str, Any]]]
    analysis_protocol: NotRequired[dict[str, Any]]
    confidence: NotRequired[dict[str, Any]]
    architecture_drift: NotRequired[dict[str, Any]]
    session_quality: NotRequired[dict[str, Any]]
    session_context: NotRequired[dict[str, Any]]
    recommendation_reminders: NotRequired[list[dict[str, str]]]
    knowledge_entries: NotRequired[list[dict[str, Any]]]
    knowledge_storage_hint: NotRequired[str]
    checklist_note: NotRequired[str]
    coordination: NotRequired[dict[str, Any]]
