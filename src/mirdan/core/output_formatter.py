"""Token-budget-aware output compression engine."""

from __future__ import annotations

from typing import Any

from mirdan.models import ModelTier, OutputFormat


def estimate_tokens(text: str) -> int:
    """Estimate token count using chars/4 heuristic.

    This avoids a ~50MB tiktoken dependency while providing
    a reasonable approximation for output budgeting.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    return max(1, len(text) // 4)


def estimate_dict_tokens(data: dict[str, Any]) -> int:
    """Estimate token count for a serialized dictionary.

    Args:
        data: The dictionary to estimate.

    Returns:
        Estimated token count.
    """
    return estimate_tokens(str(data))


def determine_format(
    max_tokens: int, compact_threshold: int, minimal_threshold: int
) -> OutputFormat:
    """Determine output format based on token budget.

    Args:
        max_tokens: Maximum token budget (0 = unlimited).
        compact_threshold: Threshold below which compact format is used.
        minimal_threshold: Threshold below which minimal format is used.

    Returns:
        The appropriate OutputFormat.
    """
    if max_tokens <= 0:
        return OutputFormat.FULL
    if max_tokens <= minimal_threshold:
        return OutputFormat.MINIMAL
    if max_tokens <= compact_threshold:
        return OutputFormat.COMPACT
    return OutputFormat.FULL


def determine_format_for_model(model_tier: ModelTier) -> OutputFormat:
    """Determine output format based on model tier.

    Haiku gets minimal (ultra-compressed) output since it has limited
    context and works best with concise instructions. Sonnet gets
    compact output. Opus gets full output.

    Args:
        model_tier: The target model tier.

    Returns:
        The appropriate OutputFormat.
    """
    if model_tier == ModelTier.HAIKU:
        return OutputFormat.MINIMAL
    if model_tier == ModelTier.SONNET:
        return OutputFormat.COMPACT
    return OutputFormat.FULL


class OutputFormatter:
    """Formats output according to token budget and model tier constraints.

    Three compression levels:
    - FULL: Current behavior, all fields included.
    - COMPACT: No code_snippets/suggestions in violations, max 3 requirements,
               truncated tool recommendations.
    - MINIMAL: Pass/fail + score only.
    """

    def __init__(
        self,
        compact_threshold: int = 4000,
        minimal_threshold: int = 1000,
    ) -> None:
        self._compact_threshold = compact_threshold
        self._minimal_threshold = minimal_threshold

    def format_enhanced_prompt(
        self,
        data: dict[str, Any],
        max_tokens: int = 0,
        model_tier: ModelTier = ModelTier.AUTO,
    ) -> dict[str, Any]:
        """Format enhance_prompt output with token budget awareness.

        Args:
            data: The full enhance_prompt response dict.
            max_tokens: Token budget (0 = no compression).
            model_tier: Target model tier for output optimization.

        Returns:
            Formatted response dict.
        """
        fmt = self._resolve_format(max_tokens, model_tier)

        if fmt == OutputFormat.FULL:
            return data

        if fmt == OutputFormat.MINIMAL:
            return self._minimal_enhanced(data)

        return self._compact_enhanced(data)

    def format_validation_result(
        self,
        data: dict[str, Any],
        max_tokens: int = 0,
        model_tier: ModelTier = ModelTier.AUTO,
    ) -> dict[str, Any]:
        """Format validate_code_quality output with token budget awareness.

        Args:
            data: The full validation response dict.
            max_tokens: Token budget (0 = no compression).
            model_tier: Target model tier for output optimization.

        Returns:
            Formatted response dict.
        """
        fmt = self._resolve_format(max_tokens, model_tier)

        if fmt == OutputFormat.FULL:
            return data

        if fmt == OutputFormat.MINIMAL:
            return self._minimal_validation(data)

        return self._compact_validation(data)

    def _resolve_format(self, max_tokens: int, model_tier: ModelTier) -> OutputFormat:
        """Resolve the effective output format from token budget and model tier."""
        token_fmt = determine_format(max_tokens, self._compact_threshold, self._minimal_threshold)
        model_fmt = (
            determine_format_for_model(model_tier)
            if model_tier != ModelTier.AUTO
            else OutputFormat.FULL
        )

        # Use the more compressed of the two
        priority = {OutputFormat.MINIMAL: 0, OutputFormat.COMPACT: 1, OutputFormat.FULL: 2}
        if priority[token_fmt] <= priority[model_fmt]:
            return token_fmt
        return model_fmt

    def _compact_enhanced(self, data: dict[str, Any]) -> dict[str, Any]:
        """Compact format: truncate lists, remove verbose fields."""
        result: dict[str, Any] = {
            "enhanced_prompt": data.get("enhanced_prompt", ""),
            "task_type": data.get("task_type", ""),
            "language": data.get("language"),
            "frameworks": data.get("frameworks", []),
            "touches_security": data.get("touches_security", False),
        }

        # Truncate quality requirements to 3
        reqs = data.get("quality_requirements", [])
        result["quality_requirements"] = reqs[:3]

        # Truncate verification steps to 3
        steps = data.get("verification_steps", [])
        result["verification_steps"] = steps[:3]

        # Simplify tool recommendations (name + priority only)
        recs = data.get("tool_recommendations", [])
        result["tool_recommendations"] = [
            {"mcp": r.get("mcp", ""), "priority": r.get("priority", "")} for r in recs[:3]
        ]

        # Carry through session_id if present
        if "session_id" in data:
            result["session_id"] = data["session_id"]

        return result

    def _minimal_enhanced(self, data: dict[str, Any]) -> dict[str, Any]:
        """Minimal format: just the essentials."""
        result: dict[str, Any] = {
            "task_type": data.get("task_type", ""),
            "language": data.get("language"),
            "touches_security": data.get("touches_security", False),
        }
        if "session_id" in data:
            result["session_id"] = data["session_id"]
        return result

    def _compact_validation(self, data: dict[str, Any]) -> dict[str, Any]:
        """Compact validation: violations without code_snippet/suggestion."""
        violations = data.get("violations", [])
        compact_violations = [
            {
                "id": v.get("id", ""),
                "severity": v.get("severity", ""),
                "message": v.get("message", ""),
                "line": v.get("line"),
            }
            for v in violations
        ]
        return {
            "passed": data.get("passed", True),
            "score": data.get("score", 1.0),
            "language_detected": data.get("language_detected", ""),
            "violations_count": data.get("violations_count", {}),
            "violations": compact_violations,
            "summary": data.get("summary", ""),
        }

    def _minimal_validation(self, data: dict[str, Any]) -> dict[str, Any]:
        """Minimal validation: pass/fail and score only."""
        return {
            "passed": data.get("passed", True),
            "score": data.get("score", 1.0),
            "summary": data.get("summary", ""),
        }
