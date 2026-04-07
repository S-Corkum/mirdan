"""Prompt optimizer — crafts optimized prompts per target model.

Both FAST (E2B/E4B on 16GB) and BRAIN (31B on 64GB+) models can do prompt
optimization. The BRAIN model produces higher-quality meta-prompting, but
the FAST model with thinking mode enabled is capable enough for context
pruning and prompt generation. Falls back to deterministic structured
assembly only if the LLM call itself fails.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mirdan.config import LLMConfig
from mirdan.llm.prompts.optimization import (
    CONTEXT_PRUNING_SCHEMA,
    OPTIMIZATION_SAMPLING,
    PRUNING_SAMPLING,
    TARGET_MODEL_PROFILES,
    build_optimization_prompt,
    build_pruning_prompt,
)
from mirdan.models import ModelRole, OptimizedPrompt

if TYPE_CHECKING:
    from mirdan.llm.manager import LLMManager

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """Crafts optimized prompts using the local LLM.

    Both FAST (E2B/E4B) and BRAIN (31B) models can do full prompt
    optimization — context pruning + LLM-powered prompt generation.
    The BRAIN model produces higher-quality output but the FAST model
    with thinking mode is capable enough for practical use on 16GB.

    Falls back to deterministic structured assembly only if the LLM
    prompt generation call fails (not because of model tier).
    Returns None when no local model is available at all.
    """

    def __init__(self, llm_manager: LLMManager | None = None, config: LLMConfig | None = None) -> None:
        self._llm = llm_manager
        self._config = config or LLMConfig()

    async def optimize(
        self,
        task_description: str,
        context_items: list[str],
        tool_recommendations: str,
        quality_requirements: str,
        target_model: str = "sonnet",
        detected_ide: str = "unknown",
    ) -> OptimizedPrompt | None:
        """Optimize a prompt for the target paid model.

        Uses BRAIN tier if available, falls back to FAST tier.

        Args:
            task_description: Developer's task intent.
            context_items: Gathered context strings.
            tool_recommendations: Tool recommendations text.
            quality_requirements: Quality requirements text.
            target_model: Target model (opus/sonnet/haiku).
            detected_ide: Detected IDE for pruning aggressiveness.

        Returns:
            OptimizedPrompt, or None if no local model is available.
        """
        if not self._llm or not self._config.prompt_optimization:
            return None

        is_cursor = detected_ide.lower() in ("cursor", "cursor_cli")
        use_brain = self._llm.is_role_available(ModelRole.BRAIN)

        # 1. Prune context (works with FAST or BRAIN)
        role = ModelRole.BRAIN if use_brain else ModelRole.FAST
        pruned_context, pruned_count = await self._prune_context(
            task_description, context_items, target_model, is_cursor, role
        )

        # 2. Generate optimized prompt via LLM (BRAIN or FAST with thinking)
        optimized_text = await self._generate_optimized_llm(
            task_description=task_description,
            pruned_context=pruned_context,
            tool_recommendations=tool_recommendations,
            quality_requirements=quality_requirements,
            target_model=target_model,
            role=role,
        )

        # Fallback: deterministic structured assembly if LLM call failed
        if not optimized_text:
            optimized_text = self._generate_optimized_fallback(
                task_description=task_description,
                pruned_context=pruned_context,
                tool_recommendations=tool_recommendations,
                quality_requirements=quality_requirements,
                target_model=target_model,
            )

        if not optimized_text:
            return None

        profile = TARGET_MODEL_PROFILES.get(target_model, TARGET_MODEL_PROFILES["sonnet"])
        model_tier = "brain" if use_brain else "fast"
        return OptimizedPrompt(
            text=optimized_text,
            context_pruned=pruned_count,
            target_model=target_model,
            optimization_notes=[
                f"Target: {profile['name']} ({profile['style']})",
                f"Model: {model_tier} ({'31B' if use_brain else 'E2B/E4B with thinking'})",
                f"Context pruned: {pruned_count} items removed",
                f"IDE: {detected_ide} ({'aggressive pruning' if is_cursor else 'standard'})",
            ],
        )

    async def _prune_context(
        self,
        task_description: str,
        context_items: list[str],
        target_model: str,
        is_cursor: bool,
        role: ModelRole,
    ) -> tuple[str, int]:
        """Prune low-relevance context items using the local LLM.

        Works with both FAST and BRAIN models — context scoring is a
        structured classification task that small models handle well.

        Args:
            task_description: Developer's task.
            context_items: Context strings to evaluate.
            target_model: Target model for budget awareness.
            is_cursor: Cursor IDE → more aggressive pruning.
            role: Which model role to use (FAST or BRAIN).

        Returns:
            Tuple of (pruned context as text, number of items pruned).
        """
        if not context_items:
            return "", 0

        if not self._llm:
            return "\n".join(context_items), 0

        prompt = build_pruning_prompt(
            task_description, context_items, target_model, is_cursor
        )

        try:
            result = await self._llm.generate_structured(
                role, prompt, CONTEXT_PRUNING_SCHEMA, **PRUNING_SAMPLING
            )
            if result:
                kept = result.get("kept", [])
                pruned = result.get("pruned", [])
                kept_text = "\n".join(item.get("item", "") for item in kept)
                return kept_text, len(pruned)
        except Exception:
            logger.warning("Context pruning failed, using original context")

        # Fallback: return all context unpruned
        return "\n".join(context_items), 0

    async def _generate_optimized_llm(
        self,
        task_description: str,
        pruned_context: str,
        tool_recommendations: str,
        quality_requirements: str,
        target_model: str,
        role: ModelRole,
    ) -> str | None:
        """Generate optimized prompt using the local LLM (FAST or BRAIN).

        Both E2B/E4B (FAST) and 31B (BRAIN) can do meta-prompting. The
        thinking mode on FAST models compensates for smaller parameter
        count on structured analytical tasks like prompt optimization.

        Args:
            task_description: Developer's task.
            pruned_context: Already-pruned context.
            tool_recommendations: Tool recs as text.
            quality_requirements: Quality reqs as text.
            target_model: Target model.
            role: Which model role to use (FAST or BRAIN).

        Returns:
            Optimized prompt text, or None on failure.
        """
        if not self._llm:
            return None

        prompt = build_optimization_prompt(
            task_description=task_description,
            pruned_context=pruned_context,
            tool_recommendations=tool_recommendations,
            quality_requirements=quality_requirements,
            target_model=target_model,
        )

        try:
            response = await self._llm.generate(
                role, prompt, **OPTIMIZATION_SAMPLING
            )
            if response and response.content:
                text: str = response.content.strip()
                return text
        except Exception:
            logger.warning("LLM prompt optimization failed (role=%s)", role.value)

        return None

    @staticmethod
    def _generate_optimized_fallback(
        task_description: str,
        pruned_context: str,
        tool_recommendations: str,
        quality_requirements: str,
        target_model: str,
    ) -> str:
        """Deterministic structured assembly fallback.

        Used only when the LLM prompt generation call itself fails
        (timeout, model error, etc). Not a tier — a safety net.

        Args:
            task_description: Developer's task.
            pruned_context: LLM-pruned context from _prune_context.
            tool_recommendations: Tool recs as text.
            quality_requirements: Quality reqs as text.
            target_model: Target model (opus/sonnet/haiku).

        Returns:
            Structured prompt text.
        """
        profile = TARGET_MODEL_PROFILES.get(target_model, TARGET_MODEL_PROFILES["sonnet"])
        parts: list[str] = []

        parts.append(f"Task: {task_description}")
        parts.append("")

        if pruned_context.strip():
            parts.append("Context:")
            parts.append(pruned_context)
            parts.append("")

        if quality_requirements.strip():
            parts.append("Quality requirements:")
            parts.append(quality_requirements)
            parts.append("")

        if tool_recommendations.strip():
            parts.append("Recommended tools:")
            parts.append(tool_recommendations)
            parts.append("")

        parts.append(f"Target: {profile['name']} — {profile['description']}")

        return "\n".join(parts)
