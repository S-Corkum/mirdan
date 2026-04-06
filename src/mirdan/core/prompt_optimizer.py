"""Prompt optimizer — BRAIN model crafts optimized prompts per target model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mirdan.config import LLMConfig

if TYPE_CHECKING:
    from mirdan.llm.manager import LLMManager
from mirdan.llm.prompts.optimization import (
    CONTEXT_PRUNING_SCHEMA,
    OPTIMIZATION_SAMPLING,
    PRUNING_SAMPLING,
    TARGET_MODEL_PROFILES,
    build_optimization_prompt,
    build_pruning_prompt,
)
from mirdan.models import ModelRole, OptimizedPrompt

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """Crafts optimized prompts using the BRAIN model (31B, FULL profile only).

    Prunes irrelevant context, selects target model profile, and generates
    a prompt a Staff Prompt Engineer would write. Returns None gracefully
    when BRAIN model is unavailable (most 16GB users).
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

        Args:
            task_description: Developer's task intent.
            context_items: Gathered context strings.
            tool_recommendations: Tool recommendations text.
            quality_requirements: Quality requirements text.
            target_model: Target model (opus/sonnet/haiku).
            detected_ide: Detected IDE for pruning aggressiveness.

        Returns:
            OptimizedPrompt with optimized text, or None if BRAIN unavailable.
        """
        if not self._llm or not self._config.prompt_optimization:
            return None

        # Check if BRAIN model is available
        if not await self._is_brain_available():
            return None

        is_cursor = detected_ide.lower() in ("cursor", "cursor_cli")

        # 1. Prune context
        pruned_context, pruned_count = await self._prune_context(
            task_description, context_items, target_model, is_cursor
        )

        # 2. Generate optimized prompt
        optimized_text = await self._generate_optimized(
            task_description=task_description,
            pruned_context=pruned_context,
            tool_recommendations=tool_recommendations,
            quality_requirements=quality_requirements,
            target_model=target_model,
        )

        if not optimized_text:
            return None

        profile = TARGET_MODEL_PROFILES.get(target_model, TARGET_MODEL_PROFILES["sonnet"])
        return OptimizedPrompt(
            text=optimized_text,
            context_pruned=pruned_count,
            target_model=target_model,
            optimization_notes=[
                f"Target: {profile['name']} ({profile['style']})",
                f"Context pruned: {pruned_count} items removed",
                f"IDE: {detected_ide} ({'aggressive pruning' if is_cursor else 'standard'})",
            ],
        )

    async def _is_brain_available(self) -> bool:
        """Check if BRAIN model is selectable via LLMManager's public API."""
        if not self._llm:
            return False
        return self._llm.is_role_available(ModelRole.BRAIN)

    async def _prune_context(
        self,
        task_description: str,
        context_items: list[str],
        target_model: str,
        is_cursor: bool,
    ) -> tuple[str, int]:
        """Prune low-relevance context items using BRAIN model.

        Args:
            task_description: Developer's task.
            context_items: Context strings to evaluate.
            target_model: Target model for budget awareness.
            is_cursor: Cursor IDE → more aggressive.

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
                ModelRole.BRAIN, prompt, CONTEXT_PRUNING_SCHEMA, **PRUNING_SAMPLING
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

    async def _generate_optimized(
        self,
        task_description: str,
        pruned_context: str,
        tool_recommendations: str,
        quality_requirements: str,
        target_model: str,
    ) -> str | None:
        """Generate optimized prompt text using BRAIN model.

        Args:
            task_description: Developer's task.
            pruned_context: Already-pruned context.
            tool_recommendations: Tool recs as text.
            quality_requirements: Quality reqs as text.
            target_model: Target model.

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
                ModelRole.BRAIN, prompt, **OPTIMIZATION_SAMPLING
            )
            if response and response.content:
                text: str = response.content.strip()
                return text
        except Exception:
            logger.warning("Prompt optimization failed")

        return None
