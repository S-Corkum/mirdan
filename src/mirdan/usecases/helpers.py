"""Shared helpers used by multiple use case modules."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mirdan.config import MirdanConfig
    from mirdan.core.active_orchestrator import ToolExecutor
    from mirdan.models import KnowledgeEntry

from mirdan.models import ModelTier

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


def _parse_model_tier(tier: str) -> ModelTier:
    """Parse a model tier string into the enum, defaulting to AUTO."""
    try:
        return ModelTier(tier.lower())
    except ValueError:
        return ModelTier.AUTO


def _process_knowledge_entries(
    entries: list[KnowledgeEntry],
    config: MirdanConfig,
    active_orchestrator: ToolExecutor,
    background_tasks: set[asyncio.Task[Any]],
) -> list[dict[str, Any]]:
    """Serialize knowledge entries, flag for client, and schedule server-side storage.

    When auto_memory is False (default): sets auto_store=True on entries above
    the confidence threshold, signaling the client to store them via enyal.

    When auto_memory is True: schedules fire-and-forget server-side storage
    via ToolExecutor and does NOT set auto_store (preventing double-storage).
    """
    orch_config = config.orchestration
    entries_out = []
    for e in entries:
        d = e.to_dict()
        if not orch_config.auto_memory and e.confidence >= orch_config.auto_memory_threshold:
            d["auto_store"] = True
        entries_out.append(d)

    if orch_config.auto_memory:
        coro = _auto_store_knowledge(
            active_orchestrator,
            entries,
            orch_config.auto_memory_threshold,
        )
        task = asyncio.create_task(coro)
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    return entries_out


async def _auto_store_knowledge(
    orchestrator: ToolExecutor,
    entries: list[KnowledgeEntry],
    threshold: float,
) -> None:
    """Fire-and-forget wrapper for auto-memory storage.

    Prevents 'Task exception was never retrieved' warnings from
    asyncio.create_task by catching and logging any errors.
    """
    try:
        await orchestrator.store_knowledge(entries, threshold)
    except Exception:
        logger.debug("Auto-memory storage failed", exc_info=True)
