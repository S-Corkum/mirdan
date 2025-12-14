"""Context Aggregator - Orchestrates context gathering from multiple MCPs."""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from mirdan.config import MirdanConfig
from mirdan.core.client_registry import MCPClientRegistry
from mirdan.core.gatherers import (
    Context7Gatherer,
    EnyalGatherer,
    FilesystemGatherer,
    GathererResult,
    GitHubGatherer,
)
from mirdan.models import ContextBundle, Intent

logger = logging.getLogger(__name__)


class ContextAggregator:
    """Orchestrates context gathering from multiple MCP sources.

    Runs gatherers concurrently and merges results into a single ContextBundle.
    Handles timeouts and errors gracefully.
    """

    def __init__(self, config: MirdanConfig) -> None:
        """Initialize aggregator with configuration.

        Args:
            config: Mirdan configuration with MCP client configs
        """
        self._config = config
        self._registry = MCPClientRegistry(config)
        self._gatherers = self._create_gatherers()

    def _create_gatherers(self) -> list[Any]:
        """Create all available gatherers.

        Returns:
            List of gatherer instances
        """
        timeout = self._config.orchestration.gatherer_timeout
        return [
            Context7Gatherer(self._registry, timeout),
            FilesystemGatherer(self._registry, timeout),
            EnyalGatherer(self._registry, timeout),
            GitHubGatherer(self._registry, timeout),
        ]

    async def gather_all(
        self,
        intent: Intent,
        context_level: str = "auto",
    ) -> ContextBundle:
        """Gather context from all available MCPs.

        Runs gatherers concurrently with timeout. Merges all results.

        Args:
            intent: Analyzed intent from user prompt
            context_level: Depth of gathering ('minimal', 'auto', 'comprehensive')

        Returns:
            Merged ContextBundle with all gathered context
        """
        # Filter to available gatherers
        available_gatherers: list[Any] = []
        for gatherer in self._gatherers:
            try:
                if await gatherer.is_available():
                    available_gatherers.append(gatherer)
                    logger.debug("Gatherer '%s' is available", gatherer.name)
                else:
                    logger.debug("Gatherer '%s' is not available", gatherer.name)
            except Exception as e:
                logger.warning("Error checking availability of '%s': %s", gatherer.name, e)

        if not available_gatherers:
            logger.info("No gatherers available, returning empty context")
            return ContextBundle()

        # Run all concurrently with global timeout
        tasks = [
            self._run_gatherer(gatherer, intent, context_level) for gatherer in available_gatherers
        ]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self._config.orchestration.gather_timeout,
            )
        except TimeoutError:
            logger.warning(
                "Global context gathering timeout (%.1fs) exceeded",
                self._config.orchestration.gather_timeout,
            )
            results = []

        # Filter and merge valid results
        valid_results: list[GathererResult] = []
        for result in results:
            if isinstance(result, GathererResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.warning("Gatherer raised exception: %s", result)

        merged = self._merge_results(valid_results)

        logger.info(
            "Context gathering complete: %d gatherers, %d succeeded",
            len(available_gatherers),
            len(valid_results),
        )

        return merged

    async def _run_gatherer(
        self,
        gatherer: Any,
        intent: Intent,
        depth: str,
    ) -> GathererResult:
        """Run a single gatherer with error handling.

        Args:
            gatherer: Gatherer instance to run
            intent: Intent to gather for
            depth: Depth parameter

        Returns:
            GathererResult (success or failure)
        """
        try:
            logger.debug("Running gatherer '%s'", gatherer.name)
            result = await gatherer.gather(intent, depth)

            # Ensure we return a proper GathererResult type
            gatherer_result: GathererResult = result
            if gatherer_result.success:
                logger.debug(
                    "Gatherer '%s' succeeded with %d patterns, %d files, %d hints",
                    gatherer.name,
                    len(gatherer_result.context.existing_patterns),
                    len(gatherer_result.context.relevant_files),
                    len(gatherer_result.context.documentation_hints),
                )
            else:
                logger.warning(
                    "Gatherer '%s' failed: %s",
                    gatherer.name,
                    gatherer_result.error,
                )

            return gatherer_result

        except Exception as e:
            logger.error("Gatherer '%s' raised exception: %s", gatherer.name, e)
            return GathererResult(
                success=False,
                context=ContextBundle(),
                error=str(e),
            )

    def _merge_results(
        self,
        results: Sequence[GathererResult],
    ) -> ContextBundle:
        """Merge multiple gatherer results into one ContextBundle.

        Args:
            results: List of gatherer results

        Returns:
            Merged ContextBundle
        """
        merged = ContextBundle()

        for result in results:
            if not result.success:
                continue

            ctx = result.context

            # Merge tech_stack (dict)
            if ctx.tech_stack:
                merged.tech_stack.update(ctx.tech_stack)

            # Merge existing_patterns (list, avoid duplicates)
            for pattern in ctx.existing_patterns:
                if pattern not in merged.existing_patterns:
                    merged.existing_patterns.append(pattern)

            # Merge relevant_files (list, avoid duplicates)
            for file in ctx.relevant_files:
                if file not in merged.relevant_files:
                    merged.relevant_files.append(file)

            # Merge documentation_hints (list, avoid duplicates)
            for hint in ctx.documentation_hints:
                if hint not in merged.documentation_hints:
                    merged.documentation_hints.append(hint)

        return merged

    async def close(self) -> None:
        """Close all MCP client connections."""
        await self._registry.close_all()
