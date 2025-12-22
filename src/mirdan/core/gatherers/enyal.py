"""Enyal gatherer for project memory and conventions."""

import logging

from mirdan.core.gatherers.base import BaseGatherer, GathererResult
from mirdan.models import ContextBundle, Intent, TaskType

logger = logging.getLogger(__name__)


# Query templates by task type
TASK_QUERIES: dict[TaskType, str] = {
    TaskType.GENERATION: "conventions patterns code style",
    TaskType.REFACTOR: "architecture decisions refactoring patterns",
    TaskType.DEBUG: "error handling debugging patterns fixes",
    TaskType.REVIEW: "review conventions code standards",
    TaskType.TEST: "testing conventions test patterns",
    TaskType.DOCUMENTATION: "documentation conventions style",
    TaskType.UNKNOWN: "conventions patterns",
}

# Depth mapping for result limits
DEPTH_RESULT_LIMITS = {
    "minimal": 3,
    "auto": 5,
    "comprehensive": 10,
}


class EnyalGatherer(BaseGatherer):
    """Gathers project conventions and decisions from enyal memory.

    Populates:
        - existing_patterns: Stored conventions and decisions
    """

    @property
    def name(self) -> str:
        return "EnyalGatherer"

    @property
    def required_mcp(self) -> str:
        return "enyal"

    async def gather(
        self,
        intent: Intent,
        depth: str = "auto",
    ) -> GathererResult:
        """Recall relevant conventions and decisions.

        Logic:
        1. Build query from task type and intent keywords
        2. Call enyal_recall with query
        3. Filter results by relevance
        4. Format as pattern descriptions

        Args:
            intent: Analyzed intent with task type
            depth: 'minimal' (3 results), 'auto' (5), 'comprehensive' (10)

        Returns:
            GathererResult with recalled patterns
        """
        context = ContextBundle()

        client = await self._registry.get_client(self.required_mcp)
        if client is None:
            return GathererResult(
                success=False,
                context=context,
                error=f"MCP '{self.required_mcp}' not available",
            )

        # Build query and get result limit
        query = self._build_query(intent)
        result_limit = DEPTH_RESULT_LIMITS.get(depth, 5)

        existing_patterns: list[str] = []

        try:
            async with client:
                result = await client.call_tool(
                    "enyal_recall",
                    {
                        "input": {
                            "query": query,
                            "limit": result_limit,
                            "min_confidence": 0.3,
                        }
                    },
                )

                if result.content:
                    for content in result.content:
                        if hasattr(content, "text"):
                            # Parse the recall results
                            patterns = self._parse_recall_results(content.text)
                            existing_patterns.extend(patterns)

        except Exception as e:
            logger.error("EnyalGatherer failed: %s", e)
            return GathererResult(
                success=False,
                context=context,
                error=str(e),
            )

        context.existing_patterns = existing_patterns[:result_limit]

        return GathererResult(
            success=True,
            context=context,
            metadata={
                "query": query,
                "patterns_found": len(existing_patterns),
            },
        )

    def _build_query(self, intent: Intent) -> str:
        """Build recall query from intent.

        Args:
            intent: Analyzed intent

        Returns:
            Query string for enyal_recall
        """
        # Start with task-type specific query
        base_query = TASK_QUERIES.get(intent.task_type, "conventions patterns")

        # Add language if available
        if intent.primary_language:
            base_query = f"{intent.primary_language} {base_query}"

        # Add framework terms if available
        if intent.frameworks:
            frameworks_str = " ".join(intent.frameworks[:2])
            base_query = f"{base_query} {frameworks_str}"

        # Add security term if relevant
        if intent.touches_security:
            base_query = f"{base_query} security"

        return base_query

    def _parse_recall_results(self, text: str) -> list[str]:
        """Parse recall results into pattern descriptions.

        Args:
            text: Raw text from enyal recall

        Returns:
            List of formatted pattern descriptions
        """
        patterns: list[str] = []

        # Try to parse as JSON-like structure first
        try:
            import json

            data = json.loads(text)
            if isinstance(data, dict):
                results = data.get("results", [])
                for item in results:
                    if isinstance(item, dict):
                        content = item.get("content", "")
                        content_type = item.get("type", "pattern")
                        confidence = item.get("confidence", 0)
                        if content and confidence >= 0.3:
                            # Format as a pattern description
                            pattern = f"[{content_type}] {content}"
                            if len(pattern) > 200:
                                pattern = pattern[:200] + "..."
                            patterns.append(pattern)
            return patterns
        except json.JSONDecodeError:
            pass

        # Fall back to line-by-line parsing
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and len(line) > 10:
                # Clean up and add as pattern
                if len(line) > 200:
                    line = line[:200] + "..."
                patterns.append(line)

        return patterns
