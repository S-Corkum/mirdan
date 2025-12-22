"""Context7 gatherer for fetching library documentation."""

import asyncio
import logging
from typing import Any

from fastmcp import Client

from mirdan.core.gatherers.base import BaseGatherer, GathererResult
from mirdan.models import ContextBundle, Intent

logger = logging.getLogger(__name__)


# Depth mapping for number of frameworks to fetch
DEPTH_FRAMEWORK_LIMITS = {
    "minimal": 1,
    "auto": 3,
    "comprehensive": 5,
}


class Context7Gatherer(BaseGatherer):
    """Gathers library documentation from context7 MCP.

    Populates:
        - documentation_hints: API docs and code examples
        - tech_stack: Library versions (if available)
    """

    @property
    def name(self) -> str:
        return "Context7Gatherer"

    @property
    def required_mcp(self) -> str:
        return "context7"

    async def gather(
        self,
        intent: Intent,
        depth: str = "auto",
    ) -> GathererResult:
        """Fetch documentation for frameworks in intent.

        Logic:
        1. Get frameworks from intent.frameworks
        2. For each framework:
           a. Call resolve-library-id to get library ID
           b. Call get-library-docs with ID and relevant topic
        3. Compile into documentation_hints

        Args:
            intent: Analyzed intent with frameworks list
            depth: 'minimal' (1 framework), 'auto' (3), 'comprehensive' (5)

        Returns:
            GathererResult with documentation hints
        """
        context = ContextBundle()

        if not intent.frameworks:
            logger.debug("No frameworks detected in intent, skipping Context7Gatherer")
            return GathererResult(
                success=True,
                context=context,
                metadata={"reason": "no_frameworks"},
            )

        client = await self._registry.get_client(self.required_mcp)
        if client is None:
            return GathererResult(
                success=False,
                context=context,
                error=f"MCP '{self.required_mcp}' not available",
            )

        # Determine how many frameworks to process based on depth
        max_frameworks = DEPTH_FRAMEWORK_LIMITS.get(depth, 3)
        frameworks_to_process = intent.frameworks[:max_frameworks]

        documentation_hints: list[str] = []
        tech_stack: dict[str, Any] = {}
        errors: list[str] = []

        try:
            async with client:
                for framework in frameworks_to_process:
                    try:
                        docs = await asyncio.wait_for(
                            self._fetch_framework_docs(client, framework),
                            timeout=self._timeout,
                        )
                        if docs:
                            if docs.get("docs"):
                                hint = f"[{framework}] {docs['docs']}"
                                documentation_hints.append(hint)
                            if docs.get("version"):
                                tech_stack[framework] = docs["version"]
                    except TimeoutError:
                        logger.warning("Timeout fetching docs for framework '%s'", framework)
                        errors.append(f"Timeout: {framework}")
                    except Exception as e:
                        logger.warning("Error fetching docs for '%s': %s", framework, e)
                        errors.append(f"Error ({framework}): {e}")

        except Exception as e:
            logger.error("Context7Gatherer failed: %s", e)
            return GathererResult(
                success=False,
                context=context,
                error=str(e),
            )

        context.documentation_hints = documentation_hints
        if tech_stack:
            context.tech_stack.update(tech_stack)

        return GathererResult(
            success=True,
            context=context,
            metadata={
                "frameworks_processed": len(frameworks_to_process),
                "docs_found": len(documentation_hints),
                "errors": errors if errors else None,
            },
        )

    async def _fetch_framework_docs(
        self,
        client: Client[Any],
        framework: str,
        topic: str | None = None,
    ) -> dict[str, Any]:
        """Fetch docs for a single framework.

        Args:
            client: Connected context7 client
            framework: Framework name (e.g., 'react', 'fastapi')
            topic: Optional topic to focus on

        Returns:
            Dict with 'library_id', 'docs', 'version' if available
        """
        result: dict[str, Any] = {}

        # Step 1: Resolve library ID
        try:
            resolve_result = await client.call_tool(
                "resolve-library-id",
                {"libraryName": framework},
            )

            # Extract library ID from result
            library_id = None
            if resolve_result.content:
                # Parse the text content to find a library ID
                for content in resolve_result.content:
                    if hasattr(content, "text"):
                        text = content.text
                        # Look for Context7-compatible library ID pattern
                        if "Context7-compatible library ID:" in text:
                            lines = text.split("\n")
                            for line in lines:
                                if "Context7-compatible library ID:" in line:
                                    # Extract the ID (format: /org/project)
                                    parts = line.split(":")
                                    if len(parts) >= 2:
                                        library_id = parts[1].strip()
                                        break
                        # Alternative: look for first /org/project pattern
                        elif not library_id:
                            import re

                            match = re.search(r"(/[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)", text)
                            if match:
                                library_id = match.group(1)
                                break

            if not library_id:
                logger.debug("Could not resolve library ID for '%s'", framework)
                return result

            result["library_id"] = library_id

        except Exception as e:
            logger.debug("Failed to resolve library ID for '%s': %s", framework, e)
            return result

        # Step 2: Get library docs
        try:
            params: dict[str, Any] = {
                "context7CompatibleLibraryID": library_id,
                "mode": "code",
            }
            if topic:
                params["topic"] = topic

            docs_result = await client.call_tool("get-library-docs", params)

            if docs_result.content:
                # Extract documentation text
                docs_text = ""
                for content in docs_result.content:
                    if hasattr(content, "text"):
                        docs_text += content.text + "\n"

                if docs_text:
                    # Truncate if too long (keep first 2000 chars)
                    if len(docs_text) > 2000:
                        docs_text = docs_text[:2000] + "..."
                    result["docs"] = docs_text.strip()

        except Exception as e:
            logger.debug("Failed to fetch docs for '%s': %s", framework, e)

        return result
