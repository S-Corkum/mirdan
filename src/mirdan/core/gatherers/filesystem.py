"""Filesystem gatherer for codebase analysis."""

import logging
from typing import Any

from fastmcp import Client

from mirdan.core.gatherers.base import BaseGatherer, GathererResult
from mirdan.models import ContextBundle, Intent, TaskType

logger = logging.getLogger(__name__)


# File patterns by language
LANGUAGE_PATTERNS: dict[str, list[str]] = {
    "python": ["**/*.py"],
    "typescript": ["**/*.ts", "**/*.tsx"],
    "javascript": ["**/*.js", "**/*.jsx"],
    "rust": ["**/*.rs"],
    "go": ["**/*.go"],
}

# Patterns to look for by task type
TASK_SEARCH_PATTERNS: dict[TaskType, list[str]] = {
    TaskType.GENERATION: ["class ", "def ", "function ", "export "],
    TaskType.REFACTOR: ["class ", "interface ", "type "],
    TaskType.DEBUG: ["try:", "catch", "except", "error", "Error"],
    TaskType.TEST: ["test_", "Test", "describe(", "it("],
}

# Depth mapping for number of files to read
DEPTH_FILE_LIMITS = {
    "minimal": 0,  # List only, no reading
    "auto": 3,
    "comprehensive": 10,
}


class FilesystemGatherer(BaseGatherer):
    """Gathers relevant files and patterns from the codebase.

    Populates:
        - relevant_files: Files matching task criteria
        - existing_patterns: Code patterns found in files
    """

    @property
    def name(self) -> str:
        return "FilesystemGatherer"

    @property
    def required_mcp(self) -> str:
        return "filesystem"

    async def gather(
        self,
        intent: Intent,
        depth: str = "auto",
    ) -> GathererResult:
        """Search codebase for relevant files and patterns.

        Logic:
        1. Determine file patterns based on language
        2. Search for files matching patterns
        3. Read relevant files to extract patterns
        4. Identify coding patterns (imports, structures)

        Args:
            intent: Analyzed intent with language and task type
            depth: 'minimal' (list only), 'auto' (3 files), 'comprehensive' (10)

        Returns:
            GathererResult with files and patterns
        """
        context = ContextBundle()

        client = await self._registry.get_client(self.required_mcp)
        if client is None:
            return GathererResult(
                success=False,
                context=context,
                error=f"MCP '{self.required_mcp}' not available",
            )

        # Get language patterns
        language = intent.primary_language or "python"
        patterns = LANGUAGE_PATTERNS.get(language, ["**/*"])

        # Determine file limit based on depth
        file_limit = DEPTH_FILE_LIMITS.get(depth, 3)

        relevant_files: list[str] = []
        existing_patterns: list[str] = []
        errors: list[str] = []

        try:
            async with client:
                # Step 1: Search for files
                found_files = await self._search_files(client, patterns, limit=10)
                relevant_files = found_files[:file_limit] if file_limit > 0 else found_files

                # Step 2: Extract patterns from files (if depth allows)
                if file_limit > 0:
                    for file_path in relevant_files[:file_limit]:
                        try:
                            file_patterns = await self._extract_patterns(
                                client, file_path, intent.task_type
                            )
                            existing_patterns.extend(file_patterns)
                        except Exception as e:
                            logger.debug("Error extracting patterns from '%s': %s", file_path, e)
                            errors.append(f"Pattern extraction failed: {file_path}")

        except Exception as e:
            logger.error("FilesystemGatherer failed: %s", e)
            return GathererResult(
                success=False,
                context=context,
                error=str(e),
            )

        context.relevant_files = relevant_files
        context.existing_patterns = existing_patterns[:10]  # Limit patterns

        return GathererResult(
            success=True,
            context=context,
            metadata={
                "files_found": len(relevant_files),
                "patterns_found": len(existing_patterns),
                "language": language,
                "errors": errors if errors else None,
            },
        )

    async def _search_files(
        self,
        client: Client[Any],
        patterns: list[str],
        limit: int = 10,
    ) -> list[str]:
        """Search for files matching patterns.

        Args:
            client: Connected filesystem client
            patterns: Glob patterns to search
            limit: Maximum files to return

        Returns:
            List of matching file paths
        """
        all_files: list[str] = []

        for pattern in patterns:
            try:
                # Use the filesystem MCP's search or list tools
                # Common tool names: search_files, list_directory, glob
                result = await client.call_tool(
                    "search_files",
                    {"query": pattern, "include": patterns[:1]},
                )

                if result.content:
                    for content in result.content:
                        if hasattr(content, "text"):
                            # Parse file paths from result
                            lines = content.text.strip().split("\n")
                            for line in lines:
                                line = line.strip()
                                if line and not line.startswith("#"):
                                    all_files.append(line)
                                    if len(all_files) >= limit:
                                        return all_files

            except Exception as e:
                logger.debug("Search with pattern '%s' failed: %s", pattern, e)
                # Try alternative tool names
                try:
                    result = await client.call_tool(
                        "list_directory",
                        {"path": ".", "recursive": True},
                    )
                    if result.content:
                        for content in result.content:
                            if hasattr(content, "text"):
                                lines = content.text.strip().split("\n")
                                for line in lines:
                                    line = line.strip()
                                    if line and any(
                                        line.endswith(ext)
                                        for ext in [".py", ".ts", ".js", ".rs", ".go"]
                                    ):
                                        all_files.append(line)
                                        if len(all_files) >= limit:
                                            return all_files
                except Exception:
                    pass

        return all_files[:limit]

    async def _extract_patterns(
        self,
        client: Client[Any],
        file_path: str,
        task_type: TaskType,
    ) -> list[str]:
        """Extract code patterns from a file.

        Args:
            client: Connected filesystem client
            file_path: Path to file to analyze
            task_type: Task type to filter relevant patterns

        Returns:
            List of pattern descriptions
        """
        patterns: list[str] = []

        try:
            result = await client.call_tool(
                "read_file",
                {"path": file_path},
            )

            if not result.content:
                return patterns

            content_text = ""
            for content in result.content:
                if hasattr(content, "text"):
                    content_text += content.text

            if not content_text:
                return patterns

            # Look for relevant patterns based on task type
            search_patterns = TASK_SEARCH_PATTERNS.get(task_type, [])

            for search_term in search_patterns:
                if search_term in content_text:
                    # Extract the line containing the pattern
                    for line in content_text.split("\n"):
                        if search_term in line:
                            line = line.strip()
                            if line and len(line) < 200:  # Reasonable length
                                pattern_desc = f"{file_path}: {line[:100]}"
                                patterns.append(pattern_desc)
                                if len(patterns) >= 3:  # Max patterns per file
                                    return patterns

        except Exception as e:
            logger.debug("Error reading file '%s': %s", file_path, e)

        return patterns
