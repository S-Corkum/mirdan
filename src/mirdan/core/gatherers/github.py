"""GitHub gatherer for repository context."""

import json
import logging
from typing import Any

from fastmcp import Client

from mirdan.core.client_registry import MCPClientRegistry
from mirdan.core.gatherers.base import BaseGatherer, GathererResult
from mirdan.models import ContextBundle, Intent

logger = logging.getLogger(__name__)


# Depth mapping for commit counts
DEPTH_COMMIT_LIMITS = {
    "minimal": 0,  # Manifest only
    "auto": 5,
    "comprehensive": 20,
}


class GitHubGatherer(BaseGatherer):
    """Gathers repository context from GitHub MCP.

    Populates:
        - tech_stack: Dependencies from package manifests
        - relevant_files: Recently changed files
    """

    def __init__(self, registry: MCPClientRegistry, timeout: float = 3.0) -> None:
        """Initialize gatherer with client registry.

        Args:
            registry: MCP client registry for connections
            timeout: Timeout for GitHub operations
        """
        super().__init__(registry, timeout)
        self._owner: str | None = None
        self._repo: str | None = None

    @property
    def name(self) -> str:
        return "GitHubGatherer"

    @property
    def required_mcp(self) -> str:
        return "github"

    def set_repo_context(self, owner: str, repo: str) -> None:
        """Set the repository context for gathering.

        Args:
            owner: Repository owner
            repo: Repository name
        """
        self._owner = owner
        self._repo = repo

    async def gather(
        self,
        intent: Intent,
        depth: str = "auto",
    ) -> GathererResult:
        """Fetch repository context from GitHub.

        Logic:
        1. Get recent commits for context
        2. Read package manifest (package.json, pyproject.toml)
        3. Extract tech stack from dependencies
        4. List recently changed files

        Args:
            intent: Analyzed intent
            depth: 'minimal' (manifest only), 'auto' (+ 5 commits), 'comprehensive' (+ 20)

        Returns:
            GathererResult with tech stack and files
        """
        context = ContextBundle()

        # Skip if no repo context set
        if not self._owner or not self._repo:
            return GathererResult(
                success=True,
                context=context,
                metadata={"reason": "no_repo_context"},
            )

        client = await self._registry.get_client(self.required_mcp)
        if client is None:
            return GathererResult(
                success=False,
                context=context,
                error=f"MCP '{self.required_mcp}' not available",
            )

        commit_limit = DEPTH_COMMIT_LIMITS.get(depth, 5)
        tech_stack: dict[str, Any] = {}
        relevant_files: list[str] = []
        errors: list[str] = []

        try:
            async with client:
                # Step 1: Parse tech stack from manifest
                try:
                    tech_stack = await self._parse_tech_stack(client)
                except Exception as e:
                    logger.debug("Failed to parse tech stack: %s", e)
                    errors.append(f"Tech stack parsing: {e}")

                # Step 2: Get recent commits (if depth allows)
                if commit_limit > 0:
                    try:
                        commits = await self._get_recent_commits(client, limit=commit_limit)
                        # Extract changed files from commits
                        for commit in commits:
                            files = commit.get("files", [])
                            for file_info in files:
                                if isinstance(file_info, str):
                                    filename = file_info
                                else:
                                    filename = file_info.get("filename", "")
                                if filename and filename not in relevant_files:
                                    relevant_files.append(filename)
                    except Exception as e:
                        logger.debug("Failed to get commits: %s", e)
                        errors.append(f"Commits: {e}")

        except Exception as e:
            logger.error("GitHubGatherer failed: %s", e)
            return GathererResult(
                success=False,
                context=context,
                error=str(e),
            )

        context.tech_stack = tech_stack
        context.relevant_files = relevant_files[:20]  # Limit files

        return GathererResult(
            success=True,
            context=context,
            metadata={
                "owner": self._owner,
                "repo": self._repo,
                "tech_stack_entries": len(tech_stack),
                "files_found": len(relevant_files),
                "errors": errors if errors else None,
            },
        )

    async def _get_recent_commits(
        self,
        client: Client[Any],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get recent commit information.

        Args:
            client: Connected GitHub client
            limit: Number of commits to fetch

        Returns:
            List of commit info dicts
        """
        commits: list[dict[str, Any]] = []

        result = await client.call_tool(
            "list_commits",
            {
                "owner": self._owner,
                "repo": self._repo,
                "perPage": limit,
            },
        )

        if result.content:
            for content in result.content:
                if hasattr(content, "text"):
                    try:
                        data = json.loads(content.text)
                        if isinstance(data, list):
                            commits = data
                        elif isinstance(data, dict) and "commits" in data:
                            commits = data["commits"]
                    except json.JSONDecodeError:
                        # Try line-by-line parsing
                        pass

        return commits[:limit]

    async def _parse_tech_stack(
        self,
        client: Client[Any],
    ) -> dict[str, Any]:
        """Parse tech stack from manifest files.

        Args:
            client: Connected GitHub client

        Returns:
            Dict of technology: version mappings
        """
        tech_stack: dict[str, Any] = {}

        # Try package.json first (JavaScript/TypeScript projects)
        try:
            result = await client.call_tool(
                "get_file_contents",
                {
                    "owner": self._owner,
                    "repo": self._repo,
                    "path": "package.json",
                },
            )

            if result.content:
                for content in result.content:
                    if hasattr(content, "text"):
                        try:
                            pkg = json.loads(content.text)
                            deps = pkg.get("dependencies", {})
                            dev_deps = pkg.get("devDependencies", {})
                            tech_stack.update(deps)
                            tech_stack.update(dev_deps)
                            tech_stack["_type"] = "nodejs"
                        except json.JSONDecodeError:
                            pass
        except Exception:
            pass

        # Try pyproject.toml (Python projects)
        if not tech_stack:
            try:
                result = await client.call_tool(
                    "get_file_contents",
                    {
                        "owner": self._owner,
                        "repo": self._repo,
                        "path": "pyproject.toml",
                    },
                )

                if result.content:
                    for content in result.content:
                        if hasattr(content, "text"):
                            # Simple TOML parsing for dependencies
                            text = content.text
                            if "dependencies" in text:
                                tech_stack["_type"] = "python"
                                # Extract dependency names (simplified)
                                lines = text.split("\n")
                                in_deps = False
                                for line in lines:
                                    if "dependencies" in line and "=" in line:
                                        in_deps = True
                                        continue
                                    if in_deps:
                                        if line.startswith("["):
                                            in_deps = False
                                        elif "=" in line or ">=" in line:
                                            parts = line.split(">=")[0].split("=")[0]
                                            dep_name = parts.strip().strip('"').strip("'")
                                            if dep_name:
                                                tech_stack[dep_name] = "*"
            except Exception:
                pass

        return tech_stack
