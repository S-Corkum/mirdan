"""Research agent — BRAIN model autonomously gathers context via MCPs."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from typing import TYPE_CHECKING, Any

from mirdan.config import LLMConfig

if TYPE_CHECKING:
    from mirdan.llm.manager import LLMManager
from mirdan.llm.prompts.research import (
    MAX_ITERATIONS,
    MAX_TOKEN_BUDGET,
    SYNTHESIS_SAMPLING,
    TOOL_SELECTION_SAMPLING,
    TOOL_SELECTION_SCHEMA,
    build_synthesis_prompt,
    build_tool_selection_prompt,
)
from mirdan.models import (
    Intent,
    MCPToolCall,
    ModelRole,
    ResearchResult,
    ToolRecommendation,
)

logger = logging.getLogger(__name__)

# Read-only tools safe for autonomous research agent use.
# The research agent gathers context — it must NEVER modify state.
# Keyed by MCP name → frozenset of allowed tool names.
RESEARCH_SAFE_TOOLS: dict[str, frozenset[str]] = {
    "context7": frozenset(
        {
            "resolve-library-id",
            "query-docs",
            "get-library-docs",
        }
    ),
    "enyal": frozenset(
        {
            "enyal_recall",
            "enyal_recall_by_scope",
            "enyal_get",
            "enyal_traverse",
            "enyal_impact",
            "enyal_edges",
            "enyal_history",
            "enyal_stats",
            "enyal_health",
            "enyal_review",
            "enyal_analytics",
        }
    ),
    "sequential-thinking": frozenset({"sequentialthinking"}),
    "github": frozenset(
        {
            "get_me",
            "list_issues",
            "search_issues",
            "issue_read",
            "list_pull_requests",
            "search_pull_requests",
            "pull_request_read",
            "search_code",
            "get_file_contents",
            "list_branches",
            "list_commits",
            "get_commit",
            "list_releases",
            "get_latest_release",
            "get_release_by_tag",
        }
    ),
}


# Validation pattern for owner/repo names (prevents injection)
_OWNER_REPO_RE = re.compile(r"^[a-zA-Z0-9._-]{1,100}$")

# Mapping of GitHub MCP tool names to gh CLI command builders.
# Each builder takes an arguments dict and returns a command list for create_subprocess_exec.
_GH_TOOL_COMMANDS: dict[str, Any] = {
    "list_commits": lambda a: [
        "gh",
        "api",
        f"repos/{a['owner']}/{a['repo']}/commits",
        "--jq",
        ".[0:10] | .[].sha",
    ],
    "list_pull_requests": lambda a: [
        "gh",
        "pr",
        "list",
        "-R",
        f"{a['owner']}/{a['repo']}",
        "--json",
        "number,title,state",
        "-L",
        "10",
    ],
    "pull_request_read": lambda a: [
        "gh",
        "pr",
        "view",
        str(int(a["number"])),
        "-R",
        f"{a['owner']}/{a['repo']}",
        "--json",
        "title,body,state,files,additions,deletions",
    ],
    "list_issues": lambda a: [
        "gh",
        "issue",
        "list",
        "-R",
        f"{a['owner']}/{a['repo']}",
        "--json",
        "number,title,state",
        "-L",
        "10",
    ],
    "search_code": lambda a: [
        "gh",
        "search",
        "code",
        a.get("query", ""),
        "--json",
        "path,repository",
        "-L",
        "10",
    ],
    "list_branches": lambda a: [
        "gh",
        "api",
        f"repos/{a['owner']}/{a['repo']}/branches",
        "--jq",
        ".[].name",
    ],
    "get_file_contents": lambda a: [
        "gh",
        "api",
        f"repos/{a['owner']}/{a['repo']}/contents/{a.get('path', '')}",
    ],
    "get_commit": lambda a: [
        "gh",
        "api",
        f"repos/{a['owner']}/{a['repo']}/commits/{a.get('ref', 'HEAD')}",
    ],
}


class ResearchAgent:
    """Autonomously gathers context by calling MCPs in an agentic loop.

    Uses the BRAIN model (31B) to select which MCP tool to call next,
    executes it via MCPClientRegistry, and synthesizes results. FULL
    profile only, experimental, off by default.

    Prefers gh CLI over GitHub MCP when available — gh auto-resolves
    owner/repo from the working directory and uses existing auth.
    """

    def __init__(
        self,
        llm_manager: LLMManager | None = None,
        registry: Any = None,
        config: LLMConfig | None = None,
    ) -> None:
        self._llm = llm_manager
        self._registry = registry  # MCPClientRegistry
        self._config = config or LLMConfig()
        self._gh_path: str | None = shutil.which("gh")
        self._gh_authed: bool | None = None  # Lazy-checked on first github call

    async def research(
        self,
        intent: Intent,
        tool_recommendations: list[ToolRecommendation],
    ) -> ResearchResult | None:
        """Run the agentic research loop.

        Args:
            intent: Analyzed task intent.
            tool_recommendations: Available tools from ToolAdvisor.

        Returns:
            ResearchResult with synthesis, or None if BRAIN unavailable.
        """
        if not self._llm or not self._config.research_agent:
            return None

        if not self._registry:
            return None

        # Check BRAIN availability
        if not await self._is_brain_available():
            return None

        # Build tool descriptions — only include MCPs with safe read-only tools
        tool_descriptions = [
            {"mcp": r.mcp, "name": r.action, "description": r.reason}
            for r in tool_recommendations
            if r.mcp in RESEARCH_SAFE_TOOLS
        ]

        results: list[dict[str, Any]] = []
        total_tokens = 0

        # Agentic loop
        for iteration in range(MAX_ITERATIONS):
            if total_tokens >= MAX_TOKEN_BUDGET:
                logger.info("Research agent token budget exhausted at iteration %d", iteration)
                break

            # Select next tool
            tool_call = await self._select_tool(intent.original_prompt, tool_descriptions, results)
            if tool_call is None:
                logger.info("Research agent completed after %d iterations", iteration)
                break

            # Execute tool
            tool_result = await self._execute_tool(tool_call)
            if tool_result is not None:
                results.append(tool_result)
                total_tokens += tool_result.get("tokens", 0)

        if not results:
            return None

        # Synthesize results
        synthesis = await self._synthesize(intent.original_prompt, results)

        return ResearchResult(
            synthesis=synthesis or "",
            sources=[
                {
                    "mcp": r.get("mcp", ""),
                    "tool": r.get("tool", ""),
                    "summary": r.get("summary", ""),
                }
                for r in results
            ],
            tool_calls_made=len(results),
            tokens_used=total_tokens,
        )

    async def _is_brain_available(self) -> bool:
        """Check if BRAIN model is selectable via LLMManager's public API."""
        if not self._llm:
            return False
        return self._llm.is_role_available(ModelRole.BRAIN)

    async def _select_tool(
        self,
        task_description: str,
        tool_descriptions: list[dict[str, str]],
        previous_results: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Ask BRAIN to select the next tool to call.

        Args:
            task_description: Developer's task.
            tool_descriptions: Available tools.
            previous_results: Results so far.

        Returns:
            Tool call dict with mcp/name/arguments, or None if done.
        """
        if not self._llm:
            return None
        prompt = build_tool_selection_prompt(task_description, tool_descriptions, previous_results)

        try:
            result = await self._llm.generate_structured(
                ModelRole.BRAIN, prompt, TOOL_SELECTION_SCHEMA, **TOOL_SELECTION_SAMPLING
            )
            if not result:
                return None

            tool = result.get("tool")
            if tool is None:
                return None  # Research complete

            # Validate tool call has required fields
            if not tool.get("mcp") or not tool.get("name"):
                return None

            selected: dict[str, Any] = tool
            return selected
        except Exception:
            logger.warning("Tool selection failed", exc_info=True)
            return None

    async def _execute_tool(self, tool_call: dict[str, Any]) -> dict[str, Any] | None:
        """Execute a single MCP tool call after allowlist validation.

        Args:
            tool_call: Dict with mcp, name, arguments.

        Returns:
            Result dict with mcp, tool, data, summary; or None on failure.
        """
        mcp_name = tool_call.get("mcp", "")
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        # SECURITY: Enforce read-only allowlist — the LLM must not modify state
        safe_tools = RESEARCH_SAFE_TOOLS.get(mcp_name)
        if safe_tools is None or tool_name not in safe_tools:
            logger.warning("Research agent blocked disallowed tool: %s/%s", mcp_name, tool_name)
            return None

        # Prefer gh CLI over GitHub MCP when available
        if mcp_name == "github" and self._gh_path:
            gh_result = await self._execute_via_gh(tool_name, arguments)
            if gh_result is not None:
                return gh_result
            # Fall through to MCP if gh failed

        call = MCPToolCall(mcp_name=mcp_name, tool_name=tool_name, arguments=arguments)

        try:
            results = await self._registry.call_tools_parallel([call])
            if results and results[0].success:
                data = results[0].data
                summary = str(data)[:500] if data else ""
                return {
                    "mcp": mcp_name,
                    "tool": tool_name,
                    "data": data,
                    "summary": summary,
                    "tokens": len(summary) // 4,  # Rough token estimate
                }
            elif results:
                logger.debug("Tool %s/%s failed: %s", mcp_name, tool_name, results[0].error)
        except Exception:
            logger.warning("Tool execution failed: %s/%s", mcp_name, tool_name, exc_info=True)

        return None

    async def _execute_via_gh(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Execute a GitHub operation via gh CLI instead of MCP.

        Prefers gh CLI because it auto-resolves owner/repo from the working
        directory's git remote and uses the user's existing auth.

        Args:
            tool_name: GitHub MCP tool name (e.g. "list_commits").
            arguments: Arguments dict from the LLM.

        Returns:
            Result dict matching _execute_tool format, or None if gh failed.
        """
        # Check auth on first call (lazy, cached)
        if self._gh_authed is None:
            self._gh_authed = await self._check_gh_auth()
        if not self._gh_authed:
            return None

        # Look up the command builder
        builder = _GH_TOOL_COMMANDS.get(tool_name)
        if builder is None:
            return None  # No gh mapping for this tool — fall through to MCP

        # Validate owner/repo if present in arguments
        if not self._validate_gh_args(arguments):
            logger.warning("Invalid gh arguments for %s, falling back to MCP", tool_name)
            return None

        try:
            cmd = builder(arguments)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=15)

            if proc.returncode != 0:
                logger.debug(
                    "gh %s failed (exit %d): %s",
                    tool_name,
                    proc.returncode,
                    stderr_bytes.decode()[:200],
                )
                return None

            output = stdout_bytes.decode(errors="replace").strip()
            # Try to parse as JSON for structured data
            try:
                data = json.loads(output)
            except json.JSONDecodeError:
                data = output

            summary = str(data)[:500] if data else ""
            return {
                "mcp": "github",
                "tool": tool_name,
                "data": data,
                "summary": summary,
                "tokens": len(summary) // 4,
            }
        except TimeoutError:
            logger.debug("gh %s timed out", tool_name)
        except FileNotFoundError:
            logger.debug("gh binary not found")
            self._gh_path = None  # Don't try again
        except Exception:
            logger.debug("gh %s unexpected error", tool_name, exc_info=True)

        return None

    async def _check_gh_auth(self) -> bool:
        """Check if gh CLI is authenticated. Cached for the session."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "gh",
                "auth",
                "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=5)
            return proc.returncode == 0
        except (TimeoutError, FileNotFoundError, OSError):
            return False

    @staticmethod
    def _validate_gh_args(arguments: dict[str, Any]) -> bool:
        """Validate LLM-provided arguments before passing to gh CLI.

        Prevents injection via malicious owner/repo/path values.
        """
        owner = arguments.get("owner", "")
        repo = arguments.get("repo", "")
        if owner and not _OWNER_REPO_RE.match(str(owner)):
            return False
        if repo and not _OWNER_REPO_RE.match(str(repo)):
            return False
        # Path must not contain traversal
        path = arguments.get("path", "")
        if ".." in str(path):
            return False
        return True

    async def _synthesize(
        self,
        task_description: str,
        results: list[dict[str, Any]],
    ) -> str | None:
        """Synthesize research results into a concise summary.

        Args:
            task_description: Developer's task.
            results: All tool results to synthesize.

        Returns:
            Synthesis text, or None on failure.
        """
        if not self._llm:
            return None
        prompt = build_synthesis_prompt(task_description, results)

        try:
            response = await self._llm.generate(ModelRole.BRAIN, prompt, **SYNTHESIS_SAMPLING)
            if response and response.content:
                text: str = response.content.strip()
                return text
        except Exception:
            logger.warning("Synthesis failed", exc_info=True)

        return None
