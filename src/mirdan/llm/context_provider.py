"""Context provider for local LLM prompts.

Gathers project context from enyal (conventions, decisions) and context7
(framework documentation) and caches it per session. Every local LLM
prompt gets project-aware context at zero additional latency after the
first call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mirdan.core.client_registry import MCPClientRegistry
    from mirdan.core.session_manager import SessionManager

from mirdan.models import MCPToolCall

logger = logging.getLogger(__name__)

# Static framework hints — zero latency, always available.
# Covers key patterns that help the E4B make framework-aware decisions.
FRAMEWORK_HINTS: dict[str, str] = {
    "fastapi": (
        "FastAPI uses Depends() with Annotated for type-safe dependency injection. "
        "Route handlers should be async. Use HTTPException for error responses. "
        "Use Pydantic models for request/response validation."
    ),
    "pydantic": (
        "Pydantic v2 uses model_validator and field_validator decorators. "
        "Use model_dump() instead of dict(). Field() for metadata and constraints. "
        "BaseModel for data validation, not dataclasses."
    ),
    "django": (
        "Django uses ORM for database access. Views use request/response pattern. "
        "Use get_object_or_404 for safe lookups. CSRF protection is mandatory for forms. "
        "Settings in settings.py, URLs in urls.py."
    ),
    "flask": (
        "Flask uses decorators for routes. Use Blueprint for modular apps. "
        "request.json for JSON body parsing. abort() for HTTP errors."
    ),
    "react": (
        "React uses JSX for UI. useState/useEffect for state and side effects. "
        "Props are read-only. Use useCallback/useMemo for performance. "
        "Keys required in lists."
    ),
    "nextjs": (
        "Next.js uses file-based routing in app/ directory. Server Components by default. "
        "Use 'use client' directive for client components. generateMetadata for SEO."
    ),
    "pytest": (
        "pytest uses fixtures for setup/teardown via conftest.py. "
        "Use @pytest.mark.parametrize for test variants. "
        "assert statements for assertions, not self.assertEqual."
    ),
    "sqlalchemy": (
        "SQLAlchemy 2.0 uses select() for queries, not query(). "
        "Use Session.execute() with select(). Mapped classes with mapped_column(). "
        "Always use parameterized queries, never string interpolation."
    ),
    "express": (
        "Express uses middleware pattern. app.use() for middleware, "
        "router.get/post for routes. res.json() for JSON responses. "
        "next() to pass to next middleware."
    ),
    "typescript": (
        "TypeScript uses strict type checking. Use interface for object shapes, "
        "type for unions/intersections. Avoid any — use unknown for untyped values. "
        "Use as const for literal types."
    ),
}


class ContextProvider:
    """Gathers and caches project context for local LLM prompts.

    Combines enyal patterns (project conventions) with framework hints
    (static + cached context7 docs) into a formatted context string.
    Session-cached: first call queries MCPs, subsequent calls return instantly.
    """

    def __init__(
        self,
        registry: MCPClientRegistry | None = None,
        session_manager: SessionManager | None = None,
    ) -> None:
        self._registry = registry
        self._session_manager = session_manager

    async def get_context(
        self,
        language: str | None = None,
        frameworks: list[str] | None = None,
        session_id: str = "",
        file_path: str = "",
    ) -> str:
        """Get project context for a local LLM prompt.

        Checks session cache first. On cache miss, queries enyal for
        conventions and adds static framework hints. Caches the result
        on the session for all subsequent calls.

        Args:
            language: Detected programming language.
            frameworks: Detected frameworks.
            session_id: Session ID for caching.
            file_path: Current file path for enyal scope weighting.

        Returns:
            Formatted context string, or empty string if no context available.
        """
        # Check session cache
        if session_id and self._session_manager:
            session = self._session_manager.get(session_id)
            if session and session.context_cache_populated:
                return session.cached_project_context

        parts: list[str] = []

        # Enyal conventions (if available)
        enyal_context = await self._recall_enyal(language, frameworks, file_path)
        if enyal_context:
            parts.append("Project conventions:")
            parts.extend(f"- {p}" for p in enyal_context)

        # Framework hints (static, always available)
        framework_parts = self._get_framework_hints(frameworks or [])
        if framework_parts:
            parts.append("")
            parts.append("Framework reference:")
            parts.extend(framework_parts)

        # Context7 docs (if available and cached from enhance_prompt)
        context7_docs = await self._get_cached_context7(session_id)
        if context7_docs:
            parts.append("")
            parts.append("Library documentation:")
            parts.extend(f"- {d}" for d in context7_docs[:3])

        context = "\n".join(parts)

        # Cache on session
        if session_id and self._session_manager:
            session = self._session_manager.get(session_id)
            if session:
                session.cached_project_context = context
                session.context_cache_populated = True

        return context

    async def _recall_enyal(
        self,
        language: str | None,
        frameworks: list[str] | None,
        file_path: str,
    ) -> list[str]:
        """Query enyal for project conventions. Single recall call.

        Args:
            language: Programming language for query.
            frameworks: Frameworks for query.
            file_path: File path for scope weighting.

        Returns:
            List of convention strings, or empty list.
        """
        if not self._registry:
            return []

        # Build focused query
        query_parts = ["conventions", "patterns"]
        if language:
            query_parts.insert(0, language)
        if frameworks:
            query_parts.extend(frameworks[:2])
        query = " ".join(query_parts)

        arguments: dict[str, Any] = {
            "input": {
                "query": query,
                "limit": 5,
                "min_confidence": 0.3,
            }
        }
        if file_path:
            arguments["input"]["file_path"] = file_path

        try:
            call = MCPToolCall(mcp_name="enyal", tool_name="enyal_recall", arguments=arguments)
            results = await self._registry.call_tools_parallel([call])
            if results and results[0].success and results[0].data:
                return self._parse_enyal_results(results[0].data)
        except Exception:
            logger.debug("Enyal recall for context provider failed")

        return []

    @staticmethod
    def _parse_enyal_results(data: Any) -> list[str]:
        """Parse enyal recall results into convention strings."""
        import json

        patterns: list[str] = []
        try:
            if isinstance(data, str):
                parsed = json.loads(data)
            elif isinstance(data, dict):
                parsed = data
            else:
                return [str(data)[:200]]

            results = parsed.get("results", [])
            for entry in results[:5]:
                if isinstance(entry, dict):
                    content = entry.get("content", "")
                    if content:
                        patterns.append(str(content)[:200])
                elif isinstance(entry, str):
                    patterns.append(entry[:200])
        except (json.JSONDecodeError, TypeError):
            if isinstance(data, str) and data.strip():
                for line in data.strip().splitlines()[:5]:
                    if line.strip():
                        patterns.append(line.strip()[:200])

        return patterns

    @staticmethod
    def _get_framework_hints(frameworks: list[str]) -> list[str]:
        """Get static framework hints for detected frameworks."""
        hints: list[str] = []
        for fw in frameworks[:3]:
            fw_lower = fw.lower()
            hint = FRAMEWORK_HINTS.get(fw_lower)
            if hint:
                hints.append(f"- [{fw}] {hint}")
        return hints

    async def _get_cached_context7(self, session_id: str) -> list[str]:
        """Get context7 docs cached from a prior enhance_prompt call.

        Context7 docs are gathered by ContextAggregator during enhance_prompt
        and stored in ContextBundle.documentation_hints. If the session has
        already run enhance_prompt, we can reuse those docs.

        This avoids making additional context7 MCP calls during validation.
        """
        # Context7 docs would need to be cached on the session by enhance_prompt.
        # For now, return empty — the session cache from enhance_prompt's
        # ContextBundle is not yet threaded through. The framework hints
        # provide the static equivalent.
        return []
