"""Mirdan MCP Server - AI Code Quality Orchestrator."""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP

from mirdan.providers import ComponentProvider

logger = logging.getLogger(__name__)

# Strong references to fire-and-forget background tasks.
_background_tasks: set[asyncio.Task[Any]] = set()

# Tool priority order for budget-aware filtering.
_TOOL_PRIORITY = [
    "validate_code_quality",
    "validate_quick",
    "enhance_prompt",
    "get_quality_standards",
    "get_quality_trends",
    "scan_dependencies",
    "scan_conventions",
]


_provider: ComponentProvider | None = None


def _get_provider() -> ComponentProvider:
    """Get or create the singleton component provider."""
    global _provider
    if _provider is not None:
        return _provider
    _provider = ComponentProvider()
    return _provider


@asynccontextmanager
async def _lifespan(app: FastMCP[Any]) -> AsyncIterator[None]:
    """Manage server lifecycle: startup initialization and shutdown cleanup."""
    _get_provider()

    budget_str = os.environ.get("MIRDAN_TOOL_BUDGET")
    if budget_str is not None and budget_str != "":
        try:
            budget = int(budget_str)
        except ValueError:
            budget = -1
        if budget >= 0:
            keep = set(_TOOL_PRIORITY[:budget])
            to_remove = [name for name in list(app._tool_manager._tools) if name not in keep]
            for name in to_remove:
                del app._tool_manager._tools[name]

    yield
    if _provider is not None:
        await _provider.close()


mcp = FastMCP("Mirdan", instructions="AI Code Quality Orchestrator", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Tool handlers — thin routing to use cases
# ---------------------------------------------------------------------------


@mcp.tool()
async def enhance_prompt(
    prompt: str,
    task_type: str = "auto",
    context_level: str = "auto",
    max_tokens: int = 0,
    model_tier: str = "auto",
    session_id: str = "",
) -> dict[str, Any]:
    """
    Automatically enhance a coding prompt with quality requirements,
    codebase context, and tool recommendations.

    Args:
        prompt: The original developer prompt
        task_type: Override auto-detection (generation|refactor|debug|review|test|planning|auto)
                   Use "analyze_only" to return just intent analysis without enhancement.
                   Use "plan_validation" to validate a plan for implementation quality.
        context_level: How much context to gather (minimal|auto|comprehensive|none)
        max_tokens: Maximum token budget for the response (0=unlimited). When set, output
                    is automatically compressed: <=1000 tokens produces minimal output,
                    <=4000 produces compact output.
        model_tier: Target model tier for output optimization (auto|opus|sonnet|haiku).
                    Haiku/Sonnet receive more compressed output.
        session_id: Resume an existing session to thread validation feedback into this
                    prompt. Persistent violations from prior validate_code_quality calls
                    are injected as priority quality requirements.

    Returns:
        Enhanced prompt with quality requirements and tool recommendations
    """
    p = _get_provider()
    uc = p.create_enhance_prompt_usecase(_background_tasks)
    return await uc.execute(
        prompt=prompt,
        task_type=task_type,
        context_level=context_level,
        max_tokens=max_tokens,
        model_tier=model_tier,
        session_id=session_id,
    )


@mcp.tool()
async def validate_code_quality(
    code: str,
    language: str = "auto",
    check_security: bool = True,
    check_architecture: bool = True,
    check_style: bool = True,
    severity_threshold: str = "warning",
    session_id: str = "",
    max_tokens: int = 0,
    model_tier: str = "auto",
    input_type: str = "code",
    compare: bool = False,
    file_path: str = "",
) -> dict[str, Any]:
    """
    Validate generated code against quality standards.

    Args:
        code: The code to validate
        language: Programming language (python|typescript|javascript|rust|go|auto)
        check_security: Validate against security standards
        check_architecture: Validate against architecture standards
        check_style: Validate against language-specific style standards
        severity_threshold: Minimum severity to include in results (error|warning|info)
        session_id: Session ID from enhance_prompt to auto-inherit language and security settings
        max_tokens: Maximum token budget for the response (0=unlimited)
        model_tier: Target model tier for output optimization (auto|opus|sonnet|haiku)
        input_type: Input type - "code" for raw code (default), "diff" for unified
                   diff format (git diff output). When "diff", only added lines are
                   validated and violation line numbers map back to original file locations
        compare: If True, treat `code` as JSON array of implementations to compare
        file_path: Optional file path for external linter analysis. When provided,
                   runs ruff/eslint/mypy on the file and merges results.

    Returns:
        Validation results with pass/fail, score, violations, and summary
    """
    p = _get_provider()
    uc = p.create_validate_code_usecase(_background_tasks)
    return await uc.execute(
        code=code,
        language=language,
        check_security=check_security,
        check_architecture=check_architecture,
        check_style=check_style,
        severity_threshold=severity_threshold,
        session_id=session_id,
        max_tokens=max_tokens,
        model_tier=model_tier,
        input_type=input_type,
        compare=compare,
        file_path=file_path,
    )


@mcp.tool()
async def validate_quick(
    code: str,
    language: str = "auto",
    max_tokens: int = 0,
    model_tier: str = "auto",
) -> dict[str, Any]:
    """Fast security-only validation for hooks and real-time feedback (<500ms target).

    Runs only security rules — skips style, architecture, framework, and custom checks.
    Ideal for PostToolUse hooks where speed matters more than comprehensive analysis.

    Args:
        code: The code to validate
        language: Programming language (python|typescript|javascript|rust|go|auto)
        max_tokens: Maximum token budget for the response (0=unlimited)
        model_tier: Target model tier for output optimization (auto|opus|sonnet|haiku)

    Returns:
        Validation results with pass/fail, score, and security-only violations
    """
    p = _get_provider()
    uc = p.create_validate_quick_usecase()
    return await uc.execute(
        code=code,
        language=language,
        max_tokens=max_tokens,
        model_tier=model_tier,
    )


@mcp.tool()
async def get_quality_standards(
    language: str,
    framework: str = "",
    category: str = "all",
) -> dict[str, Any]:
    """
    Retrieve quality standards for a language/framework combination.

    Args:
        language: Programming language (typescript, python, etc.)
        framework: Optional framework (react, fastapi, etc.)
        category: Filter to specific category (security|architecture|style|all)

    Returns:
        Quality standards for the specified language/framework
    """
    p = _get_provider()
    uc = p.create_get_quality_standards_usecase()
    return await uc.execute(language=language, framework=framework, category=category)


@mcp.tool()
async def get_quality_trends(
    project_path: str = "",
    days: int = 30,
    format: str = "",
) -> dict[str, Any]:
    """
    Get quality score trends over time from stored validation history.

    Args:
        project_path: Optional project path filter
        days: Number of days of history to analyze (default: 30)
        format: Output format — empty for default, "dashboard" for MCP Apps data

    Returns:
        Quality trend data with scores, pass rate, and trend direction
    """
    p = _get_provider()
    uc = p.create_get_quality_trends_usecase()
    return await uc.execute(project_path=project_path, days=days, format=format)


@mcp.tool()
async def scan_conventions(
    directory: str = ".",
    language: str = "auto",
) -> dict[str, Any]:
    """Scan a codebase to discover implicit conventions and patterns.

    Args:
        directory: Directory to scan (default: current directory)
        language: Language filter or "auto" to detect

    Returns:
        Scan result with discovered conventions and quality baselines
    """
    p = _get_provider()
    uc = p.create_scan_conventions_usecase()
    return await uc.execute(directory=directory, language=language)


@mcp.tool()
async def scan_dependencies(
    project_path: str = ".",
    ecosystem: str = "auto",
) -> dict[str, Any]:
    """Scan project dependencies for known vulnerabilities.

    Args:
        project_path: Project directory containing dependency manifests
        ecosystem: Filter by ecosystem (auto|PyPI|npm|crates.io|Go|Maven)

    Returns:
        Scan results with packages checked and vulnerabilities found
    """
    p = _get_provider()
    uc = p.create_scan_dependencies_usecase()
    return await uc.execute(project_path=project_path, ecosystem=ecosystem)


def main() -> None:
    """Run the Mirdan MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
