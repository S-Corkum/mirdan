"""PERF003: Synchronous blocking in async context detection rule."""

from __future__ import annotations

import re

from mirdan.core.rules.base import BaseRule, RuleContext, RuleTier
from mirdan.core.skip_regions import is_in_skip_region
from mirdan.models import Violation

# Async function declarations per language
_ASYNC_FUNC = re.compile(
    r"(?:async\s+(?:def|function|fn)\s+\w+|async\s+\w+\s*\(|async\s+\(\s*\)\s*=>|async\s+Task[<\s])",
    re.MULTILINE,
)

# Blocking calls that should not appear in async contexts
_BLOCKING_CALLS: list[tuple[re.Pattern[str], str, str]] = [
    # Python
    (
        re.compile(r"\btime\.sleep\s*\("),
        "time.sleep()",
        "Use 'await asyncio.sleep()' instead",
    ),
    (
        re.compile(r"\brequests\.(?:get|post|put|delete|patch|head)\s*\("),
        "requests (sync HTTP)",
        "Use httpx.AsyncClient or aiohttp for async HTTP calls",
    ),
    (
        re.compile(r"\bopen\s*\([^)]*\)\.read"),
        "synchronous file read",
        "Use aiofiles for async file I/O",
    ),
    # TypeScript/JavaScript
    (
        re.compile(r"\bfs\.readFileSync\s*\("),
        "fs.readFileSync()",
        "Use fs.promises.readFile() or fs/promises",
    ),
    (
        re.compile(r"\bfs\.writeFileSync\s*\("),
        "fs.writeFileSync()",
        "Use fs.promises.writeFile() or fs/promises",
    ),
    (
        re.compile(r"\bchild_process\.execSync\s*\("),
        "execSync()",
        "Use child_process.exec() with promisify or execa",
    ),
    # Java
    (
        re.compile(r"\bThread\.sleep\s*\("),
        "Thread.sleep()",
        "Use CompletableFuture.delayedExecutor() or virtual thread scheduling",
    ),
    # C#
    (
        re.compile(r"\bThread\.Sleep\s*\("),
        "Thread.Sleep()",
        "Use 'await Task.Delay()' instead",
    ),
    (
        re.compile(r"\.Result\b"),
        ".Result (sync wait on Task)",
        "Use 'await' instead of .Result to avoid deadlocks",
    ),
    (
        re.compile(r"\.GetAwaiter\(\)\.GetResult\(\)"),
        ".GetAwaiter().GetResult()",
        "Use 'await' instead to avoid thread pool starvation",
    ),
    # Rust
    (
        re.compile(r"std::thread::sleep\s*\("),
        "std::thread::sleep()",
        "Use tokio::time::sleep() in async context",
    ),
]


class PERF003SyncInAsyncRule(BaseRule):
    """Detect synchronous blocking calls inside async functions.

    Known limitation: uses a 50-line window heuristic to approximate
    function body. Short async functions followed by sync functions
    may produce false positives.
    """

    @property
    def id(self) -> str:
        return "PERF003"

    @property
    def name(self) -> str:
        return "sync-blocking-in-async"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "typescript", "javascript", "java", "csharp", "rust", "auto"})

    @property
    def tier(self) -> RuleTier:
        return RuleTier.FULL

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        """Detect synchronous blocking calls inside async functions."""
        violations: list[Violation] = []
        lines = code.split("\n")

        for async_match in _ASYNC_FUNC.finditer(code):
            if is_in_skip_region(async_match.start(), context.skip_regions):
                continue

            async_line = code[: async_match.start()].count("\n")
            # Search within the async function body (next 50 lines as heuristic)
            func_end = min(async_line + 50, len(lines))
            func_body = "\n".join(lines[async_line:func_end])

            for block_pattern, block_desc, block_fix in _BLOCKING_CALLS:
                for block_match in block_pattern.finditer(func_body):
                    block_offset = func_body[: block_match.start()].count("\n")
                    violations.append(
                        Violation(
                            id="PERF003",
                            rule="sync-blocking-in-async",
                            category="performance",
                            severity="warning",
                            message=(
                                f"Synchronous blocking call {block_desc} in async function. "
                                "This blocks the event loop/thread pool."
                            ),
                            line=async_line + block_offset + 1,
                            suggestion=block_fix,
                        )
                    )

        return violations
