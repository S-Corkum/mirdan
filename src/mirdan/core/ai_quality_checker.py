"""AI-specific and test quality checks.

Detects common AI-generated code issues that traditional linters miss:
- AI001: Placeholder code (raise NotImplementedError, pass with TODO)
- AI002: Hallucinated imports (modules not in stdlib or project deps)
- AI003: Over-engineering detection (unnecessary abstractions)
- AI004: Duplicate code block detection
- AI005: Inconsistent error handling patterns
- AI006: Unnecessary heavy imports
- AI007: Security theater detection
- AI008: Injection vulnerabilities via f-string interpolation
- TEST001-TEST010: Test quality anti-pattern detection
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from mirdan.core.rules.ai001_placeholders import (
    AI001PlaceholderRule,
    get_ast_confirmed_placeholder_lines,
)
from mirdan.core.rules.ai002_imports import PYTHON_STDLIB_MODULES, AI002ImportRule
from mirdan.core.rules.ai003_over_engineering import AI003OverEngineeringRule
from mirdan.core.rules.ai004_duplicate_blocks import AI004DuplicateBlocksRule
from mirdan.core.rules.ai005_error_handling import AI005ErrorHandlingRule
from mirdan.core.rules.ai006_heavy_imports import AI006HeavyImportsRule
from mirdan.core.rules.ai007_security_theater import AI007SecurityTheaterRule
from mirdan.core.rules.ai008_injection import AI008InjectionRule
from mirdan.core.rules.base import RuleContext, RuleRegistry, RuleTier
from mirdan.core.rules.deep_analysis_rules import (
    DEEP001SwallowedExceptionRule,
    DEEP004LostExceptionContextRule,
)
from mirdan.core.rules.perf001_n_plus_one import PERF001NPlusOneRule
from mirdan.core.rules.perf002_unbounded_collection import PERF002UnboundedCollectionRule
from mirdan.core.rules.perf003_sync_in_async import PERF003SyncInAsyncRule
from mirdan.core.rules.perf004_missing_pagination import PERF004MissingPaginationRule
from mirdan.core.rules.perf005_repeated_computation import PERF005RepeatedComputationRule
from mirdan.core.rules.rs003_undocumented_unsafe import RS003UndocumentedUnsafeRule
from mirdan.core.rules.sec014_vulnerable_deps import SEC014VulnerableDepsRule
from mirdan.core.rules.test_body_rules import (
    TEST001EmptyTestRule,
    TEST002AssertTrueRule,
    TEST003NoAssertionsRule,
    TEST005MockAbuseRule,
    TEST010BroadExceptionRule,
)
from mirdan.core.rules.test_structure_rules import (
    TEST004NoCoverageRule,
    TEST006DuplicateTestRule,
    TEST007MissingEdgeCaseRule,
    TEST008HardcodedDataRule,
    TEST009ExecutionOrderRule,
)
from mirdan.core.skip_regions import build_skip_regions
from mirdan.models import Violation

if TYPE_CHECKING:
    from mirdan.core.manifest_parser import ManifestParser
    from mirdan.core.vuln_scanner import VulnScanner

# Re-export for backward compatibility
__all__ = ["PYTHON_STDLIB_MODULES", "AIQualityChecker"]


class AIQualityChecker:
    """Detects AI-specific code quality issues.

    Thin facade over the rule registry. Each rule is implemented in its
    own module under ``mirdan.core.rules/``.
    """

    def __init__(
        self,
        project_dir: Path | None = None,
        manifest_parser: ManifestParser | None = None,
        vuln_scanner: VulnScanner | None = None,
        workspace_resolver: Any | None = None,
    ) -> None:
        self._registry = RuleRegistry()

        # Register all rules
        self._ai001 = AI001PlaceholderRule()
        self._ai002 = AI002ImportRule(
            manifest_parser=manifest_parser,
            workspace_resolver=workspace_resolver,
            project_dir=project_dir,
        )
        self._registry.register(self._ai001)
        self._registry.register(self._ai002)
        self._registry.register(AI003OverEngineeringRule())
        self._registry.register(AI004DuplicateBlocksRule())
        self._registry.register(AI005ErrorHandlingRule())
        self._registry.register(AI006HeavyImportsRule())
        self._registry.register(AI007SecurityTheaterRule())
        self._registry.register(AI008InjectionRule())
        self._registry.register(
            SEC014VulnerableDepsRule(
                manifest_parser=manifest_parser,
                vuln_scanner=vuln_scanner,
            )
        )

        # TEST rules
        self._registry.register(TEST001EmptyTestRule())
        self._registry.register(TEST002AssertTrueRule())
        self._registry.register(TEST003NoAssertionsRule())
        self._registry.register(TEST004NoCoverageRule())
        self._registry.register(TEST005MockAbuseRule())
        self._registry.register(TEST006DuplicateTestRule())
        self._registry.register(TEST007MissingEdgeCaseRule())
        self._registry.register(TEST008HardcodedDataRule())
        self._registry.register(TEST009ExecutionOrderRule())
        self._registry.register(TEST010BroadExceptionRule())

        # DEEP analysis rules
        self._registry.register(DEEP001SwallowedExceptionRule())
        self._registry.register(DEEP004LostExceptionContextRule())

        # RS003 BaseRule (undocumented unsafe blocks)
        self._registry.register(RS003UndocumentedUnsafeRule())

        # PERF rules
        self._registry.register(PERF001NPlusOneRule())
        self._registry.register(PERF002UnboundedCollectionRule())
        self._registry.register(PERF003SyncInAsyncRule())
        self._registry.register(PERF004MissingPaginationRule())
        self._registry.register(PERF005RepeatedComputationRule())

    def check(
        self,
        code: str,
        language: str,
        file_path: str = "",
        is_test: bool = False,
        implementation_code: str | None = None,
        max_tier: RuleTier = RuleTier.FULL,
    ) -> list[Violation]:
        """Run AI-specific and test quality rules.

        Args:
            code: Source code to check.
            language: Detected programming language.
            file_path: Optional file path for workspace-aware AI002 resolution.
            is_test: Whether the code is test code (enables TEST rules).
            implementation_code: Source of implementation for cross-referencing.
            max_tier: Maximum rule tier to include (for incremental validation).

        Returns:
            List of violations.
        """
        if not code or not code.strip():
            return []

        skip_regions = build_skip_regions(code, language)
        context = RuleContext(
            skip_regions=skip_regions,
            file_path=file_path,
            is_test=is_test,
            implementation_code=implementation_code,
        )

        violations = self._registry.check_by_tier(code, language, context, max_tier=max_tier)

        # Mark verifiability: AST-confirmed placeholders are verifiable
        ast_confirmed = get_ast_confirmed_placeholder_lines(code, language)
        for v in violations:
            if v.category == "ai_quality":
                if v.id == "AI001" and v.line in ast_confirmed:
                    v.verifiable = True
                else:
                    v.verifiable = False

        return violations

    def check_quick(self, code: str, language: str, file_path: str = "") -> list[Violation]:
        """Run only critical AI rules (AI001, AI007, AI008) and SEC014.

        Args:
            code: Source code to check.
            language: Detected programming language.
            file_path: Optional file path (unused, kept for API consistency).

        Returns:
            List of critical AI violations.
        """
        if not code or not code.strip():
            return []

        skip_regions = build_skip_regions(code, language)
        context = RuleContext(
            skip_regions=skip_regions,
            file_path=file_path,
        )

        violations = self._registry.check_quick(code, language, context)

        # Same verifiability logic as check()
        ast_confirmed = get_ast_confirmed_placeholder_lines(code, language)
        for v in violations:
            if v.category == "ai_quality":
                if v.id == "AI001" and v.line in ast_confirmed:
                    v.verifiable = True
                else:
                    v.verifiable = False

        return violations
