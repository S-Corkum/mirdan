"""AI-specific code quality checks.

Detects common AI-generated code issues that traditional linters miss:
- AI001: Placeholder code (raise NotImplementedError, pass with TODO)
- AI002: Hallucinated imports (modules not in stdlib or project deps)
- AI003: Over-engineering detection (unnecessary abstractions)
- AI004: Duplicate code block detection
- AI005: Inconsistent error handling patterns
- AI006: Unnecessary heavy imports
- AI007: Security theater detection
- AI008: Injection vulnerabilities via f-string interpolation
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from mirdan.core.rules.ai001_placeholders import (
    AI001PlaceholderRule,
    get_ast_confirmed_placeholder_lines,
)
from mirdan.core.rules.ai002_imports import AI002ImportRule, PYTHON_STDLIB_MODULES
from mirdan.core.rules.ai003_over_engineering import AI003OverEngineeringRule
from mirdan.core.rules.ai004_duplicate_blocks import AI004DuplicateBlocksRule
from mirdan.core.rules.ai005_error_handling import AI005ErrorHandlingRule
from mirdan.core.rules.ai006_heavy_imports import AI006HeavyImportsRule
from mirdan.core.rules.ai007_security_theater import AI007SecurityTheaterRule
from mirdan.core.rules.ai008_injection import AI008InjectionRule
from mirdan.core.rules.base import RuleContext, RuleRegistry
from mirdan.core.rules.sec014_vulnerable_deps import SEC014VulnerableDepsRule
from mirdan.core.skip_regions import build_skip_regions
from mirdan.models import Violation

if TYPE_CHECKING:
    from mirdan.core.manifest_parser import ManifestParser
    from mirdan.core.vuln_scanner import VulnScanner

# Re-export for backward compatibility
__all__ = ["AIQualityChecker", "PYTHON_STDLIB_MODULES"]


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

    def check(self, code: str, language: str, file_path: str = "") -> list[Violation]:
        """Run all AI-specific rules.

        Args:
            code: Source code to check.
            language: Detected programming language.
            file_path: Optional file path for workspace-aware AI002 resolution.

        Returns:
            List of AI-specific violations.
        """
        if not code or not code.strip():
            return []

        skip_regions = build_skip_regions(code, language)
        context = RuleContext(
            skip_regions=skip_regions,
            file_path=file_path,
        )

        violations = self._registry.check_all(code, language, context)

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
