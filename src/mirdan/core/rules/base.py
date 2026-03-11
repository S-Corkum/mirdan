"""Base classes and registry for AI quality rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol, runtime_checkable

from mirdan.models import Violation


class RuleTier(IntEnum):
    """Performance tier for rules. Controls which rules run in each validation scope.

    QUICK: Security-critical, <10ms. Runs in validate_quick scope="security".
    ESSENTIAL: Fast pattern checks, <50ms. Runs in validate_quick scope="essential".
    FULL: All checks including AST analysis. Runs in validate_code_quality.
    """

    QUICK = 0
    ESSENTIAL = 1
    FULL = 2


@dataclass
class RuleContext:
    """Context passed to each rule during checking."""

    skip_regions: list[int]
    project_deps: frozenset[str] | None = None
    file_path: str = ""
    is_test: bool = False
    implementation_code: str | None = None
    changed_lines: frozenset[int] | None = None


@runtime_checkable
class QualityRule(Protocol):
    """Protocol for quality rules."""

    @property
    def id(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def languages(self) -> frozenset[str]: ...

    @property
    def is_quick(self) -> bool: ...

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]: ...


class BaseRule(ABC):
    """Abstract base class for quality rules."""

    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def languages(self) -> frozenset[str]: ...

    @property
    def is_quick(self) -> bool:
        return False

    @property
    def tier(self) -> RuleTier:
        """Performance tier. Defaults based on is_quick for backward compatibility."""
        return RuleTier.QUICK if self.is_quick else RuleTier.FULL

    @abstractmethod
    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]: ...


@dataclass
class RuleRegistry:
    """Registry that manages and dispatches quality rules."""

    _rules: list[QualityRule] = field(default_factory=list)

    def register(self, rule: QualityRule) -> None:
        """Register a quality rule."""
        self._rules.append(rule)

    def check_all(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        """Run all applicable rules for the given language."""
        violations: list[Violation] = []
        lang = language.lower()
        for rule in self._rules:
            if lang in rule.languages or "auto" in rule.languages:
                violations.extend(rule.check(code, language, context))
        return violations

    def check_quick(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        """Run only quick (critical) rules for the given language."""
        violations: list[Violation] = []
        lang = language.lower()
        for rule in self._rules:
            if not rule.is_quick:
                continue
            if lang in rule.languages or "auto" in rule.languages:
                violations.extend(rule.check(code, language, context))
        return violations

    def check_by_tier(
        self,
        code: str,
        language: str,
        context: RuleContext,
        max_tier: RuleTier = RuleTier.FULL,
    ) -> list[Violation]:
        """Run rules up to the specified tier.

        Tier filtering happens here. Changed-lines filtering is NOT done here —
        it is applied once in CodeValidator.validate()/validate_quick() after all
        rule sources (compiled + registry) have been collected, to avoid
        double-filtering and to ensure consistent behavior across both rule systems.

        Args:
            code: Source code to check.
            language: Programming language.
            context: Rule context.
            max_tier: Maximum tier to include. Rules with tier > max_tier are skipped.

        Returns:
            List of violations from rules at or below max_tier.
        """
        violations: list[Violation] = []
        lang = language.lower()
        for rule in self._rules:
            if getattr(rule, "tier", RuleTier.FULL) > max_tier:
                continue
            if lang in rule.languages or "auto" in rule.languages:
                violations.extend(rule.check(code, language, context))
        return violations
