"""Base classes and registry for AI quality rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from mirdan.models import Violation


@dataclass
class RuleContext:
    """Context passed to each rule during checking."""

    skip_regions: list[int]
    project_deps: frozenset[str] | None = None
    file_path: str = ""


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
