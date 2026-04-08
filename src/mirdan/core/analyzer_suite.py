"""AnalyzerSuite — groups ceremony-gated optional analyzers.

Reduces constructor parameter count in use-cases by bundling the four
optional analyzers that share the ceremony-level gate pattern. Each
analyzer is independently optional (None means disabled).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mirdan.core.architecture_analyzer import ArchitectureAnalyzer
    from mirdan.core.decision_analyzer import DecisionAnalyzer
    from mirdan.core.guardrail_analyzer import GuardrailAnalyzer
    from mirdan.core.tidy_first import TidyFirstAnalyzer


@dataclass(frozen=True)
class AnalyzerSuite:
    """Ceremony-gated analyzers for enhance_prompt enrichment.

    Groups the four optional analyzers that run at STANDARD+ ceremony level.
    Immutable (frozen) — set once at composition time in providers.py.

    Each field is independently optional. A None field means that analyzer
    is disabled. The suite itself can also be None in use-case constructors
    to indicate no analyzers are available.
    """

    tidy_first: TidyFirstAnalyzer | None = None
    decision: DecisionAnalyzer | None = None
    guardrail: GuardrailAnalyzer | None = None
    architecture: ArchitectureAnalyzer | None = None
