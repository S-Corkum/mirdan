"""Pattern Matcher - Shared utility for weighted regex pattern matching."""

import re
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class PatternResult(Generic[T]):
    """Result of pattern matching.

    Attributes:
        best_match: The category with highest score, or None if no matches
        scores: Score for each category
        best_score: The highest score
        confidence: Confidence level based on score and margin
    """

    best_match: T | None
    scores: dict[T, int]
    best_score: int
    confidence: str  # "high", "medium", "low"


@dataclass
class CompiledPattern:
    """A compiled regex pattern with weight."""

    pattern: re.Pattern[str]
    weight: int
    use_multiline: bool = False


class PatternMatcher(Generic[T]):
    """Weighted pattern matching utility.

    Matches text against categorized patterns and returns the best matching
    category based on weighted scoring.

    Type parameter T is the type of categories (e.g., TaskType, str).
    """

    def __init__(
        self,
        patterns: dict[T, list[tuple[str, int]]],
        count_all_matches: bool = False,
        case_insensitive: bool = True,
        high_score_threshold: int = 8,
        high_margin_threshold: int = 3,
        medium_score_threshold: int = 4,
    ):
        """Initialize the pattern matcher.

        Args:
            patterns: Dict mapping category -> list of (pattern_str, weight)
            count_all_matches: If True, multiply weight by number of matches.
                               If False, each pattern can only contribute its weight once.
            case_insensitive: If True, patterns are case-insensitive
            high_score_threshold: Minimum score for "high" confidence
            high_margin_threshold: Minimum margin over second place for "high" confidence
            medium_score_threshold: Minimum score for "medium" confidence
        """
        self._count_all = count_all_matches
        self._case_insensitive = case_insensitive
        self._high_score = high_score_threshold
        self._high_margin = high_margin_threshold
        self._medium_score = medium_score_threshold
        self._categories = list(patterns.keys())
        self._compiled = self._compile_patterns(patterns)

    def _compile_patterns(
        self,
        patterns: dict[T, list[tuple[str, int]]],
    ) -> dict[T, list[CompiledPattern]]:
        """Compile regex patterns for each category."""
        compiled: dict[T, list[CompiledPattern]] = {}

        for category, pattern_list in patterns.items():
            compiled[category] = []
            for pattern_str, weight in pattern_list:
                flags = 0
                if self._case_insensitive:
                    flags |= re.IGNORECASE
                # Check if pattern expects multiline matching (starts with ^)
                use_multiline = pattern_str.startswith("^")
                if use_multiline:
                    flags |= re.MULTILINE

                compiled[category].append(
                    CompiledPattern(
                        pattern=re.compile(pattern_str, flags),
                        weight=weight,
                        use_multiline=use_multiline,
                    )
                )

        return compiled

    def match(self, text: str) -> PatternResult[T]:
        """Match text against all patterns and return result.

        Args:
            text: The text to match against

        Returns:
            PatternResult with best match, scores, and confidence
        """
        if not text or not text.strip():
            return PatternResult(
                best_match=None,
                scores=dict.fromkeys(self._categories, 0),
                best_score=0,
                confidence="low",
            )

        scores = self.score_all(text)
        best_category = max(scores, key=lambda k: scores[k])
        best_score = scores[best_category]

        if best_score == 0:
            return PatternResult(
                best_match=None,
                scores=scores,
                best_score=0,
                confidence="low",
            )

        # Calculate confidence based on score and margin
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]

        if best_score >= self._high_score and margin >= self._high_margin:
            confidence = "high"
        elif best_score >= self._medium_score:
            confidence = "medium"
        else:
            confidence = "low"

        return PatternResult(
            best_match=best_category,
            scores=scores,
            best_score=best_score,
            confidence=confidence,
        )

    def score_all(self, text: str) -> dict[T, int]:
        """Calculate scores for all categories.

        Args:
            text: The text to match against

        Returns:
            Dict mapping each category to its score
        """
        scores: dict[T, int] = dict.fromkeys(self._categories, 0)

        for category, patterns in self._compiled.items():
            for compiled in patterns:
                if self._count_all:
                    # Count all occurrences and multiply by weight
                    matches = len(compiled.pattern.findall(text))
                    scores[category] += matches * compiled.weight
                else:
                    # Binary match - pattern can only contribute once
                    if compiled.pattern.search(text):
                        scores[category] += compiled.weight

        return scores

    def best_match(self, text: str, default: T | None = None) -> T | None:
        """Get the best matching category.

        Args:
            text: The text to match against
            default: Value to return if no matches found

        Returns:
            The category with highest score, or default if no matches
        """
        result = self.match(text)
        return result.best_match if result.best_match is not None else default

    def matches_any(self, text: str, category: T) -> bool:
        """Check if text matches any pattern in a category.

        Args:
            text: The text to check
            category: The category to check patterns for

        Returns:
            True if any pattern in the category matches
        """
        if category not in self._compiled:
            return False

        return any(compiled.pattern.search(text) for compiled in self._compiled[category])
