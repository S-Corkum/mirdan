"""Tests for the PatternMatcher utility module."""

import pytest

from mirdan.core.pattern_matcher import PatternMatcher, PatternResult


class TestPatternResult:
    """Tests for PatternResult dataclass."""

    def test_pattern_result_creation(self) -> None:
        """Should create PatternResult with all fields."""
        result = PatternResult(
            best_match="test",
            scores={"test": 5, "other": 2},
            best_score=5,
            confidence="high",
        )
        assert result.best_match == "test"
        assert result.scores == {"test": 5, "other": 2}
        assert result.best_score == 5
        assert result.confidence == "high"

    def test_pattern_result_with_none_match(self) -> None:
        """Should allow None as best_match."""
        result = PatternResult(
            best_match=None,
            scores={"a": 0, "b": 0},
            best_score=0,
            confidence="low",
        )
        assert result.best_match is None


class TestPatternMatcher:
    """Tests for PatternMatcher class."""

    @pytest.fixture
    def simple_matcher(self) -> PatternMatcher[str]:
        """Create a simple matcher for testing."""
        patterns = {
            "greeting": [
                (r"\bhello\b", 2),
                (r"\bhi\b", 1),
                (r"\bwelcome\b", 2),
            ],
            "farewell": [
                (r"\bgoodbye\b", 2),
                (r"\bbye\b", 1),
                (r"\bsee you\b", 2),
            ],
        }
        return PatternMatcher(patterns)

    @pytest.fixture
    def weighted_matcher(self) -> PatternMatcher[str]:
        """Create a matcher with varied weights."""
        patterns = {
            "high_priority": [
                (r"\bcritical\b", 10),
                (r"\burgent\b", 8),
            ],
            "low_priority": [
                (r"\bminor\b", 2),
                (r"\btrivial\b", 1),
            ],
        }
        return PatternMatcher(patterns)

    def test_basic_matching(self, simple_matcher: PatternMatcher[str]) -> None:
        """Should match basic patterns correctly."""
        result = simple_matcher.match("hello there")
        assert result.best_match == "greeting"
        assert result.best_score == 2

    def test_multiple_matches_in_category(self, simple_matcher: PatternMatcher[str]) -> None:
        """Should accumulate scores for multiple matches in same category."""
        result = simple_matcher.match("hello and hi everyone")
        assert result.best_match == "greeting"
        assert result.scores["greeting"] == 3  # 2 + 1

    def test_no_matches(self, simple_matcher: PatternMatcher[str]) -> None:
        """Should return None when no patterns match."""
        result = simple_matcher.match("random text here")
        assert result.best_match is None
        assert result.best_score == 0
        assert result.confidence == "low"

    def test_empty_text(self, simple_matcher: PatternMatcher[str]) -> None:
        """Should handle empty text gracefully."""
        result = simple_matcher.match("")
        assert result.best_match is None
        assert result.best_score == 0

    def test_whitespace_only_text(self, simple_matcher: PatternMatcher[str]) -> None:
        """Should handle whitespace-only text."""
        result = simple_matcher.match("   \n\t  ")
        assert result.best_match is None

    def test_score_all(self, simple_matcher: PatternMatcher[str]) -> None:
        """Should return scores for all categories."""
        scores = simple_matcher.score_all("hello and goodbye")
        assert scores["greeting"] == 2
        assert scores["farewell"] == 2

    def test_best_match_method(self, simple_matcher: PatternMatcher[str]) -> None:
        """Should return best matching category."""
        match = simple_matcher.best_match("hello world")
        assert match == "greeting"

    def test_best_match_with_default(self, simple_matcher: PatternMatcher[str]) -> None:
        """Should return default when no matches."""
        match = simple_matcher.best_match("random text", default="unknown")
        assert match == "unknown"

    def test_matches_any(self, simple_matcher: PatternMatcher[str]) -> None:
        """Should check if any pattern in category matches."""
        assert simple_matcher.matches_any("hello", "greeting") is True
        assert simple_matcher.matches_any("hello", "farewell") is False
        assert simple_matcher.matches_any("random", "greeting") is False

    def test_matches_any_unknown_category(self, simple_matcher: PatternMatcher[str]) -> None:
        """Should return False for unknown category."""
        assert simple_matcher.matches_any("hello", "unknown") is False

    def test_weighted_scoring(self, weighted_matcher: PatternMatcher[str]) -> None:
        """Should weight patterns correctly."""
        result = weighted_matcher.match("this is critical")
        assert result.best_match == "high_priority"
        assert result.best_score == 10

    def test_weight_beats_count(self, weighted_matcher: PatternMatcher[str]) -> None:
        """Higher weight should beat multiple low-weight matches."""
        result = weighted_matcher.match("minor trivial minor")
        # In default mode (count_all=False), each pattern contributes once
        assert result.scores["low_priority"] == 3  # 2 + 1

        result2 = weighted_matcher.match("critical")
        assert result2.best_score == 10
        assert result2.scores["high_priority"] > result.scores["low_priority"]


class TestPatternMatcherConfiguration:
    """Tests for PatternMatcher configuration options."""

    def test_case_insensitive_default(self) -> None:
        """Should be case insensitive by default."""
        matcher = PatternMatcher({"test": [(r"\bhello\b", 1)]})
        assert matcher.best_match("HELLO") == "test"
        assert matcher.best_match("Hello") == "test"
        assert matcher.best_match("hello") == "test"

    def test_case_sensitive_option(self) -> None:
        """Should respect case_insensitive=False."""
        matcher = PatternMatcher({"test": [(r"\bhello\b", 1)]}, case_insensitive=False)
        assert matcher.best_match("hello") == "test"
        assert matcher.best_match("HELLO") is None
        assert matcher.best_match("Hello") is None

    def test_count_all_matches_false(self) -> None:
        """With count_all=False, pattern contributes weight once."""
        matcher = PatternMatcher({"test": [(r"a", 2)]}, count_all_matches=False)
        result = matcher.match("aaa")
        assert result.scores["test"] == 2  # Only counted once

    def test_count_all_matches_true(self) -> None:
        """With count_all=True, pattern weight multiplied by occurrences."""
        matcher = PatternMatcher({"test": [(r"a", 2)]}, count_all_matches=True)
        result = matcher.match("aaa")
        assert result.scores["test"] == 6  # 3 * 2


class TestConfidenceCalculation:
    """Tests for confidence level calculation."""

    def test_high_confidence(self) -> None:
        """Should return high confidence for high score with good margin."""
        matcher = PatternMatcher(
            {
                "winner": [(r"\bstrong\b", 10)],
                "loser": [(r"\bweak\b", 1)],
            },
            high_score_threshold=8,
            high_margin_threshold=3,
        )
        result = matcher.match("strong signal")
        assert result.confidence == "high"

    def test_medium_confidence(self) -> None:
        """Should return medium confidence for moderate score."""
        matcher = PatternMatcher(
            {
                "a": [(r"\bword\b", 5)],
                "b": [(r"\bother\b", 3)],
            },
            high_score_threshold=8,
            medium_score_threshold=4,
        )
        result = matcher.match("word here")
        assert result.confidence == "medium"

    def test_low_confidence(self) -> None:
        """Should return low confidence for low score."""
        matcher = PatternMatcher(
            {
                "a": [(r"\bword\b", 2)],
                "b": [(r"\bother\b", 1)],
            },
            medium_score_threshold=4,
        )
        result = matcher.match("word")
        assert result.confidence == "low"

    def test_low_confidence_close_scores(self) -> None:
        """Should return lower confidence when scores are close."""
        matcher = PatternMatcher(
            {
                "a": [(r"\bone\b", 5), (r"\btwo\b", 3)],
                "b": [(r"\bthree\b", 4), (r"\bfour\b", 3)],
            },
            high_score_threshold=8,
            high_margin_threshold=3,
        )
        result = matcher.match("one three")
        # Even though "a" wins with 5, margin is only 1 (5 - 4)
        assert result.confidence != "high"


class TestMultilinePatterns:
    """Tests for multiline pattern matching."""

    def test_multiline_pattern(self) -> None:
        """Should handle patterns starting with ^ correctly."""
        matcher = PatternMatcher(
            {
                "line_start": [(r"^def\s+\w+", 3)],
                "anywhere": [(r"function", 2)],
            }
        )
        code = """
def hello():
    pass
"""
        result = matcher.match(code)
        assert result.scores["line_start"] == 3

    def test_multiline_multiple_matches(self) -> None:
        """Should match ^ patterns on multiple lines with count_all."""
        matcher = PatternMatcher(
            {"imports": [(r"^import\s+\w+", 2)]},
            count_all_matches=True,
        )
        code = """import os
import sys
import re"""
        result = matcher.match(code)
        assert result.scores["imports"] == 6  # 3 imports * 2


class TestGenericTypeParameter:
    """Tests for generic type parameter support."""

    def test_with_string_keys(self) -> None:
        """Should work with string category keys."""
        matcher: PatternMatcher[str] = PatternMatcher(
            {
                "alpha": [(r"a", 1)],
                "beta": [(r"b", 1)],
            }
        )
        assert matcher.best_match("aaa") == "alpha"

    def test_with_enum_keys(self) -> None:
        """Should work with enum category keys."""
        from enum import Enum

        class TaskType(Enum):
            CREATE = "create"
            DELETE = "delete"

        matcher: PatternMatcher[TaskType] = PatternMatcher(
            {
                TaskType.CREATE: [(r"\bcreate\b", 2), (r"\badd\b", 1)],
                TaskType.DELETE: [(r"\bdelete\b", 2), (r"\bremove\b", 1)],
            }
        )

        result = matcher.match("please create a new file")
        assert result.best_match == TaskType.CREATE

        result2 = matcher.match("delete the old one")
        assert result2.best_match == TaskType.DELETE


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_single_category(self) -> None:
        """Should work with single category."""
        matcher = PatternMatcher({"only": [(r"test", 1)]})
        result = matcher.match("test")
        assert result.best_match == "only"

    def test_empty_pattern_list(self) -> None:
        """Should handle category with empty pattern list."""
        matcher = PatternMatcher({"empty": [], "has_patterns": [(r"x", 1)]})
        result = matcher.match("x")
        assert result.best_match == "has_patterns"
        assert result.scores["empty"] == 0

    def test_special_regex_characters(self) -> None:
        """Should handle special regex characters correctly."""
        matcher = PatternMatcher(
            {
                "parens": [(r"\(.*\)", 1)],
                "brackets": [(r"\[.*\]", 1)],
            }
        )
        assert matcher.best_match("call(arg)") == "parens"
        assert matcher.best_match("array[0]") == "brackets"

    def test_unicode_text(self) -> None:
        """Should handle unicode text."""
        matcher = PatternMatcher(
            {
                "emoji": [(r"[\U0001F600-\U0001F64F]", 1)],
                "chinese": [(r"[\u4e00-\u9fff]", 1)],
            }
        )
        result = matcher.match("Hello \U0001f600")
        assert result.scores["emoji"] == 1

    def test_very_long_text(self) -> None:
        """Should handle very long text efficiently."""
        matcher = PatternMatcher({"word": [(r"\bhello\b", 1)]})
        long_text = "random " * 10000 + "hello" + " random" * 10000
        result = matcher.match(long_text)
        assert result.best_match == "word"
