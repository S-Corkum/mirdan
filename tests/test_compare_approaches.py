"""Tests for the compare_approaches tool and ComparisonResult model."""

from mirdan.core.code_validator import CodeValidator
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import ComparisonEntry, ComparisonResult


class TestComparisonEntry:
    """Tests for the ComparisonEntry dataclass."""

    def test_to_dict(self) -> None:
        entry = ComparisonEntry(
            label="Approach A",
            score=0.9,
            passed=True,
            violation_counts={"error": 0, "warning": 1, "info": 0},
            summary="Code passes with 1 warning(s)",
        )
        d = entry.to_dict()
        assert d["label"] == "Approach A"
        assert d["score"] == 0.9
        assert d["passed"] is True
        assert d["violation_counts"]["warning"] == 1

    def test_default_values(self) -> None:
        entry = ComparisonEntry(label="A", score=1.0, passed=True)
        assert entry.violation_counts == {}
        assert entry.summary == ""


class TestComparisonResult:
    """Tests for the ComparisonResult dataclass."""

    def test_to_dict(self) -> None:
        result = ComparisonResult(
            entries=[
                ComparisonEntry(label="A", score=0.9, passed=True),
                ComparisonEntry(label="B", score=0.7, passed=False),
            ],
            winner="A",
            language_detected="python",
        )
        d = result.to_dict()
        assert d["winner"] == "A"
        assert d["count"] == 2
        assert len(d["entries"]) == 2

    def test_empty_result(self) -> None:
        result = ComparisonResult()
        d = result.to_dict()
        assert d["count"] == 0
        assert d["entries"] == []
        assert d["winner"] == ""


class TestCompareApproachesLogic:
    """Tests for the comparison logic (without server tool layer)."""

    def test_compare_clean_vs_dirty(self) -> None:
        """Clean code should score higher than code with issues."""
        clean = "def greet(name: str) -> str:\n    return f'Hello, {name}'\n"
        dirty = "result = eval(user_input)\n"

        validator = CodeValidator(QualityStandards())
        results = []
        for code, label in [(clean, "Clean"), (dirty, "Dirty")]:
            result = validator.validate(code, language="python")
            results.append(
                ComparisonEntry(
                    label=label,
                    score=result.score,
                    passed=result.passed,
                )
            )

        best = max(results, key=lambda e: e.score)
        assert best.label == "Clean"

    def test_compare_two_clean_implementations(self) -> None:
        """Two clean implementations should both pass."""
        impl_a = "def add(a: int, b: int) -> int:\n    return a + b\n"
        impl_b = "def add(x: int, y: int) -> int:\n    return x + y\n"

        validator = CodeValidator(QualityStandards())
        entries = []
        for code, label in [(impl_a, "A"), (impl_b, "B")]:
            result = validator.validate(code, language="python")
            entries.append(
                ComparisonEntry(
                    label=label,
                    score=result.score,
                    passed=result.passed,
                )
            )

        assert all(e.passed for e in entries)
        assert all(e.score > 0.8 for e in entries)

    def test_compare_multiple_implementations(self) -> None:
        """Should handle 3+ implementations."""
        impls = [
            ("def f(x):\n    return x\n", "Simple"),
            ("result = eval('x')\n", "Eval"),
            ("def f(x: int) -> int:\n    return x\n", "Typed"),
        ]

        validator = CodeValidator(QualityStandards())
        entries = []
        for code, label in impls:
            result = validator.validate(code, language="python")
            entries.append(
                ComparisonEntry(
                    label=label,
                    score=result.score,
                    passed=result.passed,
                )
            )

        comparison = ComparisonResult(
            entries=entries,
            winner=max(entries, key=lambda e: e.score).label,
        )

        assert comparison.winner != "Eval"
        assert len(comparison.entries) == 3
