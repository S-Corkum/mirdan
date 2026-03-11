"""Tests for TidyFirstAnalyzer — preparatory refactoring intelligence."""

from __future__ import annotations

import pytest

from mirdan.config import TidyFirstConfig
from mirdan.core.tidy_first import TidyFirstAnalyzer
from mirdan.models import EntityType, ExtractedEntity, Intent, TaskType


def _make_intent(file_paths: list[str]) -> Intent:
    """Create an Intent with FILE_PATH entities."""
    entities = [
        ExtractedEntity(type=EntityType.FILE_PATH, value=fp)
        for fp in file_paths
    ]
    return Intent(
        original_prompt="test",
        task_type=TaskType.GENERATION,
        entities=entities,
    )


@pytest.fixture()
def analyzer() -> TidyFirstAnalyzer:
    return TidyFirstAnalyzer(TidyFirstConfig())


@pytest.fixture()
def disabled_analyzer() -> TidyFirstAnalyzer:
    return TidyFirstAnalyzer(TidyFirstConfig(enabled=False))


class TestAnalyze:
    def test_empty_intent_no_suggestions(self, analyzer: TidyFirstAnalyzer) -> None:
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        result = analyzer.analyze(intent)
        assert len(result.suggestions) == 0
        assert len(result.target_files) == 0

    def test_nonexistent_file_skipped(self, analyzer: TidyFirstAnalyzer) -> None:
        intent = _make_intent(["/nonexistent/path/file.py"])
        result = analyzer.analyze(intent)
        assert len(result.suggestions) == 0
        assert "/nonexistent/path/file.py" in result.skipped_files

    def test_disabled_config_returns_empty(self, disabled_analyzer: TidyFirstAnalyzer) -> None:
        intent = _make_intent(["some_file.py"])
        result = disabled_analyzer.analyze(intent)
        assert len(result.suggestions) == 0
        assert len(result.target_files) == 0


class TestLongFunctionDetection:
    def test_long_function_detected(self, analyzer: TidyFirstAnalyzer, tmp_path: object) -> None:
        # Create a Python file with a long function (40 lines)
        import pathlib
        p = pathlib.Path(str(tmp_path)) / "long.py"
        lines = ["def big_function():"]
        lines.extend(f"    x_{i} = {i}" for i in range(40))
        p.write_text("\n".join(lines))

        intent = _make_intent([str(p)])
        result = analyzer.analyze(intent)
        extract_suggestions = [s for s in result.suggestions if s.type == "extract_method"]
        assert len(extract_suggestions) >= 1
        assert "big_function" in extract_suggestions[0].description

    def test_short_function_no_suggestion(self, analyzer: TidyFirstAnalyzer, tmp_path: object) -> None:
        import pathlib
        p = pathlib.Path(str(tmp_path)) / "short.py"
        lines = ["def small_function():"]
        lines.extend(f"    x_{i} = {i}" for i in range(10))
        p.write_text("\n".join(lines))

        intent = _make_intent([str(p)])
        result = analyzer.analyze(intent)
        extract_suggestions = [s for s in result.suggestions if s.type == "extract_method"]
        assert len(extract_suggestions) == 0


class TestDeepNestingDetection:
    def test_deep_nesting_detected(self, analyzer: TidyFirstAnalyzer, tmp_path: object) -> None:
        import pathlib
        p = pathlib.Path(str(tmp_path)) / "nested.py"
        code = """\
def complex_function():
    if True:
        for x in range(10):
            if x > 5:
                while x > 0:
                    x -= 1
"""
        p.write_text(code)

        intent = _make_intent([str(p)])
        result = analyzer.analyze(intent)
        nesting_suggestions = [s for s in result.suggestions if s.type == "simplify_conditional"]
        assert len(nesting_suggestions) >= 1

    def test_shallow_nesting_no_suggestion(self, analyzer: TidyFirstAnalyzer, tmp_path: object) -> None:
        import pathlib
        p = pathlib.Path(str(tmp_path)) / "shallow.py"
        code = """\
def simple_function():
    if True:
        for x in range(10):
            print(x)
"""
        p.write_text(code)

        intent = _make_intent([str(p)])
        result = analyzer.analyze(intent)
        nesting_suggestions = [s for s in result.suggestions if s.type == "simplify_conditional"]
        assert len(nesting_suggestions) == 0


class TestFileSizeDetection:
    def test_large_file_detected(self, analyzer: TidyFirstAnalyzer, tmp_path: object) -> None:
        import pathlib
        p = pathlib.Path(str(tmp_path)) / "large.py"
        lines = [f"line_{i} = {i}" for i in range(400)]
        p.write_text("\n".join(lines))

        intent = _make_intent([str(p)])
        result = analyzer.analyze(intent)
        split_suggestions = [s for s in result.suggestions if s.type == "split_file"]
        assert len(split_suggestions) >= 1
        assert split_suggestions[0].effort == "medium"


class TestMaxSuggestionsCap:
    def test_capped_at_max(self, tmp_path: object) -> None:
        import pathlib
        config = TidyFirstConfig(max_suggestions=2, min_function_length=5)
        analyzer = TidyFirstAnalyzer(config)

        # Create file with many long functions
        p = pathlib.Path(str(tmp_path)) / "many.py"
        functions = []
        for i in range(5):
            func_lines = [f"def func_{i}():"]
            func_lines.extend(f"    x_{j} = {j}" for j in range(10))
            functions.append("\n".join(func_lines))
        p.write_text("\n\n".join(functions))

        intent = _make_intent([str(p)])
        result = analyzer.analyze(intent)
        assert len(result.suggestions) <= 2


class TestEdgeCases:
    def test_binary_file_skipped(self, analyzer: TidyFirstAnalyzer, tmp_path: object) -> None:
        import pathlib
        p = pathlib.Path(str(tmp_path)) / "binary.py"
        p.write_bytes(b"\x80\x81\xfe\xff" * 100)  # Invalid UTF-8 bytes

        intent = _make_intent([str(p)])
        result = analyzer.analyze(intent)
        assert str(p) in result.skipped_files

    def test_syntax_error_falls_back_to_regex(self, analyzer: TidyFirstAnalyzer, tmp_path: object) -> None:
        import pathlib
        p = pathlib.Path(str(tmp_path)) / "bad.py"
        # Valid enough to read but invalid Python syntax
        lines = ["def broken("]
        lines.extend(f"        x_{i} = {i}" for i in range(50))
        p.write_text("\n".join(lines))

        intent = _make_intent([str(p)])
        # Should not crash
        result = analyzer.analyze(intent)
        assert str(p) in result.target_files

    def test_non_python_uses_generic(self, analyzer: TidyFirstAnalyzer, tmp_path: object) -> None:
        import pathlib
        p = pathlib.Path(str(tmp_path)) / "code.js"
        # Create deeply nested JS
        lines = ["function foo() {"]
        for i in range(4):
            indent = "    " * (i + 1)
            lines.append(f"{indent}if (true) {{")
        for i in range(3, -1, -1):
            indent = "    " * (i + 1)
            lines.append(f"{indent}}}")
        lines.append("}")
        p.write_text("\n".join(lines))

        intent = _make_intent([str(p)])
        result = analyzer.analyze(intent)
        assert str(p) in result.target_files

    def test_file_size_cap(self, tmp_path: object) -> None:
        import pathlib
        config = TidyFirstConfig(max_file_size_kb=1)  # 1KB limit
        analyzer = TidyFirstAnalyzer(config)

        p = pathlib.Path(str(tmp_path)) / "huge.py"
        # Write >1KB
        p.write_text("x = 1\n" * 500)

        intent = _make_intent([str(p)])
        result = analyzer.analyze(intent)
        assert str(p) in result.skipped_files

    def test_effort_sorting(self, analyzer: TidyFirstAnalyzer, tmp_path: object) -> None:
        import pathlib
        # Create file with both a large file (medium effort) and long function (small effort)
        p = pathlib.Path(str(tmp_path)) / "mixed.py"
        lines = []
        # Long function
        lines.append("def very_long_function():")
        lines.extend(f"    x_{i} = {i}" for i in range(50))
        # Pad to >300 non-empty lines
        lines.extend(f"constant_{i} = {i}" for i in range(300))
        p.write_text("\n".join(lines))

        intent = _make_intent([str(p)])
        config = TidyFirstConfig(max_suggestions=10)
        local_analyzer = TidyFirstAnalyzer(config)
        result = local_analyzer.analyze(intent)

        if len(result.suggestions) >= 2:
            # small/trivial effort should come before medium
            efforts = [s.effort for s in result.suggestions]
            small_idx = next((i for i, e in enumerate(efforts) if e in ("trivial", "small")), None)
            medium_idx = next((i for i, e in enumerate(efforts) if e == "medium"), None)
            if small_idx is not None and medium_idx is not None:
                assert small_idx < medium_idx
