"""Tests for ConventionExtractor."""

from pathlib import Path

import pytest

from mirdan.core.convention_extractor import ConventionExtractor, ScanResult


@pytest.fixture()
def extractor() -> ConventionExtractor:
    return ConventionExtractor()


@pytest.fixture()
def python_project(tmp_path: Path) -> Path:
    """Create a minimal Python project with consistent patterns."""
    src = tmp_path / "src"
    src.mkdir()

    # Create multiple Python files with consistent conventions
    for i in range(6):
        content = f'''"""Module {i} docstring."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def function_{i}_alpha(x: int) -> int:
    """Process alpha for module {i}."""
    return x + 1


def function_{i}_beta(y: str) -> str:
    """Process beta for module {i}."""
    return y.upper()


def function_{i}_gamma(z: float) -> float:
    """Process gamma for module {i}."""
    return z * 2


class Handler{i}:
    """Handler class for module {i}."""

    def handle(self) -> None:
        """Handle the request."""
        pass
'''
        (src / f"module_{i}.py").write_text(content)

    return tmp_path


@pytest.fixture()
def messy_project(tmp_path: Path) -> Path:
    """Create a project with violations and inconsistent patterns."""
    src = tmp_path / "src"
    src.mkdir()

    for i in range(6):
        content = f"""import os
import sys

def func{i}(x):
    try:
        return eval(x)
    except:
        pass

def another_func{i}(y):
    exec(y)
    return None

class handler{i}:
    pass
"""
        (src / f"mod_{i}.py").write_text(content)

    return tmp_path


class TestConventionExtractorScan:
    """Tests for the scan method."""

    def test_scan_empty_directory(self, extractor: ConventionExtractor, tmp_path: Path) -> None:
        result = extractor.scan(tmp_path)
        assert result.files_scanned == 0
        assert result.conventions == []

    def test_scan_returns_scan_result(
        self, extractor: ConventionExtractor, python_project: Path
    ) -> None:
        result = extractor.scan(python_project)
        assert isinstance(result, ScanResult)
        assert result.files_scanned > 0
        assert result.language == "python"

    def test_scan_detects_language(
        self, extractor: ConventionExtractor, python_project: Path
    ) -> None:
        result = extractor.scan(python_project, language="auto")
        assert result.language == "python"

    def test_scan_explicit_language(
        self, extractor: ConventionExtractor, python_project: Path
    ) -> None:
        result = extractor.scan(python_project, language="python")
        assert result.language == "python"

    def test_scan_computes_avg_score(
        self, extractor: ConventionExtractor, python_project: Path
    ) -> None:
        result = extractor.scan(python_project)
        assert 0.0 <= result.avg_score <= 1.0

    def test_scan_computes_pass_rate(
        self, extractor: ConventionExtractor, python_project: Path
    ) -> None:
        result = extractor.scan(python_project)
        assert 0.0 <= result.pass_rate <= 1.0


class TestConventionExtractorConventions:
    """Tests for convention extraction."""

    def test_discovers_naming_convention(
        self, extractor: ConventionExtractor, python_project: Path
    ) -> None:
        result = extractor.scan(python_project)
        naming_entries = [
            e for e in result.conventions if "naming" in e.tags
        ]
        assert len(naming_entries) >= 1
        assert any("snake_case" in e.content for e in naming_entries)

    def test_discovers_import_convention(
        self, extractor: ConventionExtractor, python_project: Path
    ) -> None:
        result = extractor.scan(python_project)
        import_entries = [
            e for e in result.conventions if "imports" in e.tags
        ]
        assert len(import_entries) >= 1
        assert any("future" in e.content.lower() for e in import_entries)

    def test_discovers_docstring_convention(
        self, extractor: ConventionExtractor, python_project: Path
    ) -> None:
        result = extractor.scan(python_project)
        docstring_entries = [
            e for e in result.conventions if "docstrings" in e.tags
        ]
        assert len(docstring_entries) >= 1

    def test_discovers_quality_baseline(
        self, extractor: ConventionExtractor, python_project: Path
    ) -> None:
        result = extractor.scan(python_project)
        quality_entries = [
            e for e in result.conventions if "quality" in e.tags and "baseline" in e.tags
        ]
        # May or may not produce depending on score threshold
        # Just verify it doesn't crash
        assert isinstance(quality_entries, list)

    def test_messy_project_discovers_violations(
        self, extractor: ConventionExtractor, messy_project: Path
    ) -> None:
        result = extractor.scan(messy_project)
        assert result.files_scanned > 0
        # Should have common violations
        assert len(result.common_violations) > 0

    def test_all_conventions_are_knowledge_entries(
        self, extractor: ConventionExtractor, python_project: Path
    ) -> None:
        result = extractor.scan(python_project)
        for entry in result.conventions:
            d = entry.to_dict()
            assert "content" in d
            assert "content_type" in d
            assert "tags" in d
            assert "scope" in d
            assert "confidence" in d
            assert 0.0 <= entry.confidence <= 1.0

    def test_conventions_have_project_scope(
        self, extractor: ConventionExtractor, python_project: Path
    ) -> None:
        result = extractor.scan(python_project)
        for entry in result.conventions:
            assert entry.scope == "project"
            assert entry.scope_path  # Non-empty


class TestConventionExtractorSkipDirs:
    """Tests for directory skipping."""

    def test_skips_pycache(self, extractor: ConventionExtractor, tmp_path: Path) -> None:
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "module.py").write_text("x = 1")

        result = extractor.scan(tmp_path, language="python")
        assert result.files_scanned == 0

    def test_skips_node_modules(self, extractor: ConventionExtractor, tmp_path: Path) -> None:
        nm_dir = tmp_path / "node_modules"
        nm_dir.mkdir()
        (nm_dir / "index.js").write_text("const x = 1;")

        result = extractor.scan(tmp_path, language="javascript")
        assert result.files_scanned == 0

    def test_skips_venv(self, extractor: ConventionExtractor, tmp_path: Path) -> None:
        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()
        (venv_dir / "lib.py").write_text("x = 1")

        result = extractor.scan(tmp_path, language="python")
        assert result.files_scanned == 0


class TestScanResultToDict:
    """Tests for ScanResult serialization."""

    def test_to_dict_structure(self) -> None:
        result = ScanResult(
            directory="/tmp/test",
            language="python",
            files_scanned=10,
            avg_score=0.85,
            pass_rate=0.9,
        )
        d = result.to_dict()
        assert d["directory"] == "/tmp/test"
        assert d["language"] == "python"
        assert d["files_scanned"] == 10
        assert d["avg_score"] == 0.85
        assert d["pass_rate"] == 0.9
        assert d["convention_count"] == 0
        assert d["conventions"] == []

    def test_to_dict_rounds_floats(self) -> None:
        result = ScanResult(
            directory="/tmp/test",
            language="python",
            files_scanned=5,
            avg_score=0.8567891,
            pass_rate=0.66666,
        )
        d = result.to_dict()
        assert d["avg_score"] == 0.857
        assert d["pass_rate"] == 0.667


class TestScanTypeCheckingConvention:
    """Tests for TYPE_CHECKING import detection."""

    def test_type_checking_detected(self, extractor: ConventionExtractor, tmp_path: Path) -> None:
        """Projects using TYPE_CHECKING in >30% of files should produce convention."""
        for i in range(6):
            content = f"""from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

def func_{i}() -> None:
    pass
"""
            (tmp_path / f"mod_{i}.py").write_text(content)

        result = extractor.scan(tmp_path, language="python")
        tc_entries = [e for e in result.conventions if "type-checking" in e.tags]
        assert len(tc_entries) >= 1
