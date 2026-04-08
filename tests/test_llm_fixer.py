"""Tests for LLMFixer search/replace fix application."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from mirdan.core.llm_fixer import LLMFixer
from mirdan.models import FileFixReport, FixRunReport


class TestLLMFixerFixFile:
    """Tests for LLMFixer.fix_file()."""

    @pytest.mark.asyncio
    async def test_applies_search_replace_fix(self, tmp_path: Path) -> None:
        source = tmp_path / "test.py"
        source.write_text("x = 1\nexcept:\n    pass\n")

        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = {
            "fixes": [
                {
                    "violation_id": "PY003",
                    "search": "except:",
                    "replace": "except Exception:",
                    "confidence": 0.9,
                    "description": "Replace bare except",
                },
            ],
        }

        fixer = LLMFixer(llm_manager=mock_llm)
        report = await fixer.fix_file(
            str(source),
            [{"id": "PY003", "message": "bare except", "line": 2}],
        )

        assert len(report.applied) == 1
        assert report.applied[0]["violation_id"] == "PY003"
        assert "except Exception:" in source.read_text()

    @pytest.mark.asyncio
    async def test_skips_low_confidence_fix(self, tmp_path: Path) -> None:
        source = tmp_path / "test.py"
        source.write_text("x = eval(input())\n")

        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = {
            "fixes": [
                {
                    "violation_id": "SEC001",
                    "search": "eval(input())",
                    "replace": "ast.literal_eval(input())",
                    "confidence": 0.3,  # Below threshold
                    "description": "Low confidence fix",
                },
            ],
        }

        fixer = LLMFixer(llm_manager=mock_llm)
        report = await fixer.fix_file(
            str(source),
            [{"id": "SEC001", "message": "eval usage", "line": 1}],
        )

        assert len(report.applied) == 0
        assert "SEC001" in report.skipped
        assert "eval(input())" in source.read_text()  # Unchanged

    @pytest.mark.asyncio
    async def test_skips_when_search_not_found(self, tmp_path: Path) -> None:
        source = tmp_path / "test.py"
        source.write_text("x = 1\n")

        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = {
            "fixes": [
                {
                    "violation_id": "PY001",
                    "search": "nonexistent code",  # Not in file
                    "replace": "replacement",
                    "confidence": 0.9,
                },
            ],
        }

        fixer = LLMFixer(llm_manager=mock_llm)
        report = await fixer.fix_file(
            str(source),
            [{"id": "PY001", "message": "test", "line": 1}],
        )

        assert len(report.applied) == 0
        assert "PY001" in report.skipped
        assert source.read_text() == "x = 1\n"  # Unchanged

    @pytest.mark.asyncio
    async def test_applies_multiple_fixes(self, tmp_path: Path) -> None:
        source = tmp_path / "test.py"
        source.write_text("import os\nimport sys\nx = 1\n")

        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = {
            "fixes": [
                {
                    "violation_id": "F401-1",
                    "search": "import os\n",
                    "replace": "",
                    "confidence": 0.9,
                    "description": "Remove unused import os",
                },
                {
                    "violation_id": "F401-2",
                    "search": "import sys\n",
                    "replace": "",
                    "confidence": 0.85,
                    "description": "Remove unused import sys",
                },
            ],
        }

        fixer = LLMFixer(llm_manager=mock_llm)
        report = await fixer.fix_file(
            str(source),
            [
                {"id": "F401-1", "message": "unused import os"},
                {"id": "F401-2", "message": "unused import sys"},
            ],
        )

        assert len(report.applied) == 2
        assert source.read_text() == "x = 1\n"

    @pytest.mark.asyncio
    async def test_returns_empty_report_for_missing_file(self) -> None:
        fixer = LLMFixer(llm_manager=AsyncMock())
        report = await fixer.fix_file(
            "/nonexistent/file.py",
            [{"id": "PY001", "message": "test"}],
        )

        assert len(report.applied) == 0
        assert "PY001" in report.skipped

    @pytest.mark.asyncio
    async def test_returns_empty_report_when_no_llm(self, tmp_path: Path) -> None:
        source = tmp_path / "test.py"
        source.write_text("x = 1\n")

        fixer = LLMFixer(llm_manager=None)
        report = await fixer.fix_file(
            str(source),
            [{"id": "PY001", "message": "test"}],
        )

        assert len(report.applied) == 0

    @pytest.mark.asyncio
    async def test_handles_llm_failure(self, tmp_path: Path) -> None:
        source = tmp_path / "test.py"
        source.write_text("x = 1\n")

        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = None

        fixer = LLMFixer(llm_manager=mock_llm)
        report = await fixer.fix_file(
            str(source),
            [{"id": "PY001", "message": "test"}],
        )

        assert len(report.applied) == 0
        assert source.read_text() == "x = 1\n"  # Unchanged

    @pytest.mark.asyncio
    async def test_replaces_only_first_occurrence(self, tmp_path: Path) -> None:
        source = tmp_path / "test.py"
        source.write_text("x = None\ny = None\n")

        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = {
            "fixes": [
                {
                    "violation_id": "PY001",
                    "search": "x = None",
                    "replace": "x = 0",
                    "confidence": 0.8,
                },
            ],
        }

        fixer = LLMFixer(llm_manager=mock_llm)
        await fixer.fix_file(
            str(source),
            [{"id": "PY001", "message": "test"}],
        )

        content = source.read_text()
        assert content == "x = 0\ny = None\n"  # Only first occurrence changed


class TestLLMFixerRevert:
    """Tests for file revert capability."""

    @pytest.mark.asyncio
    async def test_revert_restores_content(self, tmp_path: Path) -> None:
        source = tmp_path / "test.py"
        source.write_text("modified content")

        fixer = LLMFixer()
        await fixer.revert_file(str(source), "original content")

        assert source.read_text() == "original content"


class TestFixPrompt:
    """Tests for fix prompt construction."""

    def test_prompt_includes_code_and_violations(self) -> None:
        from mirdan.llm.prompts.fix import build_fix_prompt

        prompt = build_fix_prompt("def foo(): pass", '[{"id": "PY001"}]')
        assert "def foo(): pass" in prompt
        assert "PY001" in prompt

    def test_prompt_has_nonce_delimiters(self) -> None:
        import re

        from mirdan.llm.prompts.fix import build_fix_prompt

        prompt = build_fix_prompt("code", "[]")
        nonces = re.findall(r"SOURCE_FILE_([a-f0-9]+)>", prompt)
        assert nonces  # Has nonce
        assert len(set(nonces)) == 1  # Same nonce for open/close

    def test_schema_requires_fixes_array(self) -> None:
        from mirdan.llm.prompts.fix import FIX_SCHEMA

        assert "fixes" in FIX_SCHEMA["required"]
        items = FIX_SCHEMA["properties"]["fixes"]["items"]
        assert "search" in items["required"]
        assert "replace" in items["required"]


class TestFixModels:
    """Tests for FileFixReport and FixRunReport dataclasses."""

    def test_file_fix_report_to_dict(self) -> None:
        report = FileFixReport(
            file="test.py",
            applied=[{"violation_id": "PY001", "search": "x", "replace": "y"}],
            skipped=["PY002"],
        )
        d = report.to_dict()
        assert d["file"] == "test.py"
        assert len(d["applied"]) == 1
        assert d["skipped"] == ["PY002"]

    def test_fix_run_report_to_dict(self) -> None:
        report = FixRunReport(
            files=[FileFixReport(file="a.py", applied=[])],
            ruff_auto_fixed=["lint auto-fixed"],
            all_pass=True,
            summary="all pass",
        )
        d = report.to_dict()
        assert len(d["files"]) == 1
        assert d["all_pass"] is True
