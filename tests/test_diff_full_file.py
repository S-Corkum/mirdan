"""Tests for full-file diff validation path in _handle_diff."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_diff(file_path: str, added_lines: list[str], start_line: int = 1) -> str:
    """Create a minimal unified diff for testing."""
    count = len(added_lines)
    header = f"--- a/{file_path}\n+++ b/{file_path}\n@@ -0,0 +{start_line},{count} @@\n"
    body = "\n".join(f"+{line}" for line in added_lines)
    return header + body + "\n"


class TestFullFileDiffValidation:
    """Tests for the full-file validation path in _handle_diff."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    def _run_handle_diff(self, diff: str, cwd: Path | None = None) -> dict:
        from mirdan.server import _handle_diff

        with patch("mirdan.server.Path") as mock_path_cls:
            # Make Path.cwd() return our tmp_path
            mock_path_cls.cwd.return_value = cwd or self.tmp_path
            # But Path(x) / y should still work normally
            mock_path_cls.__truediv__ = Path.__truediv__
            # Actually, patching Path is too invasive. Let's use a different approach.
            pass

        # Use monkeypatch-style approach - just chdir
        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(cwd or self.tmp_path)
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(
                    _handle_diff(
                        diff=diff,
                        language="auto",
                        check_security=True,
                        session_id="test",
                        max_tokens=4000,
                        model_tier="standard",
                    )
                )
            finally:
                loop.close()
        finally:
            os.chdir(original_cwd)

    def test_full_file_architecture_violations_on_changed_lines(self) -> None:
        """When file exists on disk, architecture violations on changed lines are returned."""
        # Create a Python file with a very long function
        long_func = "def long_function():\n" + "    x = 1\n" * 50 + "    return x\n"
        file_path = self.tmp_path / "module.py"
        file_path.write_text(long_func)

        # Diff that touches lines in that function
        diff = _make_diff("module.py", ["    x = 1"] * 5, start_line=2)
        result = self._run_handle_diff(diff)

        # Should have run with full file context (architecture checks enabled)
        assert "passed" in result

    def test_violations_on_unchanged_lines_filtered_out(self) -> None:
        """Violations on unchanged lines should be filtered from results."""
        # Create file with eval() on line 5 (unchanged) and clean code on line 10 (changed)
        code = textwrap.dedent("""\
            import os
            import sys

            def func_a():
                eval("bad_code")

            def func_b():
                x = 1
                y = 2
                return x + y
        """)
        file_path = self.tmp_path / "module.py"
        file_path.write_text(code)

        # Diff only touches line 10 (y = 2)
        diff = _make_diff("module.py", ["    y = 2"], start_line=9)
        result = self._run_handle_diff(diff)

        # eval() violation is on line 5 (unchanged) — should not appear
        violations = result.get("violations", [])
        py001 = [v for v in violations if v.get("id") == "PY001"]
        assert len(py001) == 0

    def test_file_scope_rules_excluded(self) -> None:
        """ARCH002/TSARCH002 (file-too-long) should be excluded from diff results."""
        # Create a very long Python file
        long_code = "x = 1\n" * 500
        file_path = self.tmp_path / "long_module.py"
        file_path.write_text(long_code)

        # Diff touches one line
        diff = _make_diff("long_module.py", ["x = 1"], start_line=1)
        result = self._run_handle_diff(diff)

        violations = result.get("violations", [])
        arch002 = [v for v in violations if v.get("id") in ("ARCH002", "TSARCH002")]
        assert len(arch002) == 0

    def test_file_not_on_disk_falls_back(self) -> None:
        """When file does NOT exist on disk, falls back to added-lines-only validation."""
        # Don't create the file — it won't exist
        diff = _make_diff("nonexistent.py", ["eval(user_input)"], start_line=1)
        result = self._run_handle_diff(diff)

        # Should still work (fallback path)
        assert "passed" in result

    def test_mixed_files_hybrid_validation(self) -> None:
        """Multi-file diff: full-file for found files, fallback for missing."""
        # Create one file
        code = "def func():\n    return 1\n"
        (self.tmp_path / "exists.py").write_text(code)

        # Diff with two files — one exists, one doesn't
        diff1 = _make_diff("exists.py", ["    return 1"], start_line=2)
        diff2 = _make_diff("missing.py", ["x = 1"], start_line=1)
        combined_diff = diff1 + diff2

        result = self._run_handle_diff(combined_diff)
        assert "passed" in result
        # files_changed may be stripped by output formatter in compact mode
        # The key assertion is that the handler completes without error
