"""Tests for the diff parser component."""

from mirdan.core.diff_parser import DiffHunk, ParsedDiff, parse_unified_diff


class TestParseUnifiedDiff:
    """Tests for parse_unified_diff function."""

    def test_basic_diff(self) -> None:
        """Should parse a basic unified diff."""
        diff = """--- a/src/auth.py
+++ b/src/auth.py
@@ -1,3 +1,4 @@
 import os
+import bcrypt

 def login():
"""
        result = parse_unified_diff(diff)

        assert len(result.files_changed) == 1
        assert result.files_changed[0] == "src/auth.py"
        assert len(result.hunks) == 1
        assert len(result.hunks[0].added_lines) == 1
        assert result.hunks[0].added_lines[0] == (2, "import bcrypt")

    def test_multiple_hunks(self) -> None:
        """Should parse diff with multiple hunks."""
        diff = """--- a/app.py
+++ b/app.py
@@ -1,3 +1,4 @@
 import os
+import sys

 def main():
@@ -10,3 +11,4 @@
     pass
+    print("done")

"""
        result = parse_unified_diff(diff)

        assert len(result.hunks) == 2
        assert result.hunks[0].added_lines[0][1] == "import sys"
        assert result.hunks[1].added_lines[0][1] == '    print("done")'

    def test_multiple_files(self) -> None:
        """Should parse diff with multiple files."""
        diff = """--- a/file1.py
+++ b/file1.py
@@ -1,2 +1,3 @@
 a = 1
+b = 2
--- a/file2.py
+++ b/file2.py
@@ -1,2 +1,3 @@
 x = 1
+y = 2
"""
        result = parse_unified_diff(diff)

        assert len(result.files_changed) == 2
        assert "file1.py" in result.files_changed
        assert "file2.py" in result.files_changed

    def test_removed_lines_not_included(self) -> None:
        """Should not include removed lines in added code."""
        diff = """--- a/app.py
+++ b/app.py
@@ -1,3 +1,3 @@
 import os
-import sys
+import pathlib

"""
        result = parse_unified_diff(diff)

        added = result.get_added_code()
        assert "pathlib" in added
        assert "sys" not in added

    def test_empty_diff(self) -> None:
        """Should handle empty diff."""
        result = parse_unified_diff("")
        assert result.hunks == []
        assert result.files_changed == []

    def test_diff_without_changes(self) -> None:
        """Should handle diff with only context lines."""
        diff = """--- a/app.py
+++ b/app.py
@@ -1,3 +1,3 @@
 import os
 import sys

"""
        result = parse_unified_diff(diff)
        assert result.get_added_code() == ""

    def test_new_file(self) -> None:
        """Should parse diff for a new file."""
        diff = """--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,3 @@
+def hello():
+    return "world"
+
"""
        result = parse_unified_diff(diff)

        assert "new_file.py" in result.files_changed
        code = result.get_added_code()
        assert "def hello():" in code
        assert 'return "world"' in code

    def test_hunk_line_numbers(self) -> None:
        """Should correctly track line numbers in hunks."""
        diff = """--- a/app.py
+++ b/app.py
@@ -10,3 +10,5 @@
 existing_line
+new_line_1
+new_line_2
 another_existing
"""
        result = parse_unified_diff(diff)

        hunk = result.hunks[0]
        assert hunk.start_line == 10
        # Line 10 is context, 11 and 12 are added
        assert hunk.added_lines[0] == (11, "new_line_1")
        assert hunk.added_lines[1] == (12, "new_line_2")

    def test_no_newline_marker_ignored(self) -> None:
        """Should ignore 'No newline at end of file' markers."""
        diff = """--- a/app.py
+++ b/app.py
@@ -1,2 +1,3 @@
 line1
+line2
\\ No newline at end of file
"""
        result = parse_unified_diff(diff)
        assert len(result.hunks[0].added_lines) == 1


class TestParsedDiff:
    """Tests for ParsedDiff methods."""

    def test_get_added_code(self) -> None:
        """Should concatenate all added lines."""
        diff = ParsedDiff(
            hunks=[
                DiffHunk(
                    file_path="a.py",
                    start_line=1,
                    line_count=3,
                    added_lines=[(1, "line1"), (2, "line2")],
                ),
                DiffHunk(
                    file_path="a.py",
                    start_line=10,
                    line_count=2,
                    added_lines=[(10, "line3")],
                ),
            ],
            files_changed=["a.py"],
        )
        code = diff.get_added_code()
        assert code == "line1\nline2\nline3"

    def test_get_added_code_by_file(self) -> None:
        """Should group added code by file."""
        diff = ParsedDiff(
            hunks=[
                DiffHunk(
                    file_path="a.py",
                    start_line=1,
                    line_count=1,
                    added_lines=[(1, "a_code")],
                ),
                DiffHunk(
                    file_path="b.py",
                    start_line=1,
                    line_count=1,
                    added_lines=[(1, "b_code")],
                ),
            ],
            files_changed=["a.py", "b.py"],
        )
        by_file = diff.get_added_code_by_file()
        assert by_file["a.py"] == "a_code"
        assert by_file["b.py"] == "b_code"

    def test_get_changed_line_numbers(self) -> None:
        """Should return sorted line numbers for a file."""
        diff = ParsedDiff(
            hunks=[
                DiffHunk(
                    file_path="a.py",
                    start_line=1,
                    line_count=3,
                    added_lines=[(5, "x"), (2, "y"), (10, "z")],
                ),
            ],
            files_changed=["a.py"],
        )
        nums = diff.get_changed_line_numbers("a.py")
        assert nums == [2, 5, 10]

    def test_get_changed_line_numbers_unknown_file(self) -> None:
        """Should return empty list for unknown file."""
        diff = ParsedDiff()
        assert diff.get_changed_line_numbers("unknown.py") == []


class TestDiffIntegration:
    """Integration tests with realistic diffs."""

    def test_git_diff_format(self) -> None:
        """Should parse a realistic git diff."""
        diff = """diff --git a/src/mirdan/core/validator.py b/src/mirdan/core/validator.py
index abc1234..def5678 100644
--- a/src/mirdan/core/validator.py
+++ b/src/mirdan/core/validator.py
@@ -15,6 +15,8 @@ class Validator:
     def __init__(self):
         self.rules = []
+        self.custom_rules = []
+        self._compiled = False

     def validate(self, code):
         pass
"""
        result = parse_unified_diff(diff)

        assert "src/mirdan/core/validator.py" in result.files_changed
        code = result.get_added_code()
        assert "self.custom_rules = []" in code
        assert "self._compiled = False" in code
