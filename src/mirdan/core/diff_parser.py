"""Unified diff parser for validating only changed code."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class DiffHunk:
    """A single hunk from a unified diff."""

    file_path: str
    start_line: int
    line_count: int
    added_lines: list[tuple[int, str]] = field(default_factory=list)
    context_lines: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class ParsedDiff:
    """Parsed unified diff with extracted changed code."""

    hunks: list[DiffHunk] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)

    def get_added_code(self) -> str:
        """Get all added lines concatenated as a single code block.

        Returns:
            The added code from all hunks, preserving line ordering.
        """
        lines: list[str] = []
        for hunk in self.hunks:
            for _, line in hunk.added_lines:
                lines.append(line)
        return "\n".join(lines)

    def get_added_code_with_mapping(self) -> tuple[str, dict[int, tuple[str, int]]]:
        """Get added code with a mapping from extracted line numbers to original locations.

        Returns:
            Tuple of (concatenated added code, mapping). The mapping maps
            1-based line numbers in the extracted code to (file_path, original_line)
            tuples.
        """
        lines: list[str] = []
        mapping: dict[int, tuple[str, int]] = {}
        extracted_line = 1
        for hunk in self.hunks:
            for original_line, line in hunk.added_lines:
                lines.append(line)
                mapping[extracted_line] = (hunk.file_path, original_line)
                extracted_line += 1
        return "\n".join(lines), mapping

    def get_added_code_by_file(self) -> dict[str, str]:
        """Get added code grouped by file path.

        Returns:
            Dict mapping file path to added code string.
        """
        by_file: dict[str, list[str]] = {}
        for hunk in self.hunks:
            if hunk.file_path not in by_file:
                by_file[hunk.file_path] = []
            for _, line in hunk.added_lines:
                by_file[hunk.file_path].append(line)
        return {path: "\n".join(lines) for path, lines in by_file.items()}

    def get_changed_line_numbers(self, file_path: str) -> list[int]:
        """Get line numbers of changed lines for a specific file.

        Args:
            file_path: The file to get changed lines for.

        Returns:
            Sorted list of changed line numbers.
        """
        line_nums: list[int] = []
        for hunk in self.hunks:
            if hunk.file_path == file_path:
                line_nums.extend(num for num, _ in hunk.added_lines)
        return sorted(line_nums)


# Regex for diff header line: @@ -old_start,old_count +new_start,new_count @@
_HUNK_HEADER = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+(\d+)(?:,(\d+))?\s+@@")

# Regex for file path in diff header
_FILE_HEADER = re.compile(r"^\+\+\+\s+(?:b/)?(.+)$")


def parse_unified_diff(diff_text: str) -> ParsedDiff:
    """Parse a unified diff into structured hunks.

    Supports standard unified diff format (git diff, diff -u).

    Args:
        diff_text: The unified diff text to parse.

    Returns:
        ParsedDiff with hunks, added lines, and file info.
    """
    result = ParsedDiff()
    current_file = ""
    current_hunk: DiffHunk | None = None
    current_line = 0
    files_seen: set[str] = set()

    for raw_line in diff_text.splitlines():
        # Detect file header
        file_match = _FILE_HEADER.match(raw_line)
        if file_match:
            current_file = file_match.group(1).strip()
            if current_file not in files_seen:
                files_seen.add(current_file)
                result.files_changed.append(current_file)
            continue

        # Detect hunk header
        hunk_match = _HUNK_HEADER.match(raw_line)
        if hunk_match:
            start_line = int(hunk_match.group(1))
            count_str = hunk_match.group(2)
            line_count = int(count_str) if count_str else 1
            current_hunk = DiffHunk(
                file_path=current_file,
                start_line=start_line,
                line_count=line_count,
            )
            result.hunks.append(current_hunk)
            current_line = start_line
            continue

        if current_hunk is None:
            continue

        # Added line
        if raw_line.startswith("+"):
            content = raw_line[1:]
            current_hunk.added_lines.append((current_line, content))
            current_line += 1
        # Removed line (skip, don't advance new line counter)
        elif raw_line.startswith("-"):
            pass
        # Context line
        elif raw_line.startswith(" ") or raw_line == "":
            content = raw_line[1:] if raw_line.startswith(" ") else ""
            current_hunk.context_lines.append((current_line, content))
            current_line += 1
        # No-newline marker
        elif raw_line.startswith("\\"):
            pass

    return result
