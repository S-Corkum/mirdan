"""Skip-region detection for string literals and comments.

Provides efficient bisect-based lookup to determine whether a character
position falls inside a string literal or comment, allowing quality rules
to ignore matches in non-code regions.
"""

from __future__ import annotations

import bisect

# Languages where # starts a line comment
HASH_COMMENT_LANGUAGES = frozenset({"python", "ruby", "perl", "shell", "bash"})

# Languages that support /* */ block comments
BLOCK_COMMENT_LANGUAGES = frozenset(
    {
        "typescript",
        "javascript",
        "java",
        "go",
        "rust",
        "c",
        "cpp",
        "c++",
        "c#",
        "kotlin",
        "swift",
        "csharp",
    }
)

# Languages that support `` ` `` template literals
TEMPLATE_LITERAL_LANGUAGES = frozenset({"typescript", "javascript"})


def build_skip_regions(code: str, language: str = "unknown") -> list[int]:
    """Build a flat sorted list of skip-region boundaries.

    Scans the code once to identify all string literal and comment regions.
    Returns a flat list of [start1, end1, start2, end2, ...] suitable for
    bisect-based lookup.

    Args:
        code: Source code to scan.
        language: Detected language (controls which comment syntaxes apply).

    Returns:
        Flat sorted list of region boundaries (even indices = start, odd = end).
    """
    boundaries: list[int] = []
    lang = language.lower()
    supports_hash = lang in HASH_COMMENT_LANGUAGES
    supports_block = lang in BLOCK_COMMENT_LANGUAGES or lang == "unknown"
    supports_template = lang in TEMPLATE_LITERAL_LANGUAGES

    i = 0
    length = len(code)

    while i < length:
        c = code[i]

        # --- Triple-quoted strings (Python) - must check before single quotes ---
        if c in ('"', "'") and i + 2 < length and code[i + 1] == c and code[i + 2] == c:
            quote = code[i : i + 3]
            start = i
            i += 3
            while i < length - 2:
                if code[i] == "\\":
                    i += 2
                    continue
                if code[i : i + 3] == quote:
                    i += 3
                    break
                i += 1
            else:
                i = length  # unterminated
            boundaries.extend((start, i))
            continue

        # --- Line comments: # (Python/Ruby) ---
        if supports_hash and c == "#":
            start = i
            newline = code.find("\n", i)
            i = newline + 1 if newline != -1 else length
            boundaries.extend((start, i))
            continue

        # --- Line comments: // (C-family) ---
        if c == "/" and i + 1 < length and code[i + 1] == "/":
            start = i
            newline = code.find("\n", i)
            i = newline + 1 if newline != -1 else length
            boundaries.extend((start, i))
            continue

        # --- Block comments: /* */ (C-family) ---
        if supports_block and c == "/" and i + 1 < length and code[i + 1] == "*":
            start = i
            end = code.find("*/", i + 2)
            i = end + 2 if end != -1 else length
            boundaries.extend((start, i))
            continue

        # --- Template literals: `...` (JS/TS) ---
        if supports_template and c == "`":
            start = i
            i += 1
            while i < length:
                if code[i] == "\\":
                    i += 2
                    continue
                if code[i] == "`":
                    i += 1
                    break
                i += 1
            else:
                i = length
            boundaries.extend((start, i))
            continue

        # --- Single/double quoted strings ---
        if c in ('"', "'"):
            quote = c
            start = i
            i += 1
            while i < length:
                if code[i] == "\\":
                    i += 2
                    continue
                if code[i] == quote:
                    i += 1
                    break
                if code[i] == "\n":
                    # Unterminated single-line string
                    break
                i += 1
            else:
                i = length
            boundaries.extend((start, i))
            continue

        i += 1

    return boundaries


def is_in_skip_region(position: int, boundaries: list[int]) -> bool:
    """Check if a position falls strictly inside any skip region.

    A match AT the opening delimiter (quote or comment marker) is NOT
    considered "inside" — only positions AFTER the delimiter are.  This
    preserves the correct behavior for rules that intentionally match
    comment text (``# type: ignore``) or string-concatenation patterns
    (``"SELECT ..." + var``).

    Uses binary search on the flat boundaries list for O(log n) lookup.

    Args:
        position: Character position to check.
        boundaries: Flat list from build_skip_regions.

    Returns:
        True if position is strictly inside a string literal or comment.
    """
    if not boundaries:
        return False
    # bisect_right gives us the insertion point.
    # If the index is odd, we're between a start and end (i.e., inside a region).
    idx = bisect.bisect_right(boundaries, position)
    if idx % 2 == 0:
        return False
    # Position is within [region_start, region_end).
    # Only skip if it's STRICTLY after the start (not at the delimiter itself).
    region_start = boundaries[idx - 1]
    return position > region_start
