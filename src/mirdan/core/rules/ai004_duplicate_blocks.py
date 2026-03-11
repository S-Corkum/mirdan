"""AI004: Duplicate code block detection rule."""

from __future__ import annotations

import hashlib
import re

from mirdan.core.rules.base import BaseRule, RuleContext
from mirdan.core.skip_regions import is_in_skip_region
from mirdan.models import Violation


class AI004DuplicateBlocksRule(BaseRule):
    """Detect duplicate code blocks — function/method bodies (AI004)."""

    _RE_FUNC_BODY = re.compile(
        r"^(\s*)(?:def|function|fn|func)\s+\w+.*?[:{]\s*$",
        re.MULTILINE,
    )

    @property
    def id(self) -> str:
        return "AI004"

    @property
    def name(self) -> str:
        return "ai-duplicate-code"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "typescript", "javascript", "go", "rust", "java", "auto"})

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        """Detect duplicate code blocks (function/method bodies)."""
        violations: list[Violation] = []
        lines = code.split("\n")

        # Extract function bodies as (start_line, body_text)
        bodies: list[tuple[int, str]] = []

        i = 0
        while i < len(lines):
            line = lines[i]
            # Detect function/method definitions
            _func_re = r"^\s*(?:def|function|fn|func|pub\s+fn|async\s+def|async\s+function)\s+\w+"
            if re.match(_func_re, line):
                char_offset = sum(len(ln) + 1 for ln in lines[:i])
                if is_in_skip_region(char_offset, context.skip_regions):
                    i += 1
                    continue
                func_indent = len(line) - len(line.lstrip())
                body_lines: list[str] = []
                j = i + 1
                while j < len(lines):
                    bl = lines[j]
                    if bl.strip() == "":
                        body_lines.append("")
                        j += 1
                        continue
                    bl_indent = len(bl) - len(bl.lstrip())
                    if bl_indent <= func_indent:
                        break
                    # Normalize: strip leading whitespace beyond func_indent
                    body_lines.append(bl[func_indent:].rstrip())
                    j += 1
                if len(body_lines) > 5:  # Only check bodies > 5 lines
                    body_text = "\n".join(body_lines).strip()
                    bodies.append((i + 1, body_text))  # 1-indexed line
                i = j
            else:
                i += 1

        # Hash bodies and find duplicates
        seen: dict[str, int] = {}
        for line_no, body in bodies:
            # Normalize: collapse whitespace, strip comments
            normalized = re.sub(r"#.*$", "", body, flags=re.MULTILINE)
            normalized = re.sub(r"//.*$", "", normalized, flags=re.MULTILINE)
            normalized = re.sub(r"\s+", " ", normalized).strip()
            h = hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()
            if h in seen:
                violations.append(
                    Violation(
                        id="AI004",
                        rule="ai-duplicate-code",
                        category="ai_quality",
                        severity="warning",
                        message=(
                            f"Function body at line {line_no} is a duplicate of the"
                            f" function body at line {seen[h]}. Consider extracting"
                            " shared logic."
                        ),
                        line=line_no,
                        suggestion="Extract the duplicated logic into a shared helper function",
                    )
                )
            else:
                seen[h] = line_no

        return violations
