"""AI001: Placeholder code detection rule."""

from __future__ import annotations

import ast
import re

from mirdan.core.rules.base import BaseRule, RuleContext
from mirdan.core.skip_regions import is_in_skip_region
from mirdan.models import Violation


class AI001PlaceholderRule(BaseRule):
    """Detect AI-generated placeholder code (AI001)."""

    _RE_NOT_IMPLEMENTED = re.compile(r"raise\s+NotImplementedError\b")
    _RE_PASS_WITH_TODO = re.compile(
        r"^(\s*)pass\s*(?:#.*)?$",
        re.MULTILINE,
    )
    _RE_TODO_COMMENT = re.compile(
        r"#\s*(?:todo|fixme|placeholder|hack)\b",
        re.IGNORECASE,
    )
    _RE_ELLIPSIS_PLACEHOLDER = re.compile(
        r"\.\.\.\s*#\s*(?:todo|fixme|placeholder|hack)\b",
        re.IGNORECASE,
    )
    _RE_ABSTRACT_METHOD = re.compile(r"@abstractmethod\b")
    _RE_DEF = re.compile(r"^(\s*)def\s+", re.MULTILINE)

    @property
    def id(self) -> str:
        return "AI001"

    @property
    def name(self) -> str:
        return "ai-placeholder-code"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "auto"})

    @property
    def is_quick(self) -> bool:
        return True

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        """Detect AI-generated placeholder code."""
        if language not in ("python", "auto"):
            return []

        violations: list[Violation] = []
        lines = code.split("\n")

        # Build set of lines that are inside @abstractmethod decorated functions
        abstract_lines = self._find_abstract_method_bodies(code)

        # Check: raise NotImplementedError (outside abstract methods)
        for m in self._RE_NOT_IMPLEMENTED.finditer(code):
            if is_in_skip_region(m.start(), context.skip_regions):
                continue
            line_no = code[: m.start()].count("\n") + 1
            if line_no in abstract_lines:
                continue
            violations.append(
                Violation(
                    id="AI001",
                    rule="ai-placeholder-code",
                    category="ai_quality",
                    severity="error",
                    message=(
                        "AI-generated placeholder: raise NotImplementedError."
                        " Replace with actual implementation."
                    ),
                    line=line_no,
                    suggestion="Remove placeholder and implement the function body",
                )
            )

        # Check: pass as sole body with TODO/FIXME comment nearby
        for m in self._RE_PASS_WITH_TODO.finditer(code):
            if is_in_skip_region(m.start(), context.skip_regions):
                continue
            line_no = code[: m.start()].count("\n") + 1
            if line_no in abstract_lines:
                continue
            # Check the full line (pass may have inline comment: "pass  # TODO")
            # and also look at surrounding lines (1 above, 1 below) for TODO comments
            pass_line = lines[line_no - 1] if line_no <= len(lines) else ""
            start_line_idx = line_no - 1
            context_start = max(0, start_line_idx - 1)
            context_end = min(len(lines), start_line_idx + 2)
            ctx = "\n".join(lines[context_start:context_end])
            if self._RE_TODO_COMMENT.search(pass_line) or self._RE_TODO_COMMENT.search(ctx):
                violations.append(
                    Violation(
                        id="AI001",
                        rule="ai-placeholder-code",
                        category="ai_quality",
                        severity="error",
                        message=(
                            "AI-generated placeholder: `pass` with TODO comment."
                            " Replace with actual implementation."
                        ),
                        line=line_no,
                        suggestion="Remove placeholder and implement the function body",
                    )
                )

        # Check: ... # todo/fixme/placeholder
        for m in self._RE_ELLIPSIS_PLACEHOLDER.finditer(code):
            if is_in_skip_region(m.start(), context.skip_regions):
                continue
            line_no = code[: m.start()].count("\n") + 1
            if line_no in abstract_lines:
                continue
            violations.append(
                Violation(
                    id="AI001",
                    rule="ai-placeholder-code",
                    category="ai_quality",
                    severity="error",
                    message=(
                        "AI-generated placeholder: ellipsis with TODO comment."
                        " Replace with actual implementation."
                    ),
                    line=line_no,
                    suggestion="Remove placeholder and implement the function body",
                )
            )

        return violations

    def _find_abstract_method_bodies(self, code: str) -> frozenset[int]:
        """Find line numbers that are inside @abstractmethod decorated functions.

        Returns a frozenset of 1-indexed line numbers for the body of any
        function decorated with @abstractmethod.
        """
        lines = code.split("\n")
        abstract_body_lines: set[int] = set()

        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if self._RE_ABSTRACT_METHOD.search(stripped):
                # Found decorator -- find the next def
                j = i + 1
                while j < len(lines) and not self._RE_DEF.match(lines[j]):
                    j += 1
                if j < len(lines):
                    # Found the def -- collect its body lines
                    def_indent = len(lines[j]) - len(lines[j].lstrip())
                    k = j + 1
                    while k < len(lines):
                        line = lines[k]
                        if line.strip() == "":
                            k += 1
                            continue
                        line_indent = len(line) - len(line.lstrip())
                        if line_indent <= def_indent:
                            break
                        abstract_body_lines.add(k + 1)  # 1-indexed
                        k += 1
                    i = k
                    continue
            i += 1

        return frozenset(abstract_body_lines)


def get_ast_confirmed_placeholder_lines(code: str, language: str) -> frozenset[int]:
    """Return line numbers of placeholder function bodies confirmed via Python AST.

    Parses the code and walks FunctionDef/AsyncFunctionDef nodes. A function
    body is a confirmed placeholder if (after stripping an optional leading
    docstring) it consists of a single: ``pass``, ``...`` (Ellipsis), or
    ``raise NotImplementedError``.  Functions decorated with ``@abstractmethod``
    are excluded.

    Falls back to an empty set for non-Python code or if the code has
    syntax errors (preserving existing regex-based heuristic behavior).

    Args:
        code: Source code to analyze.
        language: Detected programming language.

    Returns:
        Frozenset of 1-indexed line numbers where AST-confirmed placeholders live.
    """
    if language not in ("python", "auto"):
        return frozenset()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return frozenset()

    confirmed: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Skip @abstractmethod decorated functions
        if any(
            (isinstance(d, ast.Name) and d.id == "abstractmethod")
            or (isinstance(d, ast.Attribute) and d.attr == "abstractmethod")
            for d in node.decorator_list
        ):
            continue
        body = node.body
        # Strip optional leading docstring
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        if len(body) != 1:
            continue
        stmt = body[0]
        is_placeholder = False
        if isinstance(stmt, ast.Pass):
            is_placeholder = True
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            # Ellipsis: `...`
            if stmt.value.value is ...:
                is_placeholder = True
        elif isinstance(stmt, ast.Raise):
            # raise NotImplementedError or raise NotImplementedError(...)
            exc = stmt.exc
            if (isinstance(exc, ast.Name) and exc.id == "NotImplementedError") or (
                isinstance(exc, ast.Call)
                and isinstance(exc.func, ast.Name)
                and exc.func.id == "NotImplementedError"
            ):
                is_placeholder = True
        if is_placeholder:
            confirmed.add(stmt.lineno)
    return frozenset(confirmed)
