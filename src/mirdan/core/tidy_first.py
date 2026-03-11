"""Tidy First refactoring intelligence.

Analyzes target files for preparatory refactoring opportunities before
the main task begins. Implements Kent Beck's "Tidy First" principle:
make the change easy, then make the easy change.

Pure analysis: reads files from disk, uses AST (Python) or regex
(other languages), returns deterministic suggestions. No side effects.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from mirdan.models import EntityType, Intent, TidyFirstAnalysis, TidySuggestion

if TYPE_CHECKING:
    from mirdan.config import TidyFirstConfig

logger = logging.getLogger(__name__)

# Effort ordering for sorting suggestions
_EFFORT_ORDER = {"trivial": 0, "small": 1, "medium": 2}


class TidyFirstAnalyzer:
    """Analyze target files for preparatory refactoring opportunities.

    Runs during enhance_prompt to surface optional structural improvements
    before the main task begins. Config-gated and deterministic.
    """

    def __init__(self, config: TidyFirstConfig) -> None:
        self._config = config

    def analyze(self, intent: Intent) -> TidyFirstAnalysis:
        """Analyze files referenced in intent for tidy opportunities.

        Extracts FILE_PATH entities from intent, reads each file from disk,
        and analyzes for structural issues. Caps at max_suggestions.

        Args:
            intent: The analyzed intent containing file entity references.

        Returns:
            TidyFirstAnalysis with suggestions, target files, and skipped files.
        """
        if not self._config.enabled:
            return TidyFirstAnalysis()

        file_entities = [e.value for e in intent.entities if e.type == EntityType.FILE_PATH]
        if not file_entities:
            return TidyFirstAnalysis()

        # Limit analysis to first 5 files
        file_entities = file_entities[:5]

        all_suggestions: list[TidySuggestion] = []
        target_files: list[str] = []
        skipped_files: list[str] = []

        for file_path_str in file_entities:
            # Resolve file path
            path = self._resolve_path(file_path_str)
            if path is None:
                skipped_files.append(file_path_str)
                continue

            # Check file size
            try:
                size = path.stat().st_size
            except OSError:
                skipped_files.append(file_path_str)
                continue

            if size > self._config.max_file_size_kb * 1024:
                skipped_files.append(file_path_str)
                continue

            # Read file
            try:
                code = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                skipped_files.append(file_path_str)
                continue

            target_files.append(file_path_str)

            # Detect language from extension
            language = self._detect_language(path)
            suggestions = self._analyze_file(code, file_path_str, language)
            all_suggestions.extend(suggestions)

        # Sort by effort (trivial first) and cap at max
        all_suggestions.sort(key=lambda s: _EFFORT_ORDER.get(s.effort, 99))
        capped = all_suggestions[: self._config.max_suggestions]

        return TidyFirstAnalysis(
            suggestions=capped,
            target_files=target_files,
            skipped_files=skipped_files,
        )

    def _resolve_path(self, file_path_str: str) -> Path | None:
        """Resolve a file path string to an actual Path on disk."""
        path = Path(file_path_str)
        if path.is_file():
            return path
        cwd_path = Path.cwd() / file_path_str
        if cwd_path.is_file():
            return cwd_path
        return None

    def _detect_language(self, path: Path) -> str:
        """Detect language from file extension."""
        ext = path.suffix.lower()
        return {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".rb": "ruby",
        }.get(ext, "unknown")

    def _analyze_file(
        self, code: str, file_path: str, language: str
    ) -> list[TidySuggestion]:
        """Analyze a single file for tidy opportunities."""
        suggestions: list[TidySuggestion] = []

        # File size check (language-agnostic)
        suggestions.extend(self._check_file_size(code, file_path))

        if language == "python":
            suggestions.extend(self._analyze_python_ast(code, file_path))
        else:
            suggestions.extend(self._analyze_generic(code, file_path, language))

        return suggestions

    def _analyze_python_ast(
        self, code: str, file_path: str
    ) -> list[TidySuggestion]:
        """Python-specific AST analysis for extract_method and nesting."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fall back to generic regex analysis
            return self._analyze_generic(code, file_path, "python")

        suggestions: list[TidySuggestion] = []
        suggestions.extend(self._check_long_functions_ast(tree, file_path))
        suggestions.extend(self._check_deep_nesting_ast(tree, file_path))
        return suggestions

    def _analyze_generic(
        self, code: str, file_path: str, language: str
    ) -> list[TidySuggestion]:
        """Language-agnostic regex analysis for nesting depth."""
        suggestions: list[TidySuggestion] = []

        # Check for deep nesting via indentation
        lines = code.split("\n")
        max_indent = 0
        max_indent_line = 0
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if not stripped or stripped.startswith(("#", "//", "/*", "*")):
                continue
            indent = len(line) - len(stripped)
            # Estimate nesting depth (4 spaces or 1 tab = 1 level)
            depth = indent // 4 if "\t" not in line else line.count("\t")
            if depth > max_indent:
                max_indent = depth
                max_indent_line = i

        if max_indent >= self._config.min_nesting_depth:
            suggestions.append(
                TidySuggestion(
                    type="simplify_conditional",
                    file_path=file_path,
                    line=max_indent_line,
                    description=(
                        f"Maximum nesting depth of {max_indent} levels — "
                        "consider extracting nested logic or using early returns."
                    ),
                    effort="small",
                    reason="Deep nesting reduces readability and increases cognitive load.",
                )
            )

        return suggestions

    def _check_long_functions_ast(
        self, tree: ast.Module, file_path: str
    ) -> list[TidySuggestion]:
        """Find functions exceeding min_function_length via AST."""
        suggestions: list[TidySuggestion] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.end_lineno is None:
                continue
            body_length = node.end_lineno - node.lineno
            if body_length > self._config.min_function_length:
                suggestions.append(
                    TidySuggestion(
                        type="extract_method",
                        file_path=file_path,
                        line=node.lineno,
                        description=(
                            f"Function '{node.name}' is {body_length} lines long — "
                            "consider extracting distinct logical sections into helpers."
                        ),
                        effort="small",
                        reason="Long functions are harder to understand and test.",
                    )
                )

        return suggestions

    def _check_deep_nesting_ast(
        self, tree: ast.Module, file_path: str
    ) -> list[TidySuggestion]:
        """Find deeply nested blocks via AST walk."""
        suggestions: list[TidySuggestion] = []
        nesting_nodes = (ast.If, ast.For, ast.While, ast.With, ast.Try)

        def _walk_depth(node: ast.AST, depth: int = 0) -> tuple[int, int]:
            """Return (max_depth, line_of_max_depth)."""
            max_depth = depth
            max_line = getattr(node, "lineno", 0)

            for child in ast.iter_child_nodes(node):
                if isinstance(child, nesting_nodes):
                    child_depth, child_line = _walk_depth(child, depth + 1)
                    if child_depth > max_depth:
                        max_depth = child_depth
                        max_line = child_line
                else:
                    child_depth, child_line = _walk_depth(child, depth)
                    if child_depth > max_depth:
                        max_depth = child_depth
                        max_line = child_line

            return max_depth, max_line

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            max_depth, max_line = _walk_depth(node)
            if max_depth >= self._config.min_nesting_depth:
                suggestions.append(
                    TidySuggestion(
                        type="simplify_conditional",
                        file_path=file_path,
                        line=max_line,
                        description=(
                            f"Function '{node.name}' has {max_depth} levels of nesting — "
                            "consider using early returns or extracting inner logic."
                        ),
                        effort="small",
                        reason="Deep nesting reduces readability and increases cognitive load.",
                    )
                )

        return suggestions

    def _check_file_size(
        self, code: str, file_path: str
    ) -> list[TidySuggestion]:
        """Suggest split_file if file exceeds threshold."""
        non_empty_lines = sum(1 for line in code.split("\n") if line.strip())
        if non_empty_lines > 300:
            return [
                TidySuggestion(
                    type="split_file",
                    file_path=file_path,
                    description=(
                        f"File has {non_empty_lines} non-empty lines — "
                        "consider splitting into focused modules."
                    ),
                    effort="medium",
                    reason="Large files are harder to navigate and have more merge conflicts.",
                )
            ]
        return []
