"""Import extraction from source code.

Python: Uses ast module for accurate extraction.
Other languages: Regex-based detection of import/require/use statements.
Returns list of (module_path, line_number) tuples.
"""

from __future__ import annotations

import ast
import re


def extract_python_imports(code: str) -> list[tuple[str, int]]:
    """Extract imports from Python code using AST.

    Handles: import foo, from foo import bar, from foo.bar import baz.
    Returns (module_path, line_number) tuples.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append((module, node.lineno))
    return imports


# Regex patterns for non-Python languages
_JS_TS_IMPORT = re.compile(
    r"""(?:import\s+.*?\s+from\s+['"](.+?)['"]|require\s*\(\s*['"](.+?)['"]\s*\))"""
)
_GO_IMPORT = re.compile(r"""import\s+["'](.+?)["']""")
_RUST_USE = re.compile(r"use\s+([\w:]+)")
_JAVA_IMPORT = re.compile(r"import\s+([\w.]+)")


def extract_generic_imports(code: str, language: str) -> list[tuple[str, int]]:
    """Extract imports using regex for non-Python languages.

    Returns (module_path, line_number) tuples.
    """
    patterns = {
        "javascript": _JS_TS_IMPORT,
        "typescript": _JS_TS_IMPORT,
        "go": _GO_IMPORT,
        "rust": _RUST_USE,
        "java": _JAVA_IMPORT,
    }

    pattern = patterns.get(language)
    if pattern is None:
        return []

    imports: list[tuple[str, int]] = []
    for i, line in enumerate(code.split("\n"), 1):
        match = pattern.search(line)
        if match:
            # Use first non-None group
            module = next((g for g in match.groups() if g is not None), "")
            if module:
                imports.append((module, i))

    return imports


def extract_imports(code: str, language: str) -> list[tuple[str, int]]:
    """Dispatch to language-specific extractor.

    Args:
        code: Source code string.
        language: Language identifier (python, javascript, etc.).

    Returns:
        List of (module_path, line_number) tuples.
    """
    if language == "python":
        return extract_python_imports(code)
    return extract_generic_imports(code, language)
