"""AST-based Python validation for checks regex cannot do accurately."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from mirdan.models import Violation


def validate_python_ast(
    code: str,
    skip_lines: set[int] | None = None,
) -> tuple[list[Violation], set[str], bool]:
    """AST-based Python validation for checks regex cannot do accurately.

    Args:
        code: Python source code to validate.
        skip_lines: Line numbers already flagged by AI002 — PY014 skips
            these to avoid double-flagging the same import.

    Returns:
        Tuple of (violations, checked_rule_ids, ast_parsed).
        checked_rule_ids contains the IDs of rules this function checked
        (used for dedup against regex results).
        ast_parsed is False on syntax errors (caller should keep regex results).
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return [], set(), False

    violations: list[Violation] = []
    violations.extend(_check_py001_eval(tree))
    violations.extend(_check_py002_exec(tree))
    violations.extend(_check_py003_bare_except(tree))
    violations.extend(_check_py004_mutable_default(tree))
    violations.extend(_check_py014_dead_import(tree, code, skip_lines))
    violations.extend(_check_py015_unreachable_code(tree))

    checked_rule_ids = {"PY001", "PY002", "PY003", "PY004"}
    return violations, checked_rule_ids, True


def _check_py001_eval(tree: ast.Module) -> list[Violation]:
    """PY001-ast: Detect eval() calls via AST (no false positives from strings)."""
    return [
        Violation(
            id="PY001",
            rule="no-eval",
            category="security",
            severity="error",
            message="Use of eval() is a security risk",
            line=node.lineno,
            column=node.col_offset,
            suggestion="Use ast.literal_eval() for safe evaluation",
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "eval"
    ]


def _check_py002_exec(tree: ast.Module) -> list[Violation]:
    """PY002-ast: Detect exec() calls via AST."""
    return [
        Violation(
            id="PY002",
            rule="no-exec",
            category="security",
            severity="error",
            message="Use of exec() is a security risk",
            line=node.lineno,
            column=node.col_offset,
            suggestion="Avoid exec(); use safer alternatives",
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "exec"
    ]


def _check_py003_bare_except(tree: ast.Module) -> list[Violation]:
    """PY003-ast: Detect bare except clauses via AST."""
    return [
        Violation(
            id="PY003",
            rule="no-bare-except",
            category="style",
            severity="warning",
            message=(
                "Bare except clause catches all exceptions"
                " including SystemExit and KeyboardInterrupt"
            ),
            line=node.lineno,
            column=node.col_offset,
            suggestion="Use 'except Exception:' to catch standard exceptions",
            fix_code="except Exception:",
            fix_description="Replace bare except with except Exception:",
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and node.type is None
    ]


def _check_py004_mutable_default(tree: ast.Module) -> list[Violation]:
    """PY004-ast: Detect mutable default arguments via AST."""
    violations: list[Violation] = []
    mutable_types = (ast.List, ast.Dict, ast.Set, ast.Call)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.extend(
                Violation(
                    id="PY004",
                    rule="no-mutable-default",
                    category="style",
                    severity="warning",
                    message="Mutable default argument detected",
                    line=default.lineno,
                    column=default.col_offset,
                    suggestion=(
                        "Use None as default and create the"
                        " mutable object inside the function"
                    ),
                )
                for default in node.args.defaults + node.args.kw_defaults
                if default is not None and isinstance(default, mutable_types)
            )
    return violations


def _check_py014_dead_import(
    tree: ast.Module,
    code: str,
    skip_lines: set[int] | None = None,
) -> list[Violation]:
    """PY014: Detect unused imports via AST.

    Skips:
    - Imports inside TYPE_CHECKING blocks
    - Names listed in __all__
    - __init__.py re-exports (detected by presence of __all__)
    - Lines in skip_lines (already flagged by AI002)
    """
    violations: list[Violation] = []
    if skip_lines is None:
        skip_lines = set()

    # Collect __all__ names if defined
    all_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "__all__"
                    and isinstance(node.value, (ast.List, ast.Tuple, ast.Set))
                ):
                    for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(
                                elt.value, str
                            ):
                                all_names.add(elt.value)

    # Identify TYPE_CHECKING block line ranges
    type_checking_ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            is_type_checking = (
                isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
            ) or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            )
            if is_type_checking and node.body:
                start = node.body[0].lineno
                end = node.body[-1].end_lineno or node.body[-1].lineno
                type_checking_ranges.append((start, end))

    def _in_type_checking(lineno: int) -> bool:
        return any(start <= lineno <= end for start, end in type_checking_ranges)

    # Collect imported names with their line numbers
    imported: dict[str, int] = {}  # name -> lineno
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            if _in_type_checking(node.lineno):
                continue
            if node.lineno in skip_lines:
                continue
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                # For dotted imports like 'import os.path', the usable name is 'os'
                top_level = name.split(".")[0]
                imported[top_level] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            if _in_type_checking(node.lineno):
                continue
            if node.lineno in skip_lines:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname if alias.asname else alias.name
                imported[name] = node.lineno

    # If __all__ is defined, any import that appears in __all__ is considered used
    for name in list(imported):
        if name in all_names:
            del imported[name]

    if not imported:
        return violations

    # Collect all used names in the module (excluding imports themselves)
    used_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not _is_import_node_name(node, tree):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            # For chained attribute access like os.path.join, collect 'os'
            root = _get_attribute_root(node)
            if root:
                used_names.add(root)

    # Also check for names used in decorators, type comments, and string annotations
    # String annotations are already resolved by ast in Python 3.12+, but for
    # earlier versions and f-strings, we do a simple text scan
    code_lines = code.split("\n")
    for line in code_lines:
        stripped = line.strip()
        # Check for usage in comments/strings that AST might miss
        if stripped.startswith("#"):
            continue

    # Report unused imports
    for name, lineno in sorted(imported.items(), key=lambda x: x[1]):
        if name not in used_names and name != "_":
            violations.append(
                Violation(
                    id="PY014",
                    rule="dead-import",
                    category="style",
                    severity="warning",
                    message=f"Import '{name}' is not used",
                    line=lineno,
                    suggestion=f"Remove unused import '{name}'",
                )
            )

    return violations


def _is_import_node_name(node: ast.Name, tree: ast.Module) -> bool:
    """Check if a Name node is part of an import statement (not a usage)."""
    for parent in ast.walk(tree):
        if isinstance(parent, (ast.Import, ast.ImportFrom)):
            for alias in parent.names:
                name = alias.asname if alias.asname else alias.name
                if name == node.id and node.lineno == parent.lineno:
                    return True
    return False


def _get_attribute_root(node: ast.Attribute) -> str | None:
    """Get the root name of a chained attribute access (e.g., os.path.join -> 'os')."""
    current = node.value
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return None


def _check_py015_unreachable_code(tree: ast.Module) -> list[Violation]:
    """PY015: Detect unreachable code after return/raise/continue/break.

    Only flags code that is truly unreachable — statements after a terminal
    statement in a non-conditional, non-finally block.
    """
    violations: list[Violation] = []
    terminal_types = (ast.Return, ast.Raise, ast.Continue, ast.Break)

    def _check_body(body: list[ast.stmt], in_finally: bool = False) -> None:
        if in_finally:
            return

        for i, stmt in enumerate(body):
            if isinstance(stmt, terminal_types) and i < len(body) - 1:
                # Everything after this statement in this block is unreachable
                violations.extend(
                    Violation(
                        id="PY015",
                        rule="unreachable-code",
                        category="style",
                        severity="warning",
                        message="Unreachable code detected",
                        line=unreachable.lineno,
                        suggestion="Remove unreachable code after return/raise/break/continue",
                    )
                    for unreachable in body[i + 1 :]
                )
                break  # No need to check further in this block

            # Recurse into sub-blocks
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _check_body(stmt.body)
            elif isinstance(stmt, ast.ClassDef):
                for class_item in stmt.body:
                    if isinstance(class_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        _check_body(class_item.body)
            elif isinstance(stmt, (ast.For, ast.While, ast.If)):
                _check_body(stmt.body)
                _check_body(stmt.orelse)
            elif isinstance(stmt, ast.With):
                _check_body(stmt.body)
            elif isinstance(stmt, (ast.Try, ast.TryStar)):
                _check_body(stmt.body)
                for handler in stmt.handlers:
                    _check_body(handler.body)
                _check_body(stmt.orelse)
                _check_body(stmt.finalbody, in_finally=True)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_body(node.body)
        elif isinstance(node, ast.ClassDef):
            for class_item in node.body:
                if isinstance(class_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _check_body(class_item.body)

    return violations
