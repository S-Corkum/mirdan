"""AI-specific code quality checks.

Detects common AI-generated code issues that traditional linters miss:
- AI001: Placeholder code (raise NotImplementedError, pass with TODO)
- AI002: Hallucinated imports (modules not in stdlib or project deps)
- AI008: Injection vulnerabilities via f-string interpolation
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from mirdan.core.code_validator import _build_skip_regions, _is_in_skip_region
from mirdan.models import Violation

# ---------------------------------------------------------------------------
# Python stdlib module names (3.11-3.13)
# ---------------------------------------------------------------------------

PYTHON_STDLIB_MODULES: frozenset[str] = frozenset({
    "__future__", "_thread", "abc", "aifc", "argparse", "array", "ast",
    "asynchat", "asyncio", "asyncore", "atexit", "audioop", "base64",
    "bdb", "binascii", "binhex", "bisect", "builtins", "bz2", "calendar",
    "cgi", "cgitb", "chunk", "cmath", "cmd", "code", "codecs", "codeop",
    "collections", "colorsys", "compileall", "concurrent", "configparser",
    "contextlib", "contextvars", "copy", "copyreg", "cProfile", "crypt",
    "csv", "ctypes", "curses", "dataclasses", "datetime", "dbm", "decimal",
    "difflib", "dis", "distutils", "doctest", "email", "encodings",
    "enum", "errno", "faulthandler", "fcntl", "filecmp", "fileinput",
    "fnmatch", "fractions", "ftplib", "functools", "gc", "getopt",
    "getpass", "gettext", "glob", "graphlib", "grp", "gzip", "hashlib",
    "heapq", "hmac", "html", "http", "idlelib", "imaplib", "imghdr",
    "imp", "importlib", "inspect", "io", "ipaddress", "itertools", "json",
    "keyword", "lib2to3", "linecache", "locale", "logging", "lzma",
    "mailbox", "mailcap", "marshal", "math", "mimetypes", "mmap",
    "modulefinder", "multiprocessing", "netrc", "nis", "nntplib",
    "numbers", "operator", "optparse", "os", "ossaudiodev",
    "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil",
    "platform", "plistlib", "poplib", "posix", "posixpath", "pprint",
    "profile", "pstats", "pty", "pwd", "py_compile", "pyclbr",
    "pydoc", "queue", "quopri", "random", "re", "readline", "reprlib",
    "resource", "rlcompleter", "runpy", "sched", "secrets", "select",
    "selectors", "shelve", "shlex", "shutil", "signal", "site", "smtpd",
    "smtplib", "sndhdr", "socket", "socketserver", "spwd", "sqlite3",
    "sre_compile", "sre_constants", "sre_parse", "ssl", "stat",
    "statistics", "string", "stringprep", "struct", "subprocess",
    "sunau", "symtable", "sys", "sysconfig", "syslog", "tabnanny",
    "tarfile", "telnetlib", "tempfile", "termios", "test", "textwrap",
    "threading", "time", "timeit", "tkinter", "token", "tokenize",
    "tomllib", "trace", "traceback", "tracemalloc", "tty", "turtle",
    "turtledemo", "types", "typing", "unicodedata", "unittest", "urllib",
    "uu", "uuid", "venv", "warnings", "wave", "weakref", "webbrowser",
    "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc",
    "zipapp", "zipfile", "zipimport", "zlib", "zoneinfo",
    # typing_extensions is not stdlib but is so ubiquitous we skip it
    "typing_extensions",
    # Common internal packages that appear as top-level
    "_abc", "_collections_abc", "_io", "_operator", "_signal",
    "_sitebuiltins", "_weakref",
    # Frequently used private stdlib modules
    "ntpath", "genericpath",
})

# Common third-party packages that are often transitive dependencies.
# When no project_dir is available, we give these the benefit of the doubt.
_COMMON_TRANSITIVE_PACKAGES: frozenset[str] = frozenset({
    "requests", "typing_extensions", "six", "certifi", "charset_normalizer",
    "idna", "urllib3", "packaging", "setuptools", "pip", "wheel",
    "attrs", "cattrs", "pydantic", "pydantic_core",
})


class AIQualityChecker:
    """Detects AI-specific code quality issues.

    Rules:
        AI001: Placeholder code detection
        AI002: Hallucinated import detection (Python only)
        AI008: Injection vulnerability via f-string interpolation
    """

    def __init__(self, project_dir: Path | None = None) -> None:
        self._project_dir = project_dir
        self._project_deps: frozenset[str] | None = None
        if project_dir:
            self._project_deps = self._load_project_deps(project_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, code: str, language: str) -> list[Violation]:
        """Run all AI-specific rules.

        Args:
            code: Source code to check.
            language: Detected programming language.

        Returns:
            List of AI-specific violations.
        """
        if not code or not code.strip():
            return []

        violations: list[Violation] = []
        skip_regions = _build_skip_regions(code, language)

        violations.extend(self._check_ai001_placeholders(code, language, skip_regions))
        violations.extend(self._check_ai002_hallucinated_imports(code, language))
        violations.extend(self._check_ai008_injection(code, language, skip_regions))

        return violations

    def check_quick(self, code: str, language: str) -> list[Violation]:
        """Run only critical AI rules (AI001, AI008).

        Args:
            code: Source code to check.
            language: Detected programming language.

        Returns:
            List of critical AI violations.
        """
        if not code or not code.strip():
            return []

        violations: list[Violation] = []
        skip_regions = _build_skip_regions(code, language)

        violations.extend(self._check_ai001_placeholders(code, language, skip_regions))
        violations.extend(self._check_ai008_injection(code, language, skip_regions))

        return violations

    # ------------------------------------------------------------------
    # AI001: Placeholder Detection
    # ------------------------------------------------------------------

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

    def _check_ai001_placeholders(
        self, code: str, language: str, skip_regions: list[int],
    ) -> list[Violation]:
        """Detect AI-generated placeholder code."""
        if language not in ("python", "auto"):
            return []

        violations: list[Violation] = []
        lines = code.split("\n")

        # Build set of lines that are inside @abstractmethod decorated functions
        abstract_lines = self._find_abstract_method_bodies(code)

        # Check: raise NotImplementedError (outside abstract methods)
        for m in self._RE_NOT_IMPLEMENTED.finditer(code):
            if _is_in_skip_region(m.start(), skip_regions):
                continue
            line_no = code[:m.start()].count("\n") + 1
            if line_no in abstract_lines:
                continue
            violations.append(Violation(
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
            ))

        # Check: pass as sole body with TODO/FIXME comment nearby
        for m in self._RE_PASS_WITH_TODO.finditer(code):
            if _is_in_skip_region(m.start(), skip_regions):
                continue
            line_no = code[:m.start()].count("\n") + 1
            if line_no in abstract_lines:
                continue
            # Check the full line (pass may have inline comment: "pass  # TODO")
            # and also look at surrounding lines (1 above, 1 below) for TODO comments
            pass_line = lines[line_no - 1] if line_no <= len(lines) else ""
            start_line_idx = line_no - 1
            context_start = max(0, start_line_idx - 1)
            context_end = min(len(lines), start_line_idx + 2)
            context = "\n".join(lines[context_start:context_end])
            if self._RE_TODO_COMMENT.search(pass_line) or self._RE_TODO_COMMENT.search(context):
                violations.append(Violation(
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
                ))

        # Check: ... # todo/fixme/placeholder
        for m in self._RE_ELLIPSIS_PLACEHOLDER.finditer(code):
            if _is_in_skip_region(m.start(), skip_regions):
                continue
            line_no = code[:m.start()].count("\n") + 1
            if line_no in abstract_lines:
                continue
            violations.append(Violation(
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
            ))

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
                # Found decorator — find the next def
                j = i + 1
                while j < len(lines) and not self._RE_DEF.match(lines[j]):
                    j += 1
                if j < len(lines):
                    # Found the def — collect its body lines
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

    # ------------------------------------------------------------------
    # AI002: Hallucinated Import Detection
    # ------------------------------------------------------------------

    _RE_IMPORT = re.compile(r"^import\s+(\w+)", re.MULTILINE)
    _RE_FROM_IMPORT = re.compile(r"^from\s+(\w+)", re.MULTILINE)

    def _check_ai002_hallucinated_imports(
        self, code: str, language: str,
    ) -> list[Violation]:
        """Detect imports for packages not in stdlib or project deps."""
        if language not in ("python", "auto"):
            return []

        # Skip if no project_dir — can't verify without context
        if self._project_deps is None:
            return []

        violations: list[Violation] = []
        imported_modules: set[tuple[str, int]] = set()

        for m in self._RE_IMPORT.finditer(code):
            module = m.group(1)
            line_no = code[:m.start()].count("\n") + 1
            imported_modules.add((module, line_no))

        for m in self._RE_FROM_IMPORT.finditer(code):
            module = m.group(1)
            # Skip relative imports (from . import ...)
            if module == ".":
                continue
            line_no = code[:m.start()].count("\n") + 1
            imported_modules.add((module, line_no))

        for module, line_no in imported_modules:
            # Skip __future__ imports
            if module == "__future__":
                continue
            # Skip relative import markers
            if module.startswith("."):
                continue
            if not self._is_known_module(module):
                violations.append(Violation(
                    id="AI002",
                    rule="ai-hallucinated-import",
                    category="ai_quality",
                    severity="warning",
                    message=(
                        f"Import '{module}' not found in project dependencies"
                        " or Python stdlib. Verify this package is installed."
                    ),
                    line=line_no,
                    suggestion=(
                        f"Check if '{module}' is a valid package"
                        " and add it to project dependencies"
                    ),
                ))

        return violations

    def _is_known_module(self, module: str) -> bool:
        """Check if a module name is known (stdlib, project dep, or common transitive)."""
        if module in PYTHON_STDLIB_MODULES:
            return True
        if self._project_deps is not None and module in self._project_deps:
            return True
        # Also check if the module is available in the current Python's stdlib
        return module in sys.stdlib_module_names

    def _load_project_deps(self, project_dir: Path) -> frozenset[str]:
        """Load project dependency names from pyproject.toml or requirements.txt.

        Returns normalized top-level package names (hyphens → underscores).
        """
        deps: set[str] = set()

        # Try pyproject.toml
        pyproject = project_dir / "pyproject.toml"
        if pyproject.exists():
            deps.update(self._parse_pyproject_deps(pyproject))

        # Try requirements.txt as fallback
        requirements = project_dir / "requirements.txt"
        if requirements.exists():
            deps.update(self._parse_requirements_deps(requirements))

        # Try package.json (for rare Python/Node hybrid projects)
        # Not needed — AI002 is Python-only

        # Add common transitive dependencies
        deps.update(_COMMON_TRANSITIVE_PACKAGES)

        # Also add the project's own package name
        deps.update(self._find_local_packages(project_dir))

        return frozenset(deps)

    def _parse_pyproject_deps(self, path: Path) -> set[str]:
        """Extract dependency names from pyproject.toml."""
        deps: set[str] = set()
        try:
            import tomllib

            with path.open("rb") as f:
                data = tomllib.load(f)

            # [project.dependencies]
            for dep_str in data.get("project", {}).get("dependencies", []):
                name = self._extract_package_name(dep_str)
                if name:
                    deps.add(name)

            # [project.optional-dependencies]
            for group_deps in data.get("project", {}).get("optional-dependencies", {}).values():
                for dep_str in group_deps:
                    name = self._extract_package_name(dep_str)
                    if name:
                        deps.add(name)

            # [tool.poetry.dependencies] (Poetry)
            for name in data.get("tool", {}).get("poetry", {}).get("dependencies", {}):
                deps.add(self._normalize_package_name(name))

            # [tool.poetry.dev-dependencies] (Poetry)
            for name in data.get("tool", {}).get("poetry", {}).get("dev-dependencies", {}):
                deps.add(self._normalize_package_name(name))

        except Exception:  # noqa: S110
            pass  # Graceful degradation: can't parse pyproject.toml

        return deps

    def _parse_requirements_deps(self, path: Path) -> set[str]:
        """Extract dependency names from requirements.txt."""
        deps: set[str] = set()
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                name = self._extract_package_name(line)
                if name:
                    deps.add(name)
        except Exception:  # noqa: S110
            pass  # Graceful degradation: can't parse requirements.txt
        return deps

    def _extract_package_name(self, dep_string: str) -> str | None:
        """Extract and normalize package name from a dependency specifier."""
        # Handle: "package>=1.0", "package[extra]>=1.0", "package; python_version>='3.8'"
        m = re.match(r"([A-Za-z0-9][\w.-]*)", dep_string.strip())
        if m:
            return self._normalize_package_name(m.group(1))
        return None

    @staticmethod
    def _normalize_package_name(name: str) -> str:
        """Normalize package name: lowercase, hyphens to underscores."""
        return re.sub(r"[-.]", "_", name.lower())

    def _find_local_packages(self, project_dir: Path) -> set[str]:
        """Find local package names (src layout or flat layout)."""
        packages: set[str] = set()
        src_dir = project_dir / "src"
        search_dirs = [src_dir] if src_dir.is_dir() else [project_dir]
        for d in search_dirs:
            if not d.is_dir():
                continue
            for child in d.iterdir():
                if child.is_dir() and (child / "__init__.py").exists():
                    packages.add(child.name)
        return packages

    # ------------------------------------------------------------------
    # AI008: Injection Vulnerability via f-string
    # ------------------------------------------------------------------

    _INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (
            re.compile(r"""f["'].*?(?:SELECT|INSERT|UPDATE|DELETE|DROP)\b.*?\{""", re.IGNORECASE),
            "SQL query built with f-string interpolation",
        ),
        (
            re.compile(r"""\beval\s*\(\s*f["']"""),
            "eval() with f-string input",
        ),
        (
            re.compile(r"""\bexec\s*\(\s*f["']"""),
            "exec() with f-string input",
        ),
        (
            re.compile(r"""\bos\.system\s*\(\s*f["']"""),
            "os.system() with f-string command",
        ),
        (
            re.compile(r"""\bsubprocess\.\w+\s*\([^)]*f["']"""),
            "subprocess call with f-string command",
        ),
    ]

    def _check_ai008_injection(
        self, code: str, language: str, skip_regions: list[int],
    ) -> list[Violation]:
        """Detect injection vulnerabilities via f-string interpolation."""
        # This rule applies to Python (f-strings) — skip other languages
        if language not in ("python", "auto"):
            return []

        violations: list[Violation] = []

        for pattern, context in self._INJECTION_PATTERNS:
            for m in pattern.finditer(code):
                if _is_in_skip_region(m.start(), skip_regions):
                    continue
                line_no = code[:m.start()].count("\n") + 1
                violations.append(Violation(
                    id="AI008",
                    rule="ai-injection-vulnerability",
                    category="security",
                    severity="error",
                    message=(
                        f"Potential injection vulnerability: {context}."
                        " Use parameterized queries or input sanitization."
                    ),
                    line=line_no,
                    suggestion=(
                        "Use parameterized queries"
                        " (e.g., cursor.execute('SELECT ...', (param,)))"
                        " instead of f-string interpolation"
                    ),
                ))

        return violations
