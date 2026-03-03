"""AI-specific code quality checks.

Detects common AI-generated code issues that traditional linters miss:
- AI001: Placeholder code (raise NotImplementedError, pass with TODO)
- AI002: Hallucinated imports (modules not in stdlib or project deps)
- AI003: Over-engineering detection (unnecessary abstractions)
- AI004: Duplicate code block detection
- AI005: Inconsistent error handling patterns
- AI006: Unnecessary heavy imports
- AI007: Security theater detection
- AI008: Injection vulnerabilities via f-string interpolation
"""

from __future__ import annotations

import hashlib
import json
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


# ---------------------------------------------------------------------------
# Node.js built-in modules
# ---------------------------------------------------------------------------

NODE_BUILTIN_MODULES: frozenset[str] = frozenset({
    "assert", "async_hooks", "buffer", "child_process", "cluster",
    "console", "constants", "crypto", "dgram", "diagnostics_channel",
    "dns", "domain", "events", "fs", "http", "http2", "https",
    "inspector", "module", "net", "os", "path", "perf_hooks",
    "process", "punycode", "querystring", "readline", "repl",
    "stream", "string_decoder", "timers", "tls", "trace_events",
    "tty", "url", "util", "v8", "vm", "wasi", "worker_threads",
    "zlib",
    # Node: prefixed versions
    "node:assert", "node:async_hooks", "node:buffer",
    "node:child_process", "node:cluster", "node:console",
    "node:crypto", "node:dgram", "node:diagnostics_channel",
    "node:dns", "node:domain", "node:events", "node:fs",
    "node:http", "node:http2", "node:https", "node:inspector",
    "node:module", "node:net", "node:os", "node:path",
    "node:perf_hooks", "node:process", "node:querystring",
    "node:readline", "node:repl", "node:stream",
    "node:string_decoder", "node:timers", "node:tls",
    "node:trace_events", "node:tty", "node:url", "node:util",
    "node:v8", "node:vm", "node:wasi", "node:worker_threads",
    "node:zlib",
})

# ---------------------------------------------------------------------------
# Go standard library top-level packages
# ---------------------------------------------------------------------------

GO_STDLIB_PACKAGES: frozenset[str] = frozenset({
    "archive", "bufio", "bytes", "compress", "container", "context",
    "crypto", "database", "debug", "embed", "encoding", "errors",
    "expvar", "flag", "fmt", "go", "hash", "html", "image", "index",
    "internal", "io", "iter", "log", "maps", "math", "mime", "net",
    "os", "path", "plugin", "reflect", "regexp", "runtime", "slices",
    "sort", "strconv", "strings", "structs", "sync", "syscall",
    "testing", "text", "time", "unicode", "unique", "unsafe", "weak",
})

# ---------------------------------------------------------------------------
# Heavy import alternatives (AI006)
# ---------------------------------------------------------------------------

_HEAVY_IMPORT_ALTERNATIVES: dict[str, dict[str, str]] = {
    "requests": {
        "pattern": r"\brequests\.get\s*\(",
        "alternative": "urllib.request.urlopen",
        "reason": "For simple GET requests, urllib.request avoids the requests dependency",
    },
    "pandas": {
        "pattern": r"\bpd\.read_csv\s*\([^)]*\)\s*$",
        "alternative": "csv module",
        "reason": "For simple CSV reading, the csv module avoids the heavy pandas dependency",
    },
    "numpy": {
        "pattern": r"\bnp\.(?:sum|mean|max|min|abs|sqrt)\s*\(\s*\[",
        "alternative": "math/statistics module",
        "reason": "For basic math on small lists, math/statistics avoids the numpy dependency",
    },
}


class AIQualityChecker:
    """Detects AI-specific code quality issues.

    Rules:
        AI001: Placeholder code detection
        AI002: Hallucinated import detection (Python, TypeScript, Go, Rust)
        AI003: Over-engineering detection
        AI004: Duplicate code block detection
        AI005: Inconsistent error handling patterns
        AI006: Unnecessary heavy imports
        AI007: Security theater detection
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
        violations.extend(self._check_ai003_over_engineering(code, language, skip_regions))
        violations.extend(self._check_ai004_duplicate_blocks(code, language, skip_regions))
        violations.extend(self._check_ai005_inconsistent_errors(code, language, skip_regions))
        violations.extend(self._check_ai006_heavy_imports(code, language, skip_regions))
        violations.extend(self._check_ai007_security_theater(code, language, skip_regions))
        violations.extend(self._check_ai008_injection(code, language, skip_regions))

        return violations

    def check_quick(self, code: str, language: str) -> list[Violation]:
        """Run only critical AI rules (AI001, AI007, AI008).

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
        violations.extend(self._check_ai007_security_theater(code, language, skip_regions))
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

    # Python imports
    _RE_IMPORT = re.compile(r"^import\s+(\w+)", re.MULTILINE)
    _RE_FROM_IMPORT = re.compile(r"^from\s+(\w+)", re.MULTILINE)

    # TypeScript/JavaScript imports
    _RE_TS_IMPORT_FROM = re.compile(
        r"""(?:^|\n)\s*import\s+.*?\s+from\s+['"]([^'"./][^'"]*?)['"]""",
    )
    _RE_TS_REQUIRE = re.compile(
        r"""\brequire\s*\(\s*['"]([^'"./][^'"]*?)['"]""",
    )

    # Go imports
    _RE_GO_IMPORT_SINGLE = re.compile(r'^\s*import\s+"([^"]+)"', re.MULTILINE)
    _RE_GO_IMPORT_BLOCK = re.compile(
        r"import\s*\((.*?)\)", re.DOTALL,
    )
    _RE_GO_IMPORT_LINE = re.compile(r'"([^"]+)"')

    # Rust imports
    _RE_RUST_USE = re.compile(r"^\s*use\s+(\w+)::", re.MULTILINE)

    def _check_ai002_hallucinated_imports(
        self, code: str, language: str,
    ) -> list[Violation]:
        """Detect imports for packages not in stdlib or project deps."""
        if language in ("python", "auto"):
            return self._check_ai002_python(code)
        if language in ("typescript", "javascript"):
            return self._check_ai002_typescript(code)
        if language == "go":
            return self._check_ai002_go(code)
        if language == "rust":
            return self._check_ai002_rust(code)
        return []

    def _check_ai002_python(self, code: str) -> list[Violation]:
        """AI002 for Python imports."""
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
            if module == ".":
                continue
            line_no = code[:m.start()].count("\n") + 1
            imported_modules.add((module, line_no))

        for module, line_no in imported_modules:
            if module == "__future__":
                continue
            if module.startswith("."):
                continue
            if not self._is_known_python_module(module):
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

    def _check_ai002_typescript(self, code: str) -> list[Violation]:
        """AI002 for TypeScript/JavaScript imports."""
        if self._project_deps is None:
            return []

        violations: list[Violation] = []
        imported_modules: set[tuple[str, int]] = set()

        for m in self._RE_TS_IMPORT_FROM.finditer(code):
            raw = m.group(1)
            # Extract package name (handle scoped: @scope/pkg → @scope/pkg)
            module = self._extract_npm_package_name(raw)
            line_no = code[:m.start()].count("\n") + 1
            imported_modules.add((module, line_no))

        for m in self._RE_TS_REQUIRE.finditer(code):
            raw = m.group(1)
            module = self._extract_npm_package_name(raw)
            line_no = code[:m.start()].count("\n") + 1
            imported_modules.add((module, line_no))

        for module, line_no in imported_modules:
            if module in NODE_BUILTIN_MODULES:
                continue
            if module.startswith("node:"):
                continue
            if module in self._project_deps:
                continue
            violations.append(Violation(
                id="AI002",
                rule="ai-hallucinated-import",
                category="ai_quality",
                severity="warning",
                message=(
                    f"Import '{module}' not found in package.json dependencies"
                    " or Node.js built-ins. Verify this package is installed."
                ),
                line=line_no,
                suggestion=(
                    f"Check if '{module}' is a valid npm package"
                    " and add it to package.json"
                ),
            ))

        return violations

    def _check_ai002_go(self, code: str) -> list[Violation]:
        """AI002 for Go imports."""
        if self._project_deps is None:
            return []

        violations: list[Violation] = []
        imported_paths: set[tuple[str, int]] = set()

        # Single import statements
        for m in self._RE_GO_IMPORT_SINGLE.finditer(code):
            path = m.group(1)
            line_no = code[:m.start()].count("\n") + 1
            imported_paths.add((path, line_no))

        # Block import statements
        for m in self._RE_GO_IMPORT_BLOCK.finditer(code):
            block = m.group(1)
            block_start = code[:m.start()].count("\n") + 1
            for i, line in enumerate(block.split("\n")):
                inner = self._RE_GO_IMPORT_LINE.search(line)
                if inner:
                    imported_paths.add((inner.group(1), block_start + i))

        for path, line_no in imported_paths:
            top_pkg = path.split("/")[0]
            # Standard library: no dots in top-level package
            if "." not in top_pkg and top_pkg in GO_STDLIB_PACKAGES:
                continue
            # Third-party: check against go.mod deps
            if path in self._project_deps:
                continue
            # Check module prefix match: import path should start with a known dep
            # e.g., "github.com/foo/bar/sub" starts with dep "github.com/foo/bar"
            if any(
                path.startswith(dep + "/") or path == dep
                for dep in self._project_deps
            ):
                continue
            # Skip stdlib paths
            if "." not in top_pkg:
                continue
            violations.append(Violation(
                id="AI002",
                rule="ai-hallucinated-import",
                category="ai_quality",
                severity="warning",
                message=(
                    f"Import '{path}' not found in go.mod dependencies."
                    " Verify this module is required."
                ),
                line=line_no,
                suggestion=(
                    f"Run 'go get {path}' to add the dependency"
                ),
            ))

        return violations

    def _check_ai002_rust(self, code: str) -> list[Violation]:
        """AI002 for Rust imports."""
        if self._project_deps is None:
            return []

        violations: list[Violation] = []
        # Rust built-in crates
        rust_builtins = frozenset({
            "std", "core", "alloc", "proc_macro", "test",
            "self", "super", "crate",
        })

        for m in self._RE_RUST_USE.finditer(code):
            crate_name = m.group(1)
            line_no = code[:m.start()].count("\n") + 1
            if crate_name in rust_builtins:
                continue
            if crate_name in self._project_deps:
                continue
            violations.append(Violation(
                id="AI002",
                rule="ai-hallucinated-import",
                category="ai_quality",
                severity="warning",
                message=(
                    f"Crate '{crate_name}' not found in Cargo.toml dependencies."
                    " Verify this crate is added."
                ),
                line=line_no,
                suggestion=(
                    f"Run 'cargo add {crate_name}' to add the dependency"
                ),
            ))

        return violations

    @staticmethod
    def _extract_npm_package_name(raw: str) -> str:
        """Extract npm package name from import path.

        Handles scoped packages: '@scope/pkg/subpath' -> '@scope/pkg'
        Regular packages: 'pkg/subpath' -> 'pkg'
        """
        if raw.startswith("@"):
            parts = raw.split("/")
            return "/".join(parts[:2]) if len(parts) >= 2 else raw
        return raw.split("/")[0]

    def _is_known_python_module(self, module: str) -> bool:
        """Check if a Python module name is known (stdlib, project dep, or common transitive)."""
        if module in PYTHON_STDLIB_MODULES:
            return True
        if self._project_deps is not None and module in self._project_deps:
            return True
        return module in sys.stdlib_module_names

    # Keep backward compatibility
    _is_known_module = _is_known_python_module

    def _load_project_deps(self, project_dir: Path) -> frozenset[str]:
        """Load project dependency names from manifests.

        Supports: pyproject.toml, requirements.txt, package.json, go.mod, Cargo.toml.
        Returns normalized top-level package names.
        """
        deps: set[str] = set()

        # Python: pyproject.toml
        pyproject = project_dir / "pyproject.toml"
        if pyproject.exists():
            deps.update(self._parse_pyproject_deps(pyproject))

        # Python: requirements.txt
        requirements = project_dir / "requirements.txt"
        if requirements.exists():
            deps.update(self._parse_requirements_deps(requirements))

        # TypeScript/JavaScript: package.json
        package_json = project_dir / "package.json"
        if package_json.exists():
            deps.update(self._parse_package_json_deps(package_json))

        # Go: go.mod
        go_mod = project_dir / "go.mod"
        if go_mod.exists():
            deps.update(self._parse_go_mod_deps(go_mod))

        # Rust: Cargo.toml
        cargo_toml = project_dir / "Cargo.toml"
        if cargo_toml.exists():
            deps.update(self._parse_cargo_toml_deps(cargo_toml))

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

    def _parse_package_json_deps(self, path: Path) -> set[str]:
        """Extract dependency names from package.json."""
        deps: set[str] = set()
        try:
            data = json.loads(path.read_text())
            for section in ("dependencies", "devDependencies", "peerDependencies"):
                for name in data.get(section, {}):
                    deps.add(name)
        except Exception:  # noqa: S110
            pass
        return deps

    def _parse_go_mod_deps(self, path: Path) -> set[str]:
        """Extract module paths from go.mod require blocks."""
        deps: set[str] = set()
        try:
            content = path.read_text()
            # Match require blocks
            require_block = re.compile(r"require\s*\((.*?)\)", re.DOTALL)
            for block in require_block.finditer(content):
                for line in block.group(1).strip().splitlines():
                    line = line.strip()
                    if line and not line.startswith("//"):
                        parts = line.split()
                        if parts:
                            deps.add(parts[0])
            # Match single require lines
            single_require = re.compile(r"^require\s+(\S+)\s+", re.MULTILINE)
            for m in single_require.finditer(content):
                deps.add(m.group(1))
        except Exception:  # noqa: S110
            pass
        return deps

    def _parse_cargo_toml_deps(self, path: Path) -> set[str]:
        """Extract crate names from Cargo.toml."""
        deps: set[str] = set()
        try:
            import tomllib
            with path.open("rb") as f:
                data = tomllib.load(f)
            for section in ("dependencies", "dev-dependencies", "build-dependencies"):
                for name in data.get(section, {}):
                    # Normalize: hyphens to underscores (Rust convention)
                    deps.add(name.replace("-", "_"))
        except Exception:  # noqa: S110
            pass
        return deps

    # ------------------------------------------------------------------
    # AI003: Over-Engineering Detection
    # ------------------------------------------------------------------

    _RE_ABSTRACT_CLASS = re.compile(
        r"^class\s+(\w+)\s*\(.*?\bABC\b.*?\)\s*:", re.MULTILINE,
    )
    _RE_CLASS_DEF = re.compile(r"^class\s+(\w+)\s*(?:\(([^)]*)\))?\s*:", re.MULTILINE)
    _RE_GENERIC_PARAMS = re.compile(r"\[([^\]]+)\]")
    _RE_FACTORY_FUNC = re.compile(
        r"^def\s+(create_|make_|build_|get_)\w+\s*\(", re.MULTILINE,
    )

    def _check_ai003_over_engineering(
        self, code: str, language: str, skip_regions: list[int],
    ) -> list[Violation]:
        """Detect over-engineering patterns common in AI-generated code."""
        if language not in ("python", "typescript", "javascript", "java", "auto"):
            return []

        violations: list[Violation] = []

        # 1. Abstract classes with only 1 concrete subclass in same file
        abstract_classes: dict[str, int] = {}
        for m in self._RE_ABSTRACT_CLASS.finditer(code):
            if _is_in_skip_region(m.start(), skip_regions):
                continue
            abstract_classes[m.group(1)] = code[:m.start()].count("\n") + 1

        for abc_name, abc_line in abstract_classes.items():
            # Count classes that inherit from this abstract class
            subclass_count = 0
            pattern = re.compile(
                rf"^class\s+\w+\s*\([^)]*\b{re.escape(abc_name)}\b[^)]*\)\s*:",
                re.MULTILINE,
            )
            for m in pattern.finditer(code):
                if not _is_in_skip_region(m.start(), skip_regions):
                    subclass_count += 1
            if subclass_count == 1:
                violations.append(Violation(
                    id="AI003",
                    rule="ai-over-engineering",
                    category="ai_quality",
                    severity="warning",
                    message=(
                        f"Abstract class '{abc_name}' has only 1 concrete subclass"
                        " in this file. Consider using a concrete class instead."
                    ),
                    line=abc_line,
                    suggestion="Remove the abstraction unless more subclasses are planned",
                ))

        # 2. Excessive generic type parameters (>5)
        for m in self._RE_CLASS_DEF.finditer(code):
            if _is_in_skip_region(m.start(), skip_regions):
                continue
            class_name = m.group(1)
            bases = m.group(2) or ""
            generic_match = self._RE_GENERIC_PARAMS.search(bases)
            if generic_match:
                params = [p.strip() for p in generic_match.group(1).split(",")]
                if len(params) > 5:
                    line_no = code[:m.start()].count("\n") + 1
                    violations.append(Violation(
                        id="AI003",
                        rule="ai-over-engineering",
                        category="ai_quality",
                        severity="warning",
                        message=(
                            f"Class '{class_name}' has {len(params)} generic type"
                            " parameters. Consider simplifying the type hierarchy."
                        ),
                        line=line_no,
                        suggestion="Reduce type parameters to 3-4 or use type aliases",
                    ))

        # 3. Factory functions that return only 1 type
        for m in self._RE_FACTORY_FUNC.finditer(code):
            if _is_in_skip_region(m.start(), skip_regions):
                continue
            func_start = m.start()
            line_no = code[:func_start].count("\n") + 1
            # Find function body (until next def/class at same indent or end)
            func_line = code[func_start:].split("\n")[0]
            indent = len(func_line) - len(func_line.lstrip())
            body_lines = []
            for line in code[func_start:].split("\n")[1:]:
                if line.strip() == "":
                    body_lines.append(line)
                    continue
                line_indent = len(line) - len(line.lstrip())
                if line_indent <= indent and line.strip():
                    break
                body_lines.append(line)
            body = "\n".join(body_lines)
            # Count distinct return types/classes
            returns = re.findall(r"return\s+(\w+)\s*\(", body)
            unique_returns = set(returns)
            if len(returns) >= 2 and len(unique_returns) == 1:
                violations.append(Violation(
                    id="AI003",
                    rule="ai-over-engineering",
                    category="ai_quality",
                    severity="warning",
                    message=(
                        f"Factory function at line {line_no} always returns the same"
                        f" type '{returns[0]}'. A factory may be unnecessary."
                    ),
                    line=line_no,
                    suggestion="Use a direct constructor call instead of a factory",
                ))

        return violations

    # ------------------------------------------------------------------
    # AI004: Duplicate Code Block Detection
    # ------------------------------------------------------------------

    _RE_FUNC_BODY = re.compile(
        r"^(\s*)(?:def|function|fn|func)\s+\w+.*?[:{]\s*$",
        re.MULTILINE,
    )

    def _check_ai004_duplicate_blocks(
        self, code: str, language: str, skip_regions: list[int],
    ) -> list[Violation]:
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
                if _is_in_skip_region(char_offset, skip_regions):
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
                violations.append(Violation(
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
                ))
            else:
                seen[h] = line_no

        return violations

    # ------------------------------------------------------------------
    # AI005: Inconsistent Error Handling
    # ------------------------------------------------------------------

    def _check_ai005_inconsistent_errors(
        self, code: str, language: str, skip_regions: list[int],
    ) -> list[Violation]:
        """Detect inconsistent error handling patterns."""
        if language not in ("python", "typescript", "javascript", "auto"):
            return []

        violations: list[Violation] = []

        if language in ("python", "auto"):
            violations.extend(self._check_ai005_python(code, skip_regions))
        if language in ("typescript", "javascript"):
            violations.extend(self._check_ai005_typescript(code, skip_regions))

        return violations

    def _check_ai005_python(self, code: str, skip_regions: list[int]) -> list[Violation]:
        """AI005 checks for Python."""
        violations: list[Violation] = []

        # Detect bare except mixed with specific except in same file
        bare_excepts = [
            code[:m.start()].count("\n") + 1
            for m in re.finditer(r"^\s*except\s*:", code, re.MULTILINE)
            if not _is_in_skip_region(m.start(), skip_regions)
        ]
        specific_excepts = [
            code[:m.start()].count("\n") + 1
            for m in re.finditer(r"^\s*except\s+\w+", code, re.MULTILINE)
            if not _is_in_skip_region(m.start(), skip_regions)
        ]

        if bare_excepts and specific_excepts:
            violations.extend(
                Violation(
                    id="AI005",
                    rule="ai-inconsistent-errors",
                    category="ai_quality",
                    severity="warning",
                    message=(
                        "Bare 'except:' mixed with specific exception handlers in the"
                        " same file. Use specific exception types consistently."
                    ),
                    line=line_no,
                    suggestion="Replace bare 'except:' with specific exception types",
                )
                for line_no in bare_excepts
            )

        return violations

    def _check_ai005_typescript(self, code: str, skip_regions: list[int]) -> list[Violation]:
        """AI005 checks for TypeScript/JavaScript."""
        violations: list[Violation] = []

        # Detect empty catch blocks mixed with handled catch blocks
        empty_catches = [
            code[:m.start()].count("\n") + 1
            for m in re.finditer(r"catch\s*\([^)]*\)\s*\{\s*\}", code)
            if not _is_in_skip_region(m.start(), skip_regions)
        ]
        handled_catches = [
            code[:m.start()].count("\n") + 1
            for m in re.finditer(r"catch\s*\([^)]*\)\s*\{[^}]+\}", code, re.DOTALL)
            if not _is_in_skip_region(m.start(), skip_regions)
            and m.group(0).strip() != "catch"
            and "{}" not in m.group(0).replace(" ", "")
        ]

        if empty_catches and handled_catches:
            violations.extend(
                Violation(
                    id="AI005",
                    rule="ai-inconsistent-errors",
                    category="ai_quality",
                    severity="warning",
                    message=(
                        "Empty catch block mixed with handled catch blocks."
                        " Handle or explicitly comment why the error is ignored."
                    ),
                    line=line_no,
                    suggestion="Add error handling or a comment explaining why it's ignored",
                )
                for line_no in empty_catches
            )

        return violations

    # ------------------------------------------------------------------
    # AI006: Unnecessary Heavy Imports
    # ------------------------------------------------------------------

    def _check_ai006_heavy_imports(
        self, code: str, language: str, skip_regions: list[int],
    ) -> list[Violation]:
        """Detect heavy library imports for trivially simple usage."""
        if language not in ("python", "auto"):
            return []

        violations: list[Violation] = []

        for lib, info in _HEAVY_IMPORT_ALTERNATIVES.items():
            # Check if the library is imported
            import_match = re.search(
                rf"^(?:import\s+{lib}|from\s+{lib}\s+import)\b",
                code,
                re.MULTILINE,
            )
            if not import_match:
                continue
            if _is_in_skip_region(import_match.start(), skip_regions):
                continue

            # Check if usage is trivially simple
            usage_pattern = re.compile(info["pattern"], re.MULTILINE)
            usages = list(usage_pattern.finditer(code))
            # Count total usages of the library
            all_usages = list(re.finditer(rf"\b{lib}\b", code))
            # Subtract the import line itself
            non_import_usages = [
                u for u in all_usages
                if u.start() != import_match.start()
                and not _is_in_skip_region(u.start(), skip_regions)
            ]

            # Only flag if there are very few usages (1-2) and they match the simple pattern
            if len(non_import_usages) <= 2 and usages:
                line_no = code[:import_match.start()].count("\n") + 1
                violations.append(Violation(
                    id="AI006",
                    rule="ai-heavy-import",
                    category="ai_quality",
                    severity="info",
                    message=(
                        f"'{lib}' imported for simple usage. {info['reason']}."
                    ),
                    line=line_no,
                    suggestion=f"Consider using {info['alternative']} instead",
                ))

        return violations

    # ------------------------------------------------------------------
    # AI007: Security Theater Detection
    # ------------------------------------------------------------------

    _RE_HASH_PASSWORD = re.compile(
        r"\bhash\s*\(\s*\w*(?:password|secret|token|key|passw|pwd)\w*\s*\)",
        re.IGNORECASE,
    )
    _RE_VALIDATE_ALWAYS_TRUE = re.compile(
        r"def\s+validate\w*\s*\([^)]*\)\s*(?:->\s*\w+\s*)?:\s*\n\s+return\s+True",
    )
    _RE_MD5_SECURITY = re.compile(
        r"""(?:hashlib\.md5|MD5\.new|md5\s*\().*?(?:password|secret|token|key|auth|cred)""",
        re.IGNORECASE,
    )
    _RE_MD5_IMPORT_USAGE = re.compile(
        r"""(?:password|secret|token|key|auth|cred).*?(?:hashlib\.md5|md5\s*\()""",
        re.IGNORECASE,
    )

    def _check_ai007_security_theater(
        self, code: str, language: str, skip_regions: list[int],
    ) -> list[Violation]:
        """Detect security theater patterns — code that looks secure but isn't."""
        if language not in ("python", "auto"):
            return []

        violations: list[Violation] = []

        # 1. Built-in hash() used on passwords/secrets
        for m in self._RE_HASH_PASSWORD.finditer(code):
            if _is_in_skip_region(m.start(), skip_regions):
                continue
            line_no = code[:m.start()].count("\n") + 1
            violations.append(Violation(
                id="AI007",
                rule="ai-security-theater",
                category="security",
                severity="error",
                message=(
                    "Built-in hash() used on sensitive data. hash() is not"
                    " cryptographically secure and not suitable for passwords."
                ),
                line=line_no,
                suggestion="Use hashlib.pbkdf2_hmac(), bcrypt, or argon2 for password hashing",
            ))

        # 2. Validation functions that always return True
        for m in self._RE_VALIDATE_ALWAYS_TRUE.finditer(code):
            if _is_in_skip_region(m.start(), skip_regions):
                continue
            line_no = code[:m.start()].count("\n") + 1
            violations.append(Violation(
                id="AI007",
                rule="ai-security-theater",
                category="security",
                severity="error",
                message=(
                    "Validation function always returns True. This provides"
                    " no actual validation."
                ),
                line=line_no,
                suggestion="Implement actual validation logic or remove the function",
            ))

        # 3. MD5 used for security purposes (not checksums)
        for m in self._RE_MD5_SECURITY.finditer(code):
            if _is_in_skip_region(m.start(), skip_regions):
                continue
            line_no = code[:m.start()].count("\n") + 1
            violations.append(Violation(
                id="AI007",
                rule="ai-security-theater",
                category="security",
                severity="error",
                message=(
                    "MD5 used with security-sensitive data. MD5 is"
                    " cryptographically broken and should not be used for"
                    " passwords, tokens, or authentication."
                ),
                line=line_no,
                suggestion="Use SHA-256+ for integrity, bcrypt/argon2 for passwords",
            ))
        # Also check reverse order (password...md5)
        for m in self._RE_MD5_IMPORT_USAGE.finditer(code):
            if _is_in_skip_region(m.start(), skip_regions):
                continue
            line_no = code[:m.start()].count("\n") + 1
            # Avoid duplicate if already caught by forward pattern
            already_caught = any(
                v.id == "AI007" and v.line == line_no for v in violations
            )
            if not already_caught:
                violations.append(Violation(
                    id="AI007",
                    rule="ai-security-theater",
                    category="security",
                    severity="error",
                    message=(
                        "MD5 used with security-sensitive data. MD5 is"
                        " cryptographically broken."
                    ),
                    line=line_no,
                    suggestion="Use SHA-256+ for integrity, bcrypt/argon2 for passwords",
                ))

        return violations

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
