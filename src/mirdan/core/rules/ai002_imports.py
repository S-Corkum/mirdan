"""AI002: Hallucinated import detection rule.

Detects imports for packages not in stdlib or project dependencies.
Supports Python, TypeScript/JavaScript, Go, and Rust.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mirdan.core.rules.base import BaseRule, RuleContext
from mirdan.models import Violation

if TYPE_CHECKING:
    from mirdan.core.manifest_parser import ManifestParser

# ---------------------------------------------------------------------------
# Python stdlib module names -- derived from the running interpreter plus
# modules removed in Python 3.12+ (so user code targeting older versions
# is not flagged as hallucinated).
# ---------------------------------------------------------------------------

PYTHON_STDLIB_MODULES: frozenset[str] = frozenset(sys.stdlib_module_names) | frozenset(
    {
        # Not stdlib but so ubiquitous we skip it
        "typing_extensions",
        # Removed in Python 3.12/3.13 -- still valid for projects targeting 3.11 and below
        "aifc",
        "asynchat",
        "asyncore",
        "audioop",
        "binhex",
        "cgi",
        "cgitb",
        "chunk",
        "crypt",
        "distutils",
        "imghdr",
        "imp",
        "lib2to3",
        "mailcap",
        "nis",
        "nntplib",
        "ossaudiodev",
        "pipes",
        "smtpd",
        "sndhdr",
        "spwd",
        "sunau",
        "telnetlib",
        "test",
        "uu",
        "xdrlib",
    }
)

# Common third-party packages that are often transitive dependencies.
# When no project_dir is available, we give these the benefit of the doubt.
_COMMON_TRANSITIVE_PACKAGES: frozenset[str] = frozenset(
    {
        "requests",
        "typing_extensions",
        "six",
        "certifi",
        "charset_normalizer",
        "idna",
        "urllib3",
        "packaging",
        "setuptools",
        "pip",
        "wheel",
        "attrs",
        "cattrs",
        "pydantic",
        "pydantic_core",
    }
)


# ---------------------------------------------------------------------------
# Node.js built-in modules
# ---------------------------------------------------------------------------

NODE_BUILTIN_MODULES: frozenset[str] = frozenset(
    {
        "assert",
        "async_hooks",
        "buffer",
        "child_process",
        "cluster",
        "console",
        "constants",
        "crypto",
        "dgram",
        "diagnostics_channel",
        "dns",
        "domain",
        "events",
        "fs",
        "http",
        "http2",
        "https",
        "inspector",
        "module",
        "net",
        "os",
        "path",
        "perf_hooks",
        "process",
        "punycode",
        "querystring",
        "readline",
        "repl",
        "stream",
        "string_decoder",
        "timers",
        "tls",
        "trace_events",
        "tty",
        "url",
        "util",
        "v8",
        "vm",
        "wasi",
        "worker_threads",
        "zlib",
        # Node: prefixed versions
        "node:assert",
        "node:async_hooks",
        "node:buffer",
        "node:child_process",
        "node:cluster",
        "node:console",
        "node:crypto",
        "node:dgram",
        "node:diagnostics_channel",
        "node:dns",
        "node:domain",
        "node:events",
        "node:fs",
        "node:http",
        "node:http2",
        "node:https",
        "node:inspector",
        "node:module",
        "node:net",
        "node:os",
        "node:path",
        "node:perf_hooks",
        "node:process",
        "node:querystring",
        "node:readline",
        "node:repl",
        "node:stream",
        "node:string_decoder",
        "node:timers",
        "node:tls",
        "node:trace_events",
        "node:tty",
        "node:url",
        "node:util",
        "node:v8",
        "node:vm",
        "node:wasi",
        "node:worker_threads",
        "node:zlib",
    }
)

# ---------------------------------------------------------------------------
# Go standard library top-level packages
# ---------------------------------------------------------------------------

GO_STDLIB_PACKAGES: frozenset[str] = frozenset(
    {
        "archive",
        "bufio",
        "bytes",
        "compress",
        "container",
        "context",
        "crypto",
        "database",
        "debug",
        "embed",
        "encoding",
        "errors",
        "expvar",
        "flag",
        "fmt",
        "go",
        "hash",
        "html",
        "image",
        "index",
        "internal",
        "io",
        "iter",
        "log",
        "maps",
        "math",
        "mime",
        "net",
        "os",
        "path",
        "plugin",
        "reflect",
        "regexp",
        "runtime",
        "slices",
        "sort",
        "strconv",
        "strings",
        "structs",
        "sync",
        "syscall",
        "testing",
        "text",
        "time",
        "unicode",
        "unique",
        "unsafe",
        "weak",
    }
)


class AI002ImportRule(BaseRule):
    """Detect hallucinated imports not in stdlib or project dependencies (AI002).

    Supports Python, TypeScript/JavaScript, Go, and Rust.  When a
    ``manifest_parser`` and/or ``workspace_resolver`` are provided the rule
    resolves per-sub-project dependencies automatically.
    """

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
        r"import\s*\((.*?)\)",
        re.DOTALL,
    )
    _RE_GO_IMPORT_LINE = re.compile(r'"([^"]+)"')

    # Rust imports
    _RE_RUST_USE = re.compile(r"^\s*use\s+(\w+)::", re.MULTILINE)

    def __init__(
        self,
        manifest_parser: ManifestParser | None = None,
        workspace_resolver: Any | None = None,  # WorkspaceResolver
        project_dir: Path | None = None,
    ) -> None:
        self._manifest_parser = manifest_parser
        self._workspace_resolver = workspace_resolver
        self._project_dir = project_dir
        self._project_deps: frozenset[str] | None = None
        self._project_deps_cache: dict[str, frozenset[str]] = {}
        if project_dir:
            self._project_deps = self._load_project_deps(project_dir)

    @property
    def id(self) -> str:
        return "AI002"

    @property
    def name(self) -> str:
        return "ai-hallucinated-import"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "typescript", "javascript", "go", "rust", "auto"})

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        """Detect imports for packages not in stdlib or project deps.

        When ``context.file_path`` is provided and a workspace resolver is
        configured, dependencies are resolved per sub-project.
        """
        file_path = context.file_path
        deps = self._get_deps_for_file(file_path) if file_path else self._project_deps
        if language in ("python", "auto"):
            return self._check_ai002_python(code, deps=deps)
        if language in ("typescript", "javascript"):
            return self._check_ai002_typescript(code, deps=deps)
        if language == "go":
            return self._check_ai002_go(code, deps=deps)
        if language == "rust":
            return self._check_ai002_rust(code, deps=deps)
        return []

    # ------------------------------------------------------------------
    # File-path-aware dependency resolution
    # ------------------------------------------------------------------

    def _get_deps_for_file(self, file_path: str) -> frozenset[str] | None:
        """Get dependencies for the sub-project containing *file_path*.

        When a workspace_resolver is available and file_path is provided,
        resolves the file to its sub-project and loads/caches deps for
        that sub-project directory. Otherwise returns the root-level
        ``self._project_deps``.
        """
        if not file_path or self._workspace_resolver is None:
            return self._project_deps

        resolved = self._workspace_resolver.resolve(file_path)
        if resolved is None:
            return self._project_deps

        cache_key = str(resolved.project_dir)
        if cache_key in self._project_deps_cache:
            return self._project_deps_cache[cache_key]

        deps = self._load_project_deps(resolved.project_dir)
        self._project_deps_cache[cache_key] = deps
        return deps

    # ------------------------------------------------------------------
    # Import extraction (shared helper)
    # ------------------------------------------------------------------

    def extract_imports(self, code: str, language: str) -> set[str]:
        """Extract imported module names from code.

        This is a public helper used by both AI002 and SEC014 (in the
        main checker).
        """
        imports: set[str] = set()
        if language in ("python", "auto"):
            for m in self._RE_IMPORT.finditer(code):
                imports.add(m.group(1))
            for m in self._RE_FROM_IMPORT.finditer(code):
                mod = m.group(1)
                if mod != "." and not mod.startswith(".") and mod != "__future__":
                    imports.add(mod)
        elif language in ("typescript", "javascript"):
            for m in self._RE_TS_IMPORT_FROM.finditer(code):
                imports.add(self._extract_npm_package_name(m.group(1)))
            for m in self._RE_TS_REQUIRE.finditer(code):
                imports.add(self._extract_npm_package_name(m.group(1)))
        return imports

    # ------------------------------------------------------------------
    # Language-specific AI002 checks
    # ------------------------------------------------------------------

    def _check_ai002_python(self, code: str, deps: frozenset[str] | None = None) -> list[Violation]:
        """AI002 for Python imports."""
        effective_deps = deps if deps is not None else self._project_deps
        if effective_deps is None:
            return []

        violations: list[Violation] = []
        imported_modules: set[tuple[str, int]] = set()

        for m in self._RE_IMPORT.finditer(code):
            module = m.group(1)
            line_no = code[: m.start()].count("\n") + 1
            imported_modules.add((module, line_no))

        for m in self._RE_FROM_IMPORT.finditer(code):
            module = m.group(1)
            if module == ".":
                continue
            line_no = code[: m.start()].count("\n") + 1
            imported_modules.add((module, line_no))

        for module, line_no in imported_modules:
            if module == "__future__":
                continue
            if module.startswith("."):
                continue
            if not self._is_known_python_module_with_deps(module, effective_deps):
                violations.append(
                    Violation(
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
                    )
                )

        return violations

    def _check_ai002_typescript(
        self, code: str, deps: frozenset[str] | None = None
    ) -> list[Violation]:
        """AI002 for TypeScript/JavaScript imports."""
        effective_deps = deps if deps is not None else self._project_deps
        if effective_deps is None:
            return []

        violations: list[Violation] = []
        imported_modules: set[tuple[str, int]] = set()

        for m in self._RE_TS_IMPORT_FROM.finditer(code):
            raw = m.group(1)
            # Extract package name (handle scoped: @scope/pkg -> @scope/pkg)
            module = self._extract_npm_package_name(raw)
            line_no = code[: m.start()].count("\n") + 1
            imported_modules.add((module, line_no))

        for m in self._RE_TS_REQUIRE.finditer(code):
            raw = m.group(1)
            module = self._extract_npm_package_name(raw)
            line_no = code[: m.start()].count("\n") + 1
            imported_modules.add((module, line_no))

        for module, line_no in imported_modules:
            if module in NODE_BUILTIN_MODULES:
                continue
            if module.startswith("node:"):
                continue
            if module in effective_deps:
                continue
            violations.append(
                Violation(
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
                        f"Check if '{module}' is a valid npm package and add it to package.json"
                    ),
                )
            )

        return violations

    def _check_ai002_go(self, code: str, deps: frozenset[str] | None = None) -> list[Violation]:
        """AI002 for Go imports."""
        effective_deps = deps if deps is not None else self._project_deps
        if effective_deps is None:
            return []

        violations: list[Violation] = []
        imported_paths: set[tuple[str, int]] = set()

        # Single import statements
        for m in self._RE_GO_IMPORT_SINGLE.finditer(code):
            path = m.group(1)
            line_no = code[: m.start()].count("\n") + 1
            imported_paths.add((path, line_no))

        # Block import statements
        for m in self._RE_GO_IMPORT_BLOCK.finditer(code):
            block = m.group(1)
            block_start = code[: m.start()].count("\n") + 1
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
            if path in effective_deps:
                continue
            # Check module prefix match: import path should start with a known dep
            # e.g., "github.com/foo/bar/sub" starts with dep "github.com/foo/bar"
            if any(path.startswith(dep + "/") or path == dep for dep in effective_deps):
                continue
            # Skip stdlib paths
            if "." not in top_pkg:
                continue
            violations.append(
                Violation(
                    id="AI002",
                    rule="ai-hallucinated-import",
                    category="ai_quality",
                    severity="warning",
                    message=(
                        f"Import '{path}' not found in go.mod dependencies."
                        " Verify this module is required."
                    ),
                    line=line_no,
                    suggestion=(f"Run 'go get {path}' to add the dependency"),
                )
            )

        return violations

    def _check_ai002_rust(self, code: str, deps: frozenset[str] | None = None) -> list[Violation]:
        """AI002 for Rust imports."""
        effective_deps = deps if deps is not None else self._project_deps
        if effective_deps is None:
            return []

        violations: list[Violation] = []
        # Rust built-in crates
        rust_builtins = frozenset(
            {
                "std",
                "core",
                "alloc",
                "proc_macro",
                "test",
                "self",
                "super",
                "crate",
            }
        )

        for m in self._RE_RUST_USE.finditer(code):
            crate_name = m.group(1)
            line_no = code[: m.start()].count("\n") + 1
            if crate_name in rust_builtins:
                continue
            if crate_name in effective_deps:
                continue
            violations.append(
                Violation(
                    id="AI002",
                    rule="ai-hallucinated-import",
                    category="ai_quality",
                    severity="warning",
                    message=(
                        f"Crate '{crate_name}' not found in Cargo.toml dependencies."
                        " Verify this crate is added."
                    ),
                    line=line_no,
                    suggestion=(f"Run 'cargo add {crate_name}' to add the dependency"),
                )
            )

        return violations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    @staticmethod
    def _is_known_python_module_with_deps(module: str, deps: frozenset[str] | None) -> bool:
        """Check if a Python module is known, using the provided deps set."""
        if module in PYTHON_STDLIB_MODULES:
            return True
        if deps is not None and module in deps:
            return True
        return module in sys.stdlib_module_names

    # ------------------------------------------------------------------
    # Dependency loading
    # ------------------------------------------------------------------

    def _load_project_deps(self, project_dir: Path) -> frozenset[str]:
        """Load project dependency names from manifests.

        If a ManifestParser was injected, delegates to it for manifest parsing.
        Otherwise falls back to inline parsing. Always adds common transitive
        packages and local package names to prevent AI002 false positives.
        """
        deps: set[str] = set()

        if self._manifest_parser:
            # Delegate manifest parsing to ManifestParser
            packages = self._manifest_parser.parse(project_dir)
            deps.update(p.name for p in packages)
        else:
            # Fallback: original inline parsing
            pyproject = project_dir / "pyproject.toml"
            if pyproject.exists():
                deps.update(self._parse_pyproject_deps(pyproject))

            requirements = project_dir / "requirements.txt"
            if requirements.exists():
                deps.update(self._parse_requirements_deps(requirements))

            package_json = project_dir / "package.json"
            if package_json.exists():
                deps.update(self._parse_package_json_deps(package_json))

            go_mod = project_dir / "go.mod"
            if go_mod.exists():
                deps.update(self._parse_go_mod_deps(go_mod))

            cargo_toml = project_dir / "Cargo.toml"
            if cargo_toml.exists():
                deps.update(self._parse_cargo_toml_deps(cargo_toml))

        # CRITICAL: Always include these -- they prevent AI002 false positives
        deps.update(_COMMON_TRANSITIVE_PACKAGES)
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

    @staticmethod
    def _find_local_packages(project_dir: Path) -> set[str]:
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
