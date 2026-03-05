"""Parse dependency manifests to extract package information."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from mirdan.models import PackageInfo

# Ecosystem mapping
_ECOSYSTEM_MAP = {
    "pyproject.toml": "PyPI",
    "requirements.txt": "PyPI",
    "setup.cfg": "PyPI",
    "package.json": "npm",
    "Cargo.toml": "crates.io",
    "go.mod": "Go",
    "pom.xml": "Maven",
    "build.gradle": "Maven",
}

# PEP 508 version specifier extraction
_RE_PEP508_VERSION = re.compile(r"[><=!~]+\s*(\d[\d.]*\d*)")
_RE_REQUIREMENTS_LINE = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9._-]*)\s*([><=!~].*)?$")
_RE_GO_REQUIRE = re.compile(r"^\s+(\S+)\s+(v[\d.]+)", re.MULTILINE)
_RE_NPM_VERSION = re.compile(r"[\^~>=<]*\s*([\d][\d.]*)")


class ManifestParser:
    """Parse dependency manifests for all supported ecosystems."""

    def __init__(self, project_dir: Path | None = None) -> None:
        self._project_dir = project_dir or Path.cwd()
        self._cache: list[PackageInfo] | None = None
        self._cache_mtimes: dict[str, float] = {}

    def parse(self, project_dir: Path | None = None) -> list[PackageInfo]:
        """Discover and parse all manifests in project directory."""
        root = project_dir or self._project_dir
        if not root.is_dir():
            return []

        # Check cache validity
        if self._cache is not None and self._cache_valid(root):
            return self._cache

        packages: list[PackageInfo] = []
        for manifest_name, ecosystem in _ECOSYSTEM_MAP.items():
            manifest_path = root / manifest_name
            if manifest_path.exists():
                try:
                    parsed = self._parse_file(manifest_path, ecosystem)
                    packages.extend(parsed)
                    self._cache_mtimes[str(manifest_path)] = manifest_path.stat().st_mtime
                except Exception:  # noqa: S112
                    continue  # Skip malformed manifests

        # Enrich from lock files for exact versions
        packages = self._enrich_from_lock_files(root, packages)

        self._cache = packages
        return packages

    def get_dep_names(self, project_dir: Path | None = None) -> frozenset[str]:
        """Get just dependency names (for AI002 compatibility)."""
        packages = self.parse(project_dir)
        return frozenset(p.name for p in packages)

    def get_version(self, package: str, ecosystem: str) -> str | None:
        """Get version of a specific package."""
        packages = self.parse()
        for p in packages:
            if p.name == package and p.ecosystem == ecosystem:
                return p.version
        return None

    def _cache_valid(self, root: Path) -> bool:
        """Check if cached results are still valid."""
        for manifest_name in _ECOSYSTEM_MAP:
            manifest_path = root / manifest_name
            cached_mtime = self._cache_mtimes.get(str(manifest_path))
            if manifest_path.exists():
                if cached_mtime is None or manifest_path.stat().st_mtime != cached_mtime:
                    return False
            elif cached_mtime is not None:
                return False  # File was deleted
        return True

    def _parse_file(
        self, path: Path, ecosystem: str
    ) -> list[PackageInfo]:
        """Dispatch to type-specific parser."""
        name = path.name
        if name == "pyproject.toml":
            return self._parse_pyproject_toml(path)
        if name == "requirements.txt":
            return self._parse_requirements_txt(path)
        if name == "setup.cfg":
            return self._parse_setup_cfg(path)
        if name == "package.json":
            return self._parse_package_json(path)
        if name == "Cargo.toml":
            return self._parse_cargo_toml(path)
        if name == "go.mod":
            return self._parse_go_mod(path)
        return []

    def _parse_pyproject_toml(self, path: Path) -> list[PackageInfo]:
        """Parse pyproject.toml dependencies."""
        packages: list[PackageInfo] = []
        with path.open("rb") as f:
            data = tomllib.load(f)

        # [project.dependencies]
        for dep_str in data.get("project", {}).get("dependencies", []):
            pkg = self._parse_pep508(dep_str, str(path))
            if pkg:
                packages.append(pkg)

        # [project.optional-dependencies]
        for group, deps in data.get("project", {}).get("optional-dependencies", {}).items():
            is_dev = group in ("dev", "test", "tests", "testing")
            for dep_str in deps:
                pkg = self._parse_pep508(dep_str, str(path), is_dev=is_dev)
                if pkg:
                    packages.append(pkg)

        return packages

    def _parse_pep508(
        self, dep_str: str, source: str, is_dev: bool = False
    ) -> PackageInfo | None:
        """Parse a PEP 508 dependency string."""
        # Strip extras and environment markers
        name_part = dep_str.split(";")[0].strip()
        # Extract name (before any version specifier)
        match = re.match(r"^([A-Za-z0-9][\w.-]*)", name_part)
        if not match:
            return None
        name = match.group(1).strip()
        # Normalize: PyPI uses lowercase with hyphens
        normalized = re.sub(r"[-_.]+", "-", name).lower()

        # Extract minimum version
        version = ""
        ver_match = _RE_PEP508_VERSION.search(name_part)
        if ver_match:
            version = ver_match.group(1)

        return PackageInfo(
            name=normalized,
            version=version,
            ecosystem="PyPI",
            source=source,
            is_dev=is_dev,
        )

    def _parse_requirements_txt(self, path: Path) -> list[PackageInfo]:
        """Parse requirements.txt dependencies."""
        packages: list[PackageInfo] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Handle inline comments
            line = line.split("#")[0].strip()
            # Split on == or >= etc.
            match = _RE_REQUIREMENTS_LINE.match(line)
            if match:
                name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
                version = ""
                if match.group(2):
                    ver_match = re.search(r"[\d][\d.]*", match.group(2))
                    if ver_match:
                        version = ver_match.group(0)
                packages.append(PackageInfo(
                    name=name,
                    version=version,
                    ecosystem="PyPI",
                    source=str(path),
                ))
        return packages

    def _parse_setup_cfg(self, path: Path) -> list[PackageInfo]:
        """Parse setup.cfg install_requires."""
        packages: list[PackageInfo] = []
        in_install_requires = False
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped == "install_requires =":
                in_install_requires = True
                continue
            if in_install_requires:
                if not stripped or (not line.startswith(" ") and not line.startswith("\t")):
                    break
                pkg = self._parse_pep508(stripped, str(path))
                if pkg:
                    packages.append(pkg)
        return packages

    def _parse_package_json(self, path: Path) -> list[PackageInfo]:
        """Parse package.json dependencies."""
        packages: list[PackageInfo] = []
        data = json.loads(path.read_text())

        for section, is_dev in [("dependencies", False), ("devDependencies", True)]:
            for name, version_str in data.get(section, {}).items():
                version = ""
                ver_match = _RE_NPM_VERSION.search(version_str)
                if ver_match:
                    version = ver_match.group(1)
                packages.append(PackageInfo(
                    name=name,
                    version=version,
                    ecosystem="npm",
                    source=str(path),
                    is_dev=is_dev,
                ))

        return packages

    def _parse_cargo_toml(self, path: Path) -> list[PackageInfo]:
        """Parse Cargo.toml dependencies."""
        packages: list[PackageInfo] = []
        with path.open("rb") as f:
            data = tomllib.load(f)

        for section in ("dependencies", "dev-dependencies"):
            is_dev = section == "dev-dependencies"
            for name, value in data.get(section, {}).items():
                if isinstance(value, str):
                    version = value
                elif isinstance(value, dict):
                    version = value.get("version", "")
                else:
                    version = ""
                packages.append(PackageInfo(
                    name=name,
                    version=version.lstrip("^~>=<"),
                    ecosystem="crates.io",
                    source=str(path),
                    is_dev=is_dev,
                ))

        return packages

    def _parse_go_mod(self, path: Path) -> list[PackageInfo]:
        """Parse go.mod require blocks."""
        packages: list[PackageInfo] = []
        content = path.read_text()

        for match in _RE_GO_REQUIRE.finditer(content):
            module_path = match.group(1)
            version = match.group(2).lstrip("v")
            # Use last path segment as package name
            name = module_path.split("/")[-1] if "/" in module_path else module_path
            packages.append(PackageInfo(
                name=name,
                version=version,
                ecosystem="Go",
                source=str(path),
            ))

        return packages

    def _enrich_from_lock_files(
        self, root: Path, packages: list[PackageInfo]
    ) -> list[PackageInfo]:
        """Override versions from lock files if available."""
        # uv.lock / poetry.lock for Python
        lock_versions: dict[str, str] = {}

        uv_lock = root / "uv.lock"
        if uv_lock.exists():
            lock_versions.update(self._parse_uv_lock(uv_lock))

        package_lock = root / "package-lock.json"
        if package_lock.exists():
            lock_versions.update(self._parse_package_lock(package_lock))

        if not lock_versions:
            return packages

        for pkg in packages:
            locked = lock_versions.get(f"{pkg.ecosystem}:{pkg.name}")
            if locked:
                pkg.version = locked

        return packages

    def _parse_uv_lock(self, path: Path) -> dict[str, str]:
        """Extract versions from uv.lock."""
        versions: dict[str, str] = {}
        try:
            with path.open("rb") as f:
                data = tomllib.load(f)
            for pkg in data.get("package", []):
                name = pkg.get("name", "")
                version = pkg.get("version", "")
                if name and version:
                    versions[f"PyPI:{name}"] = version
        except Exception:  # noqa: S110
            pass
        return versions

    def _parse_package_lock(self, path: Path) -> dict[str, str]:
        """Extract versions from package-lock.json."""
        versions: dict[str, str] = {}
        try:
            data = json.loads(path.read_text())
            for name, info in data.get("packages", {}).items():
                if name.startswith("node_modules/"):
                    pkg_name = name[len("node_modules/"):]
                    version = info.get("version", "")
                    if pkg_name and version:
                        versions[f"npm:{pkg_name}"] = version
        except Exception:  # noqa: S110
            pass
        return versions
