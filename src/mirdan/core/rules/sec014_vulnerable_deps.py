"""SEC014: Vulnerable dependency detection rule."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mirdan.core.rules.base import BaseRule, RuleContext
from mirdan.models import Violation

if TYPE_CHECKING:
    from mirdan.core.manifest_parser import ManifestParser
    from mirdan.core.vuln_scanner import VulnScanner


class SEC014VulnerableDepsRule(BaseRule):
    """Check if imported packages have known cached vulnerabilities (SEC014)."""

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

    def __init__(
        self,
        manifest_parser: ManifestParser | None = None,
        vuln_scanner: VulnScanner | None = None,
    ) -> None:
        self._manifest_parser = manifest_parser
        self._vuln_scanner = vuln_scanner

    @property
    def id(self) -> str:
        return "SEC014"

    @property
    def name(self) -> str:
        return "vulnerable-dependency"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "typescript", "javascript", "go", "rust", "auto"})

    @property
    def is_quick(self) -> bool:
        return True

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        """Check if imported packages have known cached vulnerabilities."""
        if not self._vuln_scanner or not self._manifest_parser:
            return []

        imports = self._extract_imports(code, language)
        if not imports:
            return []

        packages = self._manifest_parser.parse()
        cached_findings = self._vuln_scanner.check_cached(packages)
        if not cached_findings:
            return []

        violations: list[Violation] = []
        import_names_lower = {n.split(".")[0].lower() for n in imports}
        for finding in cached_findings:
            pkg_import_name = finding.package.replace("-", "_").lower()
            if pkg_import_name in import_names_lower:
                sev = "error" if finding.severity in ("critical", "high") else "warning"
                violations.append(
                    Violation(
                        id="SEC014",
                        rule="vulnerable-dependency",
                        category="security",
                        severity=sev,
                        message=(
                            f"Package '{finding.package}' v{finding.version} has "
                            f"vulnerability {finding.vuln_id}: {finding.summary[:100]}"
                        ),
                        suggestion=(
                            f"Upgrade to v{finding.fixed_version}"
                            if finding.fixed_version
                            else "Check advisory for remediation"
                        ),
                    )
                )
        return violations

    def _extract_imports(self, code: str, language: str) -> set[str]:
        """Extract imported module names from code."""
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
