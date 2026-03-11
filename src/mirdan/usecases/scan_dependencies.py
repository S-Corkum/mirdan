"""ScanDependencies use case — extracted from server.py."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mirdan.core.manifest_parser import ManifestParser
    from mirdan.core.vuln_scanner import VulnScanner


class ScanDependenciesUseCase:
    """Scan project dependencies for known vulnerabilities."""

    def __init__(
        self,
        manifest_parser: ManifestParser,
        vuln_scanner: VulnScanner,
    ) -> None:
        self._manifest_parser = manifest_parser
        self._vuln_scanner = vuln_scanner

    async def execute(
        self,
        project_path: str = ".",
        ecosystem: str = "auto",
    ) -> dict[str, Any]:
        """Execute the scan_dependencies use case.

        Queries the OSV database (free, no API key) to check all dependencies
        against known CVEs. Results are cached per config (default: 24 hours).

        Args:
            project_path: Project directory containing dependency manifests
            ecosystem: Filter by ecosystem (auto|PyPI|npm|crates.io|Go|Maven)

        Returns:
            Scan results with packages checked and vulnerabilities found
        """
        scan_dir = Path(project_path).resolve()

        if not scan_dir.is_dir():
            return {"error": f"Not a directory: {project_path}"}

        packages = self._manifest_parser.parse(scan_dir)
        if ecosystem != "auto":
            packages = [p for p in packages if p.ecosystem == ecosystem]

        if not packages:
            return {
                "packages_scanned": 0,
                "vulnerabilities_found": 0,
                "message": "No dependency manifests found",
                "findings": [],
            }

        findings = await self._vuln_scanner.scan(packages)

        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for f in findings:
            severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

        return {
            "packages_scanned": len(packages),
            "vulnerabilities_found": len(findings),
            "severity_counts": severity_counts,
            "findings": [f.to_dict() for f in findings],
            "ecosystems_checked": list({p.ecosystem for p in packages}),
        }
