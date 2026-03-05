"""Scan dependencies for known vulnerabilities via OSV API."""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mirdan.models import PackageInfo, VulnFinding

logger = logging.getLogger(__name__)

OSV_API_URL = "https://api.osv.dev/v1/querybatch"

# Map OSV severity labels to mirdan severity
_SEVERITY_MAP = {
    "CRITICAL": "critical",
    "HIGH": "high",
    "MODERATE": "medium",
    "MEDIUM": "medium",
    "LOW": "low",
}


class VulnCache:
    """TTL-based cache for vulnerability lookups."""

    def __init__(self, cache_path: Path, ttl: int = 86400) -> None:
        self._path = cache_path
        self._ttl = ttl
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _cache_key(self, pkg: PackageInfo) -> str:
        return f"{pkg.ecosystem}:{pkg.name}:{pkg.version}"

    def get(self, pkg: PackageInfo) -> list[VulnFinding] | None:
        """Get cached results, or None if expired/missing."""
        key = self._cache_key(pkg)
        entry = self._data.get(key)
        if not entry:
            return None
        checked_at = datetime.fromisoformat(entry["checked_at"])
        age = (datetime.now(UTC) - checked_at).total_seconds()
        if age > self._ttl:
            return None
        return [VulnFinding(**f) for f in entry["findings"]]

    def store(self, pkg: PackageInfo, findings: list[VulnFinding]) -> None:
        """Store results in cache."""
        key = self._cache_key(pkg)
        self._data[key] = {
            "checked_at": datetime.now(UTC).isoformat(),
            "findings": [f.to_dict() for f in findings],
        }
        # Limit cache size
        if len(self._data) > 1000:
            sorted_keys = sorted(self._data, key=lambda k: self._data[k]["checked_at"])
            for old_key in sorted_keys[:100]:
                del self._data[old_key]
        self._save()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))


class VulnScanner:
    """Scan packages for vulnerabilities using OSV API."""

    def __init__(self, cache_dir: Path, ttl: int = 86400) -> None:
        self._cache = VulnCache(cache_dir / "vuln-cache.json", ttl=ttl)

    def check_cached(self, packages: list[PackageInfo]) -> list[VulnFinding]:
        """Return cached vulnerability findings only (no network calls)."""
        findings: list[VulnFinding] = []
        for pkg in packages:
            cached = self._cache.get(pkg)
            if cached:
                findings.extend(cached)
        return findings

    async def scan(self, packages: list[PackageInfo]) -> list[VulnFinding]:
        """Scan packages against OSV API, updating cache."""
        uncached: list[PackageInfo] = []
        all_findings: list[VulnFinding] = []

        for pkg in packages:
            cached = self._cache.get(pkg)
            if cached is not None:
                all_findings.extend(cached)
            else:
                uncached.append(pkg)

        if uncached:
            new_findings = await asyncio.to_thread(self._query_osv, uncached)
            all_findings.extend(new_findings)

        return all_findings

    def _query_osv(self, packages: list[PackageInfo]) -> list[VulnFinding]:
        """Synchronous OSV API batch query."""
        queries = []
        for pkg in packages:
            q: dict[str, Any] = {"package": {"name": pkg.name, "ecosystem": pkg.ecosystem}}
            if pkg.version:
                q["version"] = pkg.version
            queries.append(q)

        all_findings: list[VulnFinding] = []
        for i in range(0, len(queries), 1000):
            batch = queries[i : i + 1000]
            batch_pkgs = packages[i : i + 1000]
            try:
                findings = self._execute_batch(batch, batch_pkgs)
                all_findings.extend(findings)
            except (urllib.error.URLError, OSError) as e:
                logger.warning("OSV API query failed: %s", e)
                # Cache empty results so we don't retry immediately
                for pkg in batch_pkgs:
                    self._cache.store(pkg, [])

        return all_findings

    def _execute_batch(
        self, queries: list[dict[str, Any]], packages: list[PackageInfo]
    ) -> list[VulnFinding]:
        """Execute a single batch request to OSV API."""
        body = json.dumps({"queries": queries}).encode()
        req = urllib.request.Request(  # noqa: S310
            OSV_API_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            data = json.loads(resp.read())

        findings: list[VulnFinding] = []
        for pkg, result in zip(packages, data.get("results", []), strict=False):
            pkg_findings = self._parse_vulns(pkg, result.get("vulns", []))
            self._cache.store(pkg, pkg_findings)
            findings.extend(pkg_findings)

        return findings

    def _parse_vulns(
        self, pkg: PackageInfo, vulns: list[dict[str, Any]]
    ) -> list[VulnFinding]:
        """Parse OSV vulnerability entries into VulnFindings."""
        findings: list[VulnFinding] = []
        for vuln in vulns:
            severity = self._extract_severity(vuln)
            fixed = self._extract_fixed_version(vuln, pkg.ecosystem)
            findings.append(
                VulnFinding(
                    package=pkg.name,
                    version=pkg.version,
                    ecosystem=pkg.ecosystem,
                    vuln_id=vuln.get("id", "UNKNOWN"),
                    severity=severity,
                    summary=vuln.get("summary", vuln.get("details", ""))[:200],
                    fixed_version=fixed,
                    advisory_url=next(
                        (
                            r["url"]
                            for r in vuln.get("references", [])
                            if r.get("type") == "WEB"
                        ),
                        "",
                    ),
                )
            )
        return findings

    def _extract_severity(self, vuln: dict[str, Any]) -> str:
        """Extract severity from OSV vulnerability data."""
        # Try CVSS score from severity array
        for sev in vuln.get("severity", []):
            if sev.get("type") == "CVSS_V3":
                score_str = sev.get("score", "")
                # OSV "score" field may contain either:
                # - A numeric base score (e.g., "9.8") — parsed below
                # - A CVSS vector string ("CVSS:3.1/AV:N/AC:L/...") — float() fails,
                #   falls through to ecosystem_specific/database_specific fallbacks
                try:
                    base_score = float(score_str)
                except (ValueError, TypeError):
                    continue
                if base_score >= 9.0:
                    return "critical"
                if base_score >= 7.0:
                    return "high"
                if base_score >= 4.0:
                    return "medium"
                return "low"

        # Try ecosystem_specific severity (GitHub, PyPI advisories)
        for affected in vuln.get("affected", []):
            eco_sev = (
                affected.get("ecosystem_specific", {})
                .get("severity", "")
                .upper()
            )
            if eco_sev in _SEVERITY_MAP:
                return _SEVERITY_MAP[eco_sev]

        # Try database_specific
        db_sev = vuln.get("database_specific", {}).get("severity", "").upper()
        if db_sev in _SEVERITY_MAP:
            return _SEVERITY_MAP[db_sev]

        return "medium"

    def _extract_fixed_version(self, vuln: dict[str, Any], ecosystem: str) -> str:
        """Extract the fixed version from affected ranges."""
        for affected in vuln.get("affected", []):
            for rng in affected.get("ranges", []):
                for event in rng.get("events", []):
                    if "fixed" in event:
                        return str(event["fixed"])
        return ""
