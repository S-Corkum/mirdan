"""Tests for vuln_scanner.py — vulnerability scanning and caching."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mirdan.core.vuln_scanner import VulnCache, VulnScanner
from mirdan.models import PackageInfo, VulnFinding


def _make_pkg(
    name: str = "requests",
    version: str = "2.31.0",
    ecosystem: str = "PyPI",
) -> PackageInfo:
    return PackageInfo(name=name, version=version, ecosystem=ecosystem, source="test")


def _make_finding(
    package: str = "requests",
    vuln_id: str = "CVE-2024-1234",
    severity: str = "high",
) -> VulnFinding:
    return VulnFinding(
        package=package,
        version="2.31.0",
        ecosystem="PyPI",
        vuln_id=vuln_id,
        severity=severity,
        summary="Test vulnerability",
        fixed_version="2.32.0",
    )


class TestVulnCache:
    """Tests for VulnCache."""

    def test_store_and_retrieve(self, tmp_path: Path) -> None:
        cache = VulnCache(tmp_path / "cache.json", ttl=3600)
        pkg = _make_pkg()
        finding = _make_finding()
        cache.store(pkg, [finding])
        result = cache.get(pkg)
        assert result is not None
        assert len(result) == 1
        assert result[0].vuln_id == "CVE-2024-1234"

    def test_ttl_expiry(self, tmp_path: Path) -> None:
        cache = VulnCache(tmp_path / "cache.json", ttl=1)
        pkg = _make_pkg()
        cache.store(pkg, [_make_finding()])
        time.sleep(1.1)
        result = cache.get(pkg)
        assert result is None

    def test_cache_persistence(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.json"
        cache1 = VulnCache(cache_path, ttl=3600)
        pkg = _make_pkg()
        cache1.store(pkg, [_make_finding()])

        # Create new cache from same path
        cache2 = VulnCache(cache_path, ttl=3600)
        result = cache2.get(pkg)
        assert result is not None
        assert len(result) == 1

    def test_empty_cache_returns_none(self, tmp_path: Path) -> None:
        cache = VulnCache(tmp_path / "cache.json", ttl=3600)
        assert cache.get(_make_pkg()) is None


class TestVulnScanner:
    """Tests for VulnScanner."""

    def test_check_cached_empty(self, tmp_path: Path) -> None:
        scanner = VulnScanner(cache_dir=tmp_path, ttl=3600)
        findings = scanner.check_cached([_make_pkg()])
        assert findings == []

    def test_check_cached_returns_findings(self, tmp_path: Path) -> None:
        scanner = VulnScanner(cache_dir=tmp_path, ttl=3600)
        pkg = _make_pkg()
        # Pre-populate cache
        scanner._cache.store(pkg, [_make_finding()])
        findings = scanner.check_cached([pkg])
        assert len(findings) == 1

    @pytest.mark.asyncio
    async def test_scan_with_mocked_api(self, tmp_path: Path) -> None:
        scanner = VulnScanner(cache_dir=tmp_path, ttl=3600)
        pkg = _make_pkg()

        osv_response = {
            "results": [
                {
                    "vulns": [
                        {
                            "id": "CVE-2024-5678",
                            "summary": "Test vuln summary",
                            "severity": [{"type": "CVSS_V3", "score": "9.8"}],
                            "affected": [
                                {
                                    "ranges": [
                                        {
                                            "events": [
                                                {"introduced": "0"},
                                                {"fixed": "2.32.0"},
                                            ]
                                        }
                                    ]
                                }
                            ],
                            "references": [{"type": "WEB", "url": "https://example.com/advisory"}],
                        }
                    ]
                }
            ]
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(osv_response).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            findings = await scanner.scan([pkg])

        assert len(findings) == 1
        assert findings[0].vuln_id == "CVE-2024-5678"
        assert findings[0].severity == "critical"  # 9.8 → critical
        assert findings[0].fixed_version == "2.32.0"

    @pytest.mark.asyncio
    async def test_scan_network_error_graceful(self, tmp_path: Path) -> None:
        import urllib.error

        scanner = VulnScanner(cache_dir=tmp_path, ttl=3600)
        pkg = _make_pkg()

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection failed"),
        ):
            findings = await scanner.scan([pkg])

        # Should not crash, returns empty
        assert findings == []

    def test_severity_mapping_critical(self, tmp_path: Path) -> None:
        scanner = VulnScanner(cache_dir=tmp_path)
        vuln = {"severity": [{"type": "CVSS_V3", "score": "9.8"}]}
        assert scanner._extract_severity(vuln) == "critical"

    def test_severity_mapping_high(self, tmp_path: Path) -> None:
        scanner = VulnScanner(cache_dir=tmp_path)
        vuln = {"severity": [{"type": "CVSS_V3", "score": "7.5"}]}
        assert scanner._extract_severity(vuln) == "high"

    def test_severity_mapping_medium(self, tmp_path: Path) -> None:
        scanner = VulnScanner(cache_dir=tmp_path)
        vuln = {"severity": [{"type": "CVSS_V3", "score": "5.0"}]}
        assert scanner._extract_severity(vuln) == "medium"

    def test_severity_mapping_low(self, tmp_path: Path) -> None:
        scanner = VulnScanner(cache_dir=tmp_path)
        vuln = {"severity": [{"type": "CVSS_V3", "score": "2.0"}]}
        assert scanner._extract_severity(vuln) == "low"

    def test_severity_fallback_ecosystem_specific(self, tmp_path: Path) -> None:
        scanner = VulnScanner(cache_dir=tmp_path)
        vuln = {
            "affected": [{"ecosystem_specific": {"severity": "HIGH"}}],
        }
        assert scanner._extract_severity(vuln) == "high"

    def test_severity_fallback_database_specific(self, tmp_path: Path) -> None:
        scanner = VulnScanner(cache_dir=tmp_path)
        vuln = {"database_specific": {"severity": "LOW"}}
        assert scanner._extract_severity(vuln) == "low"

    def test_severity_cvss_vector_falls_through(self, tmp_path: Path) -> None:
        """CVSS vector strings can't be parsed as float — should fall through."""
        scanner = VulnScanner(cache_dir=tmp_path)
        vuln = {
            "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N"}],
            "database_specific": {"severity": "HIGH"},
        }
        assert scanner._extract_severity(vuln) == "high"

    def test_fixed_version_extraction(self, tmp_path: Path) -> None:
        scanner = VulnScanner(cache_dir=tmp_path)
        vuln = {"affected": [{"ranges": [{"events": [{"introduced": "0"}, {"fixed": "1.2.3"}]}]}]}
        assert scanner._extract_fixed_version(vuln, "PyPI") == "1.2.3"
