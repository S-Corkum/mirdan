"""Tests for HardwareDetector with mocked platform/os calls."""

from __future__ import annotations

from unittest.mock import patch

from mirdan.llm.health import HardwareDetector
from mirdan.models import HardwareProfile


class TestHardwareDetectorDetect:
    """Tests for HardwareDetector.detect()."""

    @patch("mirdan.llm.health.HardwareDetector._detect_gpu", return_value=None)
    @patch("mirdan.llm.health.HardwareDetector.get_available_memory_mb", return_value=8000)
    @patch("mirdan.llm.health.HardwareDetector._get_total_ram_mb", return_value=16384)
    @patch("mirdan.llm.health.platform.machine", return_value="x86_64")
    @patch("mirdan.llm.health.platform.system", return_value="Linux")
    def test_standard_x86_16gb(self, *_mocks: object) -> None:
        info = HardwareDetector.detect()

        assert info.architecture == "x86_64"
        assert info.total_ram_mb == 16384
        assert info.available_ram_mb == 8000
        assert info.detected_profile == HardwareProfile.STANDARD
        assert info.metal_capable is False

    @patch("mirdan.llm.health.HardwareDetector._detect_gpu", return_value="Apple M2 Pro")
    @patch("mirdan.llm.health.HardwareDetector.get_available_memory_mb", return_value=20000)
    @patch("mirdan.llm.health.HardwareDetector._get_total_ram_mb", return_value=32768)
    @patch("mirdan.llm.health.platform.machine", return_value="arm64")
    @patch("mirdan.llm.health.platform.system", return_value="Darwin")
    def test_enhanced_arm64_32gb(self, *_mocks: object) -> None:
        info = HardwareDetector.detect()

        assert info.architecture == "arm64"
        assert info.total_ram_mb == 32768
        assert info.detected_profile == HardwareProfile.ENHANCED
        assert info.metal_capable is True
        assert info.gpu_type == "Apple M2 Pro"

    @patch("mirdan.llm.health.HardwareDetector._detect_gpu", return_value="Apple M4 Max")
    @patch("mirdan.llm.health.HardwareDetector.get_available_memory_mb", return_value=50000)
    @patch("mirdan.llm.health.HardwareDetector._get_total_ram_mb", return_value=65536)
    @patch("mirdan.llm.health.platform.machine", return_value="arm64")
    @patch("mirdan.llm.health.platform.system", return_value="Darwin")
    def test_full_arm64_64gb(self, *_mocks: object) -> None:
        info = HardwareDetector.detect()

        assert info.total_ram_mb == 65536
        assert info.detected_profile == HardwareProfile.FULL
        assert info.metal_capable is True

    @patch("mirdan.llm.health.HardwareDetector._detect_gpu", return_value=None)
    @patch("mirdan.llm.health.HardwareDetector.get_available_memory_mb", return_value=40000)
    @patch("mirdan.llm.health.HardwareDetector._get_total_ram_mb", return_value=65536)
    @patch("mirdan.llm.health.platform.machine", return_value="x86_64")
    @patch("mirdan.llm.health.platform.system", return_value="Linux")
    def test_64gb_x86_is_enhanced_not_full(self, *_mocks: object) -> None:
        """64GB on x86_64 is ENHANCED, not FULL (FULL requires arm64)."""
        info = HardwareDetector.detect()

        assert info.detected_profile == HardwareProfile.ENHANCED
        assert info.metal_capable is False


class TestHardwareProfileClassification:
    """Tests for _classify_profile edge cases."""

    def test_boundary_16gb(self) -> None:
        assert HardwareDetector._classify_profile(16384, "x86_64") == HardwareProfile.STANDARD

    def test_boundary_17gb(self) -> None:
        assert HardwareDetector._classify_profile(16385, "x86_64") == HardwareProfile.ENHANCED

    def test_boundary_48gb_arm64(self) -> None:
        assert HardwareDetector._classify_profile(49152, "arm64") == HardwareProfile.ENHANCED

    def test_boundary_49gb_arm64(self) -> None:
        assert HardwareDetector._classify_profile(49153, "arm64") == HardwareProfile.FULL

    def test_zero_ram(self) -> None:
        assert HardwareDetector._classify_profile(0, "arm64") == HardwareProfile.STANDARD


class TestHardwareDetectorTotalRam:
    """Tests for _get_total_ram_mb."""

    @patch("mirdan.llm.health.os.sysconf")
    def test_reads_from_sysconf(self, mock_sysconf: object) -> None:
        # 16GB: 4194304 pages * 4096 bytes/page = 16GB
        import unittest.mock

        assert isinstance(mock_sysconf, unittest.mock.MagicMock)
        mock_sysconf.side_effect = lambda key: {
            "SC_PHYS_PAGES": 4194304,
            "SC_PAGE_SIZE": 4096,
        }[key]

        result = HardwareDetector._get_total_ram_mb()
        assert result == 16384

    @patch("mirdan.llm.health.os.sysconf", side_effect=OSError("not supported"))
    def test_returns_zero_on_error(self, _mock: object) -> None:
        assert HardwareDetector._get_total_ram_mb() == 0


class TestHardwareDetectorAvailableMemory:
    """Tests for get_available_memory_mb."""

    @patch("mirdan.llm.health.platform.system", return_value="Linux")
    @patch("mirdan.llm.health.os.sysconf")
    def test_linux_sysconf(self, mock_sysconf: object, _sys: object) -> None:
        import unittest.mock

        assert isinstance(mock_sysconf, unittest.mock.MagicMock)
        mock_sysconf.side_effect = lambda key: {
            "SC_AVPHYS_PAGES": 2097152,  # ~8GB with 4096 page size
            "SC_PAGE_SIZE": 4096,
        }[key]

        result = HardwareDetector.get_available_memory_mb()
        assert result == 8192
