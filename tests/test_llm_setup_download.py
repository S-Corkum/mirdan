"""Tests for GGUF download and Ollama pull in the setup wizard."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mirdan.cli.llm_setup_command import (
    _download_gguf,
    _download_ollama,
    _install_llamacpp,
)

# Patch targets — these are imported inside the function body at call time,
# so we patch the source module.
_SSL = "mirdan.core.ssl_config"


# ---------------------------------------------------------------------------
# _download_gguf
# ---------------------------------------------------------------------------


class TestDownloadGguf:
    """Tests for _download_gguf()."""

    def test_skips_in_offline_mode(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch(f"{_SSL}._is_offline_mode", return_value=True):
            _download_gguf("model.gguf", "org/repo")

        out = capsys.readouterr().out
        assert "Offline mode" in out

    def test_skips_when_already_downloaded(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        models_dir = tmp_path / ".mirdan" / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "model.gguf").write_bytes(b"fake")

        with (
            patch(f"{_SSL}._is_offline_mode", return_value=False),
            patch(f"{_SSL}.configure_ssl"),
            patch("pathlib.Path.home", return_value=tmp_path),
        ):
            _download_gguf("model.gguf", "org/repo")

        out = capsys.readouterr().out
        assert "already downloaded" in out

    def test_uses_hf_endpoint_when_set(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import httpx

        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_bytes.return_value = [b"x" * 100]
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with (
            patch(f"{_SSL}._is_offline_mode", return_value=False),
            patch(f"{_SSL}.configure_ssl"),
            patch(
                f"{_SSL}.get_hf_endpoint",
                return_value="https://artifactory.corp.com/hf",
            ),
            patch("pathlib.Path.home", return_value=tmp_path),
            patch("httpx.stream", return_value=mock_response),
        ):
            _download_gguf("model.gguf", "org/repo")

        out = capsys.readouterr().out
        assert "artifactory.corp.com" in out

    def test_sends_auth_header_when_token_set(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_bytes.return_value = [b"x" * 100]
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with (
            patch(f"{_SSL}._is_offline_mode", return_value=False),
            patch(f"{_SSL}.configure_ssl"),
            patch(f"{_SSL}.get_hf_endpoint", return_value=None),
            patch(f"{_SSL}.get_hf_token", return_value="my-secret-token"),
            patch("pathlib.Path.home", return_value=tmp_path),
            patch("httpx.stream", return_value=mock_response) as mock_stream,
        ):
            _download_gguf("model.gguf", "org/repo")

        # Verify Authorization header was passed
        call_kwargs = mock_stream.call_args
        assert call_kwargs[1]["headers"]["Authorization"] == "Bearer my-secret-token"

    def test_no_auth_header_without_token(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_bytes.return_value = [b"x" * 100]
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with (
            patch(f"{_SSL}._is_offline_mode", return_value=False),
            patch(f"{_SSL}.configure_ssl"),
            patch(f"{_SSL}.get_hf_endpoint", return_value=None),
            patch(f"{_SSL}.get_hf_token", return_value=None),
            patch("pathlib.Path.home", return_value=tmp_path),
            patch("httpx.stream", return_value=mock_response) as mock_stream,
        ):
            _download_gguf("model.gguf", "org/repo")

        # No Authorization header when no token
        call_kwargs = mock_stream.call_args
        assert "Authorization" not in call_kwargs[1]["headers"]

    def test_uses_default_hf_url(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_bytes.return_value = [b"x" * 100]
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with (
            patch(f"{_SSL}._is_offline_mode", return_value=False),
            patch(f"{_SSL}.configure_ssl"),
            patch(f"{_SSL}.get_hf_endpoint", return_value=None),
            patch("pathlib.Path.home", return_value=tmp_path),
            patch("httpx.stream", return_value=mock_response),
        ):
            _download_gguf("model.gguf", "org/repo")

        out = capsys.readouterr().out
        assert "huggingface.co" in out

    def test_cleans_up_partial_on_http_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import httpx

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        exc = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_resp
        )

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(side_effect=exc)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch(f"{_SSL}._is_offline_mode", return_value=False),
            patch(f"{_SSL}.configure_ssl"),
            patch(f"{_SSL}.get_hf_endpoint", return_value=None),
            patch("pathlib.Path.home", return_value=tmp_path),
            patch("httpx.stream", return_value=mock_ctx),
        ):
            _download_gguf("model.gguf", "org/repo")

        out = capsys.readouterr().out
        assert "HTTP 404" in out
        # Partial file should be cleaned up
        assert not (tmp_path / ".mirdan" / "models" / "model.gguf").exists()

    def test_cleans_up_partial_on_network_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import httpx

        with (
            patch(f"{_SSL}._is_offline_mode", return_value=False),
            patch(f"{_SSL}.configure_ssl"),
            patch(f"{_SSL}.get_hf_endpoint", return_value=None),
            patch("pathlib.Path.home", return_value=tmp_path),
            patch("httpx.stream", side_effect=httpx.ConnectError("refused")),
        ):
            _download_gguf("model.gguf", "org/repo")

        out = capsys.readouterr().out
        assert "Network error" in out
        assert "MIRDAN_HF_ENDPOINT" in out


# ---------------------------------------------------------------------------
# _install_llamacpp
# ---------------------------------------------------------------------------


class TestInstallLlamacpp:
    """Tests for _install_llamacpp()."""

    def test_installs_with_metal_on_apple_silicon(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/uv"),
            patch("mirdan.cli.llm_setup_command.subprocess.run") as mock_run,
            patch("mirdan.cli.llm_setup_command._check_llamacpp", return_value=True),
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = _install_llamacpp(metal_capable=True)

        assert result is True
        call_kwargs = mock_run.call_args
        assert call_kwargs[1]["env"]["CMAKE_ARGS"] == "-DGGML_METAL=ON"
        # Should use uv pip install
        assert mock_run.call_args[0][0][0] == "uv"
        out = capsys.readouterr().out
        assert "Metal" in out

    def test_installs_without_metal_on_x86(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/uv"),
            patch("mirdan.cli.llm_setup_command.subprocess.run") as mock_run,
            patch("mirdan.cli.llm_setup_command._check_llamacpp", return_value=True),
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = _install_llamacpp(metal_capable=False)

        assert result is True
        call_kwargs = mock_run.call_args
        assert "CMAKE_ARGS" not in call_kwargs[1]["env"]

    def test_falls_back_to_pip_when_uv_absent(self) -> None:
        with (
            patch("shutil.which", return_value=None),
            patch("mirdan.cli.llm_setup_command.subprocess.run") as mock_run,
            patch("mirdan.cli.llm_setup_command._check_llamacpp", return_value=True),
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = _install_llamacpp(metal_capable=False)

        assert result is True
        cmd = mock_run.call_args[0][0]
        assert cmd[-1] == "llama-cpp-python"
        assert "-m" in cmd and "pip" in cmd

    def test_returns_false_on_install_failure(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            patch("shutil.which", return_value=None),
            patch("mirdan.cli.llm_setup_command.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=1, stderr="error: compilation failed\n"
            )
            result = _install_llamacpp(metal_capable=False)

        assert result is False
        out = capsys.readouterr().out
        assert "failed" in out

    def test_returns_false_on_timeout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            patch("shutil.which", return_value=None),
            patch(
                "mirdan.cli.llm_setup_command.subprocess.run",
                side_effect=subprocess.TimeoutExpired("pip", 600),
            ),
        ):
            result = _install_llamacpp(metal_capable=False)

        assert result is False
        out = capsys.readouterr().out
        assert "timed out" in out

    def test_verifies_import_after_install(self) -> None:
        with (
            patch("shutil.which", return_value=None),
            patch("mirdan.cli.llm_setup_command.subprocess.run") as mock_run,
            patch(
                "mirdan.cli.llm_setup_command._check_llamacpp", return_value=False
            ),
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = _install_llamacpp(metal_capable=False)

        # Install succeeded but import check failed
        assert result is False


# ---------------------------------------------------------------------------
# _download_ollama
# ---------------------------------------------------------------------------


class TestDownloadOllama:
    """Tests for _download_ollama()."""

    def test_calls_ollama_pull(self) -> None:
        with patch("mirdan.cli.llm_setup_command.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0)
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            _download_ollama("gemma4:e2b")

        mock_sub.run.assert_called_once_with(
            ["ollama", "pull", "gemma4:e2b"],
            timeout=600,
        )

    def test_reports_failure(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("mirdan.cli.llm_setup_command.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=1)
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            _download_ollama("gemma4:e2b")

        out = capsys.readouterr().out
        assert "failed" in out

    def test_handles_missing_ollama(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch("mirdan.cli.llm_setup_command.subprocess") as mock_sub:
            mock_sub.run.side_effect = FileNotFoundError()
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            _download_ollama("gemma4:e2b")

        out = capsys.readouterr().out
        assert "not found" in out

    def test_handles_timeout(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("mirdan.cli.llm_setup_command.subprocess") as mock_sub:
            mock_sub.run.side_effect = subprocess.TimeoutExpired("ollama", 600)
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            _download_ollama("gemma4:e2b")

        out = capsys.readouterr().out
        assert "timed out" in out
