"""Tests for mirdan triage CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mirdan.cli.triage_command import _try_sidecar, run_triage


class TestTriageSidecar:
    """Tests for _try_sidecar()."""

    def test_returns_none_when_no_port_file(self, tmp_path: Path) -> None:
        with patch("mirdan.cli.triage_command.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            result = _try_sidecar({"prompt": "test"})
            assert result is None

    def test_returns_response_on_success(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "classification": "local_only",
            "confidence": 0.9,
            "reasoning": "simple",
        }

        with (
            patch("mirdan.cli.triage_command.Path") as mock_path,
            patch("mirdan.cli.triage_command.httpx.post", return_value=mock_resp),
        ):
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = "12345"

            result = _try_sidecar({"prompt": "fix lint"})

        assert result is not None
        assert result["classification"] == "local_only"

    def test_returns_none_on_connection_error(self) -> None:
        import httpx

        with (
            patch("mirdan.cli.triage_command.Path") as mock_path,
            patch(
                "mirdan.cli.triage_command.httpx.post",
                side_effect=httpx.ConnectError("refused"),
            ),
        ):
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = "12345"

            result = _try_sidecar({"prompt": "test"})
            assert result is None


class TestRunTriage:
    """Tests for run_triage()."""

    def test_outputs_fallback_when_no_sidecar(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch("mirdan.cli.triage_command._try_sidecar", return_value=None),
            patch("mirdan.cli.triage_command._write_to_session_bridge"),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.read.return_value = '{"prompt": "fix bug"}'
            run_triage(["--stdin"])

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["classification"] == "paid_required"
        assert data["confidence"] == 0.0

    def test_outputs_sidecar_result(self, capsys: pytest.CaptureFixture[str]) -> None:
        sidecar_result = {
            "classification": "local_only",
            "confidence": 0.9,
            "reasoning": "trivial",
        }
        with (
            patch("mirdan.cli.triage_command._try_sidecar", return_value=sidecar_result),
            patch("mirdan.cli.triage_command._write_to_session_bridge"),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.read.return_value = '{"prompt": "fix import"}'
            run_triage(["--stdin"])

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["classification"] == "local_only"

    def test_handles_raw_text_stdin(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch("mirdan.cli.triage_command._try_sidecar", return_value=None),
            patch("mirdan.cli.triage_command._write_to_session_bridge"),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.read.return_value = "just plain text prompt"
            run_triage(["--stdin"])

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["classification"] == "paid_required"

    def test_requires_stdin_flag(self) -> None:
        with pytest.raises(SystemExit):
            run_triage([])

    def test_writes_to_session_bridge(self) -> None:
        with (
            patch("mirdan.cli.triage_command._try_sidecar", return_value=None),
            patch("mirdan.cli.triage_command._write_to_session_bridge") as mock_write,
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.read.return_value = '{"prompt": "test"}'
            run_triage(["--stdin"])

        mock_write.assert_called_once()
