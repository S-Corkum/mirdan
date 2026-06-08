"""Tests for ComponentProvider helpers — project-dir resolution.

Regression coverage for the 2.2.1 boot-crash fix: the MCP server must never root
`project_dir` at the filesystem root `/` (Claude Desktop / global launches), and must
honour explicit env overrides.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mirdan.providers import _resolve_project_dir


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MIRDAN_PROJECT_DIR", raising=False)
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)


class TestResolveProjectDir:
    def test_mirdan_env_override_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_env(monkeypatch)
        monkeypatch.setenv("MIRDAN_PROJECT_DIR", str(tmp_path))
        assert _resolve_project_dir(None) == tmp_path.resolve()

    def test_claude_env_honoured_when_mirdan_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_env(monkeypatch)
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
        assert _resolve_project_dir(None) == tmp_path.resolve()

    def test_mirdan_takes_precedence_over_claude(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_env(monkeypatch)
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv("MIRDAN_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(other))
        assert _resolve_project_dir(None) == tmp_path.resolve()

    def test_nonexistent_env_path_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_env(monkeypatch)
        monkeypatch.setenv("MIRDAN_PROJECT_DIR", str(tmp_path / "does-not-exist"))
        # Falls through to the discovered config dir.
        assert _resolve_project_dir(tmp_path) == tmp_path

    def test_config_dir_used_when_no_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_env(monkeypatch)
        assert _resolve_project_dir(tmp_path) == tmp_path

    def test_cwd_used_when_no_env_or_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_env(monkeypatch)
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        assert _resolve_project_dir(None) == tmp_path

    def test_root_cwd_falls_back_to_home_not_root(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_env(monkeypatch)
        monkeypatch.setattr(Path, "cwd", lambda: Path("/"))
        result = _resolve_project_dir(None)
        assert result != Path("/")
        assert result == Path.home()

    def test_env_var_set_to_root_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_env(monkeypatch)
        monkeypatch.setenv("MIRDAN_PROJECT_DIR", "/")
        monkeypatch.setattr(Path, "cwd", lambda: Path("/"))
        # An env var pointing at the filesystem root must be rejected by the guard.
        assert _resolve_project_dir(None) != Path("/")
