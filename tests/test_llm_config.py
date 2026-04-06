"""Tests for LLM configuration classes."""

import pytest
from pydantic import ValidationError

from mirdan.config import CheckRunnerConfig, LLMConfig, MirdanConfig


class TestCheckRunnerConfig:
    """Tests for CheckRunnerConfig defaults and validation."""

    def test_defaults(self) -> None:
        config = CheckRunnerConfig()
        assert config.lint_command == "ruff check"
        assert config.typecheck_command == "mypy"
        assert config.test_command == "pytest -x --tb=short"
        assert config.test_timeout == 30
        assert config.auto_fix_lint is True

    def test_custom_values(self) -> None:
        config = CheckRunnerConfig(
            lint_command="flake8",
            typecheck_command="pyright",
            test_command="pytest -v",
            test_timeout=60,
            auto_fix_lint=False,
        )
        assert config.lint_command == "flake8"
        assert config.typecheck_command == "pyright"
        assert config.test_timeout == 60
        assert config.auto_fix_lint is False


class TestLLMConfig:
    """Tests for LLMConfig defaults and validation."""

    def test_defaults(self) -> None:
        config = LLMConfig()
        assert config.enabled is False
        assert config.backend == "auto"
        assert config.ollama_url == "http://localhost:11434"
        assert config.gguf_dir == "~/.mirdan/models"
        assert config.model_keep_alive == "5m"
        assert config.n_ctx == 4096
        assert config.n_threads is None

    def test_feature_toggle_defaults(self) -> None:
        config = LLMConfig()
        assert config.triage is True
        assert config.smart_validation is True
        assert config.check_runner is True
        assert config.prompt_optimization is True
        assert config.research_agent is False

    def test_safety_defaults(self) -> None:
        config = LLMConfig()
        assert config.max_false_positive_ratio == 0.4
        assert config.validate_llm_fixes is True

    def test_nested_check_runner_config(self) -> None:
        config = LLMConfig()
        assert isinstance(config.checks, CheckRunnerConfig)
        assert config.checks.lint_command == "ruff check"

    def test_backend_validation_accepts_valid(self) -> None:
        for backend in ("auto", "ollama", "llamacpp"):
            config = LLMConfig(backend=backend)
            assert config.backend == backend

    def test_backend_validation_rejects_invalid(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(backend="invalid")

    def test_custom_check_runner(self) -> None:
        config = LLMConfig(
            checks=CheckRunnerConfig(lint_command="flake8", test_timeout=120)
        )
        assert config.checks.lint_command == "flake8"
        assert config.checks.test_timeout == 120

    def test_enabled_with_features_disabled(self) -> None:
        config = LLMConfig(
            enabled=True,
            triage=False,
            smart_validation=False,
            check_runner=False,
            prompt_optimization=False,
        )
        assert config.enabled is True
        assert config.triage is False


class TestMirdanConfigLLMIntegration:
    """Tests for LLMConfig integration with MirdanConfig."""

    def test_mirdan_config_has_llm_field(self) -> None:
        config = MirdanConfig()
        assert hasattr(config, "llm")
        assert isinstance(config.llm, LLMConfig)

    def test_llm_disabled_by_default(self) -> None:
        config = MirdanConfig()
        assert config.llm.enabled is False

    def test_llm_config_from_dict(self) -> None:
        config = MirdanConfig(llm={"enabled": True, "backend": "ollama"})  # type: ignore[arg-type]
        assert config.llm.enabled is True
        assert config.llm.backend == "ollama"

    def test_llm_config_serialization_roundtrip(self) -> None:
        config = MirdanConfig(
            llm=LLMConfig(enabled=True, backend="llamacpp", n_ctx=2048)
        )
        data = config.model_dump()
        restored = MirdanConfig(**data)
        assert restored.llm.enabled is True
        assert restored.llm.backend == "llamacpp"
        assert restored.llm.n_ctx == 2048

    def test_default_config_unchanged_for_existing_fields(self) -> None:
        """Adding LLMConfig must not alter any existing field defaults."""
        config = MirdanConfig()
        assert config.version == "1.0"
        assert config.quality.security == "strict"
        assert config.ceremony.enabled is True
        assert config.thresholds.severity_error_weight == 0.25
