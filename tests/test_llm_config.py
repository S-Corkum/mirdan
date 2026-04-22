"""Tests for LLM configuration classes."""

from pathlib import Path

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
        assert config.test_timeout == 300
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
        config = LLMConfig(checks=CheckRunnerConfig(lint_command="flake8", test_timeout=120))
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


class TestLLMConfigOllamaUrlValidation:
    """Security tests: ollama_url must point to localhost only."""

    def test_accepts_localhost(self) -> None:
        config = LLMConfig(ollama_url="http://localhost:11434")
        assert config.ollama_url == "http://localhost:11434"

    def test_accepts_127_0_0_1(self) -> None:
        config = LLMConfig(ollama_url="http://127.0.0.1:11434")
        assert config.ollama_url == "http://127.0.0.1:11434"

    def test_accepts_ipv6_loopback(self) -> None:
        config = LLMConfig(ollama_url="http://[::1]:11434")
        assert config.ollama_url == "http://[::1]:11434"

    def test_accepts_custom_port(self) -> None:
        config = LLMConfig(ollama_url="http://localhost:9999")
        assert config.ollama_url == "http://localhost:9999"

    def test_rejects_remote_host(self) -> None:
        with pytest.raises(ValidationError, match="localhost"):
            LLMConfig(ollama_url="http://evil.com:11434")

    def test_rejects_internal_ip(self) -> None:
        with pytest.raises(ValidationError, match="localhost"):
            LLMConfig(ollama_url="http://192.168.1.100:11434")

    def test_rejects_cloud_metadata(self) -> None:
        with pytest.raises(ValidationError, match="localhost"):
            LLMConfig(ollama_url="http://169.254.169.254/latest/meta-data")

    def test_rejects_ftp_scheme(self) -> None:
        with pytest.raises(ValidationError, match="http"):
            LLMConfig(ollama_url="ftp://localhost:11434")


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
        config = MirdanConfig(llm=LLMConfig(enabled=True, backend="llamacpp", n_ctx=2048))
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


class TestConfigErrorHandling:
    """Regression tests for clean error surfacing on malformed configs.

    Pre-fix, ``MirdanConfig.load`` and ``_load_yaml_dict`` propagated raw
    ``yaml.YAMLError`` exceptions, which surfaced as a Python traceback
    at the CLI.
    """

    def test_malformed_yaml_raises_config_error(self, tmp_path: "Path") -> None:
        from mirdan.config import ConfigError, MirdanConfig

        bad = tmp_path / "config.yaml"
        bad.write_text("this is : : not valid: [}\n")
        with pytest.raises(ConfigError) as exc_info:
            MirdanConfig.load(bad)
        assert "Invalid YAML" in str(exc_info.value)
        assert str(bad) in str(exc_info.value)

    def test_non_mapping_yaml_raises_config_error(self, tmp_path: "Path") -> None:
        from mirdan.config import ConfigError, MirdanConfig

        bad = tmp_path / "config.yaml"
        bad.write_text("- just\n- a\n- list\n")
        with pytest.raises(ConfigError) as exc_info:
            MirdanConfig.load(bad)
        assert "mapping" in str(exc_info.value)

    def test_failed_pydantic_validation_raises_config_error(
        self, tmp_path: "Path"
    ) -> None:
        from mirdan.config import ConfigError, MirdanConfig

        bad = tmp_path / "config.yaml"
        # llm.backend must match the regex "^(auto|ollama|llamacpp)$"
        bad.write_text("llm:\n  backend: unsupported-backend\n")
        with pytest.raises(ConfigError) as exc_info:
            MirdanConfig.load(bad)
        assert "failed validation" in str(exc_info.value).lower()

    def test_missing_config_returns_defaults(self, tmp_path: "Path") -> None:
        from mirdan.config import MirdanConfig

        missing = tmp_path / "does-not-exist.yaml"
        cfg = MirdanConfig.load(missing)
        # Missing file is the expected no-op path — return defaults.
        assert cfg.version == "1.0"

    def test_malformed_yaml_in_merge_path(self, tmp_path: "Path") -> None:
        """When both inner and outer configs exist, malformed YAML in
        either must still surface as ConfigError via the merge helper.
        """
        from mirdan.config import ConfigError, MirdanConfig

        inner_dir = tmp_path / ".mirdan"
        inner_dir.mkdir()
        (inner_dir / "config.yaml").write_text("this: is: broken\n")
        (tmp_path / ".mirdan.yaml").write_text("quality:\n  security: strict\n")

        with pytest.raises(ConfigError) as exc_info:
            MirdanConfig.find_config_with_path(tmp_path)
        assert "Invalid YAML" in str(exc_info.value)
