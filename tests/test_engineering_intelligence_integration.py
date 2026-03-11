"""Integration tests for 1.10.0 Engineering Intelligence features."""

import pytest

from mirdan.config import MirdanConfig
from mirdan.providers import ComponentProvider


@pytest.fixture
def provider(tmp_path, monkeypatch):
    """Create a ComponentProvider with default config in a temp directory."""
    monkeypatch.chdir(tmp_path)
    config = MirdanConfig()
    return ComponentProvider(config)


class TestEnhancePromptIntegration:
    """End-to-end tests for enhance_prompt with new features."""

    @pytest.mark.asyncio
    async def test_decision_guidance_present(self, provider):
        """Prompt mentioning caching should return decision_guidance at STANDARD ceremony."""
        uc = provider.create_enhance_prompt_usecase(set())
        result = await uc.execute(
            "add redis caching with memoize to user service",
            ceremony_level="standard",
        )
        assert "decision_guidance" in result

    @pytest.mark.asyncio
    async def test_cognitive_guardrails_present(self, provider):
        """Prompt mentioning payment should return cognitive_guardrails at STANDARD ceremony."""
        uc = provider.create_enhance_prompt_usecase(set())
        result = await uc.execute(
            "implement payment webhook for stripe billing",
            ceremony_level="standard",
        )
        assert "cognitive_guardrails" in result

    @pytest.mark.asyncio
    async def test_no_features_for_simple_prompt(self, provider):
        """Simple prompt should not trigger decision or guardrail features."""
        uc = provider.create_enhance_prompt_usecase(set())
        result = await uc.execute("fix typo in README")
        assert "decision_guidance" not in result
        assert "cognitive_guardrails" not in result


class TestValidateCodeIntegration:
    """End-to-end tests for validate_code with new features."""

    @pytest.mark.asyncio
    async def test_confidence_present(self, provider):
        """Validation result should include confidence field."""
        uc = provider.create_validate_code_usecase(set())
        result = await uc.execute(
            code="def hello():\n    return 'world'\n",
            language="python",
        )
        assert "confidence" in result
        assert result["confidence"]["level"] in ("high", "medium", "low")

    @pytest.mark.asyncio
    async def test_confidence_with_test_file(self, provider):
        """Confidence should be higher with associated test file."""
        uc = provider.create_validate_code_usecase(set())
        result = await uc.execute(
            code="def hello():\n    return 'world'\n",
            language="python",
            test_file="tests/test_hello.py",
        )
        assert result["confidence"]["level"] in ("high", "medium")


class TestAllFeaturesDisabled:
    """Tests with all new features disabled via config."""

    @pytest.mark.asyncio
    async def test_enhance_prompt_graceful_noop(self, tmp_path, monkeypatch):
        """All features disabled should not break enhance_prompt."""
        monkeypatch.chdir(tmp_path)
        config = MirdanConfig()
        config.decisions.enabled = False
        config.guardrails.enabled = False
        config.architecture.enabled = False
        provider = ComponentProvider(config)

        uc = provider.create_enhance_prompt_usecase(set())
        result = await uc.execute("add redis caching with memoize")
        assert "decision_guidance" not in result
        assert "cognitive_guardrails" not in result
        assert "architecture_context" not in result

    @pytest.mark.asyncio
    async def test_validate_code_without_architecture(self, tmp_path, monkeypatch):
        """Validation should still work without architecture model."""
        monkeypatch.chdir(tmp_path)
        config = MirdanConfig()
        provider = ComponentProvider(config)

        uc = provider.create_validate_code_usecase(set())
        result = await uc.execute(
            code="x = 1\n",
            language="python",
        )
        # Confidence should be present (always wired)
        assert "confidence" in result
        # architecture_drift should not be (no model loaded)
        assert "architecture_drift" not in result
