"""Integration tests for 1.10.0 Engineering Intelligence features."""

from __future__ import annotations

from pathlib import Path

import pytest

from mirdan.config import MirdanConfig
from mirdan.providers import ComponentProvider


@pytest.fixture
def provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ComponentProvider:
    """Create a ComponentProvider with default config in a temp directory."""
    monkeypatch.chdir(tmp_path)
    config = MirdanConfig()
    return ComponentProvider(config)


class TestEnhancePromptIntegration:
    """End-to-end tests for enhance_prompt with new features."""

    @pytest.mark.asyncio
    async def test_decision_guidance_present(self, provider: ComponentProvider) -> None:
        """Prompt mentioning caching should return decision_guidance at STANDARD ceremony."""
        uc = provider.create_enhance_prompt_usecase(set())
        result = await uc.execute(
            "add redis caching with memoize to user service",
            ceremony_level="standard",
        )
        assert "decision_guidance" in result

    @pytest.mark.asyncio
    async def test_cognitive_guardrails_present(self, provider: ComponentProvider) -> None:
        """Prompt mentioning payment should return cognitive_guardrails at STANDARD ceremony."""
        uc = provider.create_enhance_prompt_usecase(set())
        result = await uc.execute(
            "implement payment webhook for stripe billing",
            ceremony_level="standard",
        )
        assert "cognitive_guardrails" in result

    @pytest.mark.asyncio
    async def test_no_features_for_simple_prompt(self, provider: ComponentProvider) -> None:
        """Simple prompt should not trigger decision or guardrail features."""
        uc = provider.create_enhance_prompt_usecase(set())
        result = await uc.execute("fix typo in README")
        assert "decision_guidance" not in result
        assert "cognitive_guardrails" not in result


class TestDesignGuidanceReachesModel:
    """2.2.0: design guidance must reach the model, not just sit in a side field.

    Before 2.2.0 ``decision_guidance`` was a top-level field never injected into the
    ``enhanced_prompt`` text, and the minimal/haiku output tier dropped it entirely.
    These guard that it now reaches the prompt at full/compact tiers and that a
    compressed nudge survives the minimal tier.
    """

    _DESIGN_PROMPT = "Add a REST API endpoint for user login with error handling"

    @pytest.mark.asyncio
    async def test_injected_into_prompt_text_full_tier(self, provider: ComponentProvider) -> None:
        uc = provider.create_enhance_prompt_usecase(set())
        result = await uc.execute(
            self._DESIGN_PROMPT,
            task_type="generation",
            ceremony_level="standard",
            model_tier="opus",
        )
        text = result.get("enhanced_prompt", "")
        assert "## Design Decisions" in text
        assert "**Your decision:**" in text
        assert "State the Design" in text  # rewritten TASK_GUIDANCE

    @pytest.mark.asyncio
    async def test_survives_compact_tier(self, provider: ComponentProvider) -> None:
        """COMPACT (sonnet) must preserve the design section in enhanced_prompt."""
        uc = provider.create_enhance_prompt_usecase(set())
        result = await uc.execute(
            self._DESIGN_PROMPT,
            task_type="generation",
            ceremony_level="standard",
            model_tier="sonnet",
        )
        assert "## Design Decisions" in result.get("enhanced_prompt", "")

    @pytest.mark.asyncio
    async def test_compressed_nudge_at_minimal_tier(self, provider: ComponentProvider) -> None:
        """MINIMAL (haiku) drops the full prompt but must keep a compressed nudge."""
        uc = provider.create_enhance_prompt_usecase(set())
        result = await uc.execute(
            self._DESIGN_PROMPT,
            task_type="generation",
            ceremony_level="standard",
            model_tier="haiku",
        )
        assert result.get("design_decisions")
        assert "design_directive" in result


class TestValidateCodeIntegration:
    """End-to-end tests for validate_code with new features."""

    @pytest.mark.asyncio
    async def test_confidence_present(self, provider: ComponentProvider) -> None:
        """Validation result should include confidence field."""
        uc = provider.create_validate_code_usecase(set())
        result = await uc.execute(
            code="def hello():\n    return 'world'\n",
            language="python",
        )
        assert "confidence" in result
        assert result["confidence"]["level"] in ("high", "medium", "low")

    @pytest.mark.asyncio
    async def test_confidence_with_test_file(self, provider: ComponentProvider) -> None:
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
    async def test_enhance_prompt_graceful_noop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
    async def test_validate_code_without_architecture(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
