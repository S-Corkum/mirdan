"""Tests for model-tier output optimization."""

from __future__ import annotations

from mirdan.core.output_formatter import (
    OutputFormatter,
    determine_format,
    determine_format_for_model,
)
from mirdan.models import ModelTier, OutputFormat


class TestModelTierFormatting:
    """Tests for model-tier output format selection."""

    def test_haiku_gets_minimal(self) -> None:
        """Haiku should get MINIMAL (ultra-compressed) output."""
        fmt = determine_format_for_model(ModelTier.HAIKU)
        assert fmt == OutputFormat.MINIMAL

    def test_sonnet_gets_compact(self) -> None:
        """Sonnet should get COMPACT output."""
        fmt = determine_format_for_model(ModelTier.SONNET)
        assert fmt == OutputFormat.COMPACT

    def test_opus_gets_full(self) -> None:
        """Opus should get FULL output."""
        fmt = determine_format_for_model(ModelTier.OPUS)
        assert fmt == OutputFormat.FULL

    def test_auto_gets_full(self) -> None:
        """AUTO should get FULL output."""
        fmt = determine_format_for_model(ModelTier.AUTO)
        assert fmt == OutputFormat.FULL


class TestModelTierIntegration:
    """Tests for model-tier formatting through OutputFormatter."""

    def setup_method(self) -> None:
        """Set up formatter."""
        self.formatter = OutputFormatter(compact_threshold=4000, minimal_threshold=1000)

    def test_haiku_enhanced_prompt(self) -> None:
        """Haiku model_tier should produce minimal enhanced prompt."""
        data = {
            "enhanced_prompt": "Write auth module",
            "task_type": "generation",
            "language": "python",
            "frameworks": ["fastapi"],
            "touches_security": True,
            "quality_requirements": ["req1", "req2", "req3", "req4"],
            "verification_steps": ["step1", "step2"],
            "tool_recommendations": [{"mcp": "context7", "priority": "high"}],
            "session_id": "abc123",
        }

        result = self.formatter.format_enhanced_prompt(data, model_tier=ModelTier.HAIKU)

        # Minimal format: task_type, language, touches_security, session_id
        assert "task_type" in result
        assert "touches_security" in result
        assert "session_id" in result
        # Should NOT have verbose fields
        assert "enhanced_prompt" not in result
        assert "quality_requirements" not in result

    def test_sonnet_enhanced_prompt(self) -> None:
        """Sonnet model_tier should produce compact enhanced prompt."""
        data = {
            "enhanced_prompt": "Write auth module",
            "task_type": "generation",
            "language": "python",
            "frameworks": ["fastapi"],
            "touches_security": True,
            "quality_requirements": ["req1", "req2", "req3", "req4", "req5"],
            "verification_steps": ["step1", "step2", "step3", "step4"],
            "tool_recommendations": [
                {"mcp": "context7", "priority": "high", "reason": "docs"},
                {"mcp": "enyal", "priority": "medium", "reason": "memory"},
            ],
        }

        result = self.formatter.format_enhanced_prompt(data, model_tier=ModelTier.SONNET)

        # Compact: truncated lists
        assert len(result.get("quality_requirements", [])) <= 3
        assert len(result.get("verification_steps", [])) <= 3

    def test_opus_enhanced_prompt(self) -> None:
        """Opus model_tier should preserve full output."""
        data = {
            "enhanced_prompt": "Write auth module",
            "task_type": "generation",
            "language": "python",
            "frameworks": ["fastapi"],
            "touches_security": True,
            "quality_requirements": ["req1", "req2", "req3", "req4", "req5"],
            "verification_steps": ["step1", "step2", "step3", "step4"],
            "tool_recommendations": [],
        }

        result = self.formatter.format_enhanced_prompt(data, model_tier=ModelTier.OPUS)

        # Full: everything preserved
        assert len(result["quality_requirements"]) == 5
        assert len(result["verification_steps"]) == 4

    def test_haiku_validation_result(self) -> None:
        """Haiku should produce minimal validation output."""
        data = {
            "passed": False,
            "score": 0.7,
            "language_detected": "python",
            "violations_count": {"error": 1},
            "violations": [
                {"id": "PY003", "severity": "error", "message": "bare except", "line": 5}
            ],
            "summary": "1 error found",
            "standards_checked": ["security", "style"],
        }

        result = self.formatter.format_validation_result(data, model_tier=ModelTier.HAIKU)

        # Minimal: passed, score, summary only
        assert result["passed"] is False
        assert result["score"] == 0.7
        assert "violations" not in result

    def test_token_budget_overrides_model_tier(self) -> None:
        """Token budget should override model tier if more compressed."""
        data = {
            "passed": True,
            "score": 1.0,
            "summary": "All good",
            "violations": [],
            "violations_count": {},
            "language_detected": "python",
        }

        # Opus model tier (FULL) but very low token budget (MINIMAL)
        result = self.formatter.format_validation_result(
            data, max_tokens=500, model_tier=ModelTier.OPUS
        )

        # Token budget should win (more compressed)
        assert "violations" not in result

    def test_model_tier_overrides_token_budget(self) -> None:
        """Model tier should override token budget if more compressed."""
        data = {
            "enhanced_prompt": "Write code",
            "task_type": "generation",
            "language": "python",
            "frameworks": [],
            "touches_security": False,
            "quality_requirements": ["req1", "req2", "req3", "req4"],
            "verification_steps": ["step1", "step2"],
            "tool_recommendations": [],
        }

        # Large token budget (FULL) but haiku model (MINIMAL)
        result = self.formatter.format_enhanced_prompt(
            data, max_tokens=10000, model_tier=ModelTier.HAIKU
        )

        # Model tier should win (more compressed)
        assert "enhanced_prompt" not in result


class TestTokenBudgetFormatting:
    """Tests for token budget format determination."""

    def test_zero_tokens_is_full(self) -> None:
        """0 max_tokens should give FULL format."""
        fmt = determine_format(0, compact_threshold=4000, minimal_threshold=1000)
        assert fmt == OutputFormat.FULL

    def test_small_budget_is_minimal(self) -> None:
        """Budget <= minimal_threshold should give MINIMAL."""
        fmt = determine_format(500, compact_threshold=4000, minimal_threshold=1000)
        assert fmt == OutputFormat.MINIMAL

    def test_medium_budget_is_compact(self) -> None:
        """Budget <= compact_threshold should give COMPACT."""
        fmt = determine_format(2000, compact_threshold=4000, minimal_threshold=1000)
        assert fmt == OutputFormat.COMPACT

    def test_large_budget_is_full(self) -> None:
        """Budget > compact_threshold should give FULL."""
        fmt = determine_format(5000, compact_threshold=4000, minimal_threshold=1000)
        assert fmt == OutputFormat.FULL
