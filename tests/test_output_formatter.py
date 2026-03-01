"""Tests for the OutputFormatter component."""

from mirdan.core.output_formatter import (
    OutputFormatter,
    determine_format,
    determine_format_for_model,
    estimate_dict_tokens,
    estimate_tokens,
)
from mirdan.models import ModelTier, OutputFormat


class TestTokenEstimation:
    """Tests for token estimation functions."""

    def test_estimate_tokens_basic(self) -> None:
        """Should estimate tokens at ~4 chars per token."""
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("abcdefgh") == 2

    def test_estimate_tokens_empty(self) -> None:
        """Should return 1 for empty string (minimum)."""
        assert estimate_tokens("") == 1

    def test_estimate_tokens_long_text(self) -> None:
        """Should scale linearly with text length."""
        text = "x" * 4000
        assert estimate_tokens(text) == 1000

    def test_estimate_dict_tokens(self) -> None:
        """Should estimate tokens for a dictionary."""
        data = {"key": "value", "nested": {"a": 1}}
        result = estimate_dict_tokens(data)
        assert result > 0


class TestFormatDetermination:
    """Tests for format determination logic."""

    def test_zero_tokens_returns_full(self) -> None:
        """Zero max_tokens means unlimited (full output)."""
        assert determine_format(0, 4000, 1000) == OutputFormat.FULL

    def test_large_budget_returns_full(self) -> None:
        """Budget above compact threshold returns full."""
        assert determine_format(5000, 4000, 1000) == OutputFormat.FULL

    def test_compact_budget_returns_compact(self) -> None:
        """Budget between minimal and compact thresholds returns compact."""
        assert determine_format(2000, 4000, 1000) == OutputFormat.COMPACT

    def test_minimal_budget_returns_minimal(self) -> None:
        """Budget at or below minimal threshold returns minimal."""
        assert determine_format(500, 4000, 1000) == OutputFormat.MINIMAL

    def test_at_compact_threshold_returns_compact(self) -> None:
        """Budget exactly at compact threshold returns compact."""
        assert determine_format(4000, 4000, 1000) == OutputFormat.COMPACT

    def test_at_minimal_threshold_returns_minimal(self) -> None:
        """Budget exactly at minimal threshold returns minimal."""
        assert determine_format(1000, 4000, 1000) == OutputFormat.MINIMAL

    def test_model_tier_haiku_returns_minimal(self) -> None:
        """Haiku model tier returns minimal (ultra-compressed) format."""
        assert determine_format_for_model(ModelTier.HAIKU) == OutputFormat.MINIMAL

    def test_model_tier_sonnet_returns_compact(self) -> None:
        """Sonnet model tier returns compact format."""
        assert determine_format_for_model(ModelTier.SONNET) == OutputFormat.COMPACT

    def test_model_tier_opus_returns_full(self) -> None:
        """Opus model tier returns full format."""
        assert determine_format_for_model(ModelTier.OPUS) == OutputFormat.FULL

    def test_model_tier_auto_returns_full(self) -> None:
        """Auto model tier returns full format."""
        assert determine_format_for_model(ModelTier.AUTO) == OutputFormat.FULL


class TestOutputFormatterEnhancedPrompt:
    """Tests for enhance_prompt output formatting."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.formatter = OutputFormatter(compact_threshold=4000, minimal_threshold=1000)
        self.full_data: dict = {
            "enhanced_prompt": "Enhanced: do the thing with quality",
            "task_type": "generation",
            "language": "python",
            "frameworks": ["fastapi"],
            "extracted_entities": [{"type": "file_path", "value": "auth.py"}],
            "touches_security": True,
            "touches_rag": False,
            "touches_knowledge_graph": False,
            "ambiguity_score": 0.1,
            "clarifying_questions": [],
            "quality_requirements": [
                "Requirement 1",
                "Requirement 2",
                "Requirement 3",
                "Requirement 4",
                "Requirement 5",
            ],
            "verification_steps": [
                "Step 1",
                "Step 2",
                "Step 3",
                "Step 4",
            ],
            "tool_recommendations": [
                {"mcp": "context7", "action": "get_docs", "priority": "high", "reason": "fastapi"},
                {"mcp": "enyal", "action": "recall", "priority": "high", "reason": "conventions"},
                {"mcp": "filesystem", "action": "read", "priority": "medium", "reason": "patterns"},
                {"mcp": "github", "action": "commits", "priority": "low", "reason": "history"},
            ],
            "session_id": "abc123def456",
        }

    def test_full_format_returns_data_unchanged(self) -> None:
        """Full format should return data as-is."""
        result = self.formatter.format_enhanced_prompt(self.full_data, max_tokens=0)
        assert result == self.full_data

    def test_compact_format_truncates_requirements(self) -> None:
        """Compact format should limit quality_requirements to 3."""
        result = self.formatter.format_enhanced_prompt(self.full_data, max_tokens=2000)
        assert len(result["quality_requirements"]) == 3

    def test_compact_format_truncates_verification_steps(self) -> None:
        """Compact format should limit verification_steps to 3."""
        result = self.formatter.format_enhanced_prompt(self.full_data, max_tokens=2000)
        assert len(result["verification_steps"]) == 3

    def test_compact_format_simplifies_tool_recommendations(self) -> None:
        """Compact format should simplify tool recommendations."""
        result = self.formatter.format_enhanced_prompt(self.full_data, max_tokens=2000)
        recs = result["tool_recommendations"]
        assert len(recs) <= 3
        # Should only have mcp and priority
        for rec in recs:
            assert "mcp" in rec
            assert "priority" in rec
            assert "reason" not in rec

    def test_compact_format_removes_verbose_fields(self) -> None:
        """Compact format should remove verbose fields."""
        result = self.formatter.format_enhanced_prompt(self.full_data, max_tokens=2000)
        assert "extracted_entities" not in result
        assert "ambiguity_score" not in result
        assert "clarifying_questions" not in result

    def test_compact_format_preserves_session_id(self) -> None:
        """Compact format should preserve session_id."""
        result = self.formatter.format_enhanced_prompt(self.full_data, max_tokens=2000)
        assert result["session_id"] == "abc123def456"

    def test_minimal_format_essentials_only(self) -> None:
        """Minimal format should only include essentials."""
        result = self.formatter.format_enhanced_prompt(self.full_data, max_tokens=500)
        assert "task_type" in result
        assert "language" in result
        assert "touches_security" in result
        assert "session_id" in result
        # Should NOT include these
        assert "enhanced_prompt" not in result
        assert "quality_requirements" not in result
        assert "tool_recommendations" not in result

    def test_model_tier_haiku_produces_compact(self) -> None:
        """Haiku model tier should produce compact output."""
        result = self.formatter.format_enhanced_prompt(
            self.full_data, max_tokens=0, model_tier=ModelTier.HAIKU
        )
        # Should be compact (truncated lists)
        assert len(result.get("quality_requirements", [])) <= 3

    def test_model_tier_opus_returns_full(self) -> None:
        """Opus model tier with no token limit should return full output."""
        result = self.formatter.format_enhanced_prompt(
            self.full_data, max_tokens=0, model_tier=ModelTier.OPUS
        )
        assert result == self.full_data

    def test_token_budget_overrides_model_tier(self) -> None:
        """Token budget should override model tier when more restrictive."""
        result = self.formatter.format_enhanced_prompt(
            self.full_data, max_tokens=500, model_tier=ModelTier.OPUS
        )
        # Minimal output despite Opus tier
        assert "enhanced_prompt" not in result

    def test_compact_format_without_session_id(self) -> None:
        """Compact format should work when no session_id is present."""
        data = dict(self.full_data)
        del data["session_id"]
        result = self.formatter.format_enhanced_prompt(data, max_tokens=2000)
        assert "session_id" not in result


class TestOutputFormatterValidationResult:
    """Tests for validate_code_quality output formatting."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.formatter = OutputFormatter(compact_threshold=4000, minimal_threshold=1000)
        self.full_data: dict = {
            "passed": False,
            "score": 0.75,
            "language_detected": "python",
            "violations_count": {"error": 1, "warning": 2, "info": 0},
            "violations": [
                {
                    "id": "PY001",
                    "rule": "no-eval",
                    "category": "security",
                    "severity": "error",
                    "message": "Avoid eval()",
                    "line": 5,
                    "column": 10,
                    "code_snippet": "result = eval(data)",
                    "suggestion": "Use ast.literal_eval() instead",
                },
                {
                    "id": "PY003",
                    "rule": "no-bare-except",
                    "category": "style",
                    "severity": "warning",
                    "message": "Bare except clause",
                    "line": 7,
                    "column": 1,
                    "code_snippet": "except:",
                    "suggestion": "Catch specific exceptions",
                },
            ],
            "summary": "Code has 1 error(s) that should be fixed",
            "standards_checked": ["python", "security"],
            "limitations": [],
        }

    def test_full_format_returns_unchanged(self) -> None:
        """Full format should return data as-is."""
        result = self.formatter.format_validation_result(self.full_data, max_tokens=0)
        assert result == self.full_data

    def test_compact_format_strips_code_snippets(self) -> None:
        """Compact format should remove code_snippet and suggestion from violations."""
        result = self.formatter.format_validation_result(self.full_data, max_tokens=2000)
        for v in result["violations"]:
            assert "code_snippet" not in v
            assert "suggestion" not in v
            assert "id" in v
            assert "severity" in v
            assert "message" in v

    def test_compact_format_preserves_essentials(self) -> None:
        """Compact format should preserve pass/fail, score, summary."""
        result = self.formatter.format_validation_result(self.full_data, max_tokens=2000)
        assert result["passed"] is False
        assert result["score"] == 0.75
        assert "summary" in result

    def test_minimal_format_pass_fail_only(self) -> None:
        """Minimal format should only include pass/fail and score."""
        result = self.formatter.format_validation_result(self.full_data, max_tokens=500)
        assert result["passed"] is False
        assert result["score"] == 0.75
        assert "summary" in result
        assert "violations" not in result
        assert "violations_count" not in result


class TestOutputFormatterEdgeCases:
    """Edge case tests for OutputFormatter."""

    def test_empty_data(self) -> None:
        """Should handle empty data gracefully."""
        formatter = OutputFormatter()
        result = formatter.format_enhanced_prompt({}, max_tokens=2000)
        assert isinstance(result, dict)

    def test_negative_max_tokens_treated_as_unlimited(self) -> None:
        """Negative max_tokens should be treated as unlimited."""
        assert determine_format(-1, 4000, 1000) == OutputFormat.FULL

    def test_model_tier_auto_no_compression(self) -> None:
        """AUTO model tier should not trigger compression on its own."""
        formatter = OutputFormatter()
        data = {"key": "value"}
        result = formatter.format_enhanced_prompt(data, max_tokens=0, model_tier=ModelTier.AUTO)
        assert result == data
