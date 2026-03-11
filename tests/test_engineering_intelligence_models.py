"""Tests for 1.10.0 Engineering Intelligence config classes and models."""

from mirdan.config import (
    ArchitectureConfig,
    DecisionConfig,
    GuardrailConfig,
    MirdanConfig,
)
from mirdan.models import (
    ArchDriftResult,
    ArchLayer,
    ConfidenceAssessment,
    DecisionApproach,
    DecisionGuidance,
    GuardrailAnalysis,
    Violation,
)


class TestDecisionConfig:
    """Tests for DecisionConfig defaults and behavior."""

    def test_defaults(self) -> None:
        config = DecisionConfig()
        assert config.enabled is True
        assert config.max_decisions == 1

    def test_custom_values(self) -> None:
        config = DecisionConfig(enabled=False, max_decisions=3)
        assert config.enabled is False
        assert config.max_decisions == 3


class TestGuardrailConfig:
    """Tests for GuardrailConfig defaults and behavior."""

    def test_defaults(self) -> None:
        config = GuardrailConfig()
        assert config.enabled is True
        assert config.max_guardrails == 3

    def test_custom_values(self) -> None:
        config = GuardrailConfig(enabled=False, max_guardrails=5)
        assert config.enabled is False
        assert config.max_guardrails == 5


class TestArchitectureConfig:
    """Tests for ArchitectureConfig defaults and behavior."""

    def test_defaults(self) -> None:
        config = ArchitectureConfig()
        assert config.enabled is True
        assert config.warn_in_prompt is True

    def test_custom_values(self) -> None:
        config = ArchitectureConfig(enabled=False, warn_in_prompt=False)
        assert config.enabled is False
        assert config.warn_in_prompt is False


class TestMirdanConfigWiring:
    """Tests that new configs are wired into MirdanConfig."""

    def test_decisions_field_exists(self) -> None:
        config = MirdanConfig()
        assert isinstance(config.decisions, DecisionConfig)

    def test_guardrails_field_exists(self) -> None:
        config = MirdanConfig()
        assert isinstance(config.guardrails, GuardrailConfig)

    def test_architecture_field_exists(self) -> None:
        config = MirdanConfig()
        assert isinstance(config.architecture, ArchitectureConfig)

    def test_backward_compatible(self) -> None:
        """Existing configs without new fields should still work."""
        config = MirdanConfig()
        assert config.decisions.enabled is True
        assert config.guardrails.enabled is True
        assert config.architecture.enabled is True


class TestDecisionApproach:
    """Tests for DecisionApproach model."""

    def test_to_dict(self) -> None:
        approach = DecisionApproach(
            name="Redis",
            when_best="High read frequency, simple key-value lookups",
            when_avoid="Complex queries, tight budget",
            complexity="medium",
        )
        d = approach.to_dict()
        assert d["name"] == "Redis"
        assert d["when_best"] == "High read frequency, simple key-value lookups"
        assert d["when_avoid"] == "Complex queries, tight budget"
        assert d["complexity"] == "medium"

    def test_default_complexity(self) -> None:
        approach = DecisionApproach(name="Test", when_best="always", when_avoid="never")
        assert approach.complexity == "medium"


class TestDecisionGuidance:
    """Tests for DecisionGuidance model."""

    def test_empty(self) -> None:
        guidance = DecisionGuidance(domain="caching")
        d = guidance.to_dict()
        assert d["domain"] == "caching"
        assert d["approaches"] == []
        assert d["senior_questions"] == []

    def test_populated(self) -> None:
        guidance = DecisionGuidance(
            domain="caching",
            approaches=[
                DecisionApproach(name="Redis", when_best="high read", when_avoid="tight budget"),
            ],
            senior_questions=["What is the cache invalidation strategy?"],
        )
        d = guidance.to_dict()
        assert len(d["approaches"]) == 1
        assert d["approaches"][0]["name"] == "Redis"
        assert len(d["senior_questions"]) == 1


class TestGuardrailAnalysis:
    """Tests for GuardrailAnalysis model."""

    def test_to_dict(self) -> None:
        analysis = GuardrailAnalysis(
            domain="auth",
            guardrails=["Check OWASP top 10", "Validate JWT expiry"],
        )
        d = analysis.to_dict()
        assert d["domain"] == "auth"
        assert len(d["guardrails"]) == 2

    def test_empty_guardrails(self) -> None:
        analysis = GuardrailAnalysis(domain="general")
        assert analysis.guardrails == []


class TestConfidenceAssessment:
    """Tests for ConfidenceAssessment model."""

    def test_high_confidence(self) -> None:
        assessment = ConfidenceAssessment(
            level="high",
            reason="All checks passed with zero violations",
            attention_focus="None — code is clean",
        )
        d = assessment.to_dict()
        assert d["level"] == "high"
        assert "zero violations" in d["reason"]

    def test_low_confidence(self) -> None:
        assessment = ConfidenceAssessment(
            level="low",
            reason="Security violations detected",
            attention_focus="SQL injection in line 42",
        )
        d = assessment.to_dict()
        assert d["level"] == "low"
        assert d["attention_focus"] == "SQL injection in line 42"


class TestArchLayer:
    """Tests for ArchLayer model."""

    def test_defaults(self) -> None:
        layer = ArchLayer(name="domain")
        assert layer.name == "domain"
        assert layer.patterns == []
        assert layer.allowed_imports == []
        assert layer.forbidden_imports == []

    def test_populated(self) -> None:
        layer = ArchLayer(
            name="domain",
            patterns=["src/domain/**"],
            allowed_imports=["models"],
            forbidden_imports=["infrastructure"],
        )
        assert len(layer.patterns) == 1
        assert "infrastructure" in layer.forbidden_imports


class TestArchDriftResult:
    """Tests for ArchDriftResult model."""

    def test_empty(self) -> None:
        result = ArchDriftResult()
        d = result.to_dict()
        assert d["violations"] == []
        assert d["file_layer"] == ""
        assert d["context_warnings"] == []

    def test_with_violations(self) -> None:
        violation = Violation(
            id="ARCH004",
            rule="layer-boundary",
            category="architecture",
            severity="warning",
            message="Layer violation: domain imports infrastructure",
            line=10,
        )
        result = ArchDriftResult(
            violations=[violation],
            file_layer="domain",
            context_warnings=["File is in domain layer but imports from infrastructure"],
        )
        d = result.to_dict()
        assert len(d["violations"]) == 1
        assert d["violations"][0]["id"] == "ARCH004"
        assert d["file_layer"] == "domain"
        assert len(d["context_warnings"]) == 1
