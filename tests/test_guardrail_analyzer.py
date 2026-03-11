"""Tests for GuardrailAnalyzer."""

from mirdan.config import GuardrailConfig
from mirdan.core.guardrail_analyzer import GuardrailAnalyzer
from mirdan.models import Intent, TaskType


def _make_intent(prompt: str) -> Intent:
    return Intent(original_prompt=prompt, task_type=TaskType.GENERATION)


class TestDomainMatching:
    """Tests for guardrail domain trigger matching."""

    def test_payment_trigger(self) -> None:
        analyzer = GuardrailAnalyzer(GuardrailConfig())
        results = analyzer.analyze(_make_intent("implement payment webhook"))
        assert len(results) >= 1
        domains = [r.domain for r in results]
        assert any("Payment" in d for d in domains)

    def test_auth_trigger(self) -> None:
        analyzer = GuardrailAnalyzer(GuardrailConfig())
        results = analyzer.analyze(_make_intent("add login and permission checks"))
        assert len(results) >= 1
        domains = [r.domain for r in results]
        assert any("Authentication" in d for d in domains)

    def test_no_match(self) -> None:
        analyzer = GuardrailAnalyzer(GuardrailConfig())
        results = analyzer.analyze(_make_intent("rename variable"))
        assert results == []


class TestMaxGuardrailsCap:
    """Tests for max_guardrails configuration."""

    def test_cap_limits_total_items(self) -> None:
        analyzer = GuardrailAnalyzer(GuardrailConfig(max_guardrails=2))
        results = analyzer.analyze(
            _make_intent("implement payment authentication with file upload")
        )
        total_items = sum(len(r.guardrails) for r in results)
        assert total_items <= 2

    def test_default_cap(self) -> None:
        analyzer = GuardrailAnalyzer(GuardrailConfig(max_guardrails=3))
        results = analyzer.analyze(
            _make_intent("implement payment authentication with file upload and caching")
        )
        total_items = sum(len(r.guardrails) for r in results)
        assert total_items <= 3


class TestDisabledConfig:
    """Tests for disabled configuration."""

    def test_disabled_returns_empty(self) -> None:
        analyzer = GuardrailAnalyzer(GuardrailConfig(enabled=False))
        results = analyzer.analyze(_make_intent("implement payment webhook"))
        assert results == []


class TestMultipleDomainMatch:
    """Tests for matching across multiple domains."""

    def test_payment_and_auth(self) -> None:
        analyzer = GuardrailAnalyzer(GuardrailConfig(max_guardrails=10))
        results = analyzer.analyze(_make_intent("implement payment auth with billing and login"))
        domains = [r.domain for r in results]
        assert len(domains) >= 2

    def test_guardrails_contain_thinking_prompts(self) -> None:
        analyzer = GuardrailAnalyzer(GuardrailConfig())
        results = analyzer.analyze(_make_intent("implement payment webhook"))
        assert len(results) >= 1
        # Guardrails should be questions or thinking prompts
        for guardrail in results[0].guardrails:
            assert len(guardrail) > 10  # Not empty/trivial


class TestGuardrailAnalysisStructure:
    """Tests for GuardrailAnalysis output structure."""

    def test_to_dict(self) -> None:
        analyzer = GuardrailAnalyzer(GuardrailConfig())
        results = analyzer.analyze(_make_intent("implement payment"))
        d = results[0].to_dict()
        assert "domain" in d
        assert "guardrails" in d


class TestYAMLLoading:
    """Tests for YAML domain loading."""

    def test_all_domains_loaded(self) -> None:
        analyzer = GuardrailAnalyzer(GuardrailConfig())
        assert len(analyzer._domains) == 10

    def test_all_domains_have_required_fields(self) -> None:
        analyzer = GuardrailAnalyzer(GuardrailConfig())
        for domain in analyzer._domains:
            assert "name" in domain
            assert "triggers" in domain
            assert "guardrails" in domain
