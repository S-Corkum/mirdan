"""Tests for DecisionAnalyzer."""

from mirdan.config import DecisionConfig
from mirdan.core.decision_analyzer import DecisionAnalyzer
from mirdan.models import Intent, TaskType


def _make_intent(prompt: str) -> Intent:
    return Intent(original_prompt=prompt, task_type=TaskType.GENERATION)


class TestTriggerMatching:
    """Tests for decision domain trigger matching."""

    def test_caching_trigger(self) -> None:
        analyzer = DecisionAnalyzer(DecisionConfig())
        results = analyzer.analyze(_make_intent("add redis caching with memoize"))
        assert len(results) >= 1
        domains = [r.domain for r in results]
        assert any("Caching" in d for d in domains)

    def test_auth_trigger(self) -> None:
        analyzer = DecisionAnalyzer(DecisionConfig())
        results = analyzer.analyze(_make_intent("implement JWT authentication"))
        assert len(results) >= 1
        domains = [r.domain for r in results]
        assert any("Authentication" in d for d in domains)

    def test_no_match(self) -> None:
        analyzer = DecisionAnalyzer(DecisionConfig())
        results = analyzer.analyze(_make_intent("fix typo in README"))
        assert results == []

    def test_case_insensitive(self) -> None:
        analyzer = DecisionAnalyzer(DecisionConfig())
        results = analyzer.analyze(_make_intent("Add CACHING to the ENDPOINT"))
        assert len(results) >= 1


class TestMaxDecisionsCap:
    """Tests for max_decisions configuration."""

    def test_default_returns_one(self) -> None:
        analyzer = DecisionAnalyzer(DecisionConfig(max_decisions=1))
        # Prompt that matches multiple domains
        results = analyzer.analyze(
            _make_intent("add caching and authentication to the API endpoint")
        )
        assert len(results) <= 1

    def test_increased_cap(self) -> None:
        analyzer = DecisionAnalyzer(DecisionConfig(max_decisions=3))
        results = analyzer.analyze(
            _make_intent("add caching and authentication to the API endpoint")
        )
        assert len(results) <= 3
        # Should match at least 2 domains (caching + auth + api)
        assert len(results) >= 2


class TestDisabledConfig:
    """Tests for disabled configuration."""

    def test_disabled_returns_empty(self) -> None:
        analyzer = DecisionAnalyzer(DecisionConfig(enabled=False))
        results = analyzer.analyze(_make_intent("add caching to endpoint"))
        assert results == []


class TestGuidanceStructure:
    """Tests for DecisionGuidance output structure."""

    def test_has_approaches(self) -> None:
        analyzer = DecisionAnalyzer(DecisionConfig())
        results = analyzer.analyze(_make_intent("add caching"))
        assert len(results) >= 1
        guidance = results[0]
        assert len(guidance.approaches) > 0
        assert guidance.approaches[0].name
        assert guidance.approaches[0].when_best
        assert guidance.approaches[0].when_avoid

    def test_has_senior_questions(self) -> None:
        analyzer = DecisionAnalyzer(DecisionConfig())
        results = analyzer.analyze(_make_intent("add caching"))
        assert len(results) >= 1
        assert len(results[0].senior_questions) > 0

    def test_to_dict(self) -> None:
        analyzer = DecisionAnalyzer(DecisionConfig())
        results = analyzer.analyze(_make_intent("add caching"))
        d = results[0].to_dict()
        assert "domain" in d
        assert "approaches" in d
        assert "senior_questions" in d


class TestTemplateLoading:
    """Tests for YAML template loading."""

    def test_all_templates_loaded(self) -> None:
        analyzer = DecisionAnalyzer(DecisionConfig())
        assert len(analyzer._templates) == 8

    def test_all_templates_have_required_fields(self) -> None:
        analyzer = DecisionAnalyzer(DecisionConfig())
        for template in analyzer._templates:
            assert "name" in template
            assert "triggers" in template
            assert "approaches" in template
            assert "senior_questions" in template
