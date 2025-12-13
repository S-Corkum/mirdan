"""Tests for the Intent Analyzer module."""

import pytest

from mirdan.core.intent_analyzer import IntentAnalyzer
from mirdan.models import TaskType


@pytest.fixture
def analyzer() -> IntentAnalyzer:
    """Create an IntentAnalyzer instance."""
    return IntentAnalyzer()


class TestTaskTypeDetection:
    """Tests for task type detection."""

    def test_detects_generation_task(self, analyzer: IntentAnalyzer) -> None:
        """Should detect generation tasks."""
        intent = analyzer.analyze("add user authentication to my app")
        assert intent.task_type == TaskType.GENERATION

    def test_detects_refactor_task(self, analyzer: IntentAnalyzer) -> None:
        """Should detect refactor tasks."""
        intent = analyzer.analyze("refactor the authentication module")
        assert intent.task_type == TaskType.REFACTOR

    def test_detects_debug_task(self, analyzer: IntentAnalyzer) -> None:
        """Should detect debug tasks."""
        intent = analyzer.analyze("fix the login bug")
        assert intent.task_type == TaskType.DEBUG

    def test_detects_review_task(self, analyzer: IntentAnalyzer) -> None:
        """Should detect review tasks."""
        intent = analyzer.analyze("review the pull request for security issues")
        assert intent.task_type == TaskType.REVIEW

    def test_detects_test_task(self, analyzer: IntentAnalyzer) -> None:
        """Should detect test tasks."""
        intent = analyzer.analyze("write unit tests for the user service")
        assert intent.task_type == TaskType.TEST

    def test_returns_unknown_for_ambiguous(self, analyzer: IntentAnalyzer) -> None:
        """Should return unknown for ambiguous prompts."""
        intent = analyzer.analyze("something with the code")
        assert intent.task_type == TaskType.UNKNOWN


class TestLanguageDetection:
    """Tests for programming language detection."""

    def test_detects_python(self, analyzer: IntentAnalyzer) -> None:
        """Should detect Python."""
        intent = analyzer.analyze("create a FastAPI endpoint")
        assert intent.primary_language == "python"

    def test_detects_typescript(self, analyzer: IntentAnalyzer) -> None:
        """Should detect TypeScript."""
        intent = analyzer.analyze("add a Next.js page component")
        assert intent.primary_language == "typescript"

    def test_detects_javascript(self, analyzer: IntentAnalyzer) -> None:
        """Should detect JavaScript."""
        intent = analyzer.analyze("create a React component")
        assert intent.primary_language == "javascript"

    def test_returns_none_for_unknown(self, analyzer: IntentAnalyzer) -> None:
        """Should return None when language is unclear."""
        intent = analyzer.analyze("add a button")
        assert intent.primary_language is None


class TestFrameworkDetection:
    """Tests for framework detection."""

    def test_detects_react(self, analyzer: IntentAnalyzer) -> None:
        """Should detect React framework."""
        intent = analyzer.analyze("create a React component with hooks")
        assert "react" in intent.frameworks

    def test_detects_nextjs(self, analyzer: IntentAnalyzer) -> None:
        """Should detect Next.js framework."""
        intent = analyzer.analyze("add a Next.js API route")
        assert "next.js" in intent.frameworks

    def test_detects_fastapi(self, analyzer: IntentAnalyzer) -> None:
        """Should detect FastAPI framework."""
        intent = analyzer.analyze("create a FastAPI endpoint with Pydantic")
        assert "fastapi" in intent.frameworks

    def test_detects_multiple_frameworks(self, analyzer: IntentAnalyzer) -> None:
        """Should detect multiple frameworks."""
        intent = analyzer.analyze("add a Next.js page with Prisma and Tailwind")
        assert "next.js" in intent.frameworks
        assert "prisma" in intent.frameworks
        assert "tailwind" in intent.frameworks


class TestSecurityDetection:
    """Tests for security-related detection."""

    def test_detects_auth_security(self, analyzer: IntentAnalyzer) -> None:
        """Should detect authentication as security-related."""
        intent = analyzer.analyze("add user authentication")
        assert intent.touches_security is True

    def test_detects_password_security(self, analyzer: IntentAnalyzer) -> None:
        """Should detect password handling as security-related."""
        intent = analyzer.analyze("implement password reset")
        assert intent.touches_security is True

    def test_detects_token_security(self, analyzer: IntentAnalyzer) -> None:
        """Should detect JWT tokens as security-related."""
        intent = analyzer.analyze("add JWT token validation")
        assert intent.touches_security is True

    def test_non_security_task(self, analyzer: IntentAnalyzer) -> None:
        """Should not flag non-security tasks."""
        intent = analyzer.analyze("add a button to the header")
        assert intent.touches_security is False


class TestAmbiguityScoring:
    """Tests for ambiguity score calculation."""

    def test_short_prompts_are_ambiguous(self, analyzer: IntentAnalyzer) -> None:
        """Short prompts should have higher ambiguity."""
        intent = analyzer.analyze("fix it")
        assert intent.ambiguity_score >= 0.3

    def test_detailed_prompts_are_clear(self, analyzer: IntentAnalyzer) -> None:
        """Detailed prompts should have lower ambiguity."""
        intent = analyzer.analyze(
            "add user authentication using JWT tokens to the FastAPI backend"
        )
        assert intent.ambiguity_score < 0.5

    def test_vague_words_increase_ambiguity(self, analyzer: IntentAnalyzer) -> None:
        """Vague words should increase ambiguity score."""
        intent = analyzer.analyze("do something with that thing")
        assert intent.ambiguity_score >= 0.5


class TestEntityExtraction:
    """Tests for entity extraction integration."""

    def test_analyze_returns_entities(self, analyzer: IntentAnalyzer) -> None:
        """Should return extracted entities in Intent."""
        intent = analyzer.analyze("fix the validate_input() function in /src/utils/validators.py")
        assert len(intent.entities) >= 2

        # Check file path entity
        file_entities = [e for e in intent.entities if e.type.value == "file_path"]
        assert len(file_entities) == 1
        assert "/src/utils/validators.py" in file_entities[0].value

        # Check function entity
        func_entities = [e for e in intent.entities if e.type.value == "function_name"]
        assert len(func_entities) >= 1

    def test_entities_have_confidence(self, analyzer: IntentAnalyzer) -> None:
        """Entities should have confidence scores."""
        intent = analyzer.analyze("use requests.get to fetch /api/data.json")
        for entity in intent.entities:
            assert 0.0 <= entity.confidence <= 1.0

    def test_entities_serializable(self, analyzer: IntentAnalyzer) -> None:
        """Entities should serialize to dict."""
        intent = analyzer.analyze("modify /src/app.py")
        for entity in intent.entities:
            d = entity.to_dict()
            assert "type" in d
            assert "value" in d
            assert "confidence" in d
