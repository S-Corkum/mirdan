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
        intent = analyzer.analyze("add user authentication using JWT tokens to the FastAPI backend")
        assert intent.ambiguity_score < 0.5

    def test_vague_words_increase_ambiguity(self, analyzer: IntentAnalyzer) -> None:
        """Vague words should increase ambiguity score."""
        intent = analyzer.analyze("do something with that thing")
        assert intent.ambiguity_score >= 0.5


class TestClarifyingQuestions:
    """Tests for clarifying question generation."""

    def test_high_ambiguity_generates_questions(self, analyzer: IntentAnalyzer) -> None:
        """High ambiguity prompts should generate clarifying questions."""
        intent = analyzer.analyze("fix it")
        assert intent.ambiguity_score >= 0.6
        assert len(intent.clarifying_questions) > 0

    def test_clear_prompts_no_questions(self, analyzer: IntentAnalyzer) -> None:
        """Clear prompts should not generate questions."""
        intent = analyzer.analyze("add JWT authentication to the FastAPI backend in Python")
        assert intent.ambiguity_score < 0.6
        assert len(intent.clarifying_questions) == 0

    def test_unknown_task_type_generates_task_question(self, analyzer: IntentAnalyzer) -> None:
        """Unknown task type should generate task clarification question."""
        intent = analyzer.analyze("something with the code")
        assert intent.task_type == TaskType.UNKNOWN
        assert any("type of action" in q for q in intent.clarifying_questions)

    def test_short_prompt_generates_details_question(self, analyzer: IntentAnalyzer) -> None:
        """Short prompts should ask for more details."""
        intent = analyzer.analyze("fix it")
        assert any("more details" in q for q in intent.clarifying_questions)

    def test_vague_words_generate_specific_questions(self, analyzer: IntentAnalyzer) -> None:
        """Vague words should generate specific clarification questions."""
        intent = analyzer.analyze("do something with that thing")
        # Should have questions about vague words
        has_vague_word_question = any(
            "something" in q or "that" in q or "thing" in q for q in intent.clarifying_questions
        )
        assert has_vague_word_question

    def test_no_language_generates_language_question(self, analyzer: IntentAnalyzer) -> None:
        """Missing language should generate language question when ambiguity high."""
        # Prompt without language that triggers high ambiguity
        intent = analyzer.analyze("create something new")
        if intent.ambiguity_score >= 0.6:
            # Language question should be present (may be later in priority)
            assert any("programming language" in q for q in intent.clarifying_questions)

    def test_question_limit_is_four(self, analyzer: IntentAnalyzer) -> None:
        """Should limit to maximum 4 questions."""
        # Maximally ambiguous prompt with many vague words
        intent = analyzer.analyze("it this that something stuff thing")
        assert len(intent.clarifying_questions) <= 4

    def test_questions_have_priority_order(self, analyzer: IntentAnalyzer) -> None:
        """Task type question should come before details question."""
        intent = analyzer.analyze("something")  # Unknown task + short + vague
        if len(intent.clarifying_questions) >= 2:
            # Task type question should be first if present
            first_q = intent.clarifying_questions[0]
            assert "type of action" in first_q or "more details" in first_q


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


class TestPlanningDetection:
    """Test PLANNING task type detection."""

    def test_detect_planning_explicit_create_plan(self, analyzer: IntentAnalyzer) -> None:
        """'Create a plan to...' -> PLANNING"""
        intent = analyzer.analyze("Create a plan to implement user authentication")
        assert intent.task_type == TaskType.PLANNING

    def test_detect_planning_explicit_make_plan(self, analyzer: IntentAnalyzer) -> None:
        """'Make a plan for...' -> PLANNING"""
        intent = analyzer.analyze("Make a plan for the new caching layer")
        assert intent.task_type == TaskType.PLANNING

    def test_detect_planning_implementation_plan(self, analyzer: IntentAnalyzer) -> None:
        """'Implementation plan' -> PLANNING"""
        intent = analyzer.analyze("Write an implementation plan for the API")
        assert intent.task_type == TaskType.PLANNING

    def test_detect_planning_how_should(self, analyzer: IntentAnalyzer) -> None:
        """'How should I implement' -> PLANNING"""
        intent = analyzer.analyze("How should I implement the payment system?")
        assert intent.task_type == TaskType.PLANNING

    def test_detect_planning_break_down(self, analyzer: IntentAnalyzer) -> None:
        """'Break down' -> PLANNING"""
        intent = analyzer.analyze("Break down the feature into steps")
        assert intent.task_type == TaskType.PLANNING

    def test_planning_beats_generation(self, analyzer: IntentAnalyzer) -> None:
        """Planning words should beat generation words."""
        intent = analyzer.analyze("Plan to implement the login feature")
        assert intent.task_type == TaskType.PLANNING

    def test_generation_without_plan_words(self, analyzer: IntentAnalyzer) -> None:
        """'Add a feature' without plan words -> GENERATION not PLANNING"""
        intent = analyzer.analyze("Add a login feature to the app")
        assert intent.task_type == TaskType.GENERATION
