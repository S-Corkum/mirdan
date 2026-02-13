"""Tests for the MCP server functionality.

Tests the underlying components that power the server tools.
The @mcp.tool() decorator wraps functions, so we test the core logic directly.
"""

from mirdan.config import MirdanConfig
from mirdan.core.code_validator import CodeValidator
from mirdan.core.intent_analyzer import IntentAnalyzer
from mirdan.core.orchestrator import MCPOrchestrator
from mirdan.core.plan_validator import PlanValidator
from mirdan.core.prompt_composer import PromptComposer
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import ContextBundle, Intent, TaskType


class TestEnhancePromptLogic:
    """Tests for the enhance_prompt underlying logic."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        config = MirdanConfig()
        self.intent_analyzer = IntentAnalyzer(config.project)
        self.quality_standards = QualityStandards(config=config.quality)
        self.prompt_composer = PromptComposer(self.quality_standards, config=config.enhancement)
        self.mcp_orchestrator = MCPOrchestrator(config.orchestration)

    def test_enhance_prompt_basic(self) -> None:
        """Should enhance a basic prompt."""
        prompt = "Create a login function in Python"
        intent = self.intent_analyzer.analyze(prompt)
        tool_recommendations = self.mcp_orchestrator.suggest_tools(intent)
        context = ContextBundle()

        enhanced = self.prompt_composer.compose(intent, context, tool_recommendations)
        result = enhanced.to_dict()

        assert "enhanced_prompt" in result
        assert "task_type" in result
        assert "language" in result
        assert "frameworks" in result
        assert "quality_requirements" in result
        assert "verification_steps" in result
        assert "tool_recommendations" in result

    def test_enhance_prompt_detects_python(self) -> None:
        """Should detect Python as the language."""
        prompt = "Write a Python function to parse JSON"
        intent = self.intent_analyzer.analyze(prompt)
        tool_recommendations = self.mcp_orchestrator.suggest_tools(intent)
        context = ContextBundle()

        enhanced = self.prompt_composer.compose(intent, context, tool_recommendations)
        result = enhanced.to_dict()

        assert result["language"] == "python"
        assert result["task_type"] == "generation"

    def test_enhance_prompt_detects_frameworks(self) -> None:
        """Should detect frameworks in the prompt."""
        prompt = "Create a FastAPI endpoint for user registration"
        intent = self.intent_analyzer.analyze(prompt)

        assert "fastapi" in intent.frameworks

    def test_enhance_prompt_with_task_type_override(self) -> None:
        """Should allow task type override."""
        prompt = "Look at this code and help me understand it"
        intent = self.intent_analyzer.analyze(prompt)

        # Override the task type
        intent.task_type = TaskType.REVIEW

        tool_recommendations = self.mcp_orchestrator.suggest_tools(intent)
        context = ContextBundle()

        enhanced = self.prompt_composer.compose(intent, context, tool_recommendations)
        result = enhanced.to_dict()

        assert result["task_type"] == "review"


class TestAnalyzeIntentLogic:
    """Tests for the analyze_intent underlying logic."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        config = MirdanConfig()
        self.intent_analyzer = IntentAnalyzer(config.project)

    def test_analyze_intent_basic(self) -> None:
        """Should analyze a basic prompt."""
        prompt = "Fix the bug in the login function"
        intent = self.intent_analyzer.analyze(prompt)

        assert intent.task_type is not None
        assert isinstance(intent.primary_language, str | None)
        assert isinstance(intent.frameworks, list)
        assert isinstance(intent.touches_security, bool)
        assert isinstance(intent.ambiguity_score, float)

    def test_analyze_intent_detects_debug(self) -> None:
        """Should detect debug task type."""
        prompt = "Debug the authentication error"
        intent = self.intent_analyzer.analyze(prompt)

        assert intent.task_type == TaskType.DEBUG

    def test_analyze_intent_detects_security(self) -> None:
        """Should detect security-related prompts."""
        prompt = "Implement password hashing for user authentication"
        intent = self.intent_analyzer.analyze(prompt)

        assert intent.touches_security is True

    def test_analyze_intent_ambiguity_levels(self) -> None:
        """Should calculate ambiguity score."""
        prompt = "Help me with the code"
        intent = self.intent_analyzer.analyze(prompt)

        # Vague prompt should have higher ambiguity
        assert 0.0 <= intent.ambiguity_score <= 1.0


class TestGetQualityStandardsLogic:
    """Tests for the get_quality_standards underlying logic."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        config = MirdanConfig()
        self.quality_standards = QualityStandards(config=config.quality)

    def test_get_quality_standards_python(self) -> None:
        """Should return Python quality standards."""
        result = self.quality_standards.get_all_standards(language="python")

        assert "language_standards" in result
        assert "security_standards" in result

    def test_get_quality_standards_with_framework(self) -> None:
        """Should return framework-specific standards."""
        result = self.quality_standards.get_all_standards(language="python", framework="fastapi")

        assert "framework_standards" in result

    def test_get_quality_standards_security_category(self) -> None:
        """Should filter to security category."""
        result = self.quality_standards.get_all_standards(language="python", category="security")

        assert "security_standards" in result

    def test_get_quality_standards_unknown_language(self) -> None:
        """Should handle unknown languages gracefully."""
        result = self.quality_standards.get_all_standards(language="unknown_language")

        # Should return dict with security/architecture even if language is unknown
        assert isinstance(result, dict)


class TestSuggestToolsLogic:
    """Tests for the suggest_tools underlying logic."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        config = MirdanConfig()
        self.intent_analyzer = IntentAnalyzer(config.project)
        self.mcp_orchestrator = MCPOrchestrator(config.orchestration)

    def test_suggest_tools_basic(self) -> None:
        """Should suggest tools for a given intent."""
        prompt = "Create a React component"
        intent = self.intent_analyzer.analyze(prompt)
        recommendations = self.mcp_orchestrator.suggest_tools(intent)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_suggest_tools_recommends_context7(self) -> None:
        """Should recommend context7 for framework-related tasks."""
        prompt = "Create a FastAPI endpoint"
        intent = self.intent_analyzer.analyze(prompt)
        recommendations = self.mcp_orchestrator.suggest_tools(intent)

        # Should include context7 in recommendations
        mcp_names = [rec.mcp for rec in recommendations]
        assert "context7" in mcp_names


class TestGetVerificationChecklistLogic:
    """Tests for the get_verification_checklist underlying logic."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        config = MirdanConfig()
        self.quality_standards = QualityStandards(config=config.quality)
        self.prompt_composer = PromptComposer(self.quality_standards, config=config.enhancement)

    def test_get_verification_checklist_generation(self) -> None:
        """Should return checklist for generation task."""
        intent = Intent(
            original_prompt="",
            task_type=TaskType.GENERATION,
        )
        steps = self.prompt_composer.generate_verification_steps(intent)

        assert isinstance(steps, list)
        assert len(steps) > 0

    def test_get_verification_checklist_with_security(self) -> None:
        """Should include security checks when specified."""
        intent = Intent(
            original_prompt="",
            task_type=TaskType.GENERATION,
            touches_security=True,
        )
        steps = self.prompt_composer.generate_verification_steps(intent)

        # Should have security-related verification steps
        steps_text = " ".join(steps)
        assert any(
            word in steps_text.lower()
            for word in ["password", "security", "credentials", "secrets"]
        )

    def test_get_verification_checklist_refactor(self) -> None:
        """Should return refactor-specific checklist."""
        intent = Intent(
            original_prompt="",
            task_type=TaskType.REFACTOR,
        )
        steps = self.prompt_composer.generate_verification_steps(intent)

        # Should include preservation checks for refactor
        steps_text = " ".join(steps)
        assert "preserve" in steps_text.lower() or "functionality" in steps_text.lower()

    def test_get_verification_checklist_planning(self) -> None:
        """Should return planning-specific checklist."""
        intent = Intent(
            original_prompt="",
            task_type=TaskType.PLANNING,
        )
        steps = self.prompt_composer.generate_verification_steps(intent)

        # Planning has specific verification steps
        steps_text = " ".join(steps)
        assert "grounding" in steps_text.lower() or "verified" in steps_text.lower()


class TestValidateCodeQualityLogic:
    """Tests for the validate_code_quality underlying logic."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        config = MirdanConfig()
        self.quality_standards = QualityStandards(config=config.quality)
        self.code_validator = CodeValidator(self.quality_standards, config=config.quality)

    def test_validate_code_quality_clean_code(self) -> None:
        """Should pass clean code."""
        code = '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''
        result = self.code_validator.validate(code, language="python")

        assert result.passed is True
        assert result.score > 0.8

    def test_validate_code_quality_with_violations(self) -> None:
        """Should detect code violations."""
        code = """
def process(data):
    try:
        result = eval(data)
    except:
        pass
    return result
"""
        result = self.code_validator.validate(code, language="python")

        # Should detect eval and bare except
        assert result.passed is False
        assert len(result.violations) > 0

    def test_validate_code_quality_auto_detect_language(self) -> None:
        """Should auto-detect language."""
        code = """
fn main() {
    let x = 5;
    println!("{}", x);
}
"""
        result = self.code_validator.validate(code, language="auto")

        assert result.language_detected == "rust"

    def test_validate_code_quality_empty_code(self) -> None:
        """Should handle empty code."""
        result = self.code_validator.validate("", language="python")

        assert result.passed is True
        assert "No code provided" in result.limitations[0]


class TestValidatePlanQualityLogic:
    """Tests for the validate_plan_quality underlying logic."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        config = MirdanConfig()
        self.plan_validator = PlanValidator(config.planning)

    def test_validate_plan_quality_good_plan(self) -> None:
        """Should score a well-structured plan highly."""
        plan = """
## Research Notes (Pre-Plan Verification)

### Files Verified
- `src/auth.py`: line 45 contains login function

### Step 1: Add import

**File:** `src/auth.py`
**Action:** Edit
**Details:**
- Line 1: Add `import bcrypt`
**Verify:** Read file, confirm import exists
**Grounding:** Read of src/auth.py confirmed file structure
"""
        result = self.plan_validator.validate(plan)

        assert 0.0 <= result.overall_score <= 1.0
        assert isinstance(result.ready_for_cheap_model, bool)
        assert isinstance(result.issues, list)

    def test_validate_plan_quality_vague_plan(self) -> None:
        """Should detect vague language in plans."""
        plan = """
I think we should probably add some code around line 50.
The function should be somewhere in the auth module.
"""
        result = self.plan_validator.validate(plan)

        # Should have low clarity score due to vague language
        assert result.clarity_score < 1.0
        assert len(result.issues) > 0

    def test_validate_plan_quality_missing_sections(self) -> None:
        """Should detect missing required sections."""
        plan = "Just do the thing."
        result = self.plan_validator.validate(plan)

        # Should detect missing Research Notes section
        assert result.completeness_score < 1.0
        assert any("Research Notes" in issue for issue in result.issues)

    def test_validate_plan_quality_target_model(self) -> None:
        """Should respect target model strictness."""
        plan = (
            "## Research Notes\n\n### Files Verified\n- file.py\n\n"
            "### Step 1: Do something\n\n**File:** path.py\n**Action:** Edit\n"
            "**Details:** stuff\n**Verify:** check\n**Grounding:** read"
        )

        result_haiku = self.plan_validator.validate(plan, target_model="haiku")
        result_capable = self.plan_validator.validate(plan, target_model="capable")

        # Both should return valid results
        assert isinstance(result_haiku.ready_for_cheap_model, bool)
        assert isinstance(result_capable.ready_for_cheap_model, bool)


class TestServerComponentIntegration:
    """Integration tests for server components working together."""

    def test_full_enhance_workflow(self) -> None:
        """Should complete full enhance workflow."""
        config = MirdanConfig()
        intent_analyzer = IntentAnalyzer(config.project)
        quality_standards = QualityStandards(config=config.quality)
        prompt_composer = PromptComposer(quality_standards, config=config.enhancement)
        mcp_orchestrator = MCPOrchestrator(config.orchestration)

        # Analyze intent
        prompt = "Create a REST API endpoint with FastAPI"
        intent = intent_analyzer.analyze(prompt)

        # Get tool recommendations
        tool_recommendations = mcp_orchestrator.suggest_tools(intent)

        # Compose enhanced prompt
        context = ContextBundle()
        enhanced = prompt_composer.compose(intent, context, tool_recommendations)
        result = enhanced.to_dict()

        # Verify complete result
        assert "enhanced_prompt" in result
        assert result["task_type"] == "generation"
        assert result["language"] == "python"
        assert "fastapi" in result["frameworks"]
        assert len(result["quality_requirements"]) > 0
        assert len(result["tool_recommendations"]) > 0


class TestInputSizeLimits:
    """Tests for input size limit validation."""

    def test_check_input_size_within_limit(self) -> None:
        """Should return None when input is within limit."""
        from mirdan.server import _check_input_size

        result = _check_input_size("short input", "prompt", 50_000)
        assert result is None

    def test_check_input_size_exceeds_limit(self) -> None:
        """Should return error dict when input exceeds limit."""
        from mirdan.server import _check_input_size

        oversized = "x" * 100
        result = _check_input_size(oversized, "prompt", 50)
        assert result is not None
        assert "error" in result
        assert result["max_length"] == 50
        assert result["actual_length"] == 100

    def test_check_input_size_at_boundary(self) -> None:
        """Should return None when input is exactly at the limit."""
        from mirdan.server import _check_input_size

        exact = "x" * 100
        result = _check_input_size(exact, "prompt", 100)
        assert result is None

    def test_check_input_size_error_message_format(self) -> None:
        """Error message should include field name and sizes."""
        from mirdan.server import _check_input_size

        result = _check_input_size("x" * 200, "code", 100)
        assert result is not None
        assert "code" in result["error"]
        assert "200" in result["error"]
        assert "100" in result["error"]

    def test_constants_are_sensible(self) -> None:
        """Size limit constants should be reasonable values."""
        from mirdan.server import (
            _MAX_CODE_LENGTH,
            _MAX_PLAN_LENGTH,
            _MAX_PROMPT_LENGTH,
        )

        assert _MAX_PROMPT_LENGTH == 50_000
        assert _MAX_CODE_LENGTH == 500_000
        assert _MAX_PLAN_LENGTH == 200_000
        # Code limit should be largest (files are big)
        assert _MAX_CODE_LENGTH > _MAX_PLAN_LENGTH > _MAX_PROMPT_LENGTH


class TestInputSizeLimitsIntegration:
    """Integration tests verifying tools reject oversized input."""

    def setup_method(self) -> None:
        """Set up components matching server.py initialization."""
        self.config = MirdanConfig()
        self.intent_analyzer = IntentAnalyzer(self.config.project)
        self.quality_standards = QualityStandards(config=self.config.quality)
        self.prompt_composer = PromptComposer(
            self.quality_standards, config=self.config.enhancement
        )
        self.mcp_orchestrator = MCPOrchestrator(self.config.orchestration)
        self.code_validator = CodeValidator(
            self.quality_standards,
            config=self.config.quality,
            thresholds=self.config.thresholds,
        )
        self.plan_validator = PlanValidator(self.config.planning)

    def _simulate_enhance_prompt(self, prompt: str) -> dict:
        """Simulate enhance_prompt tool logic (sync parts only)."""
        from mirdan.server import (
            _MAX_PROMPT_LENGTH,
            _check_input_size,
        )

        if error := _check_input_size(prompt, "prompt", _MAX_PROMPT_LENGTH):
            return error
        intent = self.intent_analyzer.analyze(prompt)
        recs = self.mcp_orchestrator.suggest_tools(intent)
        context = ContextBundle()
        enhanced = self.prompt_composer.compose(intent, context, recs)
        return enhanced.to_dict()

    def _simulate_analyze_intent(self, prompt: str) -> dict:
        """Simulate analyze_intent tool logic."""
        from mirdan.server import (
            _MAX_PROMPT_LENGTH,
            _check_input_size,
        )

        if error := _check_input_size(prompt, "prompt", _MAX_PROMPT_LENGTH):
            return error
        intent = self.intent_analyzer.analyze(prompt)
        return {"task_type": intent.task_type.value}

    def _simulate_validate_code(self, code: str) -> dict:
        """Simulate validate_code_quality tool logic."""
        from mirdan.server import _MAX_CODE_LENGTH, _check_input_size

        if error := _check_input_size(code, "code", _MAX_CODE_LENGTH):
            return error
        result = self.code_validator.validate(code=code, language="python")
        return result.to_dict()

    def _simulate_validate_plan(self, plan: str) -> dict:
        """Simulate validate_plan_quality tool logic."""
        from mirdan.server import _MAX_PLAN_LENGTH, _check_input_size

        if error := _check_input_size(plan, "plan", _MAX_PLAN_LENGTH):
            return error
        result = self.plan_validator.validate(plan)
        return result.to_dict()

    def test_enhance_prompt_rejects_oversized(self) -> None:
        """enhance_prompt should return error for oversized prompt."""
        oversized = "x" * 50_001
        result = self._simulate_enhance_prompt(oversized)
        assert "error" in result
        assert "prompt" in result["error"]

    def test_enhance_prompt_accepts_normal(self) -> None:
        """enhance_prompt should succeed with normal-sized prompt."""
        result = self._simulate_enhance_prompt("add a login feature")
        assert "enhanced_prompt" in result

    def test_analyze_intent_rejects_oversized(self) -> None:
        """analyze_intent should return error for oversized prompt."""
        oversized = "x" * 50_001
        result = self._simulate_analyze_intent(oversized)
        assert "error" in result

    def test_validate_code_rejects_oversized(self) -> None:
        """validate_code_quality should return error for oversized code."""
        oversized = "x" * 500_001
        result = self._simulate_validate_code(oversized)
        assert "error" in result
        assert "code" in result["error"]

    def test_validate_code_accepts_normal(self) -> None:
        """validate_code_quality should succeed with normal code."""
        result = self._simulate_validate_code("def foo(): pass")
        assert "passed" in result

    def test_validate_plan_rejects_oversized(self) -> None:
        """validate_plan_quality should return error for oversized plan."""
        oversized = "x" * 200_001
        result = self._simulate_validate_plan(oversized)
        assert "error" in result
        assert "plan" in result["error"]

    def test_validate_plan_accepts_normal(self) -> None:
        """validate_plan_quality should succeed with normal plan."""
        result = self._simulate_validate_plan("## Plan\nStep 1: do thing")
        assert "overall_score" in result

    def test_error_response_structure_consistent(self) -> None:
        """All oversized errors should have same structure."""
        from mirdan.server import _check_input_size

        for name, limit in [
            ("prompt", 10),
            ("code", 10),
            ("plan", 10),
        ]:
            result = _check_input_size("x" * 20, name, limit)
            assert result is not None
            assert set(result.keys()) == {"error", "max_length", "actual_length"}


class TestAnalyzeIntentResponse:
    """Tests for analyze_intent response fields."""

    def test_analyze_intent_includes_touches_rag(self) -> None:
        """analyze_intent response should include touches_rag."""
        analyzer = IntentAnalyzer()
        intent = analyzer.analyze("build a RAG pipeline")

        # Simulate what server.py does
        response = {
            "touches_rag": intent.touches_rag,
            "touches_knowledge_graph": intent.touches_knowledge_graph,
        }
        assert response["touches_rag"] is True

    def test_analyze_intent_includes_touches_knowledge_graph(self) -> None:
        """analyze_intent response should include touches_knowledge_graph."""
        analyzer = IntentAnalyzer()
        intent = analyzer.analyze("build a knowledge graph")

        response = {
            "touches_knowledge_graph": intent.touches_knowledge_graph,
        }
        assert response["touches_knowledge_graph"] is True

    def test_enhanced_prompt_to_dict_includes_touches_knowledge_graph(self) -> None:
        """EnhancedPrompt.to_dict() should include touches_knowledge_graph."""
        from mirdan.core.prompt_composer import PromptComposer

        standards = QualityStandards()
        composer = PromptComposer(standards)

        intent = Intent(
            original_prompt="build a knowledge graph",
            task_type=TaskType.GENERATION,
            touches_knowledge_graph=True,
        )
        context = ContextBundle()
        enhanced = composer.compose(intent, context, [])
        result = enhanced.to_dict()
        assert "touches_knowledge_graph" in result
        assert result["touches_knowledge_graph"] is True


class TestEndToEndKnowledgeGraphFlow:
    """End-to-end tests for the full KG detection → standards → checklist flow."""

    def test_kg_prompt_produces_kg_standards_and_verification(self) -> None:
        """A KG-related prompt should produce KG standards and verification steps."""
        config = MirdanConfig()
        analyzer = IntentAnalyzer(config.project)
        standards = QualityStandards(config=config.quality)
        composer = PromptComposer(standards, config=config.enhancement)

        intent = analyzer.analyze("Build a knowledge graph with neo4j for entity resolution")

        # Intent should detect KG
        assert intent.touches_knowledge_graph is True

        # Standards should include KG-related content
        quality_reqs = standards.render_for_intent(intent)
        reqs_text = " ".join(quality_reqs).lower()
        assert "graph" in reqs_text or "entity" in reqs_text or "provenance" in reqs_text

        # Verification steps should include KG checks
        context = ContextBundle()
        enhanced = composer.compose(intent, context, [])
        steps_text = " ".join(enhanced.verification_steps).lower()
        assert "graph" in steps_text
        assert "deduplication" in steps_text

    def test_rag_and_kg_both_detected_together(self) -> None:
        """A prompt mentioning both RAG and KG should set both flags."""
        analyzer = IntentAnalyzer()
        intent = analyzer.analyze(
            "implement a GraphRAG pipeline with knowledge graph and vector embeddings"
        )
        assert intent.touches_rag is True
        assert intent.touches_knowledge_graph is True

    def test_kg_standards_respect_stringency(self) -> None:
        """KG standards count should vary with framework stringency."""
        from mirdan.config import QualityConfig

        strict = QualityStandards(config=QualityConfig(framework="strict"))
        permissive = QualityStandards(config=QualityConfig(framework="permissive"))

        intent = Intent(
            original_prompt="build a knowledge graph",
            task_type=TaskType.GENERATION,
            touches_knowledge_graph=True,
        )

        strict_result = strict.render_for_intent(intent)
        permissive_result = permissive.render_for_intent(intent)
        assert len(strict_result) > len(permissive_result)


class TestLazyInitialization:
    """Tests for lazy component initialization."""

    def test_get_components_returns_components(self) -> None:
        """_get_components() should return a _Components instance."""
        from mirdan.server import _Components, _get_components

        result = _get_components()
        assert isinstance(result, _Components)
        assert result.intent_analyzer is not None
        assert result.quality_standards is not None
        assert result.code_validator is not None

    def test_get_components_returns_same_instance(self) -> None:
        """_get_components() should return the same singleton instance."""
        from mirdan.server import _get_components

        result1 = _get_components()
        result2 = _get_components()
        assert result1 is result2
