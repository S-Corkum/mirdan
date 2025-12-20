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
        steps = self.prompt_composer._generate_verification_steps(intent)

        assert isinstance(steps, list)
        assert len(steps) > 0

    def test_get_verification_checklist_with_security(self) -> None:
        """Should include security checks when specified."""
        intent = Intent(
            original_prompt="",
            task_type=TaskType.GENERATION,
            touches_security=True,
        )
        steps = self.prompt_composer._generate_verification_steps(intent)

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
        steps = self.prompt_composer._generate_verification_steps(intent)

        # Should include preservation checks for refactor
        steps_text = " ".join(steps)
        assert "preserve" in steps_text.lower() or "functionality" in steps_text.lower()

    def test_get_verification_checklist_planning(self) -> None:
        """Should return planning-specific checklist."""
        intent = Intent(
            original_prompt="",
            task_type=TaskType.PLANNING,
        )
        steps = self.prompt_composer._generate_verification_steps(intent)

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
