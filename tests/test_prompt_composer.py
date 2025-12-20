"""Tests for the Prompt Composer module."""

import pytest

from mirdan.config import EnhancementConfig
from mirdan.core.prompt_composer import PromptComposer
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import ContextBundle, EnhancedPrompt, Intent, TaskType, ToolRecommendation


@pytest.fixture
def standards() -> QualityStandards:
    """Create a QualityStandards instance."""
    return QualityStandards()


@pytest.fixture
def composer(standards: QualityStandards) -> PromptComposer:
    """Create a PromptComposer instance."""
    return PromptComposer(standards)


class TestPromptComposerInit:
    """Tests for PromptComposer initialization."""

    def test_init_with_standards_only(self, standards: QualityStandards) -> None:
        """Should store standards and have None config."""
        composer = PromptComposer(standards)
        assert composer.standards is standards
        assert composer._config is None

    def test_init_with_config(self, standards: QualityStandards) -> None:
        """Should store both standards and config."""
        config = EnhancementConfig(verbosity="minimal")
        composer = PromptComposer(standards, config=config)
        assert composer.standards is standards
        assert composer._config is config


class TestCompose:
    """Tests for compose method."""

    def test_compose_returns_enhanced_prompt(self, composer: PromptComposer) -> None:
        """Should return EnhancedPrompt instance."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        assert isinstance(result, EnhancedPrompt)

    def test_compose_includes_quality_requirements(self, composer: PromptComposer) -> None:
        """Should populate quality_requirements."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
        )
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        assert len(result.quality_requirements) > 0

    def test_compose_includes_verification_steps(self, composer: PromptComposer) -> None:
        """Should populate verification_steps."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        assert len(result.verification_steps) >= 4  # Base steps

    def test_compose_includes_tool_recommendations(self, composer: PromptComposer) -> None:
        """Should pass through tool_recommendations."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        context = ContextBundle()
        recommendations = [ToolRecommendation(mcp="test", action="test action", priority="high")]
        result = composer.compose(intent, context, recommendations)
        assert result.tool_recommendations == recommendations


class TestVerificationSteps:
    """Tests for _generate_verification_steps method."""

    def test_base_steps_always_included(self, composer: PromptComposer) -> None:
        """Should include 4 base verification steps."""
        intent = Intent(original_prompt="test", task_type=TaskType.UNKNOWN)
        steps = composer._generate_verification_steps(intent)
        assert len(steps) >= 4
        assert any("imports" in s.lower() for s in steps)
        assert any("error handling" in s.lower() for s in steps)
        assert any("secrets" in s.lower() or "credentials" in s.lower() for s in steps)
        assert any("naming conventions" in s.lower() for s in steps)

    def test_generation_task_adds_integration_step(self, composer: PromptComposer) -> None:
        """Should add integration validation step for GENERATION tasks."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        steps = composer._generate_verification_steps(intent)
        assert any("integrates with existing patterns" in s.lower() for s in steps)

    def test_refactor_task_adds_preservation_step(self, composer: PromptComposer) -> None:
        """Should insert functionality preservation step at position 0 for REFACTOR."""
        intent = Intent(original_prompt="test", task_type=TaskType.REFACTOR)
        steps = composer._generate_verification_steps(intent)
        assert "functionality is preserved" in steps[0].lower()

    def test_refactor_task_adds_api_signature_step(self, composer: PromptComposer) -> None:
        """Should append API signature step for REFACTOR."""
        intent = Intent(original_prompt="test", task_type=TaskType.REFACTOR)
        steps = composer._generate_verification_steps(intent)
        assert any("api signatures" in s.lower() for s in steps)

    def test_debug_task_adds_root_cause_step(self, composer: PromptComposer) -> None:
        """Should insert root cause step at position 0 for DEBUG."""
        intent = Intent(original_prompt="test", task_type=TaskType.DEBUG)
        steps = composer._generate_verification_steps(intent)
        assert "root cause" in steps[0].lower()

    def test_debug_task_adds_regression_test_step(self, composer: PromptComposer) -> None:
        """Should append regression test step for DEBUG."""
        intent = Intent(original_prompt="test", task_type=TaskType.DEBUG)
        steps = composer._generate_verification_steps(intent)
        assert any("regression" in s.lower() for s in steps)

    def test_test_task_adds_coverage_steps(self, composer: PromptComposer) -> None:
        """Should add test coverage steps for TEST tasks."""
        intent = Intent(original_prompt="test", task_type=TaskType.TEST)
        steps = composer._generate_verification_steps(intent)
        assert any("edge cases" in s.lower() for s in steps)
        assert any("isolation" in s.lower() or "shared state" in s.lower() for s in steps)

    def test_security_task_adds_security_steps(self, composer: PromptComposer) -> None:
        """Should add security verification steps when touches_security=True."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            touches_security=True,
        )
        steps = composer._generate_verification_steps(intent)
        assert any("password" in s.lower() for s in steps)
        assert any("sensitive data" in s.lower() or "logged" in s.lower() for s in steps)
        assert any("sanitiz" in s.lower() for s in steps)


class TestBuildPromptText:
    """Tests for _build_prompt_text method."""

    def test_minimal_verbosity_excludes_requirements(self, standards: QualityStandards) -> None:
        """Should skip Quality Requirements section when verbosity is minimal."""
        config = EnhancementConfig(verbosity="minimal")
        composer = PromptComposer(standards, config=config)
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
        )
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        assert "## Quality Requirements" not in result.enhanced_text

    def test_balanced_verbosity_limits_requirements(self, standards: QualityStandards) -> None:
        """Should show only first 5 requirements when verbosity is balanced."""
        config = EnhancementConfig(verbosity="balanced")
        composer = PromptComposer(standards, config=config)
        # Create intent that generates many requirements
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=["fastapi", "react"],
            touches_security=True,
        )
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        # Count requirement lines (starting with "- ") in Quality Requirements section only
        req_section = result.enhanced_text.split("## Quality Requirements")
        if len(req_section) > 1:
            # Get text up to the next section header
            section_text = req_section[1].split("##")[0]
            req_lines = [line for line in section_text.split("\n") if line.strip().startswith("- ")]
            assert len(req_lines) <= 5

    def test_comprehensive_verbosity_shows_all(self, standards: QualityStandards) -> None:
        """Should show all requirements when verbosity is comprehensive."""
        config = EnhancementConfig(verbosity="comprehensive")
        composer = PromptComposer(standards, config=config)
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=["fastapi"],
            touches_security=True,
        )
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        assert "## Quality Requirements" in result.enhanced_text

    def test_include_verification_false_hides_section(self, standards: QualityStandards) -> None:
        """Should hide verification section when include_verification=False."""
        config = EnhancementConfig(include_verification=False)
        composer = PromptComposer(standards, config=config)
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        assert "## Before Completing" not in result.enhanced_text

    def test_include_tool_hints_false_hides_section(self, standards: QualityStandards) -> None:
        """Should hide tool recommendations when include_tool_hints=False."""
        config = EnhancementConfig(include_tool_hints=False)
        composer = PromptComposer(standards, config=config)
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        context = ContextBundle()
        recommendations = [ToolRecommendation(mcp="test", action="test action")]
        result = composer.compose(intent, context, recommendations)
        assert "## Recommended Tools" not in result.enhanced_text

    def test_context_section_when_patterns_exist(self, composer: PromptComposer) -> None:
        """Should render context section when patterns exist."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        context = ContextBundle(
            existing_patterns=["def my_pattern(): ..."],
            tech_stack={"python": "3.13"},
        )
        result = composer.compose(intent, context, [])
        assert "## Codebase Context" in result.enhanced_text

    def test_no_context_section_when_empty(self, composer: PromptComposer) -> None:
        """Should skip context section when context is empty."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        assert "## Codebase Context" not in result.enhanced_text

    def test_role_section_uses_language(self, composer: PromptComposer) -> None:
        """Should include language in role section."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
        )
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        assert "python developer" in result.enhanced_text.lower()

    def test_role_section_uses_frameworks(self, composer: PromptComposer) -> None:
        """Should include frameworks in role section."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            frameworks=["fastapi", "react"],
        )
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        assert "fastapi" in result.enhanced_text.lower()
        assert "react" in result.enhanced_text.lower()


class TestTaskConstraints:
    """Tests for _get_task_constraints method."""

    def test_base_constraints_always_included(self, composer: PromptComposer) -> None:
        """Should include 2 base constraints for any task type."""
        intent = Intent(original_prompt="test", task_type=TaskType.UNKNOWN)
        constraints = composer._get_task_constraints(intent)
        assert len(constraints) >= 2
        assert any("existing patterns" in c.lower() for c in constraints)
        assert any("dependencies" in c.lower() for c in constraints)

    def test_refactor_constraints(self, composer: PromptComposer) -> None:
        """Should add 3 additional constraints for REFACTOR."""
        intent = Intent(original_prompt="test", task_type=TaskType.REFACTOR)
        constraints = composer._get_task_constraints(intent)
        assert any("preserve" in c.lower() for c in constraints)
        assert any("backward compatibility" in c.lower() for c in constraints)
        assert any("api signatures" in c.lower() for c in constraints)

    def test_debug_constraints(self, composer: PromptComposer) -> None:
        """Should add 2 additional constraints for DEBUG."""
        intent = Intent(original_prompt="test", task_type=TaskType.DEBUG)
        constraints = composer._get_task_constraints(intent)
        assert any("root cause" in c.lower() for c in constraints)
        assert any(
            "minimize changes" in c.lower() or "unrelated code" in c.lower() for c in constraints
        )

    def test_generation_constraints(self, composer: PromptComposer) -> None:
        """Should add 2 additional constraints for GENERATION."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        constraints = composer._get_task_constraints(intent)
        assert any("single responsibility" in c.lower() for c in constraints)
        assert any("easy to test" in c.lower() for c in constraints)

    def test_security_constraints(self, composer: PromptComposer) -> None:
        """Should add 3 security constraints when touches_security=True."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            touches_security=True,
        )
        constraints = composer._get_task_constraints(intent)
        assert any("credentials" in c.lower() or "api keys" in c.lower() for c in constraints)
        assert any("parameterized queries" in c.lower() for c in constraints)
        assert any("sanitize" in c.lower() for c in constraints)


class TestPlanningComposition:
    """Test PLANNING-specific prompt composition."""

    @pytest.fixture
    def composer(self, standards: QualityStandards) -> PromptComposer:
        return PromptComposer(standards)

    @pytest.fixture
    def planning_intent(self) -> Intent:
        return Intent(
            original_prompt="Create a plan to implement caching",
            task_type=TaskType.PLANNING,
            primary_language="python",
            frameworks=["fastapi"],
        )

    @pytest.fixture
    def context(self) -> ContextBundle:
        return ContextBundle()

    def test_planning_includes_research_phase(
        self, composer: PromptComposer, planning_intent: Intent, context: ContextBundle
    ) -> None:
        """PLANNING prompt includes research phase requirements."""
        result = composer.compose(planning_intent, context, [])
        assert "Research" in result.enhanced_text
        assert "BEFORE" in result.enhanced_text
        assert "Read" in result.enhanced_text

    def test_planning_includes_step_format(
        self, composer: PromptComposer, planning_intent: Intent, context: ContextBundle
    ) -> None:
        """PLANNING prompt includes step format template."""
        result = composer.compose(planning_intent, context, [])
        assert "**File:**" in result.enhanced_text
        assert "**Grounding:**" in result.enhanced_text
        assert "**Verify:**" in result.enhanced_text

    def test_planning_includes_anti_slop(
        self, composer: PromptComposer, planning_intent: Intent, context: ContextBundle
    ) -> None:
        """PLANNING prompt includes anti-slop rules."""
        result = composer.compose(planning_intent, context, [])
        assert "FORBIDDEN" in result.enhanced_text
        assert "should" in result.enhanced_text.lower()
        assert "probably" in result.enhanced_text.lower()

    def test_planning_mentions_less_capable(
        self, composer: PromptComposer, planning_intent: Intent, context: ContextBundle
    ) -> None:
        """PLANNING prompt explains target is less capable model."""
        result = composer.compose(planning_intent, context, [])
        assert (
            "less capable" in result.enhanced_text.lower() or "LESS CAPABLE" in result.enhanced_text
        )

    def test_planning_has_different_verification_steps(
        self, composer: PromptComposer, planning_intent: Intent, context: ContextBundle
    ) -> None:
        """PLANNING verification steps are plan-focused, not code-focused."""
        result = composer.compose(planning_intent, context, [])
        # Planning verification is about plan quality, not code quality
        assert any("Grounding" in step for step in result.verification_steps)
        assert any("line number" in step.lower() for step in result.verification_steps)

    def test_planning_constraints_include_research(
        self, composer: PromptComposer, planning_intent: Intent
    ) -> None:
        """PLANNING constraints include research requirements."""
        constraints = composer._get_task_constraints(planning_intent)
        assert any("research" in c.lower() for c in constraints)
        assert any("atomic" in c.lower() for c in constraints)
