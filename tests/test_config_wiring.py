"""Tests for configuration wiring to components."""

from pathlib import Path

from mirdan.config import (
    EnhancementConfig,
    MirdanConfig,
    OrchestrationConfig,
    ProjectConfig,
    QualityConfig,
    ThresholdsConfig,
)
from mirdan.core.intent_analyzer import IntentAnalyzer
from mirdan.core.orchestrator import MCPOrchestrator
from mirdan.core.prompt_composer import PromptComposer
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import ContextBundle, Intent, TaskType


class TestVersionConsistency:
    """Tests that version metadata is consistent across the project."""

    def test_version_matches_pyproject(self) -> None:
        """mirdan.__version__ must match the version in pyproject.toml."""
        import tomllib

        import mirdan

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            pyproject = tomllib.load(f)

        pyproject_version = pyproject["project"]["version"]
        assert mirdan.__version__ == pyproject_version, (
            f"mirdan.__version__ ({mirdan.__version__!r}) does not match "
            f"pyproject.toml version ({pyproject_version!r}). "
            f"Update src/mirdan/__init__.py to match."
        )


class TestIntentAnalyzerConfigWiring:
    """Tests for IntentAnalyzer configuration wiring."""

    def test_uses_project_language_when_detection_empty(self) -> None:
        """Should use project config language when detection returns None."""
        config = ProjectConfig(primary_language="python")
        analyzer = IntentAnalyzer(config)

        # Prompt with no language indicators
        intent = analyzer.analyze("add a button to the page")

        assert intent.primary_language == "python"

    def test_uses_project_frameworks_when_detection_empty(self) -> None:
        """Should use project config frameworks when detection returns empty."""
        config = ProjectConfig(frameworks=["fastapi", "pydantic"])
        analyzer = IntentAnalyzer(config)

        # Prompt with no framework indicators
        intent = analyzer.analyze("add validation to the endpoint")

        assert "fastapi" in intent.frameworks
        assert "pydantic" in intent.frameworks

    def test_detection_overrides_project_config(self) -> None:
        """Detected language/frameworks should override project config."""
        config = ProjectConfig(primary_language="python", frameworks=["django"])
        analyzer = IntentAnalyzer(config)

        # Prompt with explicit React/TypeScript indicators
        intent = analyzer.analyze("create a React component with TypeScript")

        # Detection should override config
        assert intent.primary_language in ["javascript", "typescript"]
        assert "react" in intent.frameworks

    def test_works_without_config(self) -> None:
        """Should work normally when no config provided (backward compatible)."""
        analyzer = IntentAnalyzer()  # No config

        intent = analyzer.analyze("add a button")

        # Should work, language will be None since no detection or config
        assert intent.primary_language is None


class TestQualityStandardsConfigWiring:
    """Tests for QualityStandards configuration wiring."""

    def test_strict_returns_more_standards(self) -> None:
        """Strict mode should return more standards."""
        strict_config = QualityConfig(security="strict", architecture="strict")
        permissive_config = QualityConfig(security="permissive", architecture="permissive")

        strict_standards = QualityStandards(config=strict_config)
        permissive_standards = QualityStandards(config=permissive_config)

        # Create a security-related intent
        intent = Intent(
            original_prompt="implement authentication",
            task_type=TaskType.GENERATION,
            touches_security=True,
        )

        strict_result = strict_standards.render_for_intent(intent)
        permissive_result = permissive_standards.render_for_intent(intent)

        assert len(strict_result) > len(permissive_result)

    def test_moderate_is_default(self) -> None:
        """No config should behave as moderate (backward compatible)."""
        no_config = QualityStandards()  # No config
        moderate_config = QualityStandards(config=QualityConfig())  # Default config

        intent = Intent(
            original_prompt="add a feature",
            task_type=TaskType.GENERATION,
        )

        no_config_result = no_config.render_for_intent(intent)
        moderate_result = moderate_config.render_for_intent(intent)

        # Results should be similar (both moderate behavior)
        assert len(no_config_result) == len(moderate_result)

    def test_works_without_config(self) -> None:
        """Should work normally when no config provided."""
        standards = QualityStandards()

        intent = Intent(
            original_prompt="fix the bug",
            task_type=TaskType.DEBUG,
            primary_language="python",
        )

        result = standards.render_for_intent(intent)

        assert len(result) > 0


class TestQualityStandardsFrameworkConfig:
    """Tests for framework stringency configuration wiring."""

    def test_framework_stringency_affects_count(self) -> None:
        """Framework stringency should control number of framework standards."""
        strict_config = QualityConfig(framework="strict")
        permissive_config = QualityConfig(framework="permissive")

        strict_standards = QualityStandards(config=strict_config)
        permissive_standards = QualityStandards(config=permissive_config)

        intent = Intent(
            original_prompt="create a React component",
            task_type=TaskType.GENERATION,
            frameworks=["react"],
        )

        strict_result = strict_standards.render_for_intent(intent)
        permissive_result = permissive_standards.render_for_intent(intent)

        assert len(strict_result) > len(permissive_result)

    def test_framework_default_is_moderate(self) -> None:
        """Default framework stringency should be moderate."""
        config = QualityConfig()

        assert config.framework == "moderate"


class TestKnowledgeGraphStandardsTrigger:
    """Tests for knowledge graph standards triggered by touches_knowledge_graph."""

    def test_kg_flag_triggers_kg_standards(self) -> None:
        """touches_knowledge_graph=True should include KG standards."""
        standards = QualityStandards()
        intent = Intent(
            original_prompt="build a knowledge graph",
            task_type=TaskType.GENERATION,
            touches_knowledge_graph=True,
        )
        result = standards.render_for_intent(intent)
        result_text = " ".join(result).lower()
        assert "graph" in result_text or "entity" in result_text or "provenance" in result_text

    def test_kg_flag_false_skips_kg_standards(self) -> None:
        """touches_knowledge_graph=False should not include KG standards."""
        standards = QualityStandards()
        intent_with_kg = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            touches_knowledge_graph=True,
        )
        intent_without_kg = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            touches_knowledge_graph=False,
        )
        with_kg = standards.render_for_intent(intent_with_kg)
        without_kg = standards.render_for_intent(intent_without_kg)
        assert len(with_kg) > len(without_kg)

    def test_kg_independent_of_neo4j_framework(self) -> None:
        """KG standards should trigger on flag, not framework name."""
        standards = QualityStandards()
        # KG flag set but no neo4j framework
        intent = Intent(
            original_prompt="build a weaviate knowledge graph",
            task_type=TaskType.GENERATION,
            frameworks=["weaviate"],
            touches_knowledge_graph=True,
        )
        result = standards.render_for_intent(intent)
        result_text = " ".join(result).lower()
        assert "graph" in result_text or "entity" in result_text or "provenance" in result_text


class TestPromptComposerConfigWiring:
    """Tests for PromptComposer configuration wiring."""

    def test_include_verification_false_skips_section(self) -> None:
        """Should skip verification section when include_verification is False."""
        config = EnhancementConfig(include_verification=False)
        standards = QualityStandards()
        composer = PromptComposer(standards, config=config)

        intent = Intent(
            original_prompt="add a feature",
            task_type=TaskType.GENERATION,
        )
        context = ContextBundle()

        enhanced = composer.compose(intent, context, [])

        assert "Before Completing" not in enhanced.enhanced_text

    def test_include_tool_hints_false_skips_section(self) -> None:
        """Should skip tool hints section when include_tool_hints is False."""
        from mirdan.models import ToolRecommendation

        config = EnhancementConfig(include_tool_hints=False)
        standards = QualityStandards()
        composer = PromptComposer(standards, config=config)

        intent = Intent(
            original_prompt="add a feature",
            task_type=TaskType.GENERATION,
        )
        context = ContextBundle()
        recommendations = [ToolRecommendation(mcp="context7", action="fetch docs", reason="test")]

        enhanced = composer.compose(intent, context, recommendations)

        assert "Recommended Tools" not in enhanced.enhanced_text

    def test_minimal_verbosity_reduces_output(self) -> None:
        """Minimal verbosity should produce shorter output."""
        minimal_config = EnhancementConfig(verbosity="minimal")
        comprehensive_config = EnhancementConfig(verbosity="comprehensive")

        standards = QualityStandards()
        minimal_composer = PromptComposer(standards, config=minimal_config)
        comprehensive_composer = PromptComposer(standards, config=comprehensive_config)

        intent = Intent(
            original_prompt="implement user authentication with proper validation",
            task_type=TaskType.GENERATION,
            primary_language="python",
            touches_security=True,
        )
        context = ContextBundle()

        minimal_enhanced = minimal_composer.compose(intent, context, [])
        comprehensive_enhanced = comprehensive_composer.compose(intent, context, [])

        # Minimal should be shorter
        assert len(minimal_enhanced.enhanced_text) < len(comprehensive_enhanced.enhanced_text)

    def test_balanced_is_default(self) -> None:
        """No config should behave as balanced (backward compatible)."""
        standards = QualityStandards()
        no_config_composer = PromptComposer(standards)
        balanced_composer = PromptComposer(standards, config=EnhancementConfig())

        intent = Intent(
            original_prompt="add feature",
            task_type=TaskType.GENERATION,
        )
        context = ContextBundle()

        no_config_result = no_config_composer.compose(intent, context, [])
        balanced_result = balanced_composer.compose(intent, context, [])

        # Should produce similar output
        assert len(no_config_result.enhanced_text) == len(balanced_result.enhanced_text)


class TestMCPOrchestratorConfigWiring:
    """Tests for MCPOrchestrator configuration wiring."""

    def test_prefer_mcps_reorders_recommendations(self) -> None:
        """Should reorder recommendations based on prefer_mcps."""
        # Prefer filesystem first, then enyal
        config = OrchestrationConfig(prefer_mcps=["filesystem", "enyal"])
        orchestrator = MCPOrchestrator(config)

        intent = Intent(
            original_prompt="create a React component",
            task_type=TaskType.GENERATION,
            frameworks=["react"],
            uses_external_framework=True,
        )

        recommendations = orchestrator.suggest_tools(intent)

        # Find positions of filesystem and enyal recommendations
        mcp_order = [r.mcp for r in recommendations]

        # filesystem should appear before context7 (if both present)
        if "filesystem" in mcp_order and "context7" in mcp_order:
            assert mcp_order.index("filesystem") < mcp_order.index("context7")

    def test_works_without_config(self) -> None:
        """Should work normally when no config provided (backward compatible)."""
        orchestrator = MCPOrchestrator()  # No config

        intent = Intent(
            original_prompt="add authentication",
            task_type=TaskType.GENERATION,
        )

        recommendations = orchestrator.suggest_tools(intent)

        # Should return recommendations
        assert len(recommendations) >= 0  # May be empty depending on intent


class TestServerIntegration:
    """Integration tests for full config wiring."""

    def test_all_components_work_with_default_config(self) -> None:
        """All components should work with default MirdanConfig."""
        config = MirdanConfig()

        # Initialize all components as server.py does
        intent_analyzer = IntentAnalyzer(config.project)
        quality_standards = QualityStandards(config=config.quality)
        prompt_composer = PromptComposer(quality_standards, config=config.enhancement)
        mcp_orchestrator = MCPOrchestrator(config.orchestration)

        # Run through the flow
        prompt = "add user authentication with JWT"
        intent = intent_analyzer.analyze(prompt)
        recommendations = mcp_orchestrator.suggest_tools(intent)
        context = ContextBundle()
        enhanced = prompt_composer.compose(intent, context, recommendations)

        # Should complete without error
        assert enhanced.enhanced_text is not None
        assert len(enhanced.enhanced_text) > len(prompt)

    def test_custom_config_affects_all_components(self) -> None:
        """Custom config values should propagate to all components."""
        config = MirdanConfig()
        config.project = ProjectConfig(primary_language="python", frameworks=["fastapi"])
        config.quality = QualityConfig(security="strict")
        config.enhancement = EnhancementConfig(verbosity="comprehensive", include_verification=True)
        config.orchestration = OrchestrationConfig(prefer_mcps=["enyal", "filesystem"])

        intent_analyzer = IntentAnalyzer(config.project)
        quality_standards = QualityStandards(config=config.quality)
        PromptComposer(quality_standards, config=config.enhancement)
        mcp_orchestrator = MCPOrchestrator(config.orchestration)

        # Analyze a generic prompt - should get python/fastapi from config
        intent = intent_analyzer.analyze("add validation to the endpoint")

        # Project config should provide defaults
        assert intent.primary_language == "python"

        # Get recommendations and verify ordering
        recommendations = mcp_orchestrator.suggest_tools(intent)
        if len(recommendations) >= 2:
            # enyal should be preferred
            mcp_names = [r.mcp for r in recommendations]
            if "enyal" in mcp_names:
                # Enyal should appear early due to preference
                enyal_idx = mcp_names.index("enyal")
                assert enyal_idx <= 1  # Should be first or second


class TestNewConfigFields:
    """Tests for new config fields added in P0/P1 fix."""

    def test_thresholds_arch_defaults(self) -> None:
        """Architecture threshold fields should have correct defaults."""
        t = ThresholdsConfig()
        assert t.arch_max_function_length == 30
        assert t.arch_max_file_length == 300
        assert t.arch_max_nesting_depth == 4
        assert t.arch_max_class_methods == 10

    def test_thresholds_arch_custom_values(self) -> None:
        """Architecture thresholds should accept custom values."""
        t = ThresholdsConfig(
            arch_max_function_length=50,
            arch_max_file_length=500,
            arch_max_nesting_depth=6,
            arch_max_class_methods=15,
        )
        assert t.arch_max_function_length == 50
        assert t.arch_max_file_length == 500
        assert t.arch_max_nesting_depth == 6
        assert t.arch_max_class_methods == 15

    def test_quality_config_language_default(self) -> None:
        """QualityConfig.language should default to moderate."""
        q = QualityConfig()
        assert q.language == "moderate"

    def test_quality_config_language_accepts_valid(self) -> None:
        """QualityConfig.language should accept strict/moderate/permissive."""
        for level in ("strict", "moderate", "permissive"):
            q = QualityConfig(language=level)
            assert q.language == level

    def test_project_config_github_defaults(self) -> None:
        """ProjectConfig github fields should default to empty."""
        p = ProjectConfig()
        assert p.github_owner == ""
        assert p.github_repo == ""

    def test_project_config_github_custom(self) -> None:
        """ProjectConfig github fields should accept values."""
        p = ProjectConfig(github_owner="myorg", github_repo="myrepo")
        assert p.github_owner == "myorg"
        assert p.github_repo == "myrepo"

    def test_backward_compatible_mirdan_config(self) -> None:
        """MirdanConfig with no new fields should still work."""
        config = MirdanConfig()
        assert config.thresholds.arch_max_function_length == 30
        assert config.quality.language == "moderate"
        assert config.project.github_owner == ""
