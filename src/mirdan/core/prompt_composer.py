"""Prompt Composer - Assembles enhanced prompts using proven frameworks."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from mirdan.config import EnhancementConfig
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import (
    ContextBundle,
    EnhancedPrompt,
    Intent,
    TaskType,
    ToolRecommendation,
)

# Path to templates directory
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


class PromptComposer:
    """Composes enhanced prompts using proven frameworks."""

    def __init__(
        self,
        standards: QualityStandards,
        config: EnhancementConfig | None = None,
    ):
        """Initialize with quality standards and optional enhancement config.

        Args:
            standards: Quality standards repository
            config: Enhancement config for verbosity and section control
        """
        self.standards = standards
        self._config = config
        self._env = Environment(
            loader=FileSystemLoader(TEMPLATES_DIR),
            autoescape=select_autoescape(default=False),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def compose(
        self,
        intent: Intent,
        context: ContextBundle,
        tool_recommendations: list[ToolRecommendation],
    ) -> EnhancedPrompt:
        """Compose the final enhanced prompt."""
        # Get quality requirements
        quality_requirements = self.standards.render_for_intent(intent)

        # Generate verification steps based on task type
        verification_steps = self._generate_verification_steps(intent)

        # Build the enhanced prompt text
        enhanced_text = self._build_prompt_text(
            intent, context, quality_requirements, verification_steps, tool_recommendations
        )

        return EnhancedPrompt(
            enhanced_text=enhanced_text,
            intent=intent,
            tool_recommendations=tool_recommendations,
            quality_requirements=quality_requirements,
            verification_steps=verification_steps,
        )

    def _generate_verification_steps(self, intent: Intent) -> list[str]:
        """Generate verification steps based on task type."""
        base_steps = [
            "Verify all imports reference actual modules in the project",
            "Ensure error handling covers all async operations",
            "Check that no secrets or credentials are hardcoded",
            "Confirm the code follows existing naming conventions",
        ]

        if intent.task_type == TaskType.GENERATION:
            base_steps.append("Validate that new code integrates with existing patterns")

        if intent.task_type == TaskType.REFACTOR:
            base_steps.insert(0, "Verify all existing functionality is preserved")
            base_steps.append("Ensure no public API signatures changed without approval")

        if intent.task_type == TaskType.DEBUG:
            base_steps.insert(0, "Confirm the fix addresses the root cause, not just symptoms")
            base_steps.append("Add tests to prevent regression")

        if intent.task_type == TaskType.TEST:
            base_steps.extend(
                [
                    "Ensure tests cover both happy path and edge cases",
                    "Verify test isolation - no shared state between tests",
                ]
            )

        if intent.task_type == TaskType.PLANNING:
            # Planning has different verification - focused on plan quality
            return [
                "Verify every file path was confirmed with Read or Glob",
                "Verify every line number is exact (not approximated)",
                "Verify every API reference was confirmed with context7",
                "Verify every step has a Grounding field citing verification",
                "Verify no steps use vague language (should, probably, around)",
                "Verify no steps combine multiple actions",
                "Verify dependencies between steps are explicit",
                "Verify all imports, exports, tests, types are included",
            ]

        if intent.touches_rag:
            base_steps.extend(
                [
                    "Verify embedding model is consistent between indexing and querying",
                    "Verify chunk overlap is non-zero (10-20% of chunk_size)",
                    "Verify metadata is stored with vectors (source, page, model_version)",
                    "Verify similarity threshold is configured (not just top-k)",
                    "Verify error handling for embedding generation failures",
                    "Verify vector DB connection has timeout and retry logic",
                    "Verify retrieved context is validated before LLM prompt injection",
                ]
            )
            # Additional KG-specific checks
            if "neo4j" in intent.frameworks:
                base_steps.extend(
                    [
                        "Verify all Cypher queries are parameterized (no string interpolation)",
                        "Verify graph traversals have depth/node limits",
                        "Verify entity deduplication is implemented before insertion",
                    ]
                )

        if intent.touches_security:
            base_steps.extend(
                [
                    "Verify password handling uses proper hashing",
                    "Check that sensitive data is never logged",
                    "Validate input sanitization is in place",
                ]
            )

        return base_steps

    def _build_prompt_text(
        self,
        intent: Intent,
        context: ContextBundle,
        quality_requirements: list[str],
        verification_steps: list[str],
        tool_recommendations: list[ToolRecommendation],
    ) -> str:
        """Build the final prompt text using Jinja2 templates."""
        # Dispatch to planning template for PLANNING tasks
        if intent.task_type == TaskType.PLANNING:
            return self._build_planning_prompt_text(
                intent, context, quality_requirements, verification_steps, tool_recommendations
            )

        # Determine verbosity settings
        verbosity = "balanced"
        include_verification = True
        include_tool_hints = True

        if self._config:
            verbosity = self._config.verbosity
            include_verification = self._config.include_verification
            include_tool_hints = self._config.include_tool_hints

        # Prepare template context
        language = intent.primary_language or "software"
        frameworks = ", ".join(intent.frameworks) if intent.frameworks else "modern frameworks"

        # Apply verbosity limits to requirements and constraints
        constraints = self._get_task_constraints(intent)
        if verbosity == "balanced":
            quality_requirements = quality_requirements[:5]
            constraints = constraints[:4]

        # Build tech stack string
        tech_stack_str = ", ".join(f"{k}: {v}" for k, v in context.tech_stack.items())

        # Render the generation template
        template = self._env.get_template("generation.j2")
        return template.render(
            language=language,
            frameworks=frameworks,
            patterns_summary=context.summarize_patterns() if context.existing_patterns else None,
            tech_stack=tech_stack_str if context.tech_stack else None,
            original_prompt=intent.original_prompt,
            quality_requirements=quality_requirements if verbosity != "minimal" else [],
            constraints=constraints if verbosity != "minimal" else [],
            verification_steps=verification_steps if include_verification else [],
            tool_recommendations=tool_recommendations if include_tool_hints else [],
            verbosity=verbosity,
            include_verification=include_verification,
            include_tool_hints=include_tool_hints,
        ).strip()

    def _get_task_constraints(self, intent: Intent) -> list[str]:
        """Get constraints specific to the task type."""
        constraints = [
            "Follow existing patterns found in the codebase",
            "Do not introduce new dependencies without explicit approval",
        ]

        if intent.task_type == TaskType.REFACTOR:
            constraints.extend(
                [
                    "Preserve all existing functionality",
                    "Maintain backward compatibility",
                    "Do not change public API signatures without approval",
                ]
            )

        if intent.task_type == TaskType.DEBUG:
            constraints.extend(
                [
                    "Focus on the root cause, not just symptoms",
                    "Minimize changes to unrelated code",
                ]
            )

        if intent.task_type == TaskType.GENERATION:
            constraints.extend(
                [
                    "Follow the single responsibility principle",
                    "Write code that is easy to test",
                ]
            )

        if intent.task_type == TaskType.PLANNING:
            constraints.extend(
                [
                    "Complete ALL research BEFORE writing any plan steps",
                    "Every file path must be verified with Read or Glob",
                    "Every line number must be exact after Reading the file",
                    "Every API must be verified with context7 documentation",
                    "Each step must be atomic - one action only",
                    "Each step must have File, Action, Details, Verify, Grounding fields",
                    "Do NOT use vague language: should, probably, around, somewhere",
                    "Include ALL implicit requirements: imports, exports, tests, types",
                ]
            )

        if intent.touches_security:
            constraints.extend(
                [
                    "Never hardcode credentials or API keys",
                    "Use parameterized queries for all database operations",
                    "Validate and sanitize all user inputs",
                ]
            )

        return constraints

    def _build_planning_prompt_text(
        self,
        intent: Intent,
        context: ContextBundle,
        quality_requirements: list[str],
        verification_steps: list[str],
        tool_recommendations: list[ToolRecommendation],
    ) -> str:
        """Build specialized prompt text for PLANNING tasks using Jinja2 template.

        This produces a prompt designed to generate plans that can be
        implemented by less capable models (Haiku, Flash).
        """
        # Prepare template context
        language = intent.primary_language or "software"
        frameworks = ", ".join(intent.frameworks) if intent.frameworks else "modern frameworks"

        # Render the planning template
        template = self._env.get_template("planning.j2")
        return template.render(
            language=language,
            frameworks=frameworks,
            original_prompt=intent.original_prompt,
            tool_recommendations=tool_recommendations,
        ).strip()
