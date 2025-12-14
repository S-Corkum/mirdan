"""Prompt Composer - Assembles enhanced prompts using proven frameworks."""

from mirdan.config import EnhancementConfig
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import (
    ContextBundle,
    EnhancedPrompt,
    Intent,
    TaskType,
    ToolRecommendation,
)


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
        """Build the final prompt text."""
        # Determine verbosity settings
        verbosity = "balanced"
        include_verification = True
        include_tool_hints = True

        if self._config:
            verbosity = self._config.verbosity
            include_verification = self._config.include_verification
            include_tool_hints = self._config.include_tool_hints

        # Determine role
        language = intent.primary_language or "software"
        frameworks = ", ".join(intent.frameworks) if intent.frameworks else "modern frameworks"

        sections: list[str] = []

        # Role section (always included)
        sections.append(f"""## Role
Act as a senior {language} developer with expertise in {frameworks}.
You prioritize code quality, security, and maintainability.""")

        # Context section (if we have context)
        if context.existing_patterns or context.tech_stack:
            tech_stack_str = ", ".join(f"{k}: {v}" for k, v in context.tech_stack.items())
            sections.append(f"""## Codebase Context
{context.summarize_patterns()}

Tech Stack: {tech_stack_str if context.tech_stack else "Not detected"}""")

        # Task section (always included)
        sections.append(f"""## Task
{intent.original_prompt}""")

        # Quality requirements (skip if minimal verbosity)
        if quality_requirements and verbosity != "minimal":
            # Comprehensive: show all, balanced: show first 5, minimal: skip
            limit = None if verbosity == "comprehensive" else 5
            reqs_to_show = quality_requirements[:limit] if limit else quality_requirements
            requirements_text = "\n".join(f"- {req}" for req in reqs_to_show)
            sections.append(f"""## Quality Requirements
{requirements_text}""")

        # Constraints based on task type (skip if minimal verbosity)
        constraints = self._get_task_constraints(intent)
        if constraints and verbosity != "minimal":
            limit = None if verbosity == "comprehensive" else 4
            constraints_to_show = constraints[:limit] if limit else constraints
            constraints_text = "\n".join(f"- {c}" for c in constraints_to_show)
            sections.append(f"""## Constraints
{constraints_text}""")

        # Verification steps (conditional on include_verification)
        if verification_steps and include_verification:
            verification_text = "\n".join(
                f"{i + 1}. {step}" for i, step in enumerate(verification_steps)
            )
            sections.append(f"""## Before Completing
{verification_text}""")

        # Tool recommendations (conditional on include_tool_hints)
        if tool_recommendations and include_tool_hints:
            tools_text = "\n".join(
                f"- **{rec.mcp}**: {rec.action} ({rec.reason})" for rec in tool_recommendations
            )
            sections.append(f"""## Recommended Tools to Use First
{tools_text}""")

        return "\n\n".join(sections)

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

        if intent.touches_security:
            constraints.extend(
                [
                    "Never hardcode credentials or API keys",
                    "Use parameterized queries for all database operations",
                    "Validate and sanitize all user inputs",
                ]
            )

        return constraints
