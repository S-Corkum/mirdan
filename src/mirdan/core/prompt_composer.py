"""Prompt Composer - Assembles enhanced prompts using proven frameworks."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from mirdan.config import EnhancementConfig
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import (
    ContextBundle,
    EnhancedPrompt,
    Intent,
    SessionContext,
    TaskType,
    ToolRecommendation,
)

# Path to templates directory
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


class PromptComposer:
    """Composes enhanced prompts using proven frameworks."""

    # Task-type-specific guidance injected into prompts to reduce agent runtime
    # and improve first-attempt correctness (per "Beyond the Prompt" MSR 2026).
    TASK_GUIDANCE: dict[TaskType, str] = {
        TaskType.GENERATION: (
            "## Implementation Approach\n"
            "Focus on integrating with existing patterns. "
            "Before writing new code, read similar implementations in the codebase. "
            "New functions must have clear inputs/outputs and be easy to test independently.\n"
            "If decision_guidance is provided, review the trade-offs before choosing an approach."
        ),
        TaskType.REFACTOR: (
            "## Refactoring Protocol\n"
            "1. Identify all callers of the code being refactored\n"
            "2. Ensure all existing tests pass BEFORE making changes\n"
            "3. Make incremental changes — each commit should be independently correct\n"
            "4. After refactoring, verify no public API signatures changed\n"
            "If decision_guidance is provided, review the trade-offs before choosing an approach."
        ),
        TaskType.DEBUG: (
            "## Debugging Protocol\n"
            "1. Reproduce the issue first — confirm you can trigger the bug\n"
            "2. Trace the data flow to find WHERE the bug manifests\n"
            "3. Identify the ROOT CAUSE, not just the symptom\n"
            "4. Write a test that fails BEFORE your fix and passes AFTER\n"
            "5. Make the minimal change that fixes the root cause"
        ),
        TaskType.REVIEW: (
            "## Review Focus Areas\n"
            "1. Security: Check for injection, auth bypass, data exposure\n"
            "2. Correctness: Verify edge cases, error handling, race conditions\n"
            "3. Consistency: Does the code follow existing project conventions?\n"
            "4. Testability: Are changes covered by tests?"
        ),
        TaskType.TEST: (
            "## Testing Strategy\n"
            "1. Cover the happy path AND edge cases (empty, null, boundary values)\n"
            "2. Each test should verify ONE behavior — name it descriptively\n"
            "3. Tests must be isolated — no shared mutable state between tests\n"
            "4. Mock external dependencies, don't mock the code under test\n"
            "5. Assert specific values, not just that no error occurred\n"
            "6. Every test must have at least one meaningful assertion (TEST003)\n"
            "7. Never write empty test bodies or assert True placeholders (TEST001/TEST002)\n"
            "8. Avoid excessive mocking — if everything is mocked, nothing is tested (TEST005)\n"
            "9. Use pytest.raises(SpecificError), not pytest.raises(Exception) (TEST010)\n"
            "10. Include edge case tests: empty inputs, None, boundaries, error paths (TEST007)"
        ),
        TaskType.DOCUMENTATION: (
            "## Documentation Standards\n"
            "1. Document the WHY, not just the WHAT — readers can see the code\n"
            "2. Include concrete examples for non-obvious functionality\n"
            "3. Keep docs next to the code they describe\n"
            "4. Mark parameters, return values, and exceptions explicitly"
        ),
    }

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
        extra_requirements: Sequence[str] = (),
        session: SessionContext | None = None,
        tidy_suggestions: list[dict[str, Any]] | None = None,
    ) -> EnhancedPrompt:
        """Compose the final enhanced prompt."""
        # Prepend persistent violation feedback before static standards so they
        # survive the verbosity cap in _build_prompt_text (quality_requirements[:5]).
        quality_requirements = list(extra_requirements) + self.standards.render_for_intent(intent)

        # Generate verification steps based on task type, pruning when safe
        verification_steps = self.generate_verification_steps(intent, session=session)

        # Build the enhanced prompt text
        enhanced_text = self._build_prompt_text(
            intent,
            context,
            quality_requirements,
            verification_steps,
            tool_recommendations,
            tidy_suggestions=tidy_suggestions,
        )

        return EnhancedPrompt(
            enhanced_text=enhanced_text,
            intent=intent,
            tool_recommendations=tool_recommendations,
            quality_requirements=quality_requirements,
            verification_steps=verification_steps,
        )

    def generate_verification_steps(
        self,
        intent: Intent,
        session: SessionContext | None = None,
    ) -> list[str]:
        """Generate verification steps based on task type."""
        base_steps = [
            "Verify all imports reference actual modules in the project",
            "Ensure error handling covers all async operations",
            "Check that no secrets or credentials are hardcoded",
            "Confirm the code follows existing naming conventions",
        ]

        # Compress base checklist when prior validation passed — avoids re-asserting
        # already-verified constraints and reduces context waste on iteration.
        if session and session.validation_count > 0 and session.unresolved_errors == 0:
            base_steps = [
                "Previous validation passed — confirm changes don't introduce regressions"
            ]

        # Use task_types list for compound task detection (e.g., "add tests for the new feature"
        # detects both TEST and GENERATION, unioning their verification steps).
        task_type_set = set(intent.task_types) if intent.task_types else {intent.task_type}

        if TaskType.GENERATION in task_type_set:
            base_steps.append("Validate that new code integrates with existing patterns")

        if TaskType.REFACTOR in task_type_set:
            base_steps.insert(0, "Verify all existing functionality is preserved")
            base_steps.append("Ensure no public API signatures changed without approval")

        if TaskType.DEBUG in task_type_set:
            base_steps.insert(0, "Confirm the fix addresses the root cause, not just symptoms")
            base_steps.append("Add tests to prevent regression")

        if TaskType.TEST in task_type_set:
            base_steps.extend(
                [
                    "Ensure tests cover both happy path and edge cases",
                    "Verify test isolation - no shared state between tests",
                    "Verify every test has at least one meaningful assertion",
                    "Verify no empty test bodies or assert True placeholders",
                    "Verify test_file parameter is used for implementation cross-referencing",
                ]
            )

        if TaskType.PLANNING in task_type_set:
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
        if intent.touches_knowledge_graph:
            base_steps.extend(
                [
                    "Verify all graph queries are parameterized (no string interpolation)",
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
        tidy_suggestions: list[dict[str, Any]] | None = None,
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

        # Task-type-specific guidance (suppressed in minimal verbosity)
        task_guidance = ""
        if verbosity != "minimal":
            task_guidance = self.TASK_GUIDANCE.get(intent.task_type, "")

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
            task_guidance=task_guidance,
            tidy_suggestions=tidy_suggestions if verbosity != "minimal" else None,
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
