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
        # Dispatch to planning-specific method for PLANNING tasks
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
        """Build specialized prompt text for PLANNING tasks.

        This produces a prompt designed to generate plans that can be
        implemented by less capable models (Haiku, Flash).
        """
        sections: list[str] = []

        # Role section - Planning specialist
        language = intent.primary_language or "software"
        frameworks = ", ".join(intent.frameworks) if intent.frameworks else "modern frameworks"

        sections.append(f"""## Role
You are a senior software architect creating an implementation plan.
Your plans will be executed by a LESS CAPABLE AI model (like Claude Haiku or Gemini Flash).
Therefore, your plan MUST be:
- EXPLICIT: Assume NO implicit knowledge - the implementing model knows nothing
- GROUNDED: Every fact must be verified with tools before including
- ATOMIC: Each step is a single tool action
- COMPLETE: No gaps - include imports, exports, tests, configs
- LITERAL: Written so it can be executed EXACTLY as written

You are an expert in {language} with experience in {frameworks}.""")

        # Mandatory Research Phase
        sections.append("""## MANDATORY: Pre-Planning Research Phase

Before writing ANY plan steps, you MUST complete ALL of the following:

1. **Glob**: Verify the actual project structure (don't assume directories exist)
2. **Read**: Read ALL files that will be modified (not mentioned - actually READ them)
3. **Read**: Read pyproject.toml/package.json for dependencies
4. **Read**: Read existing similar implementations to understand patterns
5. **context7**: Query for EVERY external library API you will reference
6. **enyal_recall**: Get project conventions and past decisions

**CRITICAL**: You CANNOT write plan steps for files you haven't Read.
**CRITICAL**: You CANNOT reference APIs you haven't verified with context7.
**CRITICAL**: If you skip research, the plan WILL fail when implemented.""")

        # Research Notes Template
        sections.append("""## MANDATORY: Document Your Research

After completing research, BEFORE any plan steps, write a Research Notes section:

```markdown
## Research Notes (Pre-Plan Verification)

### Files Verified
- `path/to/file.py`: line 45 contains function X, line 78 contains class Y

### Project Structure
- Discovered via Glob: [actual directory tree relevant to task]

### Dependencies Confirmed
- `library-name`: version X.Y.Z (from pyproject.toml line N)

### API Documentation (context7)
- `library.method()`: takes args (a: str, b: int) -> Result

### Conventions (enyal)
- naming: uses snake_case for functions
- testing: pytest with fixtures in conftest.py
```""")

        # Step Format Template
        sections.append("""## Step Format (REQUIRED for every step)

```markdown
### Step N: [Brief descriptive title]

**File:** `exact/path/verified/via/Read.py`
  OR `NEW: path/to/new/file.py` (parent dir verified via Glob)

**Action:** Edit (or Read, Write, Bash, etc.)

**Details:**
- Line 45: Add import statement `from x import y`
- Line 78: Add function with exact signature
- [Be SPECIFIC - line numbers, function names, exact code]

**Depends On:** Steps X, Y (must complete first)

**Verify:** [How to confirm success]
- Read file after edit, confirm function exists at line 78
- Run test, confirm it passes

**Grounding:** [Which tool verified this step's facts]
- File exists: Read at research phase
- API signature: context7 query for library.method
- Convention: enyal_recall for naming pattern
```""")

        # Anti-Slop Rules
        sections.append("""## Anti-Slop Rules (VIOLATIONS WILL CAUSE IMPLEMENTATION FAILURE)

### FORBIDDEN language - the implementing model cannot interpret these:
- "should" -> Use definitive: "WILL", "DOES", "IS"
- "probably" -> Verify first, then state as fact
- "around line X" -> Read file, give EXACT line number
- "somewhere in" -> Glob/Grep, give exact path
- "I think" -> Verify with tools, then state as fact
- "similar to" -> Be specific, don't reference vague similarities
- "standard practice" -> Verify it's THIS project's practice

### FORBIDDEN assumptions - the implementing model WILL fail if you assume:
- That a file exists -> Read or Glob to verify FIRST
- That a function is at line X -> Read the file FIRST
- That an API works this way -> context7 FIRST
- That an import is available -> Read dependency file FIRST
- That the project follows pattern X -> Read existing code FIRST

### FORBIDDEN step structures:
BAD: "Update the file to add the function and fix the imports"
GOOD: Step 1: Add import at line 1, Step 2: Add function at line 45

### REQUIRED inclusions (the implementing model will NOT add these):
- Import statements for any new dependencies used
- Export statements if adding public API
- Type hints if the project uses them (check existing code)
- Test files if adding functionality
- Documentation updates if the project has docs""")

        # Tool Recommendations
        if tool_recommendations:
            tools_text = "\n".join(
                f"- **{rec.mcp}** ({rec.priority}): {rec.action}\n  Reason: {rec.reason}"
                for rec in tool_recommendations
            )
            sections.append(f"""## Use These Tools BEFORE Planning

{tools_text}

These are NOT optional. Complete ALL of these BEFORE writing any plan steps.""")

        # Task
        sections.append(f"""## Task

{intent.original_prompt}

Remember: Your plan will be executed by a less capable model.
Be explicit, be complete, be grounded in verified facts.""")

        # Quality Gate
        sections.append("""## Before Submitting Your Plan

Self-verify against these criteria:
- [ ] Every file path was verified with Read or Glob
- [ ] Every line number is accurate (not approximated)
- [ ] Every API reference was verified with context7
- [ ] Every step has a Grounding field citing verification source
- [ ] No steps use vague language (should, probably, around, etc.)
- [ ] No steps combine multiple actions
- [ ] Dependencies between steps are explicit
- [ ] All imports, exports, tests, types are included

If ANY check fails, fix BEFORE presenting the plan.""")

        return "\n\n".join(sections)
