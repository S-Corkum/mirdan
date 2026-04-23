"""Plan Validator - Validates plans for cheap model implementation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mirdan.config import PlanningConfig
from mirdan.models import PlanQualityScore

if TYPE_CHECKING:
    from mirdan.config import ThresholdsConfig


class PlanValidator:
    """Validates implementation plans for execution by less capable models."""

    # Vague language patterns that indicate hallucination risk
    VAGUE_PATTERNS: list[tuple[str, str]] = [
        (r"\bshould\b", "Use definitive language instead of 'should'"),
        (r"\bprobably\b", "Verify with tools first, don't say 'probably'"),
        (r"\blikely\b", "Verify with tools first, don't say 'likely'"),
        (r"\bmaybe\b", "Don't use 'maybe' - be definitive"),
        (r"\baround\s+line\s+\d+", "Use exact line numbers, not 'around line'"),
        (r"\bsomewhere\s+in\b", "Use exact paths, not 'somewhere in'"),
        (r"\bI\s+think\b", "Don't say 'I think' - verify and state facts"),
        (r"\bI\s+believe\b", "Don't say 'I believe' - verify and state facts"),
        (r"\bassume\b", "Don't assume - verify with tools first"),
        (r"\bmight\b", "Don't say 'might' - be definitive"),
        (r"\bpossibly\b", "Don't say 'possibly' - be definitive"),
    ]

    # Required sections for a complete plan
    REQUIRED_SECTIONS: dict[str, str] = {
        "Research Notes": r"##?\s*Research\s+Notes",
        "Files Verified": r"###?\s*Files\s+Verified",
        "Plan Steps": r"###?\s*Step\s+\d+:",
    }

    # Required fields per step
    STEP_REQUIRED_FIELDS: list[tuple[str, str]] = [
        (r"\*\*File:\*\*", "Step missing 'File:' field"),
        (r"\*\*Action:\*\*", "Step missing 'Action:' field"),
        (r"\*\*Details:\*\*", "Step missing 'Details:' field"),
        (r"\*\*Verify:\*\*", "Step missing 'Verify:' field"),
        (r"\*\*Grounding:\*\*", "Step missing 'Grounding:' field"),
    ]

    # Three-layer template required sections (2.1.0 brief-driven pipeline)
    THREE_LAYER_REQUIRED_SECTIONS: dict[str, str] = {
        "Research Notes": r"##?\s*Research\s+Notes",
        "Epic Layer": r"##?\s*Epic\s+Layer",
        "Story Layer": r"##?\s*Story\s+Layer",
    }

    # Required fields per story (three-layer only)
    THREE_LAYER_STORY_REQUIRED_FIELDS: list[tuple[str, str]] = [
        (r"\*\*As\*\*|-\s*\*\*As\*\*", "Story missing 'As' persona field"),
        (r"\*\*I want\*\*|-\s*\*\*I want\*\*", "Story missing 'I want' field"),
        (r"\*\*So that\*\*|-\s*\*\*So that\*\*", "Story missing 'So that' field"),
        (r"\*\*Acceptance Criteria:?\*\*", "Story missing Acceptance Criteria"),
    ]

    # Required fields per subtask (three-layer only); matches STEP_REQUIRED_FIELDS
    # but applies to ##### subtask headings rather than ### Step N
    THREE_LAYER_SUBTASK_REQUIRED_FIELDS: list[tuple[str, str]] = [
        (r"\*\*File:\*\*", "Subtask missing 'File:' field"),
        (r"\*\*Action:\*\*", "Subtask missing 'Action:' field"),
        (r"\*\*Details:\*\*", "Subtask missing 'Details:' field"),
        (r"\*\*Depends on:?\*\*", "Subtask missing 'Depends on:' field"),
        (r"\*\*Verify:\*\*", "Subtask missing 'Verify:' field"),
        (r"\*\*Grounding:\*\*", "Subtask missing 'Grounding:' field"),
    ]

    def __init__(
        self,
        config: PlanningConfig | None = None,
        thresholds: ThresholdsConfig | None = None,
    ):
        """Initialize with optional planning configuration.

        Args:
            config: Planning configuration for strictness levels
            thresholds: Centralized threshold values
        """
        self._config = config or PlanningConfig()
        self._thresholds = thresholds

    def validate(
        self,
        plan_text: str,
        target_model: str = "haiku",
        template_mode: str = "flat",
    ) -> PlanQualityScore:
        """Validate a plan and return quality score.

        Args:
            plan_text: The plan text to validate
            target_model: Model that will implement (affects strictness)
            template_mode: "flat" (legacy, default for backwards compat) or
                "three_layer" (2.1.0 brief-driven pipeline — epic/stories/subtasks)

        Returns:
            PlanQualityScore with scores and issues
        """
        if template_mode == "three_layer":
            return self._validate_three_layer(plan_text, target_model)
        issues: list[str] = []
        recommendations: list[str] = []

        # Check for vague language
        vague_count = 0
        for pattern, message in self.VAGUE_PATTERNS:
            matches = re.findall(pattern, plan_text, re.IGNORECASE)
            if matches:
                vague_count += len(matches)
                issues.append(f"Vague language ({len(matches)}x): {message}")

        # Check for required sections
        missing_sections = 0
        for section_name, pattern in self.REQUIRED_SECTIONS.items():
            if not re.search(pattern, plan_text, re.IGNORECASE):
                missing_sections += 1
                issues.append(f"Missing required section: {section_name}")

        # Extract and validate each step
        steps = self._extract_steps(plan_text)
        step_issues = 0
        for step_num, step_text in steps:
            for pattern, message in self.STEP_REQUIRED_FIELDS:
                if not re.search(pattern, step_text, re.IGNORECASE):
                    step_issues += 1
                    issues.append(f"Step {step_num}: {message}")

        # Check for compound steps (multiple actions in one step)
        compound_patterns = [
            r"\band\s+then\b",
            r"\bthen\s+also\b",
            r"\bfirst.*then\b",
        ]
        for pattern in compound_patterns:
            if re.search(pattern, plan_text, re.IGNORECASE):
                issues.append("Contains compound steps - split into atomic actions")
                step_issues += 1

        # Calculate scores using configured thresholds
        if self._thresholds:
            clarity_penalty = self._thresholds.plan_clarity_penalty
            completeness_penalty = self._thresholds.plan_completeness_penalty
            grounding_penalty = self._thresholds.plan_grounding_penalty
        else:
            clarity_penalty = 0.1
            completeness_penalty = 0.25
            grounding_penalty = 0.1

        clarity_score = max(0.0, 1.0 - (vague_count * clarity_penalty))
        completeness_score = max(0.0, 1.0 - (missing_sections * completeness_penalty))
        grounding_score = max(0.0, 1.0 - (step_issues * grounding_penalty))
        atomicity_score = 1.0 if not any("compound" in i.lower() for i in issues) else 0.7

        # Weighted overall score
        overall = (
            clarity_score * 0.25
            + completeness_score * 0.25
            + grounding_score * 0.30
            + atomicity_score * 0.20
        )

        # Generate recommendations
        if clarity_score < 0.9:
            recommendations.append("Replace vague language with verified facts")
        if completeness_score < 1.0:
            recommendations.append("Add missing required sections")
        if grounding_score < 0.9:
            recommendations.append("Add Grounding fields to all steps")

        # Determine if ready based on target model
        threshold = self._get_threshold(target_model)
        ready = overall >= threshold and len(issues) == 0

        return PlanQualityScore(
            overall_score=round(overall, 3),
            grounding_score=round(grounding_score, 3),
            completeness_score=round(completeness_score, 3),
            atomicity_score=round(atomicity_score, 3),
            clarity_score=round(clarity_score, 3),
            issues=issues,
            recommendations=recommendations,
            ready_for_cheap_model=ready,
        )

    def _extract_steps(self, plan_text: str) -> list[tuple[int, str]]:
        """Extract individual steps from plan text.

        Returns:
            List of (step_number, step_text) tuples
        """
        steps: list[tuple[int, str]] = []
        pattern = r"###?\s*Step\s+(\d+)[:\s]"
        matches = list(re.finditer(pattern, plan_text, re.IGNORECASE))

        for i, match in enumerate(matches):
            step_num = int(match.group(1))
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(plan_text)
            step_text = plan_text[start:end]
            steps.append((step_num, step_text))

        return steps

    def _get_threshold(self, target_model: str) -> float:
        """Get quality threshold based on target model.

        Cheaper models need stricter thresholds.
        """
        thresholds = {
            "haiku": 0.95,  # Maximum strictness
            "flash": 0.90,
            "cheap": 0.95,  # Same as haiku
            "capable": 0.75,  # More lenient for Sonnet-class
        }
        return thresholds.get(target_model, 0.9)

    def _validate_three_layer(
        self,
        plan_text: str,
        target_model: str,
    ) -> PlanQualityScore:
        """Validate a three-layer plan (epic/stories/subtasks).

        Enforces THREE_LAYER_REQUIRED_SECTIONS, per-story INVEST fields,
        and per-subtask grounding fields. Backwards-compat: the flat-template
        path in validate() is untouched.
        """
        issues: list[str] = []
        recommendations: list[str] = []

        # Vague language scan (shared with flat mode)
        vague_count = 0
        for pattern, message in self.VAGUE_PATTERNS:
            matches = re.findall(pattern, plan_text, re.IGNORECASE)
            if matches:
                vague_count += len(matches)
                issues.append(f"Vague language ({len(matches)}x): {message}")

        # Required top-level sections
        missing_sections = 0
        for section_name, pattern in self.THREE_LAYER_REQUIRED_SECTIONS.items():
            if not re.search(pattern, plan_text, re.IGNORECASE):
                missing_sections += 1
                issues.append(f"Missing required section: {section_name}")

        # Per-story INVEST field checks
        stories = self._extract_stories(plan_text)
        story_issues = 0
        for story_id, story_text in stories:
            for pattern, message in self.THREE_LAYER_STORY_REQUIRED_FIELDS:
                if not re.search(pattern, story_text, re.IGNORECASE):
                    story_issues += 1
                    issues.append(f"Story {story_id}: {message}")

        # Per-subtask grounding field checks
        subtasks = self._extract_subtasks(plan_text)
        subtask_issues = 0
        for sub_id, sub_text in subtasks:
            for pattern, message in self.THREE_LAYER_SUBTASK_REQUIRED_FIELDS:
                if not re.search(pattern, sub_text, re.IGNORECASE):
                    subtask_issues += 1
                    issues.append(f"Subtask {sub_id}: {message}")

        # Scoring — same penalty shape as flat but over three-layer issue counts
        if self._thresholds:
            clarity_penalty = self._thresholds.plan_clarity_penalty
            completeness_penalty = self._thresholds.plan_completeness_penalty
            grounding_penalty = self._thresholds.plan_grounding_penalty
        else:
            clarity_penalty = 0.1
            completeness_penalty = 0.25
            grounding_penalty = 0.05  # slightly gentler; more items being scored

        clarity_score = max(0.0, 1.0 - (vague_count * clarity_penalty))
        completeness_score = max(0.0, 1.0 - (missing_sections * completeness_penalty))
        grounding_score = max(
            0.0,
            1.0 - ((story_issues + subtask_issues) * grounding_penalty),
        )
        atomicity_score = 1.0  # three-layer atomicity enforced by subtask structure

        overall = (
            clarity_score * 0.25
            + completeness_score * 0.25
            + grounding_score * 0.30
            + atomicity_score * 0.20
        )

        if clarity_score < 0.9:
            recommendations.append("Replace vague language with verified facts")
        if completeness_score < 1.0:
            recommendations.append(
                "Add missing required sections (Research Notes, Epic Layer, Story Layer)"
            )
        if grounding_score < 0.9:
            recommendations.append(
                "Add INVEST fields to stories and 6 grounding fields to subtasks"
            )

        threshold = self._get_threshold(target_model)
        ready = overall >= threshold and len(issues) == 0

        return PlanQualityScore(
            overall_score=round(overall, 3),
            grounding_score=round(grounding_score, 3),
            completeness_score=round(completeness_score, 3),
            atomicity_score=round(atomicity_score, 3),
            clarity_score=round(clarity_score, 3),
            issues=issues,
            recommendations=recommendations,
            ready_for_cheap_model=ready,
        )

    @staticmethod
    def _extract_stories(plan_text: str) -> list[tuple[str, str]]:
        """Extract three-layer stories: ### Story N — title ... (until next ### Story)."""
        pattern = r"^###\s+Story\s+([\w.\-]+)"
        matches = list(re.finditer(pattern, plan_text, re.MULTILINE | re.IGNORECASE))
        stories: list[tuple[str, str]] = []
        for i, m in enumerate(matches):
            story_id = m.group(1)
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(plan_text)
            stories.append((story_id, plan_text[start:end]))
        return stories

    @staticmethod
    def _extract_subtasks(plan_text: str) -> list[tuple[str, str]]:
        """Extract three-layer subtasks: ##### N.N — title ... (until next #####)."""
        pattern = r"^#####\s+([\w.\-]+)"
        matches = list(re.finditer(pattern, plan_text, re.MULTILINE))
        subtasks: list[tuple[str, str]] = []
        for i, m in enumerate(matches):
            sub_id = m.group(1)
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(plan_text)
            subtasks.append((sub_id, plan_text[start:end]))
        return subtasks
