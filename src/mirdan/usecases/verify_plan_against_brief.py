"""Usecase: verify a three-layer plan covers its brief's ACs and constraints.

Mechanical checks (no LLM needed): grounding-field presence per subtask, INVEST
structure per story, out-of-scope keyword scan, required-section presence.

Semantic checks (local LLM only; graceful degradation): map each brief AC to
story ACs that cover it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mirdan.core.brief_validator import BriefValidator
from mirdan.core.plan_validator import PlanValidator
from mirdan.models import ModelRole

if TYPE_CHECKING:
    from mirdan.llm.manager import LLMManager


# Semantic check configuration.
# Minimum LLM self-reported confidence for an AC mapping to count as "mapped".
# Empirical: Gemma 4 E2B reports conf≈0 on bogus maps even while still
# populating maps_to. Threshold at 0.6 is permissive for BRAIN-tier models
# while rejecting E2B-style noise.
_MIN_MAPPING_CONFIDENCE = 0.6


_AC_MAPPING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "maps_to": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Story AC IDs (like 'Story 1 AC 2') that address this brief AC",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": ["maps_to", "confidence"],
}


class VerifyPlanAgainstBriefUseCase:
    """Verify coverage of a three-layer plan against its brief."""

    def __init__(
        self,
        llm_manager: LLMManager | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._llm = llm_manager
        self._brief_validator = BriefValidator()
        self._plan_validator = PlanValidator()
        # Anchor file-existence checks here. Defaults to CWD, which is the
        # project root during MCP-tool execution.
        self._project_root = project_root or Path.cwd()

    async def execute(self, plan_path: str, brief_path: str) -> dict[str, Any]:
        plan_p = Path(plan_path)
        brief_p = Path(brief_path)

        if not plan_p.exists():
            return self._error(f"plan file not found: {plan_path}")
        if not brief_p.exists():
            return self._error(f"brief file not found: {brief_path}")

        plan_text = plan_p.read_text()
        brief_text = brief_p.read_text()

        brief_sections = self._brief_validator._extract_sections(brief_text)
        brief_acs = self._extract_brief_acs(brief_sections)
        out_of_scope_items = self._extract_out_of_scope(brief_sections)

        plan_score = self._plan_validator.validate(plan_text, template_mode="three_layer")
        missing_grounding = self._missing_grounding_from_issues(plan_score.issues)
        invest_failures = self._invest_failures_from_issues(plan_score.issues)

        scope_violations = self._detect_scope_violations(plan_text, out_of_scope_items)

        # Mechanical upgrades (2.1.0 evidence pass): extraordinary value
        # without any LLM. These catch hallucinated files, broken dependencies,
        # and vague cross-subtask references.
        subtasks = self._plan_validator._extract_subtasks(plan_text)
        phantom_files = self._check_file_existence(subtasks)
        dependency_errors = self._check_dependencies(subtasks)
        vague_cross_references = self._check_vague_cross_references(subtasks)

        # Semantic check: brief AC → story AC mapping (skip if no LLM)
        unmapped_acs: list[str] = []
        semantic_skipped = False
        # Semantic AC mapping requires BRAIN-tier judgment (31B+).
        # Empirical measurement on Gemma 4 E2B-Q3 shows it maps every brief AC
        # to any story AC regardless of content — semantic discrimination at
        # FAST-tier is unreliable, so we explicitly require BRAIN and degrade
        # gracefully otherwise. This is documented in the 2.1.0 evidence notes.
        if self._llm is None or not self._llm.is_role_available(ModelRole.BRAIN):
            semantic_skipped = True
        else:
            story_acs_text = self._extract_story_acs_text(plan_text)
            for ac in brief_acs:
                mapping = await self._check_ac_coverage(ac, story_acs_text)
                # Treat as unmapped if the LLM returned nothing, returned an
                # empty maps_to list, OR returned a confidence below the
                # threshold. Gemma 4 at smaller sizes populates maps_to but
                # reports confidence ≈ 0 when it's really guessing — we honor
                # that signal instead of ignoring it.
                if (
                    mapping is None
                    or not mapping.get("maps_to")
                    or float(mapping.get("confidence", 0.0)) < _MIN_MAPPING_CONFIDENCE
                ):
                    unmapped_acs.append(ac)

        coverage_score = self._score(
            unmapped_acs=unmapped_acs,
            missing_grounding=missing_grounding,
            scope_violations=scope_violations,
            invest_failures=invest_failures,
            phantom_files=phantom_files,
            dependency_errors=dependency_errors,
            vague_cross_references=vague_cross_references,
            brief_ac_total=len(brief_acs) or 1,
        )

        verified = (
            not unmapped_acs
            and not missing_grounding
            and not scope_violations
            and not invest_failures
            and not phantom_files
            and not dependency_errors
            and not vague_cross_references
        )

        return {
            "verified": verified,
            "coverage_score": coverage_score,
            "unmapped_acs": unmapped_acs,
            "missing_grounding": missing_grounding,
            "out_of_scope_violations": scope_violations,
            "invest_failures": invest_failures,
            "phantom_files": phantom_files,
            "dependency_errors": dependency_errors,
            "vague_cross_references": vague_cross_references,
            "semantic_check_skipped": semantic_skipped,
            "summary": self._summary(
                verified=verified,
                unmapped_count=len(unmapped_acs),
                phantom_files_count=len(phantom_files),
                dependency_errors_count=len(dependency_errors),
                vague_cross_references_count=len(vague_cross_references),
                grounding_count=len(missing_grounding),
                scope_count=len(scope_violations),
                invest_count=len(invest_failures),
                semantic_skipped=semantic_skipped,
            ),
        }

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _error(msg: str) -> dict[str, Any]:
        return {
            "verified": False,
            "coverage_score": 0.0,
            "error": msg,
            "unmapped_acs": [],
            "missing_grounding": [],
            "out_of_scope_violations": [],
            "invest_failures": [],
        }

    @staticmethod
    def _extract_brief_acs(sections: dict[str, str]) -> list[str]:
        """Pull checklist items from Business Acceptance Criteria section."""
        body = sections.get("Business Acceptance Criteria", "")
        items = re.findall(
            r"^\s*-\s*\[\s*[ xX]?\s*\]\s+(.+?)$", body, re.MULTILINE
        )
        return [i.strip() for i in items]

    @staticmethod
    def _extract_out_of_scope(sections: dict[str, str]) -> list[str]:
        body = sections.get("Out of Scope", "")
        items: list[str] = []
        for m in re.finditer(r"^\s*-\s*(?:\*\*(.+?)\*\*|(.+?))$", body, re.MULTILINE):
            text = (m.group(1) or m.group(2) or "").strip()
            if text:
                items.append(text.rstrip("."))
        return items

    @staticmethod
    def _extract_story_acs_text(plan_text: str) -> str:
        """Concatenate story AC sections for LLM prompts."""
        stories = PlanValidator._extract_stories(plan_text)
        parts: list[str] = []
        for story_id, story_text in stories:
            ac_match = re.search(
                r"\*\*Acceptance Criteria:?\*\*(.+?)(?=####|\Z)",
                story_text,
                re.DOTALL | re.IGNORECASE,
            )
            if ac_match:
                parts.append(f"Story {story_id}:\n{ac_match.group(1).strip()}")
        return "\n\n".join(parts)

    @staticmethod
    def _missing_grounding_from_issues(issues: list[str]) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for issue in issues:
            m = re.match(r"Subtask ([\w.\-]+):\s+(.+)$", issue)
            if m:
                result.append({"subtask_id": m.group(1), "issue": m.group(2)})
        return result

    @staticmethod
    def _invest_failures_from_issues(issues: list[str]) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for issue in issues:
            m = re.match(r"Story ([\w.\-]+):\s+(.+)$", issue)
            if m:
                result.append({"story_id": m.group(1), "issue": m.group(2)})
        return result

    @staticmethod
    def _detect_scope_violations(plan_text: str, out_of_scope: list[str]) -> list[str]:
        """Flag out-of-scope keywords that appear in plan body."""
        hits: list[str] = []
        plan_lower = plan_text.lower()
        for item in out_of_scope:
            # Key-phrase match — pick the first 3-4 words as a signal
            key = " ".join(item.split()[:4]).lower().strip()
            if len(key) < 4:
                continue
            # Avoid false positives: only flag if phrase appears outside the
            # "Out of Scope" section of the plan itself. Cheap heuristic:
            # require the phrase to appear >1 time (brief will mention it once
            # in plan's scope-boundary reference; beyond that is suspicious).
            count = plan_lower.count(key)
            if count > 1:
                hits.append(item)
        return hits

    async def _check_ac_coverage(
        self, brief_ac: str, story_acs_text: str
    ) -> dict[str, Any] | None:
        """Ask the BRAIN-tier LLM whether this brief AC is addressed by any story AC.

        Uses BRAIN (31B-class) rather than FAST (E2B/E4B) because the
        semantic discrimination required — distinguishing a plausible
        mapping from confabulation — is not reliable at FAST-tier sizes
        per the 2.1.0 evidence runs.
        """
        if self._llm is None:
            return None
        prompt = (
            "Does the story AC set below address the brief AC? Return JSON with "
            "'maps_to' (list of story AC references that cover it, e.g. "
            "'Story 1 AC 2') and 'confidence' (0.0-1.0). If no story AC "
            "genuinely addresses the brief AC, return an empty maps_to list "
            "and confidence 0.0 — do not fabricate a mapping.\n\n"
            f"Brief AC: {brief_ac}\n\nStory ACs:\n{story_acs_text}"
        )
        return await self._llm.generate_structured(
            ModelRole.BRAIN, prompt, _AC_MAPPING_SCHEMA
        )

    @staticmethod
    def _score(
        unmapped_acs: list[str],
        missing_grounding: list[dict[str, str]],
        scope_violations: list[str],
        invest_failures: list[dict[str, str]],
        phantom_files: list[dict[str, str]],
        dependency_errors: list[dict[str, str]],
        vague_cross_references: list[dict[str, str]],
        brief_ac_total: int,
    ) -> float:
        score = 1.0
        score -= 0.5 * (len(unmapped_acs) / brief_ac_total)
        score -= 0.05 * len(missing_grounding)
        score -= 0.1 * len(scope_violations)
        score -= 0.1 * len(invest_failures)
        # Phantom files are the severest mechanical finding — cheap executor
        # cannot execute against a path that doesn't exist.
        score -= 0.15 * len(phantom_files)
        score -= 0.1 * len(dependency_errors)
        score -= 0.03 * len(vague_cross_references)
        return max(0.0, round(score, 3))

    @staticmethod
    def _summary(
        verified: bool,
        unmapped_count: int,
        grounding_count: int,
        scope_count: int,
        invest_count: int,
        phantom_files_count: int,
        dependency_errors_count: int,
        vague_cross_references_count: int,
        semantic_skipped: bool,
    ) -> str:
        if verified:
            suffix = " (semantic check skipped)" if semantic_skipped else ""
            return f"PASS — plan covers brief fully{suffix}"
        parts: list[str] = []
        if unmapped_count:
            parts.append(f"{unmapped_count} unmapped brief ACs")
        if grounding_count:
            parts.append(f"{grounding_count} subtasks missing grounding")
        if scope_count:
            parts.append(f"{scope_count} scope violations")
        if invest_count:
            parts.append(f"{invest_count} INVEST failures")
        if phantom_files_count:
            parts.append(f"{phantom_files_count} phantom file references")
        if dependency_errors_count:
            parts.append(f"{dependency_errors_count} dependency errors")
        if vague_cross_references_count:
            parts.append(f"{vague_cross_references_count} vague cross-references")
        return "FAIL — " + "; ".join(parts)

    # -- Mechanical upgrades ------------------------------------------------

    _FILE_FIELD_RE = re.compile(
        r"\*\*File:\*\*\s*(?P<label>NEW:\s*)?(?P<path>[^\s(]+)",
    )
    _DEPENDS_FIELD_RE = re.compile(
        r"\*\*Depends on:?\*\*\s*(?P<deps>[^\n]+)",
    )
    _VAGUE_CROSS_REF_PATTERNS: list[tuple[str, str]] = [
        (r"\bas\s+discussed\b", "'as discussed' — spell out what was discussed"),
        (r"\bas\s+mentioned\s+above\b", "'as mentioned above' — restate it inline"),
        (r"\bfrom\s+before\b", "'from before' — cite the subtask ID"),
        (r"\blike\s+(?:Step|Subtask)\s+\d+", "'like Step N' — spell out the action"),
        (r"\bthe\s+function\s+from\s+earlier\b", "vague function reference"),
        (r"\bsee\s+above\b", "'see above' — restate inline"),
    ]

    def _check_file_existence(
        self, subtasks: list[tuple[str, str]]
    ) -> list[dict[str, str]]:
        """For each subtask's ``**File:**`` field, verify the path exists.

        If the path is prefixed ``NEW:``, verify the parent directory exists
        instead. Missing paths are returned as ``phantom_files`` entries —
        the most severe mechanical finding because a cheap executor cannot
        operate on a file that doesn't exist.
        """
        phantom: list[dict[str, str]] = []
        for sub_id, body in subtasks:
            m = self._FILE_FIELD_RE.search(body)
            if not m:
                continue
            raw_path = m.group("path").strip().strip(".,;:")
            is_new = bool(m.group("label"))
            candidate = (self._project_root / raw_path).resolve()

            if is_new:
                parent = candidate.parent
                if not parent.exists():
                    phantom.append(
                        {
                            "subtask_id": sub_id,
                            "path": raw_path,
                            "issue": f"NEW: file — parent directory {parent} does not exist",
                        }
                    )
            elif not candidate.exists():
                # Skip pure placeholder tokens from the template (e.g. "x",
                # "path/to/file.py"). A path with at least one directory
                # separator AND a file extension is a real claim worth checking.
                looks_like_real_path = "/" in raw_path and "." in raw_path.split("/")[-1]
                if looks_like_real_path:
                    phantom.append(
                        {
                            "subtask_id": sub_id,
                            "path": raw_path,
                            "issue": f"File referenced but not found at {candidate}",
                        }
                    )
        return phantom

    def _check_dependencies(
        self, subtasks: list[tuple[str, str]]
    ) -> list[dict[str, str]]:
        """Validate the ``**Depends on:**`` graph — no dangling refs, no cycles."""
        errors: list[dict[str, str]] = []
        ids = {sid for sid, _ in subtasks}
        graph: dict[str, set[str]] = {}

        for sub_id, body in subtasks:
            m = self._DEPENDS_FIELD_RE.search(body)
            if not m:
                continue
            raw = m.group("deps").strip()
            # "—" or "-" or empty means "no dependencies" — OK.
            if raw in ("—", "-", "", "none", "None"):
                graph[sub_id] = set()
                continue
            # Split on commas and whitespace. Grab IDs that look like subtask IDs
            # (e.g. "1.1", "2.3", "1.1a"). Ignore prose like "see above".
            parts = re.findall(r"\b[\w.\-]+\.[\w.\-]+\b", raw)
            deps: set[str] = set()
            for p in parts:
                if p in ids:
                    deps.add(p)
                else:
                    errors.append(
                        {
                            "subtask_id": sub_id,
                            "missing_dep": p,
                            "issue": f"depends on '{p}' which is not a subtask in this plan",
                        }
                    )
            graph[sub_id] = deps

        # Cycle detection (DFS with 3-coloring).
        white, gray, black = 0, 1, 2
        color = dict.fromkeys(graph, white)

        def visit(node: str, stack: list[str]) -> None:
            if color.get(node) == gray:
                cycle_start = stack.index(node) if node in stack else 0
                cycle = " → ".join([*stack[cycle_start:], node])
                errors.append(
                    {
                        "subtask_id": node,
                        "missing_dep": "(cycle)",
                        "issue": f"dependency cycle: {cycle}",
                    }
                )
                return
            if color.get(node) == black:
                return
            color[node] = gray
            for nxt in graph.get(node, ()):
                visit(nxt, [*stack, node])
            color[node] = black

        for n in graph:
            if color[n] == white:
                visit(n, [])
        return errors

    def _check_vague_cross_references(
        self, subtasks: list[tuple[str, str]]
    ) -> list[dict[str, str]]:
        """Detect vague cross-subtask references that a cheap executor can't resolve."""
        findings: list[dict[str, str]] = []
        for sub_id, body in subtasks:
            for pattern, msg in self._VAGUE_CROSS_REF_PATTERNS:
                if re.search(pattern, body, re.IGNORECASE):
                    findings.append(
                        {"subtask_id": sub_id, "issue": msg}
                    )
        return findings
