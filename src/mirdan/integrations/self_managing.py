"""Self-managing integration for Claude Code.

After `mirdan init`, this generates `.claude/rules/mirdan-workflow.md`
that replaces all manual CLAUDE.md mirdan instructions. Combined with
hooks, mirdan manages itself entirely — zero manual configuration needed.
"""

from __future__ import annotations

import contextlib
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mirdan.cli.detect import DetectedProject
    from mirdan.core.quality_standards import QualityStandards
    from mirdan.models import CompactState


class SelfManagingIntegration:
    """Generates self-managing configuration for Claude Code.

    The generated workflow rule replaces all CLAUDE.md mirdan
    instructions with an auto-enforced quality sandwich.
    """

    def __init__(
        self,
        standards: QualityStandards | None = None,
    ) -> None:
        self._standards = standards

    def generate_workflow_rule(
        self,
        detected: DetectedProject,
        languages: list[str] | None = None,
    ) -> str:
        """Generate the mirdan-workflow.md rule content.

        This rule file teaches Claude Code how to use mirdan automatically.
        It replaces all CLAUDE.md instructions about mirdan.

        Args:
            detected: Detected project metadata.
            languages: Optional list of languages (workspace mode).

        Returns:
            Workflow rule markdown content.
        """
        # Try loading from template first
        template_content = self._load_template()
        if template_content:
            return self._render_template(template_content, detected, languages=languages)

        # Fallback: generate inline
        return self._generate_inline(detected)

    def generate_quality_context(
        self,
        session_data: dict[str, Any] | None = None,
    ) -> str:
        """Generate quality context for session injection.

        Used by SessionStart hooks to inject mirdan awareness into
        new sessions without requiring manual CLAUDE.md instructions.

        Args:
            session_data: Optional existing session data to include.

        Returns:
            Quality context string for prompt injection.
        """
        lines = [
            "## mirdan Quality Context (Auto-Injected)",
            "",
            "mirdan is active in this project. Quality workflow:",
            "1. `mcp__mirdan__enhance_prompt(task)` — before any coding task",
            "2. Write code following the quality requirements returned",
            "3. `mcp__mirdan__validate_code_quality(code)` — after writing code",
            "4. Fix all errors before marking complete",
            "",
        ]

        if session_data:
            lines.append("### Active Session")
            if session_data.get("language"):
                lines.append(f"- Language: {session_data['language']}")
            if session_data.get("touches_security"):
                lines.append("- Security-sensitive: yes (use check_security=true)")
            if session_data.get("frameworks"):
                lines.append(f"- Frameworks: {', '.join(session_data['frameworks'])}")
            lines.append("")

        return "\n".join(lines)

    def generate_compaction_state(
        self,
        compact_state: CompactState,
    ) -> str:
        """Generate state string for context compaction survival.

        Args:
            compact_state: The minimal state to preserve.

        Returns:
            Compact state string that can survive compaction.
        """
        lines = [
            "## mirdan Compacted State (Restore After Compaction)",
            "",
        ]

        if compact_state.session_id:
            lines.append(f"- Session: {compact_state.session_id}")
        if compact_state.task_type:
            lines.append(f"- Task: {compact_state.task_type}")
        if compact_state.language:
            lines.append(f"- Language: {compact_state.language}")
        if compact_state.touches_security:
            lines.append("- Security: sensitive")
        if compact_state.last_score is not None:
            lines.append(f"- Last score: {compact_state.last_score:.2f}")
        if compact_state.open_violations > 0:
            lines.append(f"- Open violations: {compact_state.open_violations}")
        if compact_state.frameworks:
            lines.append(f"- Frameworks: {', '.join(compact_state.frameworks)}")
        lines.append("")
        lines.append("To restore: call mcp__mirdan__enhance_prompt with your current task.")

        return "\n".join(lines)

    def restore_from_compaction(
        self,
        state_text: str,
    ) -> dict[str, Any]:
        """Parse compacted state text back into a dict.

        Args:
            state_text: The compacted state text.

        Returns:
            Parsed state dict.
        """
        state: dict[str, Any] = {}
        for line in state_text.split("\n"):
            line = line.strip()
            if line.startswith("- Session: "):
                state["session_id"] = line[len("- Session: ") :]
            elif line.startswith("- Task: "):
                state["task_type"] = line[len("- Task: ") :]
            elif line.startswith("- Language: "):
                state["language"] = line[len("- Language: ") :]
            elif line.startswith("- Security: "):
                state["touches_security"] = line[len("- Security: ") :] == "sensitive"
            elif line.startswith("- Last score: "):
                with contextlib.suppress(ValueError):
                    state["last_score"] = float(line[len("- Last score: ") :])
            elif line.startswith("- Open violations: "):
                with contextlib.suppress(ValueError):
                    state["open_violations"] = int(line[len("- Open violations: ") :])

            elif line.startswith("- Frameworks: "):
                fw_str = line[len("- Frameworks: ") :]
                state["frameworks"] = [f.strip() for f in fw_str.split(",")]
        return state

    def generate_session_summary(
        self,
        session_data: dict[str, Any],
        file_results: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate a markdown summary of a quality session.

        Args:
            session_data: Session quality data (validation_count, avg_score, etc.).
            file_results: Optional per-file results.

        Returns:
            Markdown-formatted session summary.
        """
        lines = [
            "# mirdan Session Quality Summary",
            "",
        ]

        validation_count = session_data.get("validation_count", 0)
        avg_score = session_data.get("avg_score", 0.0)
        files_validated = session_data.get("files_validated", 0)
        unresolved_errors = session_data.get("unresolved_errors", 0)

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Validations | {validation_count} |")
        lines.append(f"| Files validated | {files_validated} |")
        lines.append(f"| Average score | {avg_score:.3f} |")
        lines.append(f"| Unresolved errors | {unresolved_errors} |")

        status = "PASS" if unresolved_errors == 0 else "NEEDS WORK"
        lines.append(f"| Status | {status} |")
        lines.append("")

        if file_results:
            lines.append("## File Details")
            lines.append("")
            lines.append("| File | Score | Status |")
            lines.append("|------|-------|--------|")
            for fr in file_results:
                s = "PASS" if fr.get("passed", True) else "FAIL"
                lines.append(f"| `{fr.get('file', '')}` | {fr.get('score', 0):.2f} | {s} |")
            lines.append("")

        return "\n".join(lines)

    def write_workflow_rule(
        self,
        project_dir: Path,
        detected: DetectedProject,
        languages: list[str] | None = None,
    ) -> Path:
        """Generate and write the workflow rule file.

        Args:
            project_dir: Project root directory.
            detected: Detected project metadata.
            languages: Optional list of languages (workspace mode).

        Returns:
            Path to the generated rule file.
        """
        rules_dir = project_dir / ".claude" / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)

        content = self.generate_workflow_rule(detected, languages=languages)
        rule_path = rules_dir / "mirdan-workflow.md"
        rule_path.write_text(content)
        return rule_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_template(self) -> str | None:
        """Try to load workflow template from package resources."""
        try:
            pkg = files("mirdan.integrations.templates.claude_code")
            template_file = pkg / "mirdan-workflow.md"
            return template_file.read_text()
        except (ModuleNotFoundError, FileNotFoundError, TypeError, AttributeError):
            return None

    def _render_template(
        self,
        template: str,
        detected: DetectedProject,
        languages: list[str] | None = None,
    ) -> str:
        """Render template with project-specific values.

        Args:
            template: Template string with ``{{language}}`` and ``{{frameworks}}`` placeholders.
            detected: Detected project metadata.
            languages: Optional list of languages (workspace mode). When provided,
                ``{{language}}`` is replaced with a comma-separated list.
        """
        if languages and len(languages) > 1:
            lang = ", ".join(languages)
        else:
            lang = detected.primary_language or "python"
        frameworks = ", ".join(detected.frameworks) if detected.frameworks else "none detected"
        return template.replace("{{language}}", lang).replace("{{frameworks}}", frameworks)

    def _generate_inline(self, detected: DetectedProject) -> str:
        """Generate workflow rule inline (fallback if template missing)."""
        lang = detected.primary_language or "python"
        frameworks = ", ".join(detected.frameworks) if detected.frameworks else "none"

        return f"""# mirdan Quality Workflow

> Auto-generated by mirdan. Regenerate with `mirdan init --upgrade`.

## Quality Sandwich (Mandatory for All Coding Tasks)

### Before Writing Code
1. Call `mcp__mirdan__enhance_prompt(task_description)` to get:
   - Quality requirements for this specific task
   - Security sensitivity detection
   - Framework-specific guidance
   - Tool recommendations

### After Writing Code
2. Call `mcp__mirdan__validate_code_quality(code)` to validate:
   - Pass `check_security=true` if `touches_security` was flagged
   - Fix ALL errors before proceeding
   - Warnings should be addressed when practical

### Before Marking Complete
3. Ensure validation passes with no errors

## Available Tools

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `enhance_prompt` | Quality framework for a task | Start of ANY coding task |
| `validate_code_quality` | Full code validation | After writing code |
| `validate_quick` | Fast security-only check | Real-time feedback (hooks) |
| `get_quality_standards` | Language/framework rules | Before coding in {lang} |

## Project Context
- **Language**: {lang}
- **Frameworks**: {frameworks}
- **Quality enforcement**: Automatic via hooks

## Auto-Fix
When violations are found, mirdan provides fix suggestions.
Apply fixes with `mirdan fix <file>` or review suggestions inline.
"""
