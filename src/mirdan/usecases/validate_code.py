"""ValidateCodeQuality use case — extracted from server.py."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from mirdan.usecases.helpers import (
    _MAX_CODE_LENGTH,
    _check_input_size,
    _parse_model_tier,
    _process_knowledge_entries,
)

if TYPE_CHECKING:
    import asyncio

    from mirdan.config import MirdanConfig
    from mirdan.core.active_orchestrator import ToolExecutor
    from mirdan.core.agent_coordinator import AgentCoordinator
    from mirdan.core.code_validator import CodeValidator
    from mirdan.core.knowledge_producer import KnowledgeProducer
    from mirdan.core.linter_runner import LinterRunner
    from mirdan.core.output_formatter import OutputFormatter
    from mirdan.core.prompt_composer import PromptComposer
    from mirdan.core.quality_persistence import QualityPersistence
    from mirdan.core.semantic_analyzer import SemanticAnalyzer
    from mirdan.core.session_manager import SessionManager
    from mirdan.core.session_tracker import SessionTracker

logger = logging.getLogger(__name__)


class ValidateCodeUseCase:
    """Validate generated code against quality standards."""

    def __init__(
        self,
        code_validator: CodeValidator,
        session_manager: SessionManager,
        linter_runner: LinterRunner,
        violation_explainer: Any,
        quality_persistence: QualityPersistence,
        session_tracker: SessionTracker,
        semantic_analyzer: SemanticAnalyzer,
        output_formatter: OutputFormatter,
        knowledge_producer: KnowledgeProducer,
        prompt_composer: PromptComposer,
        config: MirdanConfig,
        active_orchestrator: ToolExecutor,
        background_tasks: set[asyncio.Task[Any]],
        agent_coordinator: AgentCoordinator | None = None,
    ) -> None:
        self._code_validator = code_validator
        self._session_manager = session_manager
        self._linter_runner = linter_runner
        self._violation_explainer = violation_explainer
        self._quality_persistence = quality_persistence
        self._session_tracker = session_tracker
        self._semantic_analyzer = semantic_analyzer
        self._output_formatter = output_formatter
        self._knowledge_producer = knowledge_producer
        self._prompt_composer = prompt_composer
        self._config = config
        self._active_orchestrator = active_orchestrator
        self._background_tasks = background_tasks
        self._agent_coordinator = agent_coordinator

    async def execute(
        self,
        code: str,
        language: str = "auto",
        check_security: bool = True,
        check_architecture: bool = True,
        check_style: bool = True,
        severity_threshold: str = "warning",
        session_id: str = "",
        max_tokens: int = 0,
        model_tier: str = "auto",
        input_type: str = "code",
        compare: bool = False,
        file_path: str = "",
        test_file: str = "",
        changed_lines: frozenset[int] | None = None,
    ) -> dict[str, Any]:
        """Execute the validate_code_quality use case.

        Args:
            code: The code to validate
            language: Programming language (python|typescript|javascript|rust|go|auto)
            check_security: Validate against security standards
            check_architecture: Validate against architecture standards
            check_style: Validate against language-specific style standards
            severity_threshold: Minimum severity to include in results (error|warning|info)
            session_id: Session ID from enhance_prompt to auto-inherit language and security
            max_tokens: Maximum token budget for the response (0=unlimited)
            model_tier: Target model tier for output optimization (auto|opus|sonnet|haiku)
            input_type: Input type - "code" for raw code (default), "diff" for unified
                       diff format (git diff output).
            compare: If True, treat `code` as JSON array of implementations to compare
            file_path: Optional file path for external linter analysis.

        Returns:
            Validation results with pass/fail, score, violations, and summary
        """
        from mirdan.core.linter_orchestrator import merge_linter_violations
        from mirdan.models import Intent, TaskType

        # --- Mode: compare (replaces standalone compare_approaches tool) ---
        if compare:
            return await self._validate_comparison(code, language)

        # --- Mode: diff (replaces standalone validate_diff tool) ---
        if input_type == "diff":
            return await self._validate_diff(
                code, language, check_security, session_id, max_tokens, model_tier
            )

        # --- Standard code validation ---
        # Validate input size
        if error := _check_input_size(code, "code", _MAX_CODE_LENGTH):
            return error

        # Apply session defaults if available
        resolved_language, resolved_security = self._session_manager.apply_session_defaults(
            session_id, language=language, check_security=check_security
        )

        # Resolve per-file threshold overrides
        resolved_thresholds = None
        if file_path and self._config.thresholds.file_overrides:
            resolved_thresholds = self._config.thresholds.resolve_for_file(file_path)

        _t0 = perf_counter()
        result = self._code_validator.validate(
            code=code,
            language=resolved_language,
            check_security=resolved_security,
            check_architecture=check_architecture,
            check_style=check_style,
            thresholds=resolved_thresholds,
            file_path=file_path,
            test_file=test_file,
            changed_lines=changed_lines,
        )
        _t_validate = perf_counter() - _t0

        # Run external linters if file_path provided
        if file_path:
            fp = Path(file_path)
            if fp.exists():
                linter_violations = await self._linter_runner.run(fp, result.language_detected)
                if linter_violations:
                    result = merge_linter_violations(
                        result, linter_violations, self._config.thresholds
                    )

        # Enrich violations with contextual explanations
        if result.violations:
            try:
                self._violation_explainer.enrich_violations(result.violations)
            except Exception:
                logger.debug("Failed to enrich violations", exc_info=True)

        # Persist snapshot for quality trends
        try:
            self._quality_persistence.save_snapshot(result)
        except Exception:
            logger.debug("Failed to save quality snapshot", exc_info=True)

        # Track validation in session
        session = None
        if session_id:
            session = self._session_manager.get(session_id)
        self._session_tracker.record_validation(result, file_path=file_path, session=session)

        # Detect security regression (must be AFTER record_validation — uses records[-2])
        _security_regression = self._session_tracker.detect_security_regression(
            file_path, result.violations
        )

        # Compute violation delta for re-validations within a session
        delta: dict[str, Any] = {}
        if session and session.validation_count > 1:
            key = file_path or ""
            prev_violations = self._session_tracker.get_previous_violations(key)
            curr_violations = {v.id for v in result.violations}
            resolved = prev_violations - curr_violations
            new_viol = curr_violations - prev_violations
            persistence = self._session_tracker.get_violation_persistence(key)
            if resolved:
                delta["resolved"] = sorted(resolved)
            if new_viol:
                delta["new"] = sorted(new_viol)
            persistent = {r: cnt for r, cnt in persistence.items() if cnt > 1}
            if persistent:
                delta["persistent"] = persistent

        output = result.to_dict(severity_threshold=severity_threshold)

        # Cross-session quality drift detection
        baseline = self._quality_persistence.get_baseline_score()
        if baseline is not None and baseline - result.score > 0.15:
            output["quality_drift"] = {
                "baseline": round(baseline, 3),
                "current": round(result.score, 3),
                "drift": round(baseline - result.score, 3),
            }

        # Surface security regression warning
        if _security_regression:
            output["security_regression"] = {
                "warning": (
                    "Security regression: this file previously passed security checks"
                    " but now has security violations"
                ),
                "security_violations": [
                    v.id for v in result.violations if v.category == "security"
                ],
            }

        # Generate semantic review questions (Layer 1)
        if self._config.semantic.enabled:
            semantic_checks = self._code_validator.generate_semantic_checks(
                code=code,
                language=result.language_detected,
                violations=result.violations,
            )
            if semantic_checks:
                output["semantic_checks"] = [s.to_dict() for s in semantic_checks]

            # Layer 3: Analysis protocol for security-critical code
            if resolved_security and self._config.semantic.analysis_protocol != "none":
                protocol = self._semantic_analyzer.generate_analysis_protocol(
                    code=code,
                    language=result.language_detected,
                    violations=result.violations,
                    semantic_checks=semantic_checks if semantic_checks else [],
                )
                if protocol:
                    output["analysis_protocol"] = protocol.to_dict()

        # Include session quality summary if session is active
        if session and session.validation_count > 0:
            output["session_quality"] = {
                "validation_count": session.validation_count,
                "avg_score": round(session.cumulative_score / session.validation_count, 3),
                "unresolved_errors": session.unresolved_errors,
                "files_validated": len(session.files_validated),
            }

        # Session context: violation delta and recommendation reminders
        if delta:
            output["session_context"] = delta
        if session and session.tool_recommendations:
            output["recommendation_reminders"] = session.tool_recommendations

        # Add knowledge entries for enyal storage
        knowledge_entries = self._knowledge_producer.extract_from_validation(result)
        if knowledge_entries:
            output["knowledge_entries"] = _process_knowledge_entries(
                knowledge_entries,
                self._config,
                self._active_orchestrator,
                self._background_tasks,
            )
            if not self._config.orchestration.auto_memory:
                output["knowledge_storage_hint"] = (
                    "Store entries marked auto_store=true via enyal_remember. "
                    "For convention entries, use suggest_supersedes=true to detect "
                    "and link superseded decisions. Tags and scope are pre-set."
                )

        # Add verification checklist to output (absorbs get_verification_checklist)
        intent = Intent(
            original_prompt="",
            task_type=TaskType.UNKNOWN,
            touches_security=resolved_security,
        )
        # Detect task type from session if available
        if session:
            intent.task_type = session.task_type
        output["checklist"] = self._prompt_composer.generate_verification_steps(intent)
        if session and session.validation_count > 1:
            output["checklist_note"] = (
                "Re-validation. Review session_context.resolved for progress."
            )

        # Check for multi-agent conflicts on the validated file
        if (
            self._agent_coordinator is not None
            and self._agent_coordinator.is_enabled
            and file_path
            and session_id
        ):
            coord_warnings = self._agent_coordinator.check_conflicts(session_id, file_path)
            if coord_warnings:
                output["coordination"] = {"warnings": [w.to_dict() for w in coord_warnings]}

        output["timing_ms"] = {
            "validation": round(_t_validate * 1000, 1),
            "total": round((perf_counter() - _t0) * 1000, 1),
        }

        # Apply token-budget-aware formatting
        tier = _parse_model_tier(model_tier)
        output = self._output_formatter.format_validation_result(
            output, max_tokens=max_tokens, model_tier=tier
        )

        return output

    async def _validate_diff(
        self,
        diff: str,
        language: str,
        check_security: bool,
        session_id: str,
        max_tokens: int,
        model_tier: str,
    ) -> dict[str, Any]:
        """Handle diff validation (replaces standalone validate_diff tool)."""
        from mirdan.core.diff_parser import parse_unified_diff
        from mirdan.models import ValidationResult

        if error := _check_input_size(diff, "diff", _MAX_CODE_LENGTH):
            return error

        # Parse the diff
        parsed = parse_unified_diff(diff)
        added_code, line_mapping = parsed.get_added_code_with_mapping()

        if not added_code.strip():
            return {
                "passed": True,
                "score": 1.0,
                "files_changed": parsed.files_changed,
                "summary": "No added code found in diff",
            }

        # Apply session defaults if available
        resolved_language, resolved_security = self._session_manager.apply_session_defaults(
            session_id, language=language, check_security=check_security
        )

        # Attempt full-file validation when files are resolvable on disk.
        # Full files enable architecture checks and reduce false positives.
        project_dir = Path.cwd()
        file_scope_rules = {"ARCH002", "TSARCH002"}
        full_file_violations: list[Any] = []
        fallback_files: list[str] = []

        for diff_file in parsed.files_changed:
            full_path = project_dir / diff_file
            if full_path.exists():
                try:
                    file_code = full_path.read_text(encoding="utf-8")
                    changed_lines = set(parsed.get_changed_line_numbers(diff_file))
                    file_result = self._code_validator.validate(
                        code=file_code,
                        language=resolved_language,
                        check_security=resolved_security,
                        check_architecture=True,
                        check_style=True,
                    )
                    # Filter to changed lines only, excluding file-scope rules
                    for v in file_result.violations:
                        if v.id in file_scope_rules:
                            continue
                        if v.line is not None and v.line in changed_lines:
                            full_file_violations.append(v)
                except (OSError, UnicodeDecodeError):
                    fallback_files.append(diff_file)
            else:
                fallback_files.append(diff_file)

        # For files not found on disk, fall back to added-lines-only validation
        fallback_result = None
        if fallback_files:
            # Filter added_code to only include lines from fallback files
            fallback_code_parts: list[str] = []
            fallback_line_mapping: dict[int, tuple[str, int]] = {}
            current_line = 1
            for extracted_line, (fpath, orig_line) in line_mapping.items():
                if fpath in fallback_files:
                    # Get the line content from added_code
                    added_lines = added_code.split("\n")
                    if extracted_line <= len(added_lines):
                        fallback_code_parts.append(added_lines[extracted_line - 1])
                        fallback_line_mapping[current_line] = (fpath, orig_line)
                        current_line += 1

            if fallback_code_parts:
                fallback_code = "\n".join(fallback_code_parts)
                fallback_result = self._code_validator.validate(
                    code=fallback_code,
                    language=resolved_language,
                    check_security=resolved_security,
                    check_architecture=False,
                    check_style=True,
                )
        elif not full_file_violations and not fallback_files:
            # All files found on disk but no violations on changed lines
            pass

        # If no files were on disk at all, use original added-lines-only approach
        if not parsed.files_changed or (
            not full_file_violations
            and fallback_result is None
            and fallback_files == list(parsed.files_changed)
        ):
            result = self._code_validator.validate(
                code=added_code,
                language=resolved_language,
                check_security=resolved_security,
                check_architecture=False,
                check_style=True,
            )
        else:
            # Merge full-file violations with fallback violations
            all_violations = list(full_file_violations)
            if fallback_result:
                all_violations.extend(fallback_result.violations)

            passed = not any(v.severity == "error" for v in all_violations)
            # Use the first full-file result's language detection, or fallback
            lang = resolved_language
            score = 1.0
            if all_violations:
                score = self._code_validator._calculate_score(all_violations)

            result = ValidationResult(
                passed=passed,
                score=score,
                language_detected=lang,
                violations=all_violations,
                standards_checked=["full_file_diff"],
                limitations=[],
            )

        # Persist snapshot for quality trends
        try:
            self._quality_persistence.save_snapshot(result)
        except Exception:
            logger.debug("Failed to save quality snapshot", exc_info=True)

        output = result.to_dict(severity_threshold="warning")
        output["files_changed"] = parsed.files_changed
        output["lines_added"] = sum(len(h.added_lines) for h in parsed.hunks)

        # Remap violation line numbers to original file locations for fallback violations
        if "violations" in output and fallback_result and not full_file_violations:
            for violation in output["violations"]:
                extracted_line = violation.get("line")
                if extracted_line and extracted_line in line_mapping:
                    file_path_str, original_line = line_mapping[extracted_line]
                    violation["file"] = file_path_str
                    violation["original_line"] = original_line
                    violation["line"] = original_line

        # Apply token-budget-aware formatting
        tier = _parse_model_tier(model_tier)
        output = self._output_formatter.format_validation_result(
            output, max_tokens=max_tokens, model_tier=tier
        )

        return output

    async def _validate_comparison(
        self,
        code: str,
        language: str,
    ) -> dict[str, Any]:
        """Handle multi-implementation comparison (replaces standalone compare_approaches)."""
        from mirdan.models import ComparisonEntry, ComparisonResult

        try:
            implementations = json.loads(code)
        except (json.JSONDecodeError, TypeError):
            return {
                "error": "When compare=True, code must be a JSON array of implementation strings"
            }

        if not isinstance(implementations, list):
            return {
                "error": "When compare=True, code must be a JSON array of implementation strings"
            }

        if len(implementations) < 2:
            return {"error": "At least 2 implementations are required for comparison"}
        if len(implementations) > 10:
            return {"error": "Maximum 10 implementations can be compared at once"}

        for i, impl in enumerate(implementations):
            if not isinstance(impl, str):
                return {"error": f"implementation[{i}] must be a string"}
            if error := _check_input_size(impl, f"implementation[{i}]", _MAX_CODE_LENGTH):
                return error

        labels = [f"Implementation {i + 1}" for i in range(len(implementations))]

        entries: list[ComparisonEntry] = []
        for impl, label in zip(implementations, labels, strict=True):
            result = self._code_validator.validate(
                code=impl,
                language=language,
                check_security=True,
                check_architecture=True,
                check_style=True,
            )
            output = result.to_dict(severity_threshold="warning")
            entries.append(
                ComparisonEntry(
                    label=label,
                    score=result.score,
                    passed=result.passed,
                    violation_counts=output.get("violations_count", {}),
                    summary=output.get("summary", ""),
                )
            )

        # Determine winner (highest score, then fewest errors)
        best = max(entries, key=lambda e: (e.score, -e.violation_counts.get("error", 0)))

        comparison = ComparisonResult(
            entries=entries,
            winner=best.label,
            language_detected=entries[0].summary.split()[0] if entries else language,
        )

        return comparison.to_dict()
