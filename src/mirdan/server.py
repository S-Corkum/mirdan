"""Mirdan MCP Server - AI Code Quality Orchestrator."""

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml
from fastmcp import FastMCP

from mirdan.config import MirdanConfig
from mirdan.core.active_orchestrator import ActiveOrchestrator
from mirdan.core.auto_fixer import AutoFixer
from mirdan.core.code_validator import CodeValidator
from mirdan.core.context_aggregator import ContextAggregator
from mirdan.core.convention_extractor import ConventionExtractor
from mirdan.core.diff_parser import parse_unified_diff
from mirdan.core.environment_detector import IDEType, detect_environment
from mirdan.core.intent_analyzer import IntentAnalyzer
from mirdan.core.knowledge_producer import KnowledgeProducer
from mirdan.core.linter_orchestrator import create_linter_runner, merge_linter_violations
from mirdan.core.linter_runner import LinterRunner
from mirdan.core.manifest_parser import ManifestParser
from mirdan.core.orchestrator import MCPOrchestrator
from mirdan.core.output_formatter import OutputFormatter
from mirdan.core.plan_validator import PlanValidator
from mirdan.core.prompt_composer import PromptComposer
from mirdan.core.quality_persistence import QualityPersistence
from mirdan.core.quality_standards import QualityStandards
from mirdan.core.semantic_analyzer import SemanticAnalyzer
from mirdan.core.session_manager import SessionManager
from mirdan.core.session_tracker import SessionTracker
from mirdan.core.vuln_scanner import VulnScanner
from mirdan.models import (
    ComparisonEntry,
    ComparisonResult,
    Intent,
    KnowledgeEntry,
    ModelTier,
    SessionContext,
    TaskType,
    ValidationResult,
)

logger = logging.getLogger(__name__)

# Input size limits to prevent abuse and resource exhaustion
_MAX_PROMPT_LENGTH = 50_000  # ~12k tokens
_MAX_CODE_LENGTH = 500_000  # ~125k tokens
_MAX_PLAN_LENGTH = 200_000  # ~50k tokens


def _check_input_size(value: str, name: str, max_length: int) -> dict[str, Any] | None:
    """Return an error dict if value exceeds max_length, else None."""
    if len(value) > max_length:
        return {
            "error": f"{name} exceeds maximum length ({len(value):,} > {max_length:,} characters)",
            "max_length": max_length,
            "actual_length": len(value),
        }
    return None


@dataclass
class _Components:
    """Holds all initialized Mirdan components."""

    intent_analyzer: IntentAnalyzer
    quality_standards: QualityStandards
    prompt_composer: PromptComposer
    mcp_orchestrator: MCPOrchestrator
    context_aggregator: ContextAggregator
    code_validator: CodeValidator
    plan_validator: PlanValidator
    session_manager: SessionManager
    output_formatter: OutputFormatter
    quality_persistence: QualityPersistence
    knowledge_producer: KnowledgeProducer
    linter_runner: LinterRunner
    session_tracker: SessionTracker
    auto_fixer: AutoFixer
    convention_extractor: ConventionExtractor
    violation_explainer: Any  # ViolationExplainer (lazy import)
    manifest_parser: ManifestParser
    vuln_scanner: VulnScanner
    semantic_analyzer: SemanticAnalyzer
    active_orchestrator: ActiveOrchestrator
    config: MirdanConfig


_components: _Components | None = None

# Strong references to fire-and-forget background tasks.
# Without this, asyncio tasks can be garbage-collected mid-execution.
# See: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
_background_tasks: set[asyncio.Task[Any]] = set()

# Tool priority order for budget-aware filtering.
# When MIRDAN_TOOL_BUDGET is set, only the top N tools are kept.
_TOOL_PRIORITY = [
    "validate_code_quality",
    "validate_quick",
    "enhance_prompt",
    "get_quality_standards",
    "get_quality_trends",
    "scan_dependencies",
    "scan_conventions",
]


def _create_violation_explainer() -> Any:
    """Create the ViolationExplainer instance."""
    from mirdan.core.violation_explainer import ViolationExplainer

    return ViolationExplainer()


def _get_components() -> _Components:
    """Get or create the singleton component set."""
    global _components
    if _components is not None:
        return _components

    config = MirdanConfig.find_config()

    # Apply quality profile if not default (Bug #2: apply_profile was never called)
    if config.quality_profile and config.quality_profile != "default":
        from mirdan.core.quality_profiles import apply_profile, get_profile

        try:
            profile = get_profile(config.quality_profile, config.custom_profiles)
            config_dict = {"quality": config.quality.__dict__.copy()}
            apply_profile(profile, config_dict)
            # Update the quality config with profile values
            for key, value in config_dict.get("quality", {}).items():
                if hasattr(config.quality, key):
                    setattr(config.quality, key, value)
        except ValueError:
            logger.warning("Unknown quality profile '%s', using default", config.quality_profile)

    # Detect project directory for AI002 import verification and manifest detection
    project_dir = Path.cwd()

    quality_standards = QualityStandards(config=config.quality, project_dir=project_dir)

    # Detect environment once for output optimization (Phase 2G)
    env_info = detect_environment()
    compact_threshold = config.tokens.compact_threshold
    minimal_threshold = config.tokens.minimal_threshold
    micro_threshold = config.tokens.micro_threshold

    if env_info.ide == IDEType.CLAUDE_CODE:
        # Claude Code: prefer compact mode (context window pressure from hooks/rules)
        compact_threshold = max(compact_threshold, 8000)
    elif env_info.ide == IDEType.CURSOR:
        # Cursor: prefer full mode (larger context, MCP Apps can render)
        compact_threshold = min(compact_threshold, 2000)

    # Initialize new components for semantic validation and dependency scanning
    manifest_parser = ManifestParser(project_dir=project_dir)
    cache_dir = project_dir / ".mirdan" / "cache"
    vuln_scanner = VulnScanner(cache_dir=cache_dir, ttl=config.dependencies.osv_cache_ttl)
    semantic_analyzer = SemanticAnalyzer(config=config.semantic)

    context_aggregator = ContextAggregator(config)

    _components = _Components(
        intent_analyzer=IntentAnalyzer(config.project, manifest_parser=manifest_parser),
        quality_standards=quality_standards,
        prompt_composer=PromptComposer(quality_standards, config=config.enhancement),
        mcp_orchestrator=MCPOrchestrator(config.orchestration),
        context_aggregator=context_aggregator,
        code_validator=CodeValidator(
            quality_standards,
            config=config.quality,
            thresholds=config.thresholds,
            project_dir=project_dir,
            semantic_analyzer=semantic_analyzer,
            manifest_parser=manifest_parser,
            vuln_scanner=vuln_scanner,
        ),
        plan_validator=PlanValidator(config.planning, thresholds=config.thresholds),
        session_manager=SessionManager(config.session),
        output_formatter=OutputFormatter(
            compact_threshold=compact_threshold,
            minimal_threshold=minimal_threshold,
            micro_threshold=micro_threshold,
        ),
        quality_persistence=QualityPersistence(),
        knowledge_producer=KnowledgeProducer(),
        linter_runner=create_linter_runner(config),
        session_tracker=SessionTracker(),
        auto_fixer=AutoFixer(),
        convention_extractor=ConventionExtractor(config),
        violation_explainer=_create_violation_explainer(),
        manifest_parser=manifest_parser,
        vuln_scanner=vuln_scanner,
        semantic_analyzer=semantic_analyzer,
        active_orchestrator=ActiveOrchestrator(context_aggregator.registry),
        config=config,
    )
    return _components


@asynccontextmanager
async def _lifespan(app: FastMCP[Any]) -> AsyncIterator[None]:
    """Manage server lifecycle: startup initialization and shutdown cleanup."""
    # Startup: eagerly initialize components
    _get_components()

    # Apply tool budget filtering if MIRDAN_TOOL_BUDGET is set.
    # At import time, all 5 tools are registered via @mcp.tool() so .fn access
    # works in tests. When the server actually starts (lifespan runs), excess
    # tools are pruned based on priority order.
    budget_str = os.environ.get("MIRDAN_TOOL_BUDGET")
    if budget_str is not None and budget_str != "":
        try:
            budget = int(budget_str)
        except ValueError:
            budget = -1  # invalid value → keep all tools
        if budget >= 0:
            keep = set(_TOOL_PRIORITY[:budget])
            to_remove = [name for name in list(app._tool_manager._tools) if name not in keep]
            for name in to_remove:
                del app._tool_manager._tools[name]

    yield
    # Shutdown: cleanup MCP client connections
    if _components is not None:
        await _components.context_aggregator.close()


# Initialize the MCP server with lifespan
mcp = FastMCP("Mirdan", instructions="AI Code Quality Orchestrator", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Session feedback helpers
# ---------------------------------------------------------------------------


def _get_persistent_violation_reqs(
    session: SessionContext, session_tracker: SessionTracker
) -> list[str]:
    """Build quality requirement strings from persistent violations in session history.

    Returns up to 3 enriched requirement strings for rules that failed across 2+
    consecutive validations. Each string includes the violation message, fix suggestion,
    and last-seen line number when available — giving the LLM structured context to
    converge on a fix faster (per Static Analysis Feedback Loop research).
    """
    if session.validation_count == 0 or not session.files_validated:
        return []
    persistence: dict[str, int] = {}
    details: dict[str, dict[str, Any]] = {}
    for file_path in session.files_validated:
        for rule_id, count in session_tracker.get_violation_persistence(file_path).items():
            if count >= 2:
                persistence[rule_id] = max(persistence.get(rule_id, 0), count)
        for rule_id, detail in session_tracker.get_persistent_violation_details(file_path).items():
            if rule_id not in details:
                details[rule_id] = detail
    if not persistence:
        return []
    sorted_rules = sorted(persistence.items(), key=lambda x: -x[1])[:3]
    reqs: list[str] = []
    for rule_id, count in sorted_rules:
        detail = details.get(rule_id, {})
        msg = detail.get("message", "")
        suggestion = detail.get("suggestion", "")
        line = detail.get("line")
        parts = [f"PRIORITY FIX: {rule_id}"]
        if msg:
            parts.append(f"({msg})")
        parts.append(f"— failed {count} consecutive validations")
        if line:
            parts.append(f"at line {line}")
        if suggestion:
            parts.append(f". Fix: {suggestion}")
        reqs.append(" ".join(parts))
    return reqs


# ---------------------------------------------------------------------------
# Core Tool 1: enhance_prompt
# ---------------------------------------------------------------------------


@mcp.tool()
async def enhance_prompt(
    prompt: str,
    task_type: str = "auto",
    context_level: str = "auto",
    max_tokens: int = 0,
    model_tier: str = "auto",
    session_id: str = "",
) -> dict[str, Any]:
    """
    Automatically enhance a coding prompt with quality requirements,
    codebase context, and tool recommendations.

    Args:
        prompt: The original developer prompt
        task_type: Override auto-detection (generation|refactor|debug|review|test|planning|auto)
                   Use "analyze_only" to return just intent analysis without enhancement.
                   Use "plan_validation" to validate a plan for implementation quality.
        context_level: How much context to gather (minimal|auto|comprehensive|none)
        max_tokens: Maximum token budget for the response (0=unlimited). When set, output
                    is automatically compressed: <=1000 tokens produces minimal output,
                    <=4000 produces compact output.
        model_tier: Target model tier for output optimization (auto|opus|sonnet|haiku).
                    Haiku/Sonnet receive more compressed output.
        session_id: Resume an existing session to thread validation feedback into this
                    prompt. Persistent violations from prior validate_code_quality calls
                    are injected as priority quality requirements.

    Returns:
        Enhanced prompt with quality requirements and tool recommendations
    """
    # Validate input size
    max_length = _MAX_PLAN_LENGTH if task_type == "plan_validation" else _MAX_PROMPT_LENGTH
    if error := _check_input_size(prompt, "prompt", max_length):
        return error

    c = _get_components()

    # --- Mode: analyze_only (replaces standalone analyze_intent tool) ---
    if task_type == "analyze_only":
        intent = c.intent_analyzer.analyze(prompt)
        ambiguity_level = (
            "low"
            if intent.ambiguity_score < 0.3
            else "medium"
            if intent.ambiguity_score < 0.6
            else "high"
        )
        return {
            "task_type": intent.task_type.value,
            "task_types": [t.value for t in intent.task_types],
            "language": intent.primary_language,
            "frameworks": intent.frameworks,
            "framework_versions": intent.framework_versions,
            "touches_security": intent.touches_security,
            "touches_rag": intent.touches_rag,
            "touches_knowledge_graph": intent.touches_knowledge_graph,
            "uses_external_framework": intent.uses_external_framework,
            "ambiguity_score": intent.ambiguity_score,
            "ambiguity_level": ambiguity_level,
            "extracted_entities": [e.to_dict() for e in intent.entities],
            "clarifying_questions": intent.clarifying_questions,
        }

    # --- Mode: plan_validation (replaces standalone validate_plan_quality tool) ---
    if task_type == "plan_validation":
        result = c.plan_validator.validate(prompt, "haiku")
        return result.to_dict()

    # --- Standard enhancement flow ---
    # Analyze intent
    _t0 = perf_counter()
    intent = c.intent_analyzer.analyze(prompt)

    # Override task type if specified — keep task_types consistent with the override
    if task_type != "auto":
        with contextlib.suppress(ValueError):
            overridden = TaskType(task_type)
            intent.task_type = overridden
            intent.task_types = [overridden] + [t for t in intent.task_types if t != overridden]

    # Load existing session for continuity or create new from intent.
    # When session_id is provided, persistent violation history is threaded
    # back into this prompt as priority quality requirements.
    if session_id and (existing := c.session_manager.get(session_id)):
        session = existing
    else:
        session = c.session_manager.create_from_intent(intent)

    # Compute persistent violation requirements from session history (Gap 1)
    persistent_reqs = _get_persistent_violation_reqs(session, c.session_tracker)

    # Get session-aware tool recommendations and store on session for compliance tracking
    tool_recommendations = c.mcp_orchestrator.suggest_tools(intent, session=session)
    session.tool_recommendations = [r.to_dict() for r in tool_recommendations]

    # Gather context from configured MCPs (skip if "none")
    if context_level == "none":
        from mirdan.models import ContextBundle

        context = ContextBundle()
    else:
        context = await c.context_aggregator.gather_all(intent, context_level)

    # Compose enhanced prompt with session feedback wired through
    enhanced = c.prompt_composer.compose(
        intent,
        context,
        tool_recommendations,
        extra_requirements=persistent_reqs,
        session=session,
    )

    result_dict = enhanced.to_dict()
    result_dict["session_id"] = session.session_id

    # Add knowledge entries from intent analysis
    knowledge_entries = c.knowledge_producer.extract_from_intent(
        task_type=intent.task_type.value,
        language=intent.primary_language,
        frameworks=intent.frameworks,
    )
    if knowledge_entries:
        result_dict["knowledge_entries"] = _process_knowledge_entries(knowledge_entries, c)

    # Detect environment for context
    env_info = detect_environment()
    result_dict["environment"] = env_info.to_dict()

    result_dict["timing_ms"] = {"total": round((perf_counter() - _t0) * 1000, 1)}

    # Apply token-budget-aware formatting
    tier = _parse_model_tier(model_tier)
    result_dict = c.output_formatter.format_enhanced_prompt(
        result_dict, max_tokens=max_tokens, model_tier=tier
    )

    return result_dict


# ---------------------------------------------------------------------------
# Core Tool 2: validate_code_quality
# ---------------------------------------------------------------------------


@mcp.tool()
async def validate_code_quality(
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
) -> dict[str, Any]:
    """
    Validate generated code against quality standards.

    Args:
        code: The code to validate
        language: Programming language (python|typescript|javascript|rust|go|auto)
        check_security: Validate against security standards
        check_architecture: Validate against architecture standards
        check_style: Validate against language-specific style standards
        severity_threshold: Minimum severity to include in results (error|warning|info)
        session_id: Session ID from enhance_prompt to auto-inherit language and security settings
        max_tokens: Maximum token budget for the response (0=unlimited)
        model_tier: Target model tier for output optimization (auto|opus|sonnet|haiku)
        input_type: Input type - "code" for raw code (default), "diff" for unified
                   diff format (git diff output). When "diff", only added lines are
                   validated and violation line numbers map back to original file locations
        compare: If True, treat `code` as JSON array of implementations to compare
        file_path: Optional file path for external linter analysis. When provided,
                   runs ruff/eslint/mypy on the file and merges results.

    Returns:
        Validation results with pass/fail, score, violations, and summary
    """
    # --- Mode: compare (replaces standalone compare_approaches tool) ---
    if compare:
        return await _handle_compare(code, language)

    # --- Mode: diff (replaces standalone validate_diff tool) ---
    if input_type == "diff":
        return await _handle_diff(
            code, language, check_security, session_id, max_tokens, model_tier
        )

    # --- Standard code validation ---
    # Validate input size
    if error := _check_input_size(code, "code", _MAX_CODE_LENGTH):
        return error

    c = _get_components()

    # Apply session defaults if available
    resolved_language, resolved_security = c.session_manager.apply_session_defaults(
        session_id, language=language, check_security=check_security
    )

    # Resolve per-file threshold overrides
    resolved_thresholds = None
    if file_path and c.config.thresholds.file_overrides:
        resolved_thresholds = c.config.thresholds.resolve_for_file(file_path)

    _t0 = perf_counter()
    result = c.code_validator.validate(
        code=code,
        language=resolved_language,
        check_security=resolved_security,
        check_architecture=check_architecture,
        check_style=check_style,
        thresholds=resolved_thresholds,
    )
    _t_validate = perf_counter() - _t0

    # Run external linters if file_path provided
    if file_path:
        fp = Path(file_path)
        if fp.exists():
            linter_violations = await c.linter_runner.run(fp, result.language_detected)
            if linter_violations:
                result = merge_linter_violations(result, linter_violations, c.config.thresholds)

    # Enrich violations with contextual explanations (Phase 5A)
    if result.violations:
        try:
            c.violation_explainer.enrich_violations(result.violations)
        except Exception:
            logger.debug("Failed to enrich violations", exc_info=True)

    # Persist snapshot for quality trends (Bug #1: save_snapshot was never called)
    try:
        c.quality_persistence.save_snapshot(result)
    except Exception:
        logger.debug("Failed to save quality snapshot", exc_info=True)

    # Track validation in session (Phase 2: always-on tracking)
    session = None
    if session_id:
        session = c.session_manager.get(session_id)
    c.session_tracker.record_validation(result, file_path=file_path, session=session)

    # Detect security regression (must be AFTER record_validation — uses records[-2])
    _security_regression = c.session_tracker.detect_security_regression(
        file_path, result.violations
    )

    # Compute violation delta for re-validations within a session
    delta: dict[str, Any] = {}
    if session and session.validation_count > 1:
        key = file_path or ""
        prev_violations = c.session_tracker.get_previous_violations(key)
        curr_violations = {v.id for v in result.violations}
        resolved = prev_violations - curr_violations
        new_viol = curr_violations - prev_violations
        persistence = c.session_tracker.get_violation_persistence(key)
        if resolved:
            delta["resolved"] = sorted(resolved)
        if new_viol:
            delta["new"] = sorted(new_viol)
        persistent = {r: cnt for r, cnt in persistence.items() if cnt > 1}
        if persistent:
            delta["persistent"] = persistent

    output = result.to_dict(severity_threshold=severity_threshold)

    # Cross-session quality drift detection
    baseline = c.quality_persistence.get_baseline_score()
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
            "security_violations": [v.id for v in result.violations if v.category == "security"],
        }

    # Generate semantic review questions (Layer 1)
    if c.config.semantic.enabled:
        semantic_checks = c.code_validator.generate_semantic_checks(
            code=code,
            language=result.language_detected,
            violations=result.violations,
        )
        if semantic_checks:
            output["semantic_checks"] = [s.to_dict() for s in semantic_checks]

        # Layer 3: Analysis protocol for security-critical code
        if resolved_security and c.config.semantic.analysis_protocol != "none":
            protocol = c.semantic_analyzer.generate_analysis_protocol(
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
    knowledge_entries = c.knowledge_producer.extract_from_validation(result)
    if knowledge_entries:
        output["knowledge_entries"] = _process_knowledge_entries(knowledge_entries, c)

    # Add verification checklist to output (absorbs get_verification_checklist)
    intent = Intent(
        original_prompt="",
        task_type=TaskType.UNKNOWN,
        touches_security=resolved_security,
    )
    # Detect task type from session if available
    if session:
        intent.task_type = session.task_type
    output["checklist"] = c.prompt_composer.generate_verification_steps(intent)
    if session and session.validation_count > 1:
        output["checklist_note"] = "Re-validation. Review session_context.resolved for progress."

    output["timing_ms"] = {
        "validation": round(_t_validate * 1000, 1),
        "total": round((perf_counter() - _t0) * 1000, 1),
    }

    # Apply token-budget-aware formatting
    tier = _parse_model_tier(model_tier)
    output = c.output_formatter.format_validation_result(
        output, max_tokens=max_tokens, model_tier=tier
    )

    return output


async def _handle_diff(
    diff: str,
    language: str,
    check_security: bool,
    session_id: str,
    max_tokens: int,
    model_tier: str,
) -> dict[str, Any]:
    """Handle diff validation (replaces standalone validate_diff tool)."""
    if error := _check_input_size(diff, "diff", _MAX_CODE_LENGTH):
        return error

    c = _get_components()

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
    resolved_language, resolved_security = c.session_manager.apply_session_defaults(
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
                file_result = c.code_validator.validate(
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
            fallback_result = c.code_validator.validate(
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
        result = c.code_validator.validate(
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
            score = c.code_validator._calculate_score(all_violations)

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
        c.quality_persistence.save_snapshot(result)
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
    output = c.output_formatter.format_validation_result(
        output, max_tokens=max_tokens, model_tier=tier
    )

    return output


async def _handle_compare(
    code: str,
    language: str,
) -> dict[str, Any]:
    """Handle multi-implementation comparison (replaces standalone compare_approaches tool)."""
    try:
        implementations = json.loads(code)
    except (json.JSONDecodeError, TypeError):
        return {"error": "When compare=True, code must be a JSON array of implementation strings"}

    if not isinstance(implementations, list):
        return {"error": "When compare=True, code must be a JSON array of implementation strings"}

    if len(implementations) < 2:
        return {"error": "At least 2 implementations are required for comparison"}
    if len(implementations) > 10:
        return {"error": "Maximum 10 implementations can be compared at once"}

    for i, impl in enumerate(implementations):
        if not isinstance(impl, str):
            return {"error": f"implementation[{i}] must be a string"}
        if error := _check_input_size(impl, f"implementation[{i}]", _MAX_CODE_LENGTH):
            return error

    c = _get_components()

    labels = [f"Implementation {i + 1}" for i in range(len(implementations))]

    entries: list[ComparisonEntry] = []
    for impl, label in zip(implementations, labels, strict=True):
        result = c.code_validator.validate(
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


# ---------------------------------------------------------------------------
# Core Tool 2b: validate_quick
# ---------------------------------------------------------------------------


@mcp.tool()
async def validate_quick(
    code: str,
    language: str = "auto",
    max_tokens: int = 0,
    model_tier: str = "auto",
) -> dict[str, Any]:
    """Fast security-only validation for hooks and real-time feedback (<500ms target).

    Runs only security rules — skips style, architecture, framework, and custom checks.
    Ideal for PostToolUse hooks where speed matters more than comprehensive analysis.

    Args:
        code: The code to validate
        language: Programming language (python|typescript|javascript|rust|go|auto)
        max_tokens: Maximum token budget for the response (0=unlimited)
        model_tier: Target model tier for output optimization (auto|opus|sonnet|haiku)

    Returns:
        Validation results with pass/fail, score, and security-only violations
    """
    if error := _check_input_size(code, "code", _MAX_CODE_LENGTH):
        return error

    c = _get_components()

    result = c.code_validator.validate_quick(code=code, language=language)

    output = result.to_dict(severity_threshold="warning")

    # Add auto-fix suggestions for high-confidence security/critical fixes
    auto_fixes = c.auto_fixer.quick_fix(result)
    if auto_fixes:
        output["auto_fixes"] = [
            {
                "fix_code": f.fix_code,
                "fix_description": f.fix_description,
                "confidence": f.confidence,
            }
            for f in auto_fixes
        ]

    # Apply token-budget-aware formatting
    tier = _parse_model_tier(model_tier)
    output = c.output_formatter.format_validation_result(
        output, max_tokens=max_tokens, model_tier=tier
    )

    return output


# ---------------------------------------------------------------------------
# Core Tool 3: get_quality_standards
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_quality_standards(
    language: str,
    framework: str = "",
    category: str = "all",
) -> dict[str, Any]:
    """
    Retrieve quality standards for a language/framework combination.

    Args:
        language: Programming language (typescript, python, etc.)
        framework: Optional framework (react, fastapi, etc.)
        category: Filter to specific category (security|architecture|style|all)

    Returns:
        Quality standards for the specified language/framework
    """
    c = _get_components()
    return c.quality_standards.get_all_standards(
        language=language, framework=framework, category=category
    )


# ---------------------------------------------------------------------------
# Core Tool 4: get_quality_trends
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_quality_trends(
    project_path: str = "",
    days: int = 30,
    format: str = "",
) -> dict[str, Any]:
    """
    Get quality score trends over time from stored validation history.

    Reads snapshots from `.mirdan/history/` and calculates aggregate
    statistics including average score, pass rate, and trend direction.

    Args:
        project_path: Optional project path filter
        days: Number of days of history to analyze (default: 30)
        format: Output format — empty for default, "dashboard" for MCP Apps data

    Returns:
        Quality trend data with scores, pass rate, and trend direction
    """
    if days < 1:
        return {"error": "days must be at least 1"}
    if days > 365:
        return {"error": "days cannot exceed 365"}

    c = _get_components()
    trend = c.quality_persistence.get_trends(
        days=days,
        project_path=project_path or None,
    )
    trend_dict = trend.to_dict()

    # Add quality forecasting (Phase 5D)
    if trend.snapshots:
        from mirdan.core.quality_forecaster import QualityForecaster

        forecaster = QualityForecaster()
        snap_dicts = [s.to_dict() for s in trend.snapshots]
        forecast = forecaster.forecast(snap_dicts)
        trend_dict["forecast"] = forecast.to_dict()
        regressions = forecaster.detect_regression(snap_dicts)
        if regressions:
            trend_dict["regression_alerts"] = [r.to_dict() for r in regressions]
        trend_dict["velocity"] = round(forecaster.calculate_velocity(snap_dicts), 6)

    if format == "dashboard":
        from mirdan.integrations.mcp_apps import QualityDashboard

        dashboard = QualityDashboard()
        trend_dict["dashboard"] = dashboard.score_timeline(trend_dict)

    return trend_dict


# ---------------------------------------------------------------------------
# Core Tool 5: scan_conventions
# ---------------------------------------------------------------------------


@mcp.tool()
async def scan_conventions(
    directory: str = ".",
    language: str = "auto",
) -> dict[str, Any]:
    """Scan a codebase to discover implicit conventions and patterns.

    Validates multiple source files, aggregates results, and produces
    convention entries describing naming patterns, import styles,
    docstring conventions, and recurring violation patterns.

    Args:
        directory: Directory to scan (default: current directory)
        language: Language filter or "auto" to detect

    Returns:
        Scan result with discovered conventions and quality baselines
    """
    c = _get_components()
    scan_dir = Path(directory).resolve()

    if not scan_dir.is_dir():
        return {"error": f"Not a directory: {directory}"}

    result = c.convention_extractor.scan(scan_dir, language=language)

    # Persist conventions for quality standards feedback loop
    try:
        conventions_dir = Path.cwd() / ".mirdan"
        conventions_dir.mkdir(parents=True, exist_ok=True)
        conventions_path = conventions_dir / "conventions.yaml"
        conventions_data = {
            "conventions": [e.to_dict() for e in result.conventions],
            "language": result.language,
        }
        with conventions_path.open("w") as f:
            yaml.dump(conventions_data, f, default_flow_style=False, allow_unicode=True)
    except Exception:
        logger.debug("Failed to persist conventions", exc_info=True)

    return result.to_dict()


# ---------------------------------------------------------------------------
# Core Tool 6: scan_dependencies
# ---------------------------------------------------------------------------


@mcp.tool()
async def scan_dependencies(
    project_path: str = ".",
    ecosystem: str = "auto",
) -> dict[str, Any]:
    """Scan project dependencies for known vulnerabilities.

    Queries the OSV database (free, no API key) to check all dependencies
    against known CVEs. Results are cached per config (default: 24 hours).

    Args:
        project_path: Project directory containing dependency manifests
        ecosystem: Filter by ecosystem (auto|PyPI|npm|crates.io|Go|Maven)

    Returns:
        Scan results with packages checked and vulnerabilities found
    """
    c = _get_components()
    scan_dir = Path(project_path).resolve()

    if not scan_dir.is_dir():
        return {"error": f"Not a directory: {project_path}"}

    packages = c.manifest_parser.parse(scan_dir)
    if ecosystem != "auto":
        packages = [p for p in packages if p.ecosystem == ecosystem]

    if not packages:
        return {
            "packages_scanned": 0,
            "vulnerabilities_found": 0,
            "message": "No dependency manifests found",
            "findings": [],
        }

    findings = await c.vuln_scanner.scan(packages)

    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for f in findings:
        severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

    return {
        "packages_scanned": len(packages),
        "vulnerabilities_found": len(findings),
        "severity_counts": severity_counts,
        "findings": [f.to_dict() for f in findings],
        "ecosystems_checked": list({p.ecosystem for p in packages}),
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _process_knowledge_entries(
    entries: list[KnowledgeEntry],
    components: _Components,
) -> list[dict[str, Any]]:
    """Serialize knowledge entries, flag for client, and schedule server-side storage.

    When auto_memory is False (default): sets auto_store=True on entries above
    the confidence threshold, signaling the client to store them via enyal.

    When auto_memory is True: schedules fire-and-forget server-side storage
    via ActiveOrchestrator and does NOT set auto_store (preventing double-storage).
    """
    config = components.config.orchestration
    entries_out = []
    for e in entries:
        d = e.to_dict()
        if not config.auto_memory and e.confidence >= config.auto_memory_threshold:
            d["auto_store"] = True
        entries_out.append(d)

    if config.auto_memory:
        coro = _auto_store_knowledge(
            components.active_orchestrator,
            entries,
            config.auto_memory_threshold,
        )
        task = asyncio.create_task(coro)
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    return entries_out


async def _auto_store_knowledge(
    orchestrator: ActiveOrchestrator,
    entries: list[KnowledgeEntry],
    threshold: float,
) -> None:
    """Fire-and-forget wrapper for auto-memory storage.

    Prevents 'Task exception was never retrieved' warnings from
    asyncio.create_task by catching and logging any errors.
    """
    try:
        await orchestrator.store_knowledge(entries, threshold)
    except Exception:
        logger.debug("Auto-memory storage failed", exc_info=True)


def _parse_model_tier(tier: str) -> ModelTier:
    """Parse a model tier string into the enum, defaulting to AUTO."""
    try:
        return ModelTier(tier.lower())
    except ValueError:
        return ModelTier.AUTO


def main() -> None:
    """Run the Mirdan MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
