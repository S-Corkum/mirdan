"""EnhancePrompt use case — extracted from server.py."""

from __future__ import annotations

import contextlib
import logging
from time import perf_counter
from typing import TYPE_CHECKING, Any

from mirdan.usecases.helpers import (
    _MAX_PLAN_LENGTH,
    _MAX_PROMPT_LENGTH,
    _check_input_size,
    _parse_model_tier,
    _process_knowledge_entries,
)

if TYPE_CHECKING:
    import asyncio

    from mirdan.config import MirdanConfig
    from mirdan.core.active_orchestrator import ToolExecutor
    from mirdan.core.agent_coordinator import AgentCoordinator
    from mirdan.core.ceremony import CeremonyAdvisor
    from mirdan.core.context_aggregator import ContextAggregator
    from mirdan.core.intent_analyzer import IntentAnalyzer
    from mirdan.core.knowledge_producer import KnowledgeProducer
    from mirdan.core.orchestrator import ToolAdvisor
    from mirdan.core.output_formatter import OutputFormatter
    from mirdan.core.plan_validator import PlanValidator
    from mirdan.core.prompt_composer import PromptComposer
    from mirdan.core.session_manager import SessionManager
    from mirdan.core.session_tracker import SessionTracker
    from mirdan.models import SessionContext

logger = logging.getLogger(__name__)


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


class EnhancePromptUseCase:
    """Enhance a coding prompt with quality requirements, context, and tool recommendations."""

    def __init__(
        self,
        intent_analyzer: IntentAnalyzer,
        session_manager: SessionManager,
        mcp_orchestrator: ToolAdvisor,
        context_aggregator: ContextAggregator,
        prompt_composer: PromptComposer,
        knowledge_producer: KnowledgeProducer,
        output_formatter: OutputFormatter,
        session_tracker: SessionTracker,
        plan_validator: PlanValidator,
        config: MirdanConfig,
        active_orchestrator: ToolExecutor,
        background_tasks: set[asyncio.Task[Any]],
        ceremony_advisor: CeremonyAdvisor | None = None,
        agent_coordinator: AgentCoordinator | None = None,
        analyzer_suite: Any = None,
        llm_manager: Any = None,
        training_collector: Any = None,
    ) -> None:
        self._intent_analyzer = intent_analyzer
        self._session_manager = session_manager
        self._mcp_orchestrator = mcp_orchestrator
        self._context_aggregator = context_aggregator
        self._prompt_composer = prompt_composer
        self._knowledge_producer = knowledge_producer
        self._output_formatter = output_formatter
        self._session_tracker = session_tracker
        self._plan_validator = plan_validator
        self._config = config
        self._active_orchestrator = active_orchestrator
        self._background_tasks = background_tasks
        self._ceremony_advisor = ceremony_advisor
        self._agent_coordinator = agent_coordinator
        self._analyzers = analyzer_suite
        self._llm_manager = llm_manager
        self._training_collector = training_collector

    async def execute(
        self,
        prompt: str,
        task_type: str = "auto",
        context_level: str = "auto",
        max_tokens: int = 0,
        model_tier: str = "auto",
        session_id: str = "",
        ceremony_level: str = "auto",
    ) -> dict[str, Any]:
        """Execute the enhance_prompt use case.

        Args:
            prompt: The original developer prompt
            task_type: Override auto-detection (generation|refactor|debug|review|test|planning|auto)
                       Use "analyze_only" to return just intent analysis without enhancement.
                       Use "plan_validation" to validate a plan for implementation quality.
            context_level: How much context to gather (minimal|auto|comprehensive|none)
            max_tokens: Maximum token budget for the response (0=unlimited).
            model_tier: Target model tier for output optimization (auto|opus|sonnet|haiku).
            session_id: Resume an existing session to thread validation feedback into this
                        prompt.
            ceremony_level: Guidance depth (auto|micro|light|standard|thorough).
                           "auto" scales based on task complexity.

        Returns:
            Enhanced prompt with quality requirements and tool recommendations
        """
        from mirdan.core.environment_detector import detect_environment
        from mirdan.models import CeremonyLevel, ContextBundle, TaskType

        # Validate input size
        max_length = _MAX_PLAN_LENGTH if task_type == "plan_validation" else _MAX_PROMPT_LENGTH
        if error := _check_input_size(prompt, "prompt", max_length):
            return error

        # --- Mode: analyze_only (replaces standalone analyze_intent tool) ---
        if task_type == "analyze_only":
            intent = self._intent_analyzer.analyze(prompt)
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
            result = self._plan_validator.validate(prompt, "haiku")
            return result.to_dict()

        # --- Standard enhancement flow ---
        # Analyze intent
        _t0 = perf_counter()
        intent = self._intent_analyzer.analyze(prompt)

        # Override task type if specified — keep task_types consistent with the override
        if task_type != "auto":
            with contextlib.suppress(ValueError):
                overridden = TaskType(task_type)
                intent.task_type = overridden
                intent.task_types = [overridden] + [t for t in intent.task_types if t != overridden]

        # Load existing session for continuity or create new from intent.
        # When session_id is provided, persistent violation history is threaded
        # back into this prompt as priority quality requirements.
        if session_id and (existing := self._session_manager.get(session_id)):
            session = existing
        else:
            session = self._session_manager.create_from_intent(intent)

        # Auto-claim files from extracted entities for coordination
        coordination_warnings: list[dict[str, Any]] = []
        if self._agent_coordinator is not None and self._agent_coordinator.is_enabled:
            from mirdan.models import EntityType

            file_entities = [e.value for e in intent.entities if e.type == EntityType.FILE_PATH]
            if file_entities:
                claim_type = (
                    "write"
                    if intent.task_type in (TaskType.GENERATION, TaskType.REFACTOR)
                    else "read"
                )
                warnings = self._agent_coordinator.claim_files(
                    session.session_id, file_entities, claim_type
                )
                coordination_warnings = [w.to_dict() for w in warnings]

        # Triage gate: classify task before ceremony to save paid tokens
        triage_result = None
        if self._llm_manager is not None:
            triage_result = await self._run_triage(prompt, intent, session_id)
            if triage_result is not None and self._training_collector is not None:
                self._training_collector.record_triage_sample(
                    prompt=prompt,
                    intent_summary=intent.task_type.value,
                    classification=triage_result.classification.value,
                    confidence=triage_result.confidence,
                )
            if triage_result is not None:
                from mirdan.models import TaskClassification

                # LOCAL_ONLY with high confidence: short-circuit
                if (
                    triage_result.classification == TaskClassification.LOCAL_ONLY
                    and triage_result.confidence >= 0.8
                ):
                    return self._output_formatter.format_enhanced_prompt(
                        {
                            "task_type": intent.task_type.value,
                            "language": intent.primary_language,
                            "triage": triage_result.to_dict(),
                            "ceremony_level": "micro",
                            "recommendation": "Handle locally — no paid model needed",
                            "timing_ms": {"total": round((perf_counter() - _t0) * 1000, 1)},
                        },
                        max_tokens=max_tokens,
                        model_tier=_parse_model_tier(model_tier),
                    )

                # Override ceremony level based on triage
                from mirdan.core.triage import TriageEngine

                ceremony_override = TriageEngine.get_ceremony_override(triage_result.classification)
                if ceremony_override and ceremony_level == "auto":
                    ceremony_level = ceremony_override

        # Determine ceremony level (after session is available for escalation checks)
        if self._ceremony_advisor is not None:
            if ceremony_level != "auto":
                # Explicit override — bypasses scoring and escalation (user knows best)
                try:
                    level = CeremonyLevel[ceremony_level.upper()]
                except KeyError:
                    level = CeremonyLevel.STANDARD
                policy = self._ceremony_advisor.get_policy(level)
            else:
                level = self._ceremony_advisor.determine_level(intent, len(prompt), session=session)
                policy = self._ceremony_advisor.get_policy(level)

            # MICRO: return minimal analysis with ceremony metadata
            if policy.enhancement_mode == "analyze_only":
                intent_result = {
                    "task_type": intent.task_type.value,
                    "task_types": [t.value for t in intent.task_types],
                    "language": intent.primary_language,
                    "frameworks": intent.frameworks,
                    "touches_security": intent.touches_security,
                    "ceremony_level": level.name.lower(),
                    "recommended_validation": policy.recommended_validation,
                    "ceremony_reason": self._ceremony_advisor.explain(level, intent),
                    "timing_ms": {"total": round((perf_counter() - _t0) * 1000, 1)},
                }
                return self._output_formatter.format_enhanced_prompt(
                    intent_result, max_tokens=max_tokens, model_tier=_parse_model_tier(model_tier)
                )

            # Apply ceremony policy to context_level (primary overhead reduction)
            if context_level == "auto":
                effective_context_level = policy.context_level
            else:
                effective_context_level = context_level
        else:
            level = CeremonyLevel.STANDARD
            policy = None
            effective_context_level = context_level

        # Ceremony-gated analyzers — only at STANDARD+ ceremony
        tidy_analysis = None
        decision_guidance = None
        guardrail_analysis = None
        arch_context = None
        if self._analyzers and level >= CeremonyLevel.STANDARD:
            if self._analyzers.tidy_first and intent.task_type in (
                TaskType.GENERATION,
                TaskType.REFACTOR,
            ):
                tidy_analysis = self._analyzers.tidy_first.analyze(intent)
            if self._analyzers.decision:
                decision_guidance = self._analyzers.decision.analyze(intent)
            if self._analyzers.guardrail:
                guardrail_analysis = self._analyzers.guardrail.analyze(intent)
            if self._analyzers.architecture:
                arch_context = self._analyzers.architecture.get_context_warnings(intent)

        # Compute persistent violation requirements from session history
        persistent_reqs = _get_persistent_violation_reqs(session, self._session_tracker)

        # Get session-aware tool recommendations and store on session for compliance tracking
        tool_recommendations = self._mcp_orchestrator.suggest_tools(intent, session=session)
        session.tool_recommendations = [r.to_dict() for r in tool_recommendations]

        # LIGHT ceremony: filter tool recommendations to critical-priority only
        # (session already stores unfiltered recs above for compliance tracking)
        if policy is not None and policy.filter_tool_recs:
            tool_recommendations = [r for r in tool_recommendations if r.priority == "critical"]

        # Gather context from configured MCPs (skip if "none")
        if effective_context_level == "none":
            context = ContextBundle()
        else:
            # Research agent: at THOROUGH ceremony, try BRAIN-powered research first
            research_used = False
            if self._llm_manager is not None and effective_context_level == "comprehensive":
                try:
                    from mirdan.core.research_agent import ResearchAgent

                    agent = ResearchAgent(
                        llm_manager=self._llm_manager,
                        registry=self._context_aggregator.registry,
                        config=getattr(self._llm_manager, "_config", None),
                    )
                    research = await agent.research(intent, tool_recommendations)
                    if research and research.synthesis:
                        context = ContextBundle(
                            documentation_hints=[research.synthesis],
                            existing_patterns=[s.get("summary", "") for s in research.sources],
                        )
                        research_used = True
                except Exception:
                    logger.warning("Research agent failed, falling back to ContextAggregator")

            if not research_used:
                context = await self._context_aggregator.gather_all(intent, effective_context_level)

        # Prompt optimization (BRAIN model, FULL profile only)
        if self._llm_manager is not None and hasattr(self._llm_manager, "_config"):
            try:
                from mirdan.core.prompt_optimizer import PromptOptimizer

                optimizer = PromptOptimizer(
                    llm_manager=self._llm_manager,
                    config=self._llm_manager._config,
                )
                optimized = await optimizer.optimize(
                    task_description=prompt,
                    context_items=context.existing_patterns + context.documentation_hints,
                    tool_recommendations="\n".join(
                        r.to_dict().get("action", "") for r in tool_recommendations
                    ),
                    quality_requirements="\n".join(persistent_reqs),
                    target_model=model_tier if model_tier != "auto" else "sonnet",
                    detected_ide=detect_environment().ide.value,
                )
                if optimized:
                    context.documentation_hints = [optimized.text]
                    if self._training_collector is not None:
                        self._training_collector.record_optimization_sample(
                            original_prompt=prompt,
                            optimized_prompt=optimized.text,
                            target_model=model_tier if model_tier != "auto" else "sonnet",
                        )
            except Exception:
                logger.warning("Prompt optimization failed, using original context")

        # Compose enhanced prompt with session feedback wired through
        enhanced = self._prompt_composer.compose(
            intent,
            context,
            tool_recommendations,
            extra_requirements=persistent_reqs,
            session=session,
            tidy_suggestions=(
                [s.to_dict() for s in tidy_analysis.suggestions]
                if tidy_analysis and tidy_analysis.suggestions
                else None
            ),
        )

        result_dict = enhanced.to_dict()
        result_dict["session_id"] = session.session_id

        # Add knowledge entries from intent analysis
        knowledge_entries = self._knowledge_producer.extract_from_intent(
            task_type=intent.task_type.value,
            language=intent.primary_language,
            frameworks=intent.frameworks,
        )
        if knowledge_entries:
            result_dict["knowledge_entries"] = _process_knowledge_entries(
                knowledge_entries,
                self._config,
                self._active_orchestrator,
                self._background_tasks,
            )

        # Detect environment for context
        env_info = detect_environment()
        result_dict["environment"] = env_info.to_dict()

        # Add ceremony metadata
        result_dict["ceremony_level"] = level.name.lower()
        if policy is not None and self._ceremony_advisor is not None:
            result_dict["recommended_validation"] = policy.recommended_validation
            result_dict["ceremony_reason"] = self._ceremony_advisor.explain(level, intent)

        if coordination_warnings:
            result_dict["coordination"] = {"warnings": coordination_warnings}

        if tidy_analysis is not None and tidy_analysis.suggestions:
            result_dict["tidy_suggestions"] = tidy_analysis.to_dict()

        if decision_guidance:
            result_dict["decision_guidance"] = [d.to_dict() for d in decision_guidance]

        if guardrail_analysis:
            result_dict["cognitive_guardrails"] = [g.to_dict() for g in guardrail_analysis]

        if arch_context:
            result_dict["architecture_context"] = arch_context

        result_dict["timing_ms"] = {"total": round((perf_counter() - _t0) * 1000, 1)}

        # Apply token-budget-aware formatting
        tier = _parse_model_tier(model_tier)
        result_dict = self._output_formatter.format_enhanced_prompt(
            result_dict, max_tokens=max_tokens, model_tier=tier
        )

        return result_dict

    async def _run_triage(self, prompt: str, intent: Any, session_id: str) -> Any:
        """Run triage classification, checking session bridge cache first.

        Args:
            prompt: Developer's task description.
            intent: Analyzed intent.
            session_id: Current session ID for cache lookup.

        Returns:
            TriageResult or None.
        """
        from mirdan.coordination.session_bridge import get_session_id, read_triage
        from mirdan.core.triage import TriageEngine
        from mirdan.models import TaskClassification, TriageResult

        # Check session bridge cache (hook may have already triaged)
        effective_session = session_id or get_session_id()
        cached = read_triage(effective_session)
        if cached and "classification" in cached:
            try:
                return TriageResult(
                    classification=TaskClassification(cached["classification"]),
                    confidence=cached.get("confidence", 0.0),
                    reasoning=cached.get("reasoning", "cached"),
                )
            except ValueError:
                pass

        # Run triage via engine
        engine = TriageEngine(llm_manager=self._llm_manager)
        return await engine.classify(prompt, intent)
