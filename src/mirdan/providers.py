"""Structured component provider for Mirdan MCP server."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from mirdan.config import MirdanConfig
from mirdan.core.active_orchestrator import ToolExecutor
from mirdan.core.analyzer_suite import AnalyzerSuite
from mirdan.llm.manager import LLMManager
from mirdan.core.agent_coordinator import AgentCoordinator
from mirdan.core.architecture_analyzer import ArchitectureAnalyzer
from mirdan.core.auto_fixer import AutoFixer
from mirdan.core.ceremony import CeremonyAdvisor
from mirdan.core.code_validator import CodeValidator
from mirdan.core.confidence import ConfidenceCalibrator
from mirdan.core.context_aggregator import ContextAggregator
from mirdan.core.convention_extractor import ConventionExtractor
from mirdan.core.decision_analyzer import DecisionAnalyzer
from mirdan.core.environment_detector import IDEType, detect_environment
from mirdan.core.guardrail_analyzer import GuardrailAnalyzer
from mirdan.core.intent_analyzer import IntentAnalyzer
from mirdan.core.knowledge_producer import KnowledgeProducer
from mirdan.core.linter_orchestrator import create_linter_runner
from mirdan.core.linter_runner import LinterRunner
from mirdan.core.manifest_parser import ManifestParser
from mirdan.core.orchestrator import ToolAdvisor
from mirdan.core.output_formatter import OutputFormatter
from mirdan.core.plan_validator import PlanValidator
from mirdan.core.prompt_composer import PromptComposer
from mirdan.core.quality_persistence import QualityPersistence
from mirdan.core.quality_standards import QualityStandards
from mirdan.core.semantic_analyzer import SemanticAnalyzer
from mirdan.core.session_manager import SessionManager
from mirdan.core.session_tracker import SessionTracker
from mirdan.core.tidy_first import TidyFirstAnalyzer
from mirdan.core.vuln_scanner import VulnScanner
from mirdan.usecases.enhance_prompt import EnhancePromptUseCase
from mirdan.usecases.quality_standards import GetQualityStandardsUseCase
from mirdan.usecases.quality_trends import GetQualityTrendsUseCase
from mirdan.usecases.scan_conventions import ScanConventionsUseCase
from mirdan.usecases.scan_dependencies import ScanDependenciesUseCase
from mirdan.usecases.validate_code import ValidateCodeUseCase
from mirdan.usecases.validate_quick import ValidateQuickUseCase

logger = logging.getLogger(__name__)


def _create_violation_explainer() -> Any:
    """Create the ViolationExplainer instance."""
    from mirdan.core.violation_explainer import ViolationExplainer

    return ViolationExplainer()


class ComponentProvider:
    """Structured component container with use-case factories.

    Replaces the flat _Components dataclass with a class that owns
    component initialization and exposes factory methods for each use case.
    """

    def __init__(self, config: MirdanConfig | None = None) -> None:
        if config is None:
            config = MirdanConfig.find_config()

        if config.quality_profile and config.quality_profile != "default":
            from mirdan.core.quality_profiles import apply_profile, get_profile

            try:
                profile = get_profile(config.quality_profile, config.custom_profiles)
                config_dict = {"quality": config.quality.__dict__.copy()}
                apply_profile(profile, config_dict)
                for key, value in config_dict.get("quality", {}).items():
                    if hasattr(config.quality, key):
                        setattr(config.quality, key, value)
            except ValueError:
                logger.warning(
                    "Unknown quality profile '%s', using default", config.quality_profile
                )

        project_dir = Path.cwd()

        workspace_resolver = None
        if config.is_workspace:
            from mirdan.core.workspace_resolver import WorkspaceResolver

            workspace_resolver = WorkspaceResolver(config, project_dir)

        manifest_parser = ManifestParser(project_dir=project_dir)

        quality_standards = QualityStandards(
            config=config.quality, project_dir=project_dir, manifest_parser=manifest_parser
        )

        env_info = detect_environment()
        compact_threshold = config.tokens.compact_threshold
        minimal_threshold = config.tokens.minimal_threshold
        micro_threshold = config.tokens.micro_threshold

        if env_info.ide == IDEType.CLAUDE_CODE:
            compact_threshold = max(compact_threshold, 8000)
        elif env_info.ide == IDEType.CURSOR:
            compact_threshold = min(compact_threshold, 2000)

        cache_dir = project_dir / ".mirdan" / "cache"
        vuln_scanner = VulnScanner(cache_dir=cache_dir, ttl=config.dependencies.osv_cache_ttl)
        semantic_analyzer = SemanticAnalyzer(config=config.semantic)
        context_aggregator = ContextAggregator(config)

        # AgentCoordinator must be created BEFORE SessionManager (Step 15 passes it)
        self.agent_coordinator = AgentCoordinator(config.coordination)
        self.tidy_analyzer = TidyFirstAnalyzer(config.tidy_first)
        self.decision_analyzer = DecisionAnalyzer(config.decisions)
        self.guardrail_analyzer = GuardrailAnalyzer(config.guardrails)
        self.architecture_analyzer = ArchitectureAnalyzer(config.architecture)
        self.architecture_analyzer.load_model(project_dir)
        self.analyzer_suite = AnalyzerSuite(
            tidy_first=self.tidy_analyzer,
            decision=self.decision_analyzer,
            guardrail=self.guardrail_analyzer,
            architecture=self.architecture_analyzer,
        )

        # Store all components as instance attributes
        self.intent_analyzer = IntentAnalyzer(config.project, manifest_parser=manifest_parser)
        self.quality_standards = quality_standards
        self.prompt_composer = PromptComposer(quality_standards, config=config.enhancement)
        self.mcp_orchestrator = ToolAdvisor(config.orchestration)
        self.context_aggregator = context_aggregator
        self.code_validator = CodeValidator(
            quality_standards,
            config=config.quality,
            thresholds=config.thresholds,
            project_dir=project_dir,
            semantic_analyzer=semantic_analyzer,
            manifest_parser=manifest_parser,
            vuln_scanner=vuln_scanner,
            workspace_resolver=workspace_resolver,
        )
        self.plan_validator = PlanValidator(config.planning, thresholds=config.thresholds)
        self.session_manager = SessionManager(config.session, coordinator=self.agent_coordinator)
        self.output_formatter = OutputFormatter(
            compact_threshold=compact_threshold,
            minimal_threshold=minimal_threshold,
            micro_threshold=micro_threshold,
        )
        self.quality_persistence = QualityPersistence()
        self.knowledge_producer = KnowledgeProducer()
        self.linter_runner: LinterRunner = create_linter_runner(config)
        self.session_tracker = SessionTracker()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.auto_fixer = AutoFixer()
        self.convention_extractor = ConventionExtractor(config)
        self.violation_explainer = _create_violation_explainer()
        self.manifest_parser = manifest_parser
        self.vuln_scanner = vuln_scanner
        self.semantic_analyzer = semantic_analyzer
        self.ceremony_advisor = CeremonyAdvisor(config.ceremony)
        self.active_orchestrator = ToolExecutor(context_aggregator.registry)
        self.config = config
        self.workspace_resolver = workspace_resolver
        self.llm_manager = LLMManager.create_if_enabled(config.llm)

        # Training data collector (fire-and-forget, only when explicitly enabled)
        from mirdan.llm.training_collector import TrainingCollector

        self.training_collector: TrainingCollector | None = (
            TrainingCollector(config=config.llm)
            if config.llm.enabled and config.llm.collect_training_data
            else None
        )

        # SmartValidator with fix_validator callback to break circular dep (DIP)
        from mirdan.core.smart_validator import SmartValidator

        fix_validator = (
            (lambda code, lang: self.code_validator.validate(code, lang).violations)
            if config.llm.validate_llm_fixes
            else None
        )
        self.smart_validator = SmartValidator(
            llm_manager=self.llm_manager,
            config=config.llm,
            fix_validator=fix_validator,
        )

    # -- Use-case factories ---------------------------------------------------

    def create_enhance_prompt_usecase(
        self, background_tasks: set[asyncio.Task[Any]]
    ) -> EnhancePromptUseCase:
        return EnhancePromptUseCase(
            intent_analyzer=self.intent_analyzer,
            session_manager=self.session_manager,
            mcp_orchestrator=self.mcp_orchestrator,
            context_aggregator=self.context_aggregator,
            prompt_composer=self.prompt_composer,
            knowledge_producer=self.knowledge_producer,
            output_formatter=self.output_formatter,
            session_tracker=self.session_tracker,
            plan_validator=self.plan_validator,
            config=self.config,
            active_orchestrator=self.active_orchestrator,
            background_tasks=background_tasks,
            ceremony_advisor=self.ceremony_advisor,
            agent_coordinator=self.agent_coordinator,
            analyzer_suite=self.analyzer_suite,
            llm_manager=self.llm_manager,
            training_collector=self.training_collector,
        )

    def create_validate_code_usecase(
        self, background_tasks: set[asyncio.Task[Any]]
    ) -> ValidateCodeUseCase:
        return ValidateCodeUseCase(
            code_validator=self.code_validator,
            session_manager=self.session_manager,
            linter_runner=self.linter_runner,
            violation_explainer=self.violation_explainer,
            quality_persistence=self.quality_persistence,
            session_tracker=self.session_tracker,
            semantic_analyzer=self.semantic_analyzer,
            output_formatter=self.output_formatter,
            knowledge_producer=self.knowledge_producer,
            prompt_composer=self.prompt_composer,
            config=self.config,
            active_orchestrator=self.active_orchestrator,
            background_tasks=background_tasks,
            agent_coordinator=self.agent_coordinator,
            confidence_calibrator=self.confidence_calibrator,
            architecture_analyzer=self.analyzer_suite.architecture,
            llm_manager=self.llm_manager,
            smart_validator=self.smart_validator,
            training_collector=self.training_collector,
        )

    def create_validate_quick_usecase(self) -> ValidateQuickUseCase:
        return ValidateQuickUseCase(
            code_validator=self.code_validator,
            auto_fixer=self.auto_fixer,
            output_formatter=self.output_formatter,
        )

    def create_get_quality_standards_usecase(self) -> GetQualityStandardsUseCase:
        return GetQualityStandardsUseCase(quality_standards=self.quality_standards)

    def create_get_quality_trends_usecase(self) -> GetQualityTrendsUseCase:
        return GetQualityTrendsUseCase(quality_persistence=self.quality_persistence)

    def create_scan_conventions_usecase(self) -> ScanConventionsUseCase:
        return ScanConventionsUseCase(convention_extractor=self.convention_extractor)

    def create_scan_dependencies_usecase(self) -> ScanDependenciesUseCase:
        return ScanDependenciesUseCase(
            manifest_parser=self.manifest_parser,
            vuln_scanner=self.vuln_scanner,
        )

    async def close(self) -> None:
        """Cleanup MCP client connections and LLM subsystem."""
        await self.context_aggregator.close()
        if self.llm_manager:
            await self.llm_manager.shutdown()
