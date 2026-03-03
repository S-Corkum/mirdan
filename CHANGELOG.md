# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-03

### Added

- **Self-Managing Integration** — Zero CLAUDE.md instructions needed after `mirdan init`:
  - New `SelfManagingIntegration` class generates `.claude/rules/mirdan-workflow.md`
  - Workflow rule contains quality sandwich pattern, tool table, auto-fix instructions
  - Compaction-resilient state: `generate_compaction_state()` / `restore_from_compaction()`
  - Quality context injection for session awareness
  - New module: `src/mirdan/integrations/self_managing.py`
  - New template: `src/mirdan/integrations/templates/claude_code/mirdan-workflow.md`

- **AGENTS.md Cross-Platform Standard** — Universal generator with platform-specific overlays:
  - `AgentsMDGenerator` class with universal sections (quality rules, language, security, workflow)
  - Platform overlays: Cursor (BugBot, .mdc rules), Claude Code (hooks, MCP tools, skills)
  - Convenience function: `generate_root_agents_md()` for quick generation
  - `mirdan init` now always generates root `AGENTS.md` regardless of platform
  - New module: `src/mirdan/integrations/agents_md.py`

- **Full Hook Lifecycle Coverage** — All 9 Claude Code hook events supported:
  - `HookTemplateGenerator` with per-event methods for all events
  - Events: PreToolUse (prompt reminder), PostToolUse (quick validate + auto-fix), Stop (full validation), SessionStart (inject quality context), SessionStop (persist report), SubagentStart (pass context), SubagentStop (validate output), PreCompact (serialize state), Notification (quality alerts)
  - `HookConfig` dataclass: `enabled_events`, `quick_validate_timeout`, `auto_fix_suggestions`, `compaction_resilience`, `multi_agent_awareness`, `session_hooks`, `subagent_hooks`, `notification_hooks`
  - `ALL_HOOK_EVENTS` constant listing all 9 events
  - Claude Code integration now generates all 9 hooks (up from 3)
  - New module: `src/mirdan/integrations/hook_templates.py`

- **Context Budget Awareness** — Environment-aware output compression:
  - `EnvironmentInfo.context_budget` field, detected from `MIRDAN_CONTEXT_BUDGET`, `CLAUDE_CONTEXT_REMAINING`, `CONTEXT_BUDGET` env vars
  - `OutputFormatter.format_for_compaction()` produces minimal state for context compaction
  - `OutputFormatter.format_quality_context()` with budget-aware compression
  - `SessionManager.serialize()` / `restore()` for compaction resilience
  - `CompactState` dataclass with `to_dict()` / `from_dict()` roundtrip

- **Auto-Fix Expansion** — Dedicated auto-fix engine covering all fixable rules:
  - `AutoFixer` class with template-based (30+ rules) and pattern-based (12+ regex) fixes
  - Fix confidence scoring (>= 0.7 threshold to suggest)
  - Coverage: all PY, JS, TS, RS, GO, JAVA, SEC, AI rules
  - `get_fix()`, `apply_fix()`, `batch_fix()`, `get_fix_for_violation()`
  - `get_fixable_rules()`, `coverage_report()` class methods
  - New CLI command: `mirdan fix <file>` with `--dry-run`, `--staged`, `--auto` flags
  - New modules: `src/mirdan/core/auto_fixer.py`, `src/mirdan/cli/fix_command.py`

- **`mirdan init --upgrade`** — Upgrade existing mirdan installations:
  - Detects existing config version, merges new fields with defaults
  - Regenerates all integration files (hooks, rules, AGENTS.md, workflow)
  - Zero breaking changes from v0.1.0

### Changed

- `generate_claude_code_config()` now delegates hook generation to `HookTemplateGenerator` (9 events vs 3)
- `generate_cursor_rules()` AGENTS.md generation delegated to `AgentsMDGenerator`
- `CodeValidator` fix lookup delegated to `AutoFixer` (expanded from 8 to 30+ fixable rules)
- `HookConfig` added to `MirdanConfig` for hook customization
- `mirdan init` now generates root `AGENTS.md` as step 5 (cross-platform standard)
- `_setup_claude_code()` also generates self-managing workflow rule

### Testing

- **1479 total tests** (1325 → 1479, +154 new tests), all passing
- New test files:
  - `test_auto_fixer.py` (34 tests): template fixes, pattern fixes, apply/batch fix, coverage
  - `test_hook_templates.py` (34 tests): config defaults, event generation, custom commands
  - `test_agents_md.py` (24 tests): universal generation, platform overlays, edge cases
  - `test_self_managing.py` (22 tests): workflow rule, quality context, compaction state
  - `test_context_budget.py` (20 tests): budget detection, compression, session roundtrip
- Updated test files:
  - `test_cli_init.py` (+10 tests): --upgrade flag, AGENTS.md generation, fix routing
  - `test_claude_code_integration.py` (+10 tests): hook delegation, 9-event coverage

## [0.1.0] - 2026-03-02

### Added

- **AI Quality Rules** — AI-specific code quality detection that no other tool catches:
  - `AI001` (error): Placeholder detection — catches `raise NotImplementedError`, `pass` with TODO/FIXME (skips `@abstractmethod`)
  - `AI002` (warning): Hallucinated import detection — flags imports not in Python stdlib or project dependencies
  - `AI008` (error): Injection vulnerability — catches f-string SQL, eval/exec/os.system/subprocess with f-strings
  - New module: `src/mirdan/core/ai_quality_checker.py` with `AIQualityChecker` class
  - New standards: `src/mirdan/standards/ai_quality.yaml`
  - AI rules integrated into both `validate()` (full) and `validate_quick()` (AI001 + AI008)

- **Claude Code Plugin System** — mirdan ships as a distributable Claude Code plugin:
  - `mirdan plugin export [--output-dir PATH]` — exports complete plugin structure
  - Plugin manifest (`.claude-plugin/plugin.json`), MCP config, skills, agents
  - Skills: `/mirdan:code`, `/mirdan:debug`, `/mirdan:review` with SKILL.md frontmatter
  - Agent: `quality-gate` subagent for background quality validation
  - Enhanced hooks template: PreToolUse (prompt reminder), PostToolUse (quick validation), Stop (full validation)

- **Enhanced `mirdan init --claude-code`** — generates complete integration in one command:
  - `.mcp.json` — MCP server registration (auto-detects uvx/mirdan/python -m)
  - `.claude/hooks.json` — automatic quality gates on every edit
  - `.claude/rules/mirdan-*.md` — language-specific quality rules
  - `.claude/skills/{code,debug,review}/SKILL.md` — skill files
  - `.claude/agents/quality-gate.md` — quality gate subagent
  - Merges with existing `.mcp.json` without overwriting other servers
  - Respects existing `hooks.json` (won't overwrite user customizations)

- **New CLI command**: `mirdan plugin` with `export` subcommand

### Removed

- **6 deprecated MCP tool aliases** — reduces context overhead by ~1,200 tokens/session:
  - `analyze_intent` (use `enhance_prompt` instead)
  - `suggest_tools` (use `enhance_prompt` tool_recommendations)
  - `get_verification_checklist` (use `enhance_prompt` verification_steps)
  - `validate_diff` (use `validate_code_quality` with diff input)
  - `validate_plan_quality` (use `validate_code_quality`)
  - `compare_approaches` (removed — platforms handle this natively)
- Only 5 MCP tools remain: `enhance_prompt`, `validate_code_quality`, `validate_quick`, `get_quality_standards`, `get_quality_trends`

### Changed

- `CodeValidator` accepts optional `project_dir` parameter for AI002 import verification
- `mirdan init --claude-code` now generates skills, agents, hooks, and MCP config (previously only rules)

### Testing

- **1233 total tests** (1182 → 1233, +51 new tests), all passing
- New test files: `test_ai_quality_checker.py` (64 tests), `test_plugin_export.py` (20 tests)
- Updated: `test_server_tools.py` (tool registration verification), `test_claude_code_integration.py` (Stop hook tests)

## [0.0.7] - 2026-02-13

### Added

- **AST-based Architecture Validation** for Python: function length, nesting depth, parameter count, import hygiene, class method count
- Architecture thresholds, language stringency, and GitHub config fields in `MirdanConfig`
- Block comment and template literal skip-region system for false-positive elimination (`_build_skip_regions` / `_is_in_skip_region`)
- Server tool handler tests (`test_server_tools.py`) — 57 tests covering all 7 MCP tools
- Dependabot configuration for automated dependency updates
- Coverage gate (`fail_under = 85`) in CI pipeline

### Changed

- Server uses lazy component initialization with lifecycle management (`_lifespan` context manager, `_get_components` singleton)
- Intent analyzer uses weighted scoring for language detection (replacing first-match)
- Intent analyzer uses word boundary matching to reduce false positives
- Quality standards respect language stringency for principles count

### Fixed

- Block comment (`/* */`) and template literal (`` ` ``) content no longer triggers false-positive code validation rules
- Phantom security-scanner gatherer guard (no longer errors when MCP not configured)
- Type annotation fixes across orchestrator and context aggregator
- GitHub config wiring from `ProjectConfig` to `GitHubGatherer`

### Testing

- **715 total tests** (488 → 715, +227 new tests)
- Server.py coverage: 50% → 99%
- Overall project coverage: 89% (gate: 85%)
- New test files: `test_server_tools.py` (57 tests)
- Expanded: `test_code_validator.py` (+49 block comment tests), `test_intent_analyzer.py`, `test_config_wiring.py`, `test_context_aggregator.py`, `test_server.py`

## [0.0.6] - 2026-01-24

### Added

- **RAG Pipeline Domain Standards** (`rag_pipelines.yaml`): Cross-cutting quality standards for Retrieval-Augmented Generation
  - 12 principles: embedding consistency, hybrid retrieval (vector + BM25 via RRF), semantic chunking with overlap, cross-encoder reranking, CRAG pattern, corpus sanitization, embedding versioning, batch processing, RAGAS evaluation, multimodal ingestion, parent-child retrieval, similarity threshold filtering
  - 10 forbidden patterns: mismatched embedding models, fixed-size chunking without overlap, unfiltered context injection, hardcoded chunk sizes, wrong distance metrics, missing metadata, synchronous embedding calls, top-k without threshold, no evaluation metrics, text-only multimodal processing
  - 8 code patterns: chunking, hybrid_retrieval, reranking, evaluation, embedding_versioning, crag_pattern, self_rag, multimodal_ingestion

- **Knowledge Graph Domain Standards** (`knowledge_graphs.yaml`): Cross-cutting quality standards for GraphRAG and knowledge graph construction
  - 10 principles: provenance tracking, entity deduplication, node+edge embeddings, bounded traversals, NER/RE separation, confidence scoring, schema validation, incremental updates, parameterized queries, hybrid graph+vector retrieval
  - 7 forbidden patterns: unbounded traversals, insertion without deduplication, triples without confidence/provenance, LLM extraction without schemas, queries without timeouts, string interpolation in Cypher/Gremlin, entities without schema validation
  - 6 code patterns: entity_extraction, relationship_extraction, hybrid_retrieval, graph_construction, incremental_update, graphrag_query

- **Vector Database Framework Standards** (7 new framework YAML files):
  - `chromadb.yaml`: PersistentClient, metadata, get_or_create_collection, distance functions, batch operations, metadata filters
  - `pinecone.yaml`: Namespaces, batch upserts (100 max), metadata filtering, pod configuration, gRPC client
  - `faiss.yaml`: IndexIVF for scale, vector normalization, IVF training, nprobe tuning, persistence
  - `neo4j.yaml`: Parameterized Cypher, uniqueness constraints, LIMIT clauses, MERGE, vector indexes, bounded paths
  - `weaviate.yaml`: v4 client API, vector_config, batch.rate_limit, hybrid search, multi-tenancy
  - `milvus.yaml`: MilvusClient, index type selection by scale, partition keys, hot/cold tiering, batch insert
  - `qdrant.yaml`: Production client, batch upsert, vector size validation, payload filtering, AsyncQdrantClient, gRPC

- **LangChain RAG Extensions** (`langchain.yaml`): Added RAG-specific principles, forbidden patterns, and 5 new code patterns
  - Principles: EnsembleRetriever, CrossEncoderReranker, MultiVectorRetriever, document metadata, SemanticChunker, multimodal loaders
  - Forbidden: CharacterTextSplitter with overlap=0, similarity_search k>20 without reranking, deprecated loader imports, structure-ignoring chunking
  - Patterns: hybrid_retrieval, semantic_chunking, parent_child, multimodal_ingestion, evaluation_pipeline

- **LangGraph Agentic RAG Extensions** (`langgraph.yaml`): Added RAG-specific principles, forbidden patterns, and 4 new code patterns
  - Principles: CRAG pattern, Self-RAG with reflection tokens, Adaptive RAG query routing, RAGAS evaluation-in-the-loop, max_retrieval_attempts, separate grading nodes
  - Forbidden: retrieval loops without max_attempts, grading without structured output, mixing retrieval+generation in single node
  - Patterns: crag_graph, self_rag_graph, adaptive_rag_graph, evaluation_loop

- **`touches_rag` Intent Field**: New boolean field on Intent model for RAG task detection
  - Detected via 12 RAG keywords (rag, retrieval augmented, vector store/db, embeddings, knowledge graph, graphrag, chunking, similarity search, semantic search, retriever, reranking, vector index)
  - Detected via 7 RAG framework patterns (chromadb, pinecone, faiss, neo4j, weaviate, milvus, qdrant)
  - Included in `EnhancedPrompt.to_dict()` API response

- **RAG Framework Detection** (7 new patterns in IntentAnalyzer):
  - ChromaDB: `chroma`, `chromadb`, `chroma_client`, `PersistentClient`
  - Pinecone: `pinecone`, `Pinecone`
  - FAISS: `faiss`, `FAISS`, `IndexFlat`
  - Neo4j: `neo4j`, `cypher`, `Neo4jVector`
  - Weaviate: `weaviate`, `WeaviateClient`
  - Milvus: `milvus`, `MilvusClient`, `pymilvus`
  - Qdrant: `qdrant`, `QdrantClient`

- **RAG Code Validation Rules** (RAG001–RAG002):
  - `RAG001`: Catches `chunk_overlap=0` (context lost at chunk boundaries) — warning
  - `RAG002`: Catches deprecated `langchain.document_loaders` import path — warning

- **Graph Injection Detection Rules** (SEC011–SEC013):
  - `SEC011`: Cypher f-string interpolation (graph injection vulnerability) — error
  - `SEC012`: Cypher string concatenation (graph injection vulnerability) — error
  - `SEC013`: Gremlin f-string interpolation (graph injection vulnerability) — error

- **RAG Verification Checklist**: 7 RAG-specific verification steps added to prompt composer
  - Embedding model consistency, chunk overlap, metadata storage, similarity threshold, error handling, connection retry, context validation
  - 3 additional Neo4j-specific steps: parameterized Cypher, bounded traversals, entity deduplication

- **RAG Standards Composition**: QualityStandards now composes RAG domain standards into rendered output
  - `render_for_intent()` includes RAG pipeline principles when `touches_rag=True`
  - Includes knowledge graph principles when neo4j framework detected
  - `get_all_standards(category="rag")` returns both RAG and KG standards

### Testing

- **New Test Coverage**: 57 new tests (431 → 488 total)
  - `TestRAGDetection`: 17 tests for RAG keyword and framework detection
  - `TestRAGPatternDetection`: 5 tests for RAG001–RAG002 rules
  - `TestGraphInjectionDetection`: 6 tests for SEC011–SEC013 rules
  - `TestRAGStandards`: 12 tests for standards loading and composition
  - `test_rag_standards.py`: 14 end-to-end integration tests covering full intent→standards→validation→checklist pipeline
  - Updated `test_default_standards_loads_all_categories` for 25 total standard categories

## [0.0.5] - 2026-01-24

### Added

- **LangChain 1.x Framework Support**: Quality standards for the modern LangChain agent API
  - `create_agent()` patterns, middleware lifecycle hooks, structured output strategies
  - Tool design with `@tool` decorator and Pydantic `args_schema`
  - 7 principles, 7 forbidden patterns, 4 code patterns

- **LangGraph 1.x Framework Support**: Quality standards for stateful graph workflows
  - `StateGraph` with TypedDict state, Annotated reducers, and `.compile()` patterns
  - Checkpointing (PostgresSaver/SqliteSaver), human-in-the-loop with `interrupt()`
  - 9 principles, 7 forbidden patterns, 5 code patterns

- **LangChain Deprecated-API Detection Rules** (LC001–LC004):
  - `LC001`: Catches deprecated `initialize_agent()` (use `create_agent()`)
  - `LC002`: Catches deprecated `langgraph.prebuilt` imports (moved to `langchain.agents`)
  - `LC003`: Catches legacy chain patterns (`LLMChain`, `SequentialChain`)
  - `LC004`: Catches `MemorySaver()` usage (in-memory only, not production-safe)

- **Expanded Framework Standards** (4 → 17 frameworks):
  - Django, Express, NestJS, Vue, Nuxt, SvelteKit, Tailwind CSS
  - Gin, Echo (Go), Micronaut, Quarkus (Java)
  - LangChain, LangGraph (Python AI/agents)
  - Dynamic framework loading from `standards/frameworks/` directory

- **Updated Language Standards to 2025/2026**:
  - Go, Java, JavaScript, Rust, TypeScript standards expanded with modern patterns
  - Python standards expanded with security rules (PY007–PY013): unsafe pickle, subprocess shell, yaml.load, os.system, os.path, wildcard imports, requests without timeout

- **LangChain Ecosystem Entity Detection**: Added `langchain`, `langgraph`, `langchain_core`, `langchain_openai`, `langchain_anthropic`, `langchain_community`, `langsmith` to known libraries

- **LangChain/LangGraph Intent Detection**: Framework and Python language detection from prompts mentioning `langchain`, `langgraph`, `StateGraph`, `create_agent`, `AgentExecutor`, `add_conditional_edges`

### Testing

- **New Test Coverage**: 36 new tests (395 → 431 total)
  - `TestLangChainPatternDetection`: 7 tests for LC001–LC004 rules with false-positive checks
  - `TestLangChainDetection`: 6 tests for framework and language detection
  - Framework loading assertions for langchain/langgraph in quality standards tests
  - Python security rule tests (PY007–PY013)

## [0.0.4] - 2025-12-20

### Added

- **PLANNING Task Type**: New task type optimized for creating implementation plans detailed enough for cheap models (Haiku, Flash) to execute correctly
  - `PlanValidator` component for validating plan quality and cheap-model readiness
  - `validate_plan_quality(plan, target_model)` MCP tool
  - Planning-specific prompt templates with anti-slop rules
  - Quality scoring: grounding, completeness, atomicity, clarity
  - Detection of vague language ("should", "probably", "around line", "I think")
  - Validation of required sections (Research Notes, Files Verified, step grounding)

- **PatternMatcher Utility**: Generic pattern matching utility consolidating logic across components
  - Weighted scoring with confidence levels
  - Used by IntentAnalyzer and LanguageDetector

- **BaseGatherer Abstract Class**: Eliminates duplicate boilerplate across gatherer implementations
  - Standardized `__init__` and `is_available()` methods

- **ThresholdsConfig**: Centralized configuration for magic numbers
  - Entity extraction confidence thresholds
  - Language detection score thresholds
  - Code validation severity weights
  - Plan validation penalty values

- **Jinja2 Templates**: Extracted prompt templates for better maintainability
  - `base.j2`: Shared macros for sections
  - `generation.j2`: Standard task prompts
  - `planning.j2`: Planning task prompts with anti-slop rules
  - Reduces PromptComposer from ~400 lines to ~150 lines

- **New Standards**: `planning.yaml` with principles, research requirements, and step format specification

### Fixed

- **CodeValidator False Positives**: Fixed detection of security patterns inside string literals and comments
  - Added `_is_inside_string_or_comment()` method
  - Handles single/double quotes, triple quotes, and line comments

### Changed

- **API Response Keys (Breaking)**: Standardized `EnhancedPrompt.to_dict()` response
  - `detected_task_type` → `task_type`
  - `detected_language` → `language`
  - `detected_frameworks` → `frameworks`

### Removed

- Unused "desktop-commander" and "memory" from KNOWN_MCPS
- Unused "actions" fields from MCP entries
- Unused `PlanStep` model class (replaced with new implementation)
- Duplicate import in server.py

### Documentation

- **Claude Code Integration**: Comprehensive 4-level progressive integration guide
  - Level 1: CLAUDE.md instructions for automatic orchestration
  - Level 2: Slash commands (/code, /debug, /review) with full workflows
  - Level 3: Hooks (PreToolUse, PostToolUse) for automatic enforcement
  - Level 4: Project rules for path-specific security enforcement
  - Copy-paste examples for all configuration files
  - Enterprise managed-mcp.json and managed-settings.json examples

- **Cursor Integration**: Updated for Cursor 2.2 with multi-rule architecture

### Testing

- **New Test Coverage**: 88 new tests (307 → 395 total)
  - `test_language_detector.py`: 22 tests for language detection, confidence levels, minified/test code
  - `test_server.py`: 27 tests for server component logic and workflow integration
  - `test_pattern_matcher.py`: PatternMatcher utility tests
  - `test_plan_validator.py`: 41 tests for plan validation
  - Expanded `test_code_validator.py` with false positive prevention tests

### Dependencies

- Added `jinja2>=3.1.0` for template rendering

## [0.0.2] - 2025-12-XX

### Added

- Initial release with core functionality
- Intent analysis (generation, refactor, debug, review, test)
- Language detection (Python, TypeScript, JavaScript, Go, Rust, Java)
- Code validation with security scanning
- MCP orchestration recommendations
- Quality standards for 6 languages
- Integration guides for Claude Desktop, VS Code, Cursor

[0.2.0]: https://github.com/S-Corkum/mirdan/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/S-Corkum/mirdan/compare/0.0.7...0.1.0
[0.0.7]: https://github.com/S-Corkum/mirdan/compare/0.0.6...0.0.7
[0.0.6]: https://github.com/S-Corkum/mirdan/compare/0.0.5...0.0.6
[0.0.5]: https://github.com/S-Corkum/mirdan/compare/0.0.4...0.0.5
[0.0.4]: https://github.com/S-Corkum/mirdan/compare/0.0.2...0.0.4
[0.0.2]: https://github.com/S-Corkum/mirdan/releases/tag/0.0.2
