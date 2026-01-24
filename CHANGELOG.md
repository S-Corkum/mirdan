# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.0.6]: https://github.com/S-Corkum/mirdan/compare/0.0.5...0.0.6
[0.0.5]: https://github.com/S-Corkum/mirdan/compare/0.0.4...0.0.5
[0.0.4]: https://github.com/S-Corkum/mirdan/compare/0.0.2...0.0.4
[0.0.2]: https://github.com/S-Corkum/mirdan/releases/tag/0.0.2
