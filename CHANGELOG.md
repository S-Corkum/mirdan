# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.0.5]: https://github.com/S-Corkum/mirdan/compare/0.0.4...0.0.5
[0.0.4]: https://github.com/S-Corkum/mirdan/compare/0.0.2...0.0.4
[0.0.2]: https://github.com/S-Corkum/mirdan/releases/tag/0.0.2
