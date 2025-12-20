# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.0.4]: https://github.com/S-Corkum/mirdan/compare/0.0.2...0.0.4
[0.0.2]: https://github.com/S-Corkum/mirdan/releases/tag/0.0.2
