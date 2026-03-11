# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.9.0] - 2026-03-11

### Added

- **Tidy First Refactoring Intelligence** — `enhance_prompt` analyzes target files for
  preparatory refactoring opportunities before the main task begins, implementing Kent Beck's
  "Tidy First" principle. Python files get AST-based analysis; other languages use
  indentation-based heuristics. Detects long functions (`extract_method`), deep nesting
  (`simplify_conditional`), and large files (`split_file`). Config-gated via `tidy_first`
  section. Suggestions appear in enhanced prompt output and Jinja2 template rendering.

- **Deep Semantic Analysis Expansion** — 4 new pattern categories for `SemanticAnalyzer`:
  `concurrency` (async/threading detection), `boundary` (division-by-zero, index access,
  numeric parsing), `error_propagation` (swallowed exceptions, JS catch handlers), and
  `state_machine` (string-based state comparisons). Config-gated via `deep_analysis` flag
  on `SemanticConfig`. Pattern severity now uses a tiered mapping instead of a binary
  security-only check.

- **DEEP001 / DEEP004 compiled rules** — Two new AST-based `BaseRule` subclasses:
  `DEEP001SwallowedExceptionRule` detects empty exception handlers (pass, ellipsis, bare
  return); `DEEP004LostExceptionContextRule` detects re-raised exceptions without `from`
  clause. Both run at FULL tier with `deep_analysis` category.

- **Multi-Agent Coordination Intelligence** — `AgentCoordinator` tracks file claims across
  concurrent agent sessions. Detects write-write overlaps and stale-read conflicts.
  Integrated into `enhance_prompt` (auto-claims files from intent entities) and
  `validate_code_quality` (checks for conflicts on validated files). Claims are cleaned up
  on session expiry/eviction. Config-gated via `coordination` section.

- **Adaptive Ceremony** — `enhance_prompt` automatically scales guidance depth based on
  task complexity. Trivial changes (typo fixes) get fast MICRO feedback; complex
  multi-framework tasks get deep THOROUGH analysis. Validation integrity is never compromised.
- `CeremonyLevel` enum: MICRO, LIGHT, STANDARD, THOROUGH (orderable via IntEnum)
- `CeremonyPolicy` frozen dataclass mapping each level to concrete parameter values
- `CeremonyAdvisor` — stateless scoring algorithm with escalation rules:
  - Security/RAG/KG tasks escalate to at least STANDARD
  - PLANNING tasks always get THOROUGH
  - Persistent validation failures escalate +1 level
  - High ambiguity (>=0.6) escalates to STANDARD
- `ceremony_level` parameter on `enhance_prompt` (default: "auto")
- `ceremony` config section with `enabled`, `default_level`, `min_level`,
  `security_escalation`, `ambiguity_escalation`, `ambiguity_threshold`
- Response fields: `ceremony_level`, `recommended_validation`, `ceremony_reason`
- LIGHT ceremony filters tool recommendations to critical-priority only
- MICRO ceremony returns analyze-only response with ceremony metadata

## [1.8.0] - 2026-03-11

### Added

- **Test-Awareness Integration** — 10 new compiled TEST rules (TEST001-TEST010) that detect
  AI-generated test anti-patterns: empty test bodies, assert True placeholders, missing assertions,
  tests that don't exercise code under test, excessive mocking, duplicate test logic, missing edge
  case coverage, unexplained magic values, execution order dependencies, and overly broad exception
  testing.

- **test_file parameter** — `validate_code_quality` accepts `test_file` path for cross-referencing
  implementation coverage. When validating implementation code, point to the test file to get TEST
  rule violations. When validating test code, point to the implementation for TEST004 cross-reference.

- **Incremental Validation** — Rule tier system (QUICK/ESSENTIAL/FULL) and changed-lines filtering
  enable broader coverage in fast validation paths without creating new tools.
  - `validate_quick` gains `scope` parameter: "security" (default, backward compatible) or
    "essential" (adds language-specific rules + ESSENTIAL-tier AI/TEST rules, still <500ms).
  - Both `validate_quick` and `validate_code_quality` gain `changed_lines` parameter to focus
    validation on specific line ranges (e.g., "5,10-15"). Only violations near those lines are
    reported.

- **RuleTier enum** — Rules declare their performance tier (QUICK/ESSENTIAL/FULL). Existing rules
  derive tier from `is_quick` for backward compatibility. New TEST rules declare tier explicitly.
  `RuleRegistry.check_by_tier()` dispatches by tier.

- **Enhanced test-aware prompts** — `enhance_prompt` detects testable code and injects test quality
  requirements. TEST task guidance expanded with TEST001-TEST010 awareness. GENERATION tasks include
  testability reminders.

- **Test completeness semantic checks** — SemanticAnalyzer generates test-specific review questions
  when test code is detected. Violation follow-ups for TEST001, TEST002, TEST003, TEST005.

### Changed

- **RuleContext** — Extended with `is_test`, `implementation_code`, and `changed_lines` fields
- **AIQualityChecker.check()** — Accepts `is_test`, `implementation_code`, and `max_tier` parameters;
  uses `check_by_tier()` internally instead of `check_all()`
- **CodeValidator.validate()** — Accepts `test_file` and `changed_lines` parameters
- **CodeValidator.validate_quick()** — Accepts `scope` and `changed_lines` parameters
- **validate_quick MCP tool** — Accepts `scope` and `changed_lines` string parameters
- **validate_code_quality MCP tool** — Accepts `test_file` and `changed_lines` string parameters
- **CLI `mirdan validate`** — Accepts `--scope` and `--changed-lines` flags
- **PostToolUse hook template** — Now uses `--scope essential` for broader coverage in hooks
- **testing.yaml** — Aligned YAML standard descriptions with compiled rule semantics

## [1.6.0] - 2026-03-09

### Added

- **Cursor v2.5/v2.6 Support** — Full support for Cursor v2.5 plugin system and v2.6 features:
  - **Plugin Manifest Alignment** — `.cursor-plugin/plugin.json` follows official v2.5 schema with author, license, keywords, and component paths (rules, agents, skills, commands, hooks, mcpServers)
  - **Sandbox Access Controls** — `.cursor/sandbox.json` generated with deny-default network policy and language-specific package registry allowlists (PyPI, npm, crates.io, proxy.golang.org)
  - **Async Subagent Coordination** — Background subagents document async execution patterns; foreground subagents document child subagent spawning for parallel work
  - **Bugbot Autofix Integration** — BUGBOT.md includes quality gate requirements for auto-generated fixes, merge protocol, and autofix priority classification
  - **Automations Documentation Command** — `/automations` command with setup guide for cursor.com/automations, including PR Quality Gate, Scheduled Quality Audit, and Security Review templates
  - **Debug v2.6 Runtime Instrumentation** — Debug mode rule and skill updated with runtime instrumentation guidance for breakpoints, step-through, and variable state capture
  - **Long-Running Agent Checkpoints** — AGENTS.md includes 1-hour re-validation intervals, cumulative quality drift tracking, and >4-hour summary reports

- **Upgrade Path Improvements** — `mirdan init --upgrade` now refreshes subagents, skills, and commands with latest mirdan content via `force` parameter. Previously, these files were skipped if they already existed, preventing users from receiving new content on upgrade.

### Changed

- **`generate_cursor_commands()`, `generate_cursor_subagents()`, `generate_cursor_skills()`** — Accept `force: bool = False` keyword argument to overwrite existing files with latest content
- **`_setup_cursor()`** — Accepts `force_regenerate: bool = False` parameter, passed through to idempotent generators
- **`_run_upgrade()`** — Passes `force_regenerate=True` to `_setup_cursor()` so upgrades refresh mirdan-generated content
- **`CursorAdapter`** — Accepts `force_regenerate` parameter, includes `generate_sandbox()` method, `generate_all()` calls sandbox generation
- **`CursorPluginExporter`** — Manifest moved from `manifest.json` to `.cursor-plugin/plugin.json`, includes sandbox generation via `_write_sandbox()`
- **`_CURSOR_COMMANDS` dict** — Now contains 8 entries (added `automations.md`)

### Future

- **MCP Apps** — Interactive UIs in chat via `_meta.ui` annotations and HTML dashboards (deferred to future release)
- **JetBrains ACP Adapter** — Agent Client Protocol integration for JetBrains IDEs and Zed (deferred to future release)

---

## [1.5.0] - 2026-03-09

### Added

- **Cursor Subagents** — 5 `.cursor/agents/*.md` subagent definitions following the Cursor subagent spec (docs.cursor.com/context/subagents): `mirdan-quality-validator`, `mirdan-security-scanner`, `mirdan-test-auditor`, `mirdan-slop-detector`, `mirdan-architecture-reviewer`. Auto-invoked by Cursor's agent based on description, or explicitly via `/subagent-name`.

- **Cursor Skills** — 7 `.cursor/skills/*/SKILL.md` skill definitions following the Agent Skills Standard (agentskills.io): `mirdan-code`, `mirdan-debug`, `mirdan-review`, `mirdan-plan` (auto-invoke), `mirdan-quality`, `mirdan-scan`, `mirdan-gate` (explicit invocation). Provides richer metadata, progressive disclosure, and composability compared to slash commands.

- **Cursor Cloud Environment Config** — `.cursor/environment.json` generated for Cursor Cloud Agent environments (cursor.com/schemas/environment.schema.json). Ensures mirdan is installed and available in cloud-based agent runs. Auto-detects uv vs pip for install command.

- **Command-Type Hooks** — Two shell-script hooks in `.cursor/hooks/` for fast, deterministic checks alongside existing prompt-type hooks:
  - `mirdan-shell-guard.sh` (beforeShellExecution): Blocks destructive commands (`rm -rf /`, `DROP TABLE`, force push to main, `git reset --hard`) with zero LLM overhead. Fails closed.
  - `mirdan-stop-gate.sh` (stop): Advisory reminder at task completion to run `/mirdan-gate` if uncommitted changes exist.

### Changed

- **`generate_cursor_hooks()` now produces hybrid hooks** — `beforeShellExecution` and `stop` events include both command-type (fast, deterministic) and prompt-type (context-aware) hooks. Command hooks fire first.
- **`CursorAdapter` expanded** — New methods: `generate_subagents()`, `generate_skills()`, `generate_environment()`. `generate_all()` includes all new generators.
- **`mirdan init --cursor` generates full Cursor 2.x config** — Now produces subagents, skills, environment.json, and hook scripts in addition to existing rules, commands, hooks.json, mcp.json, AGENTS.md, and BUGBOT.md.
- **`CursorPluginExporter` expanded** — Plugin export includes subagents, skills, and environment config.

---

## [1.1.0] - 2026-03-05

### Added

- **Cursor Slash Commands** — 7 plain-Markdown `.cursor/commands/*.md` files generated by `mirdan init --cursor`: `/code`, `/debug`, `/review`, `/plan`, `/quality`, `/scan`, `/gate` — injected as prompt context when typed in Cursor
- **Cursor Debug Mode Rule** — `mirdan-debug.mdc` description-based rule activates automatically in Cursor Debug Mode with root-cause-first workflow and post-fix validation
- **Cursor Agent Mode Rule** — `mirdan-agent.mdc` description-based rule activates for Background Agents and multi-agent runs with mandatory quality checkpoints (score ≥ 0.7)
- **Claude Code `/scan` Skill** — `scan/SKILL.md` for convention scanning using `mcp__mirdan__scan_conventions` with enyal recall/remember integration
- **Claude Code `/gate` Skill** — `gate/SKILL.md` for pre-commit quality gate using `uvx mirdan gate` with per-file validation fallback
- **10 AI Framework Standards** — YAML standards for: anthropic-sdk, openai-sdk, vercel-ai, llamaindex, autogen, instructor, pydantic-ai, haystack, openai-agents, mcp-server
- **C# Language Support** — `csharp.yaml` language standards (C# 13 / .NET 9), LanguageDetector patterns, IntentAnalyzer language detection, `_BLOCK_COMMENT_LANGUAGES` support
- **ASP.NET Core Framework** — `aspnetcore.yaml` standards for Minimal API, DI patterns, middleware ordering
- **ORM Standards** — `prisma.yaml` (Prisma 5.x) and `sqlalchemy.yaml` (SQLAlchemy 2.0 async style)
- **Axum Framework** — `axum.yaml` standards for Rust web development with Tower middleware
- **Versioned Standards** — `react-19.yaml` (React Compiler, use() hook, Server Actions) and `next.js-15.yaml` (async params, stable after())
- **State Management** — `zustand.yaml` (Zustand 5 with immer/devtools/persist) and `tanstack-query.yaml` (TanStack Query v5 with gcTime rename)
- **GraphQL Standards** — `graphql.yaml` covering DataLoader, query depth limits, schema-first patterns
- **OpenTelemetry Standards** — `opentelemetry.yaml` covering TracerProvider, BatchSpanProcessor, semantic conventions
- **Compiled Validation Rules** — OAI001 (deprecated openai.ChatCompletion.create), ANT001 (deprecated client.completions.create), SA001 (SQLAlchemy 1.x session.query)

### Changed

- **Claude Code skills expanded to 7** — `generate_skills()` now generates `/code`, `/debug`, `/review`, `/plan`, `/quality`, `/scan`, `/gate` (was 5)
- **Cursor `mirdan-planning.mdc` updated** — now description-based (Plan Mode activation), uses `@Docs` instead of context7, includes `mcp__mirdan__enhance_prompt` entry point
- **`mirdan init --cursor` generates commands** — `_setup_cursor()` now calls `generate_cursor_commands()` to produce `.cursor/commands/*.md`
- **`CursorAdapter.generate_all()` includes commands** — plugin export and adapter both emit the 7 command files
- **Skill `allowed-tools` expanded** — `/code` and `/debug` skills now include `mcp__enyal__enyal_recall`, `mcp__enyal__enyal_remember`, `mcp__context7__resolve-library-id`, `mcp__context7__query-docs`
- IntentAnalyzer detects 18 new frameworks: anthropic-sdk, openai-sdk, vercel-ai, llamaindex, autogen, instructor, pydantic-ai, haystack, openai-agents, mcp-server, aspnetcore, sqlalchemy, axum, zustand, tanstack-query, graphql, opentelemetry, and csharp language
- LanguageDetector detects C# from namespace, async Task, and ASP.NET attributes
- `quality_standards.py` languages list includes csharp

### Fixed

- **Agent frontmatter** — removed unsupported Claude Code fields (`background:`, `memory:`, `isolation:`, `skills:`, `mcpServers:`) from all 5 agent templates; these fields cause parse errors in Claude Code
- **Agent names aligned** — template names now match deployed names: `quality-gate`, `security-audit`, `test-quality`, `convention-check` (were `quality-validator`, `security-scanner`, `test-auditor`, `ai-slop-detector`)
- **GitHub CI removed** — `generate_github_action()` removed from `github_ci.py`; `mirdan init` no longer generates `.github/workflows/`; GitHub Actions is out of scope for mirdan
- **`mirdan-planning.mdc` double-underscore invariant** — added `mcp__mirdan__enhance_prompt` reference to satisfy project-wide tool naming test

---

## [0.4.0] - 2026-03-04

### Added

- **Platform Adapter Architecture** — Abstract `PlatformAdapter` base class for unified IDE integrations:
  - `PlatformAdapter` ABC with `generate_hooks()`, `generate_rules()`, `generate_agents()`, `generate_mcp_config()`, `generate_all()`
  - `ClaudeCodeAdapter` wrapping existing Claude Code generation functions
  - `CursorAdapter` wrapping existing + new Cursor generation functions
  - New module: `src/mirdan/integrations/base.py`

- **Cursor Hooks Generation** — Full Cursor 1.7+ hooks.json support:
  - `CursorHookStringency` enum: minimal (2), standard (3), comprehensive (4) events
  - `generate_cursor_hooks()` producing `.cursor/hooks.json` with `version: 1`
  - Events: `afterFileEdit` (validate), `preToolUse` (security reminder), `stop` (verification gate with `loop_limit`), `beforeSubmitPrompt` (quality context)
  - All hooks use prompt type for LLM evaluation
  - Idempotent: skips if hooks.json already exists (respects user customizations)

- **Enhanced AGENTS.md for Cursor** — Quality enforcement for background agents:
  - Mandatory Quality Checkpoints: before writing, after edit, before PR, periodic (30 min)
  - AI Quality Rules (AI001-AI008) inline reference table
  - Security Standards (SEC001-SEC010) inline reference table
  - Quality Thresholds: minimum 0.7 score, 0.8 for security-critical files

- **Enhanced BUGBOT.md** — Structured detection rules with regex patterns:
  - Blocking Bugs: AI001, AI008, SEC001, SEC002, SEC003 with regex patterns
  - Request Changes: AI003, AI007, SEC006-SEC010
  - Best Practice: AI004-AI006, documentation guidance
  - Regex patterns in fenced code blocks for BugBot pattern matching

- **Cursor MCP Config Generation** — `.cursor/mcp.json` with mirdan server:
  - `generate_cursor_mcp_json()` auto-detecting mirdan installation method
  - Includes `MIRDAN_TOOL_BUDGET` env var for tool count limiting
  - Merges with existing mcp.json without overwriting other servers

- **Tool Budget Strategy** — Environment-variable-controlled tool filtering:
  - `MIRDAN_TOOL_BUDGET` env var limits exposed tools by priority order
  - `_TOOL_PRIORITY` list: validate_code_quality > validate_quick > enhance_prompt > get_quality_standards > get_quality_trends
  - Filtering happens at lifespan startup, preserving `@mcp.tool()` registration for tests
  - Budget=0 exposes no tools, Budget=2 exposes top 2, unset keeps all 5

- **PlatformProfile Model** — Typed platform configuration:
  - `PlatformProfile` Pydantic model with `name`, `context_level`, `tool_budget_aware`
  - Added to `MirdanConfig` as `platform` field
  - `_write_config()` uses `config.platform.model_dump()` instead of hardcoded dicts

- **New CLI Flags** for `mirdan init`:
  - `--quality-profile NAME` — Set named quality profile (e.g. enterprise, startup)
  - `--all` — Set up both Cursor and Claude Code configurations
  - `--cursor` now generates hooks.json + mcp.json in addition to rules/agents

- **Public API**: `_detect_mirdan_command()` renamed to `detect_mirdan_command()` for cross-module reuse

### Changed

- `_setup_cursor()` now generates hooks.json and mcp.json alongside rules and agents
- `_build_config()` accepts `quality_profile` parameter and constructs `PlatformProfile`
- `_write_config()` uses `config.platform.model_dump()` instead of inline platform dicts
- BUGBOT.md now has structured rules with regex patterns (was basic text)
- AGENTS.md now includes quality checkpoints, AI/SEC rules, and threshold sections

### Stats

- 1615 total tests (up from 1544)
- 71 new v0.4.0 tests covering platform adapters, cursor hooks, enhanced agents/bugbot, tool budget, CLI flags

## [0.3.0] - 2026-03-03

### Added

- **Quality Profiles System** — Pre-configured quality dimensions for different project types:
  - `QualityProfile` dataclass with 6 dimensions: security, architecture, testing, documentation, ai_slop_detection, performance
  - 7 built-in profiles: default, startup, enterprise, fintech, library, data-science, prototype
  - `get_profile()`, `apply_profile()`, `list_profiles()` functions
  - Maps profile dimensions to QualityConfig stringency levels (strict/moderate/permissive)
  - `quality_profile` and `custom_profiles` fields added to `MirdanConfig`
  - New module: `src/mirdan/core/quality_profiles.py`

- **Full Lifecycle Hooks for Claude Code** — 7 enforcement hooks (up from 2):
  - `UserPromptSubmit` — Inject quality context before processing
  - `PreToolUse` — Security-aware reminder before Write/Edit/MultiEdit
  - `PostToolUse` — Validate code quality after Write/Edit/MultiEdit
  - `Stop` — Verification gate before task completion
  - `PreCompact` — Preserve quality state for compaction resilience
  - `SubagentStart` — Inject quality standards into subagents
  - All hooks use prompt-type for Claude Code plugin compatibility
  - `HookStringency` enum: minimal (2), standard (5), comprehensive (6) hooks
  - `generate_claude_code_hooks()` method on `HookTemplateGenerator`

- **Compaction-Resilient Rules** — `.claude/rules/` enforcement files:
  - `mirdan-always.md` — Always-loaded quality enforcement (AI001-AI008 rules)
  - `mirdan-security.md` — Path-scoped to auth/API files (SEC001-SEC013 rules)
  - `mirdan-ai-quality.md` — Path-scoped to code files (AI-specific violations)
  - Rules survive context compaction (re-read from disk automatically)

- **Modernized Skills** — 5 skills (up from 3) with modern SKILL.md features:
  - Added `/plan` skill with `context: fork` and config dynamic context
  - Added `/quality` skill with `context: fork` for isolated analysis
  - Updated `/review` skill with `context: fork` and git diff context
  - All skills use `model: inherit` and reference only current 5 MCP tools
  - Dynamic context via `!` backtick commands (git diff, config)

- **Specialized Subagent Ecosystem** — 5 agents (up from 1):
  - `security-scanner` — PROACTIVELY scans auth/API files (haiku, background, memory)
  - `architecture-reviewer` — Structural analysis for ARCH001-ARCH005 (sonnet, background)
  - `ai-slop-detector` — PROACTIVELY detects AI001-AI008 violations (haiku, background)
  - `test-auditor` — Test quality audit for meaningful coverage (haiku, background, memory)
  - `quality-validator` — Updated with `background: true` and `memory: project`

### Changed

- Hooks now use prompt-type handlers instead of command-type for Claude Code plugin compatibility
- PostToolUse matcher expanded from `Write|Edit` to `Write|Edit|MultiEdit`
- `ALL_HOOK_EVENTS` expanded to 10 events (added `UserPromptSubmit`)
- Skills reference only current 5 MCP tools (removed deprecated tool references)
- Settings.json updated from 8 tool permissions to 5 (matching current server)

### Fixed

- Plugin `settings.json` no longer references removed tools (get_verification_checklist, analyze_intent, suggest_tools, validate_plan_quality, validate_diff)
- Skill SKILL.md files no longer reference deprecated tool names

### Stats

- 1544 total tests (up from 1479)
- 73 new v0.3.0 tests covering quality profiles, hooks, rules, skills, agents

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

[0.4.0]: https://github.com/S-Corkum/mirdan/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/S-Corkum/mirdan/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/S-Corkum/mirdan/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/S-Corkum/mirdan/compare/0.0.7...0.1.0
[0.0.7]: https://github.com/S-Corkum/mirdan/compare/0.0.6...0.0.7
[0.0.6]: https://github.com/S-Corkum/mirdan/compare/0.0.5...0.0.6
[0.0.5]: https://github.com/S-Corkum/mirdan/compare/0.0.4...0.0.5
[0.0.4]: https://github.com/S-Corkum/mirdan/compare/0.0.2...0.0.4
[0.0.2]: https://github.com/S-Corkum/mirdan/releases/tag/0.0.2
