# mirdan Evolution Plan: From Quality Orchestrator to Quality Runtime

**Version**: 0.0.7 → 1.0.0
**Date**: 2026-03-02
**Status**: Phase 1 Complete (0.1.0 shipped)

---

## Executive Summary

mirdan must evolve from a **passive quality orchestrator** (manually invoked MCP tools) to an **active quality runtime** (automatically enforced via platform-native hooks, rules, skills, and subagents).

The 2026 landscape has changed dramatically:
- **Cursor**: 8 parallel agents, BugBot (2M+ PRs/month), hooks, .mdc rules, marketplace plugins
- **Claude Code**: Agent teams/swarms, custom subagents, skills, hooks, auto memory, .claude/rules/

mirdan's enduring value: **deep quality standards** (6 languages, 25 frameworks), **AI-specific quality checks** (no other tool does this), and **cross-platform enforcement** (works with both Cursor and Claude Code).

---

## Architecture Vision

```
CURRENT (0.0.7): Passive Quality Orchestrator
  User manually calls enhance_prompt → writes code → manually calls validate

FUTURE (1.0.0): Active Quality Runtime
  mirdan init → hooks/rules/skills/subagents auto-generated
  → quality gates fire automatically on every edit
  → external linters orchestrated for deep validation
  → AI-specific rules catch what no other tool catches
  → project conventions learned and enforced
  → multi-agent quality coordination
```

---

## Phase Overview

| Phase | Version | Theme | Key Deliverable |
|-------|---------|-------|-----------------|
| **1** | **0.1.0** | Automatic Enforcement | One-command setup, auto-firing hooks, linter orchestration |
| **2** | **0.2.0** | AI-Specific Intelligence | AI001-AI008 rules, diff-aware validation, modular standards |
| **3** | **0.3.0** | Learning & Adaptation | Convention learning, dynamic standards, session tracking |
| **4** | **0.4.0** | Scale | Multi-agent coordination, team/enterprise features |
| **GA** | **1.0.0** | Production Quality Runtime | All phases stable, documented, battle-tested |

Each phase is independently shippable and backward-compatible.

---

## Phase 1: Automatic Enforcement (0.1.0) — DONE

**Goal**: After running `mirdan init`, quality gates fire automatically with zero ongoing manual effort. External linters provide deep validation backing mirdan's own rules.

> **Phase 1 Status**: COMPLETE. Shipped in 0.1.0 with the following deviations from original plan:
> - Linter orchestration (1A) deferred to Phase 2 — mirdan's own rules + AI quality rules provide sufficient coverage
> - AI quality rules (AI001, AI002, AI008) pulled forward from Phase 2 into Phase 1
> - Claude Code plugin packaging added (not in original Phase 1 plan)
> - Skills, agents, and enhanced hooks generation added to `mirdan init --claude-code`
> - 6 deprecated MCP tool aliases removed (context overhead reduction)

### 1A: Complete Linter Orchestration

**Rationale**: mirdan's 40 regex rules can't compete with ruff (800+), eslint (300+), mypy, tsc. Instead of reimplementing, ORCHESTRATE them.

**Grounding**: `test_linter_runner.py` and `test_linter_parsers.py` already exist, indicating partial infrastructure. `config.py` has `LinterConfig` with `enabled_linters`, `auto_detect`, `timeout` fields.

#### Step 1.1: Verify and Complete LinterRunner

**File**: `src/mirdan/core/linter_runner.py`
**Action**: Read existing implementation, complete if partial, create if missing
**Details**:
- Class `LinterRunner` with methods:
  - `detect_linters(project_path: Path) -> list[LinterInfo]` — scan pyproject.toml for [tool.ruff], package.json for eslintConfig, tsconfig.json for tsc
  - `run_linter(linter: str, code: str, language: str, file_path: Path | None) -> list[Violation]` — execute linter via subprocess
  - `run_all(code: str, language: str, file_path: Path | None) -> list[Violation]` — run all detected linters, merge results
- Supported linters: ruff (stdin via `--stdin-filename`), eslint (stdin via `--stdin`), mypy (temp file), tsc (temp file)
- Timeout: configurable per linter (default 10s from LinterConfig)
- Graceful degradation: if linter binary not found, skip with warning in result
**Depends On**: None
**Verify**: Tests pass with mocked subprocess calls

#### Step 1.2: Verify and Complete LinterParsers

**File**: `src/mirdan/core/linter_parsers.py`
**Action**: Read existing, complete/create
**Details**:
- `parse_ruff_json(output: str) -> list[Violation]` — ruff --output-format=json
- `parse_eslint_json(output: str) -> list[Violation]` — eslint --format=json
- `parse_mypy_text(output: str) -> list[Violation]` — mypy default text output
- `parse_tsc_text(output: str) -> list[Violation]` — tsc text output
- Each parser maps external rule IDs to mirdan Violation format:
  - `id`: "EXT-{linter}-{rule}" (e.g., "EXT-RUFF-E501")
  - `category`: map to security/architecture/style
  - `severity`: map error→error, warning→warning, note→info
  - `line`, `column`: from linter output
  - `suggestion`: from linter message
**Depends On**: Step 1.1
**Verify**: Parser tests with sample linter output

#### Step 1.3: Wire Linters into CodeValidator

**File**: `src/mirdan/core/code_validator.py` (~line 200, validate() method)
**Action**: Edit
**Details**:
- After mirdan's own rule checks complete
- If LinterRunner is available and enabled in config:
  - Call `linter_runner.run_all(code, language, file_path)`
  - Merge external violations into `result.violations`
  - Recalculate `result.score` with combined violations
  - Add `result.external_violations_count` field
- If LinterRunner is not available: continue with mirdan's rules only (current behavior)
- Pass `file_path` parameter through from server.py if available
**Depends On**: Steps 1.1, 1.2
**Verify**: Read code_validator.py, confirm linter results merged into ValidationResult

#### Step 1.4: Wire Config to Validator

**File**: `src/mirdan/server.py` (line 69, `_get_components()`)
**Action**: Edit
**Details**:
- Pass `linter_config` from MirdanConfig to CodeValidator initialization
- In `validate_code_quality` tool (line 240): pass `file_path` parameter if provided
- Add optional `file_path` parameter to validate_code_quality tool signature
**Depends On**: Step 1.3
**Verify**: Server initializes validator with linter config

#### Step 1.5: Linter Orchestration Tests

**File**: `tests/test_linter_runner.py`, `tests/test_linter_parsers.py`
**Action**: Read existing, extend with new tests
**Details**:
- Mock subprocess for each linter
- Test detection from pyproject.toml/package.json/tsconfig.json
- Test parser output for each format
- Test graceful degradation when binary not found
- Test score recalculation with merged violations
- Test timeout handling
**Depends On**: Steps 1.1-1.4
**Verify**: `uv run pytest tests/test_linter_runner.py tests/test_linter_parsers.py -v`

---

### 1B: Quick Validation Mode

**Rationale**: PostToolUse hooks must complete in <500ms. Full validation takes seconds. Need a fast tier for hooks.

#### Step 1.6: Add Quick Validation to CodeValidator

**File**: `src/mirdan/core/code_validator.py`
**Action**: Edit — add new method
**Details**:
- New method: `validate_quick(code: str, language: str) -> ValidationResult`
  - Only runs security rules: SEC001-SEC013, PY007-PY013
  - Only runs critical AI rules: AI001 (placeholders), AI008 (prompt injection) (after Phase 2)
  - No AST analysis
  - No external linter invocation
  - Skip region building still applies (false-positive prevention)
  - Target execution: <500ms
- Modify existing `validate()` to accept `mode: str = "full"` parameter
  - `mode="quick"` → delegate to validate_quick()
  - `mode="full"` → current behavior (default)
**Depends On**: None (can parallel with 1A)
**Verify**: Time validate_quick() execution, confirm <500ms

#### Step 1.7: New MCP Tool — validate_quick

**File**: `src/mirdan/server.py`
**Action**: Edit — add new tool
**Details**:
- New tool: `validate_quick`
  - Parameters: `code: str`, `language: str = "auto"`
  - Returns: `{"passed": bool, "critical_count": int, "message": str}`
  - Calls `code_validator.validate_quick()`
  - Minimal output (no full violation details, just count + message)
  - No session correlation needed
**Depends On**: Step 1.6
**Verify**: MCP tool callable, returns correct format

#### Step 1.8: CLI --quick Flag

**File**: `src/mirdan/cli/validate_command.py`
**Action**: Edit — add flag
**Details**:
- Add `--quick` flag to argument parser
- When set: call `validate_quick()` instead of `validate()`
- Output: single line "PASS: No critical issues" or "FAIL: N critical issues found"
- Exit code: 0 on pass, 2 on critical failures
- Add `--exit-code` flag: when set, return non-zero on any failure (for blocking hooks)
**Depends On**: Step 1.6
**Verify**: `mirdan validate --quick --file test.py` completes in <1s

#### Step 1.9: Quick Validation Tests

**File**: `tests/test_quick_validation.py` (new)
**Details**:
- Test quick mode only runs security rules
- Test quick mode skips AST analysis
- Test quick mode skips external linters
- Test execution time is fast
- Test CLI --quick flag
- Test exit codes
**Depends On**: Steps 1.6-1.8

---

### 1C: Enhanced Hook Generation

**Rationale**: Current hooks are basic (just CLI validate). Need advanced hooks that leverage quick validation, prompt hooks, and session lifecycle.

#### Step 1.10: Claude Code Advanced Hooks Template

**File**: `src/mirdan/integrations/templates/claude-code-hooks.json` (new)
**Action**: Write
**Details**:
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "prompt",
            "prompt": "QUALITY REMINDER: If you haven't called mcp__mirdan__enhance_prompt for this task yet, do so now before making code changes. This ensures quality requirements are established."
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "mirdan validate --quick --file \"$TOOL_INPUT_FILE_PATH\" --format json 2>/dev/null || true",
            "timeout": 5000
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "mirdan validate --staged --format text 2>/dev/null || true"
          }
        ]
      }
    ]
  }
}
```
- PreToolUse: Prompt hook reminds AI to call enhance_prompt
- PostToolUse: Quick validation after each edit (non-blocking, <5s timeout)
- Stop: Full validation of staged changes when Claude finishes
**Depends On**: Steps 1.6-1.8 (quick mode)
**Verify**: Valid JSON, hook events match Claude Code documentation

#### Step 1.11: Cursor Advanced Hooks Template

**File**: `src/mirdan/integrations/templates/cursor-hooks.json` (new)
**Action**: Write
**Details**:
- Similar structure adapted for Cursor's hook format
- Post-edit: quick validation
- Pre-commit: full validation with exit code
**Depends On**: Steps 1.6-1.8

---

### 1D: Init Overhaul — Zero-Friction Platform Setup

**Rationale**: `mirdan init` should generate EVERYTHING needed for automatic enforcement. One command, zero manual configuration.

#### Step 1.12: Create Claude Code Integration Module

**File**: `src/mirdan/integrations/claude_code.py` (new)
**Action**: Write
**Details**:
- `generate_mcp_json(project_dir: Path) -> Path`
  - Detect mirdan installation (pip → "mirdan serve", uvx → "uvx mirdan serve")
  - Write `.mcp.json` with stdio transport config

- `generate_rules(rules_dir: Path, detected: DetectedProject) -> list[Path]`
  - Quality rule: `.claude/rules/mirdan-quality.md` with paths frontmatter for code files
  - Security rule: `.claude/rules/mirdan-security.md` with paths for auth/security/api/**

- `generate_skills(skills_dir: Path, detected: DetectedProject) -> list[Path]`
  - `/code` skill: `.claude/skills/code.md`
  - `/debug` skill: `.claude/skills/debug.md`
  - `/review` skill: `.claude/skills/review.md`

- `generate_agents(agents_dir: Path, detected: DetectedProject) -> list[Path]`
  - Quality gate subagent: `.claude/agents/quality-gate.md`

- `generate_hooks(claude_dir: Path, detected: DetectedProject) -> Path`
  - Advanced hooks from template (Step 1.10)
**Depends On**: Steps 1.10
**Verify**: All generated files have valid format

#### Step 1.13: Claude Code Skill Templates

**File**: `src/mirdan/integrations/templates/skill-code.md` (new)
**Action**: Write
**Details**:
```markdown
---
name: code
description: Write code with mirdan quality orchestration
---
# Quality-Orchestrated Coding

1. Call mcp__mirdan__enhance_prompt with the task description
2. Use detected_frameworks to query context7 for documentation
3. Follow quality_requirements during implementation
4. After writing code, call mcp__mirdan__validate_code_quality
5. Fix all violations before marking complete
```
Similar templates for `skill-debug.md` and `skill-review.md`.
**Depends On**: None
**Verify**: Valid skill markdown with YAML frontmatter

#### Step 1.14: Claude Code Quality-Gate Subagent Template

**File**: `src/mirdan/integrations/templates/agent-quality-gate.md` (new)
**Action**: Write
**Details**:
```markdown
---
name: quality-gate
model: haiku
tools:
  - Read
  - Glob
  - Grep
  - mcp__mirdan__validate_quick
  - mcp__mirdan__validate_code_quality
description: Background quality validation agent
---
# Quality Gate Agent

You are a code quality validator. When invoked:
1. Read the specified file(s)
2. Run mcp__mirdan__validate_quick for initial check
3. If quick check fails, run full mcp__mirdan__validate_code_quality
4. Report findings concisely
```
**Depends On**: Step 1.7 (validate_quick tool)
**Verify**: Valid subagent markdown with YAML frontmatter

#### Step 1.15: Claude Code Rules Templates

**File**: `src/mirdan/integrations/templates/rule-quality.md` (new)
**File**: `src/mirdan/integrations/templates/rule-security.md` (new)
**Action**: Write
**Details**:
- `rule-quality.md`:
```markdown
---
paths: "**/*.py,**/*.ts,**/*.tsx,**/*.js,**/*.jsx,**/*.go,**/*.rs,**/*.java"
---
# mirdan Quality Standards
When editing code files, use mirdan for quality enforcement...
```
- `rule-security.md`:
```markdown
---
paths: "**/auth/**,**/security/**,**/api/**,**/middleware/**,**/*token*,**/*session*"
---
# mirdan Security Standards
Security-sensitive code requires strict mirdan validation...
```
**Depends On**: None
**Verify**: Valid Claude Code rules format with paths frontmatter

#### Step 1.16: MCP JSON Generation

**File**: Part of `src/mirdan/integrations/claude_code.py` (generate_mcp_json)
**Action**: Part of Step 1.12
**Details**:
- Detect installation method:
  - Check if `mirdan` is in PATH → `{"command": "mirdan", "args": ["serve"]}`
  - Check if installed via uv → `{"command": "uvx", "args": ["mirdan"]}`
  - Fallback: `{"command": "python", "args": ["-m", "mirdan"]}`
- For Cursor: write `.cursor/mcp.json`
- For Claude Code: write `.mcp.json`
**Depends On**: None
**Verify**: Generated JSON is valid MCP config

#### Step 1.17: Update init_command.py

**File**: `src/mirdan/cli/init_command.py`
**Action**: Edit — major evolution of _setup_claude_code()
**Details**:
- Import new `claude_code` module
- Replace current _setup_claude_code() (lines 238-272) with:
  ```python
  def _setup_claude_code(directory: Path, detected: DetectedProject) -> None:
      from mirdan.integrations.claude_code import (
          generate_mcp_json,
          generate_rules,
          generate_skills,
          generate_agents,
          generate_hooks,
      )
      claude_dir = directory / ".claude"

      # MCP registration
      generate_mcp_json(directory)

      # Rules
      rules_dir = claude_dir / "rules"
      rules_dir.mkdir(parents=True, exist_ok=True)
      generate_rules(rules_dir, detected)

      # Skills
      skills_dir = claude_dir / "skills"
      skills_dir.mkdir(parents=True, exist_ok=True)
      generate_skills(skills_dir, detected)

      # Subagents
      agents_dir = claude_dir / "agents"
      agents_dir.mkdir(parents=True, exist_ok=True)
      generate_agents(agents_dir, detected)

      # Hooks
      generate_hooks(claude_dir, detected)
  ```
- Similarly evolve _setup_cursor() to generate mcp.json and dynamic .mdc from standards
- Add `--all` flag to generate for both platforms
**Depends On**: Steps 1.12-1.16
**Verify**: `mirdan init --claude-code` generates all expected files

#### Step 1.18: Update Cursor Integration for Dynamic Rules

**File**: `src/mirdan/integrations/cursor.py`
**Action**: Edit — evolve from static templates to dynamic generation
**Details**:
- Load actual standards from QualityStandards for each detected language/framework
- Generate .mdc rules with content from standards (not static templates)
- Generate per-framework .mdc rules (e.g., mirdan-react.mdc, mirdan-fastapi.mdc)
- Generate BUGBOT.md from security standards
- Generate .cursor/mcp.json for MCP registration
**Depends On**: None
**Verify**: Generated .mdc files contain actual standards content

#### Step 1.19: Init Overhaul Tests

**File**: `tests/test_init_overhaul.py` (new)
**Details**:
- Test Claude Code init generates: .mcp.json, rules, skills, agents, hooks
- Test Cursor init generates: mcp.json, dynamic .mdc rules, BUGBOT.md
- Test generated files have valid format
- Test idempotency (running init twice doesn't duplicate)
- Test --all flag generates for both platforms
- Test MCP JSON detection logic
- Test existing files are not overwritten
**Depends On**: Steps 1.12-1.18
**Verify**: `uv run pytest tests/test_init_overhaul.py -v`

---

### Phase 1 Summary — DONE

**Actual deliverables:**
- **Files Created**: ~20 new files
- **Files Modified**: ~10 existing files
- **Total New LOC**: ~2500
- **MCP Tools**: Reduced from 11 to 5 (removed 6 deprecated aliases)
- **New CLI Commands**: `mirdan plugin export`
- **New AI Rules**: AI001 (placeholder detection), AI002 (hallucinated imports), AI008 (injection vulnerabilities)
- **New Claude Code Features**: Skills (code, debug, review), quality-gate agent, enhanced hooks, plugin export
- **Tests**: 1182 → 1233 (+51 new tests, all passing)

**After Phase 1, the user experience is:**
```bash
pip install mirdan
cd my-project
mirdan init --claude-code

# That's it. mirdan is now:
# - Registered as MCP server (.mcp.json)
# - Quality rules active in .claude/rules/
# - Skills available: /mirdan:code, /mirdan:debug, /mirdan:review
# - Quality-gate subagent available in .claude/agents/
# - Hooks auto-fire on every edit (quick) and on stop (full)
# - AI quality rules catch placeholders, hallucinated imports, injection vulns
```

---

## Phase 2: AI-Specific Intelligence (0.2.0)

**Goal**: mirdan catches code quality issues that NO other tool catches — issues unique to AI-generated code. Also completes linter orchestration deferred from Phase 1.

> **Note**: AI001 (placeholder detection), AI002 (hallucinated imports), AI008 (injection vulnerabilities), and ai_quality.yaml were completed in Phase 1 (0.1.0). Phase 2 focuses on the remaining AI rules and linter orchestration.

### 2A: Remaining AI-Specific Quality Rules

> AI001, AI002, AI008, AIQualityChecker module, and ai_quality.yaml are DONE (shipped in 0.1.0).

#### Step 2.1: Implement Additional AI Rules

**File**: `src/mirdan/core/ai_quality_checker.py` (extend existing)
**Details**:
- **AI003 — Over-Engineering Detection** (severity: warning)
  - AST: Find abstract classes with only 1 concrete subclass in file
  - AST: Find classes with > 5 generic type parameters
  - AST: Find factory functions that only create 1 type

- **AI004 — Duplicate Code Block Detection** (severity: warning)
  - Normalize code blocks, hash function bodies
  - Flag duplicate hashes within same file (threshold: > 5 lines)

- **AI005 — Inconsistent Error Handling** (severity: warning)
  - Detect mixed try/except + if/return error patterns in same function
  - Detect bare except mixed with specific except in same file

- **AI006 — Unnecessary Heavy Import** (severity: info)
  - Detect `import requests` for simple GET, `import pandas` for simple CSV, etc.
  - Maintain allowlist of heavy→light mappings

- **AI007 — Security Theater Detection** (severity: error)
  - AST: Detect `hash()` (built-in) used on passwords
  - AST: Detect validation functions that always return True

#### Step 2.2: AI Quality Tests for New Rules

**File**: `tests/test_ai_quality_checker.py` (extend existing)

### 2A-bis: Complete Linter Orchestration (deferred from Phase 1)

---

### 2B: Diff-Aware Validation

#### Step 2.7: Enhance DiffParser

**File**: `src/mirdan/core/diff_parser.py`
**Action**: Read existing, enhance
**Details**:
- Parse unified diff format (output of `git diff`)
- Extract per-file: additions (new lines), deletions (removed lines), modifications (changed lines)
- Map old line numbers to new line numbers
- Identify which functions/classes were modified (heuristic: indentation-based)
- Return `DiffResult` with per-file change data

#### Step 2.8: Implement RegressionDetector

**File**: `src/mirdan/core/regression_detector.py` (new)
**Action**: Write
**Details**:
- Class: `RegressionDetector`
- Method: `analyze(old_code: str, new_code: str, language: str) -> RegressionResult`
  - Validate old code → get old violations
  - Validate new code → get new violations
  - Compare: which violations are new? which were fixed?
  - Calculate regression score:
    - Positive if more violations fixed than introduced
    - Negative if more violations introduced than fixed
    - Zero if unchanged
- Return: `RegressionResult(new_violations, fixed_violations, regression_score, quality_delta)`

#### Step 2.9: Integrate Diff Mode into validate_code_quality

**File**: `src/mirdan/server.py` (validate_code_quality tool, line 240)
**Action**: Edit
**Details**:
- When `input_type="diff"`:
  - Parse the diff with enhanced DiffParser
  - For each changed file: run regression detection
  - Return enhanced result with `new_violations` and `fixed_violations`
  - quality score reflects ONLY the change (not pre-existing issues)

#### Step 2.10: CLI --diff Enhancement

**File**: `src/mirdan/cli/validate_command.py`
**Action**: Edit
**Details**:
- `mirdan validate --diff` runs `git diff --staged` automatically
- Parses real diff output through DiffParser
- Reports per-file quality changes
- Shows: "3 new violations introduced, 1 existing violation fixed"

#### Step 2.11: Diff Validation Tests

**File**: `tests/test_diff_validation.py` (new or extend existing test_diff_parser.py)

---

### 2C: Modular Standards Output

#### Step 2.12: Create ModularStandards Models

**File**: `src/mirdan/models.py`
**Action**: Edit — add new dataclasses
**Details**:
```python
@dataclass
class StandardModule:
    name: str           # e.g., "python-security", "react-patterns"
    content: str        # The quality instructions text
    activation: str     # "always" | glob pattern | description
    priority: str       # "critical" | "high" | "medium" | "low"

@dataclass
class ModularStandards:
    modules: list[StandardModule]

    def render_for_cursor(self) -> list[dict]:
        """Render as .mdc file contents."""
        ...

    def render_for_claude_code(self) -> list[dict]:
        """Render as .claude/rules/ file contents."""
        ...

    def render_as_text(self) -> str:
        """Render as inline prompt text (fallback)."""
        ...
```

#### Step 2.13: Create RuleGenerator

**File**: `src/mirdan/core/rule_generator.py` (new)
**Action**: Write
**Details**:
- Takes `Intent` + composed quality standards
- Generates `ModularStandards` with modules per:
  - Language (e.g., python-standards)
  - Framework (e.g., react-patterns, fastapi-patterns)
  - Security (if touches_security)
  - AI quality (always)
- Each module is independently useful

#### Step 2.14: Integrate with enhance_prompt

**File**: `src/mirdan/server.py` (enhance_prompt tool, line 121)
**Action**: Edit
**Details**:
- After composing quality requirements
- Generate modular standards via RuleGenerator
- Add `modular_rules` field to enhance_prompt response
- Each rule has `cursor_mdc` and `claude_rule` render options
- AI assistant can write rules to platform-native locations

#### Step 2.15: Modular Standards Tests

**File**: `tests/test_rule_generator.py` (new)

---

### Phase 2 Summary

**Files Created**: ~6 new files (reduced — AI checker, ai_quality.yaml already done in 0.1.0)
**Files Modified**: ~6 existing files
**New MCP Tool Changes**: Enhanced validate_code_quality diff mode
**New Rule Categories**: AI003-AI007 (AI001, AI002, AI008 done in 0.1.0)
**Deferred from Phase 1**: Linter orchestration (1A steps)

---

## Phase 3: Learning & Adaptation (0.3.0)

**Goal**: mirdan learns from the project and gets smarter over time.

### 3A: Convention Learning Bridge

#### Step 3.1: Create ConventionBridge

**File**: `src/mirdan/core/convention_bridge.py` (new)
**Action**: Write
**Details**:
- Load `.mirdan/conventions.yaml` (output of `mirdan scan`)
- Convert convention entries to standards format (principles/forbidden/patterns)
- Merge with built-in standards, with conventions as overrides
- Support enyal as alternative convention source:
  - `enyal_recall("conventions for {language}")` → convert to standards format

#### Step 3.2: Enhance QualityStandards for Dynamic Composition

**File**: `src/mirdan/core/quality_standards.py` (line 81, `_load_default_standards()`)
**Action**: Edit
**Details**:
- After loading YAML standards
- Call ConventionBridge to load project conventions
- Merge: built-in (base) + conventions (supplement) + custom rules (override)
- Priority: custom rules > conventions > built-in defaults
- Track source of each standard (for debugging: "built-in" vs "convention" vs "custom")

#### Step 3.3: Enhanced mirdan scan with Convention Extraction

**File**: `src/mirdan/cli/scan_command.py`
**Action**: Edit
**Details**:
- After extracting conventions, optionally store in enyal:
  - `--store-enyal` flag: store conventions in enyal knowledge graph
  - Each convention → `enyal_remember(content, type="convention", scope="project")`
- Improve convention extraction:
  - Detect naming conventions from existing code
  - Detect import organization patterns
  - Detect error handling patterns
  - Detect testing patterns

### 3B: Dynamic Standards Composition

#### Step 3.4: Standards Layering System

**File**: `src/mirdan/core/quality_standards.py`
**Action**: Edit
**Details**:
- Implement layered standards resolution:
  1. Built-in YAML (base layer)
  2. `.mirdan/conventions.yaml` (project conventions layer)
  3. `.mirdan/rules/*.yaml` (custom rules layer)
  4. Config stringency filters (applied last)
- Each layer can ADD principles, ADD forbidden patterns, OVERRIDE patterns
- Clear merge semantics: later layers override earlier for same rule IDs

#### Step 3.5: Standards Version Tracking

**File**: `src/mirdan/standards/*.yaml` (all standard files)
**Action**: Edit — add version metadata
**Details**:
- Each YAML file gets `version:` field
- Config tracks which version project is using
- Deprecation warnings when project uses outdated standards version

### 3C: Session Quality Tracking

#### Step 3.6: Enhanced SessionManager

**File**: `src/mirdan/core/session_manager.py`
**Action**: Edit
**Details**:
- Track per-file quality scores within session:
  ```python
  session.file_scores: dict[str, list[float]]  # file_path → [score1, score2, ...]
  session.violations_introduced: int
  session.violations_fixed: int
  session.quality_trend: str  # "improving" | "declining" | "stable"
  ```
- Update after each validate_code_quality call
- Calculate session quality trend

#### Step 3.7: New MCP Tool — get_session_quality

**File**: `src/mirdan/server.py`
**Action**: Edit — add new tool
**Details**:
- Tool: `get_session_quality`
- Parameters: `session_id: str` (optional, uses latest if not provided)
- Returns:
  - Per-file quality scores
  - Violations introduced vs fixed
  - Quality trend (improving/declining/stable)
  - Top 3 areas needing attention
  - Recommendations for next steps

---

### Phase 3 Summary

**Files Created**: ~4 new files
**Files Modified**: ~6 existing files
**New MCP Tools**: 1 (`get_session_quality`)

---

## Phase 4: Scale (0.4.0)

**Goal**: mirdan works at multi-agent and team scale.

### 4A: Multi-Agent Quality Coordination

#### Step 4.1: Agent-Aware Session Tracking

**File**: `src/mirdan/core/session_manager.py`
**Action**: Edit
**Details**:
- Track per-agent quality within session:
  ```python
  session.agent_scores: dict[str, list[ValidationResult]]
  ```
- New method: `register_agent(agent_id: str)`
- New method: `get_agent_quality(agent_id: str) -> AgentQualityProfile`

#### Step 4.2: New MCP Tool — validate_agent_output

**File**: `src/mirdan/server.py`
**Action**: Edit — add new tool
**Details**:
- Extends validate_code_quality with `agent_id` parameter
- Tracks quality per agent in session
- Returns agent-specific quality profile

#### Step 4.3: New MCP Tool — compare_agent_outputs

**File**: `src/mirdan/server.py`
**Action**: Edit — add new tool
**Details**:
- Input: list of {agent_id, code, description}
- Validates each independently
- Compares: scores, violation counts, patterns
- Returns: ranked agents, best output, reasoning

#### Step 4.4: Quality-Aware Task Hints

**File**: `src/mirdan/server.py` (enhance_prompt)
**Action**: Edit
**Details**:
- Add `suggested_validation_tier` to enhance_prompt response
- If touches_security: suggest "strict"
- If simple refactor: suggest "light"
- Lead agent can use this for task delegation decisions

### 4B: Team Features

#### Step 4.5: Hierarchical Config Resolution

**File**: `src/mirdan/config.py`
**Action**: Edit
**Details**:
- Search order: project `.mirdan/config.yaml` > workspace `~/.mirdan/config.yaml` > global
- Merge semantics: deeper scope overrides shallower

#### Step 4.6: Standards Export/Import

**File**: `src/mirdan/cli/standards_command.py`
**Action**: Edit
**Details**:
- `mirdan standards --export > my-standards.yaml` — export project standards
- `mirdan standards --import my-standards.yaml` — import external standards
- Shareable standard packs for organizations

#### Step 4.7: Quality History Persistence

**File**: `src/mirdan/core/quality_persistence.py`
**Action**: Read existing (test_quality_persistence.py exists), enhance
**Details**:
- Store validation results in `.mirdan/history/`
- Per-file, per-session quality scores
- Trend calculation over configurable time window

---

### Phase 4 Summary

**Files Created**: ~4 new files
**Files Modified**: ~6 existing files
**New MCP Tools**: 2 (`validate_agent_output`, `compare_agent_outputs`)

---

## Cross-Cutting Concerns

### Testing Strategy

| Phase | New Test Files | Coverage Target |
|-------|---------------|-----------------|
| 1 | test_quick_validation.py, test_init_overhaul.py, + extend existing | 85% |
| 2 | test_ai_quality_checker.py, test_diff_validation.py, test_rule_generator.py | 85% |
| 3 | test_convention_bridge.py, test_session_tracking.py | 85% |
| 4 | test_multi_agent.py, test_hierarchical_config.py | 85% |

### Backward Compatibility

- All existing MCP tools retain their parameters and behavior
- New parameters are always optional with defaults matching current behavior
- ~~Deprecated tool aliases continue to work~~ **0.1.0 update**: 6 deprecated aliases removed to reduce context overhead (~1,200 tokens/session). Only 5 active tools remain.
- Config defaults mean zero-config upgrade path
- **0.1.0 update**: mirdan ships as a distributable Claude Code plugin (`mirdan plugin export`)

### Performance Budget

| Operation | Target | Phase |
|-----------|--------|-------|
| validate_quick | <500ms | 1 |
| validate_full (mirdan rules) | <2s | Current |
| validate_full (with linters) | <15s | 1 |
| enhance_prompt | <3s | Current |
| PostToolUse hook round-trip | <3s | 1 |
| AI rule checking | <1s | 2 |
| Convention loading | <500ms | 3 |

### Config Evolution

Each phase adds to MirdanConfig with backward-compatible defaults:

```python
# Phase 1 additions
class LinterConfig:  # Already exists, wire through
    enabled_linters: list[str] = []
    auto_detect: bool = True
    timeout: float = 10.0

# Phase 2 additions
class AIQualityConfig:
    enabled: bool = True
    rules: list[str] = ["AI001", "AI002", ..., "AI008"]
    placeholder_severity: str = "error"

# Phase 3 additions
class ConventionConfig:
    learning_enabled: bool = True
    enyal_integration: bool = False
    conventions_path: str = ".mirdan/conventions.yaml"

# Phase 4 additions
class MultiAgentConfig:
    per_agent_tracking: bool = True
    consistency_checking: bool = True
```

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| External linter not installed | Validation incomplete | Graceful degradation: use mirdan's rules only, warn |
| Hook performance too slow | Developer frustration | Quick mode (<500ms), async hooks, timeout guards |
| Platform API changes | Breaking integration | Abstract behind interfaces, version detection in init |
| Scope creep per phase | Delays | Each phase is MVP-first, iterate after ship |
| Standards maintenance burden | Outdated standards | Community standards packages (Phase 4B) |
| False positive AI rules | Developer annoyance | Conservative thresholds, info severity for uncertain |

---

## Success Metrics

### Phase 1 (0.1.0)
- `mirdan init --claude-code` generates all files in <5s
- PostToolUse hooks fire automatically on every edit
- Quick validation completes in <500ms
- External linters (when available) integrated into validation score

### Phase 2 (0.2.0)
- AI001-AI008 rules detect real AI-specific issues in sample codebases
- False positive rate for AI rules: <5%
- Diff-aware validation correctly attributes violations to new vs existing code

### Phase 3 (0.3.0)
- Conventions extracted from project code improve validation relevance
- Session quality tracking shows trend data across multi-edit sessions

### Phase 4 (0.4.0)
- Multi-agent outputs are validated independently and compared
- Team quality standards are shareable via export/import

### 1.0.0 (GA)
- All 4 phases stable with 85%+ test coverage
- Both Cursor and Claude Code fully supported
- Zero manual intervention after init
- Measurable improvement in AI-generated code quality

---

## Implementation Order (First 5 Steps to Start)

If starting today, begin with these 5 steps in order:

1. **Read existing linter infrastructure** (test_linter_runner.py, test_linter_parsers.py, linter_runner.py, linter_parsers.py) — understand what's already built
2. **Complete LinterRunner + LinterParsers** — wire ruff support first (mirdan's own language)
3. **Wire into CodeValidator** — external violations merge into quality score
4. **Add validate_quick mode** — <500ms security-only validation
5. **Create Claude Code integration module** — generate all platform files via init

These 5 steps deliver the highest value with the least risk.
