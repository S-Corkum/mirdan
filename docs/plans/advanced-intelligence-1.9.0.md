# Advanced Intelligence Features — Mirdan 1.9.0

> Three features that extend mirdan from a quality gate into a quality advisor:
> Tidy First refactoring intelligence, Last 30% deep semantic analysis,
> and Multi-Agent coordination.

## Research Notes (Pre-Plan Verification)

### Files Verified

- `src/mirdan/models.py`: All dataclasses verified. Key types: Intent (line 78), SessionContext (line 118), EnhancedPrompt (line 294), Violation (line 555), SemanticCheck (line 471), AnalysisProtocol (line 494). EntityType.FILE_PATH at line 72.
- `src/mirdan/server.py`: 7 tool handlers (lines 109-341). enhance_prompt at line 110, validate_code_quality at line 159. Thin routing to use cases.
- `src/mirdan/config.py`: MirdanConfig at line 352 with sections: quality, orchestration, enhancement, planning, thresholds, session, tokens, linters, hooks, platform, semantic, dependencies, workspace, ceremony. SemanticConfig at line 24 (enabled + analysis_protocol).
- `src/mirdan/providers.py`: ComponentProvider at line 51. DI pattern. Creates all use cases via factory methods (lines 146-204). CeremonyAdvisor wired at line 139.
- `src/mirdan/core/semantic_analyzer.py`: _PATTERNS dict (lines 12-49), _QUESTION_TEMPLATES (lines 53-86), _VIOLATION_FOLLOW_UPS (lines 89-125), _SECURITY_PATTERNS frozenset (line 128). SemanticAnalyzer class (line 131). generate_checks() at line 142, generate_analysis_protocol() at line 204.
- `src/mirdan/core/ceremony.py`: CeremonyAdvisor (line 9). Stateless. determine_level() at line 63. _estimate_base_level() at line 137. _apply_escalations() at line 169.
- `src/mirdan/core/code_validator.py`: CodeValidator (line 84). LANGUAGE_RULES dict (line 88). Regex-based `DetectionRule` compilation via `_compile_rules()` (line 641). Does NOT contain BaseRule/RuleRegistry registration — that lives in `ai_quality_checker.py`.
- `src/mirdan/core/ai_quality_checker.py`: AIQualityChecker (line 58). RuleRegistry-based rule registration (lines 72-106). All BaseRule subclasses (AI001-AI008, SEC014, TEST001-TEST010) are registered here via `self._registry.register()`.
- `src/mirdan/core/rules/base.py`: BaseRule ABC (line 57), RuleContext (line 27), RuleRegistry (line 86), RuleTier IntEnum (line 13). RuleContext has: skip_regions, project_deps, file_path, is_test, implementation_code, changed_lines.
- `src/mirdan/core/rules/ai005_error_handling.py`: Example compiled rule pattern. ~110 lines. BaseRule subclass with id/name/languages/check properties.
- `src/mirdan/core/rules/test_body_rules.py`: Uses ast.parse for Python AST analysis. _extract_test_functions helper (line 16). RuleTier.ESSENTIAL declared explicitly.
- `src/mirdan/core/session_manager.py`: SessionManager (line 16). create_from_intent() at line 29. _evict_expired() triggers on get/create. remove() at line 76.
- `src/mirdan/core/session_tracker.py`: SessionTracker (line 62). record_validation() at line 73. get_violation_persistence() at line 168.
- `src/mirdan/usecases/enhance_prompt.py`: EnhancePromptUseCase (line 80). 13 constructor params. execute() at line 113. Flow: intent → ceremony → session → persistent_reqs → tool_recs → context → compose → format.
- `src/mirdan/usecases/validate_code.py`: ValidateCodeUseCase (line 36). 13 constructor params. execute() at line 69.
- `src/mirdan/core/prompt_composer.py`: PromptComposer (line 23). TASK_GUIDANCE dict (line 28). compose() at line 99. Jinja2 Environment for templates (line 92). `_build_prompt_text()` at line 220 renders via `generation.j2` template — no inline string building.
- `src/mirdan/templates/generation.j2`: Jinja2 template for standard tasks (28 lines). Imports macros from `base.j2`. Renders: role, context, task_guidance, task, quality_requirements, constraints, verification, tools sections conditionally.
- `src/mirdan/templates/base.j2`: Macro library (88 lines). Defines 7 reusable macros: role_section, context_section, task_section, quality_requirements_section, constraints_section, verification_section, tools_section.
- `src/mirdan/core/gatherers/base.py`: BaseGatherer ABC (line 76). GathererResult (line 17).
- `src/mirdan/__init__.py`: `__version__ = "1.9.0"` (line 8).
- `pyproject.toml`: Dependencies: fastmcp, pyyaml, pydantic, jinja2. No others.

### Project Structure (Verified via Glob)

```
mirdan/src/mirdan/
├── __init__.py (v1.9.0)
├── server.py (7 MCP tools)
├── models.py (dataclasses)
├── config.py (Pydantic config)
├── providers.py (DI container)
├── core/
│   ├── semantic_analyzer.py
│   ├── code_validator.py
│   ├── ceremony.py
│   ├── tidy_first.py          ← NEW (Feature 9)
│   ├── agent_coordinator.py   ← NEW (Feature 12)
│   ├── session_manager.py
│   ├── session_tracker.py
│   ├── prompt_composer.py
│   ├── context_aggregator.py
│   ├── ai_quality_checker.py  ← BaseRule registration (DEEP rules go here)
│   ├── rules/
│   │   ├── base.py
│   │   ├── ai001-ai008 (8 files)
│   │   ├── sec014 (1 file)
│   │   ├── test_body_rules.py
│   │   ├── test_structure_rules.py
│   │   └── deep_analysis_rules.py  ← NEW (Feature 11)
│   └── gatherers/ (4 gatherers)
├── usecases/
│   ├── enhance_prompt.py
│   ├── validate_code.py
│   └── ...
├── standards/ (YAML files)
└── templates/
    ├── base.j2       ← macros (modify for tidy_first_section)
    ├── generation.j2  ← standard task template (modify for tidy block)
    └── planning.j2
```

### Dependencies Confirmed

- `ast` (stdlib) — Used by test_body_rules.py, python_ast_validator.py
- `re` (stdlib) — Used everywhere
- `pathlib` (stdlib) — Used everywhere
- `time` (stdlib) — Used by session_manager, session_tracker
- `dataclasses` (stdlib) — Used by models.py, rules/base.py
- Zero new external dependencies required.

### Conventions (enyal)

- DI pattern: All components created in providers.py, injected via constructor
- Rule pattern: BaseRule subclass → registered in AIQualityChecker.__init__() via RuleRegistry.register()
- Config pattern: Pydantic BaseModel → added to MirdanConfig as field
- Use case pattern: __init__ receives dependencies, execute() method does work
- Existing features in 1.9.0: Adaptive Ceremony, Test-Awareness, Incremental Validation
- Session-aware logic lives in server.py/use cases, NOT in CodeValidator (keeps validator pure)

### Similar Implementations

- `core/ceremony.py`: New analyzer module pattern (config → stateless class → deterministic output)
- `core/rules/test_body_rules.py`: AST-based compiled rule pattern
- `core/rules/ai005_error_handling.py`: Regex-based compiled rule pattern
- `core/semantic_analyzer.py`: Template-based question generation pattern

---

## Architecture Decisions

### AD-1: No New MCP Tools

All three features extend existing `enhance_prompt` and `validate_code_quality` through additional response fields. No new tools are needed because:
- Tidy First → output of enhance_prompt (`tidy_suggestions` field)
- Last 30% → output of validate_code_quality (new semantic checks + violations)
- Coordination → output of both tools (`coordination` field)

This avoids tool budget pressure (Cursor limits) and keeps the API surface minimal.

### AD-2: Three Independent Modules

Each feature is a standalone module with no cross-dependencies:
- `core/tidy_first.py` — depends only on models.py, config.py, stdlib
- `core/rules/deep_analysis_rules.py` — depends only on rules/base.py, models.py
- `core/agent_coordinator.py` — depends only on models.py, config.py

This means any feature can be disabled via config without affecting the others.

### AD-3: Optional Dependencies in Use Cases

New components are passed as `| None = None` parameters to use cases. This ensures:
- Backward compatibility (existing tests don't break)
- Graceful degradation (if config disables a feature, None is passed)
- No behavioral change unless explicitly opted in

### AD-4: Semantic Questions over Violations for Ambiguous Patterns

The "Last 30%" patterns (concurrency, boundaries, state machines) produce semantic QUESTIONS for the LLM to investigate, not violations. Only two clear-cut patterns (DEEP001 swallowed exception, DEEP004 lost exception context) become compiled rules. This prevents false-positive noise and maintains trust in mirdan's violation system.

### AD-5: Session-Scoped Coordination

Agent coordination data (file claims, conflict warnings) lives in-memory, scoped to sessions. When sessions expire (TTL), claims expire automatically. No persistent storage needed. This is correct because multi-agent sessions are inherently ephemeral.

### AD-6: Accepted Overlap in error_propagation Questions

When a swallowed exception triggers both the `error_propagation` semantic pattern (question) and the `DEEP001` violation follow-up, the same line may produce two questions with different concerns (`"error_propagation"` and `"violation_deep_dive"`). The existing dedup only removes same `(line, concern)` pairs. This is intentional: the semantic question is a general prompt to think about error propagation, while the violation follow-up is a specific directive to fix a detected swallowed exception. The slight redundancy is preferable to missing either perspective.

---

## Phase 1: Last 30% Semantic Analysis Expansion (Steps 1-8)

### Step 1: Extend _PATTERNS in semantic_analyzer.py

**File:** `src/mirdan/core/semantic_analyzer.py` (verified via Read)
**Action:** Edit
**Details:**
Add 4 new pattern categories to the `_PATTERNS` dict after the existing `"test_quality"` entry (line 43):

```python
"concurrency": [
    (r"(?:async\s+def|asyncio\.gather|asyncio\.create_task|threading\.Thread|concurrent\.futures)", "concurrent code"),
    (r"(?:global\s+\w+\s*$|Lock\(\)|Semaphore\(\)|RLock\(\))", "synchronization primitive"),
],
"boundary": [
    (r"(?<=[\w)\]])\s*/\s*[a-zA-Z_]\w*(?!\w)", "division with variable denominator"),
    (r"\[\s*[a-zA-Z_]\w*\s*(?:[+-]\s*\d+\s*)?\]", "dynamic index access"),
    (r"(?:int\s*\(|float\s*\(|parseFloat\s*\(|parseInt\s*\(|strconv\.(?:Atoi|ParseFloat))", "numeric parsing from string"),
],
"error_propagation": [
    (r"except\s+\w[\w.]*(?:\s+as\s+\w+)?:\s*$", "exception handler (verify error is not swallowed)"),
    (r"\.catch\s*\(\s*(?:\(\s*\w*\s*\)\s*=>|function\s*\()\s*\{", "JS catch handler (verify error is handled)"),
],
```

> **Note:** These patterns identify exception-handler _headers_ on a single line
> because `generate_checks()` scans line-by-line. The actual swallowed-body
> detection is handled by the DEEP001 compiled rule (AST-based). The semantic
> question here prompts the developer to think about error propagation.

```python
"state_machine": [
    (r"\b(?:status|state|phase|stage|mode)\s*(?:==|===|!=|!==)\s*['\"]", "string-based state comparison"),
    (r"(?:\.status|\.state|\.phase|\.stage)\s*=\s*['\"]", "direct string state assignment"),
],
```

**Depends On:** None
**Verify:** Read file, confirm 4 new entries in _PATTERNS dict
**Grounding:** Read of semantic_analyzer.py (lines 12-49 verified _PATTERNS structure)

### Step 2: Add question templates and update severity mapping

**File:** `src/mirdan/core/semantic_analyzer.py` (verified via Read)
**Action:** Edit
**Details:**

Add 4 new entries to `_QUESTION_TEMPLATES` dict (after line 86):
```python
"concurrency": (
    "Line {line}: {context}. If this code runs concurrently (multiple "
    "coroutines, threads, or requests), verify shared mutable state is "
    "protected. Check: can two executions interleave and corrupt data?"
),
"boundary": (
    "Line {line}: {context}. Verify behavior at boundary values: "
    "zero denominator, empty collection, negative index, max-int overflow. "
    "Add explicit guards if the caller cannot guarantee safe inputs."
),
"error_propagation": (
    "Line {line}: {context}. Verify this error path preserves diagnostic "
    "context for callers. Check: is the original exception chained "
    "(Python: 'from e', Java: 'cause')? Does the caller handle this case?"
),
"state_machine": (
    "Line {line}: {context}. Verify ALL valid states are handled. "
    "Check: what happens for unexpected values? Are transitions validated "
    "(e.g., can't go from 'cancelled' to 'active')? Consider using an Enum."
),
```

Replace `_SECURITY_PATTERNS` frozenset (line 128) with a severity mapping dict:
```python
_PATTERN_SEVERITY: dict[str, str] = {
    "sql": "warning",
    "auth": "warning",
    "crypto": "warning",
    "concurrency": "warning",
    "error_propagation": "warning",
}
```

Update the severity lookup in `generate_checks()` (line 174) from:
```python
severity="warning" if pattern_type in _SECURITY_PATTERNS else "info"
```
to:
```python
severity=_PATTERN_SEVERITY.get(pattern_type, "info")
```

**Depends On:** Step 1
**Verify:** Read file, confirm 4 new templates and severity mapping dict exists
**Grounding:** Read of semantic_analyzer.py (lines 53-86 verified _QUESTION_TEMPLATES, line 128 verified _SECURITY_PATTERNS, line 174 verified severity lookup)

### Step 3: Add violation follow-ups and update analysis protocol

**File:** `src/mirdan/core/semantic_analyzer.py` (verified via Read)
**Action:** Edit
**Details:**

Add new entries to `_VIOLATION_FOLLOW_UPS` dict (after line 125):
```python
"DEEP001": (
    "Swallowed exception. What error was the caller supposed to handle? "
    "Trace the exception to its origin — is this transient (retry?) "
    "or permanent (fail fast?)?"
),
"DEEP004": (
    "Exception context lost. The original traceback is discarded. Use "
    "`raise NewError(...) from original_error` to preserve the chain."
),
```

Update `generate_analysis_protocol()` (line 219) to include new concerns:
```python
if check.concern in (
    "sql", "auth", "crypto",
    "concurrency", "error_propagation",
    "violation_deep_dive",
)
```

Update protocol type selection (line 225) to be dynamic:
```python
security_concerns = {"sql", "auth", "crypto"}
has_security = any(fa["concern"] in security_concerns for fa in focus_areas)
has_deep = any(fa["concern"] in ("concurrency", "error_propagation") for fa in focus_areas)

if has_security and has_deep:
    protocol_type = "comprehensive_analysis"
elif has_deep:
    protocol_type = "deep_analysis"
else:
    protocol_type = "security_flow_analysis"
```

Add `deep_analysis` field to SemanticConfig:
**File:** `src/mirdan/config.py` (verified via Read, line 24)
```python
class SemanticConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable semantic analysis")
    analysis_protocol: str = Field(
        default="security",
        pattern="^(none|security|comprehensive)$",
    )
    deep_analysis: bool = Field(
        default=True,
        description="Enable last-30% deep analysis patterns (concurrency, boundaries, state machines)",
    )
```

Update `generate_checks()` to respect the flag — skip new pattern categories when `deep_analysis=False`:
```python
# At the start of the pattern loop:
deep_patterns = frozenset({"concurrency", "boundary", "error_propagation", "state_machine"})
for pattern_type, patterns in _PATTERNS.items():
    if pattern_type in deep_patterns and not self._config.deep_analysis:
        continue
    ...
```

**Depends On:** Step 2
**Verify:** Read both files, confirm follow-ups added, analysis protocol updated, config extended
**Grounding:** Read of semantic_analyzer.py (lines 89-125 for follow-ups, lines 204-238 for protocol), config.py (line 24 for SemanticConfig)

### Step 4: Create DEEP001 SwallowedExceptionRule

**File:** `NEW: src/mirdan/core/rules/deep_analysis_rules.py` (parent dir verified via Glob)
**Action:** Write
**Details:**

Create file with DEEP001SwallowedExceptionRule:
- BaseRule subclass
- id = "DEEP001", name = "swallowed-exception"
- languages = frozenset({"python", "auto"})
- tier = RuleTier.FULL
- Uses Python `ast` module to find ExceptHandler nodes
- Checks if handler body (after docstring stripping) is ONLY: `pass`, `return`, `return None`, `...`
- Does NOT flag if body contains any call expression (logging, print, raise)
- Does NOT flag `contextlib.suppress` (detected via decorator/context pattern)
- category = "deep_analysis", severity = "warning"
- verifiable = True (AST-based, definitive)

**Depends On:** None
**Verify:** Read file, confirm class follows BaseRule pattern, AST logic is correct
**Grounding:** Read of test_body_rules.py (AST analysis pattern), ai005_error_handling.py (compiled rule pattern)

### Step 5: Add DEEP004 LostExceptionContextRule

**File:** `src/mirdan/core/rules/deep_analysis_rules.py` (created in Step 4)
**Action:** Edit (append)
**Details:**

Add DEEP004LostExceptionContextRule to same file:
- BaseRule subclass
- id = "DEEP004", name = "lost-exception-context"
- languages = frozenset({"python", "auto"})
- tier = RuleTier.FULL
- Uses Python `ast` to find ExceptHandler nodes that:
  1. Bind the exception (`as e`)
  2. Contain a Raise node with a different exception type
  3. The Raise node has `cause=None` (no `from` clause)
- Does NOT flag bare `raise` (re-raise) or `raise e` (explicit re-raise)
- Does NOT flag `raise ... from None` (explicit suppression)
- category = "deep_analysis", severity = "warning"
- verifiable = True

**Depends On:** Step 4
**Verify:** Read file, confirm both rules exist with correct AST logic
**Grounding:** Python ast module: ExceptHandler.name (binding), Raise.cause (from clause)

### Step 6: Register DEEP001/DEEP004 in AIQualityChecker

**File:** `src/mirdan/core/ai_quality_checker.py` (verified via Read — BaseRule registration at lines 72-106)
**Action:** Edit
**Details:**

Add import at the top of the file (after the existing rule imports, around line 32):
```python
from mirdan.core.rules.deep_analysis_rules import (
    DEEP001SwallowedExceptionRule,
    DEEP004LostExceptionContextRule,
)
```

In `AIQualityChecker.__init__()`, after the TEST rule registrations (after line 106):
```python
# DEEP analysis rules
self._registry.register(DEEP001SwallowedExceptionRule())
self._registry.register(DEEP004LostExceptionContextRule())
```

**Depends On:** Step 5
**Verify:** Read ai_quality_checker.py, confirm both rules are imported and registered after TEST rules
**Grounding:** Read of ai_quality_checker.py (RuleRegistry.register() pattern at lines 81-106)

### Step 7: Create tests for DEEP rules

**File:** `NEW: tests/test_deep_analysis_rules.py` (parent dir verified via Glob)
**Action:** Write
**Details:**

Test cases for DEEP001:
- `test_deep001_pass_in_except_body` — `except ValueError: pass` → violation
- `test_deep001_return_none_in_except` — `except: return None` → violation
- `test_deep001_return_bare_in_except` — `except: return` → violation
- `test_deep001_logging_in_except_no_violation` — `except: logger.error(e)` → no violation
- `test_deep001_raise_in_except_no_violation` — `except: raise` → no violation
- `test_deep001_contextlib_suppress_no_violation` — `with suppress(KeyError):` → no violation
- `test_deep001_non_python_skipped` — language="javascript" → no violation

Test cases for DEEP004:
- `test_deep004_raise_different_without_from` — `except ValueError as e: raise TypeError(...)` → violation
- `test_deep004_raise_with_from_no_violation` — `except ValueError as e: raise TypeError(...) from e` → no violation
- `test_deep004_bare_raise_no_violation` — `except ValueError as e: raise` → no violation
- `test_deep004_raise_from_none_no_violation` — `except: raise X from None` → no violation
- `test_deep004_no_binding_no_violation` — `except ValueError: raise TypeError(...)` → no violation (no `as e`)
- `test_deep004_non_python_skipped` — language="go" → no violation

**Depends On:** Step 5
**Verify:** Run `pytest tests/test_deep_analysis_rules.py -v`, all pass
**Grounding:** Test patterns from tests/test_test_quality_rules.py

### Step 8: Add semantic pattern tests and run full suite

**File:** `tests/test_semantic_analyzer.py` (verified via Glob, exists)
**Action:** Edit (append new test cases)
**Details:**

Add test cases for new semantic patterns:
- `test_concurrency_pattern_async_def` — async code → generates concurrency check
- `test_boundary_pattern_division` — `x / y` → generates boundary check
- `test_error_propagation_pattern_swallowed` — `except: pass` → generates error_propagation check
- `test_state_machine_pattern_string_comparison` — `status == "active"` → generates state_machine check
- `test_deep_analysis_disabled_skips_new_patterns` — deep_analysis=False → no new pattern checks generated
- `test_severity_mapping_concurrency_is_warning` — concurrency checks have severity "warning"
- `test_severity_mapping_boundary_is_info` — boundary checks have severity "info"
- `test_analysis_protocol_includes_concurrency` — protocol type = "deep_analysis" when only concurrency concerns
- `test_analysis_protocol_comprehensive` — protocol type = "comprehensive_analysis" when security + deep concerns

Run full test suite: `pytest tests/ -x -q`

**Depends On:** Steps 1-7
**Verify:** All tests pass, zero regressions
**Grounding:** Read of tests/test_semantic_analyzer.py (existing test patterns)

---

## Phase 2: Multi-Agent Coordination (Steps 9-17)

### Step 9: Add coordination dataclasses to models.py

**File:** `src/mirdan/models.py` (verified via Read)
**Action:** Edit (append after CompactState, before SemanticCheck, around line 468)
**Details:**

Add two dataclasses:
```python
@dataclass
class FileClaim:
    """A file ownership claim by an agent session."""
    session_id: str
    file_path: str
    claim_type: str  # "read" | "write"
    timestamp: float
    agent_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "file_path": self.file_path,
            "claim_type": self.claim_type,
            "agent_label": self.agent_label,
        }


@dataclass
class ConflictWarning:
    """A warning about potential multi-agent conflicts."""
    type: str  # "write_overlap" | "stale_read"
    message: str
    conflicting_sessions: list[str]
    file_path: str
    severity: str = "warning"  # "warning" | "info"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "message": self.message,
            "conflicting_sessions": self.conflicting_sessions,
            "file_path": self.file_path,
            "severity": self.severity,
        }
```

**Depends On:** None
**Verify:** Read models.py, confirm both dataclasses exist with to_dict methods
**Grounding:** Read of models.py (existing dataclass patterns)

### Step 10: Add CoordinationConfig to config.py

**File:** `src/mirdan/config.py` (verified via Read)
**Action:** Edit
**Details:**

Add new config class (after CeremonyConfig, before MirdanConfig):
```python
class CoordinationConfig(BaseModel):
    """Multi-agent coordination configuration."""
    enabled: bool = Field(
        default=True,
        description="Enable multi-agent file claim tracking and conflict detection.",
    )
    warn_on_write_overlap: bool = Field(
        default=True,
        description="Warn when multiple sessions claim write access to the same file.",
    )
    warn_on_stale_read: bool = Field(
        default=True,
        description="Warn when a file is modified by one session while another has a read claim.",
    )
```

Add to MirdanConfig (between `ceremony` field at line 378 and `rules` field at line 379):
```python
coordination: CoordinationConfig = Field(default_factory=CoordinationConfig)
```

**Depends On:** None
**Verify:** Read config.py, confirm CoordinationConfig exists and is wired into MirdanConfig
**Grounding:** Read of config.py (CeremonyConfig at line 318, MirdanConfig at line 352, `rules` is last field at line 379)

### Step 11: Create AgentCoordinator

**File:** `NEW: src/mirdan/core/agent_coordinator.py` (parent dir verified via Glob)
**Action:** Write
**Details:**

```python
class AgentCoordinator:
    """Coordinates file access across concurrent agent sessions.

    Maintains an in-memory registry of file claims and detects conflicts
    when multiple sessions operate on overlapping files. All data is
    session-scoped and expires with sessions.

    Thread safety: FastMCP serializes all tool calls on a single event
    loop, so no locks are needed.
    """

    def __init__(self, config: CoordinationConfig) -> None:
        self._config = config
        self._claims: dict[str, list[FileClaim]] = {}  # file_path → claims

    @property
    def is_enabled(self) -> bool:
        """Whether coordination is enabled (public API to avoid private attribute access)."""
        return self._config.enabled

    def claim_files(
        self,
        session_id: str,
        file_paths: list[str],
        claim_type: str,
        agent_label: str = "",
    ) -> list[ConflictWarning]:
        """Register file claims. Returns warnings for detected conflicts."""

    def release_session(self, session_id: str) -> None:
        """Release all claims for a session."""

    def check_conflicts(self, session_id: str, file_path: str) -> list[ConflictWarning]:
        """Check for conflicts affecting a specific file and session."""

    def get_active_claims(self) -> dict[str, list[FileClaim]]:
        """Get all active claims (for debugging/visibility)."""

    def cleanup_stale(self, active_session_ids: set[str]) -> int:
        """Remove claims for expired sessions. Returns count removed."""
```

Key logic:
- `claim_files`: For each file, check existing claims. If another session has a write claim on the same file → `write_overlap` warning. Store new claims.
- `check_conflicts`: For a given file_path, check if other sessions have write claims → `stale_read` warning.
- `cleanup_stale`: Called periodically (e.g., when evicting sessions). Removes claims whose session_id is not in the active set.

~150 lines total.

**Depends On:** Steps 9, 10
**Verify:** Read file, confirm all methods implemented, no external dependencies
**Grounding:** Session management pattern from session_manager.py, DI pattern from providers.py

### Step 12: Wire AgentCoordinator in providers.py

**File:** `src/mirdan/providers.py` (verified via Read)
**Action:** Edit
**Details:**

Add import:
```python
from mirdan.core.agent_coordinator import AgentCoordinator
```

In `ComponentProvider.__init__()`, create AgentCoordinator **BEFORE** SessionManager.
Insert after `semantic_analyzer` creation (line 103) and before `SessionManager` (line 123):
```python
self.agent_coordinator = AgentCoordinator(config.coordination)
```

> **CRITICAL ORDER**: AgentCoordinator must exist before SessionManager because
> Step 15 passes it to SessionManager for claim cleanup on session eviction.
> Currently SessionManager is at line 123. Insert AgentCoordinator around line 104.

Then update SessionManager creation (line 123) to pass the coordinator:
```python
self.session_manager = SessionManager(config.session, coordinator=self.agent_coordinator)
```

In `create_enhance_prompt_usecase()` (line 146), add parameter:
```python
agent_coordinator=self.agent_coordinator,
```

In `create_validate_code_usecase()` (line 165), add parameter:
```python
agent_coordinator=self.agent_coordinator,
```

**Depends On:** Step 11
**Verify:** Read providers.py, confirm (1) AgentCoordinator created before SessionManager, (2) coordinator passed to SessionManager, (3) coordinator passed to both use case factories
**Grounding:** Read of providers.py (SessionManager at line 123, CeremonyAdvisor at line 139, use case factories at lines 146-182)

### Step 13: Extend EnhancePromptUseCase for coordination

**File:** `src/mirdan/usecases/enhance_prompt.py` (verified via Read)
**Action:** Edit
**Details:**

Add constructor parameter:
```python
agent_coordinator: AgentCoordinator | None = None,
```
Store as `self._agent_coordinator`.

Add TYPE_CHECKING import:
```python
from mirdan.core.agent_coordinator import AgentCoordinator
```

In `execute()`, after session creation (around line 200), add file claiming:
```python
# Auto-claim files from extracted entities for coordination
coordination_warnings: list[dict[str, Any]] = []
if self._agent_coordinator is not None and self._agent_coordinator.is_enabled:
    from mirdan.models import EntityType
    file_entities = [e.value for e in intent.entities if e.type == EntityType.FILE_PATH]
    if file_entities:
        claim_type = "write" if intent.task_type in (
            TaskType.GENERATION, TaskType.REFACTOR
        ) else "read"
        warnings = self._agent_coordinator.claim_files(
            session.session_id, file_entities, claim_type
        )
        coordination_warnings = [w.to_dict() for w in warnings]
```

In the result dict construction (around line 290), add:
```python
if coordination_warnings:
    result_dict["coordination"] = {"warnings": coordination_warnings}
```

**Depends On:** Step 12
**Verify:** Read enhance_prompt.py, confirm claiming logic and response field
**Grounding:** Read of enhance_prompt.py (lines 194-306, entity extraction at intent.entities)

### Step 14: Extend ValidateCodeUseCase for coordination

**File:** `src/mirdan/usecases/validate_code.py` (verified via Read)
**Action:** Edit
**Details:**

Add constructor parameter:
```python
agent_coordinator: AgentCoordinator | None = None,
```
Store as `self._agent_coordinator`.

Add TYPE_CHECKING import for AgentCoordinator.

In `execute()`, after `timing_ms` (line 289) but before `output_formatter.format_validation_result()` (line 296), add conflict check:
```python
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
```

**Depends On:** Step 12
**Verify:** Read validate_code.py, confirm conflict check and response field
**Grounding:** Read of validate_code.py (lines 69-100)

### Step 15: Extend SessionManager for claim cleanup

**File:** `src/mirdan/core/session_manager.py` (verified via Read)
**Action:** Edit
**Details:**

Add AgentCoordinator to existing TYPE_CHECKING block (file already has `from __future__ import annotations` at line 3 and `TYPE_CHECKING` import at line 8):
```python
if TYPE_CHECKING:
    from mirdan.config import SessionConfig
    from mirdan.core.agent_coordinator import AgentCoordinator  # NEW
```

Add optional coordinator reference with proper type hint:
```python
def __init__(
    self, config: SessionConfig | None = None, coordinator: AgentCoordinator | None = None
) -> None:
    ...
    self._coordinator = coordinator
```

In `_evict_expired()` (line 98), after deleting sessions, trigger cleanup:
```python
def _evict_expired(self) -> None:
    expired = [sid for sid, s in self._sessions.items() if self._is_expired(s)]
    for sid in expired:
        del self._sessions[sid]
    if expired and self._coordinator is not None:
        self._coordinator.cleanup_stale(set(self._sessions.keys()))
```

In `remove()` (line 76), release claims:
```python
def remove(self, session_id: str) -> bool:
    removed = self._sessions.pop(session_id, None) is not None
    if removed and self._coordinator is not None:
        self._coordinator.release_session(session_id)
    return removed
```

In `_enforce_max_sessions()` (line 104), release claims for capacity-evicted sessions:
```python
def _enforce_max_sessions(self) -> None:
    """Evict oldest sessions if at capacity."""
    while len(self._sessions) >= self._config.max_sessions:
        oldest_id = min(self._sessions, key=lambda sid: self._sessions[sid].last_accessed)
        del self._sessions[oldest_id]
        if self._coordinator is not None:
            self._coordinator.release_session(oldest_id)
```

> **Note:** Both `_evict_expired()` and `_enforce_max_sessions()` can silently remove
> sessions. Without coordinator notification in both, file claims for evicted
> sessions would become orphaned. The providers.py wiring (passing coordinator
> to SessionManager) is already handled in Step 12 where the creation order is established.

**Depends On:** Steps 11, 12
**Verify:** Read session_manager.py, confirm cleanup triggers and TYPE_CHECKING import
**Grounding:** Read of session_manager.py (lines 76, 98-102). `from __future__ import annotations` already present at line 3.

### Step 16: Create tests for AgentCoordinator

**File:** `NEW: tests/test_agent_coordinator.py` (parent dir verified via Glob)
**Action:** Write
**Details:**

Test cases:
- `test_claim_single_file_no_conflict` — one session claims a file → no warnings
- `test_claim_write_overlap_warns` — two sessions claim write on same file → warning
- `test_claim_read_no_overlap` — two sessions claim read on same file → no warning
- `test_claim_write_then_read_warns` — session A writes, session B reads same file → stale_read warning
- `test_release_session_clears_claims` — release → no claims remain
- `test_cleanup_stale_removes_dead_sessions` — cleanup with active set → dead claims removed
- `test_check_conflicts_returns_warnings` — validate against file with other session's write claim
- `test_disabled_config_no_claims` — enabled=False → claim_files returns empty list
- `test_warn_on_write_overlap_false_suppresses` — config flag disables write overlap warnings
- `test_warn_on_stale_read_false_suppresses` — config flag disables stale read warnings

**Depends On:** Step 11
**Verify:** Run `pytest tests/test_agent_coordinator.py -v`, all pass
**Grounding:** Test patterns from tests/test_ceremony.py

### Step 17: Run full test suite for Phase 2

**Action:** Bash
**Details:** `cd mirdan && uv run pytest tests/ -x -q && uv run ruff check src/ && uv run mypy src/`
**Depends On:** Steps 9-16
**Verify:** All tests pass, zero lint/type errors
**Grounding:** Existing CI pattern

---

## Phase 3: Tidy First Refactoring Intelligence (Steps 18-24)

### Step 18: Add Tidy First dataclasses to models.py

**File:** `src/mirdan/models.py` (verified via Read)
**Action:** Edit (append after ConflictWarning from Step 9)
**Details:**

```python
@dataclass
class TidySuggestion:
    """A preparatory refactoring suggestion (Tidy First pattern)."""
    type: str  # "extract_method" | "simplify_conditional" | "reduce_nesting" | "split_file"
    file_path: str
    line: int | None = None
    description: str = ""
    effort: str = "small"  # "trivial" | "small" | "medium"
    reason: str = ""  # Why this helps the upcoming change

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "file_path": self.file_path,
            "description": self.description,
            "effort": self.effort,
        }
        if self.line is not None:
            result["line"] = self.line
        if self.reason:
            result["reason"] = self.reason
        return result


@dataclass
class TidyFirstAnalysis:
    """Result of Tidy First analysis on target files."""
    suggestions: list[TidySuggestion] = field(default_factory=list)
    target_files: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggestions": [s.to_dict() for s in self.suggestions],
            "target_files": self.target_files,
            "skipped_files": self.skipped_files,
        }
```

**Depends On:** None
**Verify:** Read models.py, confirm both dataclasses exist with to_dict methods
**Grounding:** Read of models.py (existing dataclass patterns)

### Step 19: Add TidyFirstConfig to config.py

**File:** `src/mirdan/config.py` (verified via Read)
**Action:** Edit
**Details:**

Add new config class (after CoordinationConfig from Step 10):
```python
class TidyFirstConfig(BaseModel):
    """Tidy First refactoring intelligence configuration."""
    enabled: bool = Field(
        default=True,
        description="Enable Tidy First analysis in enhance_prompt.",
    )
    max_suggestions: int = Field(
        default=3,
        description="Maximum tidy suggestions per enhance_prompt call.",
    )
    max_file_size_kb: int = Field(
        default=50,
        description="Skip files larger than this (KB). Prevents slow analysis on generated code.",
    )
    min_function_length: int = Field(
        default=25,
        description="Minimum function body lines to suggest extract_method.",
    )
    min_nesting_depth: int = Field(
        default=3,
        description="Minimum nesting depth to suggest simplify_conditional.",
    )
```

Add to MirdanConfig (after `coordination` field, before `rules`):
```python
tidy_first: TidyFirstConfig = Field(default_factory=TidyFirstConfig)
```

**Depends On:** None
**Verify:** Read config.py, confirm TidyFirstConfig exists and is wired into MirdanConfig
**Grounding:** Read of config.py (CeremonyConfig pattern at line 318)

### Step 20: Create TidyFirstAnalyzer

**File:** `NEW: src/mirdan/core/tidy_first.py` (parent dir verified via Glob)
**Action:** Write
**Details:**

```python
class TidyFirstAnalyzer:
    """Analyze target files for preparatory refactoring opportunities.

    Implements Kent Beck's "Tidy First" principle: make the change easy,
    then make the easy change. Runs during enhance_prompt to surface
    optional structural improvements before the main task begins.

    Pure analysis: reads files from disk, uses AST (Python) or regex
    (other languages), returns deterministic suggestions. No side effects.
    """

    def __init__(self, config: TidyFirstConfig) -> None:
        self._config = config

    def analyze(self, intent: Intent) -> TidyFirstAnalysis:
        """Analyze files referenced in intent for tidy opportunities."""

    def _analyze_file(
        self, code: str, file_path: str, language: str
    ) -> list[TidySuggestion]:
        """Analyze a single file for tidy opportunities."""

    def _analyze_python_ast(
        self, code: str, file_path: str
    ) -> list[TidySuggestion]:
        """Python-specific AST analysis for extract_method and nesting."""

    def _analyze_generic(
        self, code: str, file_path: str, language: str
    ) -> list[TidySuggestion]:
        """Language-agnostic regex analysis for file size and nesting."""

    def _check_long_functions_ast(
        self, tree: ast.Module, file_path: str
    ) -> list[TidySuggestion]:
        """Find functions exceeding min_function_length via AST."""

    def _check_deep_nesting_ast(
        self, tree: ast.Module, file_path: str
    ) -> list[TidySuggestion]:
        """Find deeply nested blocks via AST walk."""

    def _check_file_size(
        self, code: str, file_path: str
    ) -> list[TidySuggestion]:
        """Suggest split_file if file exceeds threshold."""
```

Key implementation details:
- `analyze()`: Extract FILE_PATH entities from intent. For each (up to 5), try to read from disk. Analyze. Cap at max_suggestions.
- `_check_long_functions_ast()`: Walk FunctionDef/AsyncFunctionDef. Calculate body length as `node.end_lineno - node.lineno`. If > min_function_length, suggest extract_method with the function name and line number.
- `_check_deep_nesting_ast()`: Recursive walk tracking nesting depth (If, For, While, With, Try increment depth). If max depth > min_nesting_depth, suggest simplify_conditional.
- `_check_file_size()`: Count non-empty lines. If > 300 (matches arch_max_file_length default), suggest split_file with effort="medium".
- File resolution: Try `Path(file_path)` first, then `Path.cwd() / file_path`. Skip on FileNotFoundError.
- Size guard: Skip files > max_file_size_kb * 1024 bytes.

~250 lines total.

**Depends On:** Steps 18, 19
**Verify:** Read file, confirm all methods implemented, AST logic correct
**Grounding:** AST pattern from test_body_rules.py, file reading from code_validator.py

### Step 21: Wire TidyFirstAnalyzer in providers.py

**File:** `src/mirdan/providers.py` (verified via Read)
**Action:** Edit
**Details:**

Add import:
```python
from mirdan.core.tidy_first import TidyFirstAnalyzer
```

In `ComponentProvider.__init__()`, after agent_coordinator:
```python
self.tidy_analyzer = TidyFirstAnalyzer(config.tidy_first)
```

In `create_enhance_prompt_usecase()`, add parameter:
```python
tidy_analyzer=self.tidy_analyzer,
```

**Depends On:** Step 20
**Verify:** Read providers.py, confirm analyzer is created and passed
**Grounding:** Read of providers.py (DI wiring pattern)

### Step 22: Extend EnhancePromptUseCase for Tidy First

**File:** `src/mirdan/usecases/enhance_prompt.py` (verified via Read)
**Action:** Edit
**Details:**

Add constructor parameter:
```python
tidy_analyzer: TidyFirstAnalyzer | None = None,
```
Store as `self._tidy_analyzer`.

Add TYPE_CHECKING import:
```python
from mirdan.core.tidy_first import TidyFirstAnalyzer
```

In `execute()`, after ceremony determination but before context gathering (around line 244), add:
```python
# Tidy First analysis — only at STANDARD+ ceremony for GENERATION/REFACTOR
tidy_analysis = None
if (
    self._tidy_analyzer is not None
    and level >= CeremonyLevel.STANDARD
    and intent.task_type in (TaskType.GENERATION, TaskType.REFACTOR)
):
    tidy_analysis = self._tidy_analyzer.analyze(intent)
```

In the result dict construction (around line 290), add:
```python
if tidy_analysis is not None and tidy_analysis.suggestions:
    result_dict["tidy_suggestions"] = tidy_analysis.to_dict()
```

**Depends On:** Steps 20, 21
**Verify:** Read enhance_prompt.py, confirm tidy analysis call and response field
**Grounding:** Read of enhance_prompt.py (ceremony level check pattern at line 203-232)

### Step 23: Extend Jinja2 templates and PromptComposer for Tidy First section

This step modifies 3 files: `base.j2` (new macro), `generation.j2` (new block), and `prompt_composer.py` (thread parameter through).

**File 1:** `src/mirdan/templates/base.j2` (verified via Read — 88 lines, 7 macros)
**Action:** Edit (append new macro after `tools_section` macro, before end of file)
**Details:**

Add `tidy_first_section` macro:
```jinja2
{% macro tidy_first_section(tidy_suggestions) -%}
{% if tidy_suggestions -%}
## Tidy First (Optional Preparatory Refactoring)
Before implementing the main change, consider these structural improvements that would make the change easier:
{% for s in tidy_suggestions -%}
- **{{ s.type }}** ({{ s.effort }}): {{ s.description }}{% if s.file_path %} [{{ s.file_path }}{% if s.line %}:{{ s.line }}{% endif %}]{% endif %}
{% endfor %}
These are suggestions, not requirements. Apply only if they simplify your task.
{%- endif %}
{%- endmacro %}
```

**File 2:** `src/mirdan/templates/generation.j2` (verified via Read — 28 lines)
**Action:** Edit
**Details:**

Update the import line (line 2) to include the new macro:
```jinja2
{% from "base.j2" import role_section, context_section, task_section, quality_requirements_section, constraints_section, verification_section, tools_section, tidy_first_section %}
```

Add the tidy block between `task_guidance` and `task_section` (after line 11, before line 12):
```jinja2
{% if tidy_suggestions -%}
{{ tidy_first_section(tidy_suggestions) }}

{% endif -%}
```

**File 3:** `src/mirdan/core/prompt_composer.py` (verified via Read — Jinja2 rendering at line 264)
**Action:** Edit
**Details:**

Update `compose()` method signature to accept tidy analysis:
```python
def compose(
    self,
    intent: Intent,
    context: ContextBundle,
    tool_recommendations: list[ToolRecommendation],
    extra_requirements: Sequence[str] = (),
    session: SessionContext | None = None,
    tidy_suggestions: list[dict[str, Any]] | None = None,
) -> EnhancedPrompt:
```

Pass `tidy_suggestions` through to `_build_prompt_text()`:
```python
enhanced_text = self._build_prompt_text(
    intent, context, quality_requirements, verification_steps, tool_recommendations,
    tidy_suggestions=tidy_suggestions,
)
```

Update `_build_prompt_text()` signature to accept `tidy_suggestions`:
```python
def _build_prompt_text(
    self,
    intent: Intent,
    context: ContextBundle,
    quality_requirements: list[str],
    verification_steps: list[str],
    tool_recommendations: list[ToolRecommendation],
    tidy_suggestions: list[dict[str, Any]] | None = None,
) -> str:
```

Add `tidy_suggestions` to the `template.render()` call (line 265):
```python
template = self._env.get_template("generation.j2")
return template.render(
    language=language,
    frameworks=frameworks,
    patterns_summary=context.summarize_patterns() if context.existing_patterns else None,
    tech_stack=tech_stack_str if context.tech_stack else None,
    original_prompt=intent.original_prompt,
    quality_requirements=quality_requirements if verbosity != "minimal" else [],
    constraints=constraints if verbosity != "minimal" else [],
    verification_steps=verification_steps if include_verification else [],
    tool_recommendations=tool_recommendations if include_tool_hints else [],
    verbosity=verbosity,
    include_verification=include_verification,
    include_tool_hints=include_tool_hints,
    task_guidance=task_guidance,
    tidy_suggestions=tidy_suggestions if verbosity != "minimal" else None,
).strip()
```

Update the EnhancePromptUseCase call to compose():
```python
enhanced = self._prompt_composer.compose(
    intent, context, tool_recommendations,
    extra_requirements=persistent_reqs,
    session=session,
    tidy_suggestions=(
        [s.to_dict() for s in tidy_analysis.suggestions]
        if tidy_analysis and tidy_analysis.suggestions else None
    ),
)
```

**Depends On:** Step 22
**Verify:** Read all 3 files: base.j2 (new macro exists), generation.j2 (new block exists, import updated), prompt_composer.py (new parameter threaded through compose → _build_prompt_text → template.render)
**Grounding:** Read of prompt_composer.py (compose() at line 99, _build_prompt_text() renders via Jinja2 at line 264), templates/generation.j2 (28 lines, macro imports at line 2), templates/base.j2 (88 lines, 7 macros)

### Step 24: Create tests for TidyFirstAnalyzer

**File:** `NEW: tests/test_tidy_first.py` (parent dir verified via Glob)
**Action:** Write
**Details:**

Test cases:
- `test_empty_intent_no_suggestions` — intent with no file entities → empty analysis
- `test_nonexistent_file_skipped` — file path doesn't exist → in skipped_files
- `test_long_function_detected` — Python file with 40-line function → extract_method suggestion
- `test_short_function_no_suggestion` — Python file with 10-line function → no suggestion
- `test_deep_nesting_detected` — Python file with 4-level nesting → simplify_conditional suggestion
- `test_shallow_nesting_no_suggestion` — Python file with 2-level nesting → no suggestion
- `test_large_file_detected` — File with 400 lines → split_file suggestion
- `test_max_suggestions_cap` — file with many issues → capped at max_suggestions
- `test_disabled_config_returns_empty` — enabled=False → empty analysis
- `test_binary_file_skipped` — binary file → skipped gracefully
- `test_syntax_error_falls_back_to_regex` — Python file with syntax error → still analyzes via regex
- `test_non_python_uses_generic` — JavaScript file → uses regex-based analysis
- `test_file_size_cap` — file > max_file_size_kb → skipped
- `test_only_runs_for_generation_refactor` — verify it's only called for correct task types (tested at use case level)
- `test_effort_sorting` — suggestions sorted by effort (trivial first)

Use tmp_path fixture for creating test files.

**Depends On:** Step 20
**Verify:** Run `pytest tests/test_tidy_first.py -v`, all pass
**Grounding:** Test patterns from tests/test_ceremony.py, tmp_path usage from tests/test_convention_extractor.py

---

## Phase 4: Integration & Finalization (Steps 25-28)

### Step 25: Run full test suite

**Action:** Bash
**Details:** `cd mirdan && uv run pytest tests/ -x -q`
**Depends On:** Steps 1-24
**Verify:** All tests pass (expected: ~2200+ including new tests)
**Grounding:** Current test count: 2132+

### Step 26: Run linters and type checker

**Action:** Bash
**Details:** `cd mirdan && uv run ruff check src/ tests/ && uv run mypy src/`
**Depends On:** Step 25
**Verify:** Zero violations from ruff and mypy
**Grounding:** Existing CI pattern from pyproject.toml

### Step 27: Update CHANGELOG.md

**File:** `CHANGELOG.md` (verified via Read)
**Action:** Edit (add to existing [1.9.0] section)
**Details:**

Add under `### Added` in the [1.9.0] section:

```markdown
- **Tidy First Intelligence** — `enhance_prompt` analyzes target files for preparatory
  refactoring opportunities before the main task begins. Detects long functions (extract_method),
  deep nesting (simplify_conditional), and oversized files (split_file). Suggestions are optional
  and effort-rated. Only activates at STANDARD+ ceremony for GENERATION/REFACTOR tasks.

- **Last 30% Semantic Analysis** — Four new semantic check categories detect the hard-to-catch
  patterns that cause production failures: concurrency hazards (shared mutable state in async code),
  boundary conditions (unchecked division, dynamic indexing), error propagation gaps (swallowed
  exceptions, lost context), and state machine coherence (unhandled states, string-based transitions).
  Two new compiled rules: DEEP001 (swallowed exception) and DEEP004 (lost exception context).

- **Multi-Agent Coordination** — File claim registry tracks which agent sessions are modifying
  which files. Detects write-write overlaps and stale-read conflicts across concurrent agents.
  Claims are session-scoped and expire automatically. Coordination warnings appear in both
  `enhance_prompt` and `validate_code_quality` responses.

- `deep_analysis` field on SemanticConfig (default: true) controls last-30% patterns
- `tidy_first` config section with `enabled`, `max_suggestions`, `max_file_size_kb`,
  `min_function_length`, `min_nesting_depth`
- `coordination` config section with `enabled`, `warn_on_write_overlap`, `warn_on_stale_read`
```

**Depends On:** Steps 1-24
**Verify:** Read CHANGELOG.md, confirm entries under [1.9.0]
**Grounding:** Read of CHANGELOG.md (existing format)

### Step 28: Final verification

**Action:** Bash
**Details:**
```bash
cd mirdan
# Verify version
python -c "from mirdan import __version__; print(__version__)"
# Verify new modules importable
python -c "from mirdan.core.tidy_first import TidyFirstAnalyzer; print('OK')"
python -c "from mirdan.core.agent_coordinator import AgentCoordinator; print('OK')"
python -c "from mirdan.core.rules.deep_analysis_rules import DEEP001SwallowedExceptionRule, DEEP004LostExceptionContextRule; print('OK')"
# Final test run
uv run pytest tests/ -q --tb=short
```

**Depends On:** Steps 25-27
**Verify:** Version = 1.9.0, all imports succeed, all tests pass
**Grounding:** __init__.py line 8

---

## Summary

| Metric | Value |
|--------|-------|
| **New files** | 6 (3 production modules + 3 test files) |
| **Modified files** | 13 (models, config, providers, 2 use cases, semantic_analyzer, ai_quality_checker, prompt_composer, session_manager, templates/base.j2, templates/generation.j2, CHANGELOG, + test extensions) |
| **New production lines** | ~1,200 estimated |
| **New test lines** | ~800 estimated |
| **New dependencies** | 0 (stdlib only) |
| **New MCP tools** | 0 (extends existing) |
| **New compiled rules** | 2 (DEEP001, DEEP004) |
| **New semantic patterns** | 4 (concurrency, boundary, error_propagation, state_machine) |
| **New config sections** | 3 (tidy_first, coordination, semantic.deep_analysis) |
| **Version** | 1.9.0 (unchanged, unreleased) |
| **Breaking changes** | 0 |

### Dependency Graph (Implementation Order)

```
Phase 1: Last 30% Semantic     Phase 2: Multi-Agent        Phase 3: Tidy First
┌─────────────────────────┐    ┌────────────────────────┐   ┌──────────────────────┐
│ Step 1: _PATTERNS        │    │ Step 9: models.py      │   │ Step 18: models.py   │
│ Step 2: _TEMPLATES       │    │ Step 10: config.py     │   │ Step 19: config.py   │
│ Step 3: Follow-ups+cfg   │    │ Step 11: coordinator   │   │ Step 20: analyzer    │
│ Step 4: DEEP001          │    │ Step 12: providers.py  │   │ Step 21: providers   │
│ Step 5: DEEP004          │    │ Step 13: enhance_prompt│   │ Step 22: enhance     │
│ Step 6: ai_quality_chk   │    │ Step 14: validate_code │   │ Step 23: composer+j2 │
│ Step 7: Tests            │    │ Step 15: session_mgr   │   │ Step 24: Tests       │
│ Step 8: Full suite       │    │ Step 16: Tests         │   └──────────────────────┘
└─────────────────────────┘    │ Step 17: Full suite    │
                                └────────────────────────┘

Phase 4: Integration (Steps 25-28)
```

All three phases are INDEPENDENT — no cross-dependencies. They modify overlapping files (models.py, config.py, providers.py, enhance_prompt.py) but at different locations with no conflicts.
