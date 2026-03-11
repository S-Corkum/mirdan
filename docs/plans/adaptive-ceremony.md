# Adaptive Ceremony — Implementation Plan

**Version:** 1.9.0
**Date:** 2026-03-11
**Status:** PLANNED

## Executive Summary

Add a thin routing layer that automatically scales Mirdan's enforcement depth based on task complexity signals. Micro-changes get fast feedback; complex changes get deep guidance. Validation integrity is never compromised — only the *guidance* side adapts.

---

## Research Notes (Pre-Plan Verification)

### Files Verified (Read)

| File | Key Structures | Lines |
|------|---------------|-------|
| `src/mirdan/models.py` | `OutputFormat(Enum)`, `ModelTier(Enum)`, `TaskType(Enum)`, `Intent(@dataclass)`, `SessionContext(@dataclass)`, `EnhancedPrompt(@dataclass)` | 626 |
| `src/mirdan/config.py` | `QualityConfig`, `EnhancementConfig`, `MirdanConfig`, all Pydantic BaseModel | 435 |
| `src/mirdan/server.py` | `enhance_prompt()` tool handler at line 109, `validate_code_quality()` at line 152, thin routing to use cases | 345 |
| `src/mirdan/usecases/enhance_prompt.py` | `EnhancePromptUseCase`, `execute()` at line 110, `_get_persistent_violation_reqs()` at line 37 | 248 |
| `src/mirdan/core/prompt_composer.py` | `PromptComposer`, `compose()` at line 99 (params: intent, context, tool_recs, extra_reqs, session — NO verbosity param), verbosity read from `self._config.verbosity` at line 241 | 363 |
| `src/mirdan/core/quality_profiles.py` | `QualityProfile(@dataclass)`, `BUILTIN_PROFILES` dict (7 profiles), `apply_profile()` — only `quality` section is applied back in ComponentProvider | 275 |
| `src/mirdan/core/rules/base.py` | `RuleTier(IntEnum)` QUICK=0/ESSENTIAL=1/FULL=2, `RuleContext(@dataclass)`, `BaseRule(ABC)` | ~100 |
| `src/mirdan/providers.py` | `ComponentProvider` with eager init in `__init__` (all components as instance attributes, NO @cached_property), `create_enhance_prompt_usecase()` at line 144 | 206 |
| `src/mirdan/__init__.py` | `__version__ = "1.8.0"` | 8 |

### Critical Verified Constraints

1. **PromptComposer.compose() has NO verbosity parameter** — verbosity is baked into `EnhancementConfig` at construction time (`self._config.verbosity` line 241). Cannot override per-call without API change.
2. **providers.py uses eager init** — all components initialized in `__init__`, stored as instance attributes. No lazy loading or `@cached_property`.
3. **apply_profile → ComponentProvider is incomplete** — `apply_profile()` sets `semantic`/`dependencies` sections in config_dict (in addition to `quality`), but ComponentProvider only applies the `quality` section back (lines 66-70). Adding a `ceremony` section to apply_profile would create dead code — the same way semantic/dependencies profile defaults are already dead code.
4. **analyze_only path (line 144) does NOT go through OutputFormatter** — returns plain dict directly.
5. **Tool recommendations in response** — AI reads `result_dict["tool_recommendations"]`, not the prompt text. PromptComposer's `include_tool_hints` only controls prompt text, not the response dict.
6. **Server tool tests** — `tests/test_server_tools.py` calls tool handlers via `.fn` attribute, `tests/test_server.py` has `TestEnhancePromptLogic`.

### Project Structure (Glob)

```
src/mirdan/
├── __init__.py              # Version: 1.8.0
├── models.py                # CeremonyLevel + CeremonyPolicy go here
├── config.py                # CeremonyConfig goes here
├── server.py                # ceremony_level param added here
├── providers.py             # Wires CeremonyAdvisor to use case (eager init pattern)
├── core/
│   ├── ceremony.py          # NEW — CeremonyAdvisor class
│   ├── intent_analyzer.py   # No changes needed
│   ├── orchestrator.py      # No changes needed
│   ├── prompt_composer.py   # No changes needed (ceremony does NOT control verbosity)
│   ├── session_manager.py   # No changes needed
│   ├── quality_profiles.py  # No changes (apply_profile pipeline is broken for non-quality sections)
│   └── rules/base.py        # No changes (RuleTier already exists)
├── usecases/
│   └── enhance_prompt.py    # Integration point for CeremonyAdvisor + tool rec filtering
tests/
├── test_ceremony.py         # NEW — unit tests for CeremonyAdvisor
├── test_server_tools.py     # Add ceremony_level tool handler tests
├── test_server.py           # Add ceremony_level component tests
```

### Dependencies Confirmed

- No new dependencies required (pyproject.toml unchanged)
- Uses existing: `IntEnum` (stdlib), Pydantic `BaseModel`, `@dataclass`

### Conventions (enyal)

- Session-aware logic lives in server.py/usecases (orchestration), NOT CodeValidator (keeps validator pure)
- Violation delta uses set-based rule ID comparison
- Enums follow existing patterns: `TaskType(Enum)`, `RuleTier(IntEnum)`, `OutputFormat(Enum)`
- Config sections are Pydantic BaseModel with Field defaults
- Use cases accept primitives, not model objects, at the boundary

### Similar Implementations (Pattern Reference)

- **context_level parameter**: Already routes to different context gathering depths in `enhance_prompt.py` line 204. CeremonyLevel drives context_level — this is the primary overhead reduction mechanism.
- **analyze_only mode**: Already exists at `enhance_prompt.py` line 144. MICRO ceremony builds on this pattern but adds ceremony metadata fields and runs through OutputFormatter for token budget awareness.
- **RuleTier system**: Already classifies rules by speed. `validate_quick(scope="essential")` already exists for fast validation.
- **ToolRecommendation.priority field**: At models.py line 170, recommendations have `priority` (e.g., "critical", "high", "medium"). LIGHT ceremony filters to `priority == "critical"` only.

### Design Decision: Verbosity NOT Controlled by Ceremony

`PromptComposer.compose()` reads verbosity from `self._config.verbosity` (EnhancementConfig), which is set at construction time. There is no per-call override. Rather than change the compose() API (which would touch PromptComposer, its tests, and the ValidateCodeUseCase which also calls it), ceremony does NOT attempt to control prompt text verbosity. The overhead reduction comes from:
1. **context_level** — the dominant factor (context7/enyal/filesystem gathering takes seconds)
2. **enhancement_mode** — analyze_only skips compose entirely for MICRO
3. **Tool recommendation filtering** — fewer MCPs recommended = fewer round-trips

Users who want comprehensive verbosity for THOROUGH tasks can set `enhancement.verbosity: comprehensive` in config.yaml independently.

---

## Architecture Decision

### What Changes

CeremonyAdvisor is a **stateless function** that maps `(Intent, prompt_length, session_state)` → `CeremonyPolicy`. The policy is a frozen dataclass that holds values for *existing parameters* (context_level, enhancement_mode). No new control systems — just a lookup table that composes existing mechanisms.

### What Does NOT Change

- `code_validator.py` — Validation integrity is never compromised
- `validate_code_quality` tool — No API changes
- `validate_quick` tool — No API changes
- `prompt_composer.py` — No changes (ceremony does NOT control verbosity; see design decision above)
- `session_manager.py` — No changes
- `session_tracker.py` — No changes
- `orchestrator.py` — No changes (tool recommendation filtering happens in the use case layer, not orchestrator)
- `quality_profiles.py` — No changes (apply_profile → ComponentProvider pipeline doesn't support non-quality sections; ceremony config is set via config.yaml directly)
- Any existing rule files — No changes
- Hook templates — No changes (ceremony is transparent to IDE integrations)

### Why This Is Zero Technical Debt

1. **No duplication** — CeremonyPolicy provides VALUES for existing parameters, doesn't reimplement them
2. **No coupling** — CeremonyAdvisor depends only on models.py (public types). Nothing depends on ceremony.py except the use case layer
3. **No feature flags** — `CeremonyConfig.enabled` is a permanent config option, not a temporary flag
4. **No dead code risk** — Integrated into the main enhance_prompt path
5. **Fully testable** — Pure function: deterministic inputs → deterministic outputs
6. **Backward compatible** — Default behavior unchanged (auto → STANDARD for typical prompts)
7. **No partial fixes** — Does not modify apply_profile or quality_profiles.py to avoid dead code (apply_profile sets config sections that ComponentProvider doesn't read back)

---

## CeremonyLevel Enum

```python
class CeremonyLevel(IntEnum):
    """Ceremony depth for quality orchestration.

    Controls how much guidance enhance_prompt provides.
    Validation integrity is never affected — only the guidance scales.
    Orderable: MICRO < LIGHT < STANDARD < THOROUGH.
    """
    MICRO = 0       # Minimal guidance, fast feedback
    LIGHT = 1       # Lightweight guidance, essential rules
    STANDARD = 2    # Full quality sandwich (current default)
    THOROUGH = 3    # Deep analysis, architecture review
```

## CeremonyPolicy Table

| Level | enhancement_mode | context_level | recommended_validation | filter_tool_recs | session |
|-------|-----------------|---------------|----------------------|------------------|---------|
| MICRO | `analyze_only` | `none` | `quick_essential` | N/A (no recs) | Not surfaced* |
| LIGHT | `enhance` | `minimal` | `full` | critical only | Yes |
| STANDARD | `enhance` | `auto` | `full` | none (all kept) | Yes |
| THOROUGH | `enhance` | `comprehensive` | `full` | none (all kept) | Yes |

**Notes:**
- Verbosity is NOT in this table. PromptComposer.compose() reads verbosity from
  EnhancementConfig (set at init time). Ceremony does not attempt to override it per-call.
  See "Design Decision: Verbosity NOT Controlled by Ceremony" above.
- *MICRO "Not surfaced": A session is created internally (needed for escalation check),
  but `session_id` is not included in the MICRO response. The session expires via TTL.
  This avoids restructuring the session creation code for a negligible cost.

## Scoring Algorithm

```
Base score from task type:
  GENERATION: 2, REFACTOR: 2, DEBUG: 1, TEST: 1, REVIEW: 1, DOCUMENTATION: 0, PLANNING: 4, UNKNOWN: 1

Modifiers:
  + 1  if compound task (len(task_types) > 1)
  + 1  per detected framework (max 2)
  + 1  if prompt_length >= 200 chars
  + 1  if prompt_length >= 500 chars
  + 1  if entities detected (file paths, function names)

Mapping:
  score 0-1  → MICRO
  score 2    → LIGHT
  score 3-5  → STANDARD
  score 6+   → THOROUGH
```

## Escalation Rules (can only escalate, never de-escalate)

| Trigger | Escalate To | Rationale |
|---------|-------------|-----------|
| `touches_security=True` | min STANDARD | Security always gets full sandwich |
| `touches_rag=True` | min STANDARD | RAG complexity needs full guidance |
| `touches_knowledge_graph=True` | min STANDARD | KG complexity needs full guidance |
| `ambiguity_score >= threshold` | min STANDARD | Ambiguous tasks need more guidance |
| `task_type == PLANNING` | THOROUGH | Plans always need deep analysis |
| `session.unresolved_errors > 0` | +1 level | Persistent failures need more ceremony |

## Calibration Verification

| Prompt Example | Base Score | Escalation | Final Level |
|----------------|-----------|------------|-------------|
| "Fix the typo in README" | DEBUG(1) = 1 | none | MICRO |
| "Update version in config" | GEN(2) = 2 | none | LIGHT |
| "Refactor DB module for connection pooling" | REFACT(2) + entity(1) = 3 | none | STANDARD |
| "Implement JWT auth for FastAPI" | GEN(2) + framework(1) = 3 | security→STANDARD | STANDARD |
| "Add a React dashboard with real-time WebSocket charts, REST API integration using FastAPI and PostgreSQL, including auth, pagination, and error handling for the /api/v2/metrics endpoint with test coverage" | GEN(2) + frameworks(2) + compound(1) + long(1) + entity(1) = 7 | none | THOROUGH |
| "Create implementation plan for microservice" | PLANNING(4) = 4 | PLANNING→THOROUGH | THOROUGH |
| *Typical enhance_prompt usage* | 3-5 | varies | STANDARD (matches current behavior) |

---

## Implementation Steps

### Step 1: Add CeremonyLevel and CeremonyPolicy to models.py

**File:** `src/mirdan/models.py` (verified via Read, line 8-36 has existing enums)
**Action:** Edit — add after `TaskType` enum (after line 36)

**Details:**
- Add `CeremonyLevel(IntEnum)` with 4 values: MICRO=0, LIGHT=1, STANDARD=2, THOROUGH=3
- Add `CeremonyPolicy` frozen dataclass with fields: `level`, `enhancement_mode`, `context_level`, `recommended_validation`, `filter_tool_recs` (bool)
- Import `IntEnum` from enum (models.py currently imports only `Enum` at line 4; add `IntEnum`)
- `CeremonyPolicy` does NOT include `verbosity` — see design decision above

**Depends On:** Nothing (first step)

**Verify:** Read models.py, confirm CeremonyLevel and CeremonyPolicy exist and are importable

**Grounding:**
- Enum pattern: Read of models.py confirmed TaskType at line 26, OutputFormat at line 8
- IntEnum pattern: Read of rules/base.py confirmed RuleTier(IntEnum) at line 13
- Frozen dataclass pattern: Standard Python, no project-specific considerations

---

### Step 2: Add CeremonyConfig to config.py

**File:** `src/mirdan/config.py` (verified via Read, MirdanConfig at line 318)
**Action:** Edit — add CeremonyConfig class before MirdanConfig, add field to MirdanConfig

**Details:**
- Add `CeremonyConfig(BaseModel)` with fields:
  - `enabled: bool = True` — master switch
  - `default_level: str = "auto"` — default ceremony level
  - `min_level: str = "micro"` — floor level (enterprise teams can set "standard")
  - `security_escalation: bool = True` — always escalate security to STANDARD+
  - `ambiguity_escalation: bool = True` — escalate high-ambiguity to STANDARD+
  - `ambiguity_threshold: float = 0.6` — threshold for ambiguity escalation
- Add `ceremony: CeremonyConfig = Field(default_factory=CeremonyConfig)` to `MirdanConfig` (after line 343, before `rules`)
- Follow existing pattern: Pydantic BaseModel with Field descriptions

**Depends On:** Step 1 (models.py must be updated first for conceptual ordering)

**Verify:** Read config.py, confirm CeremonyConfig class exists. Instantiate `MirdanConfig()` and confirm `config.ceremony.enabled == True`

**Grounding:**
- Config pattern: Read of config.py confirmed all config sections are Pydantic BaseModel with Field defaults
- MirdanConfig structure: Read confirmed it aggregates all config sections at lines 318-344
- Existing ceremony-like config: `EnhancementConfig` at line 81 follows same pattern

---

### Step 3: Create CeremonyAdvisor in core/ceremony.py

**File:** `NEW: src/mirdan/core/ceremony.py` (parent dir verified via Glob — core/ has 30+ existing modules)
**Action:** Write new file (~140 lines)

**Details:**
- Import: `CeremonyLevel`, `CeremonyPolicy` from models, `CeremonyConfig` from config, `Intent`, `SessionContext`, `TaskType` from models
- Class `CeremonyAdvisor`:
  - Class constant `POLICIES: dict[CeremonyLevel, CeremonyPolicy]` — maps each level to its frozen policy
  - Class constant `_TASK_TYPE_SCORES: dict[TaskType, int]` — scoring weights per task type
  - `__init__(self, config: CeremonyConfig | None = None)` — stores config
  - `determine_level(self, intent: Intent, prompt_length: int, session: SessionContext | None = None) -> CeremonyLevel`:
    - If `not config.enabled`: return `CeremonyLevel.STANDARD`
    - If `config.default_level != "auto"`: use as base (with escalations still applied)
    - Otherwise: call `_estimate_base_level(intent, prompt_length)`
    - Apply `_apply_escalations(base, intent, session)`
    - Clamp to `config.min_level`
  - `get_policy(self, level: CeremonyLevel) -> CeremonyPolicy` — lookup from POLICIES dict
  - `explain(self, level: CeremonyLevel, intent: Intent) -> str` — human-readable explanation of why this level was chosen. Returns a short string like `"STANDARD: generation task with 1 framework, security escalation"`. Used in response metadata so the AI assistant understands the ceremony decision.
  - `_estimate_base_level(self, intent: Intent, prompt_length: int) -> CeremonyLevel` — scoring algorithm
  - `_apply_escalations(self, base: CeremonyLevel, intent: Intent, session: SessionContext | None) -> CeremonyLevel` — escalation rules

**Depends On:** Steps 1-2

**Verify:** Import `CeremonyAdvisor` in a Python shell. Instantiate with default config. Call `determine_level` with a mock Intent. Confirm return type is `CeremonyLevel`. Call `explain(level, intent)` and confirm it returns a non-empty string.

**Grounding:**
- Intent fields: Read of models.py confirmed `touches_security`, `touches_rag`, `touches_knowledge_graph`, `ambiguity_score`, `task_type`, `task_types`, `frameworks`, `entities` at lines 48-63
- SessionContext fields: Read confirmed `validation_count`, `unresolved_errors` at lines 101-106
- TaskType values: Read confirmed 8 values at lines 26-36
- Scoring weights calibrated against prompt examples (see Calibration Verification table above)
- `explain()` method: Required by Step 4 integration (`self._ceremony_advisor.explain(level, intent)` in response metadata)

---

### Step 4: Integrate CeremonyAdvisor into EnhancePromptUseCase

**File:** `src/mirdan/usecases/enhance_prompt.py` (verified via Read, execute() at line 110)
**Action:** Edit — add CeremonyAdvisor dependency and integrate into execute flow

**Details:**
- Add `CeremonyAdvisor` to TYPE_CHECKING imports (line 18-32)
- Add `ceremony_advisor: CeremonyAdvisor` to `__init__` params (after `config` at line 93)
- Add `ceremony_level: str = "auto"` parameter to `execute()` (after `session_id` at line 117)
- Integration logic (insert **after session creation at line 194**, before persistent violation reqs at line 196):

  **IMPORTANT:** The ceremony block MUST go after the session creation block (lines 191-194)
  because `determine_level()` needs the session for the `unresolved_errors > 0` escalation rule.
  The `session` variable is created by the walrus operator at line 191 and is always set
  (either from lookup or fresh creation) by line 194.

```python
# Determine ceremony level (after session is available for escalation checks)
# NOTE: Add CeremonyLevel to the existing import at line 136:
#   from mirdan.models import CeremonyLevel, ContextBundle, TaskType

if ceremony_level != "auto":
    # Explicit override — bypasses scoring and escalation (user knows best)
    try:
        level = CeremonyLevel[ceremony_level.upper()]
    except KeyError:
        level = CeremonyLevel.STANDARD
    policy = self._ceremony_advisor.get_policy(level)
else:
    level = self._ceremony_advisor.determine_level(
        intent, len(prompt), session=session
    )
    policy = self._ceremony_advisor.get_policy(level)

# MICRO: return minimal analysis with ceremony metadata.
# NOTE: Unlike the existing analyze_only path (line 144) which returns a plain dict,
# MICRO goes through OutputFormatter for token budget awareness and consistent output.
# A session was created above but is not surfaced in the response — MICRO
# responses are stateless from the client's perspective.
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

# Apply ceremony policy to context_level (the primary overhead reduction mechanism)
effective_context_level = policy.context_level if context_level == "auto" else context_level
```

- Pass `effective_context_level` to context gathering (replace `context_level` at line 204/207)
- After tool recommendations are stored on session (line 201), apply LIGHT filtering if policy
  requires it. **IMPORTANT:** Filtering must go AFTER `session.tool_recommendations = [...]` at
  line 201 so the session retains the unfiltered list for compliance tracking:
  ```python
  # LIGHT ceremony: filter tool recommendations to critical-priority only
  # (session already stores unfiltered recs above for compliance tracking)
  if policy.filter_tool_recs:
      tool_recommendations = [r for r in tool_recommendations if r.priority == "critical"]
  ```
- After building result_dict, add ceremony fields:
  ```python
  result_dict["ceremony_level"] = level.name.lower()
  result_dict["recommended_validation"] = policy.recommended_validation
  result_dict["ceremony_reason"] = self._ceremony_advisor.explain(level, intent)
  ```

**Depends On:** Steps 1-3

**Verify:** Read enhance_prompt.py after edit. Confirm ceremony block is AFTER session creation (lines 191-194) and BEFORE persistent_reqs (line 196). Call with ceremony_level="auto" on a short prompt — confirm ceremony_level appears in response. Call with a security-touching prompt — confirm escalation. Call with LIGHT-level prompt — confirm tool_recommendations are filtered to critical only.

**Grounding:**
- execute() flow: Read of enhance_prompt.py confirmed line-by-line flow at lines 110-247
- Session creation: Read confirmed walrus operator at line 191 creates `existing`, fallback `session = self._session_manager.create_from_intent(intent)` at line 194. `session` variable is always set after line 194.
- analyze_only pattern: Read confirmed existing early-return path at lines 144-167 (returns plain dict without OutputFormatter)
- context_level usage: Read confirmed it's checked at lines 204-207
- Tool recommendations: Read confirmed `self._mcp_orchestrator.suggest_tools()` at line 200 returns `list[ToolRecommendation]` with `.priority` field
- Output format: Read confirmed result_dict is built incrementally and formatted at line 243
- ToolRecommendation.priority: Read of models.py confirmed field at line 170 with values "critical", "high", "medium"

---

### Step 5: Wire CeremonyAdvisor in ComponentProvider

**File:** `src/mirdan/providers.py` (verified via Read, ComponentProvider at line 50, __init__ at line 57, create_enhance_prompt_usecase at line 144)
**Action:** Edit — instantiate CeremonyAdvisor in __init__ and pass to use case factory

**Details:**
- Add import at top of file: `from mirdan.core.ceremony import CeremonyAdvisor`
- Add eager initialization in `__init__` (after existing component inits, following the established pattern):
  ```python
  self.ceremony_advisor = CeremonyAdvisor(config.ceremony)
  ```
- Add `ceremony_advisor=self.ceremony_advisor` to `create_enhance_prompt_usecase()` return at line 147-160

**IMPORTANT:** providers.py uses eager init in `__init__` for ALL components (instance attributes, NOT `@cached_property`). CeremonyAdvisor follows this same pattern.

**Depends On:** Steps 3, 4

**Verify:** Read providers.py. Confirm `self.ceremony_advisor = CeremonyAdvisor(...)` in `__init__`. Confirm `ceremony_advisor=self.ceremony_advisor` in `create_enhance_prompt_usecase()`.

**Grounding:**
- Provider pattern: Read of providers.py confirmed ALL components use eager init in `__init__` as instance attributes (e.g., `self.intent_analyzer = IntentAnalyzer(...)`, `self.prompt_composer = PromptComposer(...)`)
- create_enhance_prompt_usecase: Read confirmed it passes all dependencies at lines 147-160
- No `@cached_property` or `@property` patterns exist in this file — strictly eager init

---

### Step 6: Add ceremony_level parameter to server.py enhance_prompt

**File:** `src/mirdan/server.py` (verified via Read, enhance_prompt handler at line 109)
**Action:** Edit — add `ceremony_level` parameter to tool handler, pass to use case

**Details:**
- Add `ceremony_level: str = "auto"` parameter to `enhance_prompt()` function (after `session_id` at line 116):
  ```python
  ceremony_level: str = "auto",
  ```
- Update docstring to document the parameter:
  ```
  ceremony_level: Guidance depth (auto|micro|light|standard|thorough).
                  "auto" scales based on task complexity. "micro" for trivial
                  changes, "thorough" for complex multi-framework tasks.
                  Security-touching tasks escalate to at least "standard".
  ```
- Pass to use case execute call (add at line 148):
  ```python
  ceremony_level=ceremony_level,
  ```

**Depends On:** Step 4

**Verify:** Read server.py. Confirm `ceremony_level` param in function signature and in `uc.execute()` call.

**Grounding:**
- Tool handler pattern: Read confirmed thin routing at lines 109-149
- Parameter passing: Read confirmed all params are forwarded to `uc.execute()` at lines 142-149

---

### Step 7: Create unit tests for CeremonyAdvisor

**File:** `NEW: tests/test_ceremony.py` (parent dir verified — tests/ has 86 existing test files)
**Action:** Write new file (~300 lines)

**Details:**
Test classes following project patterns (class-based, pytest fixtures, type hints):

```python
class TestCeremonyLevel:
    """Tests for CeremonyLevel enum."""
    def test_ordering(self) -> None: ...
    def test_values(self) -> None: ...

class TestCeremonyPolicy:
    """Tests for CeremonyPolicy frozen dataclass."""
    def test_micro_policy(self) -> None: ...
    def test_light_policy(self) -> None: ...
    def test_standard_policy(self) -> None: ...
    def test_thorough_policy(self) -> None: ...
    def test_policies_are_frozen(self) -> None: ...

class TestCeremonyAdvisorBaseLevel:
    """Tests for base level estimation from intent signals."""
    def test_short_debug_prompt_is_micro(self) -> None: ...
    def test_simple_generation_is_light(self) -> None: ...
    def test_generation_with_framework_is_standard(self) -> None: ...
    def test_long_compound_task_is_thorough(self) -> None: ...
    def test_planning_task_is_thorough(self) -> None: ...
    def test_documentation_only_is_micro(self) -> None: ...
    def test_entities_increase_score(self) -> None: ...
    def test_multiple_frameworks_increase_score(self) -> None: ...

class TestCeremonyAdvisorExplain:
    """Tests for explain() method."""
    def test_explain_returns_nonempty_string(self) -> None: ...
    def test_explain_includes_level_name(self) -> None: ...
    def test_explain_mentions_escalation_reason(self) -> None: ...

class TestCeremonyEscalation:
    """Tests for escalation rules."""
    def test_security_escalates_to_standard(self) -> None: ...
    def test_rag_escalates_to_standard(self) -> None: ...
    def test_knowledge_graph_escalates_to_standard(self) -> None: ...
    def test_high_ambiguity_escalates_to_standard(self) -> None: ...
    def test_planning_always_thorough(self) -> None: ...
    def test_persistent_violations_escalate_one_level(self) -> None: ...
    def test_escalation_never_decreases(self) -> None: ...
    def test_security_on_micro_escalates_to_standard(self) -> None: ...

class TestCeremonyConfig:
    """Tests for configuration overrides."""
    def test_disabled_always_returns_standard(self) -> None: ...
    def test_min_level_prevents_downgrade(self) -> None: ...
    def test_default_level_override(self) -> None: ...
    def test_security_escalation_disabled(self) -> None: ...
    def test_ambiguity_threshold_custom(self) -> None: ...

class TestCeremonyCalibration:
    """Regression tests for calibration — typical prompts produce expected levels."""
    def test_typo_fix_is_micro(self) -> None: ...
    def test_version_update_is_light(self) -> None: ...
    def test_db_refactor_is_standard(self) -> None: ...
    def test_jwt_auth_is_standard_via_security(self) -> None: ...
    def test_complex_multi_framework_task_is_thorough(self) -> None: ...
    def test_implementation_plan_is_thorough(self) -> None: ...
```

Approximately 33-38 test methods covering:
- Enum ordering and values
- Policy lookup for each level
- Base level scoring for various intent combinations
- explain() method output
- All 6 escalation rules
- Config overrides (enabled, min_level, default_level, security_escalation, ambiguity_threshold)
- Calibration regression tests (prevents drift)

**Depends On:** Steps 1-3

**Verify:** Run `uv run pytest tests/test_ceremony.py -v`. All tests pass.

**Grounding:**
- Test patterns: Research confirmed class-based tests, pytest fixtures, type hints, no conftest.py
- Fixture pattern: `@pytest.fixture` returning instance, e.g., `def advisor() -> CeremonyAdvisor`
- Assertion pattern: `assert level == CeremonyLevel.STANDARD`

---

### Step 8: Add integration tests for ceremony in enhance_prompt

**File:** `tests/test_server_tools.py` (verified via Read — contains `TestEnhancePromptTool` at line 78, calls tool handlers via `.fn` attribute at line 24)
**Action:** Edit — add test cases for ceremony_level parameter to `TestEnhancePromptTool` class

**Details:**
Add test methods to `TestEnhancePromptTool`:
- `test_enhance_prompt_returns_ceremony_level` — default "auto" returns ceremony_level in response
- `test_enhance_prompt_micro_returns_minimal_response` — short prompt returns analyze_only-style response with ceremony metadata
- `test_enhance_prompt_explicit_ceremony_override` — passing ceremony_level="thorough" forces thorough
- `test_enhance_prompt_security_escalates_micro_to_standard` — security flag overrides micro
- `test_enhance_prompt_light_filters_tool_recs` — LIGHT-level response only includes critical-priority tool recommendations

Follow existing test pattern: `_enhance_prompt = server_mod.enhance_prompt.fn` with `patch.object` for context_aggregator.

**Depends On:** Steps 1-7

**Verify:** Run `uv run pytest tests/test_server_tools.py -k ceremony -v`. All tests pass.

**Grounding:**
- Test file: Read of `tests/test_server_tools.py` confirmed `_enhance_prompt` function reference at line 24 and `TestEnhancePromptTool` class at line 78
- Test pattern: Read confirmed tests use `patch.object(server_mod._get_provider().context_aggregator, "gather_all", ...)` for mocking
- Tool handler tests: Read confirmed thin-layer testing approach — calls `.fn` directly

---

### Step 9: Update version and changelog

**File:** `src/mirdan/__init__.py` (verified via Read, `__version__ = "1.8.0"` at line 8)
**Action:** Edit — bump version to `1.9.0`

**File:** `CHANGELOG.md` (exists in mirdan root)
**Action:** Edit — add 1.9.0 entry

**Details:**
```markdown
## [1.9.0] - 2026-03-XX

### Added
- **Adaptive Ceremony**: enhance_prompt automatically scales guidance depth based on
  task complexity. Micro-changes get fast feedback; complex tasks get deep analysis.
  Validation integrity is never compromised.
- `CeremonyLevel` enum: MICRO, LIGHT, STANDARD, THOROUGH
- `ceremony_level` parameter on enhance_prompt (default: "auto")
- `ceremony` config section with `enabled`, `default_level`, `min_level`,
  `security_escalation`, `ambiguity_escalation`, `ambiguity_threshold`
- Response fields: `ceremony_level`, `recommended_validation`, `ceremony_reason`
- LIGHT ceremony filters tool recommendations to critical-priority only
```

**Depends On:** Steps 1-8 (all implementation complete)

**Verify:** Read `__init__.py` confirms `1.9.0`. Read CHANGELOG.md confirms entry.

**Grounding:**
- Version pattern: Read confirmed `__version__` at line 8
- Changelog pattern: Previous versions follow `## [X.Y.Z] - YYYY-MM-DD` format with Added/Changed/Fixed sections

---

## Step Validation Matrix

| Step | File Exists | Location Valid | API Verified | Import Available | Convention Aligned | Directory Exists |
|------|------------|----------------|-------------|-----------------|-------------------|-----------------|
| 1 | ✓ Read | ✓ Lines 8-36 | N/A | ✓ IntEnum in stdlib | ✓ Enum pattern | N/A |
| 2 | ✓ Read | ✓ Lines 318-344 | N/A | ✓ Pydantic BaseModel | ✓ Config pattern | N/A |
| 3 | NEW | N/A | N/A | ✓ All from models/config | ✓ core/ module pattern | ✓ Glob verified |
| 4 | ✓ Read | ✓ Lines 110-247 | N/A | ✓ TYPE_CHECKING | ✓ Use case pattern | N/A |
| 5 | ✓ Read | ✓ Lines 57, 144-160 | N/A | ✓ Direct import | ✓ Eager init pattern | N/A |
| 6 | ✓ Read | ✓ Lines 109-149 | N/A | N/A (pass-through) | ✓ Tool handler pattern | N/A |
| 7 | NEW | N/A | N/A | ✓ From step 3 | ✓ Test patterns verified | ✓ tests/ exists |
| 8 | ✓ Read | ✓ Line 78 (class) | N/A | ✓ | ✓ Test patterns verified | N/A |
| 9 | ✓ Read | ✓ Line 8 | N/A | N/A | ✓ Semver pattern | N/A |

## Completeness Checklist

- [x] Every aspect of the feature has a step
- [x] Steps are in correct dependency order
- [x] Every step has Grounding citations
- [x] No missing steps (imports, tests, exports, configs)
- [x] Each step is executable with ONLY the information provided
- [x] Each step has a verification method
- [x] No vague language ("should", "probably", "likely")

## Risk Assessment

| Risk | Probability | Mitigation |
|------|------------|------------|
| Scoring algorithm miscalibrated | Low | Calibration regression tests in Step 7 |
| Enterprise teams don't want MICRO ever | None | `min_level: standard` in config |
| AI assistant ignores recommended_validation | None | Advisory only, no enforcement needed |
| Existing tests break | Very Low | No existing behavior changes under default config |

## Metrics (Post-Implementation)

Track via `get_quality_trends`:
- Distribution of ceremony levels across sessions
- Average enhance_prompt response time by ceremony level
- Validation pass rate by ceremony level (should be equal — validation isn't weakened)
- Token consumption per ceremony level (MICRO should use ~80% fewer tokens than STANDARD)
