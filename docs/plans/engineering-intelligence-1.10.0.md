# Engineering Intelligence Platform — Mirdan 1.10.0

**Created:** 2026-03-11
**Status:** PLANNED
**Version:** 1.9.0 → 1.10.0
**Theme:** From Quality Gate to Engineering Intelligence

---

## Narrative

Mirdan 1.10.0 transforms from an "AI Code Quality Orchestrator" into an "AI Engineering Intelligence Platform." Three features bridge the gap between non-unicorn engineers and 10x engineers by targeting the three biggest differentiators: decision quality (50% of the gap), codebase awareness (25%), and cognitive discipline (15%).

**Zero new MCP tools. Zero new dependencies. All backward compatible.**

All three features enrich existing `enhance_prompt` and `validate_code_quality` output — the tool surface stays identical.

---

## Research Notes (Pre-Plan Verification)

### Files Verified (Read)
| File | Key Content | Lines |
|------|-------------|-------|
| `config.py` | MirdanConfig with 15 config sections, last field at line 427 (tidy_first) | 519 |
| `models.py` | 20+ dataclasses, TidyFirstAnalysis ends at line 547, Violation at 634 | 736 |
| `server.py` | 7 MCP tools, thin routing via ComponentProvider | 351 |
| `usecases/enhance_prompt.py` | Full flow: intent→ceremony→tidy→persistent_reqs→tools→context→compose | 347 |
| `usecases/validate_code.py` | Full flow: session→validate→linter→explain→persist→track→delta→semantic | 535 |
| `providers.py` | DI container, tidy_analyzer at line 110, factory methods lines 152-213 | 218 |
| `core/ceremony.py` | CeremonyAdvisor, scoring + escalation, POLICIES dict | 203 |
| `core/tidy_first.py` | **Reference pattern** for new analyzers — config→model→analyze→output | 294 |
| `core/prompt_composer.py` | TASK_GUIDANCE dict lines 29-78, compose() at line 100 | 368 |
| `core/semantic_analyzer.py` | _PATTERNS dict, _QUESTION_TEMPLATES, generate_checks() | 329 |
| `core/output_formatter.py` | Token-budget compression: FULL→COMPACT→MINIMAL→MICRO | — |
| `standards/architecture.yaml` | Simple YAML format: clean_architecture, solid, general | 15 |
| `__init__.py` | `__version__ = "1.9.0"` at line 8 | 8 |

### Project Structure (Glob Verified)
```
mirdan/src/mirdan/
├── core/
│   ├── rules/          # 14 compiled rule files
│   ├── gatherers/      # BaseGatherer + 4 concrete gatherers
│   ├── ceremony.py     # CeremonyAdvisor
│   ├── tidy_first.py   # TidyFirstAnalyzer (REFERENCE PATTERN)
│   ├── agent_coordinator.py
│   ├── semantic_analyzer.py
│   └── ... (20+ modules)
├── standards/
│   ├── frameworks/     # 33+ YAML files
│   ├── languages/      # 7 YAML files
│   ├── architecture.yaml
│   └── ... (6 YAML files)
├── usecases/           # 7 use case files
├── templates/          # Jinja2 templates
├── config.py
├── models.py
├── providers.py
├── server.py
└── __init__.py
```

### Dependencies Confirmed
- `pyyaml>=6.0` (already imported in config.py)
- `ast` (stdlib, already used in tidy_first.py)
- `fnmatch` (stdlib, already used in config.py:ThresholdsConfig)
- **Zero new dependencies needed**

### Conventions (enyal)
- All prior features follow: Config class → Model dataclasses → Core analyzer → Providers wire → UseCase integration
- Session-aware logic lives in server.py/use cases, not in validators
- Validation integrity never compromised — only guidance scales
- All config classes use Pydantic BaseModel with Field descriptors

### Similar Implementations Verified
| Feature | Pattern | Files Changed |
|---------|---------|---------------|
| TidyFirst (1.9.0) | Config→Model→Analyzer→Wire→Integrate | config.py, models.py, NEW core/tidy_first.py, providers.py, enhance_prompt.py |
| AgentCoordinator (1.9.0) | Config→Model→Coordinator→Wire→Integrate | config.py, models.py, NEW core/agent_coordinator.py, providers.py, enhance_prompt.py, validate_code.py |
| Ceremony (1.9.0) | Config→Model→Advisor→Wire→Integrate | config.py, models.py, NEW core/ceremony.py, providers.py, enhance_prompt.py |

All three features in this plan follow the **identical pattern**.

---

## Feature Specifications

### Feature 1: Decision Intelligence Engine

**Purpose:** When enhance_prompt detects a task with decision-worthy patterns, surface structured trade-off analysis. Not prescriptive — questions and trade-offs.

**Unicorn trait addressed:** Better decisions, more often (50% of gap)

**Data flow:**
```
enhance_prompt
  → IntentAnalyzer detects task type + prompt keywords
  → DecisionAnalyzer.analyze(intent) matches triggers from YAML templates
  → Returns DecisionGuidance with approaches, trade-offs, senior questions
  → Added to result_dict["decision_guidance"]
```

**Ceremony gate:** STANDARD+ only (same as TidyFirst)

**YAML template format** (`standards/decisions/caching.yaml`):
```yaml
name: "Caching Strategy"
triggers:
  - "cache"
  - "caching"
  - "memoize"
  - "ttl"
  - "invalidat"
approaches:
  - name: "In-process cache (lru_cache / Map)"
    when_best: "Single instance, read-heavy, short-lived data"
    when_avoid: "Multi-instance, needs shared invalidation"
    complexity: "low"
  - name: "Distributed cache (Redis / Memcached)"
    when_best: "Multi-instance, complex invalidation, shared state"
    when_avoid: "Simple single-instance, latency-sensitive hot path"
    complexity: "medium"
  - name: "HTTP cache headers (Cache-Control, ETag)"
    when_best: "Public data, CDN in front, cacheable responses"
    when_avoid: "Dynamic per-user data, real-time requirements"
    complexity: "low"
senior_questions:
  - "How many instances serve this endpoint?"
  - "What is the acceptable staleness window?"
  - "What triggers cache invalidation?"
  - "What happens on cache failure — degrade gracefully or error?"
```

**8 decision domains:** caching, authentication, state_management, data_access, error_handling, api_design, testing_strategy, configuration

---

### Feature 2: Cognitive Guardrails + Confidence Calibration

#### 2a: Cognitive Guardrails (enhance_prompt)

**Purpose:** Surface 2-3 domain-specific pre-flight thinking prompts before coding starts. Different from quality_requirements — these are THINKING prompts a senior would mention.

**Unicorn trait addressed:** Cognitive discipline, production awareness

**YAML format** (`standards/guardrails.yaml`):
```yaml
domains:
  payment:
    triggers: ["payment", "billing", "charge", "invoice", "subscription", "stripe"]
    guardrails:
      - "Consider idempotency — what happens if this operation runs twice?"
      - "Consider partial failure — what if payment succeeds but DB write fails?"
      - "Consider audit trail — every financial transaction needs logging"
  authentication:
    triggers: ["auth", "login", "session", "token", "password", "oauth", "jwt"]
    guardrails:
      - "Consider session invalidation — what happens when credentials change?"
      - "Consider rate limiting — prevent brute force attacks"
      - "Consider least privilege — minimize token scopes and permissions"
  data_migration:
    triggers: ["migration", "migrate", "schema change", "alter table", "backfill"]
    guardrails:
      - "Consider rollback — can this migration be reversed safely?"
      - "Consider zero-downtime — will this lock tables during deployment?"
      - "Consider in-flight data — what about active transactions?"
```

**Ceremony gate:** STANDARD+ only. Max 3 guardrails per response.

#### 2b: Confidence Calibration (validate_code_quality)

**Purpose:** After validation, provide calibrated confidence (HIGH/MEDIUM/LOW) plus ONE attention_focus item — the most important thing to manually verify.

**Unicorn trait addressed:** Trust calibration, preventing cognitive offloading

**Scoring logic:**
```
Start: HIGH
If any error violations → LOW
If any security violations → LOW
If warning count > 3 → MEDIUM
If semantic checks with "warning" severity → MEDIUM
If no test_file provided and code modifies logic → MEDIUM

attention_focus = highest-severity semantic check question
  OR "All rule checks passed — verify business logic correctness"
```

**Always active** — no ceremony gating. Small output footprint.

---

### Feature 3: Architectural Drift Detection

**Purpose:** Codify intended architecture in `.mirdan/architecture.yaml`, detect violations during validation.

**Unicorn trait addressed:** Future-cost awareness, codebase mental model (25% of gap)

**Architecture model format** (`.mirdan/architecture.yaml`):
```yaml
version: "1.0"
layers:
  - name: presentation
    patterns: ["api/**", "routes/**", "handlers/**"]
    allowed_imports: ["service", "models", "schemas"]
    forbidden_imports: ["database", "repositories"]
  - name: service
    patterns: ["services/**", "usecases/**"]
    allowed_imports: ["models", "database", "repositories"]
    forbidden_imports: ["presentation"]
  - name: database
    patterns: ["db/**", "repositories/**"]
    allowed_imports: ["models"]
    forbidden_imports: ["service", "presentation"]
```

**New validation rules:**
- `ARCH004`: layer_violation — file in layer X imports from forbidden layer Y
- `ARCH005`: unauthorized_dependency — import not in allowed_imports for layer

**Import extraction:**
- Python: AST-based (accurate, uses existing ast import pattern)
- Other languages: Regex-based (import/require/use detection)

**Integration points:**
1. `validate_code_quality`: When file_path provided + architecture.yaml exists → run analysis
2. `enhance_prompt`: At STANDARD+ → warn about architectural implications
3. `scan_conventions --architecture`: Generate initial architecture.yaml from import graph

---

## Implementation Plan

### Summary

| Phase | Feature | Steps | New Files | Modified Files | Est. Lines |
|-------|---------|-------|-----------|----------------|------------|
| 1 | Foundation (all) | 1-4 | 0 | 2 | ~140 prod |
| 2 | Confidence Calibration | 5-8 | 2 | 2 | ~200 prod, ~120 test |
| 3 | Decision + Guardrails | 9-16 | 12 | 2 | ~770 prod, ~270 test |
| 4 | Architecture Drift | 17-24 | 4 | 3 | ~520 prod, ~280 test |
| 5 | Polish | 25-29 | 0 | 5 | ~100 |
| **Total** | | **29** | **18** | **9** | **~1,730 prod, ~670 test** |

---

### Phase 1: Foundation (Config + Models)

#### Step 1: Add config classes

**File:** `mirdan/src/mirdan/config.py` (verified via Read, 519 lines)

**Action:** Edit — add 3 new BaseModel classes BEFORE MirdanConfig (after line 396, before class MirdanConfig)

**Details:**
```python
class DecisionConfig(BaseModel):
    """Decision intelligence configuration."""
    enabled: bool = Field(default=True, description="Enable decision trade-off analysis in enhance_prompt.")
    max_decisions: int = Field(default=1, description="Maximum decision domains to surface per prompt.")

class GuardrailConfig(BaseModel):
    """Cognitive guardrails configuration."""
    enabled: bool = Field(default=True, description="Enable domain-aware pre-flight guardrails.")
    max_guardrails: int = Field(default=3, description="Maximum guardrail items per prompt.")

class ArchitectureConfig(BaseModel):
    """Architectural drift detection configuration."""
    enabled: bool = Field(default=True, description="Enable architecture model validation when .mirdan/architecture.yaml exists.")
    warn_in_prompt: bool = Field(default=True, description="Surface architectural context in enhance_prompt.")
```

**Depends On:** None

**Verify:** Read file, confirm classes parse. `cd mirdan && uv run python -c "from mirdan.config import DecisionConfig, GuardrailConfig, ArchitectureConfig"`

**Grounding:** Read of config.py lines 323-396 confirmed BaseModel pattern with Field descriptors.

---

#### Step 2: Wire config fields into MirdanConfig

**File:** `mirdan/src/mirdan/config.py` (verified via Read)

**Action:** Edit — add 3 new fields to MirdanConfig class after line 427 (`tidy_first: TidyFirstConfig`)

**Details:**
```python
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    architecture: ArchitectureConfig = Field(default_factory=ArchitectureConfig)
```

**Depends On:** Step 1

**Verify:** `cd mirdan && uv run python -c "from mirdan.config import MirdanConfig; c = MirdanConfig(); print(c.decision.enabled, c.guardrails.max_guardrails, c.architecture.enabled)"`

**Grounding:** Read of config.py lines 399-428 confirmed MirdanConfig field pattern.

---

#### Step 3: Add model dataclasses

**File:** `mirdan/src/mirdan/models.py` (verified via Read, 736 lines)

**Action:** Edit — add 6 new dataclasses after TidyFirstAnalysis (after line 547)

**Details:**
```python
@dataclass
class DecisionApproach:
    """A single approach option within a decision domain."""
    name: str
    when_best: str
    when_avoid: str
    complexity: str = "medium"  # "low" | "medium" | "high"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "when_best": self.when_best,
                "when_avoid": self.when_avoid, "complexity": self.complexity}

@dataclass
class DecisionGuidance:
    """Trade-off analysis for a detected decision domain."""
    domain: str
    approaches: list[DecisionApproach] = field(default_factory=list)
    senior_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"domain": self.domain,
                "approaches": [a.to_dict() for a in self.approaches],
                "senior_questions": self.senior_questions}

@dataclass
class GuardrailAnalysis:
    """Domain-aware pre-flight cognitive guardrails."""
    domain: str
    guardrails: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"domain": self.domain, "guardrails": self.guardrails}

@dataclass
class ConfidenceAssessment:
    """Calibrated confidence level for validated code."""
    level: str  # "high" | "medium" | "low"
    reason: str
    attention_focus: str  # ONE thing to manually verify

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "reason": self.reason,
                "attention_focus": self.attention_focus}

@dataclass
class ArchLayer:
    """A layer in the architecture model."""
    name: str
    patterns: list[str] = field(default_factory=list)
    allowed_imports: list[str] = field(default_factory=list)
    forbidden_imports: list[str] = field(default_factory=list)

@dataclass
class ArchDriftResult:
    """Result of architectural drift analysis on a file."""
    violations: list[Violation] = field(default_factory=list)
    file_layer: str = ""
    context_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.violations:
            result["violations"] = [v.to_dict() for v in self.violations]
        if self.file_layer:
            result["file_layer"] = self.file_layer
        if self.context_warnings:
            result["context_warnings"] = self.context_warnings
        return result
```

**Depends On:** None (Violation already defined at line 634)

**Verify:** `cd mirdan && uv run python -c "from mirdan.models import DecisionGuidance, ConfidenceAssessment, ArchDriftResult"`

**Grounding:** Read of models.py lines 510-547 confirmed dataclass + to_dict() pattern.

---

#### Step 4: Tests for configs and models

**File:** NEW `mirdan/tests/test_engineering_intelligence_models.py`

**Action:** Write — unit tests for all new config defaults and model serialization

**Details:**
- Test each config class has correct defaults
- Test each model's to_dict() serialization
- Test MirdanConfig includes new fields
- Test ConfidenceAssessment levels
- Test DecisionGuidance with empty and populated approaches

**Depends On:** Steps 1-3

**Verify:** `cd mirdan && uv run pytest tests/test_engineering_intelligence_models.py -v`

**Grounding:** Glob confirmed tests/ directory exists with existing test files.

---

### Phase 2: Confidence Calibration

#### Step 5: Implement ConfidenceCalibrator

**File:** NEW `mirdan/src/mirdan/core/confidence.py`

**Action:** Write — ~80 lines

**Details:**
```python
"""Confidence calibration for validated code.

Aggregates validation signals into a calibrated confidence level
(HIGH/MEDIUM/LOW) with a single attention_focus item pointing the
developer to the most important manual verification.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from mirdan.models import ConfidenceAssessment, SemanticCheck, Violation

if TYPE_CHECKING:
    from mirdan.models import SessionContext

class ConfidenceCalibrator:
    """Assess confidence in validated code."""

    def assess(
        self,
        violations: list[Violation],
        semantic_checks: list[SemanticCheck],
        session: SessionContext | None = None,
        test_file: str = "",
    ) -> ConfidenceAssessment:
        """Produce a calibrated confidence assessment."""
        level = "high"
        reasons: list[str] = []

        error_count = sum(1 for v in violations if v.severity == "error")
        warning_count = sum(1 for v in violations if v.severity == "warning")
        security_count = sum(1 for v in violations if v.category == "security")

        if error_count > 0:
            level = "low"
            reasons.append(f"{error_count} error(s) found")
        if security_count > 0:
            level = "low"
            reasons.append(f"{security_count} security issue(s)")
        if warning_count > 3 and level != "low":
            level = "medium"
            reasons.append(f"{warning_count} warnings")

        warning_semantic = [s for s in semantic_checks if s.severity == "warning"]
        if warning_semantic and level == "high":
            level = "medium"
            reasons.append(f"{len(warning_semantic)} semantic concern(s) to review")

        if not test_file and level == "high":
            level = "medium"
            reasons.append("no test file cross-referenced")

        reason = "; ".join(reasons) if reasons else "all checks passed, no concerns flagged"

        # Attention focus: highest-severity semantic check
        attention_focus = "All rule checks passed — verify business logic correctness"
        for sev in ("critical", "warning", "info"):
            for check in semantic_checks:
                if check.severity == sev:
                    attention_focus = check.question
                    break
            else:
                continue
            break

        return ConfidenceAssessment(
            level=level, reason=reason, attention_focus=attention_focus
        )
```

**Depends On:** Step 3 (ConfidenceAssessment model)

**Verify:** Read file, `cd mirdan && uv run python -c "from mirdan.core.confidence import ConfidenceCalibrator"`, `uv run mypy src/mirdan/core/confidence.py`

**Grounding:** Read of validate_code.py lines 137-244 confirmed all input signals available.

---

#### Step 6: Wire ConfidenceCalibrator into providers.py

**File:** `mirdan/src/mirdan/providers.py` (verified via Read, 218 lines)

**Action:** Edit at 3 locations:
1. Add import after line 16 (after ceremony import): `from mirdan.core.confidence import ConfidenceCalibrator`
2. Create instance after line 138 (after session_tracker): `self.confidence_calibrator = ConfidenceCalibrator()`
3. Pass to factory at line 191 (in create_validate_code_usecase): `confidence_calibrator=self.confidence_calibrator,`

**Depends On:** Step 5

**Verify:** Read file, `cd mirdan && uv run ruff check src/mirdan/providers.py`

**Grounding:** Read of providers.py confirmed import pattern at lines 10-42, instance creation at lines 108-148, factory at lines 173-191.

---

#### Step 7: Integrate into ValidateCodeUseCase

**File:** `mirdan/src/mirdan/usecases/validate_code.py` (verified via Read, 535 lines)

**Action:** Edit at 3 locations:
1. Add to TYPE_CHECKING imports (after line 30): `from mirdan.core.confidence import ConfidenceCalibrator`
2. Accept in __init__ (after line 55, agent_coordinator param): `confidence_calibrator: ConfidenceCalibrator | None = None,`
   And store: `self._confidence_calibrator = confidence_calibrator`
3. Add confidence assessment after analysis_protocol section (after line 244):
```python
        # Confidence calibration
        if self._confidence_calibrator is not None:
            _semantic_list = (
                [SemanticCheck(**s) for s in output.get("semantic_checks", [])]
                if "semantic_checks" in output
                else []
            )
            confidence = self._confidence_calibrator.assess(
                violations=result.violations,
                semantic_checks=semantic_checks if semantic_checks else [],
                session=session,
                test_file=test_file,
            )
            output["confidence"] = confidence.to_dict()
```

**Depends On:** Steps 5-6

**Verify:** Read file, `cd mirdan && uv run mypy src/mirdan/usecases/validate_code.py`

**Grounding:** Read of validate_code.py confirmed output dict assembly pattern at lines 202-306, semantic_checks variable available at line 228.

---

#### Step 8: Tests for confidence calibration

**File:** NEW `mirdan/tests/test_confidence.py`

**Action:** Write — ~120 lines

**Details:**
- Test HIGH confidence: no violations, no semantic warnings
- Test LOW confidence: error violations present
- Test LOW confidence: security violations present
- Test MEDIUM confidence: >3 warnings
- Test MEDIUM confidence: warning-severity semantic checks
- Test MEDIUM confidence: no test_file
- Test attention_focus picks highest-severity semantic check
- Test attention_focus default when no semantic checks
- Test with empty inputs (no violations, no checks)

**Depends On:** Step 5

**Verify:** `cd mirdan && uv run pytest tests/test_confidence.py -v`

**Grounding:** Read of core/confidence.py (Step 5 design) confirmed all paths.

---

### Phase 3: Decision Intelligence + Cognitive Guardrails

#### Step 9: Create decision YAML templates

**File:** NEW `mirdan/src/mirdan/standards/decisions/` directory with 8 YAML files

**Action:** Write 8 files:
- `caching.yaml` — In-process vs distributed vs HTTP cache
- `authentication.yaml` — Session vs JWT vs OAuth
- `state_management.yaml` — Local vs shared vs event-sourced
- `data_access.yaml` — ORM vs query builder vs raw SQL
- `error_handling.yaml` — Exceptions vs result types vs error codes
- `api_design.yaml` — REST vs GraphQL vs RPC
- `testing_strategy.yaml` — Unit vs integration vs e2e balance
- `configuration.yaml` — Env vars vs config files vs feature flags

Each file follows the format:
```yaml
name: "Domain Name"
triggers: [list of keyword triggers]
approaches:
  - name: "Approach Name"
    when_best: "When to use"
    when_avoid: "When not to use"
    complexity: "low|medium|high"
senior_questions:
  - "Question a senior would ask"
```

**Depends On:** None

**Verify:** `cd mirdan && uv run python -c "import yaml; [yaml.safe_load(open(f'src/mirdan/standards/decisions/{n}.yaml')) for n in ['caching','authentication','state_management','data_access','error_handling','api_design','testing_strategy','configuration']]"`

**Grounding:** Read of standards/architecture.yaml confirmed YAML format.

---

#### Step 10: Implement DecisionAnalyzer

**File:** NEW `mirdan/src/mirdan/core/decision_analyzer.py`

**Action:** Write — ~150 lines

**Details:**
```python
"""Decision intelligence — surfaces trade-off analysis for detected decision domains.

Loads YAML decision templates, matches triggers against intent and prompt text,
returns structured guidance. Config-gated, ceremony-gated in enhance_prompt.
"""
class DecisionAnalyzer:
    def __init__(self, config: DecisionConfig):
        self._config = config
        self._templates: list[dict] = []
        self._load_templates()

    def _load_templates(self) -> None:
        """Load all YAML templates from standards/decisions/ directory."""
        # Path relative to this module
        ...

    def analyze(self, intent: Intent) -> list[DecisionGuidance]:
        """Match intent against decision templates, return top matches."""
        if not self._config.enabled:
            return []
        # 1. Combine prompt text + entity values into search text
        # 2. For each template, check if any trigger appears in search text
        # 3. Score by number of trigger matches
        # 4. Return top max_decisions results as DecisionGuidance objects
        ...
```

Key design: trigger matching is case-insensitive substring match on the original prompt text. Simple and fast.

**Depends On:** Steps 1-3 (config + models), Step 9 (YAML templates)

**Verify:** Read file, `cd mirdan && uv run mypy src/mirdan/core/decision_analyzer.py`

**Grounding:** Read of tidy_first.py confirmed analyzer class pattern (init with config, analyze(intent) method, Path-based resource loading).

---

#### Step 11: Create guardrails YAML

**File:** NEW `mirdan/src/mirdan/standards/guardrails.yaml`

**Action:** Write — ~100 lines

**Details:** Domain-triggered guardrail checklists covering:
- payment/billing
- authentication/authorization
- data migration/schema
- file upload/processing
- caching/invalidation
- real-time/websocket
- third-party integration/API
- concurrency/async
- deployment/infrastructure
- user data/privacy (GDPR, PII)

Each domain: 2-4 trigger words, 2-3 guardrail items (thinking prompts, not rules).

**Depends On:** None

**Verify:** `cd mirdan && uv run python -c "import yaml; d = yaml.safe_load(open('src/mirdan/standards/guardrails.yaml')); print(len(d['domains']), 'domains')"`

**Grounding:** Read of standards/architecture.yaml confirmed YAML format.

---

#### Step 12: Implement GuardrailAnalyzer

**File:** NEW `mirdan/src/mirdan/core/guardrail_analyzer.py`

**Action:** Write — ~100 lines

**Details:**
```python
"""Cognitive guardrails — domain-aware pre-flight thinking prompts.

Surfaces 2-3 domain-specific considerations a senior engineer would
mention BEFORE coding starts. Different from quality_requirements
(coding rules) — these are THINKING prompts.
"""
class GuardrailAnalyzer:
    def __init__(self, config: GuardrailConfig):
        self._config = config
        self._domains: dict[str, dict] = {}
        self._load_domains()

    def analyze(self, intent: Intent) -> list[GuardrailAnalysis]:
        """Match intent against guardrail domains, return top matches."""
        if not self._config.enabled:
            return []
        # 1. Build search text from prompt + entities
        # 2. Match triggers, score by match count
        # 3. Return top results (capped by max_guardrails across all domains)
        ...
```

**Depends On:** Steps 1-3, Step 11

**Verify:** Read file, `cd mirdan && uv run mypy src/mirdan/core/guardrail_analyzer.py`

**Grounding:** Read of tidy_first.py confirmed analyzer pattern.

---

#### Step 13: Wire into providers.py

**File:** `mirdan/src/mirdan/providers.py` (verified via Read)

**Action:** Edit at 3 locations:
1. Add imports (after line 34, after tidy_first import):
   ```python
   from mirdan.core.decision_analyzer import DecisionAnalyzer
   from mirdan.core.guardrail_analyzer import GuardrailAnalyzer
   ```
2. Create instances (after line 110, after tidy_analyzer):
   ```python
   self.decision_analyzer = DecisionAnalyzer(config.decision)
   self.guardrail_analyzer = GuardrailAnalyzer(config.guardrails)
   ```
3. Pass to EnhancePromptUseCase factory (after line 171, after tidy_analyzer param):
   ```python
   decision_analyzer=self.decision_analyzer,
   guardrail_analyzer=self.guardrail_analyzer,
   ```

**Depends On:** Steps 10, 12

**Verify:** Read file, `cd mirdan && uv run ruff check src/mirdan/providers.py`

**Grounding:** Read of providers.py confirmed pattern at lines 110-171.

---

#### Step 14: Integrate into EnhancePromptUseCase

**File:** `mirdan/src/mirdan/usecases/enhance_prompt.py` (verified via Read, 347 lines)

**Action:** Edit at 4 locations:
1. Add TYPE_CHECKING imports (after line 34):
   ```python
   from mirdan.core.decision_analyzer import DecisionAnalyzer
   from mirdan.core.guardrail_analyzer import GuardrailAnalyzer
   ```
2. Accept in __init__ (after line 101, after tidy_analyzer param):
   ```python
   decision_analyzer: DecisionAnalyzer | None = None,
   guardrail_analyzer: GuardrailAnalyzer | None = None,
   ```
   And store as `self._decision_analyzer` and `self._guardrail_analyzer`
3. Call analyzers (after line 272, after tidy analysis block):
   ```python
        # Decision intelligence — only at STANDARD+ for decision-worthy tasks
        decision_guidance = None
        if (
            self._decision_analyzer is not None
            and level >= CeremonyLevel.STANDARD
        ):
            decision_guidance = self._decision_analyzer.analyze(intent)

        # Cognitive guardrails — only at STANDARD+
        guardrail_analysis = None
        if (
            self._guardrail_analyzer is not None
            and level >= CeremonyLevel.STANDARD
        ):
            guardrail_analysis = self._guardrail_analyzer.analyze(intent)
   ```
4. Add to result_dict (after line 336, after tidy_suggestions):
   ```python
        if decision_guidance:
            result_dict["decision_guidance"] = [d.to_dict() for d in decision_guidance]

        if guardrail_analysis:
            result_dict["cognitive_guardrails"] = [g.to_dict() for g in guardrail_analysis]
   ```

**Depends On:** Step 13

**Verify:** Read file, `cd mirdan && uv run mypy src/mirdan/usecases/enhance_prompt.py`

**Grounding:** Read of enhance_prompt.py lines 265-336 confirmed ceremony gating and output dict pattern.

---

#### Step 15: Tests for DecisionAnalyzer

**File:** NEW `mirdan/tests/test_decision_analyzer.py`

**Action:** Write — ~150 lines

**Details:**
- Test trigger matching: "add caching to endpoint" → returns caching domain
- Test no match: "fix typo in README" → returns empty
- Test max_decisions cap: only top N returned
- Test disabled config: returns empty
- Test multiple domain matches: returns highest-scored
- Test case-insensitive matching
- Test YAML template loading (valid structure)

**Depends On:** Step 10

**Verify:** `cd mirdan && uv run pytest tests/test_decision_analyzer.py -v`

**Grounding:** Read of decision_analyzer.py design (Step 10).

---

#### Step 16: Tests for GuardrailAnalyzer

**File:** NEW `mirdan/tests/test_guardrail_analyzer.py`

**Action:** Write — ~120 lines

**Details:**
- Test domain matching: "implement payment webhook" → returns payment guardrails
- Test no match: "rename variable" → returns empty
- Test max_guardrails cap
- Test disabled config
- Test multiple domain match (payment + auth) → combined guardrails capped
- Test YAML loading

**Depends On:** Step 12

**Verify:** `cd mirdan && uv run pytest tests/test_guardrail_analyzer.py -v`

**Grounding:** Read of guardrail_analyzer.py design (Step 12).

---

### Phase 4: Architectural Drift Detection

#### Step 17: Implement import extractor

**File:** NEW `mirdan/src/mirdan/core/import_extractor.py`

**Action:** Write — ~120 lines

**Details:**
```python
"""Import extraction from source code.

Python: Uses ast module for accurate extraction.
Other languages: Regex-based detection of import/require/use statements.
Returns list of (module_path, line_number) tuples.
"""
def extract_python_imports(code: str) -> list[tuple[str, int]]:
    """Extract imports from Python code using AST."""
    # Handles: import foo, from foo import bar, from foo.bar import baz
    ...

def extract_generic_imports(code: str, language: str) -> list[tuple[str, int]]:
    """Extract imports using regex for non-Python languages."""
    # JavaScript/TypeScript: import ... from '...', require('...')
    # Go: import "..."
    # Rust: use ...
    # Java: import ...
    ...

def extract_imports(code: str, language: str) -> list[tuple[str, int]]:
    """Dispatch to language-specific extractor."""
    if language == "python":
        return extract_python_imports(code)
    return extract_generic_imports(code, language)
```

**Depends On:** None

**Verify:** Read file, `cd mirdan && uv run mypy src/mirdan/core/import_extractor.py`

**Grounding:** Read of tidy_first.py lines 146-159 confirmed Python AST usage for code analysis.

---

#### Step 18: Implement ArchitectureAnalyzer

**File:** NEW `mirdan/src/mirdan/core/architecture_analyzer.py`

**Action:** Write — ~200 lines

**Details:**
```python
"""Architectural drift detection.

Loads architecture model from .mirdan/architecture.yaml, validates
code against layer boundaries and import rules. Produces Violation
objects compatible with the existing validation pipeline.
"""
class ArchitectureAnalyzer:
    def __init__(self, config: ArchitectureConfig):
        self._config = config
        self._layers: list[ArchLayer] = []
        self._loaded = False

    def load_model(self, project_dir: Path) -> bool:
        """Load architecture model from .mirdan/architecture.yaml."""
        ...

    def _resolve_layer(self, file_path: str) -> str | None:
        """Determine which layer a file belongs to using fnmatch."""
        ...

    def _resolve_import_layer(self, module_path: str) -> str | None:
        """Determine which layer an imported module belongs to."""
        ...

    def analyze_file(self, file_path: str, code: str, language: str) -> ArchDriftResult:
        """Check a file's imports against architecture model."""
        # 1. Resolve file's layer
        # 2. Extract imports from code
        # 3. For each import, resolve target layer
        # 4. Check against allowed/forbidden lists
        # 5. Generate Violation objects for violations
        ...

    def get_context_warnings(self, intent: Intent) -> list[str]:
        """Generate architectural context warnings for enhance_prompt."""
        # Based on FILE_PATH entities, identify touched layers
        # Return warnings about layer responsibilities
        ...
```

**Depends On:** Steps 1-3, Step 17

**Verify:** Read file, `cd mirdan && uv run mypy src/mirdan/core/architecture_analyzer.py`

**Grounding:** Read of config.py:ThresholdsConfig.resolve_for_file() at lines 215-239 confirmed fnmatch pattern. Read of tidy_first.py confirmed file-reading and analysis pattern.

---

#### Step 19: Wire ArchitectureAnalyzer into providers.py

**File:** `mirdan/src/mirdan/providers.py`

**Action:** Edit — import, create instance (loading model from project_dir), pass to both use case factories

**Details:**
```python
# After imports
from mirdan.core.architecture_analyzer import ArchitectureAnalyzer

# After self.guardrail_analyzer (from Step 13)
self.architecture_analyzer = ArchitectureAnalyzer(config.architecture)
self.architecture_analyzer.load_model(project_dir)

# Pass to create_validate_code_usecase and create_enhance_prompt_usecase
```

**Depends On:** Step 18

**Verify:** Read file, `cd mirdan && uv run ruff check src/mirdan/providers.py`

**Grounding:** Read of providers.py confirmed project_dir at line 79, instance pattern at lines 108-148.

---

#### Step 20: Integrate into ValidateCodeUseCase

**File:** `mirdan/src/mirdan/usecases/validate_code.py`

**Action:** Edit — accept ArchitectureAnalyzer, call when file_path provided, merge violations

**Details:** After linter section (after line 159), before violation enrichment:
```python
        # Architecture drift detection
        if (
            self._architecture_analyzer is not None
            and file_path
        ):
            arch_result = self._architecture_analyzer.analyze_file(
                file_path=file_path, code=code, language=result.language_detected
            )
            if arch_result.violations:
                result.violations.extend(arch_result.violations)
                # Recalculate score with new violations
                result.score = self._code_validator._calculate_score(result.violations)
                result.passed = not any(v.severity == "error" for v in result.violations)
            if arch_result.to_dict():
                output["architecture_drift"] = arch_result.to_dict()
```

Note: output dict doesn't exist at line 159 yet — the architecture check needs to happen after result is computed but before output dict. Insert after line 149 (after `_t_validate`) and before linter section, OR after output is created at line 202. The latter is cleaner — add architecture drift as an output enrichment.

**Revised location:** After line 244 (after analysis_protocol), before session quality. This keeps it in the "enrichment" section.

**Depends On:** Steps 18-19

**Verify:** Read file, `cd mirdan && uv run mypy src/mirdan/usecases/validate_code.py`

**Grounding:** Read of validate_code.py confirmed output enrichment pattern at lines 202-306.

---

#### Step 21: Integrate into EnhancePromptUseCase

**File:** `mirdan/src/mirdan/usecases/enhance_prompt.py`

**Action:** Edit — accept ArchitectureAnalyzer, at STANDARD+ add context warnings

**Details:** After decision/guardrail block (from Step 14):
```python
        # Architecture context — only at STANDARD+ when model loaded
        arch_context = None
        if (
            self._architecture_analyzer is not None
            and level >= CeremonyLevel.STANDARD
        ):
            arch_context = self._architecture_analyzer.get_context_warnings(intent)

        # ... later in result_dict:
        if arch_context:
            result_dict["architecture_context"] = arch_context
```

**Depends On:** Step 19

**Verify:** Read file, `cd mirdan && uv run mypy src/mirdan/usecases/enhance_prompt.py`

**Grounding:** Read of enhance_prompt.py confirmed ceremony gating pattern.

---

#### Step 22: Extend scan_conventions for architecture discovery

**File:** `mirdan/src/mirdan/usecases/scan_conventions.py` (verified exists)

**Action:** Edit — add architecture scanning mode

**Details:** When `--architecture` flag is used (or detected via new parameter), scan Python files for import statements, build import graph, infer layer boundaries from directory structure, and generate initial `.mirdan/architecture.yaml`.

This is an extension to the existing scan_conventions execute() method:
- New parameter: `scan_architecture: bool = False`
- When True, use import_extractor to build import graph
- Cluster files by directory prefix
- Identify which directories import from which
- Generate suggested architecture.yaml

**Depends On:** Step 17 (import_extractor)

**Verify:** `cd mirdan && uv run mypy src/mirdan/usecases/scan_conventions.py`

**Grounding:** Verified scan_conventions.py exists in usecases/ via Glob. Extension follows existing parameter addition pattern.

---

#### Step 23: Tests for ArchitectureAnalyzer

**File:** NEW `mirdan/tests/test_architecture_analyzer.py`

**Action:** Write — ~180 lines

**Details:**
- Test layer resolution: file matches correct layer pattern
- Test allowed import: service→models passes
- Test forbidden import: presentation→database fails with ARCH004
- Test no architecture model: returns empty result
- Test Python import extraction via AST
- Test generic import extraction via regex
- Test fnmatch pattern matching
- Test disabled config
- Test get_context_warnings for enhance_prompt
- Test file not in any layer: no violations

**Depends On:** Steps 17-18

**Verify:** `cd mirdan && uv run pytest tests/test_architecture_analyzer.py -v`

**Grounding:** Read of architecture_analyzer.py design (Step 18).

---

#### Step 24: Integration tests

**File:** NEW `mirdan/tests/test_engineering_intelligence_integration.py`

**Action:** Write — ~100 lines

**Details:**
- End-to-end: enhance_prompt with "add caching to user endpoint" → verify decision_guidance present
- End-to-end: enhance_prompt with "implement payment webhook" → verify cognitive_guardrails present
- End-to-end: validate_code_quality → verify confidence field present
- End-to-end: validate_code_quality with file_path and architecture.yaml → verify architecture_drift
- Test ceremony gating: MICRO ceremony → no decision_guidance or guardrails
- Test all features disabled via config → graceful no-op

**Depends On:** All previous steps

**Verify:** `cd mirdan && uv run pytest tests/test_engineering_intelligence_integration.py -v`

**Grounding:** All integration points verified via Read of use case files.

---

### Phase 5: Polish

#### Step 25: Update OutputFormatter

**File:** `mirdan/src/mirdan/core/output_formatter.py` (verified exists)

**Action:** Edit — add compression rules for new fields

**Details:**
Compression priority (strip in order as tokens decrease):
1. COMPACT: Remove `decision_guidance`, `architecture_context`
2. MINIMAL: Remove `cognitive_guardrails`, `architecture_drift`
3. MICRO: Remove everything except `confidence` (smallest field)

The `confidence` field survives to MINIMAL because it's ~30 tokens and high-value.

**Depends On:** Steps 7, 14, 20

**Verify:** Read file, verify compression order is correct

**Grounding:** Read confirmed OutputFormatter uses token-budget compression.

---

#### Step 26: Update PromptComposer TASK_GUIDANCE

**File:** `mirdan/src/mirdan/core/prompt_composer.py` (verified via Read, line 29-78)

**Action:** Edit — add references to decision guidance in GENERATION and REFACTOR guidance strings

**Details:** Append to TASK_GUIDANCE[GENERATION]:
```
"If decision_guidance is provided above, evaluate trade-offs before choosing an approach."
```

**Depends On:** Step 14

**Verify:** Read file

**Grounding:** Read of prompt_composer.py lines 29-78 confirmed TASK_GUIDANCE dict structure.

---

#### Step 27: Update CHANGELOG.md

**File:** `mirdan/CHANGELOG.md` (verified exists)

**Action:** Edit — add 1.10.0 release section

**Details:**
```markdown
## 1.10.0 — Engineering Intelligence Platform

### Added
- **Decision Intelligence Engine**: Structured trade-off analysis for 8 common
  architectural decision domains (caching, authentication, state management, data
  access, error handling, API design, testing strategy, configuration). Surfaces
  approaches, trade-offs, and senior-level questions in enhance_prompt output.
- **Cognitive Guardrails**: Domain-aware pre-flight thinking prompts (payment,
  auth, migration, concurrency, etc.) that surface 2-3 considerations a senior
  engineer would mention before coding starts.
- **Confidence Calibration**: Calibrated HIGH/MEDIUM/LOW confidence assessment
  in validate_code_quality output, with a single attention_focus item pointing
  to the most important manual verification.
- **Architectural Drift Detection**: Codify intended architecture in
  .mirdan/architecture.yaml. New ARCH004 (layer_violation) and ARCH005
  (unauthorized_dependency) rules detect when code drifts from intended boundaries.
- Architecture discovery via `mirdan scan --architecture`.
- New config sections: `decision`, `guardrails`, `architecture`.

### Changed
- Version bump 1.9.0 → 1.10.0 (paradigm shift: Quality Gate → Engineering Intelligence)
```

**Depends On:** All steps

**Verify:** Read file

---

#### Step 28: Version bump

**File:** `mirdan/src/mirdan/__init__.py` (verified via Read, line 8)

**Action:** Edit — change `__version__ = "1.9.0"` to `__version__ = "1.10.0"`

**Depends On:** All steps

**Verify:** `cd mirdan && uv run python -c "import mirdan; print(mirdan.__version__)"`

**Grounding:** Read of __init__.py confirmed version at line 8.

---

#### Step 29: Update Claude Code rules

**Files:** `.claude/rules/mirdan-workflow.md`, `.claude/rules/mirdan-quality.md` (verified exist)

**Action:** Edit — document new output fields

**Details:** Add sections explaining:
- `decision_guidance`: Review approaches before implementing. Consider senior_questions.
- `cognitive_guardrails`: Review pre-flight items before coding.
- `confidence`: Check level and attention_focus after validation.
- `architecture_drift`: Fix any ARCH004/ARCH005 violations before completing.

**Depends On:** All steps

**Verify:** Read files

---

## Dependency Graph

```
Phase 1 (Foundation)
  Step 1 ──→ Step 2 ──→ Step 4 (tests)
  Step 3 ──────────────→ Step 4 (tests)

Phase 2 (Confidence) — depends on Phase 1
  Step 5 ──→ Step 6 ──→ Step 7
  Step 5 ──→ Step 8 (tests)

Phase 3 (Decision + Guardrails) — depends on Phase 1
  Step 9  ──→ Step 10 ──→ Step 13 ──→ Step 14
  Step 11 ──→ Step 12 ──→ Step 13
  Step 10 ──→ Step 15 (tests)
  Step 12 ──→ Step 16 (tests)

Phase 4 (Architecture) — depends on Phase 1
  Step 17 ──→ Step 18 ──→ Step 19 ──→ Step 20
                                   ──→ Step 21
  Step 17 ──→ Step 22
  Steps 17-18 ──→ Step 23 (tests)
  All ──→ Step 24 (integration tests)

Phase 5 (Polish) — depends on all
  Steps 25-29 (all parallel except 28 last)
```

**Phases 2, 3, 4 can execute in parallel** — they only share Phase 1 as a dependency.

---

## Key Design Principles

1. **Zero new MCP tools** — all features enrich existing enhance_prompt / validate_code_quality output
2. **Zero new dependencies** — uses only stdlib (ast, fnmatch) and existing deps (pyyaml, pydantic)
3. **Fully backward compatible** — new output fields are additive, existing consumers unaffected
4. **Ceremony-gated** — Decision guidance and guardrails only at STANDARD+, avoiding noise
5. **Config-gatable** — every feature has `enabled: bool` flag
6. **Not prescriptive** — surfaces trade-offs and questions, not recommendations
7. **Follows existing patterns** — every new component follows TidyFirst/AgentCoordinator pattern exactly

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Decision YAML templates low quality | Medium | High | Thorough content review, draw from established resources |
| Token budget pressure from new fields | Low | Medium | OutputFormatter compression already handles this |
| Architecture model too rigid | Low | Medium | Optional feature, auto-discovery helps bootstrap |
| False positive layer violations | Medium | Medium | High-precision fnmatch, user-defined model |
| Feature noise at STANDARD ceremony | Low | Medium | Max caps (1 decision, 3 guardrails) |

## North Star Metrics

After implementation, track:
- **Decision quality proxy**: Do users modify the AI's approach after seeing decision_guidance?
- **Attention effectiveness**: Do confidence.attention_focus items prevent downstream bugs?
- **Architecture compliance**: Do ARCH004/ARCH005 violations decrease over time?
- **Net value**: Is total time (coding + Mirdan overhead) lower than coding alone?

---

## Review Corrections (2026-03-11, revised 2026-03-11)

### Critical Fixes Required During Implementation

**C1 — Step 3: ArchDriftResult forward-references Violation (NameError)**
`models.py` does NOT have `from __future__ import annotations`. Placing `ArchDriftResult`
(with `violations: list[Violation]`) at line 547 while `Violation` is at line 634 causes
a `NameError` at import time. **Fix:** Add `from __future__ import annotations` to
`models.py` line 2 (after docstring), or place `ArchDriftResult` after `Violation` (after
line 678), or use string annotation `list["Violation"]`.

**C2 — Step 20: Architecture violations invisible in main output (split required)**
The revised insertion point (after line 244) is AFTER `output = result.to_dict()` at line 202.
Modifying `result.violations` and `result.score` at that point does NOT update the already-
serialized `output` dict. Architecture violations appear in `output["architecture_drift"]` but
are absent from `output["violations"]`, `output["violations_count"]`, `output["score"]`, and
`output["summary"]`. **Fix:** Split into two insertion points:
- **Part A** (after line 159, before line 161): Merge architecture violations into
  `result.violations`, recalculate `result.score` and `result.passed`. This ensures
  ARCH004/ARCH005 violations flow through violation enrichment and into `output`.
- **Part B** (after line 244, in enrichment section): Add
  `output["architecture_drift"] = arch_result.to_dict()` for the structured drift report.
The plan's original location (after line 159) was correct for Part A; the revision to
after line 244 is only correct for Part B.

**C3 — Step 7: Variable scoping bug + dead code in confidence integration**
Two bugs in the plan's code:
1. `_semantic_list` is created from `output.get("semantic_checks", [])` but never used (dead code).
2. `semantic_checks` is passed to `.assess()` but is only defined inside
   `if self._config.semantic.enabled:` — undefined when semantic analysis is disabled → `NameError`.
**Fix:** Delete the `_semantic_list` block entirely. Initialize `semantic_checks = []` before
line 226 (`if self._config.semantic.enabled:`). Pass `semantic_checks` to `.assess()` directly.
This is the simplest fix with no dead code.

### Moderate Fixes Required During Implementation

**M1 — Step 22: Missing server.py update for scan_architecture parameter**
Adding `scan_architecture: bool` to `ScanConventionsUseCase.execute()` requires a matching
parameter in `server.py` line 307 (`scan_conventions` tool). Add a Step 22b: edit
`server.py` to accept and pass the `scan_architecture` parameter.

**M2 — Step 25: OutputFormatter underspecified**
`_compact_enhanced()` at line 189 uses explicit dict construction. Must specify exact edits:
- `_compact_enhanced`: add `cognitive_guardrails` from data if key present (survives COMPACT)
- `_compact_validation`: add `confidence` and `architecture_drift` from data if present (survive COMPACT)
- `_minimal_validation`: add `confidence` from data if present (survives MINIMAL at ~30 tokens)

**M3 — Steps 10, 12: Use importlib.resources for YAML loading**
Both `DecisionAnalyzer._load_templates()` and `GuardrailAnalyzer._load_domains()` must use
`importlib.resources.files("mirdan.standards")` (same as `QualityStandards._load_default_standards()`
at line 124) for package-safe path resolution. Raw `Path` resolution breaks when mirdan is
installed as a package. Pattern to follow:
```python
from importlib.resources import files
standards_pkg = files("mirdan.standards")
decisions_dir = standards_pkg.joinpath("decisions")
```

**M4 — Step 5: Remove unused `session` parameter from ConfidenceCalibrator.assess()**
The `session: SessionContext | None = None` parameter is accepted but never referenced in the
method body. The feature spec's "code modifies logic" condition was simplified to just `not
test_file`. Either remove the parameter (YAGNI) or document why it's reserved. Mirdan's own
AI004 (dead code) rule would flag this.

### Minor Corrections (Non-blocking)

**N1 — Summary table counts off by 1:**
Phase 3 lists 12 new files but actual count is 13 (8 decision YAMLs + guardrails.yaml +
2 analyzers + 2 test files). Phase 4 lists 3 modified files but actual count is 4
(providers.py + validate_code.py + enhance_prompt.py + scan_conventions.py).

**N2 — Step 20 self-contradicts:**
The code block header says "After linter section (after line 159)" but the revised location
paragraph says "After line 244 (after analysis_protocol)". Per C2 fix, the answer is both:
Part A at line 159, Part B at line 244.

**N3 — Several line number references are off by 1-2 lines:**
Step 1 says "after line 396" but CoordinationConfig ends at line 397. Step 6 says "after
line 16 (after ceremony import)" but ceremony is at line 14. Step 13 says "after line 34,
after tidy_first import" but tidy_first is at line 33. All are self-correcting via the
textual descriptions.

**N4 — Missing regression test step:**
Phase 5 should include running `uv run pytest` on the full existing test suite to verify
no regressions. Individual test steps only cover new test files.

### No Technical Debt Introduced

All findings above are corrections to the plan, not new debt. The implementation follows
established patterns (TidyFirst/Ceremony/AgentCoordinator), uses existing dependencies
only, and every feature is config-gatable and ceremony-gated. No new abstractions, no new
MCP tools, no new dependencies.
