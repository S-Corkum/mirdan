# Implementation Plan: Semantic Validation + Dependency Vulnerability Scanning

## Review Corrections (Applied)

### Review 1 Corrections
| Step | Severity | Issue | Fix Applied |
|------|----------|-------|-------------|
| 9, 10 | **CRITICAL** | Plan passed manifest_parser/vuln_scanner directly to AIQualityChecker in server.py, but AIQualityChecker is lazily created inside CodeValidator (code_validator.py:737-739), not in server.py | Step 10 now passes through CodeValidator; Step 9 now forwards to AIQualityChecker at its lazy init site |
| 2 | MODERATE | apply_profile() maps only 4/6 existing dimensions; new dimensions would be defined but have zero effect | Added explicit note that mapping logic is required |
| 4 | MODERATE | ManifestParser duplicates logic already in AIQualityChecker._load_project_deps() (L611-650) which parses all 5 manifest formats | Clarified this is a refactor-and-extend of existing logic; _load_project_deps should delegate to ManifestParser |
| 7 | MINOR | Wording implied extracting an existing shared method; import extraction is actually inline per-language regex | Clarified this is a refactoring step creating a new shared method from inline code |

### Review 2 Corrections
| Step | Severity | Issue | Fix Applied |
|------|----------|-------|-------------|
| 5 | **HIGH** | `_extract_severity` has syntax errors and always returns "medium" — CVSS vector parsing is broken | Added working severity extraction with proper CVSS score → severity mapping and OSV `database_specific` fallback |
| 7 | **HIGH** | `_load_project_deps()` (L611-650) still independently parses manifests after ManifestParser exists — duplicate logic | Added sub-step to refactor `_load_project_deps` to delegate to ManifestParser |
| 8 | **HIGH** | TEMPLATE_FIXES uses values as literal strings (L312: `fix_code=fix_code`), not template-substituted. `{package}` and `{fixed_version}` would be returned as-is | Changed to use `violation.suggestion` as fix text since it already contains the upgrade instruction |
| 9 | **HIGH** | Step 9 depends on Step 7 (forwards manifest_parser/vuln_scanner that Step 7 adds to AIQualityChecker.__init__), but only listed Step 6 as dependency | Added Step 7 as dependency. Also noted Steps 9 and 10 must be sequential (both modify CodeValidator.__init__) |
| 19-21 | **HIGH** | Plan creates 3 new test files but doesn't extend 6 existing test files (`test_ai_quality_checker.py`, `test_auto_fix.py`, `test_output_formatter.py`, `test_knowledge_producer.py`, `test_code_validator.py`, `test_server.py`). 85% coverage requirement at risk | Added Step 19b to extend existing test files |
| NEW | **HIGH** | `violation_explainer.py` has SEC001-SEC013 entries but plan doesn't add SEC014 | Added to Step 7 as sub-step |
| NEW | **HIGH** | 4 integration files reference "SEC001-SEC013" and need SEC014: `agents_md.py:172-176`, `security-audit.md:33`, `mirdan-security.md:28-31`, `cursor.py` rules table | Added Step 15b for integration file updates |
| 18 | **HIGH** | `gate_command.py` uses CodeValidator directly (not server.py `_Components`). Plan references `manifest_parser.parse()` and `vuln_scanner.scan()` as if they exist in scope — they don't | Added explicit ManifestParser/VulnScanner instantiation in gate_command scope |
| 13 | MODERATE | `_compact_validation` (L239-246) uses key WHITELISTING — constructs new dict with only specific keys. Semantic_checks would be silently dropped unless explicitly added to whitelist | Clarified Step 13 must add key to whitelist dict construction |
| 10 | MODERATE | Adding 7th tool to `_TOOL_PRIORITY` affects Cursor users with `MIRDAN_TOOL_BUDGET=6` — `scan_conventions` would be dropped | Added note about tool budget priority ordering |
| 15 | MINOR | Plan says "Add SEC011-SEC013" to BUGBOT blocking bugs — they're already in the rules TABLE (L708-711), just not in the blocking bugs REGEX section (ends at SEC010 L803) | Clarified: adding to regex blocking bugs section specifically |
| dep graph | MODERATE | Steps 9 and 10 both modify CodeValidator.__init__ — parallel execution would conflict | Marked as sequential in dependency graph |

### Review 3 Corrections
| Step | Severity | Issue | Fix Applied |
|------|----------|-------|-------------|
| 7 | **HIGH** | `_load_project_deps` 3-line delegation drops `_COMMON_TRANSITIVE_PACKAGES` (L79-83: requests, typing_extensions, six, etc.) and `_find_local_packages()` (L715-726: src layout detection). Causes AI002 false positives for transitive deps and self-imports | Fixed: delegation preserves both by appending them after ManifestParser results |
| 7 | MODERATE | SEC014 should be added to `check_quick()` (L202-222) — it's a zero-cost cache lookup (no network), critical for hook-triggered validation where only AI001, AI007, AI008 currently run | Added `_check_sec014_vulnerable_deps` call to `check_quick()` |
| 11 | MODERATE | `cache_ttl` parameter accepted in `scan_dependencies` tool but never used — VulnScanner already initialized in `_get_components()` with TTL from `config.dependencies.osv_cache_ttl` | Removed `cache_ttl` parameter from tool signature |
| 8 | MINOR | Verify instruction says "confirm SEC014 in TEMPLATE_FIXES and _QUICK_FIX_RULES" but step explicitly says NOT to add to `_QUICK_FIX_RULES` | Fixed verify instruction to say "NOT in _QUICK_FIX_RULES" |
| 5 | LOW | CVSS vector string `"CVSS:3.1/AV:N/..."` in `score` field will fail `float()` — acceptable since it falls through to ecosystem/database fallbacks | Added clarifying comment in `_extract_severity` code |
| 9 | MINOR | Step 9 uses `quality_standards` as first param name but actual CodeValidator uses `standards: QualityStandards` | Fixed parameter name to match actual signature |

---

## Research Notes (Pre-Plan Verification)

### Files Verified

| File | Key Structures | Lines |
|------|---------------|-------|
| `src/mirdan/server.py` | `_Components` dataclass (L56-75), `_TOOL_PRIORITY` (L82-89), `validate_code_quality` tool (L339-475), `scan_conventions` tool (L754-779) | 800+ |
| `src/mirdan/models.py` | `Violation` (L463-503), `ValidationResult` (L507-561) | 561 |
| `src/mirdan/config.py` | `QualityConfig` (L10-21), `MirdanConfig` (L225-248), `load()` (L250-259) | 290+ |
| `src/mirdan/core/quality_profiles.py` | `QualityProfile` (L14-32), `BUILTIN_PROFILES` (L44-115), `apply_profile()` (L150-169) | 233 |
| `src/mirdan/core/code_validator.py` | `validate()` (L908-1025), `validate_quick()` (L1027-1068), `_run_checks()` loop (L941-950) | 1068+ |
| `src/mirdan/core/ai_quality_checker.py` | `check()` (L175-200), `_check_ai002_python()` (L404-446), `_load_project_deps()` (L169) | 446+ |
| `src/mirdan/core/auto_fixer.py` | `TEMPLATE_FIXES` (L39-202), `PATTERN_FIXES` (L207-274), `FixResult` (L18-30), `quick_fix()` (L425-444) | 471 |
| `src/mirdan/core/output_formatter.py` | `OutputFormat` enum, `format_validation_result()` (L135-162), `_micro_validation()` (L328-355) | 355+ |
| `src/mirdan/core/knowledge_producer.py` | `extract_from_validation()` (L20-48), `_extract_security_knowledge()` (L130-163) | 194 |
| `src/mirdan/integrations/sarif.py` | `SARIFExporter.export()` (L29-62), `_build_results()` (L102-144) | 144+ |
| `src/mirdan/integrations/hook_templates.py` | `STRINGENCY_EVENTS` (L31-56), `ALL_HOOK_EVENTS` (L83-101), generator methods | 469+ |
| `src/mirdan/integrations/cursor.py` | `_generate_bugbot_md()` (L724-825), `CursorAdapter` (L894-945) | 945+ |
| `src/mirdan/cli/__init__.py` | `main()` routing (L28-51), `_print_help()` (L146-171) | 171 |
| `pyproject.toml` | Core deps (L21-26), optional "ast" group (L29-33), "dev" group (L34-40), entry point (L42-43) | 99 |

### Project Structure

```
src/mirdan/
├── server.py              # MCP server, 6 tools
├── models.py              # Violation, ValidationResult, KnowledgeEntry, etc.
├── config.py              # MirdanConfig with 12 sub-configs
├── core/
│   ├── code_validator.py  # Main validation engine
│   ├── ai_quality_checker.py  # AI001-AI008 rules
│   ├── auto_fixer.py      # Template + pattern fixes
│   ├── output_formatter.py
│   ├── quality_profiles.py
│   ├── quality_standards.py
│   ├── knowledge_producer.py
│   ├── session_manager.py
│   ├── violation_explainer.py
│   └── ... (20+ more modules)
├── integrations/
│   ├── hook_templates.py  # Claude Code hooks
│   ├── cursor.py          # Cursor IDE integration
│   ├── sarif.py           # SARIF 2.1.0 export
│   └── ...
├── cli/
│   ├── __init__.py        # Command routing
│   ├── scan_command.py
│   ├── gate_command.py
│   └── ... (10 command modules)
└── standards/             # YAML rule definitions
```

### Dependencies Confirmed
- `fastmcp>=2.0.0` (core dep, pyproject.toml L22)
- `pyyaml>=6.0` (core dep, L23)
- `pydantic>=2.0` (core dep, L24)
- `jinja2>=3.1.0` (core dep, L25)
- Python 3.11+ required (L6) — `tomllib` available in stdlib
- No `httpx` in current deps — using `urllib.request` (stdlib) for OSV API

### API Documentation (context7)
- OSV API: `POST https://api.osv.dev/v1/querybatch` — free, no API key, batch up to 1000 packages
- `tomllib` (Python 3.11+): `tomllib.load(f)` for TOML parsing

### Conventions (from codebase)
- All dataclasses use `@dataclass` from stdlib or `BaseModel` from pydantic (config only)
- Violations use category strings: "security", "architecture", "style", "ai_quality"
- Test pattern: class-based with `setup_method`, pytest fixtures, type hints mandatory
- No conftest.py — fixtures per test file
- 85% coverage minimum enforced

### Similar Implementations
- `ai_quality_checker.py` AI002: import checking pattern (parse imports → check against known packages)
- `convention_extractor.py`: scan files → aggregate results pattern
- `linter_runner.py` + `linter_orchestrator.py`: external tool orchestration pattern

---

## Plan Overview

### Architecture

```
SEMANTIC VALIDATION                    DEPENDENCY SCANNING

Layer 1: Pattern-Informed Prompts      Tier 1: Import-Manifest Reconciliation
(semantic_analyzer.py)                 (manifest_parser.py)
  ↓                                      ↓
Layer 3: Analysis Protocol             Tier 2: OSV Vulnerability Lookup
(semantic_analyzer.py)                 (vuln_scanner.py)
  ↓                                      ↓
Both integrate into:
  server.py validate_code_quality      server.py scan_dependencies (NEW tool)
  (semantic_checks in response)        (dedicated MCP tool)
                                       ai_quality_checker.py SEC014 (cached)
```

### Key Design Decisions

1. **No new dependencies** — `urllib.request` for OSV API, `tomllib` (stdlib 3.11+) for TOML
2. **Semantic checks are questions, not violations** — they don't affect quality score; they guide the LLM
3. **VulnScanner has sync + async** — `check_cached()` for validate(), `scan()` for scan_dependencies
4. **SEC014 only fires from cache** — no network call during validate_code_quality
5. **Layer 2 (tree-sitter taint tracking) deferred** — regex-based taint detection in Layer 1 instead
6. **Backwards compatible** — all new config fields have defaults, new response fields are additive

---

## Phase 1: Foundation

### Step 1: Add SemanticConfig + DependencyConfig to config.py

**File:** `src/mirdan/config.py` (verified via Read)

**Action:** Edit

**Details:**
- Add `SemanticConfig` class after `QualityConfig` (after line 21):
  ```python
  class SemanticConfig(BaseModel):
      """Semantic validation configuration."""
      enabled: bool = Field(default=True, description="Enable semantic analysis")
      analysis_protocol: str = Field(
          default="security", pattern="^(none|security|comprehensive)$",
          description="When to generate structured analysis protocols"
      )
  ```
- Add `DependencyConfig` class after `SemanticConfig`:
  ```python
  class DependencyConfig(BaseModel):
      """Dependency vulnerability scanning configuration."""
      enabled: bool = Field(default=True, description="Enable dependency scanning")
      osv_cache_ttl: int = Field(default=86400, description="OSV cache TTL in seconds")
      scan_on_gate: bool = Field(default=True, description="Include dep scan in mirdan gate")
      fail_on_severity: str = Field(
          default="high", pattern="^(critical|high|medium|low|none)$",
          description="Minimum severity to fail quality gate"
      )
  ```
- Add both to `MirdanConfig` (after line 248):
  ```python
  semantic: SemanticConfig = Field(default_factory=SemanticConfig)
  dependencies: DependencyConfig = Field(default_factory=DependencyConfig)
  ```

**Depends On:** None

**Verify:** Read config.py, confirm both classes exist and MirdanConfig includes them. Run `uv run python -c "from mirdan.config import MirdanConfig; c = MirdanConfig(); print(c.semantic.enabled, c.dependencies.enabled)"`

**Grounding:** Read of config.py confirmed QualityConfig at L10, MirdanConfig at L225-248, field pattern from existing classes.

---

### Step 2: Add semantic + dependency dimensions to quality_profiles.py

**File:** `src/mirdan/core/quality_profiles.py` (verified via Read)

**Action:** Edit

**Details:**
- Add two fields to `QualityProfile` dataclass (after line 31, before `metadata`):
  ```python
  semantic: float = 0.5
  dependency_security: float = 0.7
  ```
- **NOTE:** `apply_profile()` currently only maps 4 of 6 dimensions (security, architecture,
  testing, documentation). The existing `ai_slop_detection` and `performance` dimensions are
  UNMAPPED. The new dimensions below must include explicit mapping logic or they'll have no effect.
- Update `apply_profile()` function (around line 150-169) to map new dimensions:
  ```python
  if "semantic" not in quality:
      quality["semantic"] = {}
  quality["semantic"]["enabled"] = profile.semantic >= 0.3
  quality["semantic"]["analysis_protocol"] = (
      "comprehensive" if profile.semantic >= 0.8
      else "security" if profile.semantic >= 0.5
      else "none"
  )
  if "dependencies" not in quality:
      quality["dependencies"] = {}
  quality["dependencies"]["enabled"] = profile.dependency_security >= 0.3
  quality["dependencies"]["scan_on_gate"] = profile.dependency_security >= 0.5
  quality["dependencies"]["fail_on_severity"] = (
      "medium" if profile.dependency_security >= 0.8
      else "high" if profile.dependency_security >= 0.5
      else "critical"
  )
  ```
- Update `BUILTIN_PROFILES` (L44-115) with new dimension values:
  | Profile | semantic | dependency_security |
  |---------|----------|-------------------|
  | default | 0.5 | 0.7 |
  | startup | 0.3 | 0.5 |
  | enterprise | 0.9 | 1.0 |
  | fintech | 1.0 | 1.0 |
  | library | 0.7 | 0.8 |
  | data-science | 0.3 | 0.5 |
  | prototype | 0.0 | 0.3 |

**Depends On:** Step 1 (config classes must exist)

**Verify:** Run `uv run python -c "from mirdan.core.quality_profiles import get_profile; p = get_profile('enterprise'); print(p.semantic, p.dependency_security)"`

**Grounding:** Read of quality_profiles.py confirmed QualityProfile at L14-32, BUILTIN_PROFILES at L44-115, apply_profile at L150-169.

---

### Step 3: Add new dataclasses to models.py

**File:** `src/mirdan/models.py` (verified via Read at L460-561)

**Action:** Edit

**Details:**
- Add after `Violation` class (after line 503), before `ValidationResult`:

  ```python
  @dataclass
  class SemanticCheck:
      """A semantic review question for the calling LLM to investigate."""
      concern: str          # taint_propagation, error_completeness, resource_management, auth_flow, concurrency
      question: str         # Specific, actionable question with line numbers
      severity: str         # critical, warning, info
      related_violation: str = ""  # Rule ID that triggered this (e.g., "SEC004")
      focus_lines: list[int] = field(default_factory=list)

      def to_dict(self) -> dict[str, Any]:
          result: dict[str, Any] = {
              "concern": self.concern,
              "question": self.question,
              "severity": self.severity,
          }
          if self.related_violation:
              result["related_violation"] = self.related_violation
          if self.focus_lines:
              result["focus_lines"] = self.focus_lines
          return result


  @dataclass
  class AnalysisProtocol:
      """Structured protocol for the LLM to self-execute deep analysis."""
      type: str             # security_flow_analysis, auth_completeness, data_handling
      focus_areas: list[dict[str, Any]] = field(default_factory=list)
      response_format: dict[str, Any] = field(default_factory=dict)

      def to_dict(self) -> dict[str, Any]:
          return {
              "type": self.type,
              "focus_areas": self.focus_areas,
              "response_format": self.response_format,
          }


  @dataclass
  class PackageInfo:
      """A dependency package parsed from a manifest."""
      name: str
      version: str
      ecosystem: str        # PyPI, npm, crates.io, Go, Maven
      source: str           # File it was parsed from
      is_dev: bool = False

      def to_dict(self) -> dict[str, Any]:
          return {
              "name": self.name,
              "version": self.version,
              "ecosystem": self.ecosystem,
              "source": self.source,
              "is_dev": self.is_dev,
          }


  @dataclass
  class VulnFinding:
      """A vulnerability found in a dependency."""
      package: str
      version: str
      ecosystem: str
      vuln_id: str          # CVE or OSV ID
      severity: str         # critical, high, medium, low
      summary: str
      fixed_version: str = ""
      advisory_url: str = ""

      def to_dict(self) -> dict[str, Any]:
          result: dict[str, Any] = {
              "package": self.package,
              "version": self.version,
              "ecosystem": self.ecosystem,
              "vuln_id": self.vuln_id,
              "severity": self.severity,
              "summary": self.summary,
          }
          if self.fixed_version:
              result["fixed_version"] = self.fixed_version
          if self.advisory_url:
              result["advisory_url"] = self.advisory_url
          return result
  ```

**Depends On:** None

**Verify:** Run `uv run python -c "from mirdan.models import SemanticCheck, PackageInfo, VulnFinding; print('OK')"`

**Grounding:** Read of models.py confirmed Violation at L463, ValidationResult at L507, dataclass pattern with to_dict().

---

## Phase 2: Core Engines

### Step 4: Create manifest_parser.py

**File:** `NEW: src/mirdan/core/manifest_parser.py` (parent dir verified via Glob)

**Action:** Write

**Details:**
Create `ManifestParser` class (~200 lines):

```python
"""Parse dependency manifests to extract package information."""

import json
import re
import tomllib
from pathlib import Path

from mirdan.models import PackageInfo


# Ecosystem mapping
_ECOSYSTEM_MAP = {
    "pyproject.toml": "PyPI",
    "requirements.txt": "PyPI",
    "setup.cfg": "PyPI",
    "package.json": "npm",
    "Cargo.toml": "crates.io",
    "go.mod": "Go",
    "pom.xml": "Maven",
    "build.gradle": "Maven",
}


class ManifestParser:
    """Parse dependency manifests for all supported ecosystems."""

    def __init__(self, project_dir: Path | None = None) -> None:
        self._project_dir = project_dir or Path.cwd()
        self._cache: list[PackageInfo] | None = None
        self._cache_mtimes: dict[str, float] = {}

    def parse(self, project_dir: Path | None = None) -> list[PackageInfo]:
        """Discover and parse all manifests in project directory."""
        root = project_dir or self._project_dir
        if not root.is_dir():
            return []

        # Check cache validity
        if self._cache is not None and self._cache_valid(root):
            return self._cache

        packages: list[PackageInfo] = []
        for manifest_name, ecosystem in _ECOSYSTEM_MAP.items():
            manifest_path = root / manifest_name
            if manifest_path.exists():
                try:
                    parsed = self._parse_file(manifest_path, ecosystem)
                    packages.extend(parsed)
                    self._cache_mtimes[str(manifest_path)] = manifest_path.stat().st_mtime
                except Exception:
                    continue  # Skip malformed manifests

        # Also check lock files for exact versions
        packages = self._enrich_from_lock_files(root, packages)

        self._cache = packages
        return packages

    def get_version(self, package: str, ecosystem: str) -> str | None:
        """Get version of a specific package."""
        packages = self.parse()
        for p in packages:
            if p.name == package and p.ecosystem == ecosystem:
                return p.version
        return None

    # ... (parsing methods for each manifest type)
```

Key methods to implement:
- `_parse_file(path, ecosystem)` → dispatcher to type-specific parsers
- `_parse_pyproject_toml(path)` → reads `[project.dependencies]` and `[project.optional-dependencies]`
- `_parse_requirements_txt(path)` → parses `package==version` lines
- `_parse_package_json(path)` → reads `dependencies` and `devDependencies`
- `_parse_cargo_toml(path)` → reads `[dependencies]` section
- `_parse_go_mod(path)` → parses `require` blocks
- `_enrich_from_lock_files(root, packages)` → overrides versions from lock files
- `_cache_valid(root)` → checks if any manifest file was modified since last parse

Version parsing rules:
- `pyproject.toml`: Use PEP 508 specifiers (e.g., `fastmcp>=2.0.0` → extract `2.0.0` as minimum)
- `requirements.txt`: `package==1.2.3` → `1.2.3`; `package>=1.0` → `1.0`
- `package.json`: `"^1.2.3"` → `1.2.3`; `"~1.2"` → `1.2.0`
- Lock files override manifest versions with exact pinned versions

**Depends On:** Step 3 (PackageInfo dataclass)

**Verify:** Create a test pyproject.toml in tmp_path, instantiate ManifestParser, call parse(), assert PackageInfo list is correct.

**Grounding:** Read of ai_quality_checker.py confirmed existing `_load_project_deps()` at L611-650 which ALREADY parses pyproject.toml, requirements.txt, package.json, go.mod, and Cargo.toml (returns `frozenset[str]`). ManifestParser is a REFACTOR AND EXTENSION of this existing logic — it adds version tracking, lock file enrichment, and ecosystem classification that `_load_project_deps` lacks. After ManifestParser is created, `_load_project_deps` should delegate to it to avoid duplicate parsing. `tomllib` is stdlib in Python 3.11+ (confirmed via pyproject.toml L6 requiring >=3.11).

---

### Step 5: Create vuln_scanner.py

**File:** `NEW: src/mirdan/core/vuln_scanner.py` (parent dir verified via Glob)

**Action:** Write

**Details:**
Create `VulnScanner` class (~250 lines):

```python
"""Scan dependencies for known vulnerabilities via OSV API."""

import asyncio
import json
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from mirdan.models import PackageInfo, VulnFinding

logger = logging.getLogger(__name__)

OSV_API_URL = "https://api.osv.dev/v1/querybatch"

# Map OSV severity to mirdan severity
_SEVERITY_MAP = {
    "CRITICAL": "critical",
    "HIGH": "high",
    "MODERATE": "medium",
    "MEDIUM": "medium",
    "LOW": "low",
}


class VulnCache:
    """TTL-based cache for vulnerability lookups."""

    def __init__(self, cache_path: Path, ttl: int = 86400) -> None:
        self._path = cache_path
        self._ttl = ttl
        self._data: dict[str, dict] = {}
        self._load()

    def _cache_key(self, pkg: PackageInfo) -> str:
        return f"{pkg.ecosystem}:{pkg.name}:{pkg.version}"

    def get(self, pkg: PackageInfo) -> list[VulnFinding] | None:
        """Get cached results, or None if expired/missing."""
        key = self._cache_key(pkg)
        entry = self._data.get(key)
        if not entry:
            return None
        checked_at = datetime.fromisoformat(entry["checked_at"])
        age = (datetime.now(timezone.utc) - checked_at).total_seconds()
        if age > self._ttl:
            return None
        return [VulnFinding(**f) for f in entry["findings"]]

    def store(self, pkg: PackageInfo, findings: list[VulnFinding]) -> None:
        """Store results in cache."""
        key = self._cache_key(pkg)
        self._data[key] = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "findings": [f.to_dict() for f in findings],
        }
        # Limit cache size
        if len(self._data) > 1000:
            # Remove oldest entries
            sorted_keys = sorted(self._data, key=lambda k: self._data[k]["checked_at"])
            for old_key in sorted_keys[:100]:
                del self._data[old_key]
        self._save()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))


class VulnScanner:
    """Scan packages for vulnerabilities using OSV API."""

    def __init__(self, cache_dir: Path, ttl: int = 86400) -> None:
        self._cache = VulnCache(cache_dir / "vuln-cache.json", ttl=ttl)

    def check_cached(self, packages: list[PackageInfo]) -> list[VulnFinding]:
        """Return cached vulnerability findings only (no network calls)."""
        findings: list[VulnFinding] = []
        for pkg in packages:
            cached = self._cache.get(pkg)
            if cached:
                findings.extend(cached)
        return findings

    async def scan(self, packages: list[PackageInfo]) -> list[VulnFinding]:
        """Scan packages against OSV API, updating cache."""
        # Separate cached vs uncached
        uncached: list[PackageInfo] = []
        all_findings: list[VulnFinding] = []

        for pkg in packages:
            cached = self._cache.get(pkg)
            if cached is not None:
                all_findings.extend(cached)
            else:
                uncached.append(pkg)

        if uncached:
            # Batch query OSV API
            new_findings = await asyncio.to_thread(self._query_osv, uncached)
            all_findings.extend(new_findings)

        return all_findings

    def _query_osv(self, packages: list[PackageInfo]) -> list[VulnFinding]:
        """Synchronous OSV API batch query."""
        queries = []
        for pkg in packages:
            q: dict = {"package": {"name": pkg.name, "ecosystem": pkg.ecosystem}}
            if pkg.version:
                q["version"] = pkg.version
            queries.append(q)

        # Batch in groups of 1000 (OSV limit)
        all_findings: list[VulnFinding] = []
        for i in range(0, len(queries), 1000):
            batch = queries[i:i + 1000]
            batch_pkgs = packages[i:i + 1000]
            try:
                findings = self._execute_batch(batch, batch_pkgs)
                all_findings.extend(findings)
            except (urllib.error.URLError, OSError) as e:
                logger.warning("OSV API query failed: %s", e)
                # Cache empty results so we don't retry immediately
                for pkg in batch_pkgs:
                    self._cache.store(pkg, [])

        return all_findings

    def _execute_batch(
        self, queries: list[dict], packages: list[PackageInfo]
    ) -> list[VulnFinding]:
        """Execute a single batch request to OSV API."""
        body = json.dumps({"queries": queries}).encode()
        req = urllib.request.Request(
            OSV_API_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        findings: list[VulnFinding] = []
        for pkg, result in zip(packages, data.get("results", [])):
            pkg_findings = self._parse_vulns(pkg, result.get("vulns", []))
            self._cache.store(pkg, pkg_findings)
            findings.extend(pkg_findings)

        return findings

    def _parse_vulns(self, pkg: PackageInfo, vulns: list[dict]) -> list[VulnFinding]:
        """Parse OSV vulnerability entries into VulnFindings."""
        findings: list[VulnFinding] = []
        for vuln in vulns:
            severity = self._extract_severity(vuln)
            fixed = self._extract_fixed_version(vuln, pkg.ecosystem)
            findings.append(VulnFinding(
                package=pkg.name,
                version=pkg.version,
                ecosystem=pkg.ecosystem,
                vuln_id=vuln.get("id", "UNKNOWN"),
                severity=severity,
                summary=vuln.get("summary", vuln.get("details", "")[:200]),
                fixed_version=fixed,
                advisory_url=next(
                    (r["url"] for r in vuln.get("references", []) if r.get("type") == "WEB"),
                    "",
                ),
            ))
        return findings

    def _extract_severity(self, vuln: dict) -> str:
        """Extract severity from OSV vulnerability data."""
        # Try CVSS score from severity array
        for sev in vuln.get("severity", []):
            if sev.get("type") == "CVSS_V3":
                score_str = sev.get("score", "")
                # OSV "score" field may contain either:
                # - A numeric base score (e.g., "9.8") — parsed below
                # - A CVSS vector string ("CVSS:3.1/AV:N/AC:L/...") — float() fails,
                #   falls through to ecosystem_specific/database_specific fallbacks
                try:
                    base_score = float(score_str)
                except (ValueError, TypeError):
                    continue
                if base_score >= 9.0:
                    return "critical"
                if base_score >= 7.0:
                    return "high"
                if base_score >= 4.0:
                    return "medium"
                return "low"

        # Try ecosystem_specific severity (GitHub, PyPI advisories)
        for affected in vuln.get("affected", []):
            eco_sev = (
                affected.get("ecosystem_specific", {})
                .get("severity", "")
                .upper()
            )
            if eco_sev in _SEVERITY_MAP:
                return _SEVERITY_MAP[eco_sev]

        # Try database_specific
        db_sev = vuln.get("database_specific", {}).get("severity", "").upper()
        if db_sev in _SEVERITY_MAP:
            return _SEVERITY_MAP[db_sev]

        # Default
        return "medium"

    def _extract_fixed_version(self, vuln: dict, ecosystem: str) -> str:
        """Extract the fixed version from affected ranges."""
        for affected in vuln.get("affected", []):
            for rng in affected.get("ranges", []):
                for event in rng.get("events", []):
                    if "fixed" in event:
                        return event["fixed"]
        return ""
```

**Depends On:** Step 3 (PackageInfo, VulnFinding dataclasses)

**Verify:** Create unit test with mocked urllib.request that returns sample OSV response. Verify VulnScanner.scan() returns correct VulnFindings. Verify cache stores and retrieves correctly.

**Grounding:** OSV API docs (verified: POST /v1/querybatch, batch up to 1000). urllib.request is stdlib. asyncio.to_thread is Python 3.9+.

---

### Step 6: Create semantic_analyzer.py

**File:** `NEW: src/mirdan/core/semantic_analyzer.py` (parent dir verified via Glob)

**Action:** Write

**Details:**
Create `SemanticAnalyzer` class (~350 lines):

```python
"""Generate semantic review questions and analysis protocols."""

import re
from mirdan.config import SemanticConfig
from mirdan.models import AnalysisProtocol, SemanticCheck, Violation
```

**Key components:**

1. **Pattern detectors** — regex patterns to detect code structures:
   ```python
   _PATTERNS = {
       "sql": [
           (r'(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b', "SQL query"),
           (r'\.(?:execute|query|prepare)\s*\(', "database call"),
       ],
       "auth": [
           (r'(?:authenticat|authoriz|login|logout|session|token|permission|role)\b', "auth logic"),
       ],
       "file_io": [
           (r'\bopen\s*\(', "file open"),
           (r'(?:read|write|readFile|writeFile|fs\.)', "file I/O"),
       ],
       "crypto": [
           (r'(?:encrypt|decrypt|hash|hmac|sign|verify|cipher|aes|rsa|sha)\b', "crypto operation"),
       ],
       "network": [
           (r'(?:requests\.(?:get|post|put|delete)|fetch\s*\(|axios\.|http\.(?:Get|Post))', "network call"),
       ],
       "error_handling": [
           (r'(?:try\s*[:{]|except\s|catch\s*\(|\.catch\s*\()', "error handler"),
       ],
       "loops": [
           (r'(?:for\s+\w+\s+in\b|while\s|for\s*\()', "loop"),
       ],
   }
   ```

2. **Question generators** — for each detected pattern, produce contextual questions:
   ```python
   _QUESTION_TEMPLATES = {
       "sql": "Line {line}: {context}. If this query incorporates any external input, verify it uses parameterized queries — not string concatenation or f-strings.",
       "auth": "Line {line}: {context}. Verify this check executes BEFORE any data access. Check: can this be bypassed by passing null/empty values?",
       "file_io": "Line {line}: {context}. Verify the file is closed in ALL exit paths including exceptions. For Python, prefer context managers (with statement).",
       "crypto": "Line {line}: {context}. Verify the algorithm is currently recommended (SHA-256+ for hashing, AES-256-GCM for encryption, RSA-2048+ for asymmetric).",
       "network": "Line {line}: {context}. Verify error handling for network failures (timeouts, DNS failures, non-2xx responses). Check if SSL verification is enabled.",
       "error_handling": "Line {line}: {context}. Verify caught exception types match what the called functions actually raise. Check for silently swallowed errors.",
       "loops": "Line {line}: {context}. Verify loop termination condition is guaranteed. Check for off-by-one errors and empty collection edge cases.",
   }
   ```

3. **Violation-informed enhancement** — existing violations trigger deeper questions:
   ```python
   _VIOLATION_FOLLOW_UPS = {
       "SEC004": "SQL string concatenation detected. Trace the concatenated variable backward to its SOURCE — where does it enter the system? Is it ever sanitized?",
       "SEC005": "SQL f-string detected. Same as SEC004 — trace the interpolated variable to its origin.",
       "SEC008": "Shell format injection. Trace {var} backward — can an attacker control its value? Check ALL code paths that assign to it.",
       "AI001": "Placeholder detected. What SPECIFIC implementation should replace this? Check the function's callers to understand the expected contract and return type.",
       "AI003": "Over-engineering flagged. Count actual call sites for this abstraction — if fewer than 3, inline it.",
   }
   ```

4. **Analysis protocol generator** (Layer 3):
   ```python
   def generate_analysis_protocol(
       self, code: str, language: str, violations: list[Violation],
       semantic_checks: list[SemanticCheck],
   ) -> AnalysisProtocol | None:
       """Generate structured analysis protocol for security-critical code."""
       # Only generate for security-related code
       focus_areas = []
       # Build focus areas from taint-related semantic checks
       for check in semantic_checks:
           if check.concern == "taint_propagation":
               focus_areas.append({
                   "concern": check.concern,
                   "question": check.question,
                   "focus_lines": check.focus_lines,
               })
       if not focus_areas:
           return None
       return AnalysisProtocol(
           type="security_flow_analysis",
           focus_areas=focus_areas,
           response_format={
               "findings": [{"line": "int", "severity": "str", "issue": "str", "recommendation": "str"}]
           },
       )
   ```

5. **Main entry point:**
   ```python
   def generate_checks(
       self, code: str, language: str, violations: list[Violation],
   ) -> list[SemanticCheck]:
       """Generate semantic review questions from code patterns and violations."""
       if not self._config.enabled:
           return []
       checks: list[SemanticCheck] = []
       lines = code.split("\n")
       # Phase 1: Pattern-based checks
       for pattern_type, patterns in _PATTERNS.items():
           for line_num, line_text in enumerate(lines, 1):
               for regex, context_label in patterns:
                   if re.search(regex, line_text, re.IGNORECASE):
                       template = _QUESTION_TEMPLATES.get(pattern_type)
                       if template:
                           checks.append(SemanticCheck(
                               concern=pattern_type,
                               question=template.format(line=line_num, context=f"{context_label} detected: `{line_text.strip()[:80]}`"),
                               severity="warning" if pattern_type in ("sql", "auth", "crypto") else "info",
                               focus_lines=[line_num],
                           ))
       # Phase 2: Violation-informed follow-ups
       for violation in violations:
           follow_up = _VIOLATION_FOLLOW_UPS.get(violation.id)
           if follow_up and violation.line:
               checks.append(SemanticCheck(
                   concern="violation_deep_dive",
                   question=f"Line {violation.line}: {follow_up}",
                   severity=violation.severity,
                   related_violation=violation.id,
                   focus_lines=[violation.line],
               ))
       # Deduplicate by line+concern
       seen: set[tuple[int, str]] = set()
       deduped: list[SemanticCheck] = []
       for check in checks:
           key = (check.focus_lines[0] if check.focus_lines else 0, check.concern)
           if key not in seen:
               seen.add(key)
               deduped.append(check)
       return deduped
   ```

**Depends On:** Step 1 (SemanticConfig), Step 3 (SemanticCheck, AnalysisProtocol, Violation)

**Verify:** Unit test: pass Python code with `cursor.execute(f"SELECT * FROM users WHERE id={user_id}")` → verify SemanticCheck with concern="sql" is generated with correct line number.

**Grounding:** Read of code_validator.py confirmed regex-based pattern matching approach. Read of models.py confirmed Violation structure. Pattern detectors derived from existing SEC rule patterns in code_validator.py.

---

## Phase 3: Validation Integration

### Step 7: Extend ai_quality_checker.py with SEC014 + ManifestParser

**File:** `src/mirdan/core/ai_quality_checker.py` (verified via Read)

**Action:** Edit

**Details:**
- Add `manifest_parser` and `vuln_scanner` parameters to `__init__` (around L165-169):
  ```python
  def __init__(self, project_dir: Path | None = None,
               manifest_parser: ManifestParser | None = None,
               vuln_scanner: VulnScanner | None = None) -> None:
      self._project_dir = project_dir
      self._project_deps = self._load_project_deps(project_dir) if project_dir else set()
      self._manifest_parser = manifest_parser
      self._vuln_scanner = vuln_scanner
  ```
- Add `_check_sec014_vulnerable_deps` method (after AI008 check, around L200):
  ```python
  def _check_sec014_vulnerable_deps(self, code: str, language: str) -> list[Violation]:
      """Check if imported packages have known cached vulnerabilities."""
      if not self._vuln_scanner or not self._manifest_parser:
          return []
      # Extract imports from code (reuse AI002 import extraction)
      imports = self._extract_imports(code, language)
      if not imports:
          return []
      # Get cached vulnerability findings only (no network call)
      packages = self._manifest_parser.parse()
      cached_findings = self._vuln_scanner.check_cached(packages)
      # Cross-reference: does this code import a vulnerable package?
      violations = []
      import_names = {imp.split(".")[0] for imp in imports}
      for finding in cached_findings:
          pkg_import_name = finding.package.replace("-", "_").lower()
          if pkg_import_name in {n.lower() for n in import_names}:
              sev = "error" if finding.severity in ("critical", "high") else "warning"
              violations.append(Violation(
                  id="SEC014",
                  rule="vulnerable-dependency",
                  category="security",
                  severity=sev,
                  message=f"Package '{finding.package}' v{finding.version} has vulnerability {finding.vuln_id}: {finding.summary[:100]}",
                  suggestion=f"Upgrade to v{finding.fixed_version}" if finding.fixed_version else "Check advisory for remediation",
              ))
      return violations
  ```
- Call `_check_sec014_vulnerable_deps` from `check()` method (around L198):
  ```python
  violations.extend(self._check_sec014_vulnerable_deps(code, language))
  ```
- **Also add to `check_quick()` (L218-220)** — SEC014 is a zero-cost cache lookup (no
  network call), making it safe and important for hook-triggered quick validation:
  ```python
  # In check_quick(), after existing AI008 call (L220):
  violations.extend(self._check_sec014_vulnerable_deps(code, language))
  ```
- Refactor: The import extraction in `_check_ai002_python` (L404-446) is currently inline
  per-language regex. Extract this into a new shared `_extract_imports(code, language)` method
  that both AI002 and SEC014 can call. This is a refactoring of existing inline code, not
  extraction of an existing shared method
- **Refactor `_load_project_deps` to delegate to ManifestParser** (L611-650):
  ```python
  def _load_project_deps(self, project_dir: Path) -> frozenset[str]:
      """Load project dependency names using ManifestParser."""
      deps: set[str] = set()
      if self._manifest_parser:
          packages = self._manifest_parser.parse(project_dir)
          deps.update(p.name for p in packages)
      else:
          # Fallback: original inline parsing (when no ManifestParser injected)
          deps.update(self._parse_pyproject_deps(project_dir / "pyproject.toml"))
          # ... (keep existing fallback logic)
      # CRITICAL: Preserve these — they prevent AI002 false positives
      deps.update(_COMMON_TRANSITIVE_PACKAGES)  # L79-83: requests, typing_extensions, etc.
      deps.update(self._find_local_packages(project_dir))  # L715-726: src layout detection
      return frozenset(deps)
  ```
  This delegates manifest parsing to ManifestParser while PRESERVING
  `_COMMON_TRANSITIVE_PACKAGES` and `_find_local_packages()` which are NOT
  manifest-derived — they're heuristic additions that prevent AI002 false positives
  for transitive dependencies and self-imports.
- **Add SEC014 to violation_explainer.py** (after SEC013 at L91-93):
  ```python
  "SEC014": (
      "Using dependencies with known vulnerabilities exposes"
      " the application to attacks that have public exploits."
  ),
  ```

**Depends On:** Step 4 (ManifestParser), Step 5 (VulnScanner)

**Verify:** Unit test: mock VulnScanner with cached finding for "requests" package, pass code with `import requests`, assert SEC014 violation is generated.

**Grounding:** Read of ai_quality_checker.py confirmed check() at L175-200, AI002 import extraction at L404-446, _load_project_deps at L169.

---

### Step 8: Add SEC014 fix template to auto_fixer.py

**File:** `src/mirdan/core/auto_fixer.py` (verified via Read)

**Action:** Edit

**Details:**
- **NOTE:** TEMPLATE_FIXES values are returned as LITERAL strings (auto_fixer.py L312:
  `fix_code=fix_code`). Template variables like `{package}` are NOT substituted.
  For SEC014, the fix is a manifest change, not a code transform. Use the violation's
  `suggestion` field instead, which already contains "Upgrade to vX.Y.Z" text.
- Add SEC014 to TEMPLATE_FIXES with a generic instruction (around L200):
  ```python
  "SEC014": (
      "# Upgrade vulnerable dependency — see violation suggestion for target version",
      "Upgrade vulnerable dependency to patched version",
      0.5,  # Low confidence — requires manifest edit, not code edit
  ),
  ```
- Do NOT add SEC014 to `_QUICK_FIX_RULES` (L419-423) because:
  - quick_fix() requires confidence >= 0.8 (L442), but SEC014 is 0.5
  - Dependency upgrades shouldn't be auto-suggested in quick mode

**Depends On:** Step 7 (SEC014 rule exists)

**Verify:** Read auto_fixer.py, confirm SEC014 in TEMPLATE_FIXES with confidence=0.5 and SEC014 NOT in _QUICK_FIX_RULES.

**Grounding:** Read of auto_fixer.py confirmed TEMPLATE_FIXES dict at L39-202, _QUICK_FIX_RULES at L419-423.

---

### Step 9: Integrate semantic_analyzer into code_validator.py

**File:** `src/mirdan/core/code_validator.py` (verified via Read)

**Action:** Edit

**Details:**
- This step is about making code_validator AWARE of semantic analysis, but the actual semantic_checks are generated in server.py (not in code_validator). The reason: semantic_checks aren't violations — they're LLM guidance. They shouldn't affect the quality score.
- Add `semantic_analyzer`, `manifest_parser`, and `vuln_scanner` parameters to `CodeValidator.__init__`:
  ```python
  def __init__(
      self,
      standards: QualityStandards,
      config: QualityConfig | None = None,
      thresholds: ThresholdsConfig | None = None,
      project_dir: Path | None = None,
      semantic_analyzer: SemanticAnalyzer | None = None,
      manifest_parser: ManifestParser | None = None,
      vuln_scanner: VulnScanner | None = None,
  ):
      # ... existing init ...
      self._semantic_analyzer = semantic_analyzer
      self._manifest_parser = manifest_parser
      self._vuln_scanner = vuln_scanner
  ```
- **CRITICAL:** Update the lazy AIQualityChecker initialization at line 737-739 to
  forward manifest_parser and vuln_scanner:
  ```python
  # Was:
  from mirdan.core.ai_quality_checker import AIQualityChecker
  self._ai_checker = AIQualityChecker(project_dir)
  # Now:
  from mirdan.core.ai_quality_checker import AIQualityChecker
  self._ai_checker = AIQualityChecker(
      project_dir,
      manifest_parser=self._manifest_parser,
      vuln_scanner=self._vuln_scanner,
  )
  ```
- Add a convenience method for server.py to call:
  ```python
  def generate_semantic_checks(self, code: str, language: str,
                                violations: list[Violation]) -> list[SemanticCheck]:
      """Generate semantic review questions (does not affect quality score)."""
      if not self._semantic_analyzer:
          return []
      return self._semantic_analyzer.generate_checks(code, language, violations)
  ```

**Depends On:** Step 6 (SemanticAnalyzer), Step 7 (AIQualityChecker accepts manifest_parser/vuln_scanner). **Must be done BEFORE or TOGETHER with Step 10** (both modify CodeValidator.__init__).

**Verify:** Read code_validator.py, confirm new method exists and delegates to semantic_analyzer. Verify AIQualityChecker lazy init at L737-739 now forwards manifest_parser and vuln_scanner.

**Grounding:** Read of code_validator.py confirmed __init__ parameters and validate() method structure. Lazy AIQualityChecker creation at L737-739.

---

## Phase 4: Server & MCP

### Step 10: Add new components to _Components in server.py

**File:** `src/mirdan/server.py` (verified via Read at L56-89)

**Action:** Edit

**Details:**
- Add imports at top of file:
  ```python
  from mirdan.core.manifest_parser import ManifestParser
  from mirdan.core.vuln_scanner import VulnScanner
  from mirdan.core.semantic_analyzer import SemanticAnalyzer
  ```
- Add fields to `_Components` dataclass (after line 73):
  ```python
  manifest_parser: ManifestParser
  vuln_scanner: VulnScanner
  semantic_analyzer: SemanticAnalyzer
  ```
- Initialize in `_get_components()` (around the component initialization block, ~L130-165):
  ```python
  manifest_parser = ManifestParser(project_dir=project_dir)
  vuln_scanner = VulnScanner(
      cache_dir=(project_dir / ".mirdan" / "cache") if project_dir else Path(".mirdan/cache"),
      ttl=config.dependencies.osv_cache_ttl,
  )
  semantic_analyzer = SemanticAnalyzer(config=config.semantic)
  ```
- **IMPORTANT:** AIQualityChecker is NOT initialized in server.py. It is lazily created
  inside `CodeValidator` at `code_validator.py:737-739`:
  ```python
  from mirdan.core.ai_quality_checker import AIQualityChecker
  self._ai_checker = AIQualityChecker(project_dir)
  ```
  Therefore, `manifest_parser` and `vuln_scanner` must be passed THROUGH CodeValidator,
  which will forward them to AIQualityChecker when it creates it.

- Pass `manifest_parser`, `vuln_scanner`, and `semantic_analyzer` to CodeValidator:
  ```python
  code_validator = CodeValidator(
      quality_standards, config=config.quality, thresholds=config.thresholds,
      project_dir=project_dir, semantic_analyzer=semantic_analyzer,
      manifest_parser=manifest_parser, vuln_scanner=vuln_scanner,
  )
  ```
  CodeValidator will then forward manifest_parser and vuln_scanner to AIQualityChecker
  at line 737-739 (see Step 9 for the CodeValidator changes).
- Update `_TOOL_PRIORITY` (line 82-89) to add `scan_dependencies`.
  **NOTE:** `_TOOL_PRIORITY` determines which tools are exposed when `MIRDAN_TOOL_BUDGET`
  is set (Cursor's 40-tool limit). Place `scan_dependencies` BEFORE `scan_conventions`
  since vulnerability scanning is higher priority than convention discovery:
  ```python
  _TOOL_PRIORITY = [
      "validate_code_quality",
      "validate_quick",
      "enhance_prompt",
      "get_quality_standards",
      "get_quality_trends",
      "scan_dependencies",
      "scan_conventions",
  ]
  ```

**Depends On:** Steps 4, 5, 6 (all three new modules)

**Verify:** Run `uv run mirdan --help` — no import errors. Run `uv run python -c "from mirdan.server import _get_components"` — no errors.

**Grounding:** Read of server.py confirmed _Components at L56-75, _TOOL_PRIORITY at L82-89, initialization block exists.

---

### Step 11: Add scan_dependencies MCP tool to server.py

**File:** `src/mirdan/server.py` (verified via Read at L754-779)

**Action:** Edit

**Details:**
- Add new tool AFTER scan_conventions (after line 779):
  ```python
  # ---------------------------------------------------------------------------
  # Core Tool 6: scan_dependencies
  # ---------------------------------------------------------------------------


  @mcp.tool()
  async def scan_dependencies(
      project_path: str = ".",
      ecosystem: str = "auto",
  ) -> dict[str, Any]:
      """Scan project dependencies for known vulnerabilities.

      Queries the OSV database (free, no API key) to check all dependencies
      against known CVEs. Results are cached per config (default: 24 hours).

      Args:
          project_path: Project directory containing dependency manifests
          ecosystem: Filter by ecosystem (auto|PyPI|npm|crates.io|Go|Maven)

      Returns:
          Scan results with packages checked and vulnerabilities found
      """
      c = _get_components()
      scan_dir = Path(project_path).resolve()

      if not scan_dir.is_dir():
          return {"error": f"Not a directory: {project_path}"}

      packages = c.manifest_parser.parse(scan_dir)
      if ecosystem != "auto":
          packages = [p for p in packages if p.ecosystem == ecosystem]

      if not packages:
          return {
              "packages_scanned": 0,
              "vulnerabilities_found": 0,
              "message": "No dependency manifests found",
              "findings": [],
          }

      findings = await c.vuln_scanner.scan(packages)

      # Group by severity
      severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
      for f in findings:
          severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

      return {
          "packages_scanned": len(packages),
          "vulnerabilities_found": len(findings),
          "severity_counts": severity_counts,
          "findings": [f.to_dict() for f in findings],
          "ecosystems_checked": list({p.ecosystem for p in packages}),
      }
  ```

**Depends On:** Step 10 (_Components updated)

**Verify:** Run mirdan server, call scan_dependencies tool from MCP client. Or test via `uv run python -c "import asyncio; from mirdan.server import scan_dependencies; print(asyncio.run(scan_dependencies()))"`.

**Grounding:** Read of server.py confirmed tool registration pattern at L754-779 (scan_conventions).

---

### Step 12: Extend validate_code_quality with semantic_checks

**File:** `src/mirdan/server.py` (verified via Read at L420-475)

**Action:** Edit

**Details:**
- After `output = result.to_dict(...)` (line 434), before session_quality (line 437), add:
  ```python
  # Generate semantic review questions (Layer 1)
  if c.config.semantic.enabled:
      semantic_checks = c.code_validator.generate_semantic_checks(
          code=code, language=result.language_detected,
          violations=result.violations,
      )
      if semantic_checks:
          output["semantic_checks"] = [s.to_dict() for s in semantic_checks]

      # Layer 3: Analysis protocol for security-critical code
      if resolved_security and c.config.semantic.analysis_protocol != "none":
          protocol = c.semantic_analyzer.generate_analysis_protocol(
              code=code, language=result.language_detected,
              violations=result.violations,
              semantic_checks=semantic_checks,
          )
          if protocol:
              output["analysis_protocol"] = protocol.to_dict()
  ```

**Depends On:** Steps 9, 10 (code_validator has semantic_analyzer, _Components updated)

**Verify:** Call validate_code_quality with code containing SQL query, verify response includes `semantic_checks` array with SQL-related question.

**Grounding:** Read of server.py confirmed output building at L434-473. Insertion point is between to_dict() and session_quality.

---

## Phase 5: Output & Integration

### Step 13: Update output_formatter.py for semantic_checks

**File:** `src/mirdan/core/output_formatter.py` (verified via Read)

**Action:** Edit

**Details:**
- **IMPORTANT:** `_compact_validation()` (L227-246) uses KEY WHITELISTING — it constructs
  a NEW dict with only specific keys (passed, score, language_detected, violations_count,
  violations, summary). Unknown keys are NOT passed through. To include semantic_checks,
  it must be EXPLICITLY added to the returned dict:
  ```python
  # In _compact_validation, after the return dict construction (L239-246):
  result = {
      "passed": data.get("passed", True),
      "score": data.get("score", 1.0),
      "language_detected": data.get("language_detected", ""),
      "violations_count": data.get("violations_count", {}),
      "violations": compact_violations,
      "summary": data.get("summary", ""),
  }
  # Add semantic_checks (limited to top 3 in compact mode)
  if "semantic_checks" in data:
      result["semantic_checks"] = data["semantic_checks"][:3]
  return result
  ```
- `_minimal_validation()` (L248-254): Same whitelisting pattern — only returns
  passed, score, summary. No change needed (semantic_checks correctly excluded).
- `_micro_validation()` (L328-355): Only returns passed, score, micro. No change needed.

**Depends On:** Step 12 (semantic_checks in response)

**Verify:** Call validate with max_tokens=3000 (triggers COMPACT), verify semantic_checks limited to 3. Call with max_tokens=500 (triggers MINIMAL), verify semantic_checks absent.

**Grounding:** Read of output_formatter.py confirmed format levels and method locations.

---

### Step 14: Update hook_templates.py with dependency-aware hooks

**File:** `src/mirdan/integrations/hook_templates.py` (verified via Read at L31-56)

**Action:** Edit

**Details:**
- Add a dependency check to the existing `_post_tool_use()` method (around L281-314).
  Add an additional hook entry for dependency manifest detection:
  ```python
  # Dependency manifest change detection
  {
      "matcher": "Write|Edit",
      "hooks": [{
          "type": "prompt",
          "prompt": (
              "If the file just modified is a dependency manifest "
              "(package.json, pyproject.toml, Cargo.toml, go.mod, requirements.txt, pom.xml), "
              "call mcp__mirdan__scan_dependencies to check for vulnerabilities "
              "in the updated dependencies."
          ),
          "timeout": 15000,
      }],
  }
  ```
- Only include this hook at STANDARD and COMPREHENSIVE stringency levels.

**Depends On:** Step 11 (scan_dependencies tool exists)

**Verify:** Generate hooks with STANDARD stringency, verify dependency manifest prompt appears in PostToolUse hooks.

**Grounding:** Read of hook_templates.py confirmed _post_tool_use at L281-314, STRINGENCY_EVENTS at L31-56.

---

### Step 15: Update cursor.py BUGBOT.md with SEC014

**File:** `src/mirdan/integrations/cursor.py` (verified via Read, _generate_bugbot_md at L724-825)

**Action:** Edit

**Details:**
- Add SEC014 section to the "Blocking Bugs (Critical)" section of BUGBOT.md (around L750):
  ```python
  # In the blocking bugs section:
  """
  ### SEC014 — Vulnerable Dependency
  If a dependency manifest is modified, verify no known-vulnerable package versions are introduced.
  ```regex
  "dependencies"
  \\[project\\.dependencies\\]
  \\[dependencies\\]
  require\\s*\\(
  ```
  """
  ```
- **Clarification:** SEC011-013 are already in the cursor.py rules TABLE (L708-711).
  They are NOT in the blocking bugs REGEX section (which ends at SEC010, L803).
  Add them to the regex blocking bugs section with appropriate patterns:
  ```python
  """
  ### SEC011 — Cypher Injection (f-string)
  ```regex
  f["'].*MATCH\\b|f["'].*MERGE\\b|f["'].*CREATE\\b
  ```

  ### SEC012 — Cypher Injection (concat)
  ```regex
  ["']\\s*\\+.*MATCH\\b|["']\\s*\\+.*MERGE\\b
  ```

  ### SEC013 — Gremlin Injection (f-string)
  ```regex
  f["'].*\\.V\\(|f["'].*\\.addV\\(
  ```
  """
  ```

**Depends On:** Step 7 (SEC014 rule exists)

**Verify:** Generate BUGBOT.md via `mirdan init --cursor` in tmp dir, verify SEC014 section present.

**Grounding:** Read of cursor.py confirmed _generate_bugbot_md at L724-825, blocking bugs REGEX section ends at SEC010 (L803), rules TABLE has SEC011-013 at L708-711.

---

### Step 15b: Update SEC014 references in integration files

**Files:** Multiple integration files (verified via Grep for "SEC013")

**Action:** Edit

**Details:**
- **`src/mirdan/integrations/agents_md.py`** (L172-176): Add SEC014 to security rules list:
  ```python
  - No vulnerable dependencies — upgrade packages with known CVEs (SEC014)
  ```
- **`src/mirdan/integrations/templates/claude_code/agents/security-audit.md`** (L33):
  Update range from "SEC001-SEC013" to "SEC001-SEC014"
- **`src/mirdan/integrations/templates/claude_code/mirdan-security.md`** (after L31):
  Add:
  ```
  - **SEC014**: No vulnerable dependencies — upgrade packages with known CVEs
  ```

**Depends On:** Step 7 (SEC014 rule exists)

**Verify:** Grep for "SEC014" in integrations/ and templates/ — should appear in all 3 files.

**Grounding:** Grep for SEC013 confirmed these 3 files reference SEC001-SEC013 and need SEC014.

---

### Step 16: Extend knowledge_producer.py for dependency knowledge

**File:** `src/mirdan/core/knowledge_producer.py` (verified via Read at L20-194)

**Action:** Edit

**Details:**
- Add `_extract_dependency_knowledge()` method (after _extract_convention_insights, ~L194):
  ```python
  def _extract_dependency_knowledge(
      self, result: ValidationResult, file_path: str,
  ) -> list[KnowledgeEntry]:
      """Extract dependency vulnerability knowledge."""
      sec014_violations = [v for v in result.violations if v.id == "SEC014"]
      if not sec014_violations:
          return []
      pattern_key = f"vuln_deps:{file_path or 'project'}"
      if pattern_key in self._seen_patterns:
          return []
      self._seen_patterns.add(pattern_key)
      packages = {v.message.split("'")[1] for v in sec014_violations if "'" in v.message}
      return [KnowledgeEntry(
          content=f"Vulnerable dependencies detected: {', '.join(sorted(packages))}. Run mirdan scan --dependencies for details.",
          content_type="fact",
          tags=["security", "dependencies", "vulnerabilities"],
          scope="project",
          scope_path=file_path or "",
          confidence=0.95,
      )]
  ```
- Call from `extract_from_validation()` (around L48):
  ```python
  entries.extend(self._extract_dependency_knowledge(result, file_path))
  ```

**Depends On:** Step 7 (SEC014 violations exist)

**Verify:** Pass ValidationResult with SEC014 violation, verify KnowledgeEntry is produced with correct tags.

**Grounding:** Read of knowledge_producer.py confirmed extraction pattern at L20-48, _extract_security_knowledge at L130-163.

---

## Phase 6: CLI

### Step 17: Extend mirdan scan with --dependencies flag

**File:** `src/mirdan/cli/scan_command.py` (verified via Glob)

**Action:** Edit

**Details:**
- Read current scan_command.py first (before editing)
- Add `--dependencies` flag parsing:
  ```python
  if "--dependencies" in args or "--deps" in args:
      return run_dependency_scan(args)
  ```
- Add `run_dependency_scan()` function:
  ```python
  def run_dependency_scan(args: list[str]) -> None:
      """Scan project dependencies for vulnerabilities."""
      import asyncio
      from mirdan.core.manifest_parser import ManifestParser
      from mirdan.core.vuln_scanner import VulnScanner
      from mirdan.config import MirdanConfig

      config = MirdanConfig.find_config()
      project_dir = Path(args[args.index("--directory") + 1] if "--directory" in args else ".")

      parser = ManifestParser(project_dir=project_dir)
      packages = parser.parse()

      if not packages:
          print("No dependency manifests found.")
          return

      print(f"Scanning {len(packages)} packages...")
      scanner = VulnScanner(
          cache_dir=project_dir / ".mirdan" / "cache",
          ttl=config.dependencies.osv_cache_ttl,
      )
      findings = asyncio.run(scanner.scan(packages))

      if not findings:
          print(f"No vulnerabilities found in {len(packages)} packages.")
          return

      # Print results table
      print(f"\nFound {len(findings)} vulnerabilities:\n")
      for f in sorted(findings, key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x.severity, 4)):
          sev_badge = {"critical": "CRIT", "high": "HIGH", "medium": "MED", "low": "LOW"}.get(f.severity, "???")
          fix_str = f" → fix: {f.fixed_version}" if f.fixed_version else ""
          print(f"  [{sev_badge}] {f.package}@{f.version}: {f.vuln_id}{fix_str}")
          print(f"         {f.summary[:80]}")
  ```

**Depends On:** Steps 4, 5 (ManifestParser, VulnScanner)

**Verify:** Run `mirdan scan --dependencies` in a project with pyproject.toml. Verify it scans and reports (or reports "No vulnerabilities found").

**Grounding:** Read of cli/__init__.py confirmed scan routing at L38-39. Glob confirmed scan_command.py exists.

---

### Step 18: Extend mirdan gate with --include-dependencies

**File:** `src/mirdan/cli/gate_command.py` (verified via Glob)

**Action:** Edit

**Details:**
- Read current gate_command.py first (before editing)
- Add `--include-dependencies` / `--include-deps` flag:
  ```python
  include_deps = "--include-dependencies" in args or "--include-deps" in args
  ```
- **IMPORTANT:** `gate_command.py` uses CodeValidator DIRECTLY (not through server.py's
  `_Components`). It does NOT have access to manifest_parser or vuln_scanner from server.py.
  Must instantiate them locally, same as Step 17 (scan_command):
  ```python
  import asyncio
  from mirdan.core.manifest_parser import ManifestParser
  from mirdan.core.vuln_scanner import VulnScanner

  # ... after code validation passes ...
  if include_deps or config.dependencies.scan_on_gate:
      project_dir = Path.cwd()
      manifest_parser = ManifestParser(project_dir=project_dir)
      packages = manifest_parser.parse()
      if packages:
          vuln_scanner = VulnScanner(
              cache_dir=project_dir / ".mirdan" / "cache",
              ttl=config.dependencies.osv_cache_ttl,
          )
          findings = asyncio.run(vuln_scanner.scan(packages))
          severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
          fail_threshold = severity_order.get(config.dependencies.fail_on_severity, 1)
          blocking = [f for f in findings if severity_order.get(f.severity, 4) <= fail_threshold]
          if blocking:
              print(f"FAIL: {len(blocking)} dependency vulnerabilities at or above '{config.dependencies.fail_on_severity}' severity")
              sys.exit(1)
  ```

**Depends On:** Steps 4, 5, 1 (ManifestParser, VulnScanner, DependencyConfig)

**Verify:** Run `mirdan gate --include-deps` in a project. Verify exit code 0 if no vulns, exit code 1 if blocking vulns.

**Grounding:** Agent verified gate_command.py uses CodeValidator directly (not server.py _Components). ManifestParser and VulnScanner must be instantiated locally.

---

## Phase 7: Tests & Documentation

### Step 19: Create test_semantic_analyzer.py

**File:** `NEW: tests/test_semantic_analyzer.py` (parent dir verified via Glob)

**Action:** Write

**Details:**
Test class `TestSemanticAnalyzer` with:
1. `test_sql_pattern_generates_check` — code with `cursor.execute(f"...")` → semantic check with concern="sql"
2. `test_auth_pattern_generates_check` — code with `authenticate(user)` → concern="auth"
3. `test_file_io_generates_check` — code with `open(path)` → concern="file_io"
4. `test_crypto_generates_check` — code with `hashlib.sha256()` → concern="crypto"
5. `test_violation_follow_up` — pass SEC004 violation → generates deeper question
6. `test_empty_code_returns_empty` — empty string → empty list
7. `test_disabled_returns_empty` — config.enabled=False → empty list
8. `test_deduplication` — same pattern on same line → only one check
9. `test_analysis_protocol_generated` — security code → protocol with focus_areas
10. `test_analysis_protocol_not_generated_when_disabled` — analysis_protocol="none" → None

Test class `TestSemanticCheckModel`:
11. `test_to_dict_minimal` — only required fields
12. `test_to_dict_full` — all fields including related_violation

**Depends On:** Step 6 (SemanticAnalyzer)

**Verify:** Run `uv run pytest tests/test_semantic_analyzer.py -v` — all pass.

**Grounding:** Test patterns verified from existing tests (class-based, setup_method, type hints, direct assertions).

---

### Step 20: Create test_manifest_parser.py

**File:** `NEW: tests/test_manifest_parser.py` (parent dir verified via Glob)

**Action:** Write

**Details:**
Test class `TestManifestParser` with:
1. `test_parse_pyproject_toml` — write pyproject.toml with dependencies → correct PackageInfo list
2. `test_parse_requirements_txt` — write requirements.txt → correct packages
3. `test_parse_package_json` — write package.json → npm packages
4. `test_parse_cargo_toml` — write Cargo.toml → crates.io packages
5. `test_parse_go_mod` — write go.mod → Go packages
6. `test_no_manifests_returns_empty` — empty dir → empty list
7. `test_malformed_manifest_doesnt_crash` — invalid TOML → empty list, no exception
8. `test_cache_invalidation` — parse, modify file, re-parse → updated results
9. `test_get_version` — specific package version lookup
10. `test_multiple_manifests` — dir with both pyproject.toml and package.json → combined

All tests use `tmp_path` fixture to create temporary manifest files.

**Depends On:** Step 4 (ManifestParser)

**Verify:** Run `uv run pytest tests/test_manifest_parser.py -v` — all pass.

**Grounding:** Test patterns from existing tests use tmp_path for file-based tests.

---

### Step 21: Create test_vuln_scanner.py

**File:** `NEW: tests/test_vuln_scanner.py` (parent dir verified via Glob)

**Action:** Write

**Details:**
Test class `TestVulnCache` with:
1. `test_store_and_retrieve` — store finding, retrieve by same package → found
2. `test_ttl_expiry` — store with ttl=1, sleep(2), retrieve → None
3. `test_cache_persistence` — store, create new VulnCache from same path → still found
4. `test_cache_size_limit` — store >1000 entries → oldest evicted

Test class `TestVulnScanner` with:
5. `test_check_cached_empty` — no cache → empty list
6. `test_check_cached_returns_findings` — pre-populated cache → findings returned
7. `test_scan_with_mocked_api` — mock urllib.request.urlopen → findings parsed correctly
8. `test_scan_network_error_graceful` — mock urlopen to raise URLError → empty findings, no crash
9. `test_severity_mapping` — various CVSS scores → correct severity strings
10. `test_fixed_version_extraction` — OSV response with fixed event → correct version

All API calls mocked with `unittest.mock.patch("urllib.request.urlopen")`.

**Depends On:** Step 5 (VulnScanner)

**Verify:** Run `uv run pytest tests/test_vuln_scanner.py -v` — all pass.

**Grounding:** Test patterns use unittest.mock.patch for external calls. VulnScanner uses urllib.request.urlopen.

---

### Step 21b: Extend existing test files for integration coverage

**Files:** Existing test files (verified via Glob: `mirdan/tests/test_*.py`)

**Action:** Edit (extend existing files)

**Details:**
The plan's new test files (Steps 19-21) cover the 3 new standalone modules. But 6 existing
test files need extensions to cover the integration points and maintain 85% coverage:

1. **`tests/test_ai_quality_checker.py`** — Add:
   - `test_sec014_cached_vuln_generates_violation` — mock VulnScanner with cached finding for "requests", pass code with `import requests`, assert SEC014 violation
   - `test_sec014_no_cache_no_violation` — empty cache → no SEC014
   - `test_sec014_unrelated_import_no_violation` — cached vuln for "requests" but code imports "flask" → no SEC014
   - `test_load_project_deps_delegates_to_manifest_parser` — verify _load_project_deps now uses ManifestParser

2. **`tests/test_auto_fix.py`** — Add:
   - `test_sec014_fix_available` — verify SEC014 returns a FixResult with confidence=0.5
   - `test_sec014_not_in_quick_fix_rules` — verify SEC014 is NOT in _QUICK_FIX_RULES

3. **`tests/test_output_formatter.py`** — Add:
   - `test_compact_validation_includes_semantic_checks` — verify semantic_checks limited to 3 in compact
   - `test_compact_validation_without_semantic_checks` — verify no error when semantic_checks absent
   - `test_minimal_validation_excludes_semantic_checks` — verify excluded
   - `test_full_validation_preserves_semantic_checks` — verify passed through

4. **`tests/test_knowledge_producer.py`** — Add:
   - `test_extract_dependency_knowledge_from_sec014` — verify KnowledgeEntry with correct tags

5. **`tests/test_code_validator.py`** — Add:
   - `test_generate_semantic_checks_returns_checks` — verify semantic_analyzer integration
   - `test_generate_semantic_checks_disabled` — verify empty when no semantic_analyzer

6. **`tests/test_server.py`** — Add:
   - `test_validate_code_quality_includes_semantic_checks` — end-to-end: code with SQL → semantic_checks in response
   - `test_scan_dependencies_tool` — end-to-end: project with pyproject.toml → scan results

**Depends On:** All previous implementation steps

**Verify:** Run `uv run pytest tests/ -v --cov=mirdan --cov-fail-under=85` — all pass, coverage ≥ 85%.

**Grounding:** Glob confirmed all 6 test files exist at `mirdan/tests/test_*.py`. Each file follows class-based test pattern with setup_method.

---

### Step 22: Update README.md

**File:** `README.md` (verified via Read)

**Action:** Edit

**Details:**
- Update "6 MCP Tools" to "7 MCP Tools" in features section
- Add `scan_dependencies` row to MCP tools table:
  ```
  | `scan_dependencies` | Scan project dependencies for known vulnerabilities (OSV database) |
  ```
- Update "12 CLI Commands" — no new commands, but note new flags:
  - Add `--dependencies` to `mirdan scan` description
  - Add `--include-dependencies` to `mirdan gate` description
- Add SEC014 to Security Rules table:
  ```
  | SEC014 | vulnerable-dependency | Package has known CVE — upgrade to patched version |
  ```
- Update auto-fixable rule count (31 → 32)
- Update total rule count (61 → 62)
- Add "Semantic Validation" to Advanced Features section:
  ```markdown
  ### Semantic Validation

  `validate_code_quality` returns `semantic_checks` — targeted review questions
  generated from code patterns. These guide the LLM to investigate specific concerns
  like taint propagation, auth flow ordering, and resource management. For security-critical
  code, an `analysis_protocol` provides a structured framework for deep analysis.
  ```
- Add "Dependency Vulnerability Scanning" to Advanced Features section:
  ```markdown
  ### Dependency Vulnerability Scanning

  `scan_dependencies` checks project dependencies against the OSV database (free, no API key).
  Supports PyPI, npm, crates.io, Go, and Maven ecosystems. Results are cached for 24 hours.

  mirdan gate --include-dependencies    # Quality gate + vuln check
  mirdan scan --dependencies            # Standalone dependency scan
  ```

**Depends On:** All previous steps

**Verify:** Read README.md, confirm tool count is 7, SEC014 listed, new features documented.

**Grounding:** Read of README.md confirmed current structure (just rewritten in previous task).

---

## Validation Checklist

After all steps complete:

```
□ uv run pytest — all tests pass
□ uv run pytest --cov=mirdan --cov-fail-under=85 — coverage ≥ 85%
□ uv run ruff check src/ — no lint errors
□ uv run mypy src/mirdan/ — no type errors
□ mirdan --help — shows all commands without import errors
□ mirdan validate --file src/mirdan/server.py — validates without crash
□ mirdan scan --dependencies — runs (may find 0 vulns, that's OK)
□ mirdan gate — exits 0 on clean project
□ Tool count: grep -c "@mcp.tool" src/mirdan/server.py → 7
□ Rule count: SEC014 appears in ai_quality_checker.py
□ SEC014 in violation_explainer.py — grep SEC014 violation_explainer.py → 1 match
□ SEC014 in agents_md.py — grep SEC014 agents_md.py → 1 match
□ SEC014 in mirdan-security.md — grep SEC014 mirdan-security.md → 1 match
□ SEC014 in security-audit.md — grep SEC014 security-audit.md → 1 match
□ Config: MirdanConfig().semantic.enabled == True
□ Config: MirdanConfig().dependencies.enabled == True
□ Profile: get_profile("enterprise").semantic == 0.9
□ Profile: get_profile("enterprise").dependency_security == 1.0
□ TEMPLATE_FIXES["SEC014"] exists — confidence=0.5
□ SEC014 NOT in _QUICK_FIX_RULES
□ _load_project_deps delegates to ManifestParser — still includes _COMMON_TRANSITIVE_PACKAGES and _find_local_packages
□ check_quick() calls _check_sec014_vulnerable_deps — grep confirms
□ Output: validate with SQL code → "semantic_checks" key in FULL response
□ Output: validate with max_tokens=3000 → semantic_checks limited to 3
□ _TOOL_PRIORITY has scan_dependencies before scan_conventions
```

---

## Dependency Graph

```
Step 1 (config) ──┬──→ Step 2 (profiles)
                  │
Step 3 (models) ──┼──→ Step 4 (manifest_parser) ──┬──→ Step 7 (SEC014 + explainer + _load_project_deps refactor)
                  │                                │         │
                  ├──→ Step 5 (vuln_scanner) ──────┘         ├──→ Step 8 (auto_fix)
                  │                                          ├──→ Step 15 (BUGBOT) + Step 15b (integration files)
                  │                                          └──→ Step 16 (knowledge_producer)
                  │
                  └──→ Step 6 (semantic_analyzer) ──→ Step 9 (code_validator) ─┐
                                                          │  [depends on 7]    │
                                                          ▼                    │
                                                     Step 10 (_Components) ◄───┘
                                                          │  [SEQUENTIAL with 9]
                                                          ▼
                                                     Step 11 (scan_deps tool)
                                                          │
                                                          ▼
                                                     Step 12 (validate semantic)
                                                          │
                                                          ▼
                                                     Step 13 (output_formatter)

Step 11 ──→ Step 14 (hooks)

Steps 4,5 ──→ Step 17 (CLI scan)
Steps 4,5 ──→ Step 18 (CLI gate)

Steps 6,4,5 ──→ Steps 19,20,21 (new test files)
All ──→ Step 21b (extend existing test files)
All ──→ Step 22 (README)
```

**CRITICAL ORDERING CONSTRAINTS:**
- Step 9 depends on Step 7 (forwards params Step 7 added to AIQualityChecker)
- Steps 9 and 10 MUST be sequential (both modify CodeValidator.__init__)

## Parallelization Opportunities

These step groups can be executed in parallel:
- **Group A:** Steps 4 + 5 + 6 (all independent new modules, depend only on Step 3)
- **Group B:** Steps 7 + 8 (SEC014 rule + fix, after Group A)
- **Group C:** Steps 15 + 15b + 16 (BUGBOT, integration files, knowledge — after Step 7)
- **Group D:** Steps 17 + 18 (CLI extensions, after Group A)
- **Group E:** Steps 19 + 20 + 21 (new test files, after corresponding modules)
- **SEQUENTIAL:** Step 9 → Step 10 → Step 11 → Step 12 → Step 13 (chain, cannot parallelize)
- **Group F:** Step 14 (hooks, after Step 11)
- **LAST:** Step 21b (extend existing tests, after all implementation) + Step 22 (README)
