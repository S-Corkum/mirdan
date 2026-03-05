# Mirdan 1.1.0 Expansion Plan
## Gap Analysis Implementation

**Based on:** Gap Analysis (2026-03-05) + Review corrections
**Target version:** 1.1.0
**Current version:** 1.0.0 (verified: `src/mirdan/__init__.py:8` — CHANGELOG only shows 0.4.0 due to uncommitted local version bump)

---

## Research Notes (Pre-Plan Verification)

### Files Verified
- `mirdan/src/mirdan/core/quality_standards.py:96` — languages list hardcoded: `["typescript", "python", "javascript", "rust", "go", "java"]`. Adding a language requires editing this line.
- `mirdan/src/mirdan/core/quality_standards.py:109` — framework YAMLs dynamically loaded by iterating `frameworks/*.yaml`. Drop a file, it's live. No code changes for new framework standards.
- `mirdan/src/mirdan/core/intent_analyzer.py:120-138` — `FRAMEWORK_PATTERNS` dict. Pattern: `"name": [r"\bregex\b", ...]`. Must add new frameworks here for detection.
- `mirdan/src/mirdan/core/intent_analyzer.py:88-117` — `LANGUAGE_PATTERNS` dict. Must add C# for language detection from prompts.
- `mirdan/src/mirdan/core/language_detector.py:43-93` — `PATTERNS` dict for code-snippet language detection. Must add C# patterns.
- `mirdan/src/mirdan/core/code_validator.py:33-35` — `_BLOCK_COMMENT_LANGUAGES` includes `"c#"`. Need to add `"csharp"` to match the identifier we'll use.
- `mirdan/src/mirdan/core/code_validator.py:347-403` — LC001-LC004 compiled rules structure: `(rule_id, rule_name, regex, severity, message, suggestion)`.
- `mirdan/tests/test_quality_standards.py:43-60` — Test pattern: assert `"principles"` and `"forbidden"` in result, `len >= 5`.
- `mirdan/tests/test_intent_analyzer.py:73+` — Framework detection test pattern: `analyzer.analyze("use [framework]")` → assert `"name" in intent.frameworks`.
- `mirdan/pyproject.toml` — strict mypy, ruff with B/S/SIM/PERF, 85% coverage gate, `uv run pytest`.

### Project Structure (Glob-verified)
- `mirdan/src/mirdan/standards/frameworks/` — 34 YAML files exist, all auto-loaded
- `mirdan/src/mirdan/standards/languages/` — 6 YAML files (python, typescript, javascript, go, java, rust)
- `mirdan/tests/` — 65 test files

### Dependencies Confirmed (pyproject.toml)
- fastmcp>=2.0.0, pyyaml>=6.0, pydantic>=2.0, jinja2>=3.1.0
- No AI client libraries (correct — mirdan is a quality tool, not an AI client)

### Prior Decisions (enyal-verified)
- **Angular EXCLUDED** — Dec 22, 2025 research: "15% decline in new projects"
- **Flask EXCLUDED** — Dec 22, 2025 research: "replaced by FastAPI"
- **CodeValidator false positive bug RESOLVED** — Fixed in 0.0.7 (Feb 13, 2026) via `_build_skip_regions`

### Existing Framework Detection
- Prisma **already detected** at `intent_analyzer.py:127` — just needs YAML
- SQLAlchemy **NOT detected** — needs both YAML and FRAMEWORK_PATTERNS entry
- C# comment handling: `_BLOCK_COMMENT_LANGUAGES` has `"c#"` — need to add `"csharp"` to match new identifier

---

## Explicitly Out of Scope

| Excluded Item | Reason |
|---------------|--------|
| Angular | Prior research decision: 15% decline in new projects |
| Flask | Prior research decision: replaced by FastAPI |
| Docker / K8s / Terraform | Infrastructure/config domain, not code quality |
| PyTorch / vLLM / Ollama | Data-science/inference infrastructure, different profile |
| Data engineering (Spark, Polars, dbt) | Too specialized, separate profile warranted |
| Desktop apps (Electron, Tauri) | Low priority |
| Build tools (Turborepo, Nx) | Niche |
| PHP / Laravel | Low priority for AI-focused tool |
| Ruby / Rails | Low priority |

---

## Phase 0: Housekeeping

*Non-code fixes. No version bump. Execute before Phase 1.*

### Step 0.1 — ~~Verify enhance_prompt modes API~~ ALREADY VERIFIED

**VERIFIED (no action needed):** `enhance_prompt` uses `task_type` parameter (NOT `mode`).
- `enhance_prompt(prompt, task_type='analyze_only')` → replaces `analyze_intent`
- `enhance_prompt(prompt, task_type='plan_validation')` → replaces `validate_plan_quality`
- Standard call output contains `tool_recommendations` and `verification_steps` fields → replaces `suggest_tools` and `get_verification_checklist`

**Grounding:** server.py:249-283 (Read): `async def enhance_prompt(prompt: str, task_type: str = "auto", ...)` and `if task_type == "analyze_only":`

---

### Step 0.2 — Update global CLAUDE.md: remove deprecated tool references

**File:** `/Users/seancorkum/.claude/CLAUDE.md`
**Action:** Read then Edit
**Details:**
- Remove standalone references to `analyze_intent` — replace with `enhance_prompt(prompt, task_type='analyze_only')`
- Remove standalone references to `suggest_tools` — replace with note that `enhance_prompt` output contains `tool_recommendations`
- Remove standalone references to `get_verification_checklist` — replace with note that `enhance_prompt` output contains `verification_steps`
- Update the MCP Framework table row for mirdan to list current 5 tools only: enhance_prompt, validate_code_quality, validate_quick, get_quality_standards, get_quality_trends

**Depends On:** Nothing (Step 0.1 pre-verified exact syntax)
**Verify:** Read the file, confirm no standalone `analyze_intent`, `suggest_tools`, or `get_verification_checklist` calls remain.
**Grounding:** CHANGELOG 0.1.0 line 237-243 (Read): lists removed tools; server.py:249-283 (Read): confirmed `task_type` parameter

---

### Step 0.3 — Update project CLAUDE.md: remove deprecated tool references

**File:** `/Users/seancorkum/projects/ai-assistance/CLAUDE.md`
**Action:** Read then Edit
**Details:** Same changes as Step 0.2, applied to the project-level CLAUDE.md.
**Depends On:** Nothing
**Verify:** Read file, confirm no deprecated tool references remain.
**Grounding:** Same as Step 0.2

---

### Step 0.4 — Update stale enyal memory about C+ grade

**Action:** `enyal_update(entry_id="be5ec43b-15f5-4c57-859e-3d2d1c3239d8", ...)`
**Details:** Already executed during plan creation session. Mark as done.
**Depends On:** Nothing
**Verify:** `enyal_recall("mirdan code review grade")` — should return updated entry
**Grounding:** CHANGELOG 0.0.7-0.4.0 (Read): tracked improvements since Dec 2025

---

## Phase 1: AI Framework Standards Expansion

*Target: 1.1.0. The #1 priority gap — AI coverage 17% → 60%+.*

**Tech debt prevention rules for this phase:**
- Query context7 for EVERY framework before writing its YAML
- Add `# Standards for [Framework] vX.Y - verified [date]` comment at top of each YAML
- FRAMEWORK_PATTERNS use import-statement patterns, not plain name matching
- Compiled rules (LC-style) only if context7 confirms a deprecated API with known migration path
- Minimum content bar: 5 principles, 3 forbidden, 3 patterns (enforced by existing tests)

---

### Step 1.1 — anthropic-sdk.yaml

**File:** `mirdan/src/mirdan/standards/frameworks/anthropic-sdk.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("anthropic")` → `context7.get-library-docs(id, topic="messages tool-use streaming")`
2. Cover: `client.messages.create()`, tool use with `tools` param, streaming with `with client.messages.stream()`, `input_tokens`/`output_tokens` usage tracking
3. Forbidden: deprecated `client.completions.create()` (old API), not caching system prompts for repeated calls, not handling `overloaded_error` (529)
4. Patterns: tool_use, streaming, token_tracking, system_prompt_caching

**Depends On:** Nothing
**Verify:** `QualityStandards().get_for_framework("anthropic-sdk")` returns dict with principles and forbidden
**Grounding:** Dynamic YAML loading at quality_standards.py:109 (Read)

---

### Step 1.2 — openai-sdk.yaml

**File:** `mirdan/src/mirdan/standards/frameworks/openai-sdk.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("openai")` → `context7.get-library-docs(id, topic="chat completions structured-output function-calling")`
2. Cover: `client.chat.completions.create()`, structured outputs with `response_format={"type": "json_schema"}`, function/tool calling, streaming with `stream=True`
3. Forbidden: deprecated `openai.ChatCompletion.create()` (pre-1.0 API), `openai.Completion.create()`, not handling rate limit errors (429), not using async client for concurrent calls
4. Add compiled rule OAI001: detect `openai\.ChatCompletion\.create\s*\(` — deprecated pre-1.0 pattern

**Depends On:** Nothing
**Verify:** `get_for_framework("openai-sdk")` returns standards; grep confirms OAI001 not yet in code_validator.py (add in Step 1.12)
**Grounding:** Dynamic YAML loading (Read); LC001-004 structure (Read) for compiled rule pattern

---

### Step 1.3 — vercel-ai.yaml

**File:** `mirdan/src/mirdan/standards/frameworks/vercel-ai.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("ai")` (Vercel AI SDK) → `context7.get-library-docs(id, topic="streamText generateText tools")`
2. Cover: `streamText()`, `generateText()`, `generateObject()` with Zod schema, tool definition with `tool()`, multi-provider via `createOpenAI`/`createAnthropic`
3. Forbidden: not streaming for long responses in UI contexts, not using `onFinish` callback for logging, mixing provider SDKs without AI SDK abstraction

**Depends On:** Nothing
**Verify:** `get_for_framework("vercel-ai")` returns standards
**Grounding:** Dynamic YAML loading (Read)

---

### Step 1.4 — llamaindex.yaml

**File:** `mirdan/src/mirdan/standards/frameworks/llamaindex.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("llamaindex")` → `context7.get-library-docs(id, topic="query engine retriever pipeline")`
2. Cover: `VectorStoreIndex`, `QueryEngine`, `RetrieverQueryEngine`, `QueryPipeline`, node postprocessors, response synthesizers
3. Forbidden: using deprecated `GPTVectorStoreIndex` (old naming), not setting `similarity_top_k`, not using async query for production load

**Depends On:** Nothing
**Verify:** `get_for_framework("llamaindex")` returns standards
**Grounding:** Dynamic YAML loading (Read)

---

### Step 1.5 — autogen.yaml

**File:** `mirdan/src/mirdan/standards/frameworks/autogen.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("autogen")` → `context7.get-library-docs(id, topic="ConversableAgent GroupChat team")`
2. Cover: `ConversableAgent`, `AssistantAgent`, `UserProxyAgent`, `GroupChat`/`GroupChatManager`, termination conditions, `human_input_mode`
3. Forbidden: no `max_turns` limit (infinite loops), hardcoding LLM config instead of using config lists, not setting `is_termination_msg`

**Depends On:** Nothing
**Verify:** `get_for_framework("autogen")` returns standards
**Grounding:** Dynamic YAML loading (Read)

---

### Step 1.6 — instructor.yaml

**File:** `mirdan/src/mirdan/standards/frameworks/instructor.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("instructor")` → `context7.get-library-docs(id, topic="patch structured output validators")`
2. Cover: `instructor.from_openai()` / `from_anthropic()` client patching, Pydantic model as `response_model`, `Partial[Model]` for streaming, field validators, `max_retries`
3. Forbidden: using `instructor.patch()` (deprecated, use `from_openai()`), not setting `max_retries` for production, using `dict` as response_model instead of Pydantic model

**Depends On:** Nothing
**Verify:** `get_for_framework("instructor")` returns standards
**Grounding:** Dynamic YAML loading (Read)

---

### Step 1.7 — pydantic-ai.yaml

**File:** `mirdan/src/mirdan/standards/frameworks/pydantic-ai.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("pydantic-ai")` → `context7.get-library-docs(id, topic="Agent RunContext tools result-type")`
2. Cover: `Agent(model, result_type=Model)`, `@agent.tool` decorator, `RunContext` for dependency injection, `agent.run()` vs `agent.run_sync()`, `ModelRetry` for retries
3. Forbidden: not typing `result_type` (loses structured output benefit), using global state instead of `RunContext.deps`, not handling `UnexpectedModelBehavior`

**Depends On:** Nothing
**Verify:** `get_for_framework("pydantic-ai")` returns standards
**Grounding:** Dynamic YAML loading (Read)

---

### Step 1.8 — haystack.yaml

**File:** `mirdan/src/mirdan/standards/frameworks/haystack.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("haystack")` → `context7.get-library-docs(id, topic="Pipeline Component DocumentStore RAG")`
2. Cover: `Pipeline` with `add_component()` and `connect()`, `@component` decorator, `DocumentStore`, `EmbeddingRetriever`, `PromptBuilder`
3. Forbidden: not using `@component` decorator for custom components, not defining `run()` with typed `Output` dataclass, mixing Haystack 1.x and 2.x APIs

**Depends On:** Nothing
**Verify:** `get_for_framework("haystack")` returns standards
**Grounding:** Dynamic YAML loading (Read)

---

### Step 1.9 — openai-agents.yaml

**File:** `mirdan/src/mirdan/standards/frameworks/openai-agents.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("openai-agents")` → `context7.get-library-docs(id, topic="Agent Runner tool handoff")`
2. Cover: `Agent(name, instructions, tools)`, `Runner.run()`, `@function_tool` decorator, `handoff()` for multi-agent, `RunContext`, guardrails
3. Forbidden: not defining output type, not using guardrails for untrusted input, sharing mutable state across agents without `RunContext`

**Depends On:** Nothing
**Verify:** `get_for_framework("openai-agents")` returns standards
**Grounding:** Dynamic YAML loading (Read)

---

### Step 1.10 — mcp-server.yaml

**File:** `mirdan/src/mirdan/standards/frameworks/mcp-server.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("fastmcp")` → `context7.get-library-docs(id, topic="tool resource server lifespan")`
2. Cover: `@mcp.tool()` decorator, Pydantic models for tool inputs, `@mcp.resource()` for read-only data, `lifespan` context manager for startup/shutdown, `Context` parameter for progress reporting
3. Forbidden: tool functions without docstrings (LLMs use docstrings for routing), not using Pydantic for input validation, global mutable state outside lifespan, tools that modify resources (use tools for mutation, resources for reads)
4. Patterns: tool_definition, resource_pattern, lifespan_management, progress_reporting

**Depends On:** Nothing
**Verify:** `get_for_framework("mcp-server")` returns standards
**Grounding:** Dynamic YAML loading (Read); we USE fastmcp in this project (pyproject.toml:21 confirmed)

---

### Step 1.11 — Update IntentAnalyzer.FRAMEWORK_PATTERNS for all 10 new frameworks

**File:** `mirdan/src/mirdan/core/intent_analyzer.py`
**Action:** Read then Edit
**Details:** Add to `FRAMEWORK_PATTERNS` dict (currently ends after `"qdrant"` at line 137). Add after the existing entries:
```python
"anthropic-sdk": [r"\bimport\s+anthropic\b", r"from\s+anthropic\s+import", r"\bclient\.messages\.create\b"],
"openai-sdk": [r"\bimport\s+openai\b", r"from\s+openai\s+import", r"\bclient\.chat\.completions\b"],
"vercel-ai": [r"from\s+['\"]ai['\"]", r"\bstreamText\b", r"\bgenerateText\b", r"\bgenerateObject\b"],
"llamaindex": [r"\bllama[_-]?index\b", r"\bVectorStoreIndex\b", r"\bQueryEngine\b", r"\bLlamaIndex\b"],
"autogen": [r"\bimport\s+autogen\b", r"\bConversableAgent\b", r"\bGroupChatManager\b"],
"instructor": [r"\bimport\s+instructor\b", r"\binstructor\.from_\w+\b", r"\bresponse_model\b"],
"pydantic-ai": [r"\bfrom\s+pydantic_ai\s+import\b", r"\bpydantic[_\-]ai\b", r"\bRunContext\b"],
"haystack": [r"\bimport\s+haystack\b", r"from\s+haystack\s+import", r"\bPipeline\b.*\bComponent\b"],
"openai-agents": [r"\bfrom\s+openai_agents\s+import\b", r"@function_tool\b", r"\bRunner\.run\b"],
"mcp-server": [r"from\s+fastmcp\s+import", r"@mcp\.tool", r"@mcp\.resource", r"\bmcp\.run\b"],
```

**Pattern notes (3 bugs fixed):**
- `pydantic.ai` → `pydantic[_\-]ai` (escaped dot was matching any char)
- `@mcp\.tool` — removed `\b` prefix: `@` is not a word char so `\b` before it never matches
- `from agents import` removed — matches any `agents.py` module; replaced with `Runner.run` which is unique to openai-agents

**Depends On:** Steps 1.1-1.10 (framework names must be consistent)
**Verify:** Read the file, confirm all 10 entries are present with correct key names matching YAML filenames
**Grounding:** intent_analyzer.py:120-138 structure (Read)

---

### Step 1.12 — Add OAI001 compiled rule for deprecated OpenAI API

**File:** `mirdan/src/mirdan/core/code_validator.py`
**Action:** Read then Edit
**Details:** Add after LC004 (line 383), inside the `"python"` language rules list:
```python
(
    "OAI001",
    "deprecated-openai-completion",
    r"\bopenai\.ChatCompletion\.create\s*\(",
    "error",
    "openai.ChatCompletion.create() is removed in openai>=1.0 - use client.chat.completions.create()",
    "Migrate to: client = openai.OpenAI(); client.chat.completions.create(model=..., messages=...)",
),
```
Also add Anthropic deprecated API rule:
```python
(
    "ANT001",
    "deprecated-anthropic-completion",
    r"\bclient\.completions\.create\s*\(",
    "error",
    "client.completions.create() is removed in anthropic>=0.20 - use client.messages.create()",
    "Migrate to: client.messages.create(model=..., messages=[{\"role\": \"user\", \"content\": ...}])",
),
```

**Depends On:** Step 1.2 (openai-sdk.yaml confirms the deprecated pattern), Step 1.1 (anthropic-sdk.yaml confirms)
**Verify:** Read file, confirm OAI001 and ANT001 are present after LC004
**Grounding:** LC001-LC004 structure at code_validator.py:347-384 (Read)

---

### Step 1.13 — Tests for Phase 1

**Files:**
- `mirdan/tests/test_quality_standards.py` (Edit)
- `mirdan/tests/test_intent_analyzer.py` (Edit)
- `mirdan/tests/test_code_validator.py` (Edit)

**Action:** Read then Edit each
**Details:**

*test_quality_standards.py* — Add test class `TestAIFrameworkStandards` with one test per framework:
```python
@pytest.mark.parametrize("framework", [
    "anthropic-sdk", "openai-sdk", "vercel-ai", "llamaindex",
    "autogen", "instructor", "pydantic-ai", "haystack",
    "openai-agents", "mcp-server",
])
def test_ai_framework_has_required_keys(self, framework: str) -> None:
    standards = QualityStandards()
    result = standards.get_for_framework(framework)
    assert "principles" in result
    assert "forbidden" in result
    assert len(result["principles"]) >= 5
    assert len(result["forbidden"]) >= 3
```

*test_intent_analyzer.py* — Add to `TestFrameworkDetection`:
- `test_detects_anthropic_sdk`: prompt "implement tool use with anthropic sdk" → "anthropic-sdk" in frameworks
- `test_detects_openai_sdk`: prompt "use openai client.chat.completions" → "openai-sdk" in frameworks
- `test_detects_mcp_server`: prompt "create a fastmcp tool with @mcp.tool" → "mcp-server" in frameworks
- (representative subset of 3-4, not all 10 to avoid test bloat)

*test_code_validator.py* — Add OAI001 and ANT001 rule detection tests following existing LC-rule test pattern.

**Depends On:** Steps 1.1-1.12
**Verify:** `uv run pytest tests/test_quality_standards.py tests/test_intent_analyzer.py tests/test_code_validator.py` passes
**Grounding:** Test patterns from test_quality_standards.py:43-60 (Read)

---

## Phase 2: C# Language + ASP.NET Core

*Target: 1.1.0. Most technically involved phase — code changes required in 4 Python files.*

---

### Step 2.1 — csharp.yaml

**File:** `mirdan/src/mirdan/standards/languages/csharp.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("csharp")` or use general knowledge — `.NET 9 / C# 13`
2. Cover: nullable reference types (enable in csproj), primary constructors (C# 12+), records for immutable data, `async Task` (never `async void` except event handlers), `IDisposable`/`IAsyncDisposable`, LINQ with deferred execution awareness, pattern matching with `switch` expressions
3. Forbidden: `async void` methods (swallows exceptions), `Thread.Sleep` (use `await Task.Delay`), `catch (Exception e)` without rethrowing or logging, mutable public fields, `var` for non-obvious types
4. Patterns: async_pattern, record_creation, pattern_matching, using_declaration

**Depends On:** Nothing
**Verify:** File exists at correct path
**Grounding:** Languages directory structure from Glob (confirmed 6 language YAMLs)

---

### Step 2.2 — Add "csharp" to quality_standards.py languages list

**File:** `mirdan/src/mirdan/core/quality_standards.py`
**Action:** Read then Edit
**Details:** At line 96, change:
```python
languages = ["typescript", "python", "javascript", "rust", "go", "java"]
```
to:
```python
languages = ["typescript", "python", "javascript", "rust", "go", "java", "csharp"]
```

**Depends On:** Step 2.1 (csharp.yaml must exist)
**Verify:** Read file at line 96, confirm "csharp" is in the list
**Grounding:** quality_standards.py:96 (Read)

---

### Step 2.3 — Add C# detection to LanguageDetector.PATTERNS

**File:** `mirdan/src/mirdan/core/language_detector.py`
**Action:** Read then Edit
**Details:** Add after `"java"` entry (line 93):
```python
"csharp": [
    (r"\bnamespace\s+\w+", 5),       # namespace keyword — strong C# signal
    (r"\busing\s+System\b", 4),       # .NET core namespace
    (r"\bpublic\s+class\s+\w+\s*:", 3),  # class with inheritance (colon, not extends)
    (r"\basync\s+Task\b", 3),         # async Task return type
    (r"\[HttpGet\]|\[HttpPost\]|\[ApiController\]", 4),  # ASP.NET attributes
    (r"\bvar\s+\w+\s*=\s*new\b", 2),  # var + new instantiation
],
```
Note: `public class` overlaps with Java — use `public class X :` (colon for inheritance) which is C#-specific; also `namespace` is essentially unique to C#.

**Depends On:** Nothing (independent of YAML)
**Verify:** Read file, confirm C# entry added
**Grounding:** language_detector.py:43-93 (Read); C# syntax knowledge verified by `namespace` keyword uniqueness

---

### Step 2.4 — Add C# to IntentAnalyzer.LANGUAGE_PATTERNS and BLOCK_COMMENT fix

**File:** `mirdan/src/mirdan/core/intent_analyzer.py`
**Action:** Read then Edit
**Details:**

*Part A* — Add to `LANGUAGE_PATTERNS` (after `"java"` entry at line 116):
```python
"csharp": [
    (r"\bc#\b", 5),
    (r"\b\.net\b", 4),
    (r"\bcsharp\b", 5),
    (r"\.cs$", 4),
    (r"\basp\.net\b", 3),
    (r"\blazor\b", 3),
],
```

*Part B* — NO ACTION NEEDED. `r"\bangular\b"` in `LANGUAGE_PATTERNS["typescript"]` (line 94, weight 3) is correct — Angular projects use TypeScript, so detecting Angular as TypeScript is the right behavior. Do not remove or reduce this weight.

No `_BLOCK_COMMENT_LANGUAGES` change needed in intent_analyzer.py — that constant lives in code_validator.py. See Step 2.5.

**Depends On:** Nothing
**Verify:** Read file, confirm csharp entry in LANGUAGE_PATTERNS
**Grounding:** intent_analyzer.py:88-117 (Read)

---

### Step 2.5 — Add "csharp" to _BLOCK_COMMENT_LANGUAGES in code_validator.py

**File:** `mirdan/src/mirdan/core/code_validator.py`
**Action:** Read then Edit
**Details:** At line 33-35, add "csharp" to the frozenset:
```python
_BLOCK_COMMENT_LANGUAGES = frozenset(
    {"typescript", "javascript", "java", "go", "rust", "c", "cpp", "c++", "c#", "kotlin", "swift", "csharp"}
)
```
This ensures block comment skip regions work correctly when language_detector returns "csharp".

**Depends On:** Nothing
**Verify:** Read lines 33-35, confirm "csharp" is present
**Grounding:** code_validator.py:33-35 (Read); "c#" already present, adding "csharp" identifier form

---

### Step 2.6 — aspnetcore.yaml

**File:** `mirdan/src/mirdan/standards/frameworks/aspnetcore.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("aspnet-core")` → `context7.get-library-docs(id, topic="minimal-api routing dependency-injection middleware")`
2. Cover: Minimal API with `app.MapGet/MapPost`, `IServiceCollection` DI pattern, `[ApiController]` + `ControllerBase`, `IActionResult` return types, middleware pipeline with `app.Use()`, `ILogger<T>` injection
3. Forbidden: not using `ILogger` (use `Console.WriteLine`), service locator anti-pattern (`IServiceProvider.GetService` in constructors), blocking `.Result` or `.Wait()` on async Tasks, not using `cancellationToken` parameters in async methods

**Depends On:** Nothing
**Verify:** File exists
**Grounding:** Dynamic YAML loading (Read)

---

### Step 2.7 — Add aspnetcore to FRAMEWORK_PATTERNS

**File:** `mirdan/src/mirdan/core/intent_analyzer.py`
**Action:** Edit (batch with Step 1.11 if not yet committed, or separate edit)
**Details:** Add to `FRAMEWORK_PATTERNS`:
```python
"aspnetcore": [r"\basp\.net\s*core\b", r"\bIActionResult\b", r"\bControllerBase\b", r"\bapp\.MapGet\b", r"\b\[ApiController\]\b"],
```

**Depends On:** Step 2.6
**Verify:** Read FRAMEWORK_PATTERNS, confirm aspnetcore entry
**Grounding:** intent_analyzer.py:120-138 (Read)

---

### Step 2.8 — Tests for Phase 2

**Files:**
- `mirdan/tests/test_language_detector.py` (Edit)
- `mirdan/tests/test_quality_standards.py` (Edit)
- `mirdan/tests/test_intent_analyzer.py` (Edit)

**Action:** Read then Edit each
**Details:**

*test_language_detector.py* — Add:
```python
def test_detects_csharp_from_namespace(self) -> None:
    detector = LanguageDetector()
    lang, confidence = detector.detect("namespace MyApp.Services { public class UserService : IUserService { } }")
    assert lang == "csharp"
    assert confidence in ("high", "medium")

def test_detects_csharp_from_async_task(self) -> None:
    detector = LanguageDetector()
    lang, _ = detector.detect("public async Task<IActionResult> GetUser(int id) { return Ok(user); }")
    assert lang == "csharp"
```

*test_quality_standards.py* — Add:
```python
def test_get_for_language_csharp(self) -> None:
    standards = QualityStandards()
    result = standards.get_for_language("csharp")
    assert "principles" in result
    assert len(result["principles"]) >= 5

def test_get_for_framework_aspnetcore(self) -> None:
    standards = QualityStandards()
    result = standards.get_for_framework("aspnetcore")
    assert "principles" in result
    assert "forbidden" in result
```

*test_intent_analyzer.py* — Add:
```python
def test_detects_csharp_language(self) -> None:
    intent = analyzer.analyze("create a C# class with async Task methods")
    assert intent.primary_language == "csharp"
```

**Depends On:** Steps 2.1-2.7
**Verify:** `uv run pytest tests/test_language_detector.py tests/test_quality_standards.py tests/test_intent_analyzer.py` passes
**Grounding:** Test patterns from existing test files (Read)

---

## Phase 3: Missing Critical ORMs

---

### Step 3.1 — prisma.yaml

**File:** `mirdan/src/mirdan/standards/frameworks/prisma.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("prisma")` → `context7.get-library-docs(id, topic="client schema migrations query")`
2. Cover: `prisma.schema` model definitions, `prisma.$transaction()` for multi-operation consistency, typed `Prisma.UserWhereInput` for queries, `prisma.$disconnect()` in cleanup, relation includes with `include`
3. Forbidden: `prisma.$queryRaw` without `Prisma.sql` template tag (SQL injection), not awaiting prisma client methods, creating `PrismaClient` instance per request (use singleton), using `upsert` without careful conflict handling

**Depends On:** Nothing (detection already exists at intent_analyzer.py:127)
**Verify:** `QualityStandards().get_for_framework("prisma")` returns standards
**Grounding:** intent_analyzer.py:127 (Read) confirms detection exists; dynamic YAML loading (Read)

---

### Step 3.2 — sqlalchemy.yaml

**File:** `mirdan/src/mirdan/standards/frameworks/sqlalchemy.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("sqlalchemy")` → `context7.get-library-docs(id, topic="2.0-style session select async")`
2. Cover: SQLAlchemy 2.0 style: `session.execute(select(User))`, `async_session` with `AsyncSession`, mapped classes with `DeclarativeBase`, `Mapped[T]` type annotations, `relationship()` with `lazy="select"` explicit
3. Forbidden: `session.query()` (1.x style — use `session.execute(select(...))`), not using `session.scalar()` for single result, `engine.execute()` (removed in 2.0), not closing sessions (use context managers)
4. Add compiled rule SA001: detect `session\.query\s*\(` — deprecated 1.x pattern

**Depends On:** Nothing
**Verify:** File exists
**Grounding:** Dynamic YAML loading (Read)

---

### Step 3.3 — Add sqlalchemy to FRAMEWORK_PATTERNS + SA001 compiled rule

**File (A):** `mirdan/src/mirdan/core/intent_analyzer.py`
**Action:** Edit
**Details:** Add to `FRAMEWORK_PATTERNS`:
```python
"sqlalchemy": [r"\bsqlalchemy\b", r"from\s+sqlalchemy\s+import", r"\bDeclarativeBase\b", r"\bAsyncSession\b"],
```

**File (B):** `mirdan/src/mirdan/core/code_validator.py`
**Action:** Edit
**Details:** Add SA001 rule after ANT001 (from Step 1.12):
```python
(
    "SA001",
    "sqlalchemy-legacy-query",
    r"\bsession\.query\s*\(",
    "warning",
    "session.query() is the SQLAlchemy 1.x style - use session.execute(select(...)) for 2.0+",
    "Replace with: result = await session.execute(select(User).where(User.id == user_id))",
),
```

**Depends On:** Step 3.2
**Verify:** Read both files, confirm additions
**Grounding:** intent_analyzer.py structure (Read); code_validator.py compiled rule structure (Read)

---

### Step 3.4 — Tests for Phase 3

**Files:** `mirdan/tests/test_quality_standards.py`, `mirdan/tests/test_intent_analyzer.py`, `mirdan/tests/test_code_validator.py`
**Action:** Read then Edit each
**Details:**
- test_quality_standards.py: `test_get_for_framework_prisma` and `test_get_for_framework_sqlalchemy`
- test_intent_analyzer.py: `test_detects_sqlalchemy` (prompt: "use SQLAlchemy AsyncSession for database queries")
- test_code_validator.py: SA001 detection test

**Depends On:** Steps 3.1-3.3
**Verify:** `uv run pytest tests/` passes
**Grounding:** Test patterns (Read)

---

## Phase 4: Rust Web + Versioned Standards

---

### Step 4.1 — axum.yaml

**File:** `mirdan/src/mirdan/standards/frameworks/axum.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("axum")` → `context7.get-library-docs(id, topic="routing state extractors middleware")`
2. Cover: `Router::new().route()`, `State<T>` extractor, `axum::extract::*` (Path, Query, Json, Extension), Tower middleware with `ServiceBuilder`, `#[axum::debug_handler]` for compile-time error messages, `IntoResponse` for custom responses
3. Forbidden: blocking operations inside handlers (use `tokio::task::spawn_blocking`), not using `State` for shared data (use `Extension` only for third-party middleware), not using `axum::debug_handler` during development, panicking in handlers

**Depends On:** Nothing
**Verify:** `get_for_framework("axum")` returns standards
**Grounding:** Dynamic YAML loading (Read)

---

### Step 4.2 — Add axum to FRAMEWORK_PATTERNS

**File:** `mirdan/src/mirdan/core/intent_analyzer.py`
**Action:** Edit
**Details:** Add:
```python
"axum": [r"\baxum\b", r"\bRouter::new\b", r"from\s+axum\s+import", r"\buse\s+axum::", r"\bIntoResponse\b"],
```

**Depends On:** Step 4.1
**Verify:** Read file
**Grounding:** intent_analyzer.py FRAMEWORK_PATTERNS (Read)

---

### Step 4.3 — react-19.yaml (versioned override)

**File:** `mirdan/src/mirdan/standards/frameworks/react-19.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("react")` → `context7.get-library-docs(id, topic="react 19 actions server-components compiler")`
2. This is a VERSIONED OVERRIDE — contains ONLY React 19-specific changes that override the base react.yaml.
3. Cover differences: React Compiler (automatic memoization, no manual `useMemo`/`useCallback` needed), `use()` hook for Promises and Context, Server Actions with `"use server"` directive, `useOptimistic()`, `useFormStatus()`, `useActionState()` (replaces `useFormState`)
4. Forbidden overrides: manual `useMemo`/`useCallback` wrapping simple values (compiler handles this), `useFormState` (renamed to `useActionState`)
5. File structure: same YAML keys as base but only changed/new values — base.update(versioned) merge strategy means these override the base

**Depends On:** Nothing (version detection uses existing mechanism)
**Verify:** When React 19 is in package.json, `QualityStandards(project_dir=dir).get_for_framework("react")` merges react.yaml with react-19.yaml
**Grounding:** quality_standards.py:193-217 versioned loading mechanism (Read)

---

### Step 4.4 — nextjs-15.yaml (versioned override)

**File:** `mirdan/src/mirdan/standards/frameworks/nextjs-15.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("next")` → `context7.get-library-docs(id, topic="next 15 caching async-params turbopack")`
2. VERSIONED OVERRIDE — Next.js 15-specific changes only.
3. Cover differences: `params` and `searchParams` are now async (must `await params`), `next/after` for post-response work, `create-next-app` defaults changed (no `src/` dir), Turbopack stable for `dev`, improved `@next/codemod` migration tool
4. Forbidden overrides: synchronously accessing `params` without `await` (breaking change in 15), using `unstable_after` (now stable as `after`)

**Depends On:** Nothing
**Verify:** Versioned loading mechanism serves this when Next.js 15.x in package.json
**Grounding:** quality_standards.py versioned loading (Read)

---

### Step 4.5 — Tests for Phase 4

**Files:** `mirdan/tests/test_quality_standards.py`, `mirdan/tests/test_intent_analyzer.py`, `mirdan/tests/test_version_aware.py`
**Action:** Read then Edit each
**Details:**

*test_version_aware.py* — First READ the file. Then:

⚠️ **Conflict:** `test_get_for_framework_with_version_no_versioned_file` (lines 117-131) currently tests that React 19 falls back to base react when `react-19.yaml` doesn't exist. After Step 4.2 adds `react-19.yaml`, this test will pass for the wrong reason (versioned file now exists, behavior changed). Fix this:
1. Change the test's framework to `vue` with version `4.0.0` — Vue has no versioned file, so the fallback behavior is correctly tested without react.
2. Add `test_react19_versioned_standards_override` that calls `get_for_framework("react", version="19.0.0")` and asserts a react-19-specific key is present in the result, verifying the merge actually ran.

Then add:
- Test that react-19.yaml overrides specific principles from react.yaml
- Test that nextjs-15.yaml overrides specific principles from nextjs.yaml

*test_quality_standards.py* — Add `test_get_for_framework_axum`

*test_intent_analyzer.py* — Add `test_detects_axum` (prompt: "create axum Router with State extractor")

**Depends On:** Steps 4.1-4.4
**Verify:** `uv run pytest tests/test_version_aware.py` passes
**Grounding:** test_version_aware.py exists (Glob confirmed)

---

## Phase 5: Structural Gaps — State Management, API Design, Observability

*All additions in this phase are YAML-only (no language model changes). Batch FRAMEWORK_PATTERNS update at end.*

---

### Step 5.1 — zustand.yaml

**File:** `mirdan/src/mirdan/standards/frameworks/zustand.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("zustand")` → `context7.get-library-docs(id, topic="store slices middleware persist")`
2. Cover: Zustand 5 `createStore` / `create` pattern, slice pattern for large stores, `immer` middleware for immutable updates, `devtools` middleware, `persist` middleware with storage config, `subscribeWithSelector` for fine-grained subscriptions
3. Forbidden: mutating state directly (must use `set(state => ({...}))`), not using `shallow` comparator for object selectors, subscribing to the entire store when only one field is needed

**Depends On:** Nothing
**Verify:** `get_for_framework("zustand")` returns standards
**Grounding:** Dynamic YAML loading (Read)

---

### Step 5.2 — tanstack-query.yaml

**File:** `mirdan/src/mirdan/standards/frameworks/tanstack-query.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("tanstack-query")` → `context7.get-library-docs(id, topic="useQuery useMutation queryClient staleTime")`
2. Cover: TanStack Query v5: `useQuery({ queryKey, queryFn })`, `useMutation({ mutationFn, onSuccess })`, `QueryClient` with `defaultOptions`, `staleTime` vs `gcTime` (renamed from `cacheTime`), `queryKey` as array for cache invalidation
3. Forbidden: `cacheTime` (renamed to `gcTime` in v5), not providing `queryKey` as array, not invalidating queries after mutations, fetching in `useEffect` instead of `useQuery`

**Depends On:** Nothing
**Verify:** `get_for_framework("tanstack-query")` returns standards
**Grounding:** Dynamic YAML loading (Read)

---

### Step 5.3 — graphql.yaml

**File:** `mirdan/src/mirdan/standards/frameworks/graphql.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("graphql")` → `context7.get-library-docs(id, topic="schema resolvers DataLoader subscriptions")`
2. Cover: schema-first vs code-first patterns, DataLoader for N+1 prevention, input validation in resolvers, `context` parameter for auth, subscription patterns, error handling with `GraphQLError`
3. Forbidden: resolving without DataLoader for batching, allowing unbounded query depth (set max depth), exposing internal error messages in production, not authenticating subscriptions

**Depends On:** Nothing
**Verify:** `get_for_framework("graphql")` returns standards
**Grounding:** Dynamic YAML loading (Read)

---

### Step 5.4 — opentelemetry.yaml

**File:** `mirdan/src/mirdan/standards/frameworks/opentelemetry.yaml` (NEW)
**Action:** Write
**Details:**
1. First: `context7.resolve-library-id("opentelemetry")` → `context7.get-library-docs(id, topic="tracing spans context-propagation semantic-conventions")`
2. Cover: `tracer.start_as_current_span()` context manager, semantic conventions for span attributes (`SpanAttributes.HTTP_METHOD`), W3C TraceContext propagation, baggage API, resource attributes for service identification, `BatchSpanProcessor` vs `SimpleSpanProcessor`
3. Forbidden: manual span context propagation (use context managers), `SimpleSpanProcessor` in production (use `BatchSpanProcessor`), not setting `service.name` resource attribute, custom attribute names that duplicate semantic conventions

**Depends On:** Nothing
**Verify:** `get_for_framework("opentelemetry")` returns standards
**Grounding:** Dynamic YAML loading (Read)

---

### Step 5.5 — Batch FRAMEWORK_PATTERNS update for Phase 5 frameworks

**File:** `mirdan/src/mirdan/core/intent_analyzer.py`
**Action:** Edit
**Details:** Add all Phase 5 frameworks to `FRAMEWORK_PATTERNS`:
```python
"zustand": [r"from\s+['\"]zustand['\"]", r"\bcreateStore\b", r"\bcreate\s*\(\s*\(set\b"],
"tanstack-query": [r"@tanstack/react-query", r"\buseQuery\b", r"\bqueryKey\b", r"\bQueryClient\b"],
"graphql": [r"\bgraphql\b", r"\btype\s+Query\s*\{", r"\btype\s+Mutation\s*\{", r"\bgql\b"],
"opentelemetry": [r"\bopentelemetry\b", r"from\s+opentelemetry", r"\bTracerProvider\b", r"\btracer\.start"],
```

**Depends On:** Steps 5.1-5.4
**Verify:** Read FRAMEWORK_PATTERNS, confirm all 4 entries
**Grounding:** intent_analyzer.py:120-138 (Read)

---

### Step 5.6 — Tests for Phase 5

**Files:** `mirdan/tests/test_quality_standards.py`, `mirdan/tests/test_intent_analyzer.py`
**Action:** Read then Edit each
**Details:**

*test_quality_standards.py* — Parametrized test for all 4 new frameworks:
```python
@pytest.mark.parametrize("framework", [
    "zustand", "tanstack-query", "graphql", "opentelemetry"
])
def test_structural_framework_has_required_keys(self, framework: str) -> None:
    ...
```

*test_intent_analyzer.py* — Representative detection tests:
- `test_detects_graphql`: prompt "define GraphQL schema with type Query"
- `test_detects_opentelemetry`: prompt "add opentelemetry tracing with TracerProvider"

**Depends On:** Steps 5.1-5.5
**Verify:** `uv run pytest tests/` passes
**Grounding:** Test patterns (Read)

---

## Final Steps

### Step F.1 — Full test suite

**Action:** Bash
**Details:** `uv run pytest --tb=short`
**Depends On:** All previous steps
**Verify:** All tests pass, coverage ≥ 85%
**Grounding:** pyproject.toml:99 `fail_under = 85`

---

### Step F.2 — Linting

**Action:** Bash
**Details:** `uv run ruff check src/ tests/` and `uv run mypy src/`
**Depends On:** All code edits
**Verify:** Zero errors
**Grounding:** pyproject.toml ruff/mypy config (Read)

---

### Step F.3 — Bump version in __init__.py

**File:** `mirdan/src/mirdan/__init__.py`
**Action:** Read then Edit
**Details:** Update `__version__` from `"1.0.0"` to `"1.1.0"`:
```python
__version__ = "1.1.0"
```

**Depends On:** Step F.1 (tests passing), Step F.2 (linting clean)
**Verify:** Read file, confirm `__version__ = "1.1.0"`
**Grounding:** `__init__.py:8` (Read) — current value is `"1.0.0"`

---

### Step F.4 — Update CHANGELOG.md

**File:** `mirdan/CHANGELOG.md`
**Action:** Read then Edit
**Details:** Add `## [1.1.0] - 2026-XX-XX` section at top with:
- Added: 10 AI framework standards (list all)
- Added: C# language standards
- Added: ASP.NET Core framework standards
- Added: Prisma, SQLAlchemy ORM standards
- Added: Axum (Rust web) framework standards
- Added: Versioned standards (react-19, nextjs-15)
- Added: State management (Zustand, TanStack Query)
- Added: API design (GraphQL) standards
- Added: Observability (OpenTelemetry) standards
- Added: OAI001, ANT001, SA001 compiled validation rules
- Changed: IntentAnalyzer detects 18 new frameworks
- Changed: LanguageDetector detects C#

**Depends On:** Step F.1 (tests passing), Step F.3 (version bumped)
**Verify:** Read CHANGELOG, confirm new section at top
**Grounding:** CHANGELOG.md format (Read)

---

### Step F.5 — Store plan completion in enyal

**Action:** `enyal_remember` or `enyal_update` on the plan entry
**Details:** Update entry `abb966d2-c30e-46d8-86a8-0833ead7018c` with completion status
**Depends On:** All previous steps
**Grounding:** enyal plan entry created during plan session

---

## Excluded Items Summary

The following were considered and **intentionally excluded** from this plan:

| Item | Decision |
|------|---------|
| Angular | Prior research: 15% decline in new projects (Dec 2025) |
| Flask | Prior research: replaced by FastAPI for new projects |
| Docker / K8s / Terraform | Infrastructure/config domain — different value proposition from code quality |
| PyTorch / vLLM / Ollama | Data science/inference infrastructure — belongs in separate `data-science` quality profile |
| Spark / Polars / dbt / DuckDB | Too specialized — warrants dedicated `data-engineering` quality profile |
| PHP / Laravel | Low priority for AI-focused developer tooling |
| Ruby / Rails | Low priority |
| Desktop (Electron, Tauri) | Niche use case |
| Build tools (Turborepo, Nx) | Low value for code quality guidance |

---

## Step Count Summary

| Phase | Steps | Type |
|-------|-------|------|
| Phase 0 | 4 | Config/docs fixes |
| Phase 1 | 13 | 10 YAMLs + detection + compiled rules + tests |
| Phase 2 | 8 | 2 YAMLs + 4 Python edits + tests |
| Phase 3 | 4 | 2 YAMLs + detection + compiled rule + tests |
| Phase 4 | 5 | 3 YAMLs + detection + tests |
| Phase 5 | 6 | 4 YAMLs + detection + tests |
| Final | 4 | Tests, lint, CHANGELOG, enyal |
| **Total** | **44** | |
