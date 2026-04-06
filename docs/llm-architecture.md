# Local Intelligence Layer — Architecture

For contributors who want to understand or modify the LLM integration code.

## Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    MIRDAN 2.0.0                                   │
│                                                                   │
│  MCP TOOL SURFACE (unchanged):                                    │
│    enhance_prompt | validate_code_quality | validate_quick        │
│    get_quality_standards | get_quality_trends                     │
│    scan_conventions | scan_dependencies                           │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  HTTP SIDECAR (localhost:auto-port)                        │   │
│  │  Endpoints: /triage, /check, /health                       │   │
│  │  For hook scripts — <5ms latency (reuses warm model)       │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  LOCAL INTELLIGENCE LAYER                                  │   │
│  │                                                            │   │
│  │  LLMManager ── ModelSelector ── HealthMonitor              │   │
│  │       │         (dynamic)        (state machine)           │   │
│  │       │                                                    │   │
│  │  HardwareDetector ── ModelRegistry                         │   │
│  │  (arch, RAM, GPU)    (known models + discovery)            │   │
│  │       │                                                    │   │
│  │  LocalLLMProtocol (abstraction)                            │   │
│  │       ├── OllamaBackend (HTTP to daemon)                   │   │
│  │       └── LlamaCppBackend (in-process, -200MB)             │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  EXISTING CORE (unchanged):                                       │
│    CodeValidator | IntentAnalyzer | PromptComposer | ...          │
└──────────────────────────────────────────────────────────────────┘
```

## Module map

```
src/mirdan/llm/
├── __init__.py           # Package exports: LocalLLMProtocol, InMemoryBackend, LLMManager
├── protocol.py           # LocalLLMProtocol (typing.Protocol) + InMemoryBackend test double
├── ollama.py             # Ollama HTTP backend (httpx)
├── llamacpp.py           # llama-cpp-python in-process backend (anyio threads)
├── registry.py           # ModelRegistry (known models + discovery) + ModelSelector
├── health.py             # HardwareDetector + HealthMonitor (state machine)
├── manager.py            # LLMManager facade + HTTP sidecar lifecycle
├── sidecar.py            # Starlette HTTP server for hook integration
├── session_bridge.py     # File-based hook↔MCP coordination
├── metrics.py            # Token savings tracking (JSONL)
├── training_collector.py # Fine-tuning data collection (JSONL)
└── prompts/
    ├── __init__.py
    ├── triage.py         # Task classification prompts + few-shot examples
    ├── checks.py         # Tool output analysis prompts
    ├── validation.py     # Smart validation prompts (injection-mitigated)
    ├── optimization.py   # Per-model prompt optimization prompts
    └── research.py       # Agentic tool selection prompts

src/mirdan/core/
├── triage.py             # TriageEngine — rules pre-filter + LLM classification
├── check_runner.py       # CheckRunner — subprocess orchestration + LLM analysis
├── smart_validator.py    # SmartValidator — FP filtering, root causes, fixes
├── prompt_optimizer.py   # PromptOptimizer — BRAIN model prompt crafting
└── research_agent.py     # ResearchAgent — agentic MCP tool loop
```

## Key patterns

### LocalLLMProtocol

All backends implement this Protocol: `generate`, `generate_structured`, `chat`, `chat_with_tools`, `is_available`, `list_models`, `health`, `close`. `InMemoryBackend` is the test double.

### Dynamic model selection

`ModelRegistry` scans Ollama tags + GGUF directory to find installed models. `ModelSelector` picks the highest-quality model that fits in available RAM minus a 2GB safety buffer. Selection happens on every `generate()` call using current available memory.

### Health state machine

```
STARTING → WARMING_UP → AVAILABLE
                       → DEGRADED (warmup failure)
                       → UNAVAILABLE (backend down)
```

Background warmup via `asyncio.Task`. MCP server responds immediately during WARMING_UP — `generate()` returns None until AVAILABLE.

### HTTP sidecar

Lightweight Starlette server on `localhost:auto-port`. Port written to `.mirdan/sidecar.port`. Hook scripts curl to this endpoint instead of cold-starting a CLI process. Endpoints call TriageEngine and CheckRunner directly.

### Session bridge

File-based coordination: hooks write JSON to `.mirdan/sessions/{session_id}/`, MCP server reads it. Prevents re-triaging when the hook already ran. Session ID from `CLAUDE_SESSION_ID` or `CURSOR_SESSION_ID` env vars.

### Graceful degradation

Every LLM consumer checks for None:
- `LLMManager.create_if_enabled()` → None if disabled
- `generate()` → None if not AVAILABLE or no model fits
- `TriageEngine.classify()` → None if no LLM
- `SmartValidator.analyze()` → None if no LLM or no violations
- Rules-only mode is always available as the baseline

## Data flow

### Hook path (Claude Code)

```
1. User types prompt
2. UserPromptSubmit hook fires
3. Hook script curls to sidecar /triage
4. Sidecar calls TriageEngine.classify()
5. Result written to session bridge + returned to stdout
6. Claude Code injects result into model context
7. Model processes with triage context
8. Model writes code
9. PostToolUse hook runs mirdan validate --quick
10. Model finishes task
11. Stop hook fires → curls to sidecar /check
12. CheckRunner runs ruff + mypy + pytest
13. Results injected, model fixes issues
```

### MCP path (Cursor)

```
1. User types prompt
2. .mdc rule mandates: "call enhance_prompt before coding"
3. Model calls enhance_prompt MCP tool
4. enhance_prompt checks session bridge for cached triage
5. If no cache: TriageEngine.classify() runs inline
6. Triage result overrides ceremony level
7. Enhanced prompt returned to model
8. Model writes code
9. afterFileEdit hook runs mirdan validate --quick
10. Model calls validate_code_quality MCP tool
11. SmartValidator enriches result with FP filtering
12. Model reviews and fixes
```

## Adding a new backend

1. Create `src/mirdan/llm/newbackend.py`
2. Implement all `LocalLLMProtocol` methods
3. Add selection logic in `LLMManager._create_backend()`
4. Add to `pyproject.toml` optional dependencies if needed
5. Write tests with mocked inference

## Adding a new feature

1. Create prompt template in `src/mirdan/llm/prompts/`
2. Create core module in `src/mirdan/core/`
3. Wire in `LLMManager.startup()` (create instance)
4. Integrate into relevant use case (enhance_prompt or validate_code)
5. Add sidecar endpoint if needed
6. Write tests with mocked LLM
