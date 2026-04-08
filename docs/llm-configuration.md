# LLM Configuration Reference

All settings go in `.mirdan.yaml` at your project root, under the `llm:` key.

## Core settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Enable local LLM features |
| `backend` | string | `"auto"` | `auto`, `ollama`, or `llamacpp` |
| `ollama_url` | string | `"http://localhost:11434"` | Ollama daemon URL |
| `gguf_dir` | string | `"~/.mirdan/models"` | Directory for GGUF model files |
| `model_keep_alive` | string | `"5m"` | Unload model after this idle period |
| `n_ctx` | int | `4096` | Context window (tokens). 4096 is sufficient. |
| `n_threads` | int | `null` | CPU threads. null = auto (cpu_count / 2) |

## Feature toggles

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `triage` | bool | `true` | Classify tasks before paid model |
| `smart_validation` | bool | `true` | LLM-enriched validation (FP filtering, root causes) |
| `check_runner` | bool | `true` | Run lint/typecheck/test locally |
| `prompt_optimization` | bool | `true` | Per-model prompt crafting (needs BRAIN model) |
| `research_agent` | bool | `false` | Experimental agentic research (needs BRAIN model) |

## Safety settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_false_positive_ratio` | float | `0.4` | Cap false positives at this ratio (0.0-1.0) |
| `validate_llm_fixes` | bool | `true` | Re-validate LLM-suggested fixes through rule engine |

## Check runner settings

Nested under `checks:`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lint_command` | string | `"ruff check"` | Lint command |
| `typecheck_command` | string | `"mypy"` | Type check command |
| `test_command` | string | `"pytest -x --tb=short"` | Test command |
| `test_timeout` | int | `30` | Max seconds for test execution |
| `auto_fix_lint` | bool | `true` | Run lint --fix for auto-fixable issues |

## Example configurations

### Minimal (16GB laptop, Ollama)

```yaml
llm:
  enabled: true
```

### Optimal (16GB laptop, llama-cpp-python)

```yaml
llm:
  enabled: true
  backend: llamacpp
```

### Full features (64GB Apple Silicon)

```yaml
llm:
  enabled: true
  prompt_optimization: true
  research_agent: true
```

### Custom test command

```yaml
llm:
  enabled: true
  checks:
    test_command: "pytest tests/unit/ -x --tb=short -q"
    test_timeout: 60
```

### Conservative safety settings

```yaml
llm:
  enabled: true
  max_false_positive_ratio: 0.2
  validate_llm_fixes: true
```
