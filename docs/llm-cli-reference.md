# LLM CLI Reference

## mirdan llm setup

Interactive setup wizard. Detects hardware, recommends model, downloads, configures `.mirdan.yaml`.

```bash
mirdan llm setup          # Interactive wizard
mirdan llm setup --check  # Non-interactive validation
```

### What it detects

- CPU architecture (x86_64 / arm64)
- Available RAM and Enyal memory usage
- Installed backends (Ollama / llama-cpp-python)
- llama.cpp compilation flags (Metal / AVX2)
- Optimal model for your hardware

## mirdan llm status

Show current LLM health, loaded model, hardware profile, features.

```bash
mirdan llm status         # Human-readable
mirdan llm status --json  # Machine-readable
```

### Example output

```
LLM enabled:  True
Backend:      auto
Architecture: arm64
RAM:          16384 MB total, 8200 MB available
Profile:      standard
GPU:          Apple M2

Features:
  triage: enabled
  smart_validation: enabled
  check_runner: enabled
  prompt_optimization: enabled
  research_agent: disabled
```

## mirdan llm warmup

Pre-load the configured model into memory. Handled automatically by the MCP server.

```bash
mirdan llm warmup
```

## mirdan llm metrics

Show token savings dashboard.

```bash
mirdan llm metrics             # Last 30 days
mirdan llm metrics --days 7    # Last 7 days
mirdan llm metrics --json      # Machine-readable
```

### Example output

```
=== mirdan LLM Metrics (30 days) ===

LLM calls:           1,247
Local tokens used:   312,000
Est. paid saved:     890,000
Savings:             41.2%
Triage count:        523

Triage distribution:
  local_only: 142
  local_assist: 98
  paid_minimal: 156
  paid_required: 127
```

## mirdan triage --stdin

Classify a task via stdin. Used by hooks, not typically run manually.

```bash
echo '{"prompt":"fix the unused import"}' | mirdan triage --stdin
```

### Output

```json
{"classification": "local_only", "confidence": 0.95, "reasoning": "Single unused import removal"}
```

The command tries the HTTP sidecar first (fast path, <5ms). Falls back to direct LLM call if the sidecar is not running.

## mirdan check --smart

Run lint + typecheck + tests with optional LLM analysis.

```bash
mirdan check --smart                    # All files
mirdan check --smart src/auth/ src/api/ # Specific paths
```

### Output

```json
{
  "lint": {"command": "ruff check", "returncode": 0, ...},
  "typecheck": {"command": "mypy", "returncode": 1, ...},
  "test": {"command": "pytest -x --tb=short", "returncode": 0, ...},
  "all_pass": false,
  "auto_fixed": ["lint auto-fixed"],
  "needs_attention": [{"tool": "typecheck", "classification": "trivial", ...}],
  "summary": "1 trivial type error"
}
```

## mirdan fine-tune status

Show collected training sample counts.

```bash
mirdan fine-tune status
```

## mirdan fine-tune export

Export training data as JSONL.

```bash
mirdan fine-tune export                  # Print to stdout
mirdan fine-tune export --format jsonl   # Explicit format
```
