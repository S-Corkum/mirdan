# Troubleshooting

## Start here

```bash
mirdan llm setup --check
```

This validates your entire setup: hardware, backends, models, and configuration.

## Common issues

### "LLM backend unavailable"

- **Ollama:** Is it running? `curl http://localhost:11434/api/tags`
- **llama-cpp-python:** Is it installed? `python -c "import llama_cpp"`
- Check `mirdan llm status` for detailed error

### "Model not found"

- Run `mirdan llm setup` to download the recommended model
- For Ollama: `ollama pull gemma4:e2b`
- For llama-cpp: check `~/.mirdan/models/` for `.gguf` files

### Slow inference (<5 tok/s)

- Check `mirdan llm status` for performance info
- If llama-cpp-python: verify Metal/AVX2 compilation:
  ```bash
  python -c "import llama_cpp; print(llama_cpp.__version__)"
  ```
- Reinstall with Metal (Apple Silicon):
  ```bash
  CMAKE_ARGS="-DGGML_METAL=ON" pip install --force-reinstall llama-cpp-python
  ```
- If Ollama: try llama-cpp-python for ~200MB less overhead

### High memory usage

- Check `model_keep_alive` setting (default `5m` — model unloads after idle)
- Use `mirdan llm status` to see current memory usage
- Consider E2B Q4 (3.5GB) instead of E4B Q3 (4.5GB) if tight on memory
- Set `model_keep_alive: 2m` for faster unloading

### Hooks not firing

- Verify hooks.json exists: `.claude/hooks.json` or `.cursor/hooks.json`
- Run `mirdan init --claude-code` or `mirdan init --cursor` to regenerate
- Check `.mirdan/sidecar.port` exists (sidecar must be running via MCP server)
- Manual test: `echo '{"prompt":"test"}' | mirdan triage --stdin`

### "triage not configured" in hook output

This is normal — it means the triage feature hasn't been enabled yet. The stub response tells the paid model to proceed normally. No impact on your workflow.

To enable: set `llm.enabled: true` in `.mirdan.yaml` and restart the MCP server.

### Enyal + mirdan memory conflict

Both fit on 16GB:

- Enyal: ~1GB (Qwen3-Embedding-0.6B)
- mirdan E4B Q3: ~4.5GB
- Total: ~5.5GB with headroom

If tight: set `ENYAL_PRELOAD_MODEL=false` for lazy loading.

### Ollama can't select Q3 quantization for E4B

Ollama tags don't expose quantization levels. `ollama pull gemma4:e4b` downloads the default Q4 (~5GB) which may be too large for 16GB.

**Solution:** Use llama-cpp-python with a Q3 GGUF file from HuggingFace. Run `mirdan llm setup` — it guides you to the right choice.

## Corporate Network / Netskope / Zscaler

If your enterprise uses SSL inspection (Netskope, Zscaler, BlueCoat), model
downloads will fail with SSL certificate errors. `mirdan llm setup` detects
this automatically and prints instructions.

### Quickest fix

```bash
pip install truststore    # Uses OS trust store — handles Netskope automatically
```

### For GGUF downloads (llama-cpp-python backend)

Point to your Artifactory HuggingFace proxy:
```bash
export MIRDAN_HF_ENDPOINT=https://artifactory.corp.com/huggingface
```

Or specify your corporate CA bundle:
```bash
export MIRDAN_SSL_CERT_FILE=/path/to/corporate-ca-bundle.crt
```

### For Ollama model pulls

Ollama is a separate daemon — mirdan can't configure it. You must:

1. Install your corporate CA cert into the **system trust store**:
   - macOS: `sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain /path/to/corp-ca.crt`
   - Linux: `sudo cp /path/to/corp-ca.crt /usr/local/share/ca-certificates/ && sudo update-ca-certificates`

2. Configure proxy (if needed):
   - macOS: `launchctl setenv HTTPS_PROXY http://proxy.corp.com:8080`
   - Linux: `sudo systemctl edit ollama` then add `Environment="HTTPS_PROXY=http://proxy.corp.com:8080"` under `[Service]`

3. Restart Ollama

### Air-gapped / offline environments

```bash
export MIRDAN_OFFLINE_MODE=true
```

Pre-download the GGUF file on a connected machine and place it in `~/.mirdan/models/`.

### If enyal is also installed

mirdan automatically reuses enyal's SSL configuration. If you've already
configured `ENYAL_HF_ENDPOINT` and `ENYAL_SSL_CERT_FILE`, mirdan picks
them up with no additional configuration.

## Debug logging

```bash
MIRDAN_LOG_LEVEL=DEBUG mirdan llm status
```
