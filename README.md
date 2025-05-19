# agent-loop

Agentic Loop in python.

Demonstrates the use of `structured outputs` with vLLM to make function and tool calling more robust.

Serve model `export HF_TOKEN=hf_` and:

```bash
vllm serve \
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16" \
    --max-model-len=1024 \
    --served-model-name=Meta-Llama-3.1-8B-Instruct-quantized.w4a16 \
    --enable-auto-tool-choice \
    --tool-call-parser=granite
```
