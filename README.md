# LLM-COMPRESSOR

LLM-COMPRESSOR is a local large-model inference experiment workspace built around AirLLM layer-sharded loading. The current milestone focuses on running oversized Hugging Face causal language models on a Windows machine with limited VRAM while keeping the Python environment, model cache, AirLLM shards, logs, and receipts isolated under `D:\AI\airllm`.

## Verified Environment

- OS: Windows
- Python: `3.12.10`
- Torch: `2.11.0+cu128`
- CUDA available: yes
- GPU: RTX 3060 Ti 8GB
- AirLLM local workspace: `D:\AI\airllm`
- AirLLM package path: `D:\AI\airllm\repo\air_llm`

## Privacy

Inference is local. Model downloads come from Hugging Face and are cached under the local workspace. No OpenAI, Anthropic, Gemini, Hugging Face Inference API, Spaces, or other remote inference endpoints are used by the test scripts.

## Known Working Model

| Model | Load | Generation | Patch Required | Notes |
| --- | --- | --- | --- | --- |
| `NousResearch/Nous-Hermes-2-Yi-34B` | pass | pass | yes | Runs through AirLLM after the local Llama rotary embedding compatibility patch. |

Run the Yi-34B verification rerun:

```powershell
Set-Location D:\AI\airllm; .\rerun_yi34b_generation.ps1
```

## Known Blocked Models

| Model | Status | Reason |
| --- | --- | --- |
| `stepfun-ai/Step-3.5-Flash` | blocked | Unsupported AirLLM architecture and too large for current disk. |
| `moonshotai/Kimi-K2.6` | blocked | Config/import compatibility issue and too large for current disk. |
| `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled` | blocked | `transformers==4.48.3` does not recognize `model_type=qwen3_5`; package pins were preserved for AirLLM compatibility. |
| `meta-llama/Llama-3.1-70B-Instruct` / `NousResearch/Meta-Llama-3.1-70B-Instruct` | blocked | Compatibility preflight passed for the accessible Llama-compatible backup, but model weights plus AirLLM shards exceeded available D: disk capacity. |

## Patch Summary

Root cause: AirLLM called `LlamaDecoderLayer` directly without passing the `position_embeddings` argument expected by `transformers==4.48.3`. The upstream `LlamaModel.forward` normally creates those shared rotary embeddings and passes them into each decoder layer, but AirLLM bypasses that full model forward path when streaming layer shards.

Patched file:

- `repo\air_llm\airllm\airllm_base.py`

The patch computes rotary position embeddings from `self.model.model.rotary_emb(seq, position_ids)` when available and passes them into decoder-layer calls. Backup and diff receipts are stored under `receipts`.

## Key Receipts

- `receipts\setup_log.txt`
- `receipts\verify_log.txt`
- `receipts\pip_freeze.txt`
- `receipts\yi34b_failure_trace_summary.txt`
- `receipts\yi34b_airllm_code_map.txt`
- `receipts\yi34b_patch.diff`
- `receipts\yi34b_generation_rerun.txt`
- `receipts\yi34b_generation_rerun.json`
- `receipts\yi34b_generation_rerun_stdout.log`
- `receipts\yi34b_generation_rerun_stderr.log`

## Storage Warning

AirLLM layer-sharded loading can require roughly 2x the model weight size because both original Hugging Face weights and AirLLM split shards may be present. The Yi-34B test used about 128 GiB for weights plus shards. 70B-class models need more storage than the current D: free space when model weights, AirLLM shards, and safety margin are considered.

## Repository Hygiene

Do not commit model weights, AirLLM shards, local caches, `.venv`, notebooks, `*.safetensors`, `*.bin`, `*.gguf`, token files, or environment files. These are protected by `.gitignore` and should remain local.
