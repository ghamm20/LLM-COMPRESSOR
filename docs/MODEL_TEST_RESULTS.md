# Model Test Results

This document summarizes the local AirLLM model tests captured in `D:\AI\airllm\receipts`.

| Model | Result | Estimated Weights | Estimated AirLLM Shards | Estimated Total With Safety | Download Attempted | Reason |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `NousResearch/Nous-Hermes-2-Yi-34B` | PASS | 64.05 GiB | 64.05 GiB | 140.92 GiB | yes | Preflight passed as `LlamaForCausalLM`; AirLLM load passed; generation passed after local rotary `position_embeddings` patch. |
| `stepfun-ai/Step-3.5-Flash` | PARTIAL | larger than practical for current disk | larger than practical for current disk | exceeds current free space | no | Blocked by unsupported AirLLM architecture and insufficient disk. |
| `moonshotai/Kimi-K2.6` | PARTIAL | larger than practical for current disk | larger than practical for current disk | exceeds current free space | no | Blocked by config/import failure, unsupported AirLLM indication, and insufficient disk. |
| `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled` | PARTIAL | preflight stopped before weights | preflight stopped before shards | not applicable | no | Blocked because `transformers==4.48.3` does not recognize `model_type=qwen3_5`; package pins were preserved. |
| `meta-llama/Llama-3.1-70B-Instruct` | PARTIAL | 70B-class, gated primary | 70B-class | exceeds current free space | no | Primary model required Hugging Face auth; no weights downloaded. |
| `NousResearch/Meta-Llama-3.1-70B-Instruct` | PARTIAL | 70B-class | 70B-class | exceeds current free space | no | Llama-compatible preflight passed, but estimated weights plus AirLLM shards plus safety margin exceeded D: free space. |

## Yi-34B Result

- Model: `NousResearch/Nous-Hermes-2-Yi-34B`
- Architecture: `LlamaForCausalLM`
- Model type: `llama`
- Load: pass
- Generation: pass
- Output sample: `Explain local LLM inference in one sentence. Local LLM inference refers to the process of using a pre-trained`
- Generation time for 16 new tokens: about 3253 seconds on RTX 3060 Ti 8GB
- Peak GPU memory reserved: about 1.07 GiB
- Disk before rerun: 246.36 GiB free
- Disk after rerun: 246.36 GiB free

## Next Recommended Models

Prefer Llama/Mistral/Yi-era models already supported by AirLLM's architecture mapping and small enough to fit both original weights and AirLLM shards:

- `01-ai/Yi-34B-Chat` if local disk capacity remains acceptable and Hugging Face metadata preflight passes.
- A 13B Llama-compatible instruct model for faster regression checks.
- A Mistral 7B instruct model for quick AirLLM smoke/regression testing after patch changes.

Avoid newer architectures that require newer Transformers model types unless the AirLLM compatibility matrix and pinned package constraints are intentionally revisited.
