# Model Test Results

This document summarizes the local AirLLM model tests captured in `D:\AI\airllm\receipts`.

| Model | Result | Estimated Weights | Estimated AirLLM Shards | Estimated Total With Safety | Download Attempted | Reason |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `NousResearch/Nous-Hermes-2-Yi-34B` | PASS | 64.05 GiB | 64.05 GiB | 140.92 GiB | yes | Preflight passed as `LlamaForCausalLM`; AirLLM load passed; generation passed after local rotary `position_embeddings` patch. |
| `01-ai/Yi-34B-Chat` | PASS | 64.05 GiB | 64.05 GiB | 140.92 GiB | yes | AirLLM mapped it to `AirLLMLlama2`; existing local snapshot and shards loaded; 1-token and 4-token local-only generation passed. |
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

## Yi-34B-Chat Result

- Model: `01-ai/Yi-34B-Chat`
- Architecture: `LlamaForCausalLM`
- AirLLM class: `AirLLMLlama2`
- Existing shards loaded: yes
- Model redownloaded during rerun: no
- Load: pass
- 1-token generation: pass
- 4-token generation: pass
- Output sample, 1-token: `Local inference means that`
- Output sample, 4-token: `Local inference means that模型在本地`
- Runtime: about 210 seconds/token on RTX 3060 Ti 8GB
- Practical classification: compatibility proof, not a daily interactive model
- Old `position_embeddings` bug returned: no
- Disk before/after reruns: 123.02 GiB free / 123.02 GiB free
- Receipts: `yi34b_chat_minimal_rerun.*`, `yi34b_chat_4token_rerun.*`, and `yi34b_chat_partial_forensics.*`

The earlier 16-token Yi-34B-Chat run was partial because Windows rebooted during generation. Windows System logs showed `python.exe` delaying shutdown just before a kernel-initiated reboot, so the process never reached the final receipt writer. The follow-up local-only reruns confirmed that the downloaded snapshot and AirLLM shards were usable.

The 4-token rerun wrote valid PASS JSON/TXT receipts, then redirected stdout hit a `UnicodeEncodeError` while printing generated non-ASCII text to the console. The launcher/test scripts now set `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1`, and the Yi-34B-Chat Python runners print ASCII-safe JSON unless stdout reports UTF-8.

## Next Recommended Models

Prefer Llama/Mistral/Yi-era models already supported by AirLLM's architecture mapping and small enough to fit both original weights and AirLLM shards:

- A 13B Llama-compatible instruct model for faster regression checks.
- A Mistral 7B instruct model for quick AirLLM smoke/regression testing after patch changes.
- A shorter-token Yi-34B-Chat regression check before attempting another 16-token run.

Avoid newer architectures that require newer Transformers model types unless the AirLLM compatibility matrix and pinned package constraints are intentionally revisited.
