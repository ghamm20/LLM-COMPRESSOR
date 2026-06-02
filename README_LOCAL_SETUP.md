# AirLLM Local Setup

This workspace is intentionally isolated on the detachable drive:

```text
D:\AI\airllm
├── .venv
├── cache
├── models
├── receipts
├── scripts
└── repo
```

## What Was Inspected

The uploaded archive is a GitHub-style `airllm-main.zip`. The installable Python package lives in:

```text
D:\AI\airllm\repo\air_llm
```

The package metadata is `repo\air_llm\setup.py`, declaring `airllm==2.11.0` and these runtime dependencies: `torch`, `transformers`, `accelerate`, `safetensors`, `optimum`, `huggingface-hub`, `scipy`, and `tqdm`.

The top-level `repo\requirements.txt` appears to be training/finetuning extras, not the runtime package requirements. It pins older packages including `bitsandbytes==0.39.0`, `scikit-learn==1.2.2`, old Git-based `peft`, old Git-based `accelerate`, and Git-based `transformers`. On Windows with Python 3.12, those pins are likely to conflict or require source builds, so `setup_windows.ps1` does not install them by default.

Runtime compatibility pins applied by the setup script:

```text
optimum==1.27.0
transformers==4.48.3
sentencepiece==0.2.1
```

Reason: AirLLM imports `optimum.bettertransformer`, which is absent from `optimum>=2`, and `optimum.bettertransformer` requires `transformers<4.49`. AirLLM also imports the Baichuan tokenizer at package import time, which requires `sentencepiece`.

## Setup

From PowerShell:

```powershell
cd D:\AI\airllm
.\setup_windows.ps1
```

If local PowerShell execution policy blocks scripts, use a process-scoped bypass:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\AI\airllm\setup_windows.ps1
```

The setup script creates or reuses `D:\AI\airllm\.venv`, activates it, upgrades `pip`, `setuptools`, and `wheel`, installs a CUDA-capable PyTorch wheel when `nvidia-smi` reports a supported CUDA runtime, falls back to CPU PyTorch if CUDA install or validation fails, and installs AirLLM editable from `D:\AI\airllm\repo\air_llm`.

## Verification

```powershell
cd D:\AI\airllm
.\verify_airllm.ps1
```

Required receipts:

```text
D:\AI\airllm\receipts\setup_log.txt
D:\AI\airllm\receipts\verify_log.txt
D:\AI\airllm\receipts\pip_freeze.txt
```

Additional machine-readable summaries are written to:

```text
D:\AI\airllm\receipts\setup_summary.json
D:\AI\airllm\receipts\verify_summary.json
```

## Cache And Model Paths

Process-scoped environment variables are set by the scripts only. They are not written permanently.

Model downloads default to:

```text
D:\AI\airllm\models
```

Practical package/tool caches default to:

```text
D:\AI\airllm\cache
```

The setup and verification scripts also set process-scoped `TEMP` and `TMP` to `D:\AI\airllm\cache\temp` to reduce incidental use of the Windows default temp directory.

## Real Model Command

The smoke test does not download a model. To trigger a real AirLLM model load with a relatively small Llama-family model:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "cd D:\AI\airllm; $env:HF_HOME='D:\AI\airllm\models\huggingface'; $env:HUGGINGFACE_HUB_CACHE='D:\AI\airllm\models\huggingface\hub'; $env:TRANSFORMERS_CACHE='D:\AI\airllm\models\huggingface\transformers'; $env:TORCH_HOME='D:\AI\airllm\cache\torch'; .\.venv\Scripts\python.exe -c \"from airllm import AutoModel; import torch; model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0'; device='cuda:0' if torch.cuda.is_available() else 'cpu'; model=AutoModel.from_pretrained(model_id, device=device, layer_shards_saving_path=r'D:\AI\airllm\models\TinyLlama-1.1B-Chat-v1.0\airllm_shards'); print('Loaded', model_id, 'on', device)\""
```

This will download model files and may create AirLLM layer shards. Use a gated model only after logging in to Hugging Face or passing an appropriate token.

## Disk, RAM, And VRAM Notes

AirLLM can reduce VRAM pressure by loading layers, but it still needs disk space for the original Hugging Face model and AirLLM's layer-sharded copy. Plan for at least two copies of the model during first preparation.

The README examples include 70B and 405B models. Those are very large downloads and can require hundreds of GB of disk and substantial system RAM even if VRAM use is reduced. The detected 8 GB class GPU should not be treated as proof that every model will run well.

## Windows Troubleshooting

If `torch.cuda.is_available()` is false, the setup script will not claim GPU support. Check the NVIDIA driver, rerun setup, and inspect `D:\AI\airllm\receipts\setup_log.txt`.

If script execution is blocked, use the process-scoped `-ExecutionPolicy Bypass` command shown above.

If editable install fails around `optimum.bettertransformer`, the minimal fix is likely to pin an older compatible `optimum` release rather than changing system Python or installing global packages.

If training extras are needed later, use a separate plan. The top-level `requirements.txt` is not part of the runtime verification path and may require Python 3.10 or 3.11 plus updated Windows-compatible pins.

## Verification Status

PASS as of June 2, 2026.

```text
Project path: D:\AI\airllm
Python: Python 3.12.10
Torch: 2.11.0+cu128
CUDA detected by nvidia-smi: 13.1
Torch CUDA runtime: 12.8
GPU detected: NVIDIA GeForce RTX 3060 Ti, 8192 MiB
AirLLM import: pass
Smoke test: pass
pip check: pass
```

Free space observations:

```text
D: free before first install preflight: 381.62 GB
D: free after verified setup: 374.32 GB
```

Dependency fixes applied:

```text
setuptools constrained to <82 because torch 2.11.0 requires setuptools<82
optimum pinned to 1.27.0 because AirLLM imports optimum.bettertransformer
transformers pinned to 4.48.3 because optimum.bettertransformer requires transformers<4.49
sentencepiece installed because AirLLM imports Baichuan tokenizer support at package import time
```

No large model was downloaded during verification.
