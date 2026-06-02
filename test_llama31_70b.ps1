$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$CacheDir = Join-Path $Root "cache"
$ModelsDir = Join-Path $Root "models"
$ReceiptsDir = Join-Path $Root "receipts"
$TempDir = Join-Path $CacheDir "temp"
$Runner = Join-Path $TempDir "llama31_70b_runner.py"

New-Item -ItemType Directory -Force -Path `
    (Join-Path $CacheDir "pip"), `
    (Join-Path $CacheDir "torch"), `
    $TempDir, `
    (Join-Path $CacheDir "huggingface\datasets"), `
    (Join-Path $ModelsDir "huggingface\hub"), `
    (Join-Path $ModelsDir "huggingface\transformers"), `
    $ReceiptsDir | Out-Null

$env:PIP_CACHE_DIR = Join-Path $CacheDir "pip"
$env:TEMP = $TempDir
$env:TMP = $TempDir
$env:TORCH_HOME = Join-Path $CacheDir "torch"
$env:XDG_CACHE_HOME = $CacheDir
$env:HF_DATASETS_CACHE = Join-Path $CacheDir "huggingface\datasets"
$env:HF_HOME = Join-Path $ModelsDir "huggingface"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $ModelsDir "huggingface\hub"
$env:TRANSFORMERS_CACHE = Join-Path $ModelsDir "huggingface\transformers"
$env:AIRLLM_LOCAL_MODELS = $ModelsDir

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Missing venv python: $VenvPython"
}

$PythonSource = @'
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\AI\airllm")
CACHE = ROOT / "cache"
MODELS = ROOT / "models"
RECEIPTS = ROOT / "receipts"
TEXT_RECEIPT = RECEIPTS / "llama31_70b_test.txt"
JSON_RECEIPT = RECEIPTS / "llama31_70b_test.json"
PRIMARY_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
BACKUP_MODEL = "NousResearch/Meta-Llama-3.1-70B-Instruct"
PROMPT = "Explain in one sentence what local LLM inference means."
MAX_NEW_TOKENS = 16
DISK_SAFETY_MULTIPLIER = 1.05

SUPPORTED_ARCH_MARKERS = {
    "Qwen2ForCausalLM": "AirLLMQWen2",
    "QWen": "AirLLMQWen",
    "Baichuan": "AirLLMBaichuan",
    "ChatGLM": "AirLLMChatGLM",
    "InternLM": "AirLLMInternLM",
    "Mistral": "AirLLMMistral",
    "Mixtral": "AirLLMMixtral",
    "Llama": "AirLLMLlama2",
}
SUPPORTED_MODEL_TYPES = {
    "qwen2": "AirLLMQWen2",
    "qwen": "AirLLMQWen",
    "baichuan": "AirLLMBaichuan",
    "chatglm": "AirLLMChatGLM",
    "internlm": "AirLLMInternLM",
    "mistral": "AirLLMMistral",
    "mixtral": "AirLLMMixtral",
    "llama": "AirLLMLlama2",
}
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")


def set_local_environment() -> None:
    (CACHE / "pip").mkdir(parents=True, exist_ok=True)
    (CACHE / "torch").mkdir(parents=True, exist_ok=True)
    (CACHE / "temp").mkdir(parents=True, exist_ok=True)
    (CACHE / "huggingface" / "datasets").mkdir(parents=True, exist_ok=True)
    (MODELS / "huggingface" / "hub").mkdir(parents=True, exist_ok=True)
    (MODELS / "huggingface" / "transformers").mkdir(parents=True, exist_ok=True)
    (MODELS / "airllm_shards").mkdir(parents=True, exist_ok=True)
    RECEIPTS.mkdir(parents=True, exist_ok=True)

    os.environ["PIP_CACHE_DIR"] = str(CACHE / "pip")
    os.environ["TEMP"] = str(CACHE / "temp")
    os.environ["TMP"] = str(CACHE / "temp")
    os.environ["TORCH_HOME"] = str(CACHE / "torch")
    os.environ["XDG_CACHE_HOME"] = str(CACHE)
    os.environ["HF_DATASETS_CACHE"] = str(CACHE / "huggingface" / "datasets")
    os.environ["HF_HOME"] = str(MODELS / "huggingface")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS / "huggingface" / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(MODELS / "huggingface" / "transformers")
    os.environ["AIRLLM_LOCAL_MODELS"] = str(MODELS)


def bytes_to_gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024 ** 3), 2)


def d_free() -> dict[str, Any]:
    usage = shutil.disk_usage("D:\\")
    return {
        "free_bytes": usage.free,
        "free_gib": bytes_to_gib(usage.free),
        "total_gib": bytes_to_gib(usage.total),
    }


def safe_model_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id.replace("/", "__")).strip("_")


def get_token() -> str | None:
    try:
        from huggingface_hub import get_token as hf_get_token

        return hf_get_token()
    except Exception:
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def is_auth_error(exc: Exception) -> bool:
    text = repr(exc).lower()
    markers = [
        "gated",
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "restricted",
        "must be authenticated",
        "access to model",
        "requires you to be authenticated",
    ]
    return any(marker in text for marker in markers)


def get_file_name(file_info: Any) -> str:
    return getattr(file_info, "rfilename", None) or getattr(file_info, "path", None) or str(file_info)


def get_file_size(file_info: Any) -> int | None:
    for attr in ("size", "blob_size"):
        value = getattr(file_info, attr, None)
        if isinstance(value, int):
            return value
    return None


def is_weight_file(name: str) -> bool:
    return name.lower().endswith(WEIGHT_SUFFIXES)


def get_ram_snapshot() -> dict[str, Any]:
    try:
        import psutil

        proc = psutil.Process()
        mem = psutil.virtual_memory()
        return {
            "process_rss_gib": bytes_to_gib(proc.memory_info().rss),
            "system_used_gib": bytes_to_gib(mem.used),
            "system_available_gib": bytes_to_gib(mem.available),
            "system_total_gib": bytes_to_gib(mem.total),
        }
    except Exception as exc:
        return {"error": repr(exc)}


def get_gpu_snapshot() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        return {
            "cuda_available": True,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "memory_allocated_gib": bytes_to_gib(torch.cuda.memory_allocated()),
            "memory_reserved_gib": bytes_to_gib(torch.cuda.memory_reserved()),
            "max_memory_allocated_gib": bytes_to_gib(torch.cuda.max_memory_allocated()),
            "max_memory_reserved_gib": bytes_to_gib(torch.cuda.max_memory_reserved()),
        }
    except Exception as exc:
        return {"error": repr(exc)}


def determine_airllm_support(architectures: list[str], model_type: str | None) -> tuple[str, str, str | None]:
    for architecture in architectures:
        for marker, airllm_class in SUPPORTED_ARCH_MARKERS.items():
            if marker in architecture:
                return "yes", f"Architecture {architecture!r} matches AirLLM marker {marker!r}.", airllm_class

    normalized = (model_type or "").lower()
    if normalized in SUPPORTED_MODEL_TYPES:
        return (
            "yes" if normalized == "llama" else "uncertain",
            f"Model type {model_type!r} maps to {SUPPORTED_MODEL_TYPES[normalized]}.",
            SUPPORTED_MODEL_TYPES[normalized],
        )

    return "no", "Architecture/model_type does not match AirLLM 2.11 AutoModel supported markers.", None


def metadata_preflight(model_id: str, token: str | None) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    token_arg = token if token else None
    model_info = api.model_info(model_id, files_metadata=True, token=token_arg)
    siblings = list(getattr(model_info, "siblings", None) or [])
    weight_files = []
    known_weight_bytes = 0
    unknown_weight_file_count = 0
    for sibling in siblings:
        name = get_file_name(sibling)
        size = get_file_size(sibling)
        if is_weight_file(name):
            record = {"name": name, "size_bytes": size, "size_gib": bytes_to_gib(size)}
            weight_files.append(record)
            if size is None:
                unknown_weight_file_count += 1
            else:
                known_weight_bytes += size

    estimated_total_bytes = known_weight_bytes * 2
    estimated_with_margin_bytes = int(estimated_total_bytes * DISK_SAFETY_MULTIPLIER)
    return {
        "status": "pass",
        "model_id": model_id,
        "sha": getattr(model_info, "sha", None),
        "gated": getattr(model_info, "gated", None),
        "private": getattr(model_info, "private", None),
        "library_name": getattr(model_info, "library_name", None),
        "pipeline_tag": getattr(model_info, "pipeline_tag", None),
        "weight_file_count": len(weight_files),
        "unknown_weight_file_count": unknown_weight_file_count,
        "known_weight_bytes": known_weight_bytes,
        "estimated_download_gib": bytes_to_gib(known_weight_bytes),
        "estimated_airllm_shards_gib": bytes_to_gib(known_weight_bytes),
        "estimated_total_disk_needed_gib": bytes_to_gib(estimated_total_bytes),
        "estimated_total_with_safety_margin_gib": bytes_to_gib(estimated_with_margin_bytes),
        "largest_weight_files": sorted(weight_files, key=lambda item: item.get("size_bytes") or 0, reverse=True)[:12],
    }


def config_preflight(model_id: str, token: str | None) -> dict[str, Any]:
    from transformers import AutoConfig

    kwargs = {"trust_remote_code": True}
    if token:
        kwargs["token"] = token
    config = AutoConfig.from_pretrained(model_id, **kwargs)
    config_dict = config.to_dict()
    architectures = list(config_dict.get("architectures") or [])
    model_type = config_dict.get("model_type")
    compatible, reason, mapped_class = determine_airllm_support(architectures, model_type)
    preview_keys = [
        "architectures",
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "torch_dtype",
        "vocab_size",
    ]
    return {
        "status": "pass",
        "config_class": type(config).__name__,
        "config_module": type(config).__module__,
        "architectures": architectures,
        "model_type": model_type,
        "airllm_compatible": compatible,
        "airllm_reason": reason,
        "airllm_mapped_class": mapped_class,
        "config_preview": {key: config_dict.get(key) for key in preview_keys},
    }


def select_model(token: str | None, result: dict[str, Any]) -> tuple[str | None, str]:
    candidates = [PRIMARY_MODEL, BACKUP_MODEL]
    auth_blocked = []
    for model_id in candidates:
        try:
            metadata = metadata_preflight(model_id, token)
            gated = metadata.get("gated")
            if gated and not token:
                result["candidate_preflights"][model_id] = {
                    "metadata": metadata,
                    "config": {"status": "not_attempted", "reason": "model is gated and no local Hugging Face token/login is available"},
                }
                auth_blocked.append(model_id)
                continue
            config = config_preflight(model_id, token)
            result["candidate_preflights"][model_id] = {"metadata": metadata, "config": config}
            return model_id, "selected"
        except Exception as exc:
            status = "auth_blocked" if is_auth_error(exc) else "fail"
            result["candidate_preflights"][model_id] = {
                "metadata": {"status": status, "error": repr(exc)},
                "config": {"status": "not_attempted"},
            }
            if status == "auth_blocked":
                auth_blocked.append(model_id)
                continue
    if auth_blocked:
        return None, "needs_hf_auth"
    return None, "no_candidate_passed_preflight"


def write_receipts(result: dict[str, Any]) -> None:
    JSON_RECEIPT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "AirLLM Llama 3.1 70B local test",
        f"Generated: {result.get('generated_at')}",
        f"Status: {result.get('status')}",
        f"Model used: {result.get('model_used')}",
        f"Download attempted: {result.get('download_attempted')}",
        f"AirLLM load: {result.get('airllm_load', {}).get('status')}",
        f"Generation: {result.get('generation', {}).get('status')}",
        f"Disk free before: {result.get('disk', {}).get('before', {}).get('free_gib')} GiB",
        f"Disk free after: {result.get('disk', {}).get('after', {}).get('free_gib')} GiB",
        f"Estimated total disk needed: {result.get('selected_preflight', {}).get('metadata', {}).get('estimated_total_disk_needed_gib')} GiB",
        f"Estimated total with margin: {result.get('selected_preflight', {}).get('metadata', {}).get('estimated_total_with_safety_margin_gib')} GiB",
        "",
        "Cache paths:",
        json.dumps(result.get("cache_paths"), indent=2, ensure_ascii=False),
        "",
        "Candidate preflights:",
        json.dumps(result.get("candidate_preflights"), indent=2, ensure_ascii=False),
        "",
        "GPU:",
        json.dumps(result.get("gpu"), indent=2, ensure_ascii=False),
        "",
        "RAM:",
        json.dumps(result.get("ram"), indent=2, ensure_ascii=False),
        "",
        "Timing:",
        json.dumps(result.get("timing"), indent=2, ensure_ascii=False),
        "",
        "Prompt:",
        str(result.get("prompt")),
        "",
        "Output sample:",
        str(result.get("output_sample")),
        "",
        "Errors:",
        json.dumps(result.get("errors"), indent=2, ensure_ascii=False),
    ]
    TEXT_RECEIPT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    set_local_environment()
    token = get_token()
    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "primary_model": PRIMARY_MODEL,
        "backup_model": BACKUP_MODEL,
        "model_used": None,
        "status": "started",
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "hf_token_present": bool(token),
        "download_attempted": False,
        "airllm_load": {"status": "not_attempted"},
        "generation": {"status": "not_attempted"},
        "candidate_preflights": {},
        "selected_preflight": {},
        "disk": {"before": d_free(), "after": None, "used_gib": None},
        "gpu": get_gpu_snapshot(),
        "ram": get_ram_snapshot(),
        "timing": {"load_seconds": None, "generation_seconds": None},
        "output_sample": "",
        "errors": [],
        "cache_paths": {
            "HF_HOME": os.environ["HF_HOME"],
            "HUGGINGFACE_HUB_CACHE": os.environ["HUGGINGFACE_HUB_CACHE"],
            "TRANSFORMERS_CACHE": os.environ["TRANSFORMERS_CACHE"],
            "TORCH_HOME": os.environ["TORCH_HOME"],
            "TEMP": os.environ["TEMP"],
            "TMP": os.environ["TMP"],
        },
        "receipts": {
            "text": str(TEXT_RECEIPT),
            "json": str(JSON_RECEIPT),
        },
    }

    exit_code = 1
    try:
        model_id, selection_reason = select_model(token, result)
        result["selection_reason"] = selection_reason
        if selection_reason == "needs_hf_auth":
            result["status"] = "NEEDS_HF_AUTH"
            return 2
        if not model_id:
            result["status"] = "PARTIAL"
            return 3

        selected = result["candidate_preflights"][model_id]
        result["model_used"] = model_id
        result["selected_preflight"] = selected
        metadata = selected["metadata"]
        config = selected["config"]
        free_bytes = result["disk"]["before"]["free_bytes"]
        needed_bytes = int((metadata.get("known_weight_bytes") or 0) * 2 * DISK_SAFETY_MULTIPLIER)

        if metadata.get("unknown_weight_file_count"):
            result["status"] = "PARTIAL"
            result["errors"].append("Refusing download: at least one weight file has unknown size.")
            return 3
        if needed_bytes <= 0:
            result["status"] = "PARTIAL"
            result["errors"].append("Refusing download: could not estimate model weight size.")
            return 3
        if free_bytes < needed_bytes:
            result["status"] = "PARTIAL"
            result["errors"].append(
                "Refusing download: D: free space is insufficient for weights plus AirLLM shards with safety margin."
            )
            return 3
        if config.get("airllm_compatible") != "yes":
            result["status"] = "PARTIAL"
            result["errors"].append(f"Refusing download: AirLLM support is {config.get('airllm_compatible')}.")
            return 3

        import torch
        from airllm import AutoModel

        if not torch.cuda.is_available():
            result["status"] = "FAIL"
            result["errors"].append("Torch CUDA is not available.")
            return 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        device = "cuda:0"
        shard_dir = MODELS / "airllm_shards" / safe_model_name(model_id)
        shard_dir.mkdir(parents=True, exist_ok=True)
        result["shard_dir"] = str(shard_dir)
        kwargs: dict[str, Any] = {
            "device": device,
            "layer_shards_saving_path": str(shard_dir),
        }
        if token:
            kwargs["hf_token"] = token

        result["download_attempted"] = True
        load_start = time.perf_counter()
        model = AutoModel.from_pretrained(model_id, **kwargs)
        result["timing"]["load_seconds"] = round(time.perf_counter() - load_start, 3)
        result["airllm_load"] = {"status": "pass"}

        input_tokens = model.tokenizer(
            [PROMPT],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=128,
            padding=False,
        )
        input_ids = input_tokens["input_ids"].to(device)
        gen_start = time.perf_counter()
        generation_output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            return_dict_in_generate=True,
        )
        result["timing"]["generation_seconds"] = round(time.perf_counter() - gen_start, 3)
        result["generation"] = {"status": "pass"}
        result["output_sample"] = model.tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
        result["status"] = "PASS"
        exit_code = 0
    except Exception as exc:
        result["errors"].append(repr(exc))
        if is_auth_error(exc):
            result["status"] = "NEEDS_HF_AUTH"
            exit_code = 2
        else:
            result["status"] = "FAIL"
            exit_code = 1
        if result["airllm_load"]["status"] == "not_attempted" and result.get("download_attempted"):
            result["airllm_load"] = {"status": "fail", "error": repr(exc)}
        elif result["generation"]["status"] == "not_attempted" and result["airllm_load"].get("status") == "pass":
            result["generation"] = {"status": "fail", "error": repr(exc)}
    finally:
        result["disk"]["after"] = d_free()
        result["disk"]["used_gib"] = bytes_to_gib(result["disk"]["before"]["free_bytes"] - result["disk"]["after"]["free_bytes"])
        result["gpu"] = get_gpu_snapshot()
        result["ram"] = get_ram_snapshot()
        write_receipts(result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Wrote {TEXT_RECEIPT}")
        print(f"Wrote {JSON_RECEIPT}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
'@

Set-Content -Path $Runner -Value $PythonSource -Encoding UTF8
& $VenvPython $Runner
exit $LASTEXITCODE
