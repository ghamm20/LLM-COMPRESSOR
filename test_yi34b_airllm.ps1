param(
    [switch]$Inner
)

$ErrorActionPreference = "Stop"

$Root = "D:\AI\airllm"
$VenvDir = Join-Path $Root ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Activate = Join-Path $VenvDir "Scripts\Activate.ps1"
$CacheDir = Join-Path $Root "cache"
$ModelsDir = Join-Path $Root "models"
$ReceiptsDir = Join-Path $Root "receipts"
$TempDir = Join-Path $CacheDir "temp"
$Runner = Join-Path $TempDir "yi34b_airllm_runner.py"
$StdoutLog = Join-Path $ReceiptsDir "yi34b_console_stdout.log"
$StderrLog = Join-Path $ReceiptsDir "yi34b_console_stderr.log"

New-Item -ItemType Directory -Force -Path `
    (Join-Path $CacheDir "pip"), `
    (Join-Path $CacheDir "torch"), `
    $TempDir, `
    (Join-Path $CacheDir "huggingface\datasets"), `
    (Join-Path $ModelsDir "huggingface\hub"), `
    (Join-Path $ModelsDir "huggingface\transformers"), `
    $ReceiptsDir | Out-Null

if (-not $Inner) {
    Remove-Item -LiteralPath $StdoutLog, $StderrLog -Force -ErrorAction SilentlyContinue
    $process = Start-Process -FilePath "powershell" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath, "-Inner") `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog `
        -WindowStyle Hidden `
        -PassThru `
        -Wait
    exit $process.ExitCode
}

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
if (-not (Test-Path -LiteralPath $Activate)) {
    throw "Missing venv activation script: $Activate"
}
. $Activate

$PythonSource = @'
from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\AI\airllm")
CACHE = ROOT / "cache"
MODELS = ROOT / "models"
RECEIPTS = ROOT / "receipts"
TEXT_RECEIPT = RECEIPTS / "yi34b_airllm_test.txt"
JSON_RECEIPT = RECEIPTS / "yi34b_airllm_test.json"
MODEL_ID = "NousResearch/Nous-Hermes-2-Yi-34B"
PROMPT = "Explain local LLM inference in one sentence."
MAX_NEW_TOKENS = 24
SAFETY_MULTIPLIER = 1.10
SHARD_DIR = MODELS / "NousResearch_Nous-Hermes-2-Yi-34B" / "airllm_shards"

AIRLLM_AUTO_MARKERS = [
    ("Qwen2ForCausalLM", "AirLLMQWen2"),
    ("QWen", "AirLLMQWen"),
    ("Baichuan", "AirLLMBaichuan"),
    ("ChatGLM", "AirLLMChatGLM"),
    ("InternLM", "AirLLMInternLM"),
    ("Mistral", "AirLLMMistral"),
    ("Mixtral", "AirLLMMixtral"),
    ("Llama", "AirLLMLlama2"),
]
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
NON_AIRLLM_WEIGHT_SUFFIXES = (".gguf",)


def set_local_environment() -> None:
    (CACHE / "pip").mkdir(parents=True, exist_ok=True)
    (CACHE / "torch").mkdir(parents=True, exist_ok=True)
    (CACHE / "temp").mkdir(parents=True, exist_ok=True)
    (CACHE / "huggingface" / "datasets").mkdir(parents=True, exist_ok=True)
    (MODELS / "huggingface" / "hub").mkdir(parents=True, exist_ok=True)
    (MODELS / "huggingface" / "transformers").mkdir(parents=True, exist_ok=True)
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
    return round(float(value) / (1024**3), 2)


def compact_value(value: Any, max_items: int = 24) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return compact_value(asdict(value), max_items=max_items)
    if hasattr(value, "to_dict"):
        try:
            return compact_value(value.to_dict(), max_items=max_items)
        except Exception:
            pass
    if hasattr(value, "__dict__") and not isinstance(value, type):
        try:
            return compact_value(vars(value), max_items=max_items)
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): compact_value(v, max_items=max_items) for k, v in list(value.items())[:max_items]}
    if isinstance(value, (list, tuple)):
        return [compact_value(v, max_items=max_items) for v in list(value)[:max_items]]
    return repr(value)


def d_drive_free() -> dict[str, Any]:
    usage = shutil.disk_usage("D:\\")
    return {
        "free_bytes": usage.free,
        "free_gib": bytes_to_gib(usage.free),
        "total_gib": bytes_to_gib(usage.total),
    }


def get_token() -> str | None:
    try:
        from huggingface_hub import get_token as hf_get_token

        return hf_get_token()
    except Exception:
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def is_auth_error(exc: Exception) -> bool:
    text = repr(exc).lower()
    return any(marker in text for marker in [
        "gated",
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "restricted",
        "must be authenticated",
        "requires you to be authenticated",
    ])


def file_name(file_info: Any) -> str:
    return getattr(file_info, "rfilename", None) or getattr(file_info, "path", None) or str(file_info)


def file_size(file_info: Any) -> int | None:
    for attr in ("size", "blob_size"):
        value = getattr(file_info, attr, None)
        if isinstance(value, int):
            return value
    return None


def is_weight_file(name: str) -> bool:
    return name.lower().endswith(WEIGHT_SUFFIXES)


def is_non_airllm_weight(name: str) -> bool:
    return name.lower().endswith(NON_AIRLLM_WEIGHT_SUFFIXES)


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


def determine_airllm_compatibility(architectures: list[str], model_type: str | None) -> dict[str, Any]:
    first_architecture = architectures[0] if architectures else ""
    for marker, airllm_class in AIRLLM_AUTO_MARKERS:
        if marker in first_architecture:
            return {
                "status": "yes",
                "mapped_class": airllm_class,
                "reason": f"AirLLM AutoModel would match architecture {first_architecture!r} with marker {marker!r}.",
                "first_architecture": first_architecture,
                "model_type": model_type,
                "auto_markers": [marker for marker, _ in AIRLLM_AUTO_MARKERS],
            }
    if model_type in {"llama", "yi"}:
        return {
            "status": "uncertain",
            "mapped_class": None,
            "reason": f"Model type {model_type!r} is Llama/Yi-era, but AirLLM dispatches by architecture string and this architecture does not clearly match its markers.",
            "first_architecture": first_architecture,
            "model_type": model_type,
            "auto_markers": [marker for marker, _ in AIRLLM_AUTO_MARKERS],
        }
    return {
        "status": "no",
        "mapped_class": None,
        "reason": "Architecture/model_type does not match AirLLM 2.11 AutoModel dispatch markers.",
        "first_architecture": first_architecture,
        "model_type": model_type,
        "auto_markers": [marker for marker, _ in AIRLLM_AUTO_MARKERS],
    }


def metadata_preflight(token: str | None) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    model_info = api.model_info(MODEL_ID, files_metadata=True, token=token if token else None)
    siblings = list(getattr(model_info, "siblings", None) or [])
    weight_files = []
    non_airllm_weights = []
    known_weight_bytes = 0
    unknown_weight_count = 0
    for sibling in siblings:
        name = file_name(sibling)
        size = file_size(sibling)
        record = {"name": name, "size_bytes": size, "size_gib": bytes_to_gib(size)}
        if is_weight_file(name):
            weight_files.append(record)
            if size is None:
                unknown_weight_count += 1
            else:
                known_weight_bytes += size
        elif is_non_airllm_weight(name):
            non_airllm_weights.append(record)

    estimated_shards_bytes = known_weight_bytes
    estimated_total_bytes = known_weight_bytes + estimated_shards_bytes
    estimated_total_with_margin_bytes = int(estimated_total_bytes * SAFETY_MULTIPLIER)
    return {
        "status": "pass",
        "sha": getattr(model_info, "sha", None),
        "gated": getattr(model_info, "gated", None),
        "private": getattr(model_info, "private", None),
        "library_name": getattr(model_info, "library_name", None),
        "pipeline_tag": getattr(model_info, "pipeline_tag", None),
        "tags": list(getattr(model_info, "tags", None) or [])[:40],
        "parameter_hints": {
            "safetensors": compact_value(getattr(model_info, "safetensors", None)),
            "card_data": compact_value(getattr(model_info, "cardData", None), max_items=30),
        },
        "storage_hints": {
            "repo_file_count": len(siblings),
            "weight_file_count": len(weight_files),
            "unknown_weight_file_count": unknown_weight_count,
            "known_weight_bytes": known_weight_bytes,
            "estimated_weight_gib": bytes_to_gib(known_weight_bytes),
            "estimated_airllm_shard_gib": bytes_to_gib(estimated_shards_bytes),
            "estimated_total_disk_needed_gib": bytes_to_gib(estimated_total_bytes),
            "estimated_total_with_safety_margin_gib": bytes_to_gib(estimated_total_with_margin_bytes),
            "safety_multiplier": SAFETY_MULTIPLIER,
            "largest_weight_files": sorted(weight_files, key=lambda item: item.get("size_bytes") or 0, reverse=True)[:16],
            "non_airllm_weight_like_files": sorted(non_airllm_weights, key=lambda item: item.get("size_bytes") or 0, reverse=True)[:16],
        },
    }


def config_preflight(token: str | None) -> dict[str, Any]:
    from transformers import AutoConfig

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if token:
        kwargs["token"] = token
    config = AutoConfig.from_pretrained(MODEL_ID, **kwargs)
    config_dict = config.to_dict()
    architectures = list(config_dict.get("architectures") or [])
    model_type = config_dict.get("model_type")
    compatibility = determine_airllm_compatibility(architectures, model_type)
    preview_keys = [
        "architectures",
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "torch_dtype",
        "vocab_size",
    ]
    return {
        "status": "pass",
        "config_class": type(config).__name__,
        "config_module": type(config).__module__,
        "model_type": model_type,
        "architectures": architectures,
        "config_preview": {key: config_dict.get(key) for key in preview_keys},
        "airllm_architecture_compatibility": compatibility,
    }


def write_receipts(result: dict[str, Any]) -> None:
    JSON_RECEIPT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "AirLLM Yi 34B local test",
        f"Generated: {result.get('generated_at')}",
        f"Status: {result.get('status')}",
        f"Model used: {result.get('model_id')}",
        f"Download attempted: {result.get('download_attempted')}",
        f"AirLLM architecture compatibility: {result.get('airllm_architecture_compatibility', {}).get('status')}",
        f"AirLLM compatibility reason: {result.get('airllm_architecture_compatibility', {}).get('reason')}",
        f"AirLLM load: {result.get('airllm_load', {}).get('status')}",
        f"Generation: {result.get('generation', {}).get('status')}",
        f"Disk free before: {result.get('disk', {}).get('before', {}).get('free_gib')} GiB",
        f"Disk free after: {result.get('disk', {}).get('after', {}).get('free_gib')} GiB",
        f"Estimated weight size: {result.get('metadata', {}).get('storage_hints', {}).get('estimated_weight_gib')} GiB",
        f"Estimated AirLLM shard size: {result.get('metadata', {}).get('storage_hints', {}).get('estimated_airllm_shard_gib')} GiB",
        f"Estimated total with safety margin: {result.get('metadata', {}).get('storage_hints', {}).get('estimated_total_with_safety_margin_gib')} GiB",
        "",
        "Metadata:",
        json.dumps(result.get("metadata"), indent=2, ensure_ascii=False),
        "",
        "Config:",
        json.dumps(result.get("config"), indent=2, ensure_ascii=False),
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
        "Output sample:",
        str(result.get("output_sample") or ""),
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
        "model_id": MODEL_ID,
        "status": "started",
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "hf_token_present": bool(token),
        "download_attempted": False,
        "metadata": {"status": "not_started"},
        "config": {"status": "not_started"},
        "airllm_architecture_compatibility": {"status": "unknown"},
        "airllm_load": {"status": "not_attempted"},
        "generation": {"status": "not_attempted"},
        "disk": {"before": d_drive_free(), "after": None, "used_gib": None},
        "gpu": get_gpu_snapshot(),
        "ram": get_ram_snapshot(),
        "timing": {
            "load_start": None,
            "load_end": None,
            "load_seconds": None,
            "generation_start": None,
            "generation_end": None,
            "generation_seconds": None,
        },
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
        "shard_dir": str(SHARD_DIR),
        "receipts": {
            "text": str(TEXT_RECEIPT),
            "json": str(JSON_RECEIPT),
        },
    }

    exit_code = 1
    try:
        print(f"AirLLM Yi 34B test root: {ROOT}")
        print(f"Model: {MODEL_ID}")
        print(f"D: free before any download: {result['disk']['before']['free_gib']} GiB")

        try:
            result["metadata"] = metadata_preflight(token)
        except Exception as exc:
            result["metadata"] = {"status": "auth_blocked" if is_auth_error(exc) else "fail", "error": repr(exc)}
            result["status"] = "PARTIAL" if is_auth_error(exc) else "FAIL"
            result["errors"].append(f"Metadata preflight failed: {repr(exc)}")
            return 3 if is_auth_error(exc) else 1

        try:
            result["config"] = config_preflight(token)
            result["airllm_architecture_compatibility"] = result["config"]["airllm_architecture_compatibility"]
        except Exception as exc:
            result["config"] = {"status": "fail", "error": repr(exc)}
            result["airllm_architecture_compatibility"] = {"status": "no", "reason": "Config could not be loaded locally with trust_remote_code=True."}
            result["status"] = "PARTIAL"
            result["errors"].append(f"Config preflight failed: {repr(exc)}")
            return 3

        storage = result["metadata"]["storage_hints"]
        compatibility = result["airllm_architecture_compatibility"]
        needed_gib = storage.get("estimated_total_with_safety_margin_gib")
        free_gib = result["disk"]["before"]["free_gib"]
        print(f"model_type: {result['config'].get('model_type')}")
        print(f"architectures: {result['config'].get('architectures')}")
        print(f"estimated weights: {storage.get('estimated_weight_gib')} GiB")
        print(f"estimated shards: {storage.get('estimated_airllm_shard_gib')} GiB")
        print(f"estimated total with safety margin: {needed_gib} GiB")
        print(f"AirLLM compatibility: {compatibility.get('status')} - {compatibility.get('reason')}")

        if storage.get("unknown_weight_file_count"):
            result["status"] = "PARTIAL"
            result["errors"].append("Refusing download: at least one weight file has unknown size.")
            return 3
        if not storage.get("known_weight_bytes"):
            result["status"] = "PARTIAL"
            result["errors"].append("Refusing download: no safetensors/bin/pt/pth/ckpt weights were found in metadata.")
            return 3
        if compatibility.get("status") != "yes":
            result["status"] = "PARTIAL"
            result["errors"].append("Refusing download: AirLLM AutoModel does not clearly support this architecture.")
            return 3
        if needed_gib is None or free_gib is None or needed_gib > free_gib:
            result["status"] = "PARTIAL"
            result["errors"].append("Refusing download: estimated weights plus AirLLM shards plus safety margin exceed free D: space.")
            return 3

        import torch
        from airllm import AutoModel

        if not torch.cuda.is_available():
            result["status"] = "FAIL"
            result["errors"].append("Torch CUDA is not available.")
            return 1

        SHARD_DIR.mkdir(parents=True, exist_ok=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result["download_attempted"] = True

        load_start = time.perf_counter()
        result["timing"]["load_start"] = datetime.now(timezone.utc).isoformat()
        model_kwargs: dict[str, Any] = {
            "device": "cuda:0",
            "layer_shards_saving_path": str(SHARD_DIR),
        }
        if token:
            model_kwargs["hf_token"] = token
        model = AutoModel.from_pretrained(MODEL_ID, **model_kwargs)
        result["timing"]["load_end"] = datetime.now(timezone.utc).isoformat()
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
        input_ids = input_tokens["input_ids"].to("cuda:0")
        generation_start = time.perf_counter()
        result["timing"]["generation_start"] = datetime.now(timezone.utc).isoformat()
        generation_output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            return_dict_in_generate=True,
        )
        result["timing"]["generation_end"] = datetime.now(timezone.utc).isoformat()
        result["timing"]["generation_seconds"] = round(time.perf_counter() - generation_start, 3)
        result["generation"] = {"status": "pass"}
        result["output_sample"] = model.tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)[:2000]
        result["status"] = "PASS"
        exit_code = 0
    except Exception as exc:
        result["errors"].append(repr(exc))
        if result["download_attempted"] and result["airllm_load"]["status"] == "not_attempted":
            result["airllm_load"] = {"status": "fail", "error": repr(exc)}
        elif result["airllm_load"].get("status") == "pass" and result["generation"]["status"] == "not_attempted":
            result["generation"] = {"status": "fail", "error": repr(exc)}
        result["status"] = "FAIL"
        exit_code = 1
    finally:
        result["disk"]["after"] = d_drive_free()
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
