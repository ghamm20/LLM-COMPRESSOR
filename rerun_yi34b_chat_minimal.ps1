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
$Runner = Join-Path $TempDir "yi34b_chat_minimal_runner.py"
$StdoutLog = Join-Path $ReceiptsDir "yi34b_chat_minimal_stdout.log"
$StderrLog = Join-Path $ReceiptsDir "yi34b_chat_minimal_stderr.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

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
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"

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
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\AI\airllm")
CACHE = ROOT / "cache"
MODELS = ROOT / "models"
RECEIPTS = ROOT / "receipts"
MODEL_ID = "01-ai/Yi-34B-Chat"
LOCAL_MODEL_ROOT = MODELS / "huggingface" / "hub" / "models--01-ai--Yi-34B-Chat"
SHARD_DIR = MODELS / "01-ai_Yi-34B-Chat" / "airllm_shards"
SPLIT_DIR = SHARD_DIR / "splitted_model"
TEXT_RECEIPT = RECEIPTS / "yi34b_chat_minimal_rerun.txt"
JSON_RECEIPT = RECEIPTS / "yi34b_chat_minimal_rerun.json"
STDOUT_LOG = RECEIPTS / "yi34b_chat_minimal_stdout.log"
STDERR_LOG = RECEIPTS / "yi34b_chat_minimal_stderr.log"
PROMPT = "Local inference means"
MAX_NEW_TOKENS = 1


def bytes_to_gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024**3), 2)


def disk_free() -> dict[str, Any]:
    usage = shutil.disk_usage("D:\\")
    return {
        "free_bytes": usage.free,
        "free_gib": bytes_to_gib(usage.free),
        "used_gib": bytes_to_gib(usage.used),
        "total_gib": bytes_to_gib(usage.total),
    }


def dir_size(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "file_count": 0, "size_gib": None}
    total = 0
    count = 0
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            total += item.stat().st_size
    return {"path": str(path), "exists": True, "file_count": count, "size_gib": bytes_to_gib(total)}


def get_snapshot_path() -> Path:
    snapshots = LOCAL_MODEL_ROOT / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(f"Missing local Hugging Face snapshots directory: {snapshots}")
    candidates = [
        path for path in snapshots.iterdir()
        if path.is_dir() and (path / "config.json").exists() and (path / "model.safetensors.index.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No complete local snapshot with config.json and model.safetensors.index.json under {snapshots}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


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
            return {
                "cuda_available": False,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
            }
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


def patch_fingerprint() -> dict[str, Any]:
    patch_file = ROOT / "repo" / "air_llm" / "airllm" / "airllm_base.py"
    text = patch_file.read_text(encoding="utf-8")
    markers = [
        "def get_position_embeddings_args",
        "rotary_emb(seq, position_ids)",
        '"position_embeddings"',
    ]
    return {
        "file": str(patch_file),
        "present": all(marker in text for marker in markers),
        "markers": {marker: marker in text for marker in markers},
    }


def block_network_downloads() -> None:
    import huggingface_hub

    def blocked_snapshot_download(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network download blocked: rerun must use existing local Yi-34B-Chat snapshot and AirLLM shards only.")

    huggingface_hub.snapshot_download = blocked_snapshot_download


def check_local_inputs(snapshot_path: Path) -> list[str]:
    errors: list[str] = []
    required_snapshot_files = [
        snapshot_path / "config.json",
        snapshot_path / "generation_config.json",
        snapshot_path / "model.safetensors.index.json",
        snapshot_path / "tokenizer.model",
        snapshot_path / "tokenizer_config.json",
    ]
    for path in required_snapshot_files:
        if not path.exists():
            errors.append(f"Missing local snapshot file: {path}")
    required_split_files = [
        SPLIT_DIR / "model.embed_tokens.safetensors",
        SPLIT_DIR / "model.layers.0.safetensors",
        SPLIT_DIR / "model.layers.59.safetensors",
        SPLIT_DIR / "model.norm.safetensors",
        SPLIT_DIR / "lm_head.safetensors",
    ]
    for path in required_split_files:
        if not path.exists():
            errors.append(f"Missing AirLLM split shard: {path}")
    return errors


def generated_sequences(output: Any) -> Any:
    if hasattr(output, "sequences"):
        return output.sequences
    return output


def write_receipts(result: dict[str, Any]) -> None:
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    JSON_RECEIPT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "Yi-34B-Chat minimal local-only rerun",
        f"Generated: {result.get('generated_at')}",
        f"Status: {result.get('status')}",
        f"Model id: {result.get('model_id')}",
        f"Local snapshot: {result.get('local_snapshot')}",
        f"Existing shards loaded: {result.get('existing_shards_loaded')}",
        f"Download attempted: {result.get('download_attempted')}",
        f"AirLLM load: {result.get('airllm_load', {}).get('status')}",
        f"Generation: {result.get('generation', {}).get('status')}",
        f"Generated token count: {result.get('generation', {}).get('generated_token_count')}",
        f"Runtime per generated token: {result.get('generation', {}).get('seconds_per_generated_token')}",
        f"Old position_embeddings bug returned: {result.get('old_position_embeddings_bug_returned')}",
        f"Disk free before: {result.get('disk', {}).get('before', {}).get('free_gib')} GiB",
        f"Disk free after: {result.get('disk', {}).get('after', {}).get('free_gib')} GiB",
        "",
        "Output sample:",
        str(result.get("output_sample") or ""),
        "",
        "Timing:",
        json.dumps(result.get("timing"), indent=2, ensure_ascii=False),
        "",
        "GPU:",
        json.dumps(result.get("gpu"), indent=2, ensure_ascii=False),
        "",
        "RAM:",
        json.dumps(result.get("ram"), indent=2, ensure_ascii=False),
        "",
        "Local sizes:",
        json.dumps(result.get("local_sizes"), indent=2, ensure_ascii=False),
        "",
        "Errors:",
        json.dumps(result.get("errors"), indent=2, ensure_ascii=False),
        "",
        "Traceback:",
        str(result.get("traceback") or ""),
    ]
    TEXT_RECEIPT.write_text("\n".join(lines), encoding="utf-8")


def stdout_json_ensure_ascii() -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or ""
    return "utf" not in encoding.lower()


def main() -> int:
    for key, value in {
        "PIP_CACHE_DIR": CACHE / "pip",
        "TEMP": CACHE / "temp",
        "TMP": CACHE / "temp",
        "TORCH_HOME": CACHE / "torch",
        "XDG_CACHE_HOME": CACHE,
        "HF_DATASETS_CACHE": CACHE / "huggingface" / "datasets",
        "HF_HOME": MODELS / "huggingface",
        "HUGGINGFACE_HUB_CACHE": MODELS / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": MODELS / "huggingface" / "transformers",
        "AIRLLM_LOCAL_MODELS": MODELS,
    }.items():
        os.environ[key] = str(value)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    for path in [CACHE / "pip", CACHE / "torch", CACHE / "temp", RECEIPTS]:
        path.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "local_snapshot": None,
        "shard_dir": str(SHARD_DIR),
        "split_dir": str(SPLIT_DIR),
        "download_attempted": False,
        "existing_shards_loaded": False,
        "old_position_embeddings_bug_returned": False,
        "airllm_load": {"status": "not_attempted"},
        "generation": {"status": "not_attempted", "generated_token_count": None, "seconds_per_generated_token": None},
        "output_sample": "",
        "disk": {"before": disk_free(), "after": None, "used_gib": None},
        "gpu": get_gpu_snapshot(),
        "ram": get_ram_snapshot(),
        "local_sizes": {
            "hf_cache": dir_size(LOCAL_MODEL_ROOT),
            "shards": dir_size(SHARD_DIR),
            "split_dir": dir_size(SPLIT_DIR),
        },
        "position_embeddings_patch": patch_fingerprint(),
        "timing": {
            "load_start": None,
            "load_end": None,
            "load_seconds": None,
            "generation_start": None,
            "generation_end": None,
            "generation_seconds": None,
        },
        "cache_paths": {
            "HF_HOME": os.environ["HF_HOME"],
            "HUGGINGFACE_HUB_CACHE": os.environ["HUGGINGFACE_HUB_CACHE"],
            "TRANSFORMERS_CACHE": os.environ["TRANSFORMERS_CACHE"],
            "TORCH_HOME": os.environ["TORCH_HOME"],
            "TEMP": os.environ["TEMP"],
            "TMP": os.environ["TMP"],
            "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
            "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
        },
        "receipts": {
            "text": str(TEXT_RECEIPT),
            "json": str(JSON_RECEIPT),
            "stdout": str(STDOUT_LOG),
            "stderr": str(STDERR_LOG),
        },
        "errors": [],
        "traceback": "",
    }

    exit_code = 1
    try:
        snapshot_path = get_snapshot_path()
        result["local_snapshot"] = str(snapshot_path)
        local_errors = check_local_inputs(snapshot_path)
        if local_errors:
            result["status"] = "FAIL"
            result["errors"].extend(local_errors)
            return 1

        print(f"Yi-34B-Chat minimal rerun using local snapshot: {snapshot_path}")
        print(f"Existing AirLLM split dir: {SPLIT_DIR}")
        print(f"D: free before rerun: {result['disk']['before']['free_gib']} GiB")

        block_network_downloads()

        import torch
        from airllm import AutoModel

        if not torch.cuda.is_available():
            result["status"] = "FAIL"
            result["errors"].append("Torch CUDA is not available.")
            return 1

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        load_start = time.perf_counter()
        result["timing"]["load_start"] = datetime.now(timezone.utc).isoformat()
        model = AutoModel.from_pretrained(
            str(snapshot_path),
            device="cuda:0",
            layer_shards_saving_path=str(SHARD_DIR),
        )
        result["timing"]["load_end"] = datetime.now(timezone.utc).isoformat()
        result["timing"]["load_seconds"] = round(time.perf_counter() - load_start, 3)
        result["airllm_load"] = {"status": "pass", "model_class": type(model).__name__}
        result["existing_shards_loaded"] = True

        encoded = model.tokenizer(
            [PROMPT],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=32,
            padding=False,
        )
        input_ids = encoded["input_ids"].to("cuda:0")
        input_token_count = int(input_ids.shape[-1])

        generation_start = time.perf_counter()
        result["timing"]["generation_start"] = datetime.now(timezone.utc).isoformat()
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=False,
            return_dict_in_generate=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result["timing"]["generation_end"] = datetime.now(timezone.utc).isoformat()
        generation_seconds = round(time.perf_counter() - generation_start, 3)
        result["timing"]["generation_seconds"] = generation_seconds

        sequences = generated_sequences(output)
        generated_token_count = int(sequences.shape[-1] - input_token_count)
        decoded = model.tokenizer.decode(sequences[0], skip_special_tokens=True)
        result["generation"] = {
            "status": "pass" if decoded.strip() else "fail",
            "input_token_count": input_token_count,
            "generated_token_count": generated_token_count,
            "seconds_per_generated_token": round(generation_seconds / max(generated_token_count, 1), 3),
        }
        result["output_sample"] = decoded[:2000]
        result["status"] = "PASS" if decoded.strip() and generated_token_count >= 1 else "FAIL"
        if result["status"] != "PASS":
            result["errors"].append("Generation returned without a non-empty decoded output and at least one generated token.")
        exit_code = 0 if result["status"] == "PASS" else 1
    except Exception as exc:
        tb = traceback.format_exc()
        result["traceback"] = tb
        result["errors"].append(repr(exc))
        result["old_position_embeddings_bug_returned"] = "position_embeddings" in tb or "LlamaDecoderLayer" in tb
        if result["airllm_load"].get("status") == "not_attempted":
            result["airllm_load"] = {"status": "fail", "error": repr(exc)}
        elif result["generation"].get("status") == "not_attempted":
            result["generation"] = {"status": "fail", "error": repr(exc)}
        result["status"] = "FAIL"
        exit_code = 1
    finally:
        result["disk"]["after"] = disk_free()
        result["disk"]["used_gib"] = bytes_to_gib(result["disk"]["before"]["free_bytes"] - result["disk"]["after"]["free_bytes"])
        result["gpu"] = get_gpu_snapshot()
        result["ram"] = get_ram_snapshot()
        result["local_sizes"] = {
            "hf_cache": dir_size(LOCAL_MODEL_ROOT),
            "shards": dir_size(SHARD_DIR),
            "split_dir": dir_size(SPLIT_DIR),
        }
        write_receipts(result)
        print(json.dumps(result, indent=2, ensure_ascii=stdout_json_ensure_ascii()))
        print(f"Wrote {TEXT_RECEIPT}")
        print(f"Wrote {JSON_RECEIPT}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
'@

Set-Content -LiteralPath $Runner -Value $PythonSource -Encoding UTF8
& $VenvPython $Runner
exit $LASTEXITCODE
