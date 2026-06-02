$ErrorActionPreference = "Stop"

$Root = "D:\AI\airllm"
$Receipts = Join-Path $Root "receipts"
$TempDir = Join-Path $Root "cache\temp"
$StdoutLog = Join-Path $Receipts "yi34b_generation_rerun_stdout.log"
$StderrLog = Join-Path $Receipts "yi34b_generation_rerun_stderr.log"
$Runner = Join-Path $TempDir "rerun_yi34b_generation_runner.py"

New-Item -ItemType Directory -Force -Path $Receipts | Out-Null
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "cache\torch") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "models\huggingface\hub") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "models\huggingface\transformers") | Out-Null

$env:HF_HOME = Join-Path $Root "models\huggingface"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $Root "models\huggingface\hub"
$env:TRANSFORMERS_CACHE = Join-Path $Root "models\huggingface\transformers"
$env:TORCH_HOME = Join-Path $Root "cache\torch"
$env:TEMP = $TempDir
$env:TMP = $TempDir
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"

& (Join-Path $Root ".venv\Scripts\Activate.ps1")

$PythonCode = @'
import gc
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"D:\AI\airllm")
RECEIPTS = ROOT / "receipts"
MODEL_ID = "NousResearch/Nous-Hermes-2-Yi-34B"
MODEL_SNAPSHOT = (
    ROOT
    / "models"
    / "huggingface"
    / "hub"
    / "models--NousResearch--Nous-Hermes-2-Yi-34B"
    / "snapshots"
    / "fcb0a8847e76aea14aba9aa44009d4418ad7c18f"
)
SHARD_DIR = ROOT / "models" / "NousResearch_Nous-Hermes-2-Yi-34B" / "airllm_shards"
PROMPT = "Explain local LLM inference in one sentence."
TEXT_RECEIPT = RECEIPTS / "yi34b_generation_rerun.txt"
JSON_RECEIPT = RECEIPTS / "yi34b_generation_rerun.json"

def gib(value):
    return round(float(value) / (1024 ** 3), 2)

def disk_info(path):
    total, used, free = shutil.disk_usage(path)
    return {"total_gib": gib(total), "used_gib": gib(used), "free_gib": gib(free)}

def tensor_info(value):
    if value is None:
        return None
    return {
        "type": type(value).__name__,
        "shape": list(value.shape) if hasattr(value, "shape") else None,
        "dtype": str(value.dtype) if hasattr(value, "dtype") else None,
        "device": str(value.device) if hasattr(value, "device") else None,
    }

def ram_info():
    try:
        import psutil
        proc = psutil.Process()
        vm = psutil.virtual_memory()
        return {
            "process_rss_gib": gib(proc.memory_info().rss),
            "system_used_gib": gib(vm.used),
            "system_available_gib": gib(vm.available),
            "system_total_gib": gib(vm.total),
        }
    except Exception as exc:
        return {"error": repr(exc)}

def gpu_info(torch):
    data = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        data["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        data["memory_allocated_gib"] = gib(torch.cuda.memory_allocated())
        data["memory_reserved_gib"] = gib(torch.cuda.memory_reserved())
        data["max_memory_allocated_gib"] = gib(torch.cuda.max_memory_allocated())
        data["max_memory_reserved_gib"] = gib(torch.cuda.max_memory_reserved())
    return data

def write_receipts(result):
    JSON_RECEIPT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    lines = [
        "AirLLM Yi-34B generation rerun",
        f"Generated: {result.get('generated_at')}",
        f"Status: {result.get('status')}",
        f"Model: {MODEL_ID}",
        f"Snapshot: {MODEL_SNAPSHOT}",
        f"Shard dir: {SHARD_DIR}",
        f"Prompt: {PROMPT}",
        f"Output sample: {result.get('output_sample', '')}",
        "",
        json.dumps(result, indent=2, default=str),
    ]
    TEXT_RECEIPT.write_text("\n".join(lines), encoding="utf-8")

def main():
    for key, value in {
        "HF_HOME": ROOT / "models" / "huggingface",
        "HUGGINGFACE_HUB_CACHE": ROOT / "models" / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": ROOT / "models" / "huggingface" / "transformers",
        "TORCH_HOME": ROOT / "cache" / "torch",
        "TEMP": ROOT / "cache" / "temp",
        "TMP": ROOT / "cache" / "temp",
    }.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "model_id": MODEL_ID,
        "model_snapshot": str(MODEL_SNAPSHOT),
        "shard_dir": str(SHARD_DIR),
        "prompt": PROMPT,
        "max_new_tokens": 16,
        "status": "running",
        "download_attempted": False,
        "airllm_load": {"status": "not attempted"},
        "generation": {"status": "not attempted"},
        "output_sample": "",
        "errors": [],
        "traceback": "",
        "disk": {"before": disk_info(ROOT), "after": None},
        "timing": {},
    }
    write_receipts(result)

    try:
        import torch
        from airllm import AutoModel

        result["gpu_before"] = gpu_info(torch)
        result["ram_before"] = ram_info()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        load_start = time.perf_counter()
        result["timing"]["load_start"] = datetime.now(timezone.utc).isoformat()
        model = AutoModel.from_pretrained(
            str(MODEL_SNAPSHOT),
            layer_shards_saving_path=str(SHARD_DIR),
            profiling_mode=False,
            prefetching=True,
        )
        result["timing"]["load_end"] = datetime.now(timezone.utc).isoformat()
        result["timing"]["load_seconds"] = round(time.perf_counter() - load_start, 3)
        result["airllm_load"] = {
            "status": "pass",
            "model_class": type(model).__name__,
            "inner_model_class": type(getattr(model, "model", None)).__name__,
        }
        write_receipts(result)

        input_tokens = model.tokenizer(
            [PROMPT],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=128,
            padding=True,
        )
        input_ids = input_tokens["input_ids"].to(model.device)

        gen_start = time.perf_counter()
        result["timing"]["generation_start"] = datetime.now(timezone.utc).isoformat()
        output = model.generate(
            input_ids,
            max_new_tokens=16,
            do_sample=False,
            use_cache=False,
            return_dict_in_generate=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result["timing"]["generation_end"] = datetime.now(timezone.utc).isoformat()
        result["timing"]["generation_seconds"] = round(time.perf_counter() - gen_start, 3)

        full_text = model.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        new_text = model.tokenizer.decode(output.sequences[0][input_ids.shape[-1]:], skip_special_tokens=True)
        result["generation"] = {
            "status": "pass",
            "sequences": tensor_info(output.sequences),
            "new_text": new_text,
        }
        result["output_sample"] = full_text
        result["status"] = "PASS" if new_text.strip() else "FAIL"
        if not new_text.strip():
            result["errors"].append("Generation returned no new decoded text.")
    except Exception as exc:
        result["status"] = "FAIL"
        if result["airllm_load"]["status"] == "not attempted":
            result["airllm_load"]["status"] = "fail"
        elif result["generation"]["status"] == "not attempted":
            result["generation"]["status"] = "fail"
        result["errors"].append(repr(exc))
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], file=sys.stderr)
    finally:
        try:
            import torch
            result["gpu_after"] = gpu_info(torch)
        except Exception as exc:
            result["gpu_after"] = {"error": repr(exc)}
        result["ram_after"] = ram_info()
        result["disk"]["after"] = disk_info(ROOT)
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_receipts(result)
        gc.collect()

    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
'@

Set-Content -LiteralPath $Runner -Value $PythonCode -Encoding UTF8
& (Join-Path $Root ".venv\Scripts\python.exe") $Runner > $StdoutLog 2> $StderrLog
$ExitCode = $LASTEXITCODE

Get-Content -LiteralPath $StdoutLog
if ($ExitCode -ne 0) {
    Get-Content -LiteralPath $StderrLog
    exit $ExitCode
}
