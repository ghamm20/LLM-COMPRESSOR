from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
MODELS = ROOT / "models"
RECEIPTS = ROOT / "receipts"


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


def safe_model_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id.replace("/", "__")).strip("_")


def bytes_to_gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024**3), 2)


def d_drive_free() -> dict[str, Any]:
    usage = shutil.disk_usage("D:\\")
    return {
        "free_bytes": usage.free,
        "free_gib": bytes_to_gib(usage.free),
        "total_gib": bytes_to_gib(usage.total),
    }


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


def get_gpu_snapshot(torch_module: Any) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "device_count": torch_module.cuda.device_count(),
        "devices": [torch_module.cuda.get_device_name(i) for i in range(torch_module.cuda.device_count())],
        "memory_allocated_gib": bytes_to_gib(torch_module.cuda.memory_allocated()),
        "memory_reserved_gib": bytes_to_gib(torch_module.cuda.memory_reserved()),
        "max_memory_allocated_gib": bytes_to_gib(torch_module.cuda.max_memory_allocated()),
        "max_memory_reserved_gib": bytes_to_gib(torch_module.cuda.max_memory_reserved()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny local AirLLM inference test.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--prompt", default="Give a one paragraph explanation of local LLM inference.")
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def write_receipts(result: dict[str, Any], text_path: Path, json_path: Path) -> None:
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "AirLLM large-model local inference test",
        f"Generated: {result.get('generated_at')}",
        f"Model: {result.get('model_id')}",
        f"Status: {result.get('status')}",
        f"Root: {ROOT}",
        f"Shard dir: {result.get('shard_dir')}",
        f"Download attempted: {result.get('download_attempted')}",
        f"AirLLM load: {result.get('airllm_load', {}).get('status')}",
        f"Generation: {result.get('generation', {}).get('status')}",
        f"Disk before: {result.get('disk', {}).get('before', {}).get('free_gib')} GiB",
        f"Disk after: {result.get('disk', {}).get('after', {}).get('free_gib')} GiB",
        f"Disk used: {result.get('disk', {}).get('used_gib')} GiB",
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
    text_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_local_environment()

    safe_name = safe_model_name(args.model_id)
    text_path = RECEIPTS / f"large_model_test_{safe_name}.txt"
    json_path = RECEIPTS / f"large_model_test_{safe_name}.json"
    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "model_id": args.model_id,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "shard_dir": str(shard_dir),
        "offline": bool(args.offline),
        "download_attempted": not bool(args.offline),
        "status": "started",
        "cache_paths": {
            "HF_HOME": os.environ["HF_HOME"],
            "HUGGINGFACE_HUB_CACHE": os.environ["HUGGINGFACE_HUB_CACHE"],
            "TRANSFORMERS_CACHE": os.environ["TRANSFORMERS_CACHE"],
            "TORCH_HOME": os.environ["TORCH_HOME"],
            "TEMP": os.environ["TEMP"],
            "TMP": os.environ["TMP"],
        },
        "disk": {"before": d_drive_free(), "after": None, "used_gib": None},
        "airllm_load": {"status": "not_started"},
        "generation": {"status": "not_started"},
        "timing": {
            "load_seconds": None,
            "time_to_first_token_seconds": None,
            "time_to_first_token_note": "AirLLM generate is synchronous here; exact streaming TTFT is not available.",
            "generation_seconds": None,
        },
        "gpu": None,
        "ram": None,
        "output_sample": None,
        "errors": [],
        "receipts": {"text": str(text_path), "json": str(json_path)},
    }

    exit_code = 1
    try:
        import torch
        from airllm import AutoModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        result["device"] = device
        result["gpu"] = get_gpu_snapshot(torch)
        result["ram"] = get_ram_snapshot()

        load_kwargs: dict[str, Any] = {
            "device": device,
            "layer_shards_saving_path": str(shard_dir),
        }
        if args.offline:
            load_kwargs["local_files_only"] = True

        start = time.perf_counter()
        model = AutoModel.from_pretrained(args.model_id, **load_kwargs)
        result["timing"]["load_seconds"] = round(time.perf_counter() - start, 3)
        result["airllm_load"] = {"status": "pass"}

        input_tokens = model.tokenizer(
            [args.prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=128,
            padding=False,
        )
        input_ids = input_tokens["input_ids"].to(device)

        generation_start = time.perf_counter()
        generation_output = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
        )
        result["timing"]["generation_seconds"] = round(time.perf_counter() - generation_start, 3)
        result["generation"] = {"status": "pass"}

        output = model.tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
        result["output_sample"] = output[:2000]
        result["status"] = "pass"
        exit_code = 0
    except Exception as exc:
        result["errors"].append(repr(exc))
        if result["airllm_load"]["status"] == "not_started":
            result["airllm_load"] = {"status": "fail", "error": repr(exc)}
        elif result["generation"]["status"] == "not_started":
            result["generation"] = {"status": "fail", "error": repr(exc)}
        result["status"] = "fail"
    finally:
        try:
            import torch

            result["gpu"] = get_gpu_snapshot(torch)
        except Exception as exc:
            result["gpu"] = {"error": repr(exc)}
        result["ram"] = get_ram_snapshot()
        result["disk"]["after"] = d_drive_free()
        before = result["disk"]["before"]["free_bytes"]
        after = result["disk"]["after"]["free_bytes"]
        result["disk"]["used_gib"] = bytes_to_gib(before - after)
        write_receipts(result, text_path, json_path)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Wrote {text_path}")
        print(f"Wrote {json_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
