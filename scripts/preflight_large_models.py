from __future__ import annotations

import json
import os
import shutil
import sys
import textwrap
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
MODELS = ROOT / "models"
RECEIPTS = ROOT / "receipts"

MODELS_TO_CHECK = [
    "stepfun-ai/Step-3.5-Flash",
    "moonshotai/Kimi-K2.6",
]

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

WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
)
WEIGHT_EXCLUDE_SUFFIXES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)


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


def compact_value(value: Any, max_items: int = 20) -> Any:
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


def get_file_size(file_info: Any) -> int | None:
    for attr in ("size", "blob_size"):
        value = getattr(file_info, attr, None)
        if isinstance(value, int):
            return value
    return None


def is_weight_file(name: str) -> bool:
    lower = name.lower()
    if lower.endswith(WEIGHT_EXCLUDE_SUFFIXES):
        return False
    return lower.endswith(WEIGHT_SUFFIXES)


def get_sibling_name(file_info: Any) -> str:
    return getattr(file_info, "rfilename", None) or getattr(file_info, "path", None) or str(file_info)


def get_safetensors_hint(model_info: Any) -> dict[str, Any] | None:
    safetensors = getattr(model_info, "safetensors", None)
    if safetensors is None:
        return None
    if hasattr(safetensors, "__dict__"):
        return compact_value(vars(safetensors))
    return compact_value(safetensors)


def normalize_config_dict(config: Any) -> dict[str, Any]:
    try:
        return config.to_dict()
    except Exception:
        return {}


def determine_airllm_support(architectures: list[str], model_type: str | None) -> tuple[str, str, str | None]:
    for architecture in architectures:
        for marker, airllm_class in SUPPORTED_ARCH_MARKERS.items():
            if marker in architecture:
                return (
                    "yes",
                    f"Architecture {architecture!r} matches AirLLM marker {marker!r}.",
                    airllm_class,
                )

    normalized_model_type = (model_type or "").lower()
    for marker, airllm_class in SUPPORTED_MODEL_TYPES.items():
        if normalized_model_type == marker:
            return (
                "uncertain",
                f"Model type {model_type!r} is known, but architecture did not directly match AirLLM's AutoModel markers.",
                airllm_class,
            )

    if architectures:
        return (
            "no",
            "Architecture is not in AirLLM 2.11 AutoModel's supported marker list; AirLLM would likely fall back incorrectly to Llama.",
            None,
        )

    return (
        "uncertain",
        "No architecture class was available from config metadata.",
        None,
    )


def get_d_drive_free() -> dict[str, Any]:
    usage = shutil.disk_usage("D:\\")
    return {
        "free_bytes": usage.free,
        "free_gib": bytes_to_gib(usage.free),
        "total_gib": bytes_to_gib(usage.total),
    }


@dataclass
class PreflightPaths:
    json: str
    text: str


def preflight_model(model_id: str) -> dict[str, Any]:
    from huggingface_hub import HfApi
    from transformers import AutoConfig

    result: dict[str, Any] = {
        "model_id": model_id,
        "metadata_query": {"status": "not_started"},
        "config_load": {"status": "not_started"},
        "config": None,
        "architectures": [],
        "model_type": None,
        "parameter_hints": {},
        "storage_hints": {},
        "airllm_support": {
            "compatible": "uncertain",
            "reason": "not evaluated",
            "mapped_class": None,
        },
        "blocking_reasons": [],
        "recommendation_before_download": "Do not download until preflight completes.",
    }

    api = HfApi()
    model_info = None
    try:
        model_info = api.model_info(
            repo_id=model_id,
            files_metadata=True,
            securityStatus=True,
        )
        result["metadata_query"] = {
            "status": "pass",
            "sha": getattr(model_info, "sha", None),
            "last_modified": str(getattr(model_info, "lastModified", None)),
            "private": getattr(model_info, "private", None),
            "gated": getattr(model_info, "gated", None),
            "tags": list(getattr(model_info, "tags", None) or [])[:50],
            "pipeline_tag": getattr(model_info, "pipeline_tag", None),
            "library_name": getattr(model_info, "library_name", None),
        }

        siblings = list(getattr(model_info, "siblings", None) or [])
        files = []
        weight_files = []
        total_known_weight_bytes = 0
        unknown_weight_file_count = 0
        for sibling in siblings:
            name = get_sibling_name(sibling)
            size = get_file_size(sibling)
            file_record = {
                "name": name,
                "size_bytes": size,
                "size_gib": bytes_to_gib(size),
            }
            files.append(file_record)
            if is_weight_file(name):
                weight_files.append(file_record)
                if size is None:
                    unknown_weight_file_count += 1
                else:
                    total_known_weight_bytes += size

        safetensors_hint = get_safetensors_hint(model_info)
        result["parameter_hints"] = {
            "safetensors": safetensors_hint,
            "card_data_model_size": compact_value(getattr(model_info, "cardData", None)),
        }
        result["storage_hints"] = {
            "file_count": len(files),
            "weight_file_count": len(weight_files),
            "unknown_weight_file_count": unknown_weight_file_count,
            "known_weight_bytes": total_known_weight_bytes,
            "known_weight_gib": bytes_to_gib(total_known_weight_bytes),
            "estimated_download_gib": bytes_to_gib(total_known_weight_bytes),
            "estimated_airllm_shard_gib": bytes_to_gib(total_known_weight_bytes),
            "estimated_download_plus_shards_gib": bytes_to_gib(total_known_weight_bytes * 2),
            "largest_files": sorted(weight_files, key=lambda item: item.get("size_bytes") or 0, reverse=True)[:20],
        }
    except Exception as exc:
        result["metadata_query"] = {
            "status": "fail",
            "error": repr(exc),
        }

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config_dict = normalize_config_dict(config)
        architectures = list(config_dict.get("architectures") or [])
        model_type = config_dict.get("model_type")
        support_status, support_reason, mapped_class = determine_airllm_support(architectures, model_type)

        result["config_load"] = {
            "status": "pass",
            "config_class": type(config).__name__,
            "module": type(config).__module__,
        }
        result["config"] = compact_value(config_dict, max_items=80)
        result["architectures"] = architectures
        result["model_type"] = model_type
        result["airllm_support"] = {
            "compatible": support_status,
            "reason": support_reason,
            "mapped_class": mapped_class,
            "airllm_supported_markers": list(SUPPORTED_ARCH_MARKERS.keys()),
        }
    except Exception as exc:
        result["config_load"] = {
            "status": "fail",
            "error": repr(exc),
        }
        result["airllm_support"] = {
            "compatible": "no",
            "reason": "Transformers AutoConfig could not load with trust_remote_code=True in the verified AirLLM environment.",
            "mapped_class": None,
            "airllm_supported_markers": list(SUPPORTED_ARCH_MARKERS.keys()),
        }

    free = get_d_drive_free()
    result["d_drive_free"] = free
    estimated_needed = result["storage_hints"].get("estimated_download_plus_shards_gib")
    compatible = result["airllm_support"]["compatible"]
    blocking_reasons: list[str] = []
    if result["metadata_query"]["status"] != "pass":
        blocking_reasons.append("metadata query failed")
    if result["config_load"]["status"] != "pass":
        blocking_reasons.append("config cannot be loaded by this local Transformers/AirLLM environment")
    if compatible == "no":
        blocking_reasons.append("AirLLM AutoModel does not appear to support this architecture")
    if estimated_needed is not None and estimated_needed > free["free_gib"]:
        blocking_reasons.append("estimated original weights plus AirLLM shards exceed free D: space")
    if compatible == "uncertain":
        blocking_reasons.append("AirLLM support is uncertain and should be reviewed manually")

    result["blocking_reasons"] = blocking_reasons
    if blocking_reasons:
        result["recommendation_before_download"] = "Do not download: " + "; ".join(blocking_reasons) + "."
    else:
        result["recommendation_before_download"] = "Preflight passed for metadata/config/support; download may be attempted locally if you accept the disk/time cost."

    return result


def write_text_report(results: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("AirLLM large-model preflight")
    lines.append(f"Generated: {results['generated_at']}")
    lines.append(f"Root: {ROOT}")
    lines.append(f"D: free: {results['d_drive_free']['free_gib']} GiB")
    lines.append("")

    for model in results["models"]:
        lines.append("=" * 88)
        lines.append(model["model_id"])
        lines.append("=" * 88)
        lines.append(f"Metadata query: {model['metadata_query']['status']}")
        if model["metadata_query"]["status"] == "fail":
            lines.append(f"Metadata error: {model['metadata_query'].get('error')}")
        lines.append(f"Config load: {model['config_load']['status']}")
        if model["config_load"]["status"] == "fail":
            lines.append(f"Config error: {model['config_load'].get('error')}")
        lines.append(f"Architecture class names: {model.get('architectures')}")
        lines.append(f"Model type: {model.get('model_type')}")
        lines.append(f"AirLLM compatible: {model['airllm_support']['compatible']}")
        lines.append(f"AirLLM reason: {model['airllm_support']['reason']}")
        lines.append(f"Mapped AirLLM class: {model['airllm_support'].get('mapped_class')}")
        storage = model["storage_hints"]
        lines.append(f"Weight files: {storage.get('weight_file_count')} known, {storage.get('unknown_weight_file_count')} unknown sizes")
        lines.append(f"Estimated download: {storage.get('estimated_download_gib')} GiB")
        lines.append(f"Estimated AirLLM shards: {storage.get('estimated_airllm_shard_gib')} GiB")
        lines.append(f"Estimated download + shards: {storage.get('estimated_download_plus_shards_gib')} GiB")
        lines.append(f"Recommendation: {model['recommendation_before_download']}")
        lines.append("")
        lines.append("Parameter/storage hints:")
        lines.append(json.dumps(model.get("parameter_hints"), indent=2, ensure_ascii=False))
        lines.append("")
        lines.append("Config preview:")
        lines.append(textwrap.indent(json.dumps(model.get("config"), indent=2, ensure_ascii=False), "  "))
        lines.append("")
        lines.append("Largest weight-like files:")
        lines.append(textwrap.indent(json.dumps(storage.get("largest_files"), indent=2, ensure_ascii=False), "  "))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    set_local_environment()

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "cache_paths": {
            "HF_HOME": os.environ["HF_HOME"],
            "HUGGINGFACE_HUB_CACHE": os.environ["HUGGINGFACE_HUB_CACHE"],
            "TRANSFORMERS_CACHE": os.environ["TRANSFORMERS_CACHE"],
            "TORCH_HOME": os.environ["TORCH_HOME"],
            "TEMP": os.environ["TEMP"],
            "TMP": os.environ["TMP"],
        },
        "d_drive_free": get_d_drive_free(),
        "models": [],
        "receipts": {},
    }

    print("AirLLM large-model preflight")
    print(f"Root: {ROOT}")
    print(f"D: free before large model tests: {results['d_drive_free']['free_gib']} GiB")
    print("No model weights will be downloaded by this script.")
    print()

    for model_id in MODELS_TO_CHECK:
        print(f"Checking {model_id} ...")
        model_result = preflight_model(model_id)
        results["models"].append(model_result)
        print(f"  metadata: {model_result['metadata_query']['status']}")
        print(f"  config: {model_result['config_load']['status']}")
        print(f"  architectures: {model_result.get('architectures')}")
        print(f"  model_type: {model_result.get('model_type')}")
        print(f"  estimated download GiB: {model_result['storage_hints'].get('estimated_download_gib')}")
        print(f"  estimated shards GiB: {model_result['storage_hints'].get('estimated_airllm_shard_gib')}")
        print(f"  AirLLM compatible: {model_result['airllm_support']['compatible']}")
        print(f"  recommendation: {model_result['recommendation_before_download']}")
        print()

    json_path = RECEIPTS / "large_model_preflight.json"
    text_path = RECEIPTS / "large_model_preflight.txt"
    results["receipts"] = {
        "json": str(json_path),
        "text": str(text_path),
    }

    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_text_report(results, text_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
