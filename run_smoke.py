from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT / "repo"
CACHE = ROOT / "cache"
MODELS = ROOT / "models"


def set_local_env_defaults() -> None:
    os.environ.setdefault("PIP_CACHE_DIR", str(CACHE / "pip"))
    os.environ.setdefault("TEMP", str(CACHE / "temp"))
    os.environ.setdefault("TMP", str(CACHE / "temp"))
    os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
    os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE / "huggingface" / "datasets"))
    os.environ.setdefault("HF_HOME", str(MODELS / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODELS / "huggingface" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS / "huggingface" / "transformers"))
    os.environ.setdefault("AIRLLM_LOCAL_MODELS", str(MODELS))


def find_tiny_local_model() -> Path | None:
    search_roots = [MODELS, REPO]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for config in search_root.rglob("config.json"):
            parent = config.parent
            weight_files = list(parent.glob("*.safetensors")) + list(parent.glob("*.bin"))
            if not weight_files:
                continue
            total_size = sum(path.stat().st_size for path in weight_files)
            if total_size <= 500 * 1024 * 1024:
                return parent
    return None


def main() -> int:
    set_local_env_defaults()

    result: dict[str, object] = {
        "root": str(ROOT),
        "repo": str(REPO),
        "models": str(MODELS),
        "python": sys.version.replace("\n", " "),
        "torch": {"import_ok": False},
        "airllm": {"import_ok": False},
        "tiny_local_model": None,
        "download_performed": False,
    }

    try:
        import torch

        result["torch"] = {
            "import_ok": True,
            "version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        result["torch"] = {"import_ok": False, "error": repr(exc)}

    try:
        import airllm
        from airllm import AirLLMLlama2, AutoModel

        result["airllm"] = {
            "import_ok": True,
            "module_file": getattr(airllm, "__file__", None),
            "automodel_from_pretrained_callable": callable(getattr(AutoModel, "from_pretrained", None)),
            "llama_class": AirLLMLlama2.__name__,
        }
    except Exception as exc:
        result["airllm"] = {"import_ok": False, "error": repr(exc)}

    tiny_model = find_tiny_local_model()
    if tiny_model is not None:
        result["tiny_local_model"] = str(tiny_model)
        result["tiny_model_note"] = (
            "A tiny local model path was found, but model loading is intentionally not run "
            "inside smoke verification because AirLLM may shard and copy weights."
        )
    else:
        result["tiny_model_note"] = (
            "No tiny local model was found in repo or models; skipped model loading to avoid downloads."
        )

    next_command = (
        'powershell -NoProfile -ExecutionPolicy Bypass -Command '
        '"cd D:\\AI\\airllm; '
        '$env:HF_HOME=\'D:\\AI\\airllm\\models\\huggingface\'; '
        '$env:HUGGINGFACE_HUB_CACHE=\'D:\\AI\\airllm\\models\\huggingface\\hub\'; '
        '$env:TRANSFORMERS_CACHE=\'D:\\AI\\airllm\\models\\huggingface\\transformers\'; '
        '$env:TORCH_HOME=\'D:\\AI\\airllm\\cache\\torch\'; '
        '.\\.venv\\Scripts\\python.exe -c \\"'
        "from airllm import AutoModel; import torch; "
        "model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0'; "
        "device='cuda:0' if torch.cuda.is_available() else 'cpu'; "
        "model=AutoModel.from_pretrained(model_id, device=device, "
        "layer_shards_saving_path=r'D:\\AI\\airllm\\models\\TinyLlama-1.1B-Chat-v1.0\\airllm_shards'); "
        "print('Loaded', model_id, 'on', device)"
        '\\""'
    )
    result["next_real_model_command"] = next_command

    print(json.dumps(result, indent=2))
    print()
    print("Smoke test did not download or load a model.")
    print("Next real-model command:")
    print(next_command)

    torch_ok = bool(result["torch"].get("import_ok")) if isinstance(result["torch"], dict) else False
    airllm_ok = bool(result["airllm"].get("import_ok")) if isinstance(result["airllm"], dict) else False
    return 0 if torch_ok and airllm_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
