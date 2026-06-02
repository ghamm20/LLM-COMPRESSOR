import gc
import importlib.metadata
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

TEXT_RECEIPT = RECEIPTS / "yi34b_debug_generation.txt"
JSON_RECEIPT = RECEIPTS / "yi34b_debug_generation.json"
STDOUT_LOG = RECEIPTS / "yi34b_debug_generation_stdout.log"
STDERR_LOG = RECEIPTS / "yi34b_debug_generation_stderr.log"


def set_local_environment():
    paths = {
        "HF_HOME": ROOT / "models" / "huggingface",
        "HUGGINGFACE_HUB_CACHE": ROOT / "models" / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": ROOT / "models" / "huggingface" / "transformers",
        "TORCH_HOME": ROOT / "cache" / "torch",
        "TEMP": ROOT / "cache" / "temp",
        "TMP": ROOT / "cache" / "temp",
        "HF_DATASETS_CACHE": ROOT / "cache" / "huggingface" / "datasets",
        "PIP_CACHE_DIR": ROOT / "cache" / "pip",
    }
    for key, value in paths.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


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


def write_receipts(results):
    JSON_RECEIPT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    lines = [
        "AirLLM Yi-34B forensic generation debug",
        f"Generated: {results.get('generated_at')}",
        f"Model: {MODEL_ID}",
        f"Snapshot: {MODEL_SNAPSHOT}",
        f"Shard dir: {SHARD_DIR}",
        "",
    ]
    for key, value in results.get("checks", {}).items():
        lines.append(f"[{key}] {value.get('status')}")
        if value.get("summary"):
            lines.append(str(value["summary"]))
        if value.get("data") is not None:
            lines.append(json.dumps(value["data"], indent=2, default=str))
        if value.get("error"):
            lines.append(f"ERROR: {value['error']}")
        if value.get("traceback"):
            lines.append(value["traceback"])
        lines.append("")
    TEXT_RECEIPT.write_text("\n".join(lines), encoding="utf-8")


def run_check(results, key, func):
    print(f"\n=== {key} ===")
    entry = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": None,
        "data": None,
        "summary": "",
        "error": "",
        "traceback": "",
    }
    results["checks"][key] = entry
    write_receipts(results)
    start = time.perf_counter()
    try:
        data = func()
        entry["status"] = "pass"
        entry["data"] = data
        entry["summary"] = data.get("summary", "") if isinstance(data, dict) else ""
        print(json.dumps(data, indent=2, default=str))
    except Exception as exc:
        entry["status"] = "fail"
        entry["error"] = repr(exc)
        entry["traceback"] = traceback.format_exc()
        print(entry["traceback"], file=sys.stderr)
    finally:
        entry["finished_at"] = datetime.now(timezone.utc).isoformat()
        entry["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        write_receipts(results)


def main():
    set_local_environment()
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    with STDOUT_LOG.open("w", encoding="utf-8") as stdout_file, STDERR_LOG.open("w", encoding="utf-8") as stderr_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, stdout_file)
        sys.stderr = Tee(original_stderr, stderr_file)
        try:
            results = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "root": str(ROOT),
                "model_id": MODEL_ID,
                "model_snapshot": str(MODEL_SNAPSHOT),
                "shard_dir": str(SHARD_DIR),
                "prompt": PROMPT,
                "checks": {},
            }
            context = {}

            def environment_check():
                import airllm
                import optimum
                import torch
                import transformers

                data = {
                    "python": sys.version.replace("\n", " "),
                    "torch": torch.__version__,
                    "torch_cuda_version": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(),
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "transformers": transformers.__version__,
                    "optimum": getattr(optimum, "__version__", importlib.metadata.version("optimum")),
                    "airllm_import_path": str(Path(airllm.__file__).resolve()),
                    "disk_d": disk_info(ROOT),
                    "env": {key: os.environ.get(key) for key in [
                        "HF_HOME",
                        "HUGGINGFACE_HUB_CACHE",
                        "TRANSFORMERS_CACHE",
                        "TORCH_HOME",
                        "TEMP",
                        "TMP",
                        "HF_HUB_OFFLINE",
                        "TRANSFORMERS_OFFLINE",
                    ]},
                    "snapshot_exists": MODEL_SNAPSHOT.exists(),
                    "shards_exist": (SHARD_DIR / "splitted_model").exists(),
                    "summary": "Environment and local cache paths recorded.",
                }
                return data

            def tokenizer_check():
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    str(MODEL_SNAPSHOT),
                    trust_remote_code=True,
                    local_files_only=True,
                )
                encoded = tokenizer(PROMPT, return_tensors="pt", return_attention_mask=True)
                context["tokenizer"] = tokenizer
                context["encoded"] = encoded
                return {
                    "tokenizer_class": type(tokenizer).__name__,
                    "tokenizer_module": type(tokenizer).__module__,
                    "eos_token": tokenizer.eos_token,
                    "eos_token_id": tokenizer.eos_token_id,
                    "bos_token": tokenizer.bos_token,
                    "bos_token_id": tokenizer.bos_token_id,
                    "pad_token": tokenizer.pad_token,
                    "pad_token_id": tokenizer.pad_token_id,
                    "input_ids": tensor_info(encoded["input_ids"]),
                    "attention_mask": tensor_info(encoded.get("attention_mask")),
                    "tokens": encoded["input_ids"][0].tolist(),
                    "summary": "Tokenizer loaded from local snapshot and prompt tokenized.",
                }

            def install_attention_probe():
                import transformers.models.llama.modeling_llama as modeling_llama

                original_forward = modeling_llama.LlamaAttention.forward
                probe_state = {"calls": 0, "first_none_call": None}

                def probed_forward(self, *args, **kwargs):
                    probe_state["calls"] += 1
                    if probe_state["first_none_call"] is None and kwargs.get("position_embeddings") is None:
                        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
                        attention_mask = kwargs.get("attention_mask")
                        position_ids = kwargs.get("position_ids")
                        cache_position = kwargs.get("cache_position")
                        probe_state["first_none_call"] = {
                            "layer_idx": getattr(self, "layer_idx", None),
                            "hidden_states": tensor_info(hidden_states),
                            "attention_mask": tensor_info(attention_mask),
                            "position_ids": tensor_info(position_ids),
                            "cache_position": tensor_info(cache_position),
                            "use_cache": kwargs.get("use_cache"),
                            "position_embeddings": None,
                        }
                        print(
                            "ATTENTION_PROBE position_embeddings=None "
                            + json.dumps(probe_state["first_none_call"], default=str),
                            file=sys.stderr,
                        )
                    return original_forward(self, *args, **kwargs)

                modeling_llama.LlamaAttention.forward = probed_forward
                context["attention_probe"] = probe_state
                return {
                    "patched_runtime_only": "transformers.models.llama.modeling_llama.LlamaAttention.forward",
                    "summary": "Runtime-only attention probe installed; no files changed.",
                }

            def airllm_load_check():
                from airllm import AutoModel

                model = AutoModel.from_pretrained(
                    str(MODEL_SNAPSHOT),
                    layer_shards_saving_path=str(SHARD_DIR),
                    profiling_mode=False,
                    prefetching=True,
                )
                context["model"] = model
                return {
                    "model_class": type(model).__name__,
                    "model_module": type(model).__module__,
                    "has_generate": hasattr(model, "generate"),
                    "has_forward": hasattr(model, "forward"),
                    "has_model": hasattr(model, "model"),
                    "has_tokenizer": hasattr(model, "tokenizer"),
                    "has_generation_config": hasattr(model, "generation_config"),
                    "inner_model_class": type(getattr(model, "model", None)).__name__,
                    "device": str(getattr(model, "device", "")),
                    "running_device": str(getattr(model, "running_device", "")),
                    "attention_probe": context.get("attention_probe"),
                    "summary": "AirLLM model loaded from local snapshot and existing shards.",
                }

            def get_model_and_encoded():
                if "model" not in context:
                    raise RuntimeError("AirLLM model was not loaded.")
                if "encoded" not in context:
                    raise RuntimeError("Tokenizer output is unavailable.")
                return context["model"], context["encoded"]

            def sync_cuda():
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            def forward_raw():
                model, encoded = get_model_and_encoded()
                sync_cuda()
                output = model(encoded["input_ids"], return_dict=True)
                sync_cuda()
                return {
                    "logits": tensor_info(output.logits),
                    "past_key_values": type(output.past_key_values).__name__ if output.past_key_values is not None else None,
                    "attention_probe": context.get("attention_probe"),
                    "summary": "Raw AirLLM forward call completed.",
                }

            def forward_use_cache_false():
                model, encoded = get_model_and_encoded()
                sync_cuda()
                output = model(input_ids=encoded["input_ids"], use_cache=False, return_dict=True)
                sync_cuda()
                return {
                    "logits": tensor_info(output.logits),
                    "past_key_values": type(output.past_key_values).__name__ if output.past_key_values is not None else None,
                    "attention_probe": context.get("attention_probe"),
                    "summary": "AirLLM forward with use_cache=False completed.",
                }

            def forward_attention_mask():
                model, encoded = get_model_and_encoded()
                sync_cuda()
                output = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded.get("attention_mask"),
                    use_cache=False,
                    return_dict=True,
                )
                sync_cuda()
                return {
                    "logits": tensor_info(output.logits),
                    "past_key_values": type(output.past_key_values).__name__ if output.past_key_values is not None else None,
                    "attention_probe": context.get("attention_probe"),
                    "summary": "AirLLM forward with attention_mask completed.",
                }

            def generate_one_token():
                model, encoded = get_model_and_encoded()
                input_ids = encoded["input_ids"].to(model.device)
                sync_cuda()
                output = model.generate(
                    input_ids,
                    max_new_tokens=1,
                    do_sample=False,
                    use_cache=False,
                    return_dict_in_generate=True,
                )
                sync_cuda()
                text = model.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
                return {
                    "sequences": tensor_info(output.sequences),
                    "decoded": text,
                    "attention_probe": context.get("attention_probe"),
                    "summary": "AirLLM generation with max_new_tokens=1 completed.",
                }

            def generate_four_tokens():
                model, encoded = get_model_and_encoded()
                input_ids = encoded["input_ids"].to(model.device)
                sync_cuda()
                output = model.generate(
                    input_ids,
                    max_new_tokens=4,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
                sync_cuda()
                text = model.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
                return {
                    "sequences": tensor_info(output.sequences),
                    "decoded": text,
                    "attention_probe": context.get("attention_probe"),
                    "summary": "AirLLM generation with max_new_tokens=4 completed.",
                }

            def generate_readme_style():
                model, _encoded = get_model_and_encoded()
                input_tokens = model.tokenizer(
                    [PROMPT],
                    return_tensors="pt",
                    return_attention_mask=False,
                    truncation=True,
                    max_length=128,
                    padding=True,
                )
                input_ids = input_tokens["input_ids"].to(model.device)
                sync_cuda()
                output = model.generate(
                    input_ids,
                    max_new_tokens=4,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
                sync_cuda()
                text = model.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
                return {
                    "input_ids": tensor_info(input_ids),
                    "sequences": tensor_info(output.sequences),
                    "decoded": text,
                    "attention_probe": context.get("attention_probe"),
                    "summary": "AirLLM README/example generation style completed.",
                }

            run_check(results, "A_environment", environment_check)
            run_check(results, "B_tokenizer", tokenizer_check)
            run_check(results, "C_attention_probe", install_attention_probe)
            run_check(results, "C_airllm_load", airllm_load_check)
            run_check(results, "D_forward_raw", forward_raw)
            run_check(results, "D_forward_use_cache_false", forward_use_cache_false)
            run_check(results, "D_forward_attention_mask", forward_attention_mask)
            run_check(results, "E_generate_max_new_tokens_1", generate_one_token)
            run_check(results, "F_generate_max_new_tokens_4", generate_four_tokens)
            run_check(results, "G_generate_readme_style", generate_readme_style)

            results["finished_at"] = datetime.now(timezone.utc).isoformat()
            results["disk_after"] = disk_info(ROOT)
            write_receipts(results)
            print(f"\nWrote {TEXT_RECEIPT}")
            print(f"Wrote {JSON_RECEIPT}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            gc.collect()


if __name__ == "__main__":
    main()
