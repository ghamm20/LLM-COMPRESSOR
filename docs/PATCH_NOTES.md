# Patch Notes

## Yi-34B Generation Failure

The Yi-34B AirLLM load succeeded, but generation failed before the first decoder layer completed.

Original error:

```text
TypeError: cannot unpack non-iterable NoneType object
```

## Stack Trace Summary

The debug receipt captured the failing call path:

```text
D:\AI\airllm\repo\air_llm\airllm\airllm_base.py:569
  new_seq = layer(seq, **kwargs)[0]

D:\AI\airllm\.venv\Lib\site-packages\transformers\models\llama\modeling_llama.py:335
  hidden_states, self_attn_weights = self.self_attn(...)

D:\AI\airllm\.venv\Lib\site-packages\transformers\models\llama\modeling_llama.py:273
  cos, sin = position_embeddings
```

Runtime probe details:

- `position_embeddings`: `None`
- `layer_idx`: `0`
- `hidden_states`: shape `[1, 10, 7168]`, dtype `torch.float16`, device `cuda:0`
- `attention_mask`: shape `[1, 1, 10, 10]`, dtype `torch.bool`, device `cuda:0`
- `position_ids`: shape `[1, 10]`, dtype `torch.int64`, device `cuda:0`
- `use_cache`: `False`

## Root Cause

`transformers==4.48.3` Llama creates shared rotary embeddings in `LlamaModel.forward`:

```python
position_embeddings = self.rotary_emb(hidden_states, position_ids)
```

It then passes those embeddings into each `LlamaDecoderLayer`. AirLLM streams the model one layer at a time and calls each `LlamaDecoderLayer` directly, bypassing the full `LlamaModel.forward` path. As a result, `position_embeddings` stayed `None`, and `LlamaAttention.forward` failed while trying to unpack it.

## Patched File

- `D:\AI\airllm\repo\air_llm\airllm\airllm_base.py`

## Behavioral Fix

The patch adds a small helper that detects the model rotary embedding module and computes the missing argument:

```python
{"position_embeddings": rotary_emb(seq, position_ids)}
```

AirLLM now adds that argument to decoder-layer calls when `get_pos_emb_args` did not already provide one. This mirrors the behavior of `transformers==4.48.3` Llama without upgrading packages or replacing AirLLM.

## Proof Receipts

- Failure trace summary: `D:\AI\airllm\receipts\yi34b_failure_trace_summary.txt`
- Code map: `D:\AI\airllm\receipts\yi34b_airllm_code_map.txt`
- Debug JSON: `D:\AI\airllm\receipts\yi34b_debug_generation.json`
- Patch diff: `D:\AI\airllm\receipts\yi34b_patch.diff`
- Backup: `D:\AI\airllm\receipts\patched_backups\airllm_base.py.before_yi34b_position_embeddings_patch`
- Passing rerun JSON: `D:\AI\airllm\receipts\yi34b_generation_rerun.json`
- Passing rerun text: `D:\AI\airllm\receipts\yi34b_generation_rerun.txt`

## Result

After the patch, `NousResearch/Nous-Hermes-2-Yi-34B` generated actual text through AirLLM locally. No remote inference endpoint was used.
