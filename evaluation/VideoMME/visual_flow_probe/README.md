# Visual Flow Probe for VideoMME

This directory contains an isolated Phase-1 diagnostic experiment for visual-token
responsibility in Qwen3-VL on VideoMME. It does not import or enable any existing
vLLM compression method. The default backend is vLLM. It uses a scoped runtime
patch on vLLM's Qwen3 decoder attention to capture diagnostic decoder
self-attention and to zero selected value vectors. The legacy Hugging Face model
backend remains available via `--backend hf`, but it is not the default.

## Scientific Question

Do a small number of video placeholder tokens have substantially higher causal
responsibility for the model's answer than low-responsibility or randomly
selected video tokens?

This is not a compression method. Phase 1 intentionally keeps sequence length,
M-RoPE positions, visual placeholders, and attention denominators unchanged
during interventions.

## Responsibility Score

The runner captures decoder self-attention from middle language-model layers.
With the vLLM backend, this is computed inside the worker from Q/K after RoPE in
Qwen3's decoder attention during prefill. Attention is averaged over heads and
selected layers using the convention:

```text
A[q, k] = attention from query position q to key/source position k
```

Only strict chronological edges are used. For target positions `T`:

```text
b[i] = 1 / |T| if i in T else 0
h[i] = b[i] + sum_{j > i} A[j, i] * h[j]
```

The visual source responsibility is `h[i]` normalized over video placeholder
positions. A direct-attention baseline, `mean_t A[t, i]`, is also saved.

## Target Modes

`decision` is the primary target mode. For each baseline-generated answer token
at position `p`, the target node is `p - 1`, because logits at `p - 1` predict
the answer token.

`post_answer` uses the answer token positions themselves. It is an auxiliary
FlowTracer-style sink because answer token embeddings are already present.

## Intervention

The primary causal intervention is value-zeroing. With the vLLM backend, Qwen3
uses a fused `qkv_proj`, so the runtime patch zeros only the V slice for selected
full-sequence visual positions during prefill. Query and key projections,
sequence length, placeholders, and positions are unchanged. The zeroed values
enter the prefill KV cache; cached decode steps are not reinterpreted as full
sequence positions. The HF backend uses equivalent scoped module hooks on
decoder `v_proj` modules.

## Primary Endpoint

The primary pre-registered comparison is:

- target mode: `decision`
- score: multihop responsibility
- ratio: `0.20`
- intervention: value-zero
- control: temporally matched random
- endpoint: drop in log probability of the original baseline answer

Random repetitions are averaged within sample before paired bootstrap
confidence intervals are computed.

Evidence for a high-responsibility-token phenomenon requires, on a nontrivial
sample:

1. top-responsibility ablation causes a larger baseline-answer logprob drop than
   temporally matched random;
2. the paired confidence interval is clearly above zero;
3. top ablation is stronger than bottom ablation;
4. the pattern is not explained entirely by temporal position;
5. results are reasonably stable across `decision` and `post_answer`.

Do not draw strong conclusions from a one-sample smoke test.

## Outputs

`results.jsonl` contains one record per sample, including baseline answer,
targets, responsibility concentration, interventions, seeds, and skip reasons.

`responsibilities/<question_id>.npz` stores visual sequence positions, local
indices, responsibility, direct attention, `(t, y, x)` grid coordinates, target
positions, and selected intervention sets. Full attention matrices are not saved.

`summary.json` and `summary.csv` contain grouped metrics and paired bootstrap
intervals. The originally pre-registered primary ratio is 0.20; when a run only
contains another ratio such as 0.10, `summary.json` also reports the first
available decision/responsibility ratio as a clearly marked fallback and includes
all-ratio paired comparisons. `run_config.json` records CLI args, model/config
info, versions, git status, selected layers, and CUDA metadata.

`--resume` skips question IDs already present in `results.jsonl`.

## Memory Notes

Defaults are conservative: batch size 1, `max_frames=8`, limited pixels, no raw
attention persistence, `use_cache=False` for the flow pass, and
`max_flow_seq_len=2048`. Samples exceeding the flow length limit are skipped with
a structured reason.

## Example Commands

One-sample primary smoke:

```bash
/work/nvme/bglg/adeng2/conda_envs/qwen3vl-codec-eval/bin/python \
  evaluation/VideoMME/visual_flow_probe/run_visual_flow_probe.py \
  --backend vllm \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --data-path /work/nvme/bglg/adeng2/hf_cache/videomme \
  --video-dir /work/nvme/bglg/adeng2/hf_cache/videomme/data \
  --output-dir outputs/visual_flow_probe/decision_r20_smoke \
  --duration short \
  --target-mode decision \
  --ratios 0.20 \
  --random-repeats 5 \
  --max-samples 1 \
  --max-frames 8 \
  --max-flow-seq-len 2048 \
  --seed 42 \
  --resume
```

Exploratory post-answer run:

```bash
/work/nvme/bglg/adeng2/conda_envs/qwen3vl-codec-eval/bin/python \
  evaluation/VideoMME/visual_flow_probe/run_visual_flow_probe.py \
  --backend vllm \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --data-path /work/nvme/bglg/adeng2/hf_cache/videomme \
  --video-dir /work/nvme/bglg/adeng2/hf_cache/videomme/data \
  --output-dir outputs/visual_flow_probe/post_answer_r20 \
  --duration short \
  --target-mode post_answer \
  --ratios 0.05,0.10,0.20 \
  --random-repeats 5 \
  --max-samples 100 \
  --max-frames 8 \
  --max-flow-seq-len 2048 \
  --seed 42 \
  --resume
```
