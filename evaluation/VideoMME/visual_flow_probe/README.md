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

## Implementation-Aligned Pseudocode

The two diagnostics below use the same attention convention and reachability DP.
They differ only in when layer averaging happens and whether visual scores are
normalized.

### Layer 17-34 Averaged Responsibility

This is the path used by `run_visual_flow_probe.py` for the layer-17-to-34 f8
sweep. `--layer-end` is exclusive, so `--layer-start 17 --layer-end 35`
captures decoder layers `17, ..., 34`.

```text
Input:
  VideoMME sample
  Qwen3-VL vLLM model and processor
  selected_layers = [17, 18, ..., 34]
  target_mode = decision by default

1. Build the VideoMME prompt.
   messages, annotation = build_prompt(sample)
   prepared = prepare_vllm_prompt(processor, messages)

2. Run deterministic baseline generation.
   baseline = deterministic_vllm_generate(model, prepared.vllm_input)

3. Build the teacher-forced diagnostic sequence.
   teacher_input = prompt_tokens + baseline_generated_answer_tokens
   answer_positions = positions of baseline answer tokens

   if target_mode == "decision":
       target_positions = [p - 1 for p in answer_positions]
   else if target_mode == "post_answer":
       target_positions = answer_positions

4. Enable vLLM decoder-attention capture on selected_layers.

   In each selected Qwen3Attention.forward:
       qkv = qkv_proj(hidden_states)
       q, k, v = split(qkv)

       q = q_norm(q)
       k = k_norm(k)
       q, k = rotary_emb(positions, q, k)

       logits[head, query, key] =
           dot(q[query, head], k[key, head]) * scaling

       mask logits where key > query
       attn[head, query, key] = softmax(logits over key dimension)
       A_layer[query, key] = mean over heads of attn[:, query, key]

       store A_layer on CPU

5. Force one prefill pass on the teacher-forced input.
   llm.generate([teacher_input], max_tokens=1)

6. Average attention across layers before running flow.
   A = mean_layer A_layer

   Shape:
     A: [seq_len, seq_len]

   Convention:
     A[query, key] = attention from query position to key/source position
     A[later_query, earlier_key] represents earlier_key -> later_query

   Assert causal orientation:
     A[query, key] ~= 0 for key > query

7. Locate visual placeholder positions.
   video_token_id = get_vllm_video_token_id(processor, model)
   visual_positions = where teacher_input_ids == video_token_id

   spatial_merge_size = get_vllm_spatial_merge_size(processor, model)
   map visual_positions to (video, t, y, x) using video_grid_thw and
   spatial_merge_size.

8. Compute multihop answer reachability on the averaged attention graph.

   b[i] = 1 / len(target_positions), if i is a target position
          0, otherwise

   h = zeros(seq_len)
   for i from seq_len - 1 down to 0:
       h[i] = b[i] + sum_{j > i} A[j, i] * h[j]

9. Extract and normalize visual responsibility.
   raw_visual = h[visual_positions]
   responsibility = raw_visual / sum(raw_visual)

10. Also compute the direct-attention baseline.
    direct[v] = mean over target t of A[t, visual_positions[v]]
    if sum(direct) > 0:
        direct = direct / sum(direct)

Output:
  responsibility: [num_visual_tokens], normalized to sum to 1
  direct_attention: [num_visual_tokens], normalized when possible
  target_positions
  visual token grid metadata
```

The important implementation detail is that this path averages the selected
layers' attention matrices first and runs the reachability DP once. It does not
run one DP per layer and then average the resulting responsibility vectors.

### Layer-Wise Responsibility Matrix

This is the path used by `run_layer_matrix_dump.py` and the 10-example plotting
script `run_layer_dynamics.py`. It saves raw per-layer visual reachability
curves, so values are not normalized across visual tokens.

```text
Input:
  VideoMME sample
  Qwen3-VL vLLM model and processor
  layer_start, layer_end, layer_stride
  target_mode = decision by default

Example full-depth setting:
  layer_start = 0
  layer_end = 36       # exclusive, captures layers 0..35
  layer_stride = 1

1. Build the VideoMME prompt.
   messages, annotation = build_prompt(sample)
   prepared = prepare_vllm_prompt(processor, messages)

2. Run deterministic baseline generation.
   baseline = deterministic_vllm_generate(model, prepared.vllm_input)

3. Build the teacher-forced diagnostic sequence.
   teacher_input = prompt_tokens + baseline_generated_answer_tokens
   answer_positions = positions of baseline answer tokens

   if target_mode == "decision":
       target_positions = [p - 1 for p in answer_positions]
   else if target_mode == "post_answer":
       target_positions = answer_positions

4. Select layers.
   selected_layers = range(layer_start, layer_end, layer_stride)

5. Enable per-layer vLLM decoder-attention capture.

   In each selected Qwen3Attention.forward:
       qkv = qkv_proj(hidden_states)
       q, k, v = split(qkv)

       q = q_norm(q)
       k = k_norm(k)
       q, k = rotary_emb(positions, q, k)

       logits[head, query, key] =
           dot(q[query, head], k[key, head]) * scaling

       mask logits where key > query
       attn[head, query, key] = softmax(logits over key dimension)
       A_layer[query, key] = mean over heads of attn[:, query, key]

       store A_layer separately for this layer

6. Force one prefill pass on the teacher-forced input.
   llm.generate([teacher_input], max_tokens=1)

7. Locate visual placeholder positions and grid coordinates.
   video_token_id = get_vllm_video_token_id(processor, model)
   visual_positions = where teacher_input_ids == video_token_id
   map visual_positions to (video, t, y, x) using video_grid_thw and
   spatial_merge_size.

8. For each captured layer independently:

   A_l = captured attention matrix for layer l

   Assert:
     A_l shape == [seq_len, seq_len]
     A_l is causal

   b[i] = 1 / len(target_positions), if i is a target position
          0, otherwise

   h_l = zeros(seq_len)
   for i from seq_len - 1 down to 0:
       h_l[i] = b[i] + sum_{j > i} A_l[j, i] * h_l[j]

   curve_l = h_l[visual_positions]

   Check:
     curve_l is finite
     curve_l is non-negative

   Do not normalize curve_l over visual tokens.

9. Stack curves in selected-layer order.
   responsibility_matrix[layer_row, visual_local_index] = curve_l[visual_local_index]

Output:
  responsibility_matrix: [num_selected_layers, num_visual_tokens]
  layers: selected layer indices
  visual_seq_positions
  visual_local_indices
  temporal_grid_indices
  y_grid_indices
  x_grid_indices
  video_grid_thw
  target_positions
```

Because the layer-wise matrix stores raw `h_l[visual_positions]`, different
layers may have different value ranges. This is intentional for layer-dynamics
diagnostics and matches the plotting scripts.

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
