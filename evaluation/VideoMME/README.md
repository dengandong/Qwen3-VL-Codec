# VideoMME Compression Experiments

This directory contains the active experiment code for Qwen3-VL video token
compression under vLLM.

## Main Files

```text
run_videomme.py
  Inference/evaluation entrypoint.  Applies runtime patches before LLM(...)
  construction and writes JSONL predictions.

vllm_qwen3_vl_codec_guided.py
  Codec-guided pre-ViT and post-ViT selection plus random-selection baseline.

vllm_qwen3_vl_vcast.py
  V-CAST post-ViT pruning.

vllm_qwen3_vl_ttf.py
  TTF retain-ratio pruning with the original global-anchor formulation.

vllm_qwen3_vl_ttf_v2.py
  TTF V2 dynamic local temporal-anchor variant.

vllm_qwen3_vl_echoprune.py
  EchoPrune query-guided fixed-budget pruning.

test_vllm_qwen3_vl_*.py
  CPU tests for algorithm and compaction invariants.
```

## Launchers

```text
run_baseline_array.sbatch       dense baselines
run_codec_only_array.sbatch     codec-guided pre/post-ViT
run_codec_random_array.sbatch   random baseline using codec budgets
run_vcast_array.sbatch          V-CAST
run_ttf_array.sbatch            TTF
run_ttf_v2_array.sbatch         TTF V2
run_echoprune_array.sbatch      EchoPrune
```

Smoke scripts are available for codec, V-CAST, and EchoPrune.  The frame
similarity plotting helper for TTF is `plot_ttf_frame_similarity.py`.

## Required Environment Variables

```bash
export DATA_DIR=/path/to/VideoMME
export VIDEO_DIR=/path/to/VideoMME/data
export MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct
export VLLM_WORKER_MULTIPROC_METHOD=fork
export FORCE_QWENVL_VIDEO_READER=decord
```

## Method Examples

TTF V2, max 64 frames, retain 12.5%:

```bash
MAX_FRAMES=64 TTF_RETAIN_RATIO=0.125 \
  sbatch evaluation/VideoMME/run_ttf_v2_array.sbatch
```

Codec-guided f64/f128 pre/post-ViT:

```bash
sbatch evaluation/VideoMME/run_codec_only_array.sbatch
```

EchoPrune, retain 12.5%:

```bash
ECHOPRUNE_RETAIN_RATIO=0.125 \
  sbatch evaluation/VideoMME/run_echoprune_array.sbatch
```

## Evaluation

```bash
python evaluation/VideoMME/run_videomme.py eval \
  --data-dir "${DATA_DIR}" \
  --input-file /path/to/predictions.jsonl \
  --output-file /path/to/eval_results.csv \
  --api-type dash \
  --nproc 4
```

The runner preserves `task_type`, `sub_category`, and `duration` metadata in
predictions so downstream summaries can report task-level performance.
