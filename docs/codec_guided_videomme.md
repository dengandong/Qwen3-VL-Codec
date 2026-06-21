# Codec-Guided Qwen3-VL VideoMME Evaluation

This repository adds codec-guided patch/token selection for Qwen3-VL running through vLLM.

## What Is Included

- `evaluation/VideoMME/vllm_qwen3_vl_codec_guided.py`
  Runtime monkey-patch for vLLM Qwen3-VL.
- `evaluation/VideoMME/run_videomme.py`
  VideoMME inference/evaluation entrypoint with codec-guided options.
- `evaluation/VideoMME/run_codec_only_2xa100.sh`
  Local 2-GPU runner for the four codec experiments.
- `evaluation/VideoMME/run_codec_only_array.sbatch`
  Slurm array runner for the same four codec experiments.
- `evaluation/VideoMME/run_codec_smoke_array.sbatch`
  Small GPU smoke test for `pre_vit` and `post_vit`.

## Required Files

Place the guidance files in the repository root:

```text
codec_f128_eq16.zip
codec_f64_eqf8.zip
```

These files are larger than GitHub's normal 100MB file limit and are not stored in ordinary git history unless Git LFS is configured.

## Environment

Use the same package family as the cluster run:

- Python 3.11
- PyTorch with CUDA
- vLLM 0.12.x
- transformers with Qwen3-VL support
- qwen-vl-utils from this repo or an equivalent installed version
- datasets, pandas, numpy, tqdm

The vLLM worker multiprocessing method must be `fork` so the runtime monkey-patch is inherited by EngineCore workers:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=fork
```

## Data Layout

Set these variables before running:

```bash
export DATA_DIR=/path/to/videomme
export VIDEO_DIR=/path/to/videomme/data
export MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct
```

`VIDEO_DIR` should contain files like:

```text
${VIDEO_DIR}/${videoID}.mp4
```

## Smoke Test

Before launching full evaluation, run a 1-sample smoke test. On Slurm:

```bash
sbatch evaluation/VideoMME/run_codec_smoke_array.sbatch
```

A successful pruning run should include log lines like:

```text
[CodecGuided-vLLM] video[0] keep_groups=2889/23040 keep_patches=11556/92160
```

If you see `grid mismatch`, the runtime video preprocessing does not match the guidance zip.

## Run Four Codec Experiments On 2xA100

From the repo root:

```bash
export DATA_DIR=/path/to/videomme
export VIDEO_DIR=/path/to/videomme/data
export MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct
export BASE_OUT=/path/to/output/qwen3vl_codec_eval

bash evaluation/VideoMME/run_codec_only_2xa100.sh
```

The script runs two jobs concurrently, one per GPU:

```text
codec_f128_eq16_pre_vit   short / medium / long
codec_f128_eq16_post_vit  short / medium / long
codec_f64_eq8_pre_vit     short / medium / long
codec_f64_eq8_post_vit    short / medium / long
```

Outputs are written to:

```text
${BASE_OUT}/${experiment}/${duration}/predictions.jsonl
${BASE_OUT}/logs/${experiment}_${duration}.log
```

## Run On Slurm

```bash
sbatch evaluation/VideoMME/run_codec_only_array.sbatch
```

The Slurm script runs a 12-task array with at most four tasks active:

```text
#SBATCH --array=0-11%4
```

Tune `%4`, `--mem`, and `PREPARE_BATCH_SIZE` for your cluster.

## Important Runtime Choices

- `--max-pixels 786432`
- `--total-pixels 117964800`
- `--prepare-batch-size 1`
- `VLLM_WORKER_MULTIPROC_METHOD=fork`

`run_videomme.py` also passes `do_resize=False` to the HF/vLLM video processor because `qwen_vl_utils.process_vision_info` has already decoded and resized the video. This avoids a second resize that would invalidate the guidance grid.

## Evaluate Predictions

After inference:

```bash
python evaluation/VideoMME/run_videomme.py eval \
  --input-file /path/to/predictions.jsonl \
  --output-file /path/to/eval_results.csv \
  --judge-model gpt-4o-mini
```

The evaluation output includes accuracy by `duration`, `sub_category`, and `task_type`.
