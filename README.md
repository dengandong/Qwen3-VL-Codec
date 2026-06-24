# Qwen3-VL Video Token Compression

This repository is a trimmed Qwen3-VL fork for studying video visual-token
compression under vLLM.  The active code path is the VideoMME runner plus a set
of runtime Qwen3-VL/vLLM patches that shorten the visual placeholder sequence,
M-RoPE positions, and LLM prefill/KV cache length.

It is not intended to be a general Qwen3-VL demo, cookbook, fine-tuning, or
multi-benchmark repository.

## What Is Included

```text
evaluation/VideoMME/
  run_videomme.py                    # VideoMME inference/evaluation entrypoint
  vllm_qwen3_vl_codec_guided.py       # codec-guided pre/post-ViT pruning
  vllm_qwen3_vl_vcast.py              # V-CAST post-ViT pruning
  vllm_qwen3_vl_ttf.py                # TTF retain-ratio implementation
  vllm_qwen3_vl_ttf_v2.py             # TTF V2 dynamic local-anchor variant
  vllm_qwen3_vl_echoprune.py          # EchoPrune query-guided pruning
  run_*_array.sbatch                  # Slurm launchers
  test_vllm_qwen3_vl_*.py             # CPU algorithm/invariant tests

docs/
  codec_guided_videomme.md            # codec-guided experiment notes

codec_f128_eq16.zip                   # local codec guidance, max_frames=128
codec_f64_eqf8.zip                    # local codec guidance, max_frames=64
```

The two codec zip files are ignored by git because they are large experiment
artifacts.  Keep them in the repository root when running codec-guided methods.

## Implemented Methods

- **Baseline**: dense Qwen3-VL inference at fixed `max_frames`.
- **Codec-guided**: selects visual groups from precomputed codec priors, with
  `pre_vit`, `post_vit`, and deterministic random-selection baseline modes.
- **V-CAST**: online post-ViT token pruning with fixed retain ratio.
- **TTF**: Temporal Token Fusion-style post-ViT matching with fixed retain
  ratio so vLLM can compact the prompt before scheduling.
- **TTF V2**: replaces the global anchor with a per-frame dynamic temporal
  anchor chosen from `[t-2, t+2]` by frame-level similarity.
- **EchoPrune**: query-guided relevance minus temporal redundancy, with fixed
  visual-token budget and sparse M-RoPE gather.

Only one pruning method can be enabled per run.  `run_videomme.py` enforces this
so results are not accidentally double-pruned.

## Environment

The cluster experiments use:

```text
Python 3.11
PyTorch with CUDA
vLLM 0.12.x
transformers with Qwen3-VL support
qwen-vl-utils >= 0.0.14
decord or a working qwen-vl-utils video backend
datasets, pandas, numpy, tqdm, pillow
```

Recommended runtime variables:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=fork
export FORCE_QWENVL_VIDEO_READER=decord
export PYTHONNOUSERSITE=1
```

The monkey patches are applied in Python before `LLM(...)` is instantiated.
`fork` ensures the vLLM EngineCore worker inherits those patches.

## Data Layout

Set these paths before launching jobs:

```bash
export DATA_DIR=/path/to/VideoMME
export VIDEO_DIR=/path/to/VideoMME/data
export MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct
```

`VIDEO_DIR` should contain files named by VideoMME video id, for example:

```text
${VIDEO_DIR}/fFjv93ACGo8.mp4
```

## Quick Commands

Baseline:

```bash
MAX_FRAMES=64 sbatch evaluation/VideoMME/run_baseline_array.sbatch
```

Codec-guided pre/post-ViT:

```bash
sbatch evaluation/VideoMME/run_codec_only_array.sbatch
```

Random selection with the same codec budgets:

```bash
sbatch evaluation/VideoMME/run_codec_random_array.sbatch
```

V-CAST, retain 12.5%:

```bash
VCAST_RETAIN_RATIO=0.125 VCAST_EXP_SUFFIX=rr0125 \
  sbatch evaluation/VideoMME/run_vcast_array.sbatch
```

TTF, retain 12.5%:

```bash
MAX_FRAMES=64 TTF_RETAIN_RATIO=0.125 \
  sbatch evaluation/VideoMME/run_ttf_array.sbatch
```

TTF V2, retain 12.5%:

```bash
MAX_FRAMES=64 TTF_RETAIN_RATIO=0.125 \
  sbatch evaluation/VideoMME/run_ttf_v2_array.sbatch
```

EchoPrune, retain 12.5%:

```bash
ECHOPRUNE_RETAIN_RATIO=0.125 \
  sbatch evaluation/VideoMME/run_echoprune_array.sbatch
```

## Direct Runner Example

```bash
python evaluation/VideoMME/run_videomme.py infer \
  --model-path "${MODEL_PATH}" \
  --data-dir "${DATA_DIR}" \
  --video-dir "${VIDEO_DIR}" \
  --duration short \
  --output-file /path/to/predictions.jsonl \
  --fps 2 \
  --min-pixels 3584 \
  --max-pixels 786432 \
  --min-frames 4 \
  --max-frames 64 \
  --total-pixels 117964800 \
  --prepare-batch-size 1 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 131072 \
  --ttf-mode post_vit \
  --ttf-version v2 \
  --ttf-budget-mode retain_ratio \
  --ttf-retain-ratio 0.125 \
  --ttf-window-radius 1 \
  --ttf-v2-temporal-anchor-radius 2 \
  --ttf-debug-verify
```

## Evaluation

For JSONL predictions:

```bash
python evaluation/VideoMME/run_videomme.py eval \
  --data-dir "${DATA_DIR}" \
  --input-file /path/to/predictions.jsonl \
  --output-file /path/to/eval_results.csv \
  --api-type dash \
  --nproc 4
```

The output includes overall accuracy plus breakdowns by `duration`,
`sub_category`, and `task_type`.

## Tests

CPU tests do not require model weights or VideoMME data:

```bash
python -m py_compile \
  evaluation/VideoMME/run_videomme.py \
  evaluation/VideoMME/vllm_qwen3_vl_codec_guided.py \
  evaluation/VideoMME/vllm_qwen3_vl_vcast.py \
  evaluation/VideoMME/vllm_qwen3_vl_ttf.py \
  evaluation/VideoMME/vllm_qwen3_vl_ttf_v2.py \
  evaluation/VideoMME/vllm_qwen3_vl_echoprune.py

python -m pytest -q \
  evaluation/VideoMME/test_vllm_qwen3_vl_ttf.py \
  evaluation/VideoMME/test_vllm_qwen3_vl_ttf_v2.py \
  evaluation/VideoMME/test_vllm_qwen3_vl_echoprune.py
```

## Notes

- The vLLM patches do not permanently edit site-packages; they patch classes at
  runtime.
- `--ttf-budget-mode retain_ratio` is the vLLM-compatible path.  It fixes the
  retained visual count before scheduler admission so the prompt and KV cache
  are truly compact.
- Debug flags such as `--ttf-debug-verify` and `--echoprune-debug-verify`
  assert that visual placeholders, embeddings, and M-RoPE positions have the
  same compact length.
- The repository keeps only the files needed for VideoMME compression
  experiments.  Upstream Qwen3-VL cookbooks, demos, fine-tuning code, Docker
  demo files, and unrelated benchmark runners were removed from this project
  branch.
