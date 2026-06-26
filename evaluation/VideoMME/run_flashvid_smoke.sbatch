#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_flashvid_smoke
#SBATCH --account=bglg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs/%x-%j.out
#SBATCH --error=/work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs/%x-%j.err

set -euo pipefail

cd /u/adeng2/code/Qwen3-VL/evaluation/VideoMME
mkdir -p /work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs

export QWEN3VL_ENV_BIN="${QWEN3VL_ENV_BIN:-/work/nvme/bglg/adeng2/conda_envs/qwen3vl-codec-eval/bin}"
export PATH="${QWEN3VL_ENV_BIN}:${PATH}"
export HF_HOME="${HF_HOME:-/work/nvme/bglg/adeng2/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/work/nvme/bglg/adeng2/hf_cache/datasets}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-fork}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export QWEN3VL_PATCH_DECORD_WITH_OPENCV="${QWEN3VL_PATCH_DECORD_WITH_OPENCV:-0}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_DIR="${DATA_DIR:-/work/nvme/bglg/adeng2/hf_cache/videomme}"
VIDEO_DIR="${VIDEO_DIR:-/work/nvme/bglg/adeng2/hf_cache/videomme/data}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
OUT_DIR="${OUT_DIR:-/work/nvme/bglg/adeng2/qwen3vl_flashvid_smoke_${SLURM_JOB_ID:-manual}}"
mkdir -p "${OUT_DIR}"

FLASHVID_DEBUG_VERIFY="${FLASHVID_DEBUG_VERIFY:-1}"
DEBUG_ARGS=()
if [[ "${FLASHVID_DEBUG_VERIFY}" == "1" ]]; then
  DEBUG_ARGS+=(--flashvid-debug-verify)
fi

python run_videomme.py infer \
  --model-path "${MODEL_PATH}" \
  --data-dir "${DATA_DIR}" \
  --video-dir "${VIDEO_DIR}" \
  --duration "${DURATION:-short}" \
  --output-file "${OUT_DIR}/predictions.jsonl" \
  --max-samples "${MAX_SAMPLES:-1}" \
  --prepare-batch-size 1 \
  --max-new-tokens 32 \
  --temperature 0.0 \
  --top-p 1.0 \
  --top-k 1 \
  --repetition-penalty 1.0 \
  --presence-penalty 0.0 \
  --fps 2 \
  --min-pixels 3584 \
  --max-pixels 786432 \
  --min-frames 4 \
  --max-frames "${MAX_FRAMES:-32}" \
  --total-pixels 117964800 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --flashvid-mode post_vit \
  --flashvid-profile "${FLASHVID_PROFILE:-official_qwen3}" \
  --flashvid-budget-mode "${FLASHVID_BUDGET_MODE:-direct}" \
  --flashvid-retention-ratio "${FLASHVID_RETENTION_RATIO:-0.20}" \
  --flashvid-expansion "${FLASHVID_EXPANSION:-1.25}" \
  --flashvid-alpha "${FLASHVID_ALPHA:-0.70}" \
  --flashvid-temporal-threshold "${FLASHVID_TEMPORAL_THRESHOLD:-0.80}" \
  --flashvid-segment-threshold "${FLASHVID_SEGMENT_THRESHOLD:-0.90}" \
  --flashvid-min-segment-num "${FLASHVID_MIN_SEGMENT_NUM:-4}" \
  --flashvid-budget-correction "${FLASHVID_BUDGET_CORRECTION:-official_ceil}" \
  --flashvid-dpc-k-max "${FLASHVID_DPC_K_MAX:-7}" \
  --flashvid-deepstack-mode "${FLASHVID_DEEPSTACK_MODE:-official_gather}" \
  --flashvid-cls-attn-chunk-size "${FLASHVID_CLS_ATTN_CHUNK_SIZE:-128}" \
  --flashvid-temporal-match-chunk-size "${FLASHVID_TEMPORAL_MATCH_CHUNK_SIZE:-256}" \
  "${DEBUG_ARGS[@]}"
