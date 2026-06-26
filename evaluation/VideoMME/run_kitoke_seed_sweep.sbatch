#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_kitoke_seed
#SBATCH --account=bglg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --array=0-9%1
#SBATCH --output=/work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs/%x-%A_%a.out
#SBATCH --error=/work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs/%x-%A_%a.err

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
RUN_ID="${SLURM_ARRAY_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${BASE_OUT:-/work/nvme/bglg/adeng2/qwen3vl_kitoke_seed_${RUN_ID}}"
SEED="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
OUT_DIR="${BASE_OUT}/seed_${SEED}/${DURATION:-short}"
mkdir -p "${OUT_DIR}"

python run_videomme.py infer \
  --model-path "${MODEL_PATH}" \
  --data-dir "${DATA_DIR}" \
  --video-dir "${VIDEO_DIR}" \
  --duration "${DURATION:-short}" \
  --output-file "${OUT_DIR}/predictions.jsonl" \
  --prepare-batch-size "${PREPARE_BATCH_SIZE:-1}" \
  --max-new-tokens 32768 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --repetition-penalty 1.0 \
  --presence-penalty 1.5 \
  --fps 2 \
  --min-pixels 3584 \
  --max-pixels 786432 \
  --min-frames 4 \
  --max-frames "${MAX_FRAMES:-64}" \
  --total-pixels 117964800 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --kitoke-mode post_vit \
  --kitoke-retain-ratio "${KITOKE_RETAIN_RATIO:-0.10}" \
  --kitoke-kernel-alpha "${KITOKE_KERNEL_ALPHA:-800}" \
  --kitoke-selection-method pivotal \
  --kitoke-selection-seed "${SEED}" \
  --kitoke-seed-policy "${KITOKE_SEED_POLICY:-stable_video}" \
  --kitoke-pivotal-pairing "${KITOKE_PIVOTAL_PAIRING:-random_rounds}" \
  --kitoke-diff-threshold "${KITOKE_DIFF_THRESHOLD:-110}" \
  --kitoke-delta-threshold "${KITOKE_DELTA_THRESHOLD:-70}" \
  --kitoke-relative-delta-threshold "${KITOKE_RELATIVE_DELTA_THRESHOLD:-0.4}" \
  --kitoke-empty-interval-policy "${KITOKE_EMPTY_INTERVAL_POLICY:-repair_swap}" \
  --kitoke-merge-mode "${KITOKE_MERGE_MODE:-weighted}" \
  --kitoke-deepstack-mode "${KITOKE_DEEPSTACK_MODE:-same_weighted_merge}"
