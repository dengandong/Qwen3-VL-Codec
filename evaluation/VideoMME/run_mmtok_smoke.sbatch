#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_mmtok_smoke
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
OUT_DIR="${OUT_DIR:-/work/nvme/bglg/adeng2/qwen3vl_mmtok_smoke_${SLURM_JOB_ID:-manual}}"
mkdir -p "${OUT_DIR}"

MMTOK_DEBUG_VERIFY="${MMTOK_DEBUG_VERIFY:-1}"
DEBUG_ARGS=()
if [[ "${MMTOK_DEBUG_VERIFY}" == "1" ]]; then
  DEBUG_ARGS+=(--mmtok-debug-verify)
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
  --mmtok-mode post_vit \
  --mmtok-profile "${MMTOK_PROFILE:-video_scalable}" \
  --mmtok-retain-ratio "${MMTOK_RETAIN_RATIO:-0.20}" \
  --mmtok-budget-rounding "${MMTOK_BUDGET_ROUNDING:-floor}" \
  --mmtok-alpha "${MMTOK_ALPHA:-0.50}" \
  --mmtok-tv-temperature "${MMTOK_TV_TEMPERATURE:-0.01}" \
  --mmtok-vv-temperature "${MMTOK_VV_TEMPERATURE:-0.20}" \
  --mmtok-temperature-mode "${MMTOK_TEMPERATURE_MODE:-fixed}" \
  --mmtok-adaptive-vv-candidates "${MMTOK_ADAPTIVE_VV_CANDIDATES:-0.05,0.10,0.15,0.20}" \
  --mmtok-vv-target-mode "${MMTOK_VV_TARGET_MODE:-stratified_3d}" \
  --mmtok-vv-target-tokens "${MMTOK_VV_TARGET_TOKENS:-1024}" \
  --mmtok-greedy-mode "${MMTOK_GREEDY_MODE:-stochastic}" \
  --mmtok-stochastic-epsilon "${MMTOK_STOCHASTIC_EPSILON:-0.10}" \
  --mmtok-selection-seed "${MMTOK_SELECTION_SEED:-3407}" \
  --mmtok-query-source "${MMTOK_QUERY_SOURCE:-question_options}" \
  --mmtok-candidate-chunk-size "${MMTOK_CANDIDATE_CHUNK_SIZE:-512}" \
  --mmtok-target-chunk-size "${MMTOK_TARGET_CHUNK_SIZE:-256}" \
  --mmtok-exact-max-tokens "${MMTOK_EXACT_MAX_TOKENS:-1024}" \
  "${DEBUG_ARGS[@]}"
