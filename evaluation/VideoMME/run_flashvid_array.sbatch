#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_flashvid_eval
#SBATCH --account=bglg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%1
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
BASE_OUT="${BASE_OUT:-/work/nvme/bglg/adeng2/qwen3vl_flashvid_eval_${RUN_ID}}"
PREPARE_BATCH_SIZE="${PREPARE_BATCH_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

FLASHVID_PROFILE="${FLASHVID_PROFILE:-official_qwen3}"
FLASHVID_BUDGET_MODE="${FLASHVID_BUDGET_MODE:-direct}"
FLASHVID_RETENTION_RATIO="${FLASHVID_RETENTION_RATIO:-0.125}"
FLASHVID_EXPANSION="${FLASHVID_EXPANSION:-1.25}"
FLASHVID_ALPHA="${FLASHVID_ALPHA:-0.70}"
FLASHVID_TEMPORAL_THRESHOLD="${FLASHVID_TEMPORAL_THRESHOLD:-0.80}"
FLASHVID_SEGMENT_THRESHOLD="${FLASHVID_SEGMENT_THRESHOLD:-0.90}"
FLASHVID_MIN_SEGMENT_NUM="${FLASHVID_MIN_SEGMENT_NUM:-4}"
FLASHVID_BUDGET_CORRECTION="${FLASHVID_BUDGET_CORRECTION:-official_ceil}"
FLASHVID_DPC_K_MAX="${FLASHVID_DPC_K_MAX:-7}"
FLASHVID_DEEPSTACK_MODE="${FLASHVID_DEEPSTACK_MODE:-official_gather}"
FLASHVID_CLS_ATTN_CHUNK_SIZE="${FLASHVID_CLS_ATTN_CHUNK_SIZE:-128}"
FLASHVID_TEMPORAL_MATCH_CHUNK_SIZE="${FLASHVID_TEMPORAL_MATCH_CHUNK_SIZE:-256}"
FLASHVID_DEBUG_VERIFY="${FLASHVID_DEBUG_VERIFY:-0}"

TASKS=(
  "flashvid_f64_rr0125 64 short"
  "flashvid_f64_rr0125 64 medium"
  "flashvid_f64_rr0125 64 long"
  "flashvid_f128_rr0125 128 short"
  "flashvid_f128_rr0125 128 medium"
  "flashvid_f128_rr0125 128 long"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
read -r EXP_NAME MAX_FRAMES DURATION <<< "${TASKS[$TASK_ID]}"

OUT_DIR="${BASE_OUT}/${EXP_NAME}/${DURATION}"
mkdir -p "${OUT_DIR}"
OUT_FILE="${OUT_DIR}/predictions.jsonl"

echo "experiment=${EXP_NAME}"
echo "duration=${DURATION}"
echo "max_frames=${MAX_FRAMES}"
echo "flashvid_profile=${FLASHVID_PROFILE}"
echo "flashvid_budget_mode=${FLASHVID_BUDGET_MODE}"
echo "flashvid_retention_ratio=${FLASHVID_RETENTION_RATIO}"
echo "flashvid_alpha=${FLASHVID_ALPHA}"
echo "flashvid_temporal_threshold=${FLASHVID_TEMPORAL_THRESHOLD}"
echo "flashvid_segment_threshold=${FLASHVID_SEGMENT_THRESHOLD}"
echo "flashvid_budget_correction=${FLASHVID_BUDGET_CORRECTION}"
echo "output=${OUT_FILE}"
echo "prepare_batch_size=${PREPARE_BATCH_SIZE}"
echo "vllm_worker_multiproc_method=${VLLM_WORKER_MULTIPROC_METHOD}"
echo "force_qwenvl_video_reader=${FORCE_QWENVL_VIDEO_READER}"

DEBUG_ARGS=()
if [[ "${FLASHVID_DEBUG_VERIFY}" == "1" ]]; then
  DEBUG_ARGS+=(--flashvid-debug-verify)
fi

python run_videomme.py infer \
  --model-path "${MODEL_PATH}" \
  --data-dir "${DATA_DIR}" \
  --video-dir "${VIDEO_DIR}" \
  --duration "${DURATION}" \
  --output-file "${OUT_FILE}" \
  --prepare-batch-size "${PREPARE_BATCH_SIZE}" \
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
  --max-frames "${MAX_FRAMES}" \
  --total-pixels 117964800 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --flashvid-mode post_vit \
  --flashvid-profile "${FLASHVID_PROFILE}" \
  --flashvid-budget-mode "${FLASHVID_BUDGET_MODE}" \
  --flashvid-retention-ratio "${FLASHVID_RETENTION_RATIO}" \
  --flashvid-expansion "${FLASHVID_EXPANSION}" \
  --flashvid-alpha "${FLASHVID_ALPHA}" \
  --flashvid-temporal-threshold "${FLASHVID_TEMPORAL_THRESHOLD}" \
  --flashvid-segment-threshold "${FLASHVID_SEGMENT_THRESHOLD}" \
  --flashvid-min-segment-num "${FLASHVID_MIN_SEGMENT_NUM}" \
  --flashvid-budget-correction "${FLASHVID_BUDGET_CORRECTION}" \
  --flashvid-dpc-k-max "${FLASHVID_DPC_K_MAX}" \
  --flashvid-deepstack-mode "${FLASHVID_DEEPSTACK_MODE}" \
  --flashvid-cls-attn-chunk-size "${FLASHVID_CLS_ATTN_CHUNK_SIZE}" \
  --flashvid-temporal-match-chunk-size "${FLASHVID_TEMPORAL_MATCH_CHUNK_SIZE}" \
  "${DEBUG_ARGS[@]}"
