#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_echoprune_eval
#SBATCH --account=bglg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
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
BASE_OUT="${BASE_OUT:-/work/nvme/bglg/adeng2/qwen3vl_echoprune_eval_${RUN_ID}}"
PREPARE_BATCH_SIZE="${PREPARE_BATCH_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
ECHOPRUNE_RETAIN_RATIO="${ECHOPRUNE_RETAIN_RATIO:-0.125}"
ECHOPRUNE_TEMPERATURE="${ECHOPRUNE_TEMPERATURE:-0.50}"
ECHOPRUNE_MATCH_SCOPE="${ECHOPRUNE_MATCH_SCOPE:-full}"
ECHOPRUNE_WINDOW_SIZE="${ECHOPRUNE_WINDOW_SIZE:-3}"
ECHOPRUNE_FIRST_FRAME_POLICY="${ECHOPRUNE_FIRST_FRAME_POLICY:-paper}"
ECHOPRUNE_QUERY_SOURCE="${ECHOPRUNE_QUERY_SOURCE:-question_options}"
ECHOPRUNE_MATCH_CHUNK_SIZE="${ECHOPRUNE_MATCH_CHUNK_SIZE:-256}"
ECHOPRUNE_DEBUG_VERIFY="${ECHOPRUNE_DEBUG_VERIFY:-0}"

TASKS=(
  "echoprune_f64_rr0125 64 short"
  "echoprune_f64_rr0125 64 medium"
  "echoprune_f64_rr0125 64 long"
  "echoprune_f128_rr0125 128 short"
  "echoprune_f128_rr0125 128 medium"
  "echoprune_f128_rr0125 128 long"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
read -r EXP_NAME MAX_FRAMES DURATION <<< "${TASKS[$TASK_ID]}"

OUT_DIR="${BASE_OUT}/${EXP_NAME}/${DURATION}"
mkdir -p "${OUT_DIR}"
OUT_FILE="${OUT_DIR}/predictions.jsonl"

echo "experiment=${EXP_NAME}"
echo "duration=${DURATION}"
echo "max_frames=${MAX_FRAMES}"
echo "echoprune_retain_ratio=${ECHOPRUNE_RETAIN_RATIO}"
echo "echoprune_temperature=${ECHOPRUNE_TEMPERATURE}"
echo "echoprune_match_scope=${ECHOPRUNE_MATCH_SCOPE}"
echo "echoprune_first_frame_policy=${ECHOPRUNE_FIRST_FRAME_POLICY}"
echo "echoprune_query_source=${ECHOPRUNE_QUERY_SOURCE}"
echo "output=${OUT_FILE}"
echo "prepare_batch_size=${PREPARE_BATCH_SIZE}"
echo "vllm_worker_multiproc_method=${VLLM_WORKER_MULTIPROC_METHOD}"
echo "force_qwenvl_video_reader=${FORCE_QWENVL_VIDEO_READER}"
echo "patch_decord_with_opencv=${QWEN3VL_PATCH_DECORD_WITH_OPENCV}"
echo "python_no_user_site=${PYTHONNOUSERSITE}"

DEBUG_ARGS=()
if [[ "${ECHOPRUNE_DEBUG_VERIFY}" == "1" ]]; then
  DEBUG_ARGS+=(--echoprune-debug-verify)
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
  --echoprune-mode post_vit \
  --echoprune-retain-ratio "${ECHOPRUNE_RETAIN_RATIO}" \
  --echoprune-temperature "${ECHOPRUNE_TEMPERATURE}" \
  --echoprune-match-scope "${ECHOPRUNE_MATCH_SCOPE}" \
  --echoprune-window-size "${ECHOPRUNE_WINDOW_SIZE}" \
  --echoprune-first-frame-policy "${ECHOPRUNE_FIRST_FRAME_POLICY}" \
  --echoprune-query-source "${ECHOPRUNE_QUERY_SOURCE}" \
  --echoprune-match-chunk-size "${ECHOPRUNE_MATCH_CHUNK_SIZE}" \
  "${DEBUG_ARGS[@]}"
