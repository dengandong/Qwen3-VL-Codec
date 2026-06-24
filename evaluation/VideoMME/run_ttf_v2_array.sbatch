#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_ttf_v2_eval
#SBATCH --account=bglg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs/%x-%A_%a.out
#SBATCH --error=/work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs/%x-%A_%a.err

set -euo pipefail

cd /u/adeng2/code/Qwen3-VL/evaluation/VideoMME
mkdir -p /work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs

export QWEN3VL_ENV_BIN="${QWEN3VL_ENV_BIN:-/work/nvme/bglg/adeng2/conda_envs/qwen3vl-codec-eval/bin}"
export PATH="${QWEN3VL_ENV_BIN}:${PATH}"
export HF_HOME="${HF_HOME:-/work/nvme/bglg/adeng2/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/work/nvme/bglg/adeng2/hf_cache/datasets}"
export VLLM_WORKER_MULTIPROC_METHOD=fork
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export QWEN3VL_PATCH_DECORD_WITH_OPENCV="${QWEN3VL_PATCH_DECORD_WITH_OPENCV:-0}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_DIR="${DATA_DIR:-/work/nvme/bglg/adeng2/hf_cache/videomme}"
VIDEO_DIR="${VIDEO_DIR:-/work/nvme/bglg/adeng2/hf_cache/videomme/data}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
RUN_ID="${SLURM_ARRAY_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${BASE_OUT:-/work/nvme/bglg/adeng2/qwen3vl_ttf_v2_eval_${RUN_ID}}"
PREPARE_BATCH_SIZE="${PREPARE_BATCH_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_FRAMES="${MAX_FRAMES:-64}"
TTF_RETAIN_RATIO="${TTF_RETAIN_RATIO:-0.125}"
TTF_WINDOW_RADIUS="${TTF_WINDOW_RADIUS:-1}"
TTF_ANCHOR="${TTF_ANCHOR:-auto}"
TTF_ORDER="${TTF_ORDER:-paper}"
TTF_V2_TEMPORAL_ANCHOR_RADIUS="${TTF_V2_TEMPORAL_ANCHOR_RADIUS:-2}"
TTF_DEBUG_VERIFY="${TTF_DEBUG_VERIFY:-1}"

DURATIONS=(short medium long)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
DURATION="${DURATIONS[$TASK_ID]}"
EXP_NAME="ttf_v2_f${MAX_FRAMES}_rr0125"

OUT_DIR="${BASE_OUT}/${EXP_NAME}/${DURATION}"
mkdir -p "${OUT_DIR}"
OUT_FILE="${OUT_DIR}/predictions.jsonl"

echo "experiment=${EXP_NAME}"
echo "duration=${DURATION}"
echo "max_frames=${MAX_FRAMES}"
echo "ttf_retain_ratio=${TTF_RETAIN_RATIO}"
echo "ttf_window_radius=${TTF_WINDOW_RADIUS}"
echo "ttf_anchor=${TTF_ANCHOR}"
echo "ttf_order=${TTF_ORDER}"
echo "ttf_v2_temporal_anchor_radius=${TTF_V2_TEMPORAL_ANCHOR_RADIUS}"
echo "ttf_debug_verify=${TTF_DEBUG_VERIFY}"
echo "output=${OUT_FILE}"
echo "prepare_batch_size=${PREPARE_BATCH_SIZE}"
echo "vllm_worker_multiproc_method=${VLLM_WORKER_MULTIPROC_METHOD}"
echo "force_qwenvl_video_reader=${FORCE_QWENVL_VIDEO_READER}"
echo "patch_decord_with_opencv=${QWEN3VL_PATCH_DECORD_WITH_OPENCV}"
echo "python_no_user_site=${PYTHONNOUSERSITE}"

TTF_DEBUG_ARGS=()
if [[ "${TTF_DEBUG_VERIFY}" == "1" ]]; then
  TTF_DEBUG_ARGS+=(--ttf-debug-verify)
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
  --ttf-mode post_vit \
  --ttf-version v2 \
  --ttf-budget-mode retain_ratio \
  --ttf-retain-ratio "${TTF_RETAIN_RATIO}" \
  --ttf-window-radius "${TTF_WINDOW_RADIUS}" \
  --ttf-anchor "${TTF_ANCHOR}" \
  --ttf-order "${TTF_ORDER}" \
  --ttf-v2-temporal-anchor-radius "${TTF_V2_TEMPORAL_ANCHOR_RADIUS}" \
  "${TTF_DEBUG_ARGS[@]}"
