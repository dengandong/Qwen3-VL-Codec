#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_mmtok_sweep
#SBATCH --account=bglg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --array=0-19%1
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
BASE_OUT="${BASE_OUT:-/work/nvme/bglg/adeng2/qwen3vl_mmtok_temperature_sweep_${RUN_ID}}"
PREPARE_BATCH_SIZE="${PREPARE_BATCH_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_FRAMES="${MAX_FRAMES:-64}"
MMTOK_RETAIN_RATIO="${MMTOK_RETAIN_RATIO:-0.20}"
MMTOK_ALPHA="${MMTOK_ALPHA:-0.50}"
MMTOK_PROFILE="${MMTOK_PROFILE:-video_scalable}"
MMTOK_TEMPERATURE_MODE="${MMTOK_TEMPERATURE_MODE:-fixed}"
MMTOK_VV_TARGET_TOKENS="${MMTOK_VV_TARGET_TOKENS:-1024}"
MMTOK_GREEDY_MODE="${MMTOK_GREEDY_MODE:-stochastic}"
MMTOK_STOCHASTIC_EPSILON="${MMTOK_STOCHASTIC_EPSILON:-0.10}"
MMTOK_SELECTION_SEED="${MMTOK_SELECTION_SEED:-3407}"
MMTOK_DEBUG_VERIFY="${MMTOK_DEBUG_VERIFY:-0}"

TV_TEMPS=(0.005 0.010 0.020 0.050)
VV_TEMPS=(0.050 0.100 0.150 0.200 0.300)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
TV_INDEX=$((TASK_ID / ${#VV_TEMPS[@]}))
VV_INDEX=$((TASK_ID % ${#VV_TEMPS[@]}))
TV_TEMP="${TV_TEMPS[$TV_INDEX]}"
VV_TEMP="${VV_TEMPS[$VV_INDEX]}"

EXP_NAME="mmtok_f${MAX_FRAMES}_rr${MMTOK_RETAIN_RATIO}_tv${TV_TEMP}_vv${VV_TEMP}_${MMTOK_TEMPERATURE_MODE}"
OUT_DIR="${BASE_OUT}/${EXP_NAME}/${DURATION:-short}"
mkdir -p "${OUT_DIR}"
OUT_FILE="${OUT_DIR}/predictions.jsonl"

echo "experiment=${EXP_NAME}"
echo "duration=${DURATION:-short}"
echo "max_frames=${MAX_FRAMES}"
echo "mmtok_profile=${MMTOK_PROFILE}"
echo "mmtok_retain_ratio=${MMTOK_RETAIN_RATIO}"
echo "mmtok_alpha=${MMTOK_ALPHA}"
echo "mmtok_tv_temperature=${TV_TEMP}"
echo "mmtok_vv_temperature=${VV_TEMP}"
echo "mmtok_temperature_mode=${MMTOK_TEMPERATURE_MODE}"
echo "mmtok_vv_target_tokens=${MMTOK_VV_TARGET_TOKENS}"
echo "mmtok_greedy_mode=${MMTOK_GREEDY_MODE}"
echo "output=${OUT_FILE}"

DEBUG_ARGS=()
if [[ "${MMTOK_DEBUG_VERIFY}" == "1" ]]; then
  DEBUG_ARGS+=(--mmtok-debug-verify)
fi

SAMPLE_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  SAMPLE_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

python run_videomme.py infer \
  --model-path "${MODEL_PATH}" \
  --data-dir "${DATA_DIR}" \
  --video-dir "${VIDEO_DIR}" \
  --duration "${DURATION:-short}" \
  --output-file "${OUT_FILE}" \
  "${SAMPLE_ARGS[@]}" \
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
  --mmtok-mode post_vit \
  --mmtok-profile "${MMTOK_PROFILE}" \
  --mmtok-retain-ratio "${MMTOK_RETAIN_RATIO}" \
  --mmtok-budget-rounding floor \
  --mmtok-alpha "${MMTOK_ALPHA}" \
  --mmtok-tv-temperature "${TV_TEMP}" \
  --mmtok-vv-temperature "${VV_TEMP}" \
  --mmtok-temperature-mode "${MMTOK_TEMPERATURE_MODE}" \
  --mmtok-adaptive-vv-candidates "${MMTOK_ADAPTIVE_VV_CANDIDATES:-0.05,0.10,0.15,0.20}" \
  --mmtok-vv-target-mode stratified_3d \
  --mmtok-vv-target-tokens "${MMTOK_VV_TARGET_TOKENS}" \
  --mmtok-greedy-mode "${MMTOK_GREEDY_MODE}" \
  --mmtok-stochastic-epsilon "${MMTOK_STOCHASTIC_EPSILON}" \
  --mmtok-selection-seed "${MMTOK_SELECTION_SEED}" \
  --mmtok-query-source question_options \
  --mmtok-candidate-chunk-size "${MMTOK_CANDIDATE_CHUNK_SIZE:-512}" \
  --mmtok-target-chunk-size "${MMTOK_TARGET_CHUNK_SIZE:-256}" \
  --mmtok-exact-max-tokens "${MMTOK_EXACT_MAX_TOKENS:-1024}" \
  "${DEBUG_ARGS[@]}"
