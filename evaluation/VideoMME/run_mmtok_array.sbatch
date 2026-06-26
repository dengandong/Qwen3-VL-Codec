#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_mmtok_eval
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
BASE_OUT="${BASE_OUT:-/work/nvme/bglg/adeng2/qwen3vl_mmtok_eval_${RUN_ID}}"
PREPARE_BATCH_SIZE="${PREPARE_BATCH_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

MMTOK_PROFILE="${MMTOK_PROFILE:-video_scalable}"
MMTOK_RETAIN_RATIO="${MMTOK_RETAIN_RATIO:-0.125}"
MMTOK_ALPHA="${MMTOK_ALPHA:-0.50}"
MMTOK_TV_TEMPERATURE="${MMTOK_TV_TEMPERATURE:-0.0100}"
MMTOK_VV_TEMPERATURE="${MMTOK_VV_TEMPERATURE:-0.2000}"
MMTOK_TEMPERATURE_MODE="${MMTOK_TEMPERATURE_MODE:-fixed}"
MMTOK_VV_TARGET_MODE="${MMTOK_VV_TARGET_MODE:-stratified_3d}"
MMTOK_VV_TARGET_TOKENS="${MMTOK_VV_TARGET_TOKENS:-1024}"
MMTOK_GREEDY_MODE="${MMTOK_GREEDY_MODE:-stochastic}"
MMTOK_STOCHASTIC_EPSILON="${MMTOK_STOCHASTIC_EPSILON:-0.10}"
MMTOK_SELECTION_SEED="${MMTOK_SELECTION_SEED:-3407}"
MMTOK_QUERY_SOURCE="${MMTOK_QUERY_SOURCE:-question_options}"
MMTOK_CANDIDATE_CHUNK_SIZE="${MMTOK_CANDIDATE_CHUNK_SIZE:-512}"
MMTOK_TARGET_CHUNK_SIZE="${MMTOK_TARGET_CHUNK_SIZE:-256}"
MMTOK_EXACT_MAX_TOKENS="${MMTOK_EXACT_MAX_TOKENS:-1024}"
MMTOK_DEBUG_VERIFY="${MMTOK_DEBUG_VERIFY:-0}"

TASKS=(
  "mmtok_f64_rr0125 64 short"
  "mmtok_f64_rr0125 64 medium"
  "mmtok_f64_rr0125 64 long"
  "mmtok_f128_rr0125 128 short"
  "mmtok_f128_rr0125 128 medium"
  "mmtok_f128_rr0125 128 long"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
read -r EXP_NAME MAX_FRAMES DURATION <<< "${TASKS[$TASK_ID]}"

OUT_DIR="${BASE_OUT}/${EXP_NAME}/${DURATION}"
mkdir -p "${OUT_DIR}"
OUT_FILE="${OUT_DIR}/predictions.jsonl"

echo "experiment=${EXP_NAME}"
echo "duration=${DURATION}"
echo "max_frames=${MAX_FRAMES}"
echo "mmtok_profile=${MMTOK_PROFILE}"
echo "mmtok_retain_ratio=${MMTOK_RETAIN_RATIO}"
echo "mmtok_alpha=${MMTOK_ALPHA}"
echo "mmtok_tv_temperature=${MMTOK_TV_TEMPERATURE}"
echo "mmtok_vv_temperature=${MMTOK_VV_TEMPERATURE}"
echo "mmtok_temperature_mode=${MMTOK_TEMPERATURE_MODE}"
echo "mmtok_vv_target_mode=${MMTOK_VV_TARGET_MODE}"
echo "mmtok_vv_target_tokens=${MMTOK_VV_TARGET_TOKENS}"
echo "mmtok_greedy_mode=${MMTOK_GREEDY_MODE}"
echo "mmtok_stochastic_epsilon=${MMTOK_STOCHASTIC_EPSILON}"
echo "mmtok_selection_seed=${MMTOK_SELECTION_SEED}"
echo "output=${OUT_FILE}"
echo "prepare_batch_size=${PREPARE_BATCH_SIZE}"
echo "vllm_worker_multiproc_method=${VLLM_WORKER_MULTIPROC_METHOD}"
echo "force_qwenvl_video_reader=${FORCE_QWENVL_VIDEO_READER}"
echo "patch_decord_with_opencv=${QWEN3VL_PATCH_DECORD_WITH_OPENCV}"
echo "python_no_user_site=${PYTHONNOUSERSITE}"

DEBUG_ARGS=()
if [[ "${MMTOK_DEBUG_VERIFY}" == "1" ]]; then
  DEBUG_ARGS+=(--mmtok-debug-verify)
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
  --mmtok-mode post_vit \
  --mmtok-profile "${MMTOK_PROFILE}" \
  --mmtok-retain-ratio "${MMTOK_RETAIN_RATIO}" \
  --mmtok-budget-rounding floor \
  --mmtok-alpha "${MMTOK_ALPHA}" \
  --mmtok-tv-temperature "${MMTOK_TV_TEMPERATURE}" \
  --mmtok-vv-temperature "${MMTOK_VV_TEMPERATURE}" \
  --mmtok-temperature-mode "${MMTOK_TEMPERATURE_MODE}" \
  --mmtok-adaptive-vv-candidates "${MMTOK_ADAPTIVE_VV_CANDIDATES:-0.05,0.10,0.15,0.20}" \
  --mmtok-vv-target-mode "${MMTOK_VV_TARGET_MODE}" \
  --mmtok-vv-target-tokens "${MMTOK_VV_TARGET_TOKENS}" \
  --mmtok-greedy-mode "${MMTOK_GREEDY_MODE}" \
  --mmtok-stochastic-epsilon "${MMTOK_STOCHASTIC_EPSILON}" \
  --mmtok-selection-seed "${MMTOK_SELECTION_SEED}" \
  --mmtok-query-source "${MMTOK_QUERY_SOURCE}" \
  --mmtok-candidate-chunk-size "${MMTOK_CANDIDATE_CHUNK_SIZE}" \
  --mmtok-target-chunk-size "${MMTOK_TARGET_CHUNK_SIZE}" \
  --mmtok-exact-max-tokens "${MMTOK_EXACT_MAX_TOKENS}" \
  "${DEBUG_ARGS[@]}"
