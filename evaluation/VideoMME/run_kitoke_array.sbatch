#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_kitoke_eval
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
BASE_OUT="${BASE_OUT:-/work/nvme/bglg/adeng2/qwen3vl_kitoke_eval_${RUN_ID}}"
PREPARE_BATCH_SIZE="${PREPARE_BATCH_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

KITOKE_RETAIN_RATIO="${KITOKE_RETAIN_RATIO:-0.125}"
KITOKE_KERNEL_ALPHA="${KITOKE_KERNEL_ALPHA:-800}"
KITOKE_SELECTION_METHOD="${KITOKE_SELECTION_METHOD:-pivotal}"
KITOKE_SELECTION_SEED="${KITOKE_SELECTION_SEED:-3407}"
KITOKE_SEED_POLICY="${KITOKE_SEED_POLICY:-stable_video}"
KITOKE_PIVOTAL_PAIRING="${KITOKE_PIVOTAL_PAIRING:-random_rounds}"
KITOKE_DIFF_THRESHOLD="${KITOKE_DIFF_THRESHOLD:-110}"
KITOKE_DELTA_THRESHOLD="${KITOKE_DELTA_THRESHOLD:-70}"
KITOKE_RELATIVE_DELTA_THRESHOLD="${KITOKE_RELATIVE_DELTA_THRESHOLD:-0.4}"
KITOKE_EDGE_POLICY="${KITOKE_EDGE_POLICY:-absolute_only}"
KITOKE_EMPTY_INTERVAL_POLICY="${KITOKE_EMPTY_INTERVAL_POLICY:-repair_swap}"
KITOKE_MERGE_MODE="${KITOKE_MERGE_MODE:-weighted}"
KITOKE_DEEPSTACK_MODE="${KITOKE_DEEPSTACK_MODE:-same_weighted_merge}"
KITOKE_KERNEL_ROW_CHUNK_SIZE="${KITOKE_KERNEL_ROW_CHUNK_SIZE:-256}"
KITOKE_KERNEL_COL_CHUNK_SIZE="${KITOKE_KERNEL_COL_CHUNK_SIZE:-512}"
KITOKE_FRAME_MATCH_CHUNK_SIZE="${KITOKE_FRAME_MATCH_CHUNK_SIZE:-256}"
KITOKE_INTERVAL_MATCH_CHUNK_SIZE="${KITOKE_INTERVAL_MATCH_CHUNK_SIZE:-256}"
KITOKE_DEBUG_VERIFY="${KITOKE_DEBUG_VERIFY:-0}"

TASKS=(
  "kitoke_f64_rr0125 64 short"
  "kitoke_f64_rr0125 64 medium"
  "kitoke_f64_rr0125 64 long"
  "kitoke_f128_rr0125 128 short"
  "kitoke_f128_rr0125 128 medium"
  "kitoke_f128_rr0125 128 long"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
read -r EXP_NAME MAX_FRAMES DURATION <<< "${TASKS[$TASK_ID]}"

OUT_DIR="${BASE_OUT}/${EXP_NAME}/${DURATION}"
mkdir -p "${OUT_DIR}"
OUT_FILE="${OUT_DIR}/predictions.jsonl"

echo "experiment=${EXP_NAME}"
echo "duration=${DURATION}"
echo "max_frames=${MAX_FRAMES}"
echo "kitoke_retain_ratio=${KITOKE_RETAIN_RATIO}"
echo "kitoke_kernel_alpha=${KITOKE_KERNEL_ALPHA}"
echo "kitoke_selection_method=${KITOKE_SELECTION_METHOD}"
echo "kitoke_selection_seed=${KITOKE_SELECTION_SEED}"
echo "kitoke_merge_mode=${KITOKE_MERGE_MODE}"
echo "kitoke_deepstack_mode=${KITOKE_DEEPSTACK_MODE}"
echo "output=${OUT_FILE}"

DEBUG_ARGS=()
if [[ "${KITOKE_DEBUG_VERIFY}" == "1" ]]; then
  DEBUG_ARGS+=(--kitoke-debug-verify)
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
  --kitoke-mode post_vit \
  --kitoke-retain-ratio "${KITOKE_RETAIN_RATIO}" \
  --kitoke-kernel-alpha "${KITOKE_KERNEL_ALPHA}" \
  --kitoke-selection-method "${KITOKE_SELECTION_METHOD}" \
  --kitoke-selection-seed "${KITOKE_SELECTION_SEED}" \
  --kitoke-seed-policy "${KITOKE_SEED_POLICY}" \
  --kitoke-pivotal-pairing "${KITOKE_PIVOTAL_PAIRING}" \
  --kitoke-diff-threshold "${KITOKE_DIFF_THRESHOLD}" \
  --kitoke-delta-threshold "${KITOKE_DELTA_THRESHOLD}" \
  --kitoke-relative-delta-threshold "${KITOKE_RELATIVE_DELTA_THRESHOLD}" \
  --kitoke-edge-policy "${KITOKE_EDGE_POLICY}" \
  --kitoke-empty-interval-policy "${KITOKE_EMPTY_INTERVAL_POLICY}" \
  --kitoke-merge-mode "${KITOKE_MERGE_MODE}" \
  --kitoke-deepstack-mode "${KITOKE_DEEPSTACK_MODE}" \
  --kitoke-kernel-row-chunk-size "${KITOKE_KERNEL_ROW_CHUNK_SIZE}" \
  --kitoke-kernel-col-chunk-size "${KITOKE_KERNEL_COL_CHUNK_SIZE}" \
  --kitoke-frame-match-chunk-size "${KITOKE_FRAME_MATCH_CHUNK_SIZE}" \
  --kitoke-interval-match-chunk-size "${KITOKE_INTERVAL_MATCH_CHUNK_SIZE}" \
  "${DEBUG_ARGS[@]}"
