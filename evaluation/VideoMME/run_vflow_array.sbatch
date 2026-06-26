#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_vflow_eval
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
BASE_OUT="${BASE_OUT:-/work/nvme/bglg/adeng2/qwen3vl_vflow_eval_${RUN_ID}}"
PREPARE_BATCH_SIZE="${PREPARE_BATCH_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

VFLOW_RETAIN_RATIO="${VFLOW_RETAIN_RATIO:-0.125}"
VFLOW_SIGNAL="${VFLOW_SIGNAL:-responsibility}"
VFLOW_KEEP="${VFLOW_KEEP:-high}"
VFLOW_RESPONSIBILITY_DIR="${VFLOW_RESPONSIBILITY_DIR:-/work/nvme/bglg/adeng2/qwen3vl_visual_flow_probe_l17_34_f8_sweep_2560780/decision_layers17_34_r01_10_20_50_80}"
VFLOW_DEBUG_VERIFY="${VFLOW_DEBUG_VERIFY:-0}"
VFLOW_QUIET="${VFLOW_QUIET:-0}"

TASKS=(
  "vflow_f64_rr0125 64 short"
  "vflow_f64_rr0125 64 medium"
  "vflow_f64_rr0125 64 long"
  "vflow_f128_rr0125 128 short"
  "vflow_f128_rr0125 128 medium"
  "vflow_f128_rr0125 128 long"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
read -r EXP_NAME MAX_FRAMES DURATION <<< "${TASKS[$TASK_ID]}"

OUT_DIR="${BASE_OUT}/${EXP_NAME}/${DURATION}"
mkdir -p "${OUT_DIR}"
OUT_FILE="${OUT_DIR}/predictions.jsonl"

echo "experiment=${EXP_NAME}"
echo "duration=${DURATION}"
echo "max_frames=${MAX_FRAMES}"
echo "vflow_retain_ratio=${VFLOW_RETAIN_RATIO}"
echo "vflow_signal=${VFLOW_SIGNAL}"
echo "vflow_keep=${VFLOW_KEEP}"
echo "vflow_responsibility_dir=${VFLOW_RESPONSIBILITY_DIR}"
echo "output=${OUT_FILE}"

DEBUG_ARGS=()
if [[ "${VFLOW_DEBUG_VERIFY}" == "1" ]]; then
  DEBUG_ARGS+=(--vflow-debug-verify)
fi
if [[ "${VFLOW_QUIET}" == "1" ]]; then
  DEBUG_ARGS+=(--vflow-quiet)
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
  --vflow-mode post_vit \
  --vflow-retain-ratio "${VFLOW_RETAIN_RATIO}" \
  --vflow-responsibility-dir "${VFLOW_RESPONSIBILITY_DIR}" \
  --vflow-signal "${VFLOW_SIGNAL}" \
  --vflow-keep "${VFLOW_KEEP}" \
  "${DEBUG_ARGS[@]}"
