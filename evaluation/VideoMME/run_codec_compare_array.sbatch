#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_vm_cmp
#SBATCH --account=bglg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --array=0-23%8
#SBATCH --output=/work/nvme/bglg/adeng2/qwen3vl_vllm_compare_jobs/%x-%A_%a.out
#SBATCH --error=/work/nvme/bglg/adeng2/qwen3vl_vllm_compare_jobs/%x-%A_%a.err

set -euo pipefail

cd /u/adeng2/code/Qwen3-VL/evaluation/VideoMME
mkdir -p /work/nvme/bglg/adeng2/qwen3vl_vllm_compare_jobs

export QWEN3VL_ENV_BIN="${QWEN3VL_ENV_BIN:-/work/nvme/bglg/adeng2/conda_envs/qwen3vl-codec-eval/bin}"
export PATH="${QWEN3VL_ENV_BIN}:${PATH}"
export HF_HOME="${HF_HOME:-/work/nvme/bglg/adeng2/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/work/nvme/bglg/adeng2/hf_cache/datasets}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_DIR="${DATA_DIR:-/work/nvme/bglg/adeng2/hf_cache/videomme}"
VIDEO_DIR="${VIDEO_DIR:-/work/nvme/bglg/adeng2/hf_cache/videomme/data}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
RUN_ID="${SLURM_ARRAY_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${BASE_OUT:-/work/nvme/bglg/adeng2/qwen3vl_vllm_compare_${RUN_ID}}"
PREPARE_BATCH_SIZE="${PREPARE_BATCH_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

EXPERIMENTS=(
  "codec_f128_eq16_pre_vit pre_vit f128_eq16 128"
  "codec_f128_eq16_post_vit post_vit f128_eq16 128"
  "codec_f64_eq8_pre_vit pre_vit f64_eq8 64"
  "codec_f64_eq8_post_vit post_vit f64_eq8 64"
  "native_f128 none custom 128"
  "native_f64 none custom 64"
  "native_f16 none custom 16"
  "native_f8 none custom 8"
)
DURATIONS=(short medium long)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
EXP_ID=$((TASK_ID / 3))
DUR_ID=$((TASK_ID % 3))

read -r EXP_NAME CODEC_MODE CODEC_PROFILE MAX_FRAMES <<< "${EXPERIMENTS[$EXP_ID]}"
DURATION="${DURATIONS[$DUR_ID]}"

OUT_DIR="${BASE_OUT}/${EXP_NAME}/${DURATION}"
mkdir -p "${OUT_DIR}"
OUT_FILE="${OUT_DIR}/predictions.jsonl"

echo "experiment=${EXP_NAME}"
echo "duration=${DURATION}"
echo "codec_mode=${CODEC_MODE}"
echo "codec_profile=${CODEC_PROFILE}"
echo "max_frames=${MAX_FRAMES}"
echo "output=${OUT_FILE}"

ARGS=(
  infer
  --model-path "${MODEL_PATH}"
  --data-dir "${DATA_DIR}"
  --video-dir "${VIDEO_DIR}"
  --duration "${DURATION}"
  --output-file "${OUT_FILE}"
  --prepare-batch-size "${PREPARE_BATCH_SIZE}"
  --max-new-tokens 32768
  --temperature 0.7
  --top-p 0.8
  --top-k 20
  --repetition-penalty 1.0
  --presence-penalty 1.5
  --fps 2
  --min-pixels 3584
  --max-pixels 401408
  --min-frames 4
  --max-frames "${MAX_FRAMES}"
  --total-pixels 19267584
  --tensor-parallel-size 1
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
)

if [[ "${CODEC_MODE}" != "none" ]]; then
  ARGS+=(--codec-guided-mode "${CODEC_MODE}" --codec-guide-profile "${CODEC_PROFILE}")
else
  ARGS+=(--codec-guided-mode none)
fi

python run_videomme.py "${ARGS[@]}"
