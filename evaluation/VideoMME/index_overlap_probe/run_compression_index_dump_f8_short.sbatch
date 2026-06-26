#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_index_dump_f8_short
#SBATCH --account=bglg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%1
#SBATCH --output=/work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs/%x-%A_%a.out
#SBATCH --error=/work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs/%x-%A_%a.err

set -euo pipefail

cd /u/adeng2/code/Qwen3-VL
mkdir -p /work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs

export QWEN3VL_ENV_BIN="${QWEN3VL_ENV_BIN:-/work/nvme/bglg/adeng2/conda_envs/qwen3vl-codec-eval/bin}"
export PATH="${QWEN3VL_ENV_BIN}:${PATH}"
export HF_HOME="${HF_HOME:-/work/nvme/bglg/adeng2/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/work/nvme/bglg/adeng2/hf_cache/datasets}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-fork}"
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_PATH="${DATA_PATH:-/work/nvme/bglg/adeng2/hf_cache/videomme}"
VIDEO_DIR="${VIDEO_DIR:-/work/nvme/bglg/adeng2/hf_cache/videomme/data}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
RUN_ID="${SLURM_ARRAY_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${BASE_OUT:-/work/nvme/bglg/adeng2/qwen3vl_index_overlap_f8_short_${RUN_ID}}"

TASKS=(
  "vcast 0.20"
  "vcast 0.80"
  "ttf 0.20"
  "ttf 0.80"
  "echoprune 0.20"
  "echoprune 0.80"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
read -r METHOD DROP_RATIO <<< "${TASKS[$TASK_ID]}"

echo "experiment=index_overlap_compression_dump"
echo "method=${METHOD}"
echo "drop_ratio=${DROP_RATIO}"
echo "duration=short"
echo "max_frames=8"
echo "output_dir=${BASE_OUT}"

python -u evaluation/VideoMME/index_overlap_probe/run_compression_index_dump.py \
  --method "${METHOD}" \
  --drop-ratio "${DROP_RATIO}" \
  --model-path "${MODEL_PATH}" \
  --data-path "${DATA_PATH}" \
  --video-dir "${VIDEO_DIR}" \
  --output-dir "${BASE_OUT}" \
  --duration short \
  --fps 2 \
  --min-frames 4 \
  --max-frames 8 \
  --min-pixels 3584 \
  --max-pixels 100352 \
  --total-pixels 3211264 \
  --seed 42 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --resume
