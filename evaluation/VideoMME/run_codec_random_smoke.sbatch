#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_rand_smoke
#SBATCH --account=bglg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/work/nvme/bglg/adeng2/qwen3vl_codec_smoke_jobs/%x-%A_%a.out
#SBATCH --error=/work/nvme/bglg/adeng2/qwen3vl_codec_smoke_jobs/%x-%A_%a.err

set -euo pipefail

cd /u/adeng2/code/Qwen3-VL/evaluation/VideoMME
mkdir -p /work/nvme/bglg/adeng2/qwen3vl_codec_smoke_jobs

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
BASE_OUT="${BASE_OUT:-/work/nvme/bglg/adeng2/qwen3vl_codec_random_smoke_${RUN_ID}}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
CODEC_RANDOM_SEED="${CODEC_RANDOM_SEED:-3407}"

EXPERIMENTS=(
  "random_f64_eq8_post_vit f64_eq8 64"
  "random_f128_eq16_post_vit f128_eq16 128"
)

read -r EXP_NAME CODEC_PROFILE MAX_FRAMES <<< "${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="${BASE_OUT}/${EXP_NAME}"
OUT_FILE="${OUT_DIR}/predictions.jsonl"
mkdir -p "${OUT_DIR}"

echo "experiment=${EXP_NAME}"
echo "codec_profile=${CODEC_PROFILE}"
echo "max_frames=${MAX_FRAMES}"
echo "codec_random_seed=${CODEC_RANDOM_SEED}"
echo "output=${OUT_FILE}"
echo "force_qwenvl_video_reader=${FORCE_QWENVL_VIDEO_READER}"
echo "patch_decord_with_opencv=${QWEN3VL_PATCH_DECORD_WITH_OPENCV}"
echo "python_no_user_site=${PYTHONNOUSERSITE}"

python run_videomme.py infer \
  --model-path "${MODEL_PATH}" \
  --data-dir "${DATA_DIR}" \
  --video-dir "${VIDEO_DIR}" \
  --duration short \
  --output-file "${OUT_FILE}" \
  --max-samples "${MAX_SAMPLES}" \
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
  --max-frames "${MAX_FRAMES}" \
  --total-pixels 117964800 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 131072 \
  --codec-guided-mode post_vit \
  --codec-guide-profile "${CODEC_PROFILE}" \
  --codec-selection-strategy random \
  --codec-random-seed "${CODEC_RANDOM_SEED}"
