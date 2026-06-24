#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_ttf_framesim
#SBATCH --account=bglg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs/%x-%j.out
#SBATCH --error=/work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs/%x-%j.err

set -euo pipefail

cd /u/adeng2/code/Qwen3-VL/evaluation/VideoMME
mkdir -p /work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs

export QWEN3VL_ENV_BIN="${QWEN3VL_ENV_BIN:-/work/nvme/bglg/adeng2/conda_envs/qwen3vl-codec-eval/bin}"
export PATH="${QWEN3VL_ENV_BIN}:${PATH}"
export HF_HOME="${HF_HOME:-/work/nvme/bglg/adeng2/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/work/nvme/bglg/adeng2/hf_cache/datasets}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python plot_ttf_frame_similarity.py \
  --model-path "${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}" \
  --data-dir "${DATA_DIR:-/work/nvme/bglg/adeng2/hf_cache/videomme}" \
  --video-dir "${VIDEO_DIR:-/work/nvme/bglg/adeng2/hf_cache/videomme/data}" \
  --duration "${DURATION:-short}" \
  --num-videos "${NUM_VIDEOS:-5}" \
  --fps "${FPS:-2}" \
  --min-frames "${MIN_FRAMES:-4}" \
  --max-frames "${MAX_FRAMES:-128}" \
  --min-pixels "${MIN_PIXELS:-3584}" \
  --max-pixels "${MAX_PIXELS:-786432}" \
  --total-pixels "${TOTAL_PIXELS:-117964800}" \
  --output-dir "${OUTPUT_DIR:-/work/nvme/bglg/adeng2/qwen3vl_ttf_frame_similarity}"
