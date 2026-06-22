#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_vcast_smoke
#SBATCH --account=bglg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs/%x-%j.out
#SBATCH --error=/work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs/%x-%j.err

set -euo pipefail

cd /u/adeng2/code/Qwen3-VL/evaluation/VideoMME
mkdir -p /work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs

export PATH="/projects/bffz/adeng2/conda_envs/longvt-rft-eval/bin:${PATH}"
export HF_HOME="${HF_HOME:-/work/nvme/bglg/adeng2/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/work/nvme/bglg/adeng2/hf_cache/datasets}"
export VLLM_WORKER_MULTIPROC_METHOD=fork
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT_DIR="${OUT_DIR:-/work/nvme/bglg/adeng2/qwen3vl_vcast_smoke_${SLURM_JOB_ID:-manual}}"
mkdir -p "${OUT_DIR}"

python run_videomme.py infer \
  --model-path "${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}" \
  --data-dir "${DATA_DIR:-/work/nvme/bglg/adeng2/hf_cache/videomme}" \
  --video-dir "${VIDEO_DIR:-/work/nvme/bglg/adeng2/hf_cache/videomme/data}" \
  --duration "${DURATION:-short}" \
  --output-file "${OUT_DIR}/predictions.jsonl" \
  --max-samples "${MAX_SAMPLES:-1}" \
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
  --max-frames "${MAX_FRAMES:-64}" \
  --total-pixels 117964800 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 131072 \
  --vcast-mode post_vit \
  --vcast-retain-ratio "${VCAST_RETAIN_RATIO:-0.25}" \
  --vcast-min-k "${VCAST_MIN_K:-1}"
