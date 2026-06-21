#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_codec_smoke
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

export PATH="/projects/bffz/adeng2/conda_envs/longvt-rft-eval/bin:${PATH}"
export HF_HOME="${HF_HOME:-/work/nvme/bglg/adeng2/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/work/nvme/bglg/adeng2/hf_cache/datasets}"
export VLLM_WORKER_MULTIPROC_METHOD=fork
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_DIR="${DATA_DIR:-/work/nvme/bglg/adeng2/hf_cache/videomme}"
VIDEO_DIR="${VIDEO_DIR:-/work/nvme/bglg/adeng2/hf_cache/videomme/data}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
RUN_ID="${SLURM_ARRAY_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${BASE_OUT:-/work/nvme/bglg/adeng2/qwen3vl_codec_smoke_${RUN_ID}}"

MODES=(pre_vit post_vit)
MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="${BASE_OUT}/${MODE}"
OUT_FILE="${OUT_DIR}/predictions.jsonl"
mkdir -p "${OUT_DIR}"

echo "mode=${MODE}"
echo "output=${OUT_FILE}"

python run_videomme.py infer \
  --model-path "${MODEL_PATH}" \
  --data-dir "${DATA_DIR}" \
  --video-dir "${VIDEO_DIR}" \
  --duration short \
  --output-file "${OUT_FILE}" \
  --max-samples 1 \
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
  --max-frames 64 \
  --total-pixels 117964800 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 32768 \
  --codec-guided-mode "${MODE}" \
  --codec-guide-profile f64_eq8
