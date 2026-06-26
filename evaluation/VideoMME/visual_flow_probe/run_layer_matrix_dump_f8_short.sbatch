#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_vflow_layer_matrix_f8_short
#SBATCH --account=bglg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=06:00:00
#SBATCH --output=/work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs/%x-%j.out
#SBATCH --error=/work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs/%x-%j.err

set -euo pipefail

cd /u/adeng2/code/Qwen3-VL
mkdir -p /work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs

export QWEN3VL_ENV_BIN="${QWEN3VL_ENV_BIN:-/work/nvme/bglg/adeng2/conda_envs/qwen3vl-codec-eval/bin}"
export PATH="${QWEN3VL_ENV_BIN}:${PATH}"
export HF_HOME="${HF_HOME:-/work/nvme/bglg/adeng2/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/work/nvme/bglg/adeng2/hf_cache/datasets}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"

DATA_PATH="${DATA_PATH:-/work/nvme/bglg/adeng2/hf_cache/videomme}"
VIDEO_DIR="${VIDEO_DIR:-/work/nvme/bglg/adeng2/hf_cache/videomme/data}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
RUN_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-visualizations/vflow_layer_matrix_f8_short_${RUN_ID}}"

TARGET_MODE="${TARGET_MODE:-decision}"
MAX_FRAMES="${MAX_FRAMES:-8}"
MAX_FLOW_SEQ_LEN="${MAX_FLOW_SEQ_LEN:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-8}"
SEED="${SEED:-42}"
SAVE_DTYPE="${SAVE_DTYPE:-float32}"

FPS="${FPS:-2}"
MIN_FRAMES="${MIN_FRAMES:-4}"
MIN_PIXELS="${MIN_PIXELS:-3584}"
MAX_PIXELS="${MAX_PIXELS:-100352}"
TOTAL_PIXELS="${TOTAL_PIXELS:-3211264}"

MAX_SAMPLE_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLE_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

echo "experiment=qwen3vl_vflow_layer_matrix_f8_short"
echo "duration=short"
echo "target_mode=${TARGET_MODE}"
echo "max_frames=${MAX_FRAMES}"
echo "max_flow_seq_len=${MAX_FLOW_SEQ_LEN}"
echo "layers=0..35"
echo "output_dir=${OUTPUT_DIR}"
echo "save_dtype=${SAVE_DTYPE}"
echo "data_path=${DATA_PATH}"
echo "video_dir=${VIDEO_DIR}"
echo "model_path=${MODEL_PATH}"

python -u evaluation/VideoMME/visual_flow_probe/run_layer_matrix_dump.py \
  --model-path "${MODEL_PATH}" \
  --data-path "${DATA_PATH}" \
  --video-dir "${VIDEO_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --duration short \
  --target-mode "${TARGET_MODE}" \
  --max-frames "${MAX_FRAMES}" \
  --max-flow-seq-len "${MAX_FLOW_SEQ_LEN}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-answer-tokens "${MAX_ANSWER_TOKENS}" \
  --fps "${FPS}" \
  --min-frames "${MIN_FRAMES}" \
  --min-pixels "${MIN_PIXELS}" \
  --max-pixels "${MAX_PIXELS}" \
  --total-pixels "${TOTAL_PIXELS}" \
  --layer-start 0 \
  --layer-end 36 \
  --layer-stride 1 \
  --save-dtype "${SAVE_DTYPE}" \
  --seed "${SEED}" \
  --resume \
  "${MAX_SAMPLE_ARGS[@]}"
