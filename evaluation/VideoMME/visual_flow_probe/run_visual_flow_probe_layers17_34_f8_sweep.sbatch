#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_vflow_l17_34_f8
#SBATCH --account=bglg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%1
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
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"

DATA_PATH="${DATA_PATH:-/work/nvme/bglg/adeng2/hf_cache/videomme}"
VIDEO_DIR="${VIDEO_DIR:-/work/nvme/bglg/adeng2/hf_cache/videomme/data}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
RUN_ID="${SLURM_ARRAY_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${BASE_OUT:-/work/nvme/bglg/adeng2/qwen3vl_visual_flow_probe_l17_34_f8_sweep_${RUN_ID}}"

TARGET_MODE="${TARGET_MODE:-decision}"
RATIOS="${RATIOS:-0.01,0.10,0.20,0.50,0.80}"
RANDOM_REPEATS="${RANDOM_REPEATS:-5}"
MAX_FRAMES="${MAX_FRAMES:-8}"
MAX_FLOW_SEQ_LEN="${MAX_FLOW_SEQ_LEN:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-8}"
SEED="${SEED:-42}"
BOOTSTRAP_RESAMPLES="${BOOTSTRAP_RESAMPLES:-2000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-3407}"

LAYER_START="${LAYER_START:-17}"
# run_visual_flow_probe.py treats layer-end as exclusive, so 35 captures 17..34.
LAYER_END="${LAYER_END:-35}"
LAYER_STRIDE="${LAYER_STRIDE:-1}"

FPS="${FPS:-2}"
MIN_FRAMES="${MIN_FRAMES:-4}"
MIN_PIXELS="${MIN_PIXELS:-3584}"
MAX_PIXELS="${MAX_PIXELS:-100352}"
TOTAL_PIXELS="${TOTAL_PIXELS:-3211264}"

DURATIONS=(short medium long)
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
DURATION="${DURATION:-${DURATIONS[$TASK_ID]}}"
OUT_DIR="${BASE_OUT}/${TARGET_MODE}_layers17_34_r01_10_20_50_80/${DURATION}"

MAX_SAMPLE_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLE_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

echo "experiment=qwen3vl_visual_flow_probe_layers17_34_f8_sweep"
echo "duration=${DURATION}"
echo "target_mode=${TARGET_MODE}"
echo "ratios=${RATIOS}"
echo "random_repeats=${RANDOM_REPEATS}"
echo "max_frames=${MAX_FRAMES}"
echo "max_flow_seq_len=${MAX_FLOW_SEQ_LEN}"
echo "layers=${LAYER_START}..$((LAYER_END - 1)) stride=${LAYER_STRIDE}"
echo "output_dir=${OUT_DIR}"
echo "data_path=${DATA_PATH}"
echo "video_dir=${VIDEO_DIR}"
echo "model_path=${MODEL_PATH}"

python -u evaluation/VideoMME/visual_flow_probe/run_visual_flow_probe.py \
  --backend vllm \
  --model-path "${MODEL_PATH}" \
  --data-path "${DATA_PATH}" \
  --video-dir "${VIDEO_DIR}" \
  --output-dir "${OUT_DIR}" \
  --duration "${DURATION}" \
  --target-mode "${TARGET_MODE}" \
  --ratios "${RATIOS}" \
  --random-repeats "${RANDOM_REPEATS}" \
  --max-frames "${MAX_FRAMES}" \
  --max-flow-seq-len "${MAX_FLOW_SEQ_LEN}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-answer-tokens "${MAX_ANSWER_TOKENS}" \
  --layer-start "${LAYER_START}" \
  --layer-end "${LAYER_END}" \
  --layer-stride "${LAYER_STRIDE}" \
  --fps "${FPS}" \
  --min-frames "${MIN_FRAMES}" \
  --min-pixels "${MIN_PIXELS}" \
  --max-pixels "${MAX_PIXELS}" \
  --total-pixels "${TOTAL_PIXELS}" \
  --seed "${SEED}" \
  --bootstrap-resamples "${BOOTSTRAP_RESAMPLES}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}" \
  --resume \
  "${MAX_SAMPLE_ARGS[@]}"
