#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-fork}"

DATA_DIR="${DATA_DIR:-/path/to/videomme}"
VIDEO_DIR="${VIDEO_DIR:-${DATA_DIR}/data}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
BASE_OUT="${BASE_OUT:-${REPO_ROOT}/outputs/qwen3vl_codec_eval_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
PREPARE_BATCH_SIZE="${PREPARE_BATCH_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-900}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

mkdir -p "${BASE_OUT}" "${LOG_DIR}"

TASKS=(
  "codec_f128_eq16_pre_vit pre_vit f128_eq16 128 short"
  "codec_f128_eq16_pre_vit pre_vit f128_eq16 128 medium"
  "codec_f128_eq16_pre_vit pre_vit f128_eq16 128 long"
  "codec_f128_eq16_post_vit post_vit f128_eq16 128 short"
  "codec_f128_eq16_post_vit post_vit f128_eq16 128 medium"
  "codec_f128_eq16_post_vit post_vit f128_eq16 128 long"
  "codec_f64_eq8_pre_vit pre_vit f64_eq8 64 short"
  "codec_f64_eq8_pre_vit pre_vit f64_eq8 64 medium"
  "codec_f64_eq8_pre_vit pre_vit f64_eq8 64 long"
  "codec_f64_eq8_post_vit post_vit f64_eq8 64 short"
  "codec_f64_eq8_post_vit post_vit f64_eq8 64 medium"
  "codec_f64_eq8_post_vit post_vit f64_eq8 64 long"
)

running=0
task_idx=0
for task in "${TASKS[@]}"; do
  read -r EXP_NAME CODEC_MODE CODEC_PROFILE MAX_FRAMES DURATION <<< "${task}"
  OUT_DIR="${BASE_OUT}/${EXP_NAME}/${DURATION}"
  OUT_FILE="${OUT_DIR}/predictions.jsonl"
  LOG_FILE="${LOG_DIR}/${EXP_NAME}_${DURATION}.log"
  mkdir -p "${OUT_DIR}"

  if [[ -f "${OUT_FILE}" ]] && [[ "$(wc -l < "${OUT_FILE}")" -ge "${EXPECTED_SAMPLES}" ]]; then
    echo "[skip] ${EXP_NAME}/${DURATION}: ${OUT_FILE}"
    continue
  fi

  gpu=$((task_idx % MAX_PARALLEL))
  task_idx=$((task_idx + 1))

  echo "[launch] gpu=${gpu} ${EXP_NAME}/${DURATION}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
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
      --codec-guided-mode "${CODEC_MODE}" \
      --codec-guide-profile "${CODEC_PROFILE}"
  ) > "${LOG_FILE}" 2>&1 &

  running=$((running + 1))
  if [[ "${running}" -ge "${MAX_PARALLEL}" ]]; then
    wait -n
    running=$((running - 1))
  fi
done

wait
echo "[done] outputs: ${BASE_OUT}"
