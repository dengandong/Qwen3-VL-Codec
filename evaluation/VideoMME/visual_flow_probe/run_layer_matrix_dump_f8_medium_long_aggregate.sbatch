#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_vflow_layer_matrix_f8_ml_agg
#SBATCH --account=bglg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH --output=/work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs/%x-%A_%a.out
#SBATCH --error=/work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs/%x-%A_%a.err

set -euo pipefail

cd /u/adeng2/code/Qwen3-VL
mkdir -p /work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs

export QWEN3VL_ENV_BIN="${QWEN3VL_ENV_BIN:-/work/nvme/bglg/adeng2/conda_envs/qwen3vl-codec-eval/bin}"
export PATH="${QWEN3VL_ENV_BIN}:${PATH}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

SOURCE_RUN_ID="${SOURCE_RUN_ID:?SOURCE_RUN_ID is required}"
TASKS=(
  "medium"
  "long"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
DURATION="${TASKS[$TASK_ID]}"
OUTPUT_DIR="visualizations/vflow_layer_matrix_f8_${DURATION}_${SOURCE_RUN_ID}"

echo "aggregate_duration=${DURATION}"
echo "source_run_id=${SOURCE_RUN_ID}"
echo "output_dir=${OUTPUT_DIR}"

python -u evaluation/VideoMME/visual_flow_probe/aggregate_layer_matrix_dump.py \
  --output-dir "${OUTPUT_DIR}"
