#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_index_overlap_summary
#SBATCH --account=bglg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs/%x-%j.out
#SBATCH --error=/work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs/%x-%j.err

set -euo pipefail

cd /u/adeng2/code/Qwen3-VL
mkdir -p /work/nvme/bglg/adeng2/qwen3vl_codec_eval_jobs

export QWEN3VL_ENV_BIN="${QWEN3VL_ENV_BIN:-/work/nvme/bglg/adeng2/conda_envs/qwen3vl-codec-eval/bin}"
export PATH="${QWEN3VL_ENV_BIN}:${PATH}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

DUMP_ROOT="${DUMP_ROOT:?DUMP_ROOT is required}"
VFLOW_RESPONSIBILITY_DIR="${VFLOW_RESPONSIBILITY_DIR:?VFLOW_RESPONSIBILITY_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:-${DUMP_ROOT}/overlap_with_vflow_l17_34}"

echo "experiment=index_overlap_summary"
echo "dump_root=${DUMP_ROOT}"
echo "vflow_responsibility_dir=${VFLOW_RESPONSIBILITY_DIR}"
echo "output_dir=${OUTPUT_DIR}"

python -u evaluation/VideoMME/index_overlap_probe/compare_vflow_overlap.py \
  --dump-root "${DUMP_ROOT}" \
  --vflow-responsibility-dir "${VFLOW_RESPONSIBILITY_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --methods vcast,ttf,echoprune,random_drop \
  --drop-ratios 0.20,0.80 \
  --vflow-kinds top,bottom \
  --random-repeats 5 \
  --seed 42
