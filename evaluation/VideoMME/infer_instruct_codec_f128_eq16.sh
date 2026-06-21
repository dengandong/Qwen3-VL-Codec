#!/bin/bash

# Codec-guided VideoMME inference for Qwen3-VL with vLLM.
# Guidance profile: max_frames=128, equivalent 16-frame token budget.
# Set CODEC_GUIDED_MODE=post_vit to run post-ViT token-level pruning.

CODEC_GUIDED_MODE="${CODEC_GUIDED_MODE:-pre_vit}"

python run_videomme.py infer \
    --model-path /path/to/Qwen3-VL-Instruct \
    --data-dir /path/to/VideoMME \
    --duration short \
    --output-file "results/videomme_short_codec_f128_eq16_${CODEC_GUIDED_MODE}.jsonl" \
    --codec-guided-mode "${CODEC_GUIDED_MODE}" \
    --codec-guide-profile f128_eq16 \
    --max-new-tokens 32768 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --repetition-penalty 1.0 \
    --presence-penalty 1.5 \
    --fps 2 \
    --min-pixels 3584 \
    --max-pixels 401408 \
    --min-frames 4 \
    --max-frames 128 \
    --total-pixels 19267584

