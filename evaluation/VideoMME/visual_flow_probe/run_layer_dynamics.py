from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from visual_flow_probe.flow import compute_answer_reachability, map_video_tokens_to_grid, visual_positions_from_input_ids  # type: ignore
    from visual_flow_probe.io_utils import atomic_write_json, build_prompt, load_videomme_samples, save_npz_atomic  # type: ignore
    from visual_flow_probe.metrics import parse_choice  # type: ignore
    from visual_flow_probe.vllm_adapter import (  # type: ignore
        build_vllm_teacher_forced,
        capture_vllm_decoder_attention_by_layer,
        deterministic_vllm_generate,
        get_vllm_model_config_summary,
        get_vllm_spatial_merge_size,
        get_vllm_video_token_id,
        load_vllm_and_processor,
        prepare_vllm_prompt,
        select_vllm_target_positions,
    )
else:
    from .flow import compute_answer_reachability, map_video_tokens_to_grid, visual_positions_from_input_ids
    from .io_utils import atomic_write_json, build_prompt, load_videomme_samples, save_npz_atomic
    from .metrics import parse_choice
    from .vllm_adapter import (
        build_vllm_teacher_forced,
        capture_vllm_decoder_attention_by_layer,
        deterministic_vllm_generate,
        get_vllm_model_config_summary,
        get_vllm_spatial_merge_size,
        get_vllm_video_token_id,
        load_vllm_and_processor,
        prepare_vllm_prompt,
        select_vllm_target_positions,
    )


LOGGER = logging.getLogger("vflow_layer_dynamics")


@dataclass(frozen=True)
class Candidate:
    question_id: str
    confidence: float
    baseline_logprob_mean: float
    baseline_choice: str | None
    ground_truth: str | None
    baseline_correct: bool | None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot per-layer visual-token reachability curves for VFlow examples")
    p.add_argument("--model-path", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--data-path", default="/work/nvme/bglg/adeng2/hf_cache/videomme")
    p.add_argument("--video-dir", default="/work/nvme/bglg/adeng2/hf_cache/videomme/data")
    p.add_argument("--source-results", default="/work/nvme/bglg/adeng2/qwen3vl_visual_flow_probe_vllm_r10_20260624_192312/decision_r10/short/results.jsonl")
    p.add_argument("--output-dir", default="visualizations/vflow_layer_dynamics_f8_short")
    p.add_argument("--duration", choices=["short", "medium", "long"], default="short")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--require-baseline-correct", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--target-mode", choices=["decision", "post_answer"], default="decision")
    p.add_argument("--max-answer-tokens", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--max-flow-seq-len", type=int, default=2048)
    p.add_argument("--causal-atol", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=float, default=2)
    p.add_argument("--min-frames", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--min-pixels", type=int, default=3584)
    p.add_argument("--max-pixels", type=int, default=100352)
    p.add_argument("--total-pixels", type=int, default=3211264)
    p.add_argument("--use-subtitle", action="store_true")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--log-level", default="INFO")
    return p


def _valid_labels(annotation: dict[str, Any]) -> list[str]:
    choices = annotation.get("choices") or {}
    if isinstance(choices, dict) and choices:
        return list(choices.keys())
    return list("ABCDE")


def load_high_confidence_candidates(path: Path, *, top_k: int, require_correct: bool) -> list[Candidate]:
    if not path.exists():
        raise FileNotFoundError(f"source results not found: {path}")
    candidates: list[Candidate] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("skip"):
                continue
            if rec.get("baseline_parse_status") != "ok":
                continue
            baseline_correct = rec.get("baseline_correct")
            if require_correct and baseline_correct is not True:
                continue
            lp = float(rec.get("baseline_answer_logprob_mean", float("-inf")))
            if not math.isfinite(lp):
                continue
            confidence = float(math.exp(min(0.0, lp)))
            candidates.append(
                Candidate(
                    question_id=str(rec.get("question_id")),
                    confidence=confidence,
                    baseline_logprob_mean=lp,
                    baseline_choice=rec.get("baseline_choice"),
                    ground_truth=rec.get("ground_truth_choice"),
                    baseline_correct=baseline_correct if isinstance(baseline_correct, bool) else None,
                )
            )
    candidates.sort(key=lambda x: (-x.confidence, -x.baseline_logprob_mean, x.question_id))
    if len(candidates) < top_k:
        raise RuntimeError(f"only found {len(candidates)} high-confidence candidates in {path}; requested {top_k}")
    return candidates[:top_k]


def _layer_color(layer: int, total: int) -> tuple[int, int, int]:
    # HSV-like palette with full saturation/value, implemented without matplotlib.
    h = (float(layer) / max(1.0, float(total))) * 6.0
    i = int(h) % 6
    f = h - int(h)
    q = int(255 * (1.0 - f))
    t = int(255 * f)
    if i == 0:
        return 255, t, 0
    if i == 1:
        return q, 255, 0
    if i == 2:
        return 0, 255, t
    if i == 3:
        return 0, q, 255
    if i == 4:
        return t, 0, 255
    return 255, 0, q


def render_layer_curve_png(
    path: Path,
    curves: np.ndarray,
    layers: list[int],
    *,
    title: str,
    confidence: float,
    baseline_text: str,
    target_positions: list[int],
) -> None:
    """Render raw per-layer h[visual_token] curves with a shared linear y-axis."""
    if curves.ndim != 2:
        raise ValueError(f"curves must be [layers, visual_tokens], got {curves.shape}")
    height = 950
    width = 1650
    left, right, top, bottom = 92, 260, 82, 96
    plot_w = width - left - right
    plot_h = height - top - bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    finite = curves[np.isfinite(curves)]
    ymax = float(finite.max()) if finite.size else 1.0
    ymin = 0.0
    if ymax <= 0:
        ymax = 1.0
    n = curves.shape[1]

    def x_to_px(i: int) -> int:
        if n <= 1:
            return left
        return int(left + (i / (n - 1)) * plot_w)

    def y_to_px(v: float) -> int:
        frac = (float(v) - ymin) / (ymax - ymin)
        frac = max(0.0, min(1.0, frac))
        return int(top + (1.0 - frac) * plot_h)

    # Axes and grid.
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(20, 20, 20), width=1)
    for g in range(6):
        yv = ymin + (ymax - ymin) * g / 5.0
        y = y_to_px(yv)
        draw.line([left, y, left + plot_w, y], fill=(230, 230, 230))
        draw.text((8, y - 7), f"{yv:.3g}", fill=(0, 0, 0), font=font)
    for g in range(6):
        xv = int(round((n - 1) * g / 5.0)) if n > 1 else 0
        x = x_to_px(xv)
        draw.line([x, top, x, top + plot_h], fill=(238, 238, 238))
        draw.text((x - 16, top + plot_h + 8), str(xv), fill=(0, 0, 0), font=font)

    draw.text((left, 16), title, fill=(0, 0, 0), font=font)
    draw.text(
        (left, 36),
        f"raw multihop reachability h[visual]; confidence={confidence:.4f}; baseline={baseline_text!r}; targets={target_positions}",
        fill=(0, 0, 0),
        font=font,
    )
    draw.text((left + plot_w // 2 - 70, height - 36), "visual token local index", fill=(0, 0, 0), font=font)
    draw.text((8, 58), "raw h", fill=(0, 0, 0), font=font)

    total_layers = max(layers) + 1 if layers else curves.shape[0]
    for row, layer in enumerate(layers):
        color = _layer_color(layer, total_layers)
        vals = curves[row]
        points = [(x_to_px(i), y_to_px(float(v))) for i, v in enumerate(vals) if math.isfinite(float(v))]
        if len(points) >= 2:
            draw.line(points, fill=color, width=1)
        elif len(points) == 1:
            x, y = points[0]
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=color)

    # Compact legend.
    legend_x = left + plot_w + 18
    legend_y = top
    for row, layer in enumerate(layers):
        y = legend_y + row * 21
        color = _layer_color(layer, total_layers)
        draw.line([legend_x, y + 7, legend_x + 28, y + 7], fill=color, width=3)
        draw.text((legend_x + 34, y), f"layer {layer}", fill=(0, 0, 0), font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    img.save(tmp, format="PNG")
    tmp.replace(path)


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    random_seed = int(args.seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    candidates = load_high_confidence_candidates(
        Path(args.source_results),
        top_k=int(args.top_k),
        require_correct=bool(args.require_baseline_correct),
    )
    qids = {c.question_id for c in candidates}
    by_qid = {str(s.get("question_id")): s for s in load_videomme_samples(args.data_path, duration=args.duration)}
    missing = sorted(qids - set(by_qid))
    if missing:
        raise RuntimeError(f"source candidates missing from VideoMME {args.duration} split: {missing}")

    llm, processor = load_vllm_and_processor(args)
    model_summary = get_vllm_model_config_summary(llm)
    num_layers = int(model_summary.get("num_hidden_layers") or 0)
    if num_layers <= 0:
        raise RuntimeError(f"could not infer decoder layer count from {model_summary}")
    layer_indices = list(range(num_layers))

    atomic_write_json(
        output_dir / "run_config.json",
        {
            "args": vars(args),
            "selected_candidates": [c.__dict__ for c in candidates],
            "model_config": model_summary,
            "layer_indices": layer_indices,
            "purpose": "Per-layer raw multihop visual-token reachability curves; no visual-token normalization.",
        },
    )

    for rank, cand in enumerate(candidates, start=1):
        t0 = time.time()
        sample = by_qid[cand.question_id]
        messages, annotation = build_prompt(sample, args)
        valid_labels = _valid_labels(annotation)
        prepared = prepare_vllm_prompt(processor, messages)
        baseline = deterministic_vllm_generate(llm, prepared.vllm_input, max_new_tokens=args.max_new_tokens)
        baseline_choice, parse_status = parse_choice(baseline.text, valid_labels)
        if baseline.token_ids.numel() == 0 or baseline.prompt_token_ids.numel() == 0:
            raise RuntimeError(f"{cand.question_id}: empty vLLM baseline output or prompt ids")

        teacher = build_vllm_teacher_forced(
            prepared,
            baseline.text,
            baseline.prompt_token_ids,
            baseline.token_ids,
            max_answer_tokens=args.max_answer_tokens,
        )
        target_positions = select_vllm_target_positions(teacher, args.target_mode)
        if int(teacher.input_ids.shape[1]) > int(args.max_flow_seq_len):
            raise RuntimeError(
                f"{cand.question_id}: teacher sequence {teacher.input_ids.shape[1]} exceeds max_flow_seq_len={args.max_flow_seq_len}"
            )

        capture = capture_vllm_decoder_attention_by_layer(
            llm,
            teacher.vllm_input,
            layer_indices=layer_indices,
            max_seq_len=args.max_flow_seq_len,
            causal_atol=args.causal_atol,
        )
        if capture.prompt_token_ids.numel() > 0:
            teacher.input_ids = capture.prompt_token_ids.view(1, -1).to(torch.long)
            actual_answer = teacher.input_ids[0, teacher.prompt_length :].detach().cpu().to(torch.long)
            if args.max_answer_tokens is not None:
                actual_answer = actual_answer[: int(args.max_answer_tokens)]
            if actual_answer.numel() > 0:
                teacher.answer_token_ids = actual_answer
                teacher.answer_positions = list(range(teacher.prompt_length, teacher.prompt_length + int(actual_answer.numel())))
                teacher.decision_target_positions = [p - 1 for p in teacher.answer_positions]
                teacher.post_answer_target_positions = teacher.answer_positions
                target_positions = select_vllm_target_positions(teacher, args.target_mode)

        video_token_id = get_vllm_video_token_id(processor, llm)
        visual_positions, mask_diag = visual_positions_from_input_ids(teacher.input_ids.detach().cpu(), video_token_id)
        spatial_merge_size = get_vllm_spatial_merge_size(processor, llm)
        mapping = map_video_tokens_to_grid(visual_positions, teacher.video_grid_thw.detach().cpu(), spatial_merge_size)

        curves = []
        captured_layers = sorted(capture.attentions_by_layer)
        if captured_layers != layer_indices:
            raise RuntimeError(f"{cand.question_id}: captured layers {captured_layers} != expected {layer_indices}")
        seq_len = int(teacher.input_ids.shape[1])
        for layer in captured_layers:
            attn = capture.attentions_by_layer[layer]
            if tuple(attn.shape) != (seq_len, seq_len):
                raise RuntimeError(f"{cand.question_id} layer {layer}: attention shape {tuple(attn.shape)} != {(seq_len, seq_len)}")
            reachability = compute_answer_reachability(attn, target_positions, check_causal=False)
            curve = reachability.index_select(0, visual_positions).detach().cpu().to(torch.float64)
            if not torch.isfinite(curve).all() or (curve < 0).any():
                raise RuntimeError(f"{cand.question_id} layer {layer}: non-finite or negative curve")
            curves.append(curve.numpy())

        curve_arr = np.stack(curves, axis=0)
        stem = f"{rank:02d}_{cand.question_id}"
        png_path = output_dir / f"{stem}_layer_curves.png"
        npz_path = output_dir / f"{stem}_layer_curves.npz"
        json_path = output_dir / f"{stem}_metadata.json"
        render_layer_curve_png(
            png_path,
            curve_arr,
            captured_layers,
            title=f"{stem}: VFlow raw per-layer visual reachability",
            confidence=cand.confidence,
            baseline_text=baseline.text,
            target_positions=target_positions,
        )
        save_npz_atomic(
            npz_path,
            curves=curve_arr,
            layers=np.asarray(captured_layers, dtype=np.int64),
            visual_seq_positions=mapping.visual_seq_positions,
            visual_local_indices=mapping.visual_local_indices,
            temporal_grid_indices=mapping.temporal_grid_indices,
            y_grid_indices=mapping.y_grid_indices,
            x_grid_indices=mapping.x_grid_indices,
            video_grid_thw=mapping.video_grid_thw,
            target_positions=np.asarray(target_positions, dtype=np.int64),
        )
        meta = {
            "rank": rank,
            "question_id": cand.question_id,
            "confidence_from_source": cand.confidence,
            "source_baseline_logprob_mean": cand.baseline_logprob_mean,
            "source_baseline_choice": cand.baseline_choice,
            "source_ground_truth": cand.ground_truth,
            "source_baseline_correct": cand.baseline_correct,
            "rerun_baseline_text": baseline.text,
            "rerun_baseline_choice": baseline_choice,
            "rerun_parse_status": parse_status,
            "ground_truth": annotation.get("answer"),
            "prompt_length": teacher.prompt_length,
            "sequence_length": seq_len,
            "visual_token_count": int(visual_positions.numel()),
            "target_positions": target_positions,
            "captured_layers": captured_layers,
            "capture_counts_by_layer": capture.capture_counts_by_layer,
            "video_grid_thw": mapping.video_grid_thw.tolist(),
            "video_mask_diagnostic": mask_diag,
            "png": str(png_path),
            "npz": str(npz_path),
            "elapsed_sec": time.time() - t0,
        }
        atomic_write_json(json_path, meta)
        _write_jsonl(manifest_path, meta)
        LOGGER.info(
            "wrote %s visual_tokens=%d seq_len=%d elapsed=%.1fs",
            png_path,
            int(visual_positions.numel()),
            seq_len,
            meta["elapsed_sec"],
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    LOGGER.info("layer dynamics complete: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
