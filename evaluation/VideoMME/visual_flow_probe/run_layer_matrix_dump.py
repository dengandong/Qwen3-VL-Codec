from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

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


LOGGER = logging.getLogger("vflow_layer_matrix_dump")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dump per-layer raw visual-token multihop responsibility matrices for VideoMME"
    )
    p.add_argument("--model-path", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--data-path", default="/work/nvme/bglg/adeng2/hf_cache/videomme")
    p.add_argument("--video-dir", default="/work/nvme/bglg/adeng2/hf_cache/videomme/data")
    p.add_argument("--output-dir", default="visualizations/vflow_layer_matrix_f8_short")
    p.add_argument("--duration", choices=["short", "medium", "long"], default="short")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--sample-ids", default=None)
    p.add_argument("--resume", action="store_true")
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
    p.add_argument("--layer-start", type=int, default=0)
    p.add_argument("--layer-end", type=int, default=None)
    p.add_argument("--layer-stride", type=int, default=1)
    p.add_argument("--save-dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--log-level", default="INFO")
    return p


def _valid_labels(annotation: dict[str, Any]) -> list[str]:
    choices = annotation.get("choices") or {}
    if isinstance(choices, dict) and choices:
        return list(choices.keys())
    answer = annotation.get("answer")
    if answer in ["A", "B", "C", "D", "E"]:
        return list("ABCDE")[: max(4, ord(str(answer)) - ord("A") + 1)]
    return list("ABCDE")


def _safe_stem(qid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(qid))


def _parse_sample_ids(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {x.strip() for x in str(value).split(",") if x.strip()}


def _filter_samples(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    sample_ids = _parse_sample_ids(args.sample_ids)
    if sample_ids is not None:
        samples = [s for s in samples if str(s.get("question_id")) in sample_ids]
    if args.start_index:
        samples = samples[int(args.start_index) :]
    if args.max_samples is not None:
        samples = samples[: int(args.max_samples)]
    return samples


def _completed_qids(manifest_path: Path) -> set[str]:
    done: set[str] = set()
    if not manifest_path.exists():
        return done
    with manifest_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("question_id")
            if qid is not None:
                done.add(str(qid))
    return done


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _layer_indices(num_layers: int, *, start: int, end: int | None, stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError("--layer-stride must be positive")
    lo = max(0, min(num_layers, int(start)))
    hi = num_layers if end is None else max(lo + 1, min(num_layers, int(end)))
    return list(range(lo, hi, int(stride)))


def _process_one(
    sample: dict[str, Any],
    *,
    rank: int,
    total: int,
    llm: Any,
    processor: Any,
    args: argparse.Namespace,
    layer_indices: list[int],
    sample_dir: Path,
) -> dict[str, Any]:
    t0 = time.time()
    qid = str(sample.get("question_id"))
    messages, annotation = build_prompt(sample, args)
    valid_labels = _valid_labels(annotation)
    prepared = prepare_vllm_prompt(processor, messages)
    baseline = deterministic_vllm_generate(llm, prepared.vllm_input, max_new_tokens=args.max_new_tokens)
    baseline_choice, parse_status = parse_choice(baseline.text, valid_labels)
    if baseline.token_ids.numel() == 0 or baseline.prompt_token_ids.numel() == 0:
        raise RuntimeError(f"{qid}: empty vLLM baseline output or prompt ids")

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
            f"{qid}: teacher sequence {teacher.input_ids.shape[1]} exceeds max_flow_seq_len={args.max_flow_seq_len}"
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

    captured_layers = sorted(capture.attentions_by_layer)
    if captured_layers != layer_indices:
        raise RuntimeError(f"{qid}: captured layers {captured_layers} != expected {layer_indices}")

    seq_len = int(teacher.input_ids.shape[1])
    curves: list[np.ndarray] = []
    for layer in captured_layers:
        attn = capture.attentions_by_layer[layer]
        if tuple(attn.shape) != (seq_len, seq_len):
            raise RuntimeError(f"{qid} layer {layer}: attention shape {tuple(attn.shape)} != {(seq_len, seq_len)}")
        reachability = compute_answer_reachability(attn, target_positions, check_causal=False)
        curve = reachability.index_select(0, visual_positions).detach().cpu()
        if not torch.isfinite(curve).all() or (curve < 0).any():
            raise RuntimeError(f"{qid} layer {layer}: non-finite or negative responsibility")
        curves.append(curve.numpy())

    dtype = np.float32 if args.save_dtype == "float32" else np.float64
    matrix = np.stack(curves, axis=0).astype(dtype, copy=False)
    stem = f"{rank:04d}_{_safe_stem(qid)}"
    npz_path = sample_dir / f"{stem}_layer_responsibility.npz"
    save_npz_atomic(
        npz_path,
        responsibility=matrix,
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
        "total": total,
        "question_id": qid,
        "skip": False,
        "duration": sample.get("duration", args.duration),
        "baseline_text": baseline.text,
        "baseline_choice": baseline_choice,
        "baseline_parse_status": parse_status,
        "ground_truth": annotation.get("answer"),
        "baseline_correct": None if baseline_choice is None else baseline_choice == annotation.get("answer"),
        "prompt_length": teacher.prompt_length,
        "sequence_length": seq_len,
        "visual_token_count": int(visual_positions.numel()),
        "responsibility_shape": list(matrix.shape),
        "target_positions": target_positions,
        "captured_layers": captured_layers,
        "capture_counts_by_layer": capture.capture_counts_by_layer,
        "video_grid_thw": mapping.video_grid_thw.tolist(),
        "video_mask_diagnostic": mask_diag,
        "npz": str(npz_path),
        "elapsed_sec": time.time() - t0,
    }
    LOGGER.info(
        "processed %d/%d qid=%s shape=%s seq_len=%d elapsed=%.1fs",
        rank,
        total,
        qid,
        tuple(matrix.shape),
        seq_len,
        meta["elapsed_sec"],
    )
    return meta


def _write_aggregate(output_dir: Path, manifest_path: Path) -> None:
    records = []
    with manifest_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("skip"):
                records.append(rec)
    if not records:
        LOGGER.warning("no valid records to aggregate")
        return

    shapes = {tuple(rec["responsibility_shape"]) for rec in records}
    if len(shapes) != 1:
        atomic_write_json(
            output_dir / "aggregate_skipped.json",
            {
                "reason": "responsibility shapes are not uniform",
                "shapes": sorted([list(s) for s in shapes]),
                "valid_records": len(records),
            },
        )
        LOGGER.warning("not writing aggregate npz because shapes differ: %s", shapes)
        return

    matrices = []
    qids = []
    prompt_lengths = []
    sequence_lengths = []
    visual_token_counts = []
    baseline_choices = []
    ground_truths = []
    target_lists = []
    for rec in records:
        with np.load(rec["npz"]) as arr:
            matrices.append(arr["responsibility"])
        qids.append(str(rec["question_id"]))
        prompt_lengths.append(int(rec["prompt_length"]))
        sequence_lengths.append(int(rec["sequence_length"]))
        visual_token_counts.append(int(rec["visual_token_count"]))
        baseline_choices.append("" if rec.get("baseline_choice") is None else str(rec.get("baseline_choice")))
        ground_truths.append("" if rec.get("ground_truth") is None else str(rec.get("ground_truth")))
        target_lists.append([int(x) for x in rec.get("target_positions", [])])

    stacked = np.stack(matrices, axis=0)
    max_targets = max((len(x) for x in target_lists), default=0)
    target_positions = np.full((len(records), max_targets), -1, dtype=np.int64)
    target_lengths = np.zeros(len(records), dtype=np.int64)
    for i, vals in enumerate(target_lists):
        target_lengths[i] = len(vals)
        if vals:
            target_positions[i, : len(vals)] = np.asarray(vals, dtype=np.int64)

    with np.load(records[0]["npz"]) as first:
        layers = first["layers"]
        visual_local_indices = first["visual_local_indices"]
        temporal_grid_indices = first["temporal_grid_indices"]
        y_grid_indices = first["y_grid_indices"]
        x_grid_indices = first["x_grid_indices"]
    save_npz_atomic(
        output_dir / "all_layer_responsibility.npz",
        responsibility=stacked,
        question_ids=np.asarray(qids),
        layers=layers,
        visual_local_indices=visual_local_indices,
        temporal_grid_indices=temporal_grid_indices,
        y_grid_indices=y_grid_indices,
        x_grid_indices=x_grid_indices,
        prompt_lengths=np.asarray(prompt_lengths, dtype=np.int64),
        sequence_lengths=np.asarray(sequence_lengths, dtype=np.int64),
        visual_token_counts=np.asarray(visual_token_counts, dtype=np.int64),
        target_positions=target_positions,
        target_position_lengths=target_lengths,
        baseline_choices=np.asarray(baseline_choices),
        ground_truths=np.asarray(ground_truths),
    )
    atomic_write_json(
        output_dir / "aggregate_summary.json",
        {
            "valid_records": len(records),
            "responsibility_shape": list(stacked.shape),
            "dtype": str(stacked.dtype),
            "aggregate_npz": str(output_dir / "all_layer_responsibility.npz"),
        },
    )
    LOGGER.info("wrote aggregate matrix shape=%s", stacked.shape)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    output_dir = Path(args.output_dir)
    sample_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    if not args.resume and manifest_path.exists():
        manifest_path.unlink()

    samples = _filter_samples(load_videomme_samples(args.data_path, duration=args.duration), args)
    if args.resume:
        done = _completed_qids(manifest_path)
        samples = [s for s in samples if str(s.get("question_id")) not in done]
        LOGGER.info("resume enabled: skipping %d completed samples", len(done))
    total = len(samples)
    LOGGER.info("samples to process: %d", total)

    llm, processor = load_vllm_and_processor(args)
    model_summary = get_vllm_model_config_summary(llm)
    num_layers = int(model_summary.get("num_hidden_layers") or 0)
    if num_layers <= 0:
        raise RuntimeError(f"could not infer decoder layer count from {model_summary}")
    layer_indices = _layer_indices(
        num_layers,
        start=int(args.layer_start),
        end=args.layer_end,
        stride=int(args.layer_stride),
    )
    atomic_write_json(
        output_dir / "run_config.json",
        {
            "args": vars(args),
            "model_config": model_summary,
            "layer_indices": layer_indices,
            "purpose": "Full-split per-layer raw multihop visual-token responsibility matrix dump.",
        },
    )

    start_rank = len(_completed_qids(manifest_path)) + 1 if args.resume else 1
    for offset, sample in enumerate(samples):
        rank = start_rank + offset
        qid = str(sample.get("question_id"))
        try:
            rec = _process_one(
                sample,
                rank=rank,
                total=total,
                llm=llm,
                processor=processor,
                args=args,
                layer_indices=layer_indices,
                sample_dir=sample_dir,
            )
        except Exception as exc:
            rec = {
                "rank": rank,
                "total": total,
                "question_id": qid,
                "skip": True,
                "skip_reason": str(exc),
                "traceback": traceback.format_exc(limit=6),
            }
            LOGGER.exception("failed qid=%s", qid)
        _append_jsonl(manifest_path, rec)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_aggregate(output_dir, manifest_path)
    LOGGER.info("matrix dump complete: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
