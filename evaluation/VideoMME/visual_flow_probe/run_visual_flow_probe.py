from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from visual_flow_probe.attention_capture import (  # type: ignore
        capture_averaged_decoder_attention,
        default_layer_indices,
        find_decoder_self_attention_modules,
    )
    from visual_flow_probe.flow import (  # type: ignore
        compute_answer_reachability,
        concentration_statistics,
        direct_target_attention,
        map_video_tokens_to_grid,
        normalize_visual_responsibility,
        visual_positions_from_input_ids,
    )
    from visual_flow_probe.interventions import build_selection_sets, stable_int_seed, zero_selected_visual_values  # type: ignore
    from visual_flow_probe.io_utils import (  # type: ignore
        append_jsonl,
        atomic_write_json,
        build_prompt,
        completed_question_ids,
        git_info,
        load_videomme_samples,
        read_jsonl,
        save_npz_atomic,
    )
    from visual_flow_probe.metrics import parse_choice, write_summary_files  # type: ignore
    from visual_flow_probe.model_adapter import (  # type: ignore
        answer_logprob,
        build_teacher_forced_inputs,
        deterministic_generate,
        get_spatial_merge_size,
        get_video_token_id,
        input_without_mm_token_type,
        load_hf_model_and_processor,
        model_config_summary,
        move_to_device,
        prepare_prompt_inputs,
        select_target_positions,
    )
    from visual_flow_probe.vllm_adapter import (  # type: ignore
        build_vllm_teacher_forced,
        capture_vllm_decoder_attention,
        deterministic_vllm_generate,
        get_vllm_model_config_summary,
        get_vllm_spatial_merge_size,
        get_vllm_video_token_id,
        load_vllm_and_processor,
        prepare_vllm_prompt,
        select_vllm_target_positions,
        vllm_answer_logprob,
        vllm_zero_generate_and_logprob,
    )
else:
    from .attention_capture import (
        capture_averaged_decoder_attention,
        default_layer_indices,
        find_decoder_self_attention_modules,
    )
    from .flow import (
        compute_answer_reachability,
        concentration_statistics,
        direct_target_attention,
        map_video_tokens_to_grid,
        normalize_visual_responsibility,
        visual_positions_from_input_ids,
    )
    from .interventions import build_selection_sets, stable_int_seed, zero_selected_visual_values
    from .io_utils import (
        append_jsonl,
        atomic_write_json,
        build_prompt,
        completed_question_ids,
        git_info,
        load_videomme_samples,
        read_jsonl,
        save_npz_atomic,
    )
    from .metrics import parse_choice, write_summary_files
    from .model_adapter import (
        answer_logprob,
        build_teacher_forced_inputs,
        deterministic_generate,
        get_spatial_merge_size,
        get_video_token_id,
        input_without_mm_token_type,
        load_hf_model_and_processor,
        model_config_summary,
        move_to_device,
        prepare_prompt_inputs,
        select_target_positions,
    )
    from .vllm_adapter import (
        build_vllm_teacher_forced,
        capture_vllm_decoder_attention,
        deterministic_vllm_generate,
        get_vllm_model_config_summary,
        get_vllm_spatial_merge_size,
        get_vllm_video_token_id,
        load_vllm_and_processor,
        prepare_vllm_prompt,
        select_vllm_target_positions,
        vllm_answer_logprob,
        vllm_zero_generate_and_logprob,
    )


LOGGER = logging.getLogger("visual_flow_probe")


def _parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def _parse_csv_strings(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VideoMME visual-token responsibility Phase-1 probe")
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", required=True, help="VideoMME dataset directory/HF path or parquet file")
    p.add_argument("--video-dir", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--duration", choices=["short", "medium", "long", "all"], default="short")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--sample-ids", default=None, help="Comma-separated question_id list")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--fps", type=float, default=2)
    p.add_argument("--min-frames", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--min-pixels", type=int, default=3584)
    p.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    p.add_argument("--total-pixels", type=int, default=4096 * 28 * 28)
    p.add_argument("--use-subtitle", action="store_true")

    p.add_argument("--target-mode", choices=["decision", "post_answer"], default="decision")
    p.add_argument("--max-answer-tokens", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--max-flow-seq-len", type=int, default=2048)

    p.add_argument("--layer-start", type=int, default=None)
    p.add_argument("--layer-end", type=int, default=None)
    p.add_argument("--layer-stride", type=int, default=1)
    p.add_argument("--causal-atol", type=float, default=1e-4)

    p.add_argument("--ratios", default="0.20")
    p.add_argument("--random-repeats", type=int, default=5)
    p.add_argument("--bootstrap-resamples", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=3407)
    p.add_argument("--no-verify-hook-removal", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--log-level", default="INFO")
    return p


def _filter_samples(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    sample_ids = _parse_csv_strings(args.sample_ids)
    if sample_ids is not None:
        samples = [s for s in samples if str(s.get("question_id")) in sample_ids]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.num_shards > 1:
        samples = [s for i, s in enumerate(samples) if i % args.num_shards == args.shard_index]
    if args.start_index:
        samples = samples[args.start_index :]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    return samples


def _valid_labels(annotation: dict[str, Any]) -> list[str]:
    choices = annotation.get("choices") or {}
    if isinstance(choices, dict):
        return list(choices.keys())
    answer = annotation.get("answer")
    if answer in ["A", "B", "C", "D", "E"]:
        return list("ABCDE")[: max(4, ord(answer) - ord("A") + 1)]
    return list("ABCDE")


def _sample_base_record(
    sample: dict[str, Any],
    annotation: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "question_id": str(sample.get("question_id", annotation.get("question_id") if annotation else "")),
        "video_id": sample.get("videoID"),
        "duration": sample.get("duration", annotation.get("duration") if annotation else args.duration),
        "category": sample.get("category", sample.get("domain")),
        "task_category": sample.get("task_category", sample.get("task_type")),
        "target_mode": args.target_mode,
        "preprocessing": {
            "fps": args.fps,
            "min_frames": args.min_frames,
            "max_frames": args.max_frames,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "total_pixels": args.total_pixels,
            "use_subtitle": bool(args.use_subtitle),
        },
    }


def _skip_record(
    sample: dict[str, Any],
    annotation: dict[str, Any] | None,
    args: argparse.Namespace,
    *,
    stage: str,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec = _sample_base_record(sample, annotation, args)
    rec.update({"skip": True, "skip_stage": stage, "skip_reason": reason})
    if extra:
        rec["skip_extra"] = extra
    return rec


def _torch_cuda_info() -> dict[str, Any]:
    info = {"cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}
    if torch.cuda.is_available():
        info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def _npz_selected_payload(selections: dict[str, Any]) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    for score_type, by_ratio in selections.items():
        for ratio, by_cond in by_ratio.items():
            ratio_key = str(ratio).replace(".", "p")
            for condition, value in by_cond.items():
                if isinstance(value, list):
                    for rep, arr in enumerate(value):
                        arrays[f"selected_{score_type}_{ratio_key}_{condition}_{rep}"] = arr
                else:
                    arrays[f"selected_{score_type}_{ratio_key}_{condition}"] = value
    return arrays


def _run_intervention(
    model: torch.nn.Module,
    processor: Any,
    prompt_model_inputs: dict[str, torch.Tensor],
    teacher_model_inputs: dict[str, torch.Tensor],
    selected_seq_positions: torch.Tensor,
    *,
    prompt_length: int,
    answer_token_ids: torch.Tensor,
    baseline_choice: str | None,
    ground_truth: str | None,
    valid_labels: list[str],
    max_new_tokens: int,
    baseline_logprob_total: float,
) -> dict[str, Any]:
    with zero_selected_visual_values(model, selected_seq_positions) as state:
        gen_text, gen_ids, _ = deterministic_generate(
            model,
            processor,
            prompt_model_inputs,
            max_new_tokens=max_new_tokens,
        )
        lp_total, lp_mean = answer_logprob(
            model,
            teacher_model_inputs,
            prompt_length=prompt_length,
            answer_token_ids=answer_token_ids,
        )
    if state.modified_calls <= 0:
        raise RuntimeError("value-zero hooks were installed but no v_proj output was modified")
    choice, parse_status = parse_choice(gen_text, valid_labels)
    correct = None if choice is None or ground_truth is None else choice == ground_truth
    baseline_correct = None if baseline_choice is None or ground_truth is None else baseline_choice == ground_truth
    return {
        "generated_text": gen_text,
        "generated_token_ids": gen_ids.tolist(),
        "parsed_choice": choice,
        "parse_status": parse_status,
        "answer_changed": None if baseline_choice is None or choice is None else choice != baseline_choice,
        "correct": correct,
        "correct_to_wrong": bool(baseline_correct and correct is False),
        "wrong_to_correct": bool(baseline_correct is False and correct is True),
        "baseline_answer_logprob_total": lp_total,
        "baseline_answer_logprob_mean": lp_mean,
        "logprob_drop": baseline_logprob_total - lp_total,
        "zero_hook_modified_calls": state.modified_calls,
        "zero_hook_module_count": state.module_count,
    }


def _run_vllm_intervention(
    llm: Any,
    processor: Any,
    prompt_vllm_input: dict[str, Any],
    teacher_vllm_input: dict[str, Any],
    selected_seq_positions: torch.Tensor,
    *,
    prompt_length: int,
    answer_token_ids: torch.Tensor,
    baseline_choice: str | None,
    ground_truth: str | None,
    valid_labels: list[str],
    max_new_tokens: int,
    baseline_logprob_total: float,
) -> dict[str, Any]:
    gen, lp_total, lp_mean, zero_state = vllm_zero_generate_and_logprob(
        llm,
        prompt_vllm_input,
        teacher_vllm_input,
        selected_seq_positions,
        max_new_tokens=max_new_tokens,
        prompt_length=prompt_length,
        answer_token_ids=answer_token_ids,
    )
    choice, parse_status = parse_choice(gen.text, valid_labels)
    correct = None if choice is None or ground_truth is None else choice == ground_truth
    baseline_correct = None if baseline_choice is None or ground_truth is None else baseline_choice == ground_truth
    return {
        "generated_text": gen.text,
        "generated_token_ids": gen.token_ids.tolist(),
        "parsed_choice": choice,
        "parse_status": parse_status,
        "answer_changed": None if baseline_choice is None or choice is None else choice != baseline_choice,
        "correct": correct,
        "correct_to_wrong": bool(baseline_correct and correct is False),
        "wrong_to_correct": bool(baseline_correct is False and correct is True),
        "baseline_answer_logprob_total": lp_total,
        "baseline_answer_logprob_mean": lp_mean,
        "logprob_drop": baseline_logprob_total - lp_total,
        "zero_hook_modified_calls": int(zero_state.get("modified_calls", 0)),
        "zero_hook_module_count": int(zero_state.get("module_count", 0)),
        "intervention_backend": "vllm_qkv_v_slice_zero",
    }


def process_sample_vllm(
    sample: dict[str, Any],
    *,
    llm: Any,
    processor: Any,
    args: argparse.Namespace,
    layer_indices: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    annotation = None
    try:
        messages, annotation = build_prompt(sample, args)
        rec = _sample_base_record(sample, annotation, args)
        valid_labels = _valid_labels(annotation)
        ground_truth = annotation.get("answer")

        prepared = prepare_vllm_prompt(processor, messages)
        baseline = deterministic_vllm_generate(
            llm,
            prepared.vllm_input,
            max_new_tokens=args.max_new_tokens,
        )
        baseline_choice, baseline_parse_status = parse_choice(baseline.text, valid_labels)
        baseline_correct = None if baseline_choice is None else baseline_choice == ground_truth
        if baseline.token_ids.numel() == 0:
            return _skip_record(sample, annotation, args, stage="baseline", reason="empty_baseline_answer")
        if baseline.prompt_token_ids.numel() == 0:
            return _skip_record(sample, annotation, args, stage="vllm_output", reason="missing_prompt_token_ids")

        teacher = build_vllm_teacher_forced(
            prepared,
            baseline.text,
            baseline.prompt_token_ids,
            baseline.token_ids,
            max_answer_tokens=args.max_answer_tokens,
        )
        target_positions = select_vllm_target_positions(teacher, args.target_mode)
        if int(teacher.input_ids.shape[1]) > args.max_flow_seq_len:
            return _skip_record(
                sample,
                annotation,
                args,
                stage="flow_length_guard",
                reason="sequence_exceeds_max_flow_seq_len",
                extra={"seq_len": int(teacher.input_ids.shape[1]), "max_flow_seq_len": args.max_flow_seq_len},
            )

        attn_result = capture_vllm_decoder_attention(
            llm,
            teacher.vllm_input,
            layer_indices=layer_indices,
            max_seq_len=args.max_flow_seq_len,
            causal_atol=args.causal_atol,
        )
        attention = attn_result.attention
        if attn_result.prompt_token_ids.numel() > 0:
            teacher.input_ids = attn_result.prompt_token_ids.view(1, -1).to(torch.long)
            actual_answer = teacher.input_ids[0, teacher.prompt_length :].detach().cpu().to(torch.long)
            if args.max_answer_tokens is not None:
                actual_answer = actual_answer[: int(args.max_answer_tokens)]
            if actual_answer.numel() > 0:
                teacher.answer_token_ids = actual_answer
                teacher.answer_positions = list(
                    range(teacher.prompt_length, teacher.prompt_length + int(actual_answer.numel()))
                )
                teacher.decision_target_positions = [p - 1 for p in teacher.answer_positions]
                teacher.post_answer_target_positions = teacher.answer_positions
                target_positions = select_vllm_target_positions(teacher, args.target_mode)

        if tuple(attention.shape) != (int(teacher.input_ids.shape[1]), int(teacher.input_ids.shape[1])):
            return _skip_record(
                sample,
                annotation,
                args,
                stage="attention_capture",
                reason="attention_shape_mismatch",
                extra={
                    "attention_shape": list(attention.shape),
                    "teacher_input_len": int(teacher.input_ids.shape[1]),
                },
            )

        video_token_id = get_vllm_video_token_id(processor, llm)
        visual_positions, mask_diag = visual_positions_from_input_ids(
            teacher.input_ids.detach().cpu(),
            video_token_id,
            mm_token_type_ids=None,
        )
        if visual_positions.numel() == 0:
            return _skip_record(sample, annotation, args, stage="video_mask", reason="no_video_tokens", extra=mask_diag)
        spatial_merge_size = get_vllm_spatial_merge_size(processor, llm)
        mapping = map_video_tokens_to_grid(
            visual_positions,
            teacher.video_grid_thw.detach().cpu(),
            spatial_merge_size,
        )

        h = compute_answer_reachability(attention, target_positions)
        responsibility = normalize_visual_responsibility(h, visual_positions)
        direct = direct_target_attention(attention, target_positions, visual_positions)
        score_stats = concentration_statistics(responsibility, temporal_grid_indices=mapping.temporal_grid_indices)

        baseline_lp_total, baseline_lp_mean = vllm_answer_logprob(
            llm,
            teacher.vllm_input,
            prompt_length=teacher.prompt_length,
            answer_token_ids=teacher.answer_token_ids,
        )

        ratios = _parse_csv_floats(args.ratios)
        selections = build_selection_sets(
            {"responsibility": responsibility, "direct_attention": direct},
            ratios,
            mapping.temporal_grid_indices,
            question_id=str(annotation["question_id"]),
            seed=args.seed,
            random_repeats=args.random_repeats,
        )
        interventions: list[dict[str, Any]] = []
        for score_type, by_ratio in selections.items():
            for ratio, by_cond in by_ratio.items():
                for condition, selected in by_cond.items():
                    reps = selected if isinstance(selected, list) else [selected]
                    for rep_idx, local_indices in enumerate(reps):
                        selected_seq_positions = mapping.visual_seq_positions.index_select(0, local_indices)
                        row = _run_vllm_intervention(
                            llm,
                            processor,
                            prepared.vllm_input,
                            teacher.vllm_input,
                            selected_seq_positions,
                            prompt_length=teacher.prompt_length,
                            answer_token_ids=teacher.answer_token_ids,
                            baseline_choice=baseline_choice,
                            ground_truth=ground_truth,
                            valid_labels=valid_labels,
                            max_new_tokens=args.max_new_tokens,
                            baseline_logprob_total=baseline_lp_total,
                        )
                        row.update(
                            {
                                "target_mode": args.target_mode,
                                "score_type": score_type,
                                "ratio": float(ratio),
                                "condition": condition,
                                "repeat": rep_idx if isinstance(selected, list) else None,
                                "selected_local_indices": local_indices.tolist(),
                                "selected_seq_positions": selected_seq_positions.tolist(),
                                "seed": stable_int_seed(args.seed, annotation["question_id"], ratio, rep_idx, condition),
                            }
                        )
                        interventions.append(row)

        hook_removal_verified = None
        if not args.no_verify_hook_removal:
            verify = deterministic_vllm_generate(
                llm,
                prepared.vllm_input,
                max_new_tokens=args.max_new_tokens,
            )
            hook_removal_verified = bool(torch.equal(verify.token_ids, baseline.token_ids) and verify.text == baseline.text)
            if not hook_removal_verified:
                raise RuntimeError(
                    "vLLM baseline generation changed after clearing intervention state; "
                    f"baseline={baseline.text!r} verify={verify.text!r}"
                )

        npz_payload = {
            "visual_seq_positions": mapping.visual_seq_positions,
            "visual_local_indices": mapping.visual_local_indices,
            "responsibility": responsibility,
            "direct_attention": direct,
            "temporal_grid_indices": mapping.temporal_grid_indices,
            "y_grid_indices": mapping.y_grid_indices,
            "x_grid_indices": mapping.x_grid_indices,
            "video_indices": mapping.video_indices,
            "video_grid_thw": mapping.video_grid_thw,
            "target_positions": torch.tensor(target_positions, dtype=torch.long),
            **_npz_selected_payload(selections),
        }
        resp_path = output_dir / "responsibilities" / f"{annotation['question_id']}.npz"
        save_npz_atomic(resp_path, **npz_payload)

        rec.update(
            {
                "skip": False,
                "backend": "vllm",
                "annotation": annotation,
                "prompt_length": int(baseline.prompt_token_ids.numel()),
                "flow_sequence_length": int(teacher.input_ids.shape[1]),
                "visual_token_count": int(visual_positions.numel()),
                "video_mask_diagnostic": mask_diag,
                "video_grid_thw": mapping.video_grid_thw.tolist(),
                "video_metadata": prepared.video_metadata,
                "spatial_merge_size": spatial_merge_size,
                "selected_layer_indices": attn_result.layer_indices,
                "captured_attention_layers": attn_result.captured_layers,
                "captured_attention_calls": attn_result.captured_calls,
                "target_positions": target_positions,
                "answer_positions": teacher.answer_positions,
                "baseline_generated_text": baseline.text,
                "baseline_generated_token_ids": baseline.token_ids.tolist(),
                "baseline_full_output_ids": list(getattr(baseline.raw_output.outputs[0], "token_ids", []) or baseline.token_ids.tolist()),
                "baseline_choice": baseline_choice,
                "baseline_parse_status": baseline_parse_status,
                "ground_truth_choice": ground_truth,
                "baseline_correct": baseline_correct,
                "baseline_answer_logprob_total": baseline_lp_total,
                "baseline_answer_logprob_mean": baseline_lp_mean,
                "score_concentration": score_stats,
                "responsibility_npz": str(resp_path),
                "interventions": interventions,
                "random_seed": args.seed,
                "hook_removal_verified": hook_removal_verified,
            }
        )
        return rec
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _skip_record(
            sample,
            annotation,
            args,
            stage="cuda_oom",
            reason=str(exc),
            extra={"traceback": traceback.format_exc(limit=5)},
        )
    except Exception as exc:
        return _skip_record(
            sample,
            annotation,
            args,
            stage="exception",
            reason=f"{type(exc).__name__}: {exc}",
            extra={"traceback": traceback.format_exc(limit=10)},
        )


def process_sample(
    sample: dict[str, Any],
    *,
    model: torch.nn.Module,
    processor: Any,
    args: argparse.Namespace,
    layer_indices: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    annotation = None
    try:
        messages, annotation = build_prompt(sample, args)
        rec = _sample_base_record(sample, annotation, args)
        valid_labels = _valid_labels(annotation)
        ground_truth = annotation.get("answer")

        prepared = prepare_prompt_inputs(processor, messages)
        prompt_inputs_cpu = prepared.inputs
        prompt_len = int(prompt_inputs_cpu["input_ids"].shape[1])
        prompt_inputs_device = move_to_device(prompt_inputs_cpu, next(model.parameters()).device)
        prompt_model_inputs = input_without_mm_token_type(prompt_inputs_device)

        baseline_text, baseline_answer_ids, full_output_ids = deterministic_generate(
            model,
            processor,
            prompt_model_inputs,
            max_new_tokens=args.max_new_tokens,
        )
        baseline_choice, baseline_parse_status = parse_choice(baseline_text, valid_labels)
        baseline_correct = None if baseline_choice is None else baseline_choice == ground_truth
        if baseline_answer_ids.numel() == 0:
            return _skip_record(sample, annotation, args, stage="baseline", reason="empty_baseline_answer")

        teacher = build_teacher_forced_inputs(
            prompt_inputs_device,
            baseline_answer_ids,
            max_answer_tokens=args.max_answer_tokens,
        )
        target_positions = select_target_positions(teacher, args.target_mode)
        if int(teacher.inputs["input_ids"].shape[1]) > args.max_flow_seq_len:
            return _skip_record(
                sample,
                annotation,
                args,
                stage="flow_length_guard",
                reason="sequence_exceeds_max_flow_seq_len",
                extra={"seq_len": int(teacher.inputs["input_ids"].shape[1]), "max_flow_seq_len": args.max_flow_seq_len},
            )

        teacher_model_inputs = input_without_mm_token_type(teacher.inputs)
        attn_result = capture_averaged_decoder_attention(
            model,
            teacher_model_inputs,
            layer_indices=layer_indices,
            causal_atol=args.causal_atol,
        )
        attention = attn_result.attention

        video_token_id = get_video_token_id(model, processor)
        visual_positions, mask_diag = visual_positions_from_input_ids(
            teacher.inputs["input_ids"].detach().cpu(),
            video_token_id,
            mm_token_type_ids=teacher.inputs.get("mm_token_type_ids"),
        )
        if visual_positions.numel() == 0:
            return _skip_record(sample, annotation, args, stage="video_mask", reason="no_video_tokens", extra=mask_diag)
        spatial_merge_size = get_spatial_merge_size(model, processor)
        mapping = map_video_tokens_to_grid(
            visual_positions,
            teacher.inputs["video_grid_thw"].detach().cpu(),
            spatial_merge_size,
        )

        h = compute_answer_reachability(attention, target_positions)
        responsibility = normalize_visual_responsibility(h, visual_positions)
        direct = direct_target_attention(attention, target_positions, visual_positions)
        score_stats = concentration_statistics(responsibility, temporal_grid_indices=mapping.temporal_grid_indices)

        baseline_lp_total, baseline_lp_mean = answer_logprob(
            model,
            teacher_model_inputs,
            prompt_length=teacher.prompt_length,
            answer_token_ids=teacher.answer_token_ids,
        )

        ratios = _parse_csv_floats(args.ratios)
        selections = build_selection_sets(
            {"responsibility": responsibility, "direct_attention": direct},
            ratios,
            mapping.temporal_grid_indices,
            question_id=str(annotation["question_id"]),
            seed=args.seed,
            random_repeats=args.random_repeats,
        )
        interventions: list[dict[str, Any]] = []
        for score_type, by_ratio in selections.items():
            for ratio, by_cond in by_ratio.items():
                for condition, selected in by_cond.items():
                    reps = selected if isinstance(selected, list) else [selected]
                    for rep_idx, local_indices in enumerate(reps):
                        selected_seq_positions = mapping.visual_seq_positions.index_select(0, local_indices)
                        row = _run_intervention(
                            model,
                            processor,
                            prompt_model_inputs,
                            teacher_model_inputs,
                            selected_seq_positions,
                            prompt_length=teacher.prompt_length,
                            answer_token_ids=teacher.answer_token_ids,
                            baseline_choice=baseline_choice,
                            ground_truth=ground_truth,
                            valid_labels=valid_labels,
                            max_new_tokens=args.max_new_tokens,
                            baseline_logprob_total=baseline_lp_total,
                        )
                        row.update(
                            {
                                "target_mode": args.target_mode,
                                "score_type": score_type,
                                "ratio": float(ratio),
                                "condition": condition,
                                "repeat": rep_idx if isinstance(selected, list) else None,
                                "selected_local_indices": local_indices.tolist(),
                                "selected_seq_positions": selected_seq_positions.tolist(),
                                "seed": stable_int_seed(args.seed, annotation["question_id"], ratio, rep_idx, condition),
                            }
                        )
                        interventions.append(row)

        hook_removal_verified = None
        if not args.no_verify_hook_removal:
            verify_text, verify_ids, _ = deterministic_generate(
                model,
                processor,
                prompt_model_inputs,
                max_new_tokens=args.max_new_tokens,
            )
            hook_removal_verified = bool(torch.equal(verify_ids, baseline_answer_ids) and verify_text == baseline_text)
            if not hook_removal_verified:
                raise RuntimeError(
                    "baseline generation changed after removing intervention hooks; "
                    f"baseline={baseline_text!r} verify={verify_text!r}"
                )

        npz_payload = {
            "visual_seq_positions": mapping.visual_seq_positions,
            "visual_local_indices": mapping.visual_local_indices,
            "responsibility": responsibility,
            "direct_attention": direct,
            "temporal_grid_indices": mapping.temporal_grid_indices,
            "y_grid_indices": mapping.y_grid_indices,
            "x_grid_indices": mapping.x_grid_indices,
            "video_indices": mapping.video_indices,
            "video_grid_thw": mapping.video_grid_thw,
            "target_positions": torch.tensor(target_positions, dtype=torch.long),
            **_npz_selected_payload(selections),
        }
        resp_path = output_dir / "responsibilities" / f"{annotation['question_id']}.npz"
        save_npz_atomic(resp_path, **npz_payload)

        rec.update(
            {
                "skip": False,
                "annotation": annotation,
                "prompt_length": prompt_len,
                "flow_sequence_length": int(teacher.inputs["input_ids"].shape[1]),
                "visual_token_count": int(visual_positions.numel()),
                "video_mask_diagnostic": mask_diag,
                "video_grid_thw": mapping.video_grid_thw.tolist(),
                "spatial_merge_size": spatial_merge_size,
                "selected_layer_indices": attn_result.layer_indices,
                "captured_attention_layers": attn_result.captured_layers,
                "target_positions": target_positions,
                "answer_positions": teacher.answer_positions,
                "baseline_generated_text": baseline_text,
                "baseline_generated_token_ids": baseline_answer_ids.tolist(),
                "baseline_full_output_ids": full_output_ids[0].tolist(),
                "baseline_choice": baseline_choice,
                "baseline_parse_status": baseline_parse_status,
                "ground_truth_choice": ground_truth,
                "baseline_correct": baseline_correct,
                "baseline_answer_logprob_total": baseline_lp_total,
                "baseline_answer_logprob_mean": baseline_lp_mean,
                "score_concentration": score_stats,
                "responsibility_npz": str(resp_path),
                "interventions": interventions,
                "random_seed": args.seed,
                "hook_removal_verified": hook_removal_verified,
            }
        )
        return rec
    except torch.cuda.OutOfMemoryError as exc:
        for p in model.parameters():
            if p.is_cuda:
                torch.cuda.empty_cache()
                break
        return _skip_record(
            sample,
            annotation,
            args,
            stage="cuda_oom",
            reason=str(exc),
            extra={"traceback": traceback.format_exc(limit=5)},
        )
    except Exception as exc:
        return _skip_record(
            sample,
            annotation,
            args,
            stage="exception",
            reason=f"{type(exc).__name__}: {exc}",
            extra={"traceback": traceback.format_exc(limit=10)},
        )


def save_run_config(args: argparse.Namespace, model: torch.nn.Module, layer_indices: list[int], output_dir: Path) -> None:
    import transformers

    repo_root = Path(__file__).resolve().parents[3]
    cfg = {
        "args": vars(args),
        "model_path": args.model_path,
        "model_config": model_config_summary(model),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "selected_layer_indices": layer_indices,
        "git": git_info(repo_root),
        "cuda": _torch_cuda_info(),
    }
    atomic_write_json(output_dir / "run_config.json", cfg)


def save_run_config_vllm(args: argparse.Namespace, llm: Any, layer_indices: list[int], output_dir: Path) -> None:
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    try:
        import vllm

        vllm_version = getattr(vllm, "__version__", None)
    except Exception:
        vllm_version = None
    repo_root = Path(__file__).resolve().parents[3]
    cfg = {
        "args": vars(args),
        "model_path": args.model_path,
        "model_config": get_vllm_model_config_summary(llm),
        "backend": "vllm",
        "vllm_version": vllm_version,
        "transformers_version": transformers_version,
        "torch_version": torch.__version__,
        "selected_layer_indices": layer_indices,
        "attention_intervention": "runtime qkv V-slice zeroing in vLLM Qwen3Attention",
        "git": git_info(repo_root),
        "cuda": _torch_cuda_info(),
    }
    atomic_write_json(output_dir / "run_config.json", cfg)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    duration = None if args.duration == "all" else args.duration
    samples = load_videomme_samples(args.data_path, duration=duration)
    samples = _filter_samples(samples, args)
    if args.resume:
        done = completed_question_ids(results_path)
        samples = [s for s in samples if str(s.get("question_id")) not in done]
        LOGGER.info("resume enabled: skipping %d completed samples", len(done))
    LOGGER.info("samples to process: %d", len(samples))

    if args.backend == "vllm":
        llm, processor = load_vllm_and_processor(args)
        summary = get_vllm_model_config_summary(llm)
        num_layers = int(summary.get("num_hidden_layers") or 0)
        if num_layers <= 0:
            raise RuntimeError(f"could not determine vLLM decoder layer count from {summary}")
        layer_indices = default_layer_indices(
            num_layers,
            start=args.layer_start,
            end=args.layer_end,
            stride=args.layer_stride,
        )
        save_run_config_vllm(args, llm, layer_indices, output_dir)

        for idx, sample in enumerate(samples):
            qid = sample.get("question_id")
            LOGGER.info("processing %d/%d question_id=%s backend=vllm", idx + 1, len(samples), qid)
            rec = process_sample_vllm(
                sample,
                llm=llm,
                processor=processor,
                args=args,
                layer_indices=layer_indices,
                output_dir=output_dir,
            )
            append_jsonl(results_path, rec)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        model, processor = load_hf_model_and_processor(args.model_path, device=args.device, dtype=args.dtype)
        num_layers = len(find_decoder_self_attention_modules(model))
        layer_indices = default_layer_indices(
            num_layers,
            start=args.layer_start,
            end=args.layer_end,
            stride=args.layer_stride,
        )
        save_run_config(args, model, layer_indices, output_dir)

        for idx, sample in enumerate(samples):
            qid = sample.get("question_id")
            LOGGER.info("processing %d/%d question_id=%s backend=hf", idx + 1, len(samples), qid)
            rec = process_sample(
                sample,
                model=model,
                processor=processor,
                args=args,
                layer_indices=layer_indices,
                output_dir=output_dir,
            )
            append_jsonl(results_path, rec)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    records = read_jsonl(results_path)
    write_summary_files(records, output_dir, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed)
    LOGGER.info("wrote results to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
