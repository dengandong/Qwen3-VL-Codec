from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from visual_flow_probe.io_utils import append_jsonl, build_prompt, completed_question_ids, git_info, load_videomme_samples  # type: ignore
else:
    from ..visual_flow_probe.io_utils import append_jsonl, build_prompt, completed_question_ids, git_info, load_videomme_samples


LOGGER = logging.getLogger("compression_index_dump")


def _parse_sample_ids(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def _filter_samples(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    ids = _parse_sample_ids(args.sample_ids)
    if ids is not None:
        samples = [s for s in samples if str(s.get("question_id")) in ids]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.num_shards > 1:
        samples = [s for i, s in enumerate(samples) if i % args.num_shards == args.shard_index]
    if args.start_index:
        samples = samples[int(args.start_index) :]
    if args.max_samples is not None:
        samples = samples[: int(args.max_samples)]
    return samples


def _safe_drop_name(drop_ratio: float) -> str:
    return f"drop_{float(drop_ratio):.4f}".replace(".", "p")


def _apply_method_patch(args: argparse.Namespace, retain_ratio: float) -> None:
    method = args.method
    if method == "vcast":
        from vllm_qwen3_vl_vcast import apply_patch

        apply_patch(mode="post_vit", retain_ratio=retain_ratio, min_k=args.vcast_min_k)
        return
    if method == "ttf":
        from vllm_qwen3_vl_ttf import apply_patch

        apply_patch(
            mode="post_vit",
            budget_mode="retain_ratio",
            retain_ratio=retain_ratio,
            threshold=args.ttf_threshold,
            window_radius=args.ttf_window_radius,
            anchor=args.ttf_anchor,
            order=args.ttf_order,
        )
        return
    if method == "echoprune":
        from vllm_qwen3_vl_echoprune import apply_patch

        apply_patch(
            mode="post_vit",
            retain_ratio=retain_ratio,
            temperature=args.echoprune_temperature,
            match_scope=args.echoprune_match_scope,
            window_size=args.echoprune_window_size,
            first_frame_policy=args.echoprune_first_frame_policy,
            query_source=args.echoprune_query_source,
            match_chunk_size=args.echoprune_match_chunk_size,
        )
        return
    raise ValueError(f"unsupported method: {method}")


def _make_sampling_params(max_tokens: int):
    from vllm import SamplingParams

    return SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(max_tokens), stop_token_ids=[])


def _load_llm_and_processor(args: argparse.Namespace):
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from transformers import AutoProcessor
    from vllm import LLM

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"video": 1},
        seed=args.seed,
        disable_log_stats=True,
    )
    return llm, processor


def _reset_vllm_mm_cache(llm: Any) -> bool:
    """Best-effort reset for vLLM multimodal caches between per-question dumps."""

    seen: set[int] = set()
    stack: list[Any] = [llm]
    for attr in ("llm_engine", "engine", "_engine"):
        obj = getattr(llm, attr, None)
        if obj is not None:
            stack.append(obj)
    called = False
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        reset = getattr(obj, "reset_mm_cache", None)
        if callable(reset):
            reset()
            called = True
        for attr in ("engine_core", "_engine_core", "processor", "input_preprocessor"):
            child = getattr(obj, attr, None)
            if child is not None:
                stack.append(child)
    return called


def _prepare_vllm_prompt(processor: Any, messages: list[dict[str, Any]], args: argparse.Namespace, annotation: dict[str, Any]) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=getattr(processor.image_processor, "patch_size", None),
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    video_kwargs = dict(video_kwargs or {})
    video_kwargs["do_resize"] = False
    if args.method == "echoprune":
        from vllm_qwen3_vl_echoprune import build_echoprune_query_text

        query = build_echoprune_query_text(
            annotation=annotation,
            messages=messages,
            query_source=args.echoprune_query_source,
        )
        video_kwargs["echoprune_query_texts"] = [query] * len(video_inputs)
    mm_data: dict[str, Any] = {"video": video_inputs}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump Qwen3-VL visual token keep/drop indices for compression methods")
    p.add_argument("--method", choices=["vcast", "ttf", "echoprune"], required=True)
    p.add_argument("--drop-ratio", type=float, required=True, help="Fraction of dense visual tokens intended to drop")
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--video-dir", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--duration", choices=["short", "medium", "long"], default="short")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--sample-ids", default=None)
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

    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--max-trigger-tokens", type=int, default=1)

    p.add_argument("--vcast-min-k", type=int, default=1)
    p.add_argument("--ttf-threshold", type=float, default=0.70)
    p.add_argument("--ttf-window-radius", type=int, default=1)
    p.add_argument("--ttf-anchor", choices=["auto", "first", "last"], default="auto")
    p.add_argument("--ttf-order", choices=["paper", "temporal"], default="paper")
    p.add_argument("--echoprune-temperature", type=float, default=0.50)
    p.add_argument("--echoprune-match-scope", choices=["full", "local"], default="full")
    p.add_argument("--echoprune-window-size", type=int, default=3)
    p.add_argument("--echoprune-first-frame-policy", choices=["paper", "global"], default="paper")
    p.add_argument("--echoprune-query-source", choices=["question_options", "user_text", "all_text"], default="question_options")
    p.add_argument("--echoprune-match-chunk-size", type=int, default=256)
    p.add_argument("--log-level", default="INFO")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    if not (0.0 <= float(args.drop_ratio) < 1.0):
        raise ValueError("--drop-ratio must be in [0,1)")
    retain_ratio = 1.0 - float(args.drop_ratio)
    if retain_ratio <= 0.0:
        raise ValueError("retain ratio would be <= 0")

    output_dir = Path(args.output_dir)
    dump_root = output_dir / "dumps"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["QWEN3VL_INDEX_DUMP_DIR"] = str(dump_root)
    os.environ.setdefault("QWEN3VL_INDEX_DUMP_STRICT", "1")
    os.environ.setdefault("QWEN3VL_VCAST_QUIET", "1")
    os.environ.setdefault("QWEN3VL_TTF_QUIET", "1")
    os.environ.setdefault("QWEN3VL_ECHOPRUNE_QUIET", "1")

    _apply_method_patch(args, retain_ratio)
    llm, processor = _load_llm_and_processor(args)

    if __package__ is None or __package__ == "":
        from vllm_qwen3_vl_index_dump import clear_index_dump_context, set_index_dump_context  # type: ignore
    else:
        from ..vllm_qwen3_vl_index_dump import clear_index_dump_context, set_index_dump_context

    samples = load_videomme_samples(args.data_path, duration=args.duration)
    samples = _filter_samples(samples, args)
    results_path = output_dir / args.method / _safe_drop_name(float(args.drop_ratio)) / "dump_results.jsonl"
    done = completed_question_ids(results_path) if args.resume else set()
    params = _make_sampling_params(args.max_trigger_tokens)

    config = {
        "method": args.method,
        "drop_ratio": float(args.drop_ratio),
        "retain_ratio": retain_ratio,
        "duration": args.duration,
        "max_frames": args.max_frames,
        "data_path": args.data_path,
        "video_dir": args.video_dir,
        "model_path": args.model_path,
        "git": git_info(Path(__file__).resolve().parents[3]),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    LOGGER.info("dumping %s samples method=%s drop=%.2f retain=%.2f", len(samples), args.method, args.drop_ratio, retain_ratio)
    logged_reset = False
    for idx, sample in enumerate(samples):
        qid = str(sample.get("question_id"))
        if qid in done:
            continue
        record = {"question_id": qid, "method": args.method, "drop_ratio": float(args.drop_ratio), "ok": False}
        try:
            messages, annotation = build_prompt(sample, args)
            prompt_input = _prepare_vllm_prompt(processor, messages, args, annotation)
            reset_called = _reset_vllm_mm_cache(llm)
            if not logged_reset:
                LOGGER.info("vLLM multimodal cache reset available=%s", reset_called)
                logged_reset = True
            llm.apply_model(
                lambda model, qid=qid: set_index_dump_context(
                    model,
                    qid,
                    method=args.method,
                    drop_ratio=float(args.drop_ratio),
                    duration=args.duration,
                    extra={"sample_index": idx, "video_id": sample.get("videoID"), "retain_ratio": retain_ratio},
                )
            )
            llm.generate([prompt_input], sampling_params=params)
            llm.apply_model(clear_index_dump_context)
            expected = dump_root / args.method / _safe_drop_name(float(args.drop_ratio)) / f"{qid}.json"
            record.update({"ok": expected.exists(), "dump_path": str(expected)})
            if not expected.exists():
                record["error"] = "dump file was not produced"
        except Exception as exc:
            try:
                llm.apply_model(clear_index_dump_context)
            except Exception:
                pass
            record.update({"ok": False, "error": str(exc), "traceback": traceback.format_exc(limit=8)})
            LOGGER.exception("failed qid=%s", qid)
        append_jsonl(results_path, record)
        if (idx + 1) % 25 == 0:
            LOGGER.info("processed %d/%d", idx + 1, len(samples))
    LOGGER.info("done: %s", output_dir)


def _safe_drop_name(drop_ratio: float) -> str:
    return f"drop_{float(drop_ratio):.4f}".replace(".", "p")


if __name__ == "__main__":
    main()
