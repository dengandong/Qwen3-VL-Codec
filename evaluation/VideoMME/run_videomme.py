import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
import torch
import warnings
import string
import traceback

# Pick qwen-vl-utils' video backend before importing qwen_vl_utils.  If this is
# left unset, the current environment prefers torchcodec, whose installed binary
# is incompatible with this PyTorch/FFmpeg stack and falls back to slow
# torchvision after printing a large traceback.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

# vLLM imports
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

# Local imports from refactored files
from dataset_utils import load_videomme_dataset, build_videomme_prompt
from eval_utils import build_judge, eval_single_sample

# Set vLLM multiprocessing method unless the launcher already chose one.
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

def configure_video_reader_backend():
    """Optionally patch qwen-vl-utils decord backend with OpenCV.

    This fallback is only for environments where the native decord/FFmpeg build
    is broken. The default path uses real decord so experiments benefit from
    FFmpeg-backed random access when available.
    """
    if os.environ.get("QWEN3VL_PATCH_DECORD_WITH_OPENCV", "0") != "1":
        return

    try:
        from qwen_vl_utils import vision_process as vp
        import cv2
    except Exception as exc:
        print(f"[VideoReader] OpenCV decord patch unavailable: {exc}")
        return

    if getattr(vp, "_qwen3vl_opencv_decord_patched", False):
        return

    def _read_video_opencv(ele):
        video_path = ele["video"]
        if isinstance(video_path, str) and video_path.startswith("file://"):
            video_path = video_path[7:]

        st = time.time()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = float(cap.get(cv2.CAP_PROP_FPS))
        start_frame, end_frame, total_frames = vp.calculate_video_frame_range(
            ele,
            total_frames,
            video_fps,
        )
        nframes = vp.smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()

        frames = []
        for frame_idx in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                raise RuntimeError(f"OpenCV failed to read frame {frame_idx} from {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame).permute(2, 0, 1))

        cap.release()
        video = torch.stack(frames, dim=0)
        sample_fps = nframes / max(total_frames, 1e-6) * video_fps
        vp.logger.info(
            f"opencv-decord:  video_path={video_path!r}, total_frames={total_frames}, "
            f"video_fps={video_fps}, time={time.time() - st:.3f}s"
        )
        video_metadata = dict(
            fps=video_fps,
            frames_indices=idx,
            total_num_frames=total_frames,
            video_backend="opencv-decord",
        )
        return video, video_metadata, sample_fps

    vp.VIDEO_READER_BACKENDS["decord"] = _read_video_opencv
    vp._qwen3vl_opencv_decord_patched = True
    print("[VideoReader] patched qwen-vl-utils decord backend with OpenCV random frame reader")

def prepare_inputs_for_vllm(
    messages,
    processor,
    codec_video_id=None,
    echoprune_query_text=None,
    mmtok_query_text=None,
    vflow_question_id=None,
    vflow_video_id=None,
    vflow_duration=None,
):
    """
    Prepare inputs for vLLM (following the examples in README.md).
    
    Args:
        messages: List of messages in standard conversation format
        processor: AutoProcessor instance
    
    Returns:
        dict: Input format required by vLLM
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # qwen_vl_utils 0.0.14+ required
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs
        video_kwargs = dict(video_kwargs)
        # qwen_vl_utils.process_vision_info already decodes and resizes videos.
        # Keep HF/vLLM's video processor from resizing the tensor a second time.
        video_kwargs["do_resize"] = False
        if codec_video_id is not None:
            video_kwargs["codec_video_ids"] = [str(codec_video_id)] * len(video_inputs)
        if echoprune_query_text is not None:
            video_kwargs["echoprune_query_texts"] = [str(echoprune_query_text)] * len(video_inputs)
        if mmtok_query_text is not None:
            video_kwargs["mmtok_query_texts"] = [str(mmtok_query_text)] * len(video_inputs)
        if vflow_question_id is not None:
            video_kwargs["vflow_question_ids"] = [str(vflow_question_id)] * len(video_inputs)
        if vflow_video_id is not None:
            video_kwargs["vflow_video_ids"] = [str(vflow_video_id)] * len(video_inputs)
        if vflow_duration is not None:
            video_kwargs["vflow_durations"] = [str(vflow_duration)] * len(video_inputs)
    
    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }

def run_inference(args):
    """Run inference on the VideoMME dataset using vLLM."""
    print("\n" + "="*80)
    print("🚀 VideoMME Inference with vLLM (High-Speed Mode)")
    print("="*80 + "\n")
    
    configure_video_reader_backend()
    configure_vflow(args)
    configure_kitoke(args)
    configure_flashvid(args)
    configure_mmtok(args)
    configure_echoprune(args)
    configure_ttf(args)
    configure_vcast(args)
    configure_codec_guidance(args)

    # Load dataset
    data = load_videomme_dataset(args.data_dir, duration=args.duration)
    print(f"✓ Loaded {len(data)} samples from VideoMME (duration={args.duration})")
    
    # Limit samples for testing if specified
    if args.max_samples is not None and args.max_samples > 0:
        data = data[:args.max_samples]
        print(f"⚠️  Testing mode: Processing only first {len(data)} samples")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Load system prompt if provided
    sys_prompt = None
    if args.sys_prompt and os.path.exists(args.sys_prompt):
        with open(args.sys_prompt, 'r') as f:
            sys_prompt = f.read().strip()
        print(f"✓ Loaded system prompt from {args.sys_prompt}")

    # Set up generation parameters (vLLM SamplingParams format)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        stop_token_ids=[],
    )
    
    print(f"\n⚙️  Generation parameters (vLLM SamplingParams):")
    print(f"   max_tokens={sampling_params.max_tokens}")
    print(f"   temperature={sampling_params.temperature}, top_p={sampling_params.top_p}, top_k={sampling_params.top_k}")
    print(f"   repetition_penalty={sampling_params.repetition_penalty}")
    print(f"   presence_penalty={sampling_params.presence_penalty}")
    
    print(f"\n⚙️  Video processing parameters:")
    print(f"   fps={args.fps}")
    print(f"   min_pixels={args.min_pixels}, max_pixels={args.max_pixels}")
    print(f"   min_frames={args.min_frames}, max_frames={args.max_frames}")
    print(f"   total_pixels={args.total_pixels}")
    print(f"   use_subtitle={args.use_subtitle}")
    
    if sampling_params.presence_penalty > 0:
        print(f"   ✅ Anti-repetition enabled (presence_penalty={sampling_params.presence_penalty})")
    
    if sampling_params.temperature <= 0.02 and sampling_params.top_k == 1:
        print(f"   ✅ Using FAST greedy-like decoding")
    else:
        print(f"   ⚠️  Using sampling decoding (slower but more diverse)")
    print()

    # Load processor for input preparation
    print(f"Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("✓ Processor loaded\n")
    
    # Initialize vLLM
    print(f"Initializing vLLM with model: {args.model_path}")
    print(f"   GPU count: {torch.cuda.device_count()}")
    print(f"   Tensor parallel size: {args.tensor_parallel_size}")
    
    llm_kwargs = dict(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"video": args.max_videos_per_prompt},
        seed=args.seed,
    )
    if (
        getattr(args, "ttf_mode", "none") != "none"
        or getattr(args, "echoprune_mode", "none") != "none"
        or getattr(args, "mmtok_mode", "none") != "none"
        or getattr(args, "flashvid_mode", "none") != "none"
        or getattr(args, "kitoke_mode", "none") != "none"
    ):
        # This only enables vLLM's multimodal-pruning lifecycle so our patched
        # Qwen3-VL model can recompute sparse M-RoPE after encoder output.
        # Retained counts are controlled by the method-specific args.
        llm_kwargs["video_pruning_rate"] = 1e-6

    llm = LLM(**llm_kwargs)
    print("✓ vLLM initialized successfully\n")
    
    # Chunked inference. Preparing all videos up front can consume too much CPU
    # memory for VideoMME, especially at max_frames=128.
    print("="*80)
    print("🚀 Running vLLM batch inference (chunked preparation)")
    print("="*80)
    start_time = time.time()
    chunk_size = max(1, int(args.prepare_batch_size))
    results = []

    with open(args.output_file, 'w') as f:
        for start in tqdm(range(0, len(data), chunk_size), desc="Inference chunks"):
            chunk = data[start:start + chunk_size]
            chunk_inputs = []
            chunk_annotations = []
            chunk_messages = []

            for data_item in chunk:
                messages, annotation = build_videomme_prompt(
                    data_item,
                    args.data_dir,
                    use_subtitle=args.use_subtitle,
                    fps=args.fps,
                    min_frames=args.min_frames,
                    max_frames=args.max_frames,
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                    total_pixels=args.total_pixels,
                    sys_prompt=sys_prompt,
                    video_dir=args.video_dir,
                )

                echoprune_query_text = None
                if getattr(args, "echoprune_mode", "none") != "none":
                    from vllm_qwen3_vl_echoprune import build_echoprune_query_text

                    echoprune_query_text = build_echoprune_query_text(
                        annotation,
                        messages,
                        query_source=args.echoprune_query_source,
                    )
                mmtok_query_text = None
                if getattr(args, "mmtok_mode", "none") != "none":
                    from vllm_qwen3_vl_mmtok import build_mmtok_query_text

                    mmtok_query_text = build_mmtok_query_text(
                        annotation,
                        messages,
                        query_source=args.mmtok_query_source,
                    )

                vllm_input = prepare_inputs_for_vllm(
                    messages,
                    processor,
                    codec_video_id=data_item.get("videoID") if args.codec_guided_mode != "none" else None,
                    echoprune_query_text=echoprune_query_text,
                    mmtok_query_text=mmtok_query_text,
                    vflow_question_id=annotation.get("question_id") if getattr(args, "vflow_mode", "none") != "none" else None,
                    vflow_video_id=data_item.get("videoID") if getattr(args, "vflow_mode", "none") != "none" else None,
                    vflow_duration=args.duration if getattr(args, "vflow_mode", "none") != "none" else None,
                )

                chunk_inputs.append(vllm_input)
                chunk_annotations.append(annotation)
                chunk_messages.append(messages)

            outputs = llm.generate(chunk_inputs, sampling_params=sampling_params)

            for annotation, messages, output in zip(chunk_annotations, chunk_messages, outputs):
                response = output.outputs[0].text
                response_final = str(response).split("</think>")[-1].strip()
                result = {
                    "question_id": annotation['question_id'],
                    "annotation": annotation,
                    "task": f"VideoMME_{args.duration}_{'w_subtitle' if args.use_subtitle else 'wo_subtitle'}",
                    "result": {"gen": response_final, "gen_raw": response},
                    "messages": messages
                }
                results.append(result)
                f.write(json.dumps(result) + '\n')
                f.flush()

            del chunk_inputs, chunk_annotations, chunk_messages, outputs

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n✓ Inference completed in {total_time:.2f} seconds")
    print(f"  Average: {total_time/len(data):.2f} seconds/sample")
    print(f"  Throughput: {len(data)/total_time:.2f} samples/second\n")
    
    print(f"\n✓ Results saved to {args.output_file}")
    print(f"✓ Total samples processed: {len(results)}")


def configure_vflow(args):
    mode = getattr(args, "vflow_mode", "none")
    if mode == "none":
        return
    conflicts = []
    if getattr(args, "codec_guided_mode", "none") != "none":
        conflicts.append("--codec-guided-mode")
    if getattr(args, "vcast_mode", "none") != "none":
        conflicts.append("--vcast-mode")
    if getattr(args, "ttf_mode", "none") != "none":
        conflicts.append("--ttf-mode")
    if getattr(args, "echoprune_mode", "none") != "none":
        conflicts.append("--echoprune-mode")
    if getattr(args, "mmtok_mode", "none") != "none":
        conflicts.append("--mmtok-mode")
    if getattr(args, "flashvid_mode", "none") != "none":
        conflicts.append("--flashvid-mode")
    if getattr(args, "kitoke_mode", "none") != "none":
        conflicts.append("--kitoke-mode")
    if conflicts:
        raise ValueError("--vflow-mode cannot be enabled together with " + ", ".join(conflicts))

    target_visual_tokens = getattr(args, "vflow_target_visual_tokens", None)
    if target_visual_tokens is not None and target_visual_tokens <= 0:
        target_visual_tokens = None
    responsibility_dir = getattr(args, "vflow_responsibility_dir", None)
    if not responsibility_dir:
        raise ValueError("--vflow-responsibility-dir is required when --vflow-mode is enabled")

    os.environ["QWEN3VL_VFLOW_MODE"] = mode
    os.environ["QWEN3VL_VFLOW_RETAIN_RATIO"] = str(args.vflow_retain_ratio)
    os.environ["QWEN3VL_VFLOW_TARGET_VISUAL_TOKENS"] = (
        "" if target_visual_tokens is None else str(int(target_visual_tokens))
    )
    os.environ["QWEN3VL_VFLOW_RESPONSIBILITY_DIR"] = str(responsibility_dir)
    os.environ["QWEN3VL_VFLOW_SIGNAL"] = args.vflow_signal
    os.environ["QWEN3VL_VFLOW_KEEP"] = args.vflow_keep
    os.environ["QWEN3VL_VFLOW_DEBUG_VERIFY"] = "1" if args.vflow_debug_verify else "0"
    os.environ["QWEN3VL_VFLOW_QUIET"] = "1" if args.vflow_quiet else "0"

    from vllm_qwen3_vl_vflow import apply_patch as apply_qwen3_vl_vflow_patch

    apply_qwen3_vl_vflow_patch(
        mode=mode,
        retain_ratio=args.vflow_retain_ratio,
        responsibility_dir=responsibility_dir,
        target_visual_tokens=target_visual_tokens,
        signal=args.vflow_signal,
        keep=args.vflow_keep,
        debug_verify=args.vflow_debug_verify,
        quiet=args.vflow_quiet,
    )

    print("\n⚙️  VFlow vLLM patch:")
    print(f"   mode={mode}")
    print(f"   retain_ratio={args.vflow_retain_ratio}")
    print(f"   target_visual_tokens={target_visual_tokens}")
    print(f"   responsibility_dir={responsibility_dir}")
    print(f"   signal={args.vflow_signal}")
    print(f"   keep={args.vflow_keep}")
    print(f"   debug_verify={args.vflow_debug_verify}")


def configure_codec_guidance(args):
    mode = getattr(args, "codec_guided_mode", "none")
    if mode == "none":
        return
    if getattr(args, "flashvid_mode", "none") != "none":
        raise ValueError("--codec-guided-mode and --flashvid-mode cannot be enabled together")
    if getattr(args, "echoprune_mode", "none") != "none":
        raise ValueError("--codec-guided-mode and --echoprune-mode cannot be enabled together")
    if getattr(args, "vcast_mode", "none") != "none":
        raise ValueError("--codec-guided-mode and --vcast-mode cannot be enabled together")
    if getattr(args, "ttf_mode", "none") != "none":
        raise ValueError("--codec-guided-mode and --ttf-mode cannot be enabled together")
    if getattr(args, "mmtok_mode", "none") != "none":
        raise ValueError("--codec-guided-mode and --mmtok-mode cannot be enabled together")
    if getattr(args, "kitoke_mode", "none") != "none":
        raise ValueError("--codec-guided-mode and --kitoke-mode cannot be enabled together")

    repo_root = Path(__file__).resolve().parents[2]
    profile_to_zip = {
        "f128_eq16": repo_root / "codec_f128_eq16.zip",
        "f64_eq8": repo_root / "codec_f64_eqf8.zip",
    }
    profile_to_max_frames = {
        "f128_eq16": 128,
        "f64_eq8": 64,
    }

    if args.codec_guide_zip:
        guide_zip = Path(args.codec_guide_zip)
    else:
        guide_zip = profile_to_zip.get(args.codec_guide_profile)
        if guide_zip is None:
            raise ValueError("--codec-guide-zip is required when --codec-guide-profile custom is used")

    if not guide_zip.exists():
        raise FileNotFoundError(f"Codec guidance zip not found: {guide_zip}")

    if not args.no_codec_profile_overrides:
        profiled_max_frames = profile_to_max_frames.get(args.codec_guide_profile)
        if profiled_max_frames is not None:
            args.max_frames = profiled_max_frames

    os.environ["QWEN3VL_CODEC_GUIDED_MODE"] = mode
    os.environ["QWEN3VL_CODEC_GUIDE_ZIP"] = str(guide_zip)
    os.environ["QWEN3VL_CODEC_SELECTION_STRATEGY"] = args.codec_selection_strategy
    os.environ["QWEN3VL_CODEC_RANDOM_SEED"] = str(args.codec_random_seed)

    from vllm_qwen3_vl_codec_guided import apply_patch as apply_qwen3_vl_codec_patch

    apply_qwen3_vl_codec_patch(
        mode=mode,
        guide_zip=str(guide_zip),
        selection_strategy=args.codec_selection_strategy,
        random_seed=args.codec_random_seed,
    )

    print("\n⚙️  Codec-guided vLLM patch:")
    print(f"   mode={mode}")
    print(f"   selection_strategy={args.codec_selection_strategy}")
    print(f"   random_seed={args.codec_random_seed}")
    print(f"   profile={args.codec_guide_profile}")
    print(f"   guide_zip={guide_zip}")
    print(f"   max_frames={args.max_frames}")


def configure_vcast(args):
    mode = getattr(args, "vcast_mode", "none")
    if mode == "none":
        return
    if getattr(args, "flashvid_mode", "none") != "none":
        raise ValueError("--vcast-mode and --flashvid-mode cannot be enabled together")
    if getattr(args, "echoprune_mode", "none") != "none":
        raise ValueError("--vcast-mode and --echoprune-mode cannot be enabled together")
    if getattr(args, "codec_guided_mode", "none") != "none":
        raise ValueError("--vcast-mode and --codec-guided-mode cannot be enabled together")
    if getattr(args, "ttf_mode", "none") != "none":
        raise ValueError("--vcast-mode and --ttf-mode cannot be enabled together")
    if getattr(args, "mmtok_mode", "none") != "none":
        raise ValueError("--vcast-mode and --mmtok-mode cannot be enabled together")
    if getattr(args, "kitoke_mode", "none") != "none":
        raise ValueError("--vcast-mode and --kitoke-mode cannot be enabled together")

    os.environ["QWEN3VL_VCAST_MODE"] = mode
    os.environ["QWEN3VL_VCAST_RETAIN_RATIO"] = str(args.vcast_retain_ratio)
    os.environ["QWEN3VL_VCAST_MIN_K"] = str(args.vcast_min_k)

    from vllm_qwen3_vl_vcast import apply_patch as apply_qwen3_vl_vcast_patch

    apply_qwen3_vl_vcast_patch(
        mode=mode,
        retain_ratio=args.vcast_retain_ratio,
        min_k=args.vcast_min_k,
    )

    print("\n⚙️  V-CAST vLLM patch:")
    print(f"   mode={mode}")
    print(f"   retain_ratio={args.vcast_retain_ratio}")
    print(f"   min_k={args.vcast_min_k}")


def configure_ttf(args):
    mode = getattr(args, "ttf_mode", "none")
    if mode == "none":
        return
    if getattr(args, "flashvid_mode", "none") != "none":
        raise ValueError("--ttf-mode and --flashvid-mode cannot be enabled together")
    if getattr(args, "echoprune_mode", "none") != "none":
        raise ValueError("--ttf-mode and --echoprune-mode cannot be enabled together")
    if getattr(args, "codec_guided_mode", "none") != "none":
        raise ValueError("--ttf-mode and --codec-guided-mode cannot be enabled together")
    if getattr(args, "vcast_mode", "none") != "none":
        raise ValueError("--ttf-mode and --vcast-mode cannot be enabled together")
    if getattr(args, "mmtok_mode", "none") != "none":
        raise ValueError("--ttf-mode and --mmtok-mode cannot be enabled together")
    if getattr(args, "kitoke_mode", "none") != "none":
        raise ValueError("--ttf-mode and --kitoke-mode cannot be enabled together")

    version = getattr(args, "ttf_version", "v1")
    if version == "v2":
        os.environ["QWEN3VL_TTF_V2_MODE"] = mode
        os.environ["QWEN3VL_TTF_V2_THRESHOLD"] = str(args.ttf_threshold)
        os.environ["QWEN3VL_TTF_V2_BUDGET_MODE"] = args.ttf_budget_mode
        os.environ["QWEN3VL_TTF_V2_RETAIN_RATIO"] = str(args.ttf_retain_ratio)
        os.environ["QWEN3VL_TTF_V2_WINDOW_RADIUS"] = str(args.ttf_window_radius)
        os.environ["QWEN3VL_TTF_V2_ANCHOR"] = args.ttf_anchor
        os.environ["QWEN3VL_TTF_V2_ORDER"] = args.ttf_order
        os.environ["QWEN3VL_TTF_V2_TEMPORAL_ANCHOR_RADIUS"] = str(
            args.ttf_v2_temporal_anchor_radius
        )
        os.environ["QWEN3VL_TTF_V2_DEBUG_VERIFY"] = "1" if args.ttf_debug_verify else "0"

        from vllm_qwen3_vl_ttf_v2 import apply_patch as apply_qwen3_vl_ttf_patch

        apply_qwen3_vl_ttf_patch(
            mode=mode,
            threshold=args.ttf_threshold,
            budget_mode=args.ttf_budget_mode,
            retain_ratio=args.ttf_retain_ratio,
            window_radius=args.ttf_window_radius,
            anchor=args.ttf_anchor,
            order=args.ttf_order,
            temporal_anchor_radius=args.ttf_v2_temporal_anchor_radius,
            debug_verify=args.ttf_debug_verify,
        )
    else:
        os.environ["QWEN3VL_TTF_MODE"] = mode
        os.environ["QWEN3VL_TTF_THRESHOLD"] = str(args.ttf_threshold)
        os.environ["QWEN3VL_TTF_BUDGET_MODE"] = args.ttf_budget_mode
        os.environ["QWEN3VL_TTF_RETAIN_RATIO"] = str(args.ttf_retain_ratio)
        os.environ["QWEN3VL_TTF_WINDOW_RADIUS"] = str(args.ttf_window_radius)
        os.environ["QWEN3VL_TTF_ANCHOR"] = args.ttf_anchor
        os.environ["QWEN3VL_TTF_ORDER"] = args.ttf_order
        os.environ["QWEN3VL_TTF_DEBUG_VERIFY"] = "1" if args.ttf_debug_verify else "0"

        from vllm_qwen3_vl_ttf import apply_patch as apply_qwen3_vl_ttf_patch

        apply_qwen3_vl_ttf_patch(
            mode=mode,
            threshold=args.ttf_threshold,
            budget_mode=args.ttf_budget_mode,
            retain_ratio=args.ttf_retain_ratio,
            window_radius=args.ttf_window_radius,
            anchor=args.ttf_anchor,
            order=args.ttf_order,
            debug_verify=args.ttf_debug_verify,
        )

    print("\n⚙️  TTF vLLM patch:")
    print(f"   version={version}")
    print(f"   mode={mode}")
    print(f"   budget_mode={args.ttf_budget_mode}")
    print(f"   retain_ratio={args.ttf_retain_ratio}")
    print(f"   threshold={args.ttf_threshold}")
    print(f"   window_radius={args.ttf_window_radius}")
    print(f"   anchor={args.ttf_anchor}")
    print(f"   order={args.ttf_order}")
    if version == "v2":
        print(f"   temporal_anchor_radius={args.ttf_v2_temporal_anchor_radius}")
    print(f"   debug_verify={args.ttf_debug_verify}")


def configure_echoprune(args):
    mode = getattr(args, "echoprune_mode", "none")
    if mode == "none":
        return
    conflicts = []
    if getattr(args, "flashvid_mode", "none") != "none":
        conflicts.append("--flashvid-mode")
    if getattr(args, "codec_guided_mode", "none") != "none":
        conflicts.append("--codec-guided-mode")
    if getattr(args, "vcast_mode", "none") != "none":
        conflicts.append("--vcast-mode")
    if getattr(args, "ttf_mode", "none") != "none":
        conflicts.append("--ttf-mode")
    if getattr(args, "mmtok_mode", "none") != "none":
        conflicts.append("--mmtok-mode")
    if getattr(args, "kitoke_mode", "none") != "none":
        conflicts.append("--kitoke-mode")
    if conflicts:
        raise ValueError(
            "--echoprune-mode cannot be enabled together with "
            + ", ".join(conflicts)
        )

    target_visual_tokens = getattr(args, "echoprune_target_visual_tokens", None)
    if target_visual_tokens is not None and target_visual_tokens <= 0:
        target_visual_tokens = None

    os.environ["QWEN3VL_ECHOPRUNE_MODE"] = mode
    os.environ["QWEN3VL_ECHOPRUNE_RETAIN_RATIO"] = str(args.echoprune_retain_ratio)
    os.environ["QWEN3VL_ECHOPRUNE_TARGET_VISUAL_TOKENS"] = (
        "" if target_visual_tokens is None else str(int(target_visual_tokens))
    )
    os.environ["QWEN3VL_ECHOPRUNE_TEMPERATURE"] = str(args.echoprune_temperature)
    os.environ["QWEN3VL_ECHOPRUNE_MATCH_SCOPE"] = args.echoprune_match_scope
    os.environ["QWEN3VL_ECHOPRUNE_WINDOW_SIZE"] = str(args.echoprune_window_size)
    os.environ["QWEN3VL_ECHOPRUNE_FIRST_FRAME_POLICY"] = args.echoprune_first_frame_policy
    os.environ["QWEN3VL_ECHOPRUNE_QUERY_SOURCE"] = args.echoprune_query_source
    os.environ["QWEN3VL_ECHOPRUNE_MATCH_CHUNK_SIZE"] = str(args.echoprune_match_chunk_size)
    os.environ["QWEN3VL_ECHOPRUNE_DEBUG_VERIFY"] = "1" if args.echoprune_debug_verify else "0"

    from vllm_qwen3_vl_echoprune import apply_patch as apply_qwen3_vl_echoprune_patch

    apply_qwen3_vl_echoprune_patch(
        mode=mode,
        retain_ratio=args.echoprune_retain_ratio,
        target_visual_tokens=target_visual_tokens,
        temperature=args.echoprune_temperature,
        match_scope=args.echoprune_match_scope,
        window_size=args.echoprune_window_size,
        first_frame_policy=args.echoprune_first_frame_policy,
        query_source=args.echoprune_query_source,
        match_chunk_size=args.echoprune_match_chunk_size,
        debug_verify=args.echoprune_debug_verify,
    )

    print("\n⚙️  EchoPrune vLLM patch:")
    print(f"   mode={mode}")
    print(f"   retain_ratio={args.echoprune_retain_ratio}")
    print(f"   target_visual_tokens={target_visual_tokens}")
    print(f"   temperature={args.echoprune_temperature}")
    print(f"   match_scope={args.echoprune_match_scope}")
    print(f"   window_size={args.echoprune_window_size}")
    print(f"   first_frame_policy={args.echoprune_first_frame_policy}")
    print(f"   query_source={args.echoprune_query_source}")
    print(f"   match_chunk_size={args.echoprune_match_chunk_size}")
    print(f"   debug_verify={args.echoprune_debug_verify}")


def _parse_float_tuple_csv(value):
    if value is None:
        return (0.05, 0.10, 0.15, 0.20)
    if isinstance(value, (tuple, list)):
        return tuple(float(x) for x in value)
    return tuple(float(x.strip()) for x in str(value).split(",") if x.strip())


def configure_mmtok(args):
    mode = getattr(args, "mmtok_mode", "none")
    if mode == "none":
        return
    conflicts = []
    if getattr(args, "flashvid_mode", "none") != "none":
        conflicts.append("--flashvid-mode")
    if getattr(args, "codec_guided_mode", "none") != "none":
        conflicts.append("--codec-guided-mode")
    if getattr(args, "vcast_mode", "none") != "none":
        conflicts.append("--vcast-mode")
    if getattr(args, "ttf_mode", "none") != "none":
        conflicts.append("--ttf-mode")
    if getattr(args, "echoprune_mode", "none") != "none":
        conflicts.append("--echoprune-mode")
    if getattr(args, "kitoke_mode", "none") != "none":
        conflicts.append("--kitoke-mode")
    if conflicts:
        raise ValueError("--mmtok-mode cannot be enabled together with " + ", ".join(conflicts))

    target_visual_tokens = getattr(args, "mmtok_target_visual_tokens", None)
    if target_visual_tokens is not None and target_visual_tokens <= 0:
        target_visual_tokens = None
    adaptive_candidates = _parse_float_tuple_csv(args.mmtok_adaptive_vv_candidates)

    os.environ["QWEN3VL_MMTOK_MODE"] = mode
    os.environ["QWEN3VL_MMTOK_PROFILE"] = args.mmtok_profile
    os.environ["QWEN3VL_MMTOK_RETAIN_RATIO"] = str(args.mmtok_retain_ratio)
    os.environ["QWEN3VL_MMTOK_TARGET_VISUAL_TOKENS"] = (
        "" if target_visual_tokens is None else str(int(target_visual_tokens))
    )
    os.environ["QWEN3VL_MMTOK_BUDGET_ROUNDING"] = args.mmtok_budget_rounding
    os.environ["QWEN3VL_MMTOK_ALPHA"] = str(args.mmtok_alpha)
    os.environ["QWEN3VL_MMTOK_TV_TEMPERATURE"] = str(args.mmtok_tv_temperature)
    os.environ["QWEN3VL_MMTOK_VV_TEMPERATURE"] = str(args.mmtok_vv_temperature)
    os.environ["QWEN3VL_MMTOK_TEMPERATURE_MODE"] = args.mmtok_temperature_mode
    os.environ["QWEN3VL_MMTOK_ADAPTIVE_VV_CANDIDATES"] = ",".join(str(x) for x in adaptive_candidates)
    os.environ["QWEN3VL_MMTOK_VV_TARGET_MODE"] = args.mmtok_vv_target_mode
    os.environ["QWEN3VL_MMTOK_VV_TARGET_TOKENS"] = str(args.mmtok_vv_target_tokens)
    os.environ["QWEN3VL_MMTOK_GREEDY_MODE"] = args.mmtok_greedy_mode
    os.environ["QWEN3VL_MMTOK_STOCHASTIC_EPSILON"] = str(args.mmtok_stochastic_epsilon)
    os.environ["QWEN3VL_MMTOK_SELECTION_SEED"] = str(args.mmtok_selection_seed)
    os.environ["QWEN3VL_MMTOK_QUERY_SOURCE"] = args.mmtok_query_source
    os.environ["QWEN3VL_MMTOK_CANDIDATE_CHUNK_SIZE"] = str(args.mmtok_candidate_chunk_size)
    os.environ["QWEN3VL_MMTOK_TARGET_CHUNK_SIZE"] = str(args.mmtok_target_chunk_size)
    os.environ["QWEN3VL_MMTOK_EXACT_MAX_TOKENS"] = str(args.mmtok_exact_max_tokens)
    os.environ["QWEN3VL_MMTOK_DEBUG_VERIFY"] = "1" if args.mmtok_debug_verify else "0"

    from vllm_qwen3_vl_mmtok import apply_patch as apply_qwen3_vl_mmtok_patch

    apply_qwen3_vl_mmtok_patch(
        mode=mode,
        profile=args.mmtok_profile,
        retain_ratio=args.mmtok_retain_ratio,
        target_visual_tokens=target_visual_tokens,
        budget_rounding=args.mmtok_budget_rounding,
        alpha=args.mmtok_alpha,
        tv_temperature=args.mmtok_tv_temperature,
        vv_temperature=args.mmtok_vv_temperature,
        temperature_mode=args.mmtok_temperature_mode,
        adaptive_vv_candidates=adaptive_candidates,
        vv_target_mode=args.mmtok_vv_target_mode,
        vv_target_tokens=args.mmtok_vv_target_tokens,
        greedy_mode=args.mmtok_greedy_mode,
        stochastic_epsilon=args.mmtok_stochastic_epsilon,
        selection_seed=args.mmtok_selection_seed,
        query_source=args.mmtok_query_source,
        candidate_chunk_size=args.mmtok_candidate_chunk_size,
        target_chunk_size=args.mmtok_target_chunk_size,
        exact_max_tokens=args.mmtok_exact_max_tokens,
        debug_verify=args.mmtok_debug_verify,
    )

    print("\n⚙️  MMTok vLLM patch:")
    print(f"   mode={mode}")
    print(f"   profile={args.mmtok_profile}")
    print(f"   retain_ratio={args.mmtok_retain_ratio}")
    print(f"   target_visual_tokens={target_visual_tokens}")
    print(f"   alpha={args.mmtok_alpha}")
    print(f"   tv_temperature={args.mmtok_tv_temperature}")
    print(f"   vv_temperature={args.mmtok_vv_temperature}")
    print(f"   temperature_mode={args.mmtok_temperature_mode}")
    print(f"   adaptive_vv_candidates={adaptive_candidates}")
    print(f"   vv_target_mode={args.mmtok_vv_target_mode}")
    print(f"   vv_target_tokens={args.mmtok_vv_target_tokens}")
    print(f"   greedy_mode={args.mmtok_greedy_mode}")
    print(f"   stochastic_epsilon={args.mmtok_stochastic_epsilon}")
    print(f"   selection_seed={args.mmtok_selection_seed}")
    print(f"   query_source={args.mmtok_query_source}")
    print(f"   debug_verify={args.mmtok_debug_verify}")


def configure_kitoke(args):
    mode = getattr(args, "kitoke_mode", "none")
    if mode == "none":
        return
    conflicts = []
    if getattr(args, "codec_guided_mode", "none") != "none":
        conflicts.append("--codec-guided-mode")
    if getattr(args, "vcast_mode", "none") != "none":
        conflicts.append("--vcast-mode")
    if getattr(args, "ttf_mode", "none") != "none":
        conflicts.append("--ttf-mode")
    if getattr(args, "echoprune_mode", "none") != "none":
        conflicts.append("--echoprune-mode")
    if getattr(args, "mmtok_mode", "none") != "none":
        conflicts.append("--mmtok-mode")
    if getattr(args, "flashvid_mode", "none") != "none":
        conflicts.append("--flashvid-mode")
    if conflicts:
        raise ValueError("--kitoke-mode cannot be enabled together with " + ", ".join(conflicts))

    target_visual_tokens = getattr(args, "kitoke_target_visual_tokens", None)
    if target_visual_tokens is not None and target_visual_tokens <= 0:
        target_visual_tokens = None

    os.environ["QWEN3VL_KITOKE_MODE"] = mode
    os.environ["QWEN3VL_KITOKE_RETAIN_RATIO"] = str(args.kitoke_retain_ratio)
    os.environ["QWEN3VL_KITOKE_TARGET_VISUAL_TOKENS"] = (
        "" if target_visual_tokens is None else str(int(target_visual_tokens))
    )
    os.environ["QWEN3VL_KITOKE_KERNEL_ALPHA"] = str(args.kitoke_kernel_alpha)
    os.environ["QWEN3VL_KITOKE_SELECTION_METHOD"] = args.kitoke_selection_method
    os.environ["QWEN3VL_KITOKE_SELECTION_SEED"] = str(args.kitoke_selection_seed)
    os.environ["QWEN3VL_KITOKE_SEED_POLICY"] = args.kitoke_seed_policy
    os.environ["QWEN3VL_KITOKE_PIVOTAL_PAIRING"] = args.kitoke_pivotal_pairing
    os.environ["QWEN3VL_KITOKE_DIFF_THRESHOLD"] = str(args.kitoke_diff_threshold)
    os.environ["QWEN3VL_KITOKE_DELTA_THRESHOLD"] = str(args.kitoke_delta_threshold)
    os.environ["QWEN3VL_KITOKE_RELATIVE_DELTA_THRESHOLD"] = str(args.kitoke_relative_delta_threshold)
    os.environ["QWEN3VL_KITOKE_EDGE_POLICY"] = args.kitoke_edge_policy
    os.environ["QWEN3VL_KITOKE_EMPTY_INTERVAL_POLICY"] = args.kitoke_empty_interval_policy
    os.environ["QWEN3VL_KITOKE_MERGE_MODE"] = args.kitoke_merge_mode
    os.environ["QWEN3VL_KITOKE_DEEPSTACK_MODE"] = args.kitoke_deepstack_mode
    os.environ["QWEN3VL_KITOKE_KERNEL_ROW_CHUNK_SIZE"] = str(args.kitoke_kernel_row_chunk_size)
    os.environ["QWEN3VL_KITOKE_KERNEL_COL_CHUNK_SIZE"] = str(args.kitoke_kernel_col_chunk_size)
    os.environ["QWEN3VL_KITOKE_FRAME_MATCH_CHUNK_SIZE"] = str(args.kitoke_frame_match_chunk_size)
    os.environ["QWEN3VL_KITOKE_INTERVAL_MATCH_CHUNK_SIZE"] = str(args.kitoke_interval_match_chunk_size)
    os.environ["QWEN3VL_KITOKE_DEBUG_VERIFY"] = "1" if args.kitoke_debug_verify else "0"

    from vllm_qwen3_vl_kitoke import apply_patch as apply_qwen3_vl_kitoke_patch

    apply_qwen3_vl_kitoke_patch(
        mode=mode,
        retain_ratio=args.kitoke_retain_ratio,
        target_visual_tokens=target_visual_tokens,
        kernel_alpha=args.kitoke_kernel_alpha,
        selection_method=args.kitoke_selection_method,
        selection_seed=args.kitoke_selection_seed,
        seed_policy=args.kitoke_seed_policy,
        pivotal_pairing=args.kitoke_pivotal_pairing,
        diff_threshold=args.kitoke_diff_threshold,
        delta_threshold=args.kitoke_delta_threshold,
        relative_delta_threshold=args.kitoke_relative_delta_threshold,
        edge_policy=args.kitoke_edge_policy,
        empty_interval_policy=args.kitoke_empty_interval_policy,
        merge_mode=args.kitoke_merge_mode,
        deepstack_mode=args.kitoke_deepstack_mode,
        kernel_row_chunk_size=args.kitoke_kernel_row_chunk_size,
        kernel_col_chunk_size=args.kitoke_kernel_col_chunk_size,
        frame_match_chunk_size=args.kitoke_frame_match_chunk_size,
        interval_match_chunk_size=args.kitoke_interval_match_chunk_size,
        debug_verify=args.kitoke_debug_verify,
    )

    print("\n⚙️  KiToke vLLM patch:")
    print(f"   mode={mode}")
    print(f"   retain_ratio={args.kitoke_retain_ratio}")
    print(f"   target_visual_tokens={target_visual_tokens}")
    print(f"   kernel_alpha={args.kitoke_kernel_alpha}")
    print(f"   selection_method={args.kitoke_selection_method}")
    print(f"   selection_seed={args.kitoke_selection_seed}")
    print(f"   seed_policy={args.kitoke_seed_policy}")
    print(f"   pivotal_pairing={args.kitoke_pivotal_pairing}")
    print(f"   thresholds=({args.kitoke_diff_threshold}, {args.kitoke_delta_threshold}, {args.kitoke_relative_delta_threshold})")
    print(f"   edge_policy={args.kitoke_edge_policy}")
    print(f"   empty_interval_policy={args.kitoke_empty_interval_policy}")
    print(f"   merge_mode={args.kitoke_merge_mode}")
    print(f"   deepstack_mode={args.kitoke_deepstack_mode}")
    print(f"   debug_verify={args.kitoke_debug_verify}")


def configure_flashvid(args):
    mode = getattr(args, "flashvid_mode", "none")
    if mode == "none":
        return
    conflicts = []
    if getattr(args, "codec_guided_mode", "none") != "none":
        conflicts.append("--codec-guided-mode")
    if getattr(args, "vcast_mode", "none") != "none":
        conflicts.append("--vcast-mode")
    if getattr(args, "ttf_mode", "none") != "none":
        conflicts.append("--ttf-mode")
    if getattr(args, "echoprune_mode", "none") != "none":
        conflicts.append("--echoprune-mode")
    if getattr(args, "mmtok_mode", "none") != "none":
        conflicts.append("--mmtok-mode")
    if getattr(args, "kitoke_mode", "none") != "none":
        conflicts.append("--kitoke-mode")
    if conflicts:
        raise ValueError("--flashvid-mode cannot be enabled together with " + ", ".join(conflicts))

    llm_retention_ratio = getattr(args, "flashvid_llm_retention_ratio", None)
    if llm_retention_ratio is not None and llm_retention_ratio <= 0:
        llm_retention_ratio = None

    os.environ["QWEN3VL_FLASHVID_MODE"] = mode
    os.environ["QWEN3VL_FLASHVID_PROFILE"] = args.flashvid_profile
    os.environ["QWEN3VL_FLASHVID_BUDGET_MODE"] = args.flashvid_budget_mode
    os.environ["QWEN3VL_FLASHVID_RETENTION_RATIO"] = str(args.flashvid_retention_ratio)
    os.environ["QWEN3VL_FLASHVID_EXPANSION"] = str(args.flashvid_expansion)
    os.environ["QWEN3VL_FLASHVID_ALPHA"] = str(args.flashvid_alpha)
    os.environ["QWEN3VL_FLASHVID_TOKEN_SELECTION_METHOD"] = args.flashvid_token_selection_method or ""
    os.environ["QWEN3VL_FLASHVID_TEMPORAL_THRESHOLD"] = str(args.flashvid_temporal_threshold)
    os.environ["QWEN3VL_FLASHVID_DO_SEGMENT"] = "1" if args.flashvid_do_segment else "0"
    os.environ["QWEN3VL_FLASHVID_SEGMENT_THRESHOLD"] = str(args.flashvid_segment_threshold)
    os.environ["QWEN3VL_FLASHVID_MIN_SEGMENT_NUM"] = str(args.flashvid_min_segment_num)
    os.environ["QWEN3VL_FLASHVID_COMPLEMENTARY_SEGMENT"] = (
        "1" if args.flashvid_complementary_segment else "0"
    )
    os.environ["QWEN3VL_FLASHVID_BUDGET_CORRECTION"] = args.flashvid_budget_correction
    os.environ["QWEN3VL_FLASHVID_DPC_K_MAX"] = str(args.flashvid_dpc_k_max)
    os.environ["QWEN3VL_FLASHVID_DEEPSTACK_MODE"] = args.flashvid_deepstack_mode
    os.environ["QWEN3VL_FLASHVID_CLS_ATTN_CHUNK_SIZE"] = str(args.flashvid_cls_attn_chunk_size)
    os.environ["QWEN3VL_FLASHVID_TEMPORAL_MATCH_CHUNK_SIZE"] = str(args.flashvid_temporal_match_chunk_size)
    os.environ["QWEN3VL_FLASHVID_PRUNING_LAYER"] = str(args.flashvid_pruning_layer)
    os.environ["QWEN3VL_FLASHVID_LLM_RETENTION_RATIO"] = (
        "" if llm_retention_ratio is None else str(llm_retention_ratio)
    )
    os.environ["QWEN3VL_FLASHVID_DEBUG_VERIFY"] = "1" if args.flashvid_debug_verify else "0"

    from vllm_qwen3_vl_flashvid import apply_patch as apply_qwen3_vl_flashvid_patch

    apply_qwen3_vl_flashvid_patch(
        mode=mode,
        profile=args.flashvid_profile,
        budget_mode=args.flashvid_budget_mode,
        retention_ratio=args.flashvid_retention_ratio,
        expansion=args.flashvid_expansion,
        alpha=args.flashvid_alpha,
        token_selection_method=args.flashvid_token_selection_method,
        temporal_threshold=args.flashvid_temporal_threshold,
        do_segment=args.flashvid_do_segment,
        segment_threshold=args.flashvid_segment_threshold,
        min_segment_num=args.flashvid_min_segment_num,
        complementary_segment=args.flashvid_complementary_segment,
        budget_correction=args.flashvid_budget_correction,
        dpc_k_max=args.flashvid_dpc_k_max,
        deepstack_mode=args.flashvid_deepstack_mode,
        cls_attention_query_chunk_size=args.flashvid_cls_attn_chunk_size,
        temporal_match_chunk_size=args.flashvid_temporal_match_chunk_size,
        pruning_layer=args.flashvid_pruning_layer,
        llm_retention_ratio=llm_retention_ratio,
        debug_verify=args.flashvid_debug_verify,
    )

    print("\n⚙️  FlashVID vLLM patch:")
    print(f"   mode={mode}")
    print(f"   profile={args.flashvid_profile}")
    print(f"   budget_mode={args.flashvid_budget_mode}")
    print(f"   retention_ratio={args.flashvid_retention_ratio}")
    print(f"   expansion={args.flashvid_expansion}")
    print(f"   alpha={args.flashvid_alpha}")
    print(f"   token_selection_method={args.flashvid_token_selection_method}")
    print(f"   temporal_threshold={args.flashvid_temporal_threshold}")
    print(f"   do_segment={args.flashvid_do_segment}")
    print(f"   segment_threshold={args.flashvid_segment_threshold}")
    print(f"   min_segment_num={args.flashvid_min_segment_num}")
    print(f"   complementary_segment={args.flashvid_complementary_segment}")
    print(f"   budget_correction={args.flashvid_budget_correction}")
    print(f"   dpc_k_max={args.flashvid_dpc_k_max}")
    print(f"   deepstack_mode={args.flashvid_deepstack_mode}")
    print(f"   cls_attn_chunk_size={args.flashvid_cls_attn_chunk_size}")
    print(f"   temporal_match_chunk_size={args.flashvid_temporal_match_chunk_size}")
    print(f"   pruning_layer={args.flashvid_pruning_layer}")
    print(f"   llm_retention_ratio={llm_retention_ratio}")
    print(f"   debug_verify={args.flashvid_debug_verify}")


def run_evaluation(args):
    """Run evaluation on inference results."""
    # Load results
    results = []
    with open(args.input_file, 'r') as f:
        for line in f:
            job = json.loads(line)
            annotation = job["annotation"]
            annotation["prediction"] = job["result"]["gen"]
            annotation["index"] = job["question_id"]
            annotation["category"] = annotation.get("category", annotation.get("domain", "unknown"))
            annotation["domain"] = annotation.get("domain", annotation["category"])
            annotation["duration"] = annotation.get("duration", "unknown")
            annotation["sub_category"] = annotation.get("sub_category", "unknown")
            annotation["task_category"] = annotation.get(
                "task_category",
                annotation.get("task_type", annotation["sub_category"]),
            )
            annotation["task_type"] = annotation.get("task_type", annotation["task_category"])
            results.append(annotation)
            
    data = pd.DataFrame.from_records(results)
    data = data.sort_values(by='index')
    data['prediction'] = [str(x) for x in data['prediction']]
    
    # Build choices columns (A, B, C, D) from annotation
    for idx, row in data.iterrows():
        choices = row['choices']
        for k, v in choices.items():
            data.at[idx, k] = v
    
    # Build judge model
    model = build_judge(
        model=getattr(args, 'eval_model', 'gpt-3.5-turbo-0125'),
        api_type=getattr(args, 'api_type', 'dash')
    )
    
    # Prepare evaluation tasks
    eval_tasks = []
    for idx, item in data.iterrows():
        eval_tasks.append((model, item))
    
    # Run evaluation
    eval_results = []
    
    # Normal mode: process all samples with threading
    from concurrent.futures import ThreadPoolExecutor
    nproc = getattr(args, 'nproc', 4)
    with ThreadPoolExecutor(max_workers=nproc) as executor:
        for result in tqdm(executor.map(eval_single_sample, eval_tasks), 
                         total=len(eval_tasks), desc="Evaluating"):
            eval_results.append(result)
    
    # Calculate overall accuracy
    accuracy = sum(r['hit'] for r in eval_results) / len(eval_results)
    
    # Calculate accuracy by category
    results_by_category = {}
    for result in eval_results:
        category = result.get('domain', 'unknown')
        if category not in results_by_category:
            results_by_category[category] = []
        results_by_category[category].append(result)
    
    accuracy_by_category = {}
    for category, cat_results in results_by_category.items():
        cat_accuracy = sum(r['hit'] for r in cat_results) / len(cat_results)
        accuracy_by_category[category] = cat_accuracy
        print(f"Accuracy for {category}: {cat_accuracy:.4f} ({sum(r['hit'] for r in cat_results)}/{len(cat_results)})")
    
    # Calculate accuracy by sub_category
    results_by_subcategory = {}
    for result in eval_results:
        sub_category = result.get('sub_category', 'unknown')
        if sub_category not in results_by_subcategory:
            results_by_subcategory[sub_category] = []
        results_by_subcategory[sub_category].append(result)
    
    accuracy_by_subcategory = {}
    for sub_category, subcat_results in results_by_subcategory.items():
        subcat_accuracy = sum(r['hit'] for r in subcat_results) / len(subcat_results)
        accuracy_by_subcategory[sub_category] = subcat_accuracy

    # Calculate accuracy by task category. VideoMME uses task_category; keep a
    # task_type alias for downstream scripts that expect that name.
    results_by_task_category = {}
    for result in eval_results:
        task_category = result.get('task_category', result.get('task_type', result.get('sub_category', 'unknown')))
        if task_category not in results_by_task_category:
            results_by_task_category[task_category] = []
        results_by_task_category[task_category].append(result)

    accuracy_by_task_category = {}
    for task_category, task_results in results_by_task_category.items():
        task_accuracy = sum(r['hit'] for r in task_results) / len(task_results)
        accuracy_by_task_category[task_category] = task_accuracy

    # Calculate accuracy by duration.
    results_by_duration = {}
    for result in eval_results:
        duration = result.get('duration', 'unknown')
        if duration not in results_by_duration:
            results_by_duration[duration] = []
        results_by_duration[duration].append(result)

    accuracy_by_duration = {}
    for duration, duration_results in results_by_duration.items():
        duration_accuracy = sum(r['hit'] for r in duration_results) / len(duration_results)
        accuracy_by_duration[duration] = duration_accuracy
    
    # Save results
    output_df = pd.DataFrame(eval_results)
    output_df.to_csv(args.output_file, index=False)
    
    # Save accuracy
    with open(args.output_file.replace('.csv', '_acc.json'), 'w') as f:
        json.dump({
            "overall_accuracy": accuracy,
            "accuracy_by_category": accuracy_by_category,
            "accuracy_by_subcategory": accuracy_by_subcategory,
            "accuracy_by_task_category": accuracy_by_task_category,
            "accuracy_by_task_type": accuracy_by_task_category,
            "accuracy_by_duration": accuracy_by_duration
        }, f, indent=2)
    
    # Also save as TSV format (consistent with original implementation)
    tsv_file = args.output_file.replace('.csv', '.tsv')
    output_df.to_csv(tsv_file, sep='\t', index=False)
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"{'='*50}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"{'='*50}\n")

def main():
    parser = argparse.ArgumentParser(description="VideoMME Evaluation with vLLM")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference with vLLM")
    infer_parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    infer_parser.add_argument("--data-dir", type=str, required=True, help="VideoMME data directory")
    infer_parser.add_argument("--video-dir", type=str, default=None,
                            help="Directory containing VideoMME mp4 files. Defaults to <data-dir>/videos")
    infer_parser.add_argument("--duration", type=str, default="short", 
                            choices=["short", "medium", "long"],
                            help="Video duration type (short/medium/long)")
    infer_parser.add_argument("--use-subtitle", action="store_true", 
                            help="Use subtitles if available")
    infer_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    infer_parser.add_argument("--sys-prompt", type=str, default=None, 
                            help="Path to system prompt file")
    infer_parser.add_argument("--max-samples", type=int, default=None,
                            help="Maximum number of samples to process (for testing, default: None = all samples)")
    infer_parser.add_argument("--prepare-batch-size", type=int, default=16,
                            help="Number of videos to prepare and send to vLLM per generate call")
    
    # Video processing parameters
    infer_parser.add_argument("--fps", type=int, default=2, help="Frames per second (default: 2)")
    infer_parser.add_argument("--min-pixels", type=int, default=128*28*28, 
                            help="Minimum pixels per frame (default: 128*28*28)")
    infer_parser.add_argument("--max-pixels", type=int, default=512*28*28,
                            help="Maximum pixels per frame (default: 512*28*28)")
    infer_parser.add_argument("--min-frames", type=int, default=4, 
                            help="Minimum number of frames (default: 4)")
    infer_parser.add_argument("--max-frames", type=int, default=512,
                            help="Maximum number of frames (default: 512)")
    infer_parser.add_argument("--total-pixels", type=int, default=24576*28*28,
                            help="Total pixels across all frames (default: 24576*28*28)")
    infer_parser.add_argument("--vflow-mode", type=str, default="none",
                            choices=["none", "post_vit"],
                            help="Enable offline VFlow-responsibility-guided post-ViT pruning")
    infer_parser.add_argument("--vflow-retain-ratio", type=float, default=0.125,
                            help="VFlow visual token retain ratio (default: 0.125)")
    infer_parser.add_argument("--vflow-target-visual-tokens", type=int, default=None,
                            help="Optional absolute VFlow visual-token budget per video")
    infer_parser.add_argument("--vflow-responsibility-dir", type=str, default=None,
                            help="Directory containing VFlow responsibility .npz files or a responsibilities/ subdir")
    infer_parser.add_argument("--vflow-signal", type=str, default="responsibility",
                            choices=["responsibility", "direct_attention"],
                            help="VFlow score array to use from each .npz file")
    infer_parser.add_argument("--vflow-keep", type=str, default="high",
                            choices=["high", "top", "low", "bottom"],
                            help="Keep high- or low-score VFlow tokens (default: high)")
    infer_parser.add_argument("--vflow-debug-verify", action="store_true",
                            help="Enable strict VFlow invariant logging/checks")
    infer_parser.add_argument("--vflow-quiet", action="store_true",
                            help="Suppress per-video VFlow logs")
    infer_parser.add_argument("--codec-guided-mode", type=str, default="none",
                            choices=["none", "pre_vit", "post_vit"],
                            help="Enable codec-guided vLLM Qwen3-VL pruning")
    infer_parser.add_argument("--codec-guide-profile", type=str, default="f128_eq16",
                            choices=["f128_eq16", "f64_eq8", "custom"],
                            help="Codec guidance profile: f128_eq16=max_frames 128/equiv 16, f64_eq8=max_frames 64/equiv 8")
    infer_parser.add_argument("--codec-guide-zip", type=str, default=None,
                            help="Custom codec guidance zip. Required for --codec-guide-profile custom")
    infer_parser.add_argument("--no-codec-profile-overrides", action="store_true",
                            help="Do not override --max-frames from the selected codec guidance profile")
    infer_parser.add_argument("--codec-selection-strategy", type=str, default="codec",
                            choices=["codec", "random"],
                            help="Select codec-ranked groups or a deterministic random baseline with the same budget")
    infer_parser.add_argument("--codec-random-seed", type=int, default=3407,
                            help="Stable seed for --codec-selection-strategy random")
    infer_parser.add_argument("--vcast-mode", type=str, default="none",
                            choices=["none", "post_vit"],
                            help="Enable V-CAST online post-ViT token pruning for vLLM Qwen3-VL")
    infer_parser.add_argument("--vcast-retain-ratio", type=float, default=0.25,
                            help="V-CAST retain ratio over video tokens (default: 0.25)")
    infer_parser.add_argument("--vcast-min-k", type=int, default=1,
                            help="Minimum retained tokens per video frame for V-CAST (default: 1)")
    infer_parser.add_argument("--ttf-mode", type=str, default="none",
                            choices=["none", "post_vit"],
                            help="Enable TTF post-ViT token fusion for vLLM Qwen3-VL")
    infer_parser.add_argument("--ttf-version", type=str, default="v1",
                            choices=["v1", "v2"],
                            help="TTF implementation version. v2 uses dynamic local temporal anchors")
    infer_parser.add_argument("--ttf-budget-mode", type=str, default="retain_ratio",
                            choices=["threshold", "retain_ratio"],
                            help="TTF budget mode. retain_ratio is the vLLM-compatible fixed-length mode")
    infer_parser.add_argument("--ttf-retain-ratio", type=float, default=0.25,
                            help="TTF retained ratio when --ttf-budget-mode retain_ratio (default: 0.25)")
    infer_parser.add_argument("--ttf-threshold", type=float, default=0.70,
                            help="TTF cosine threshold for threshold mode (default: 0.70)")
    infer_parser.add_argument("--ttf-window-radius", type=int, default=1,
                            help="TTF local anchor search radius (default: 1)")
    infer_parser.add_argument("--ttf-anchor", type=str, default="auto",
                            choices=["auto", "first", "last"],
                            help="TTF anchor frame policy (default: auto)")
    infer_parser.add_argument("--ttf-order", type=str, default="paper",
                            choices=["paper", "temporal"],
                            help="TTF output order; temporal is compatibility mode (default: paper)")
    infer_parser.add_argument("--ttf-v2-temporal-anchor-radius", type=int, default=2,
                            help="TTF V2 dynamic anchor temporal radius; 2 means [t-2,t+2]")
    infer_parser.add_argument("--ttf-debug-verify", action="store_true",
                            help="Enable strict TTF invariant logging/checks")
    infer_parser.add_argument("--echoprune-mode", type=str, default="none",
                            choices=["none", "post_vit"],
                            help="Enable EchoPrune query-guided post-ViT pruning for vLLM Qwen3-VL")
    infer_parser.add_argument("--echoprune-retain-ratio", type=float, default=0.20,
                            help="EchoPrune visual token retain ratio (default: 0.20)")
    infer_parser.add_argument("--echoprune-target-visual-tokens", type=int, default=None,
                            help="Optional absolute EchoPrune visual-token budget per video")
    infer_parser.add_argument("--echoprune-temperature", type=float, default=0.50,
                            help="EchoPrune temporal echo softmax temperature (default: 0.50)")
    infer_parser.add_argument("--echoprune-match-scope", type=str, default="full",
                            choices=["full", "local"],
                            help="EchoPrune temporal echo candidate scope (default: full)")
    infer_parser.add_argument("--echoprune-window-size", type=int, default=3,
                            help="Odd local-window size for --echoprune-match-scope local (default: 3)")
    infer_parser.add_argument("--echoprune-first-frame-policy", type=str, default="paper",
                            choices=["paper", "global"],
                            help="EchoPrune first-frame budget policy (default: paper)")
    infer_parser.add_argument("--echoprune-query-source", type=str, default="question_options",
                            choices=["question_options", "user_text", "all_text"],
                            help="Text used for EchoPrune query embeddings (default: question_options)")
    infer_parser.add_argument("--echoprune-match-chunk-size", type=int, default=256,
                            help="Chunk size for EchoPrune matching/relevance matmuls (default: 256)")
    infer_parser.add_argument("--echoprune-debug-verify", action="store_true",
                            help="Enable strict EchoPrune invariant logging/checks")
    infer_parser.add_argument("--mmtok-mode", type=str, default="none",
                            choices=["none", "post_vit"],
                            help="Enable MMTok query-guided post-ViT pruning for vLLM Qwen3-VL")
    infer_parser.add_argument("--mmtok-profile", type=str, default="auto",
                            choices=["paper_exact", "video_scalable", "auto"],
                            help="MMTok profile: exact paper-style video extension, scalable video mode, or auto")
    infer_parser.add_argument("--mmtok-retain-ratio", type=float, default=0.20,
                            help="MMTok visual token retain ratio (default: 0.20)")
    infer_parser.add_argument("--mmtok-target-visual-tokens", type=int, default=None,
                            help="Optional absolute MMTok visual-token budget per video")
    infer_parser.add_argument("--mmtok-budget-rounding", type=str, default="floor",
                            choices=["floor", "round"],
                            help="How retain_ratio is converted to integer visual-token budget")
    infer_parser.add_argument("--mmtok-alpha", type=float, default=0.50,
                            help="MMTok visual coverage weight alpha (default: 0.50)")
    infer_parser.add_argument("--mmtok-tv-temperature", type=float, default=0.01,
                            help="MMTok text-vision softmax temperature (default: 0.01)")
    infer_parser.add_argument("--mmtok-vv-temperature", type=float, default=0.20,
                            help="MMTok vision-vision softmax temperature (default: 0.20)")
    infer_parser.add_argument("--mmtok-temperature-mode", type=str, default="fixed",
                            choices=["fixed", "adaptive_vv"],
                            help="MMTok temperature mode; adaptive_vv selects vv temperature per video")
    infer_parser.add_argument("--mmtok-adaptive-vv-candidates", type=str, default="0.05,0.10,0.15,0.20",
                            help="Comma-separated candidate vv temperatures for adaptive_vv")
    infer_parser.add_argument("--mmtok-vv-target-mode", type=str, default="stratified_3d",
                            choices=["full", "stratified_3d"],
                            help="MMTok visual target row mode (default: stratified_3d)")
    infer_parser.add_argument("--mmtok-vv-target-tokens", type=int, default=1024,
                            help="MMTok scalable visual target coreset size (default: 1024)")
    infer_parser.add_argument("--mmtok-greedy-mode", type=str, default="auto",
                            choices=["exact", "stochastic", "auto"],
                            help="MMTok greedy optimizer mode")
    infer_parser.add_argument("--mmtok-stochastic-epsilon", type=float, default=0.10,
                            help="MMTok stochastic greedy epsilon (default: 0.10)")
    infer_parser.add_argument("--mmtok-selection-seed", type=int, default=3407,
                            help="MMTok request-local stochastic selection seed")
    infer_parser.add_argument("--mmtok-query-source", type=str, default="question_options",
                            choices=["question", "question_options", "user_text", "all_text"],
                            help="Text used for MMTok query embeddings (default: question_options)")
    infer_parser.add_argument("--mmtok-candidate-chunk-size", type=int, default=512,
                            help="Candidate chunk size for MMTok global softmax/logZ")
    infer_parser.add_argument("--mmtok-target-chunk-size", type=int, default=256,
                            help="Target chunk size reserved for MMTok diagnostics/scalable path")
    infer_parser.add_argument("--mmtok-exact-max-tokens", type=int, default=1024,
                            help="Maximum dense tokens allowed for exact MMTok profile/greedy")
    infer_parser.add_argument("--mmtok-debug-verify", action="store_true",
                            help="Enable strict MMTok invariant and temperature diagnostics")
    infer_parser.add_argument("--flashvid-mode", type=str, default="none",
                            choices=["none", "post_vit", "hybrid"],
                            help="Enable FlashVID vision-side post-ViT merging for vLLM Qwen3-VL")
    infer_parser.add_argument("--flashvid-profile", type=str, default="official_qwen3",
                            choices=["official_qwen3", "paper_adts_v2", "custom"],
                            help="FlashVID profile: official Qwen3 ADTS v1, paper ADTS v2, or custom")
    infer_parser.add_argument("--flashvid-budget-mode", type=str, default="direct",
                            choices=["direct", "paper_hybrid"],
                            help="FlashVID budget interpretation; direct is default for post_vit")
    infer_parser.add_argument("--flashvid-retention-ratio", type=float, default=0.20,
                            help="FlashVID requested visual token retention ratio (default: 0.20)")
    infer_parser.add_argument("--flashvid-expansion", type=float, default=1.25,
                            help="FlashVID hybrid pre-LLM expansion factor (default: 1.25)")
    infer_parser.add_argument("--flashvid-alpha", type=float, default=0.70,
                            help="FlashVID ADTS budget fraction alpha (default: 0.70)")
    infer_parser.add_argument("--flashvid-token-selection-method", type=str, default=None,
                            choices=["attn", "div", "attn_div", "attn_div_v2"],
                            help="Override FlashVID token selection method")
    infer_parser.add_argument("--flashvid-temporal-threshold", type=float, default=0.80,
                            help="FlashVID TSTM strict temporal merge threshold (default: 0.80)")
    infer_parser.add_argument("--flashvid-do-segment", action=argparse.BooleanOptionalAction, default=True,
                            help="Enable/disable FlashVID DySeg dynamic segmentation")
    infer_parser.add_argument("--flashvid-segment-threshold", type=float, default=0.90,
                            help="FlashVID DySeg transition threshold (default: 0.90)")
    infer_parser.add_argument("--flashvid-min-segment-num", type=int, default=4,
                            help="FlashVID minimum segment count with complementary cuts (default: 4)")
    infer_parser.add_argument("--flashvid-complementary-segment", action=argparse.BooleanOptionalAction, default=True,
                            help="Enable/disable FlashVID complementary segmentation cuts")
    infer_parser.add_argument("--flashvid-budget-correction", type=str, default="official_ceil",
                            choices=["official_ceil", "exact_total"],
                            help="FlashVID DPC frame budget correction policy")
    infer_parser.add_argument("--flashvid-dpc-k-max", type=int, default=7,
                            help="FlashVID DPC-kNN maximum k (default: 7)")
    infer_parser.add_argument("--flashvid-deepstack-mode", type=str, default="official_gather",
                            choices=["official_gather", "hierarchical_mean"],
                            help="FlashVID deepstack handling; official default gathers representative rows")
    infer_parser.add_argument("--flashvid-cls-attn-chunk-size", type=int, default=128,
                            help="Query chunk size for FlashVID incoming vision attention")
    infer_parser.add_argument("--flashvid-temporal-match-chunk-size", type=int, default=256,
                            help="Current-token chunk size for FlashVID TSTM matching")
    infer_parser.add_argument("--flashvid-pruning-layer", type=int, default=28,
                            help="FlashVID hybrid pruning layer index (hybrid currently fail-fast)")
    infer_parser.add_argument("--flashvid-llm-retention-ratio", type=float, default=None,
                            help="Optional explicit inner-LLM retention ratio for hybrid")
    infer_parser.add_argument("--flashvid-debug-verify", action="store_true",
                            help="Enable strict FlashVID invariant logging/checks")
    infer_parser.add_argument("--kitoke-mode", type=str, default="none",
                            choices=["none", "post_vit"],
                            help="Enable KiToke post-ViT kernel/interval token merging for vLLM Qwen3-VL")
    infer_parser.add_argument("--kitoke-retain-ratio", type=float, default=0.10,
                            help="KiToke visual token retain ratio (default: 0.10)")
    infer_parser.add_argument("--kitoke-target-visual-tokens", type=int, default=None,
                            help="Optional absolute KiToke visual-token budget per video")
    infer_parser.add_argument("--kitoke-kernel-alpha", type=float, default=800.0,
                            help="KiToke Gaussian kernel denominator alpha (default: 800)")
    infer_parser.add_argument("--kitoke-selection-method", type=str, default="pivotal",
                            choices=["pivotal", "multinomial", "topk"],
                            help="KiToke fixed-size selection method")
    infer_parser.add_argument("--kitoke-selection-seed", type=int, default=3407,
                            help="KiToke engineering reproducibility seed")
    infer_parser.add_argument("--kitoke-seed-policy", type=str, default="stable_video",
                            choices=["stable_video", "fixed"],
                            help="KiToke seed policy (default: stable_video)")
    infer_parser.add_argument("--kitoke-pivotal-pairing", type=str, default="random_rounds",
                            choices=["random_rounds", "sequential"],
                            help="KiToke pivotal sampling pair schedule")
    infer_parser.add_argument("--kitoke-diff-threshold", type=float, default=110.0,
                            help="KiToke absolute transition boundary threshold (default: 110)")
    infer_parser.add_argument("--kitoke-delta-threshold", type=float, default=70.0,
                            help="KiToke local absolute deviation threshold (default: 70)")
    infer_parser.add_argument("--kitoke-relative-delta-threshold", type=float, default=0.40,
                            help="KiToke local relative deviation threshold (default: 0.4)")
    infer_parser.add_argument("--kitoke-edge-policy", type=str, default="absolute_only",
                            choices=["absolute_only", "one_sided"],
                            help="KiToke edge transition local-deviation policy")
    infer_parser.add_argument("--kitoke-empty-interval-policy", type=str, default="repair_swap",
                            choices=["paper_strict", "repair_swap", "coarsen_then_repair"],
                            help="KiToke policy for intervals with no sampled representative")
    infer_parser.add_argument("--kitoke-merge-mode", type=str, default="weighted",
                            choices=["weighted", "uniform", "none"],
                            help="KiToke merge mode; weighted is paper-compatible default")
    infer_parser.add_argument("--kitoke-deepstack-mode", type=str, default="same_weighted_merge",
                            choices=["same_weighted_merge", "representative_gather"],
                            help="KiToke Qwen3-VL deepstack handling")
    infer_parser.add_argument("--kitoke-kernel-row-chunk-size", type=int, default=256,
                            help="KiToke KDE row chunk size")
    infer_parser.add_argument("--kitoke-kernel-col-chunk-size", type=int, default=512,
                            help="KiToke KDE candidate chunk size")
    infer_parser.add_argument("--kitoke-frame-match-chunk-size", type=int, default=256,
                            help="KiToke adjacent-frame matching chunk size")
    infer_parser.add_argument("--kitoke-interval-match-chunk-size", type=int, default=256,
                            help="KiToke interval assignment chunk size")
    infer_parser.add_argument("--kitoke-debug-verify", action="store_true",
                            help="Enable strict KiToke invariant logging/checks")
    
    # vLLM specific parameters
    infer_parser.add_argument("--tensor-parallel-size", type=int, default=None, 
                            help="Tensor parallel size (default: number of GPUs)")
    infer_parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                            help="GPU memory utilization (0.0-1.0, default: 0.9)")
    infer_parser.add_argument("--max-model-len", type=int, default=128000,
                            help="Maximum model context length (default: 128000)")
    infer_parser.add_argument("--max-videos-per-prompt", type=int, default=1,
                            help="Maximum videos per prompt (default: 1)")
    infer_parser.add_argument("--seed", type=int, default=3407, help="Random seed (default: 3407)")
    
    # Generation parameters used for Qwen3-VL Instruct VideoMME runs.
    infer_parser.add_argument("--max-new-tokens", type=int, default=32768, 
                            help="Maximum number of tokens to generate (default: 32768)")
    infer_parser.add_argument("--temperature", type=float, default=0.7, 
                            help="Temperature for sampling (default: 0.7)")
    infer_parser.add_argument("--top-p", type=float, default=0.8, 
                            help="Top-p for sampling (default: 0.8)")
    infer_parser.add_argument("--top-k", type=int, default=20, 
                            help="Top-k for sampling (default: 20)")
    infer_parser.add_argument("--repetition-penalty", type=float, default=1.0,
                            help="Repetition penalty (default: 1.0)")
    infer_parser.add_argument("--presence-penalty", type=float, default=1.5,
                            help="Presence penalty (default: 1.5)")
    
    # Evaluation parser
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--data-dir", type=str, required=True, help="VideoMME data directory")
    eval_parser.add_argument("--input-file", type=str, required=True, help="Input file with inference results")
    eval_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    eval_parser.add_argument("--eval-model", type=str, default="gpt-3.5-turbo-0125",
                            help="Model to use for evaluation (default: gpt-3.5-turbo-0125)")
    eval_parser.add_argument("--api-type", type=str, default="dash", choices=["dash", "mit"],
                            help="API type for evaluation")
    eval_parser.add_argument("--nproc", type=int, default=4, help="Number of processes to use")
    
    args = parser.parse_args()
    
    # Automatically set tensor_parallel_size
    if args.command == 'infer' and args.tensor_parallel_size is None:
        args.tensor_parallel_size = torch.cuda.device_count()
        print(f"Auto-set tensor_parallel_size to {args.tensor_parallel_size}")
    
    if args.command == 'infer':
        run_inference(args)
    elif args.command == 'eval':
        run_evaluation(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
