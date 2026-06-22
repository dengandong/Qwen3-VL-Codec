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
    """Patch qwen-vl-utils decord backend when its FFmpeg build is broken.

    On the current cluster image, decord 0.6.0 is linked against FFmpeg 8 and
    fails with "Option 'pix_fmts' is not a runtime option".  qwen-vl-utils then
    falls back to torchvision, which decodes whole videos and is much slower.
    Keep the public reader name as "decord" for existing scripts, but service it
    with OpenCV random frame reads when enabled.
    """
    if os.environ.get("QWEN3VL_PATCH_DECORD_WITH_OPENCV", "1") == "0":
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

def prepare_inputs_for_vllm(messages, processor, codec_video_id=None):
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
    
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"video": args.max_videos_per_prompt},
        seed=args.seed,
    )
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

                vllm_input = prepare_inputs_for_vllm(
                    messages,
                    processor,
                    codec_video_id=data_item.get("videoID") if args.codec_guided_mode != "none" else None,
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


def configure_codec_guidance(args):
    mode = getattr(args, "codec_guided_mode", "none")
    if mode == "none":
        return
    if getattr(args, "vcast_mode", "none") != "none":
        raise ValueError("--codec-guided-mode and --vcast-mode cannot be enabled together")

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

    from vllm_qwen3_vl_codec_guided import apply_patch as apply_qwen3_vl_codec_patch

    apply_qwen3_vl_codec_patch(mode=mode, guide_zip=str(guide_zip))

    print("\n⚙️  Codec-guided vLLM patch:")
    print(f"   mode={mode}")
    print(f"   profile={args.codec_guide_profile}")
    print(f"   guide_zip={guide_zip}")
    print(f"   max_frames={args.max_frames}")


def configure_vcast(args):
    mode = getattr(args, "vcast_mode", "none")
    if mode == "none":
        return
    if getattr(args, "codec_guided_mode", "none") != "none":
        raise ValueError("--vcast-mode and --codec-guided-mode cannot be enabled together")

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
    infer_parser.add_argument("--vcast-mode", type=str, default="none",
                            choices=["none", "post_vit"],
                            help="Enable V-CAST online post-ViT token pruning for vLLM Qwen3-VL")
    infer_parser.add_argument("--vcast-retain-ratio", type=float, default=0.25,
                            help="V-CAST retain ratio over video tokens (default: 0.25)")
    infer_parser.add_argument("--vcast-min-k", type=int, default=1,
                            help="Minimum retained tokens per video frame for V-CAST (default: 1)")
    
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
    
    # Generation parameters (aligned with MMMU/RealWorldQA for Instruct model)
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
