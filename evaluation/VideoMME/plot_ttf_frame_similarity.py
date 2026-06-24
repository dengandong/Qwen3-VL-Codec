#!/usr/bin/env python3
"""Plot TTF frame/global similarity curves for sampled VideoMME videos."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from qwen_vl_utils import process_vision_info  # noqa: E402
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: E402


def _patch_video_reader() -> None:
    try:
        from run_videomme import configure_video_reader_backend

        configure_video_reader_backend()
    except Exception as exc:
        print(f"[frame-sim] video reader patch skipped: {exc}", flush=True)


def _load_dataset_video_ids(
    data_dir: str,
    video_dir: Path,
    duration: str,
    num_videos: int,
) -> list[str]:
    try:
        from datasets import load_dataset

        ids: list[str] = []
        for row in load_dataset(data_dir)["test"]:
            if duration != "all" and row.get("duration") != duration:
                continue
            video_id = str(row["videoID"])
            if video_id in ids:
                continue
            if not (video_dir / f"{video_id}.mp4").exists():
                continue
            ids.append(video_id)
            if len(ids) >= num_videos:
                return ids
    except Exception as exc:
        print(f"[frame-sim] dataset selection failed, fallback to mp4 listing: {exc}", flush=True)

    return [path.stem for path in sorted(video_dir.glob("*.mp4"))[:num_videos]]


def _build_video_message(
    video_path: Path,
    *,
    fps: float,
    min_frames: int,
    max_frames: int,
    min_pixels: int,
    max_pixels: int,
    total_pixels: int,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "video": str(video_path),
                    "fps": fps,
                    "min_frames": min_frames,
                    "max_frames": max_frames,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "total_pixels": total_pixels,
                },
                {"text": "Analyze this video."},
            ],
        }
    ]


def _prepare_video_inputs(processor: AutoProcessor, messages: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=False,
    )
    if video_inputs is None:
        raise RuntimeError("process_vision_info returned no video inputs")
    video_kwargs = dict(video_kwargs)
    video_kwargs["do_resize"] = False
    video_kwargs["do_sample_frames"] = False
    del image_inputs
    inputs = processor.video_processor(
        videos=video_inputs,
        return_tensors="pt",
        **video_kwargs,
    )
    return inputs


def _first_video_embeddings(
    model: Qwen3VLForConditionalGeneration,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if "pixel_values_videos" not in inputs or "video_grid_thw" not in inputs:
        raise RuntimeError(f"processor output lacks video tensors: keys={list(inputs)}")

    visual_device = next(model.visual.parameters()).device
    pixel_values = inputs["pixel_values_videos"].to(visual_device)
    grid_thw = inputs["video_grid_thw"].to(visual_device)
    with torch.inference_mode():
        video_embeds, _ = model.get_video_features(pixel_values, grid_thw)
    if not isinstance(video_embeds, (tuple, list)) or len(video_embeds) == 0:
        raise RuntimeError(f"unexpected video_embeds type: {type(video_embeds)!r}")
    return video_embeds[0].detach(), grid_thw[0].detach().cpu()


def _frame_similarity_curve(
    video_embeds: torch.Tensor,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> dict[str, Any]:
    t, h, w = [int(x) for x in grid_thw.tolist()]
    grid_h = h // int(spatial_merge_size)
    grid_w = w // int(spatial_merge_size)
    tokens_per_frame = grid_h * grid_w
    expected = t * tokens_per_frame
    if video_embeds.shape[0] != expected:
        raise RuntimeError(
            "embedding/grid mismatch: "
            f"embeds={video_embeds.shape[0]} expected={expected} "
            f"grid_thw={(t, h, w)} merge={spatial_merge_size}"
        )

    x = video_embeds.reshape(t, tokens_per_frame, -1).float()
    frame_mean = x.mean(dim=1)
    global_mean = frame_mean.mean(dim=0, keepdim=True)
    sim = (
        F.normalize(frame_mean, dim=-1, eps=1e-6)
        * F.normalize(global_mean, dim=-1, eps=1e-6)
    ).sum(dim=-1)
    anchor_idx = int(torch.argmax(sim).item())
    return {
        "similarity": sim.detach().cpu().tolist(),
        "anchor_idx": anchor_idx,
        "anchor_similarity": float(sim[anchor_idx].detach().cpu().item()),
        "grid_thw": [t, h, w],
        "merged_grid_hw": [grid_h, grid_w],
        "tokens_per_frame": tokens_per_frame,
        "num_temporal_grids": t,
    }


def _plot_curves(records: list[dict[str, Any]], output_png: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
        for rec in records:
            sims = rec["similarity"]
            xs = list(range(len(sims)))
            label = f"{rec['video_id']} (anchor={rec['anchor_idx']})"
            ax.plot(xs, sims, linewidth=1.7, label=label)
            ax.scatter(
                [rec["anchor_idx"]],
                [rec["anchor_similarity"]],
                s=28,
                marker="o",
                zorder=4,
            )
        ax.set_title("Qwen3-VL TTF frame/global similarity on VideoMME (max_frames=128)")
        ax.set_xlabel("frame/grid index after Qwen3-VL temporal patching")
        ax.set_ylabel("cosine(frame token mean, global video mean)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=1, loc="best")
        fig.tight_layout()
        fig.savefig(output_png)
        plt.close(fig)
        return
    except ModuleNotFoundError:
        pass

    _plot_curves_with_pil(records, output_png)


def _plot_curves_with_pil(records: list[dict[str, Any]], output_png: Path) -> None:
    width, height = 1800, 980
    left, right, top, bottom = 130, 420, 95, 145
    plot_w = width - left - right
    plot_h = height - top - bottom
    bg = "white"
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (148, 103, 189),
        (255, 127, 14),
        (23, 190, 207),
    ]

    max_x = max(max(len(rec["similarity"]) - 1, 1) for rec in records)
    all_sims = [float(x) for rec in records for x in rec["similarity"]]
    y_min = min(all_sims)
    y_max = max(all_sims)
    pad = max((y_max - y_min) * 0.08, 1e-4)
    y_min -= pad
    y_max += pad

    def px(x: float) -> float:
        return left + (x / max_x) * plot_w

    def py(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(30, 30, 30), width=2)
    for i in range(6):
        yy = top + i * plot_h / 5
        draw.line([left, yy, left + plot_w, yy], fill=(225, 225, 225), width=1)
        value = y_max - i * (y_max - y_min) / 5
        draw.text((20, yy - 8), f"{value:.4f}", fill=(30, 30, 30), font=font)
    for i in range(9):
        xx = left + i * plot_w / 8
        draw.line([xx, top, xx, top + plot_h], fill=(235, 235, 235), width=1)
        value = round(i * max_x / 8)
        draw.text((xx - 12, top + plot_h + 12), str(value), fill=(30, 30, 30), font=font)

    title = "Qwen3-VL TTF frame/global similarity on VideoMME (max_frames=128)"
    draw.text((left, 30), title, fill=(0, 0, 0), font=font)
    draw.text(
        (left + plot_w // 2 - 170, height - 70),
        "frame/grid index after Qwen3-VL temporal patching",
        fill=(0, 0, 0),
        font=font,
    )
    draw.text((20, top - 35), "cosine similarity", fill=(0, 0, 0), font=font)

    legend_x = left + plot_w + 35
    legend_y = top + 20
    for idx, rec in enumerate(records):
        color = colors[idx % len(colors)]
        sims = [float(x) for x in rec["similarity"]]
        points = [(px(i), py(v)) for i, v in enumerate(sims)]
        if len(points) > 1:
            draw.line(points, fill=color, width=4, joint="curve")
        anchor_x = px(rec["anchor_idx"])
        anchor_y = py(rec["anchor_similarity"])
        draw.ellipse(
            [anchor_x - 7, anchor_y - 7, anchor_x + 7, anchor_y + 7],
            fill=color,
            outline=(0, 0, 0),
            width=1,
        )
        draw.line([legend_x, legend_y + idx * 44 + 7, legend_x + 32, legend_y + idx * 44 + 7], fill=color, width=5)
        label = f"{rec['video_id']}  anchor={rec['anchor_idx']}  sim={rec['anchor_similarity']:.4f}"
        draw.text((legend_x + 42, legend_y + idx * 44), label, fill=(0, 0, 0), font=font)

    image.save(output_png)


def _write_csv(records: list[dict[str, Any]], output_csv: Path) -> None:
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "frame_grid_index",
                "similarity",
                "is_anchor",
                "grid_thw",
                "merged_grid_hw",
                "tokens_per_frame",
            ],
        )
        writer.writeheader()
        for rec in records:
            for idx, sim in enumerate(rec["similarity"]):
                writer.writerow(
                    {
                        "video_id": rec["video_id"],
                        "frame_grid_index": idx,
                        "similarity": sim,
                        "is_anchor": idx == rec["anchor_idx"],
                        "grid_thw": "x".join(map(str, rec["grid_thw"])),
                        "merged_grid_hw": "x".join(map(str, rec["merged_grid_hw"])),
                        "tokens_per_frame": rec["tokens_per_frame"],
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--data-dir", default="/work/nvme/bglg/adeng2/hf_cache/videomme")
    parser.add_argument("--video-dir", default="/work/nvme/bglg/adeng2/hf_cache/videomme/data")
    parser.add_argument("--duration", default="short", choices=["short", "medium", "long", "all"])
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--min-pixels", type=int, default=3584)
    parser.add_argument("--max-pixels", type=int, default=786432)
    parser.add_argument("--total-pixels", type=int, default=117964800)
    parser.add_argument("--output-dir", default="/work/nvme/bglg/adeng2/qwen3vl_ttf_frame_similarity")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _patch_video_reader()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = Path(args.video_dir)
    if args.video_ids:
        video_ids = args.video_ids[: args.num_videos]
    else:
        video_ids = _load_dataset_video_ids(
            args.data_dir,
            video_dir,
            args.duration,
            args.num_videos,
        )
    if not video_ids:
        raise RuntimeError(f"No VideoMME videos found under {video_dir}")

    print(f"[frame-sim] videos={video_ids}", flush=True)
    print(f"[frame-sim] loading processor/model from {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    spatial_merge_size = int(model.config.vision_config.spatial_merge_size)

    records = []
    for video_id in video_ids:
        video_path = video_dir / f"{video_id}.mp4"
        print(f"[frame-sim] processing {video_id}: {video_path}", flush=True)
        messages = _build_video_message(
            video_path,
            fps=args.fps,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            total_pixels=args.total_pixels,
        )
        inputs = _prepare_video_inputs(processor, messages)
        embeds, grid_thw = _first_video_embeddings(model, inputs)
        rec = _frame_similarity_curve(embeds, grid_thw, spatial_merge_size)
        rec["video_id"] = video_id
        records.append(rec)
        print(
            f"[frame-sim] {video_id} grid={rec['grid_thw']} "
            f"merged_hw={rec['merged_grid_hw']} anchor={rec['anchor_idx']} "
            f"anchor_sim={rec['anchor_similarity']:.6f}",
            flush=True,
        )

    output_png = output_dir / "ttf_frame_global_similarity_f128_5videos.png"
    output_csv = output_dir / "ttf_frame_global_similarity_f128_5videos.csv"
    output_json = output_dir / "ttf_frame_global_similarity_f128_5videos.json"
    _plot_curves(records, output_png)
    _write_csv(records, output_csv)
    with output_json.open("w") as f:
        json.dump(
            {
                "model_path": args.model_path,
                "data_dir": args.data_dir,
                "video_dir": args.video_dir,
                "duration": args.duration,
                "max_frames": args.max_frames,
                "records": records,
            },
            f,
            indent=2,
        )
    print(f"[frame-sim] wrote {output_png}", flush=True)
    print(f"[frame-sim] wrote {output_csv}", flush=True)
    print(f"[frame-sim] wrote {output_json}", flush=True)


if __name__ == "__main__":
    main()
