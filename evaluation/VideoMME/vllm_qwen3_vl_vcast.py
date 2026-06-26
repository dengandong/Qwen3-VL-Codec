"""V-CAST Qwen3-VL patches for vLLM.

This module mirrors the repository's codec-guided vLLM monkey-patch style, but
computes token retention online from ViT features following the V-CAST scoring
logic in /u/adeng2/code/V-CAST/compressor/v_cast.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


_PATCHED = False


def _enabled_mode() -> str:
    return os.environ.get("QWEN3VL_VCAST_MODE", "off").strip().lower()


def _retain_ratio() -> float:
    return float(os.environ.get("QWEN3VL_VCAST_RETAIN_RATIO", "0.25"))


def _min_k() -> int:
    return int(os.environ.get("QWEN3VL_VCAST_MIN_K", "1"))


def _budget_temp() -> float:
    return float(os.environ.get("QWEN3VL_VCAST_BUDGET_TEMP", "0.7"))


def _verbose() -> bool:
    return os.environ.get("QWEN3VL_VCAST_QUIET", "0") != "1"


def _tokens_per_frame(row: Any, spatial_merge_size: int) -> tuple[int, int, int, int]:
    t, h, w = [int(x) for x in (row.detach().cpu().tolist() if isinstance(row, torch.Tensor) else row)]
    m = max(1, int(spatial_merge_size))
    gh = max(1, h // m)
    gw = max(1, w // m)
    return t, gh, gw, gh * gw


def _compute_curvature(frame_reps: torch.Tensor) -> torch.Tensor:
    t = int(frame_reps.shape[0])
    if t <= 1:
        return torch.ones((t,), device=frame_reps.device, dtype=torch.float32)
    v_in = frame_reps[1:-1] - frame_reps[:-2]
    v_out = frame_reps[2:] - frame_reps[1:-1]
    curv = 1.0 - F.cosine_similarity(v_in, v_out, dim=-1, eps=1e-6)
    ones = torch.ones((1,), device=frame_reps.device, dtype=curv.dtype)
    return torch.cat([ones, curv, ones], dim=0)


def _allocate_budget_per_frame(
    weights: torch.Tensor,
    total_budget: int,
    *,
    min_k: int,
    max_k: int,
) -> torch.Tensor:
    t = int(weights.shape[0])
    if t <= 0:
        return torch.zeros((0,), device=weights.device, dtype=torch.long)

    total_budget = int(max(0, min(int(total_budget), int(t * max_k))))
    min_k = int(max(0, min(int(min_k), int(max_k))))
    min_k_eff = min_k if total_budget >= t * min_k else 0

    alloc = torch.full((t,), int(min_k_eff), device=weights.device, dtype=torch.long)
    remaining = int(total_budget - int(alloc.sum().item()))
    if remaining <= 0:
        return alloc

    weights = weights.float().clamp_min(0.0)
    if float(weights.sum().item()) <= 0.0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum().clamp_min(1e-6)

    raw = weights * float(remaining)
    extra = torch.floor(raw).to(torch.long)
    max_extra = max(0, max_k - min_k_eff)
    extra = torch.minimum(extra, torch.full_like(extra, int(max_extra)))
    alloc = alloc + extra

    remaining = int(total_budget - int(alloc.sum().item()))
    if remaining > 0:
        frac = raw - torch.floor(raw)
        frac = frac.masked_fill(alloc >= max_k, -1.0)
        for _ in range(remaining):
            idx = torch.argmax(frac)
            if float(frac[idx].item()) < 0.0:
                break
            alloc[idx] += 1
            if alloc[idx] >= max_k:
                frac[idx] = -1.0
    elif remaining < 0:
        over = -remaining
        order = torch.argsort(weights, descending=False)
        for idx in order:
            if over <= 0:
                break
            if alloc[idx] > min_k_eff:
                alloc[idx] -= 1
                over -= 1

    return alloc


def _static_counts_per_frame(row: Any, spatial_merge_size: int, retain_ratio: float, min_k: int) -> list[int]:
    t, _, _, tpf = _tokens_per_frame(row, spatial_merge_size)
    total = max(1, min(t * tpf, int(round(float(t * tpf) * float(retain_ratio)))))
    if min_k > 0:
        total = max(total, min(t * tpf, t * min(int(min_k), tpf)))
    base = total // t
    rem = total % t
    return [min(tpf, base + (1 if i < rem else 0)) for i in range(t)]


def _vcast_select_for_video(
    frames: torch.Tensor,
    retain_ratio: float,
    min_k: int,
) -> torch.Tensor:
    """Return local token indices for one video chunk shaped [T, HW, D]."""
    t, tokens_per_frame, _ = frames.shape
    if t <= 0 or tokens_per_frame <= 0:
        return torch.zeros((0,), device=frames.device, dtype=torch.long)

    frame_reps = F.normalize(frames.mean(dim=1), dim=-1, eps=1e-6)
    curvature = _compute_curvature(frame_reps)
    weights = torch.softmax(curvature.float() / float(_budget_temp()), dim=0)

    total_budget = int(round(float(t * tokens_per_frame) * float(retain_ratio)))
    total_budget = max(1, min(int(t * tokens_per_frame), total_budget))
    min_k = int(max(0, min(int(min_k), int(tokens_per_frame))))
    if min_k > 0:
        total_budget = max(total_budget, int(t * min_k))
    k_t = _allocate_budget_per_frame(
        weights,
        total_budget,
        min_k=min_k,
        max_k=tokens_per_frame,
    )

    keep_indices = []
    for frame_idx in range(t):
        k = int(k_t[frame_idx].item())
        if k <= 0:
            continue
        frame_tokens = frames[frame_idx]
        rep = frame_reps[frame_idx]
        sim = F.cosine_similarity(frame_tokens, rep.unsqueeze(0), dim=-1, eps=1e-6)
        outlier = (1.0 - sim).float()
        norm = frame_tokens.float().norm(dim=-1)
        norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-6)
        score = outlier + norm
        topk_idx = torch.topk(score, k=k, largest=True, sorted=False).indices
        topk_idx, _ = torch.sort(topk_idx)
        keep_indices.append(topk_idx + frame_idx * tokens_per_frame)

    if not keep_indices:
        return torch.zeros((0,), device=frames.device, dtype=torch.long)
    return torch.cat(keep_indices, dim=0).to(torch.long)


def _vcast_select_for_grid(
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    retain_ratio: float,
    min_k: int,
) -> torch.Tensor:
    chunks = []
    offset = 0
    for row_idx, row in enumerate(grid_thw):
        t, _, _, tpf = _tokens_per_frame(row, spatial_merge_size)
        num = int(t * tpf)
        if num <= 0:
            continue
        if t <= 1:
            local = torch.arange(num, device=hidden_states.device, dtype=torch.long)
        else:
            frames = hidden_states[offset : offset + num].view(t, tpf, -1)
            local = _vcast_select_for_video(frames, retain_ratio, min_k)
            if local.numel() == 0:
                local = torch.arange(num, device=hidden_states.device, dtype=torch.long)
        try:
            from vllm_qwen3_vl_index_dump import dump_video_selection

            dump_video_selection(
                method="vcast",
                video_index=row_idx,
                grid_thw=row,
                spatial_merge_size=spatial_merge_size,
                dense_token_count=num,
                keep_indices=local,
                output_indices=local,
                extra={
                    "retain_ratio": float(retain_ratio),
                    "min_k": int(min_k),
                    "budget_temp": float(_budget_temp()),
                },
            )
        except Exception as exc:
            if os.environ.get("QWEN3VL_INDEX_DUMP_STRICT", "0") == "1":
                raise
            if os.environ.get("QWEN3VL_INDEX_DUMP_DIR"):
                print(f"[V-CAST-vLLM] index dump failed for video[{row_idx}]: {exc}")
        chunks.append(local + offset)
        if _verbose() and local.numel() < num:
            print(f"[V-CAST-vLLM] video[{row_idx}] keep_tokens={local.numel()}/{num}")
        offset += num
    if not chunks:
        return torch.arange(hidden_states.shape[0], device=hidden_states.device, dtype=torch.long)
    return torch.cat(chunks, dim=0).to(torch.long)


def _selected_video_sizes(grid_thw: torch.Tensor, spatial_merge_size: int, retain_ratio: float, min_k: int) -> list[int]:
    sizes = []
    for row in grid_thw:
        counts = _static_counts_per_frame(row, spatial_merge_size, retain_ratio, min_k)
        sizes.append(int(sum(counts)))
    return sizes


def apply_patch(mode: str = "post_vit", retain_ratio: float = 0.25, min_k: int = 1) -> None:
    """Patch vLLM Qwen3-VL classes in the current process."""
    global _PATCHED
    mode = (mode or "off").strip().lower()
    if mode in {"none", "off", ""}:
        return
    if mode != "post_vit":
        raise ValueError(f"Unsupported V-CAST mode for vLLM: {mode}")

    os.environ["QWEN3VL_VCAST_MODE"] = mode
    os.environ["QWEN3VL_VCAST_RETAIN_RATIO"] = str(retain_ratio)
    os.environ["QWEN3VL_VCAST_MIN_K"] = str(min_k)

    import vllm.model_executor.models.qwen3_vl as qwen3_vl

    if _PATCHED:
        return

    orig_get_prompt_updates = qwen3_vl.Qwen3VLMultiModalProcessor._get_prompt_updates
    orig_process_video_input = qwen3_vl.Qwen3VLForConditionalGeneration._process_video_input
    orig_iter_mm_grid_hw = qwen3_vl.Qwen3VLForConditionalGeneration.iter_mm_grid_hw
    orig_get_mrope_input_positions = qwen3_vl.Qwen3VLForConditionalGeneration.get_mrope_input_positions

    def patched_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        mode_now = _enabled_mode()
        if mode_now != "post_vit":
            return orig_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs)

        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        hf_config = self.info.get_hf_config()

        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        vision_end_token_id = hf_config.vision_end_token_id
        merge_length = image_processor.merge_size**2

        def get_image_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            num_tokens = int(grid_thw.prod()) // merge_length
            return [hf_processor.image_token_id] * num_tokens

        def get_video_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            video, metadata = mm_items["video"][item_idx]
            del video
            do_sample_frames = hf_processor_mm_kwargs.get("do_sample_frames")
            sampled_fps = hf_processor_mm_kwargs.get("fps")
            if qwen3_vl.is_list_of(sampled_fps, float):
                sampled_fps = sampled_fps[item_idx]
            timestamps = self.info._get_video_second_idx(metadata, out_item, do_sample_frames, sampled_fps)
            assert len(timestamps) == int(grid_thw[0])

            counts = _static_counts_per_frame(
                grid_thw,
                int(image_processor.merge_size),
                _retain_ratio(),
                _min_k(),
            )
            placeholder = []
            for curr_time, count in zip(timestamps, counts):
                frame_idx = tokenizer.encode(f"<{curr_time:.1f} seconds>", add_special_tokens=False)
                placeholder.extend(frame_idx)
                placeholder.extend(
                    [vision_start_token_id]
                    + [video_token_id] * int(count)
                    + [vision_end_token_id]
                )
            return qwen3_vl.PromptUpdateDetails.select_token_id(placeholder, video_token_id)

        return [
            qwen3_vl.PromptReplacement(
                modality="image",
                target=hf_processor.image_token,
                replacement=get_image_replacement_qwen3vl,
            ),
            qwen3_vl.PromptReplacement(
                modality="video",
                target="<|vision_start|><|video_pad|><|vision_end|>",
                replacement=get_video_replacement_qwen3vl,
            ),
        ]

    def patched_vision_forward(self, x, grid_thw):
        mode_now = _enabled_mode()
        hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        hidden_states = self.patch_embed(hidden_states)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_np = np.array(grid_thw, dtype=np.int32)
            grid_tensor = torch.tensor(grid_thw, dtype=torch.int64)
        else:
            grid_thw_list = grid_thw.tolist()
            grid_np = grid_thw.detach().cpu().numpy() if isinstance(grid_thw, torch.Tensor) else np.array(grid_thw, dtype=np.int32)
            grid_tensor = grid_thw.to(dtype=torch.int64) if isinstance(grid_thw, torch.Tensor) else torch.tensor(grid_thw, dtype=torch.int64)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw_list)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

        cu_seqlens = np.repeat(grid_np[:, 1] * grid_np[:, 2], grid_np[:, 0]).cumsum(axis=0, dtype=np.int32)
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])
        cu_seqlens = torch.from_numpy(cu_seqlens)

        hidden_states = hidden_states.unsqueeze(1)
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)
        cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen,
            )
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature_lists.append(self.deepstack_merger_list[idx](hidden_states))

        hidden_states = self.merger(hidden_states)
        if mode_now == "post_vit":
            keep = _vcast_select_for_grid(
                hidden_states,
                grid_tensor,
                int(self.spatial_merge_size),
                _retain_ratio(),
                _min_k(),
            )
            if keep.numel() != hidden_states.shape[0]:
                hidden_states = hidden_states.index_select(0, keep)
                deepstack_feature_lists = [x.index_select(0, keep) for x in deepstack_feature_lists]

        hidden_states = torch.cat([hidden_states] + deepstack_feature_lists, dim=1)
        return hidden_states

    def patched_process_video_input(self, video_input):
        mode_now = _enabled_mode()
        if mode_now != "post_vit":
            return orig_process_video_input(self, video_input)

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
            if self.use_data_parallel:
                raise RuntimeError("V-CAST vLLM patch does not support mm_encoder_tp_mode=data yet.")
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

        sizes = _selected_video_sizes(
            grid_thw,
            int(self.visual.spatial_merge_size),
            _retain_ratio(),
            _min_k(),
        )
        return video_embeds.split(sizes)

    def patched_iter_mm_grid_hw(self, input_tokens, mm_features):
        mode_now = _enabled_mode()
        if mode_now != "post_vit":
            yield from orig_iter_mm_grid_hw(self, input_tokens, mm_features)
            return
        video_token_id = self.config.video_token_id
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            offset = mm_feature.mm_position.offset
            if mm_feature.modality == "image":
                t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
                assert t == 1
                yield offset, h // spatial_merge_size, w // spatial_merge_size
            elif mm_feature.modality == "video":
                grid = mm_feature.data["video_grid_thw"].data
                counts = _static_counts_per_frame(grid, spatial_merge_size, _retain_ratio(), _min_k())
                for count in counts:
                    offset = input_tokens.index(video_token_id, offset)
                    yield offset, 1, int(count)
                    offset += int(count)
            else:
                raise ValueError(f"Unsupported modality: {mm_feature.modality}")

    qwen3_vl.Qwen3VLMultiModalProcessor._get_prompt_updates = patched_get_prompt_updates
    qwen3_vl.Qwen3_VisionTransformer.forward = patched_vision_forward
    qwen3_vl.Qwen3VLForConditionalGeneration._process_video_input = patched_process_video_input
    qwen3_vl.Qwen3VLForConditionalGeneration.iter_mm_grid_hw = patched_iter_mm_grid_hw
    qwen3_vl.Qwen3VLForConditionalGeneration.get_mrope_input_positions = orig_get_mrope_input_positions

    _PATCHED = True
    print(f"[V-CAST-vLLM] enabled mode={mode} retain_ratio={retain_ratio} min_k={min_k}")
