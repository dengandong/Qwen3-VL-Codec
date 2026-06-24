"""Temporal Token Fusion V2 utilities and vLLM Qwen3-VL patch hooks.

TTF V2 removes the single global anchor frame.  For each frame, it chooses a
dynamic anchor from a local temporal window by frame-level cosine similarity,
then applies the original local spatial token matching against that anchor.

The pure PyTorch implementation in this file is independent of vLLM so it can
be tested on CPU.  The vLLM integration supports the fixed retain-ratio path,
where prompt placeholders can be compacted before scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


_PATCHED = False
_ORIGINALS: dict[str, Any] = {}


@dataclass(frozen=True)
class TTFPlan:
    anchor_idx: int
    anchor_indices_per_frame: torch.LongTensor | None
    keep_flat_indices: torch.LongTensor
    output_flat_indices: torch.LongTensor
    output_coords: torch.LongTensor
    num_tokens_per_original_frame: list[int]
    output_frame_order: list[int]
    best_similarity: torch.Tensor | None
    matched_anchor_indices: torch.LongTensor | None
    original_token_count: int
    retained_token_count: int
    effective_threshold: float | None = None


def _validate_params(
    threshold: float,
    window_radius: int,
    anchor: str,
    order: str,
    budget_mode: str = "threshold",
    retain_ratio: float = 1.0,
    temporal_anchor_radius: int = 2,
) -> None:
    if not math.isfinite(float(threshold)):
        raise ValueError(f"TTF threshold must be finite, got {threshold!r}")
    if int(window_radius) < 0:
        raise ValueError(f"TTF window_radius must be >= 0, got {window_radius}")
    if anchor not in {"auto", "dynamic_local"}:
        raise ValueError(
            f"Unsupported TTF V2 anchor={anchor!r}; expected auto/dynamic_local"
        )
    if order not in {"paper", "temporal"}:
        raise ValueError(f"Unsupported TTF order={order!r}; expected paper/temporal")
    if budget_mode not in {"threshold", "retain_ratio"}:
        raise ValueError(
            f"Unsupported TTF budget_mode={budget_mode!r}; expected threshold/retain_ratio"
        )
    if not math.isfinite(float(retain_ratio)) or float(retain_ratio) <= 0.0:
        raise ValueError(f"TTF retain_ratio must be finite and > 0, got {retain_ratio!r}")
    if int(temporal_anchor_radius) < 1:
        raise ValueError(
            "TTF V2 temporal_anchor_radius must be >= 1, "
            f"got {temporal_anchor_radius}"
        )


def _target_retained_count(total_tokens: int, anchor_tokens: int, retain_ratio: float) -> int:
    target = int(round(float(total_tokens) * float(retain_ratio)))
    return max(1, min(int(total_tokens), target))


def _select_dynamic_anchor_indices(
    features: torch.Tensor,
    temporal_anchor_radius: int = 2,
) -> torch.LongTensor:
    """Choose one dynamic temporal anchor for every frame.

    For frame ``t``, candidates are frames in ``[t-R, t+R]`` excluding ``t``.
    Similarity is computed over projector/merger frame means in float32. Ties
    are resolved by the first candidate in chronological order.
    """
    t = int(features.shape[0])
    device = features.device
    if t == 1:
        return torch.zeros(1, device=device, dtype=torch.long)

    frame_mean = features.float().mean(dim=(1, 2))
    frame_mean = F.normalize(frame_mean, dim=-1, eps=1e-6)

    radius = int(temporal_anchor_radius)
    sim = frame_mean @ frame_mean.transpose(0, 1)
    frame_ids = torch.arange(t, device=device, dtype=torch.long)
    distance = (frame_ids[:, None] - frame_ids[None, :]).abs()
    valid = (distance <= radius) & (distance > 0)
    sim = sim.masked_fill(~valid, -torch.inf)
    return torch.argmax(sim, dim=1).to(torch.long)


def _row_major_coords(
    t: int,
    h: int,
    w: int,
    *,
    device: torch.device,
) -> torch.LongTensor:
    flat = torch.arange(t * h * w, device=device, dtype=torch.long)
    ti = flat // (h * w)
    rem = flat % (h * w)
    ri = rem // w
    ci = rem % w
    return torch.stack([ti, ri, ci], dim=1)


def _anchor_neighborhood_indices(
    h: int,
    w: int,
    window_radius: int,
    *,
    device: torch.device,
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    rows = torch.arange(h, device=device, dtype=torch.long).repeat_interleave(w)
    cols = torch.arange(w, device=device, dtype=torch.long).repeat(h)

    offsets = [
        (dr, dc)
        for dr in range(-window_radius, window_radius + 1)
        for dc in range(-window_radius, window_radius + 1)
    ]
    cand = []
    for dr, dc in offsets:
        rr = (rows + dr).clamp_(0, h - 1)
        cc = (cols + dc).clamp_(0, w - 1)
        cand.append(rr * w + cc)
    cand_idx = torch.stack(cand, dim=1)

    # Clip at the border can map several offsets to the same true anchor token.
    # Keep only the first occurrence in deterministic offset order.
    unique_mask = torch.ones_like(cand_idx, dtype=torch.bool)
    for k in range(cand_idx.shape[1]):
        if k == 0:
            continue
        duplicate = (cand_idx[:, :k] == cand_idx[:, k : k + 1]).any(dim=1)
        unique_mask[:, k] = ~duplicate
    return cand_idx, unique_mask


def build_ttf_plan(
    features: torch.Tensor,
    *,
    threshold: float = 0.70,
    window_radius: int = 1,
    anchor: str = "auto",
    order: str = "paper",
    budget_mode: str = "threshold",
    retain_ratio: float = 1.0,
    temporal_anchor_radius: int = 2,
    disable_fusion: bool = False,
) -> TTFPlan:
    """Build a TTF V2 selection plan for one video.

    Args:
        features: Projector/merger output shaped ``[T, H, W, C]``.
        threshold: Tokens with best dynamic-anchor cosine similarity
            greater than or equal to this value are removed.
        window_radius: Local spatial search radius around the same row/column.
        anchor: ``auto`` or ``dynamic_local``. Both use the V2 dynamic anchor.
        order: Kept for CLI compatibility. Since V2 has no global anchor frame,
            both ``paper`` and ``temporal`` output chronological row-major order.
        temporal_anchor_radius: Temporal frame window radius. Default 2 means
            frame ``t`` searches ``[t-2, t+2]`` excluding itself.
        disable_fusion: Internal testing/baseline switch that preserves dense
            temporal row-major order.
    """
    _validate_params(
        threshold,
        window_radius,
        anchor,
        order,
        budget_mode,
        retain_ratio,
        temporal_anchor_radius,
    )
    if features.ndim != 4:
        raise ValueError(f"TTF features must be [T,H,W,C], got shape={tuple(features.shape)}")

    t, h, w, c = [int(x) for x in features.shape]
    if t <= 0 or h <= 0 or w <= 0 or c <= 0:
        raise ValueError(f"TTF features must have positive dimensions, got {tuple(features.shape)}")

    device = features.device
    total = t * h * w
    anchor_indices = _select_dynamic_anchor_indices(features, temporal_anchor_radius)
    anchor_idx = -1

    if t == 1 or disable_fusion:
        output = torch.arange(total, device=device, dtype=torch.long)
        coords = _row_major_coords(t, h, w, device=device)
        return TTFPlan(
            anchor_idx=0,
            anchor_indices_per_frame=torch.zeros(t, device=device, dtype=torch.long),
            keep_flat_indices=output,
            output_flat_indices=output,
            output_coords=coords,
            num_tokens_per_original_frame=[h * w] * t,
            output_frame_order=list(range(t)),
            best_similarity=None,
            matched_anchor_indices=None,
            original_token_count=total,
            retained_token_count=total,
            effective_threshold=None,
        )

    tokens_per_frame = h * w
    keep_mask = torch.zeros((t, tokens_per_frame), device=device, dtype=torch.bool)

    flat = features.reshape(t, tokens_per_frame, c)
    normalized = F.normalize(flat.float(), dim=-1, eps=1e-6)
    cand_idx, cand_unique = _anchor_neighborhood_indices(
        h,
        w,
        window_radius,
        device=device,
    )

    best_full = torch.empty((t, tokens_per_frame), device=device, dtype=torch.float32)
    matched_full = torch.full((t, tokens_per_frame), -1, device=device, dtype=torch.long)
    for frame_idx in range(t):
        anchor_frame_idx = int(anchor_indices[frame_idx].item())
        anchor_tokens = normalized[anchor_frame_idx]
        anchor_candidates = anchor_tokens[cand_idx]  # [N, K, C]
        sim = torch.einsum("nc,nkc->nk", normalized[frame_idx], anchor_candidates)
        sim = sim.masked_fill(~cand_unique, -torch.inf)
        best_sim, best_k = sim.max(dim=-1)
        matched = cand_idx.gather(1, best_k.unsqueeze(-1)).squeeze(-1)
        best_full[frame_idx] = best_sim
        matched_full[frame_idx] = matched

    effective_threshold = None
    if budget_mode == "threshold":
        keep_mask = best_full < float(threshold)
        if not bool(keep_mask.any().item()):
            # Avoid producing an invalid zero-placeholder video in threshold
            # mode. The fixed retain-ratio path used by vLLM is exact and does
            # not need this fallback.
            keep_mask.reshape(-1)[torch.argmin(best_full.reshape(-1))] = True
    else:
        target = _target_retained_count(total, tokens_per_frame, float(retain_ratio))
        score = best_full.reshape(-1)
        order_idx = torch.argsort(score, stable=True)
        selected = order_idx[:target]
        effective_threshold = float(score[order_idx[target - 1]].item())
        keep_mask.reshape(-1)[selected] = True

    frame_order = list(range(t))

    output_chunks = []
    for frame_idx in frame_order:
        local = torch.nonzero(keep_mask[frame_idx], as_tuple=False).flatten()
        if local.numel() > 0:
            output_chunks.append(local + frame_idx * tokens_per_frame)
    if output_chunks:
        output_flat = torch.cat(output_chunks).to(torch.long)
    else:
        # This should be unreachable because retain-ratio keeps at least one
        # token and threshold mode falls back to the least redundant token.
        output_flat = torch.arange(tokens_per_frame, device=device, dtype=torch.long)

    keep_flat = torch.nonzero(keep_mask.reshape(-1), as_tuple=False).flatten().to(torch.long)
    coords_all = _row_major_coords(t, h, w, device=device)
    coords = coords_all.index_select(0, output_flat)
    counts = [int(keep_mask[idx].sum().item()) for idx in range(t)]

    return TTFPlan(
        anchor_idx=anchor_idx,
        anchor_indices_per_frame=anchor_indices,
        keep_flat_indices=keep_flat,
        output_flat_indices=output_flat,
        output_coords=coords,
        num_tokens_per_original_frame=counts,
        output_frame_order=frame_order,
        best_similarity=best_full,
        matched_anchor_indices=matched_full,
        original_token_count=total,
        retained_token_count=int(output_flat.numel()),
        effective_threshold=effective_threshold,
    )


def build_ttf_plans_for_videos(
    videos: Sequence[torch.Tensor],
    *,
    threshold: float = 0.70,
    window_radius: int = 1,
    anchor: str = "auto",
    order: str = "paper",
    budget_mode: str = "threshold",
    retain_ratio: float = 1.0,
    temporal_anchor_radius: int = 2,
    disable_fusion: bool = False,
) -> list[TTFPlan]:
    return [
        build_ttf_plan(
            video,
            threshold=threshold,
            window_radius=window_radius,
            anchor=anchor,
            order=order,
            budget_mode=budget_mode,
            retain_ratio=retain_ratio,
            temporal_anchor_radius=temporal_anchor_radius,
            disable_fusion=disable_fusion,
        )
        for video in videos
    ]


def apply_ttf_plan_to_flat_embeddings(
    embeddings: torch.Tensor,
    plan: TTFPlan,
) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected flat embeddings [N,C], got {tuple(embeddings.shape)}")
    if embeddings.shape[0] != plan.original_token_count:
        raise ValueError(
            "Embedding rows do not match TTF plan: "
            f"expected={plan.original_token_count} actual={embeddings.shape[0]}"
        )
    return embeddings.index_select(0, plan.output_flat_indices.to(embeddings.device))


def apply_ttf_plan_to_deepstack(
    deepstack_features: Iterable[torch.Tensor],
    plan: TTFPlan,
) -> list[torch.Tensor]:
    return [apply_ttf_plan_to_flat_embeddings(layer, plan) for layer in deepstack_features]


def gather_dense_mrope_positions(
    dense_positions: torch.Tensor,
    plan: TTFPlan,
) -> torch.Tensor:
    """Gather sparse M-RoPE positions using a TTF plan.

    Accepts either ``[3, N]`` or ``[N, 3]`` dense position tensors and preserves
    the input orientation.
    """
    if dense_positions.ndim != 2:
        raise ValueError(f"dense_positions must be rank 2, got {tuple(dense_positions.shape)}")
    idx = plan.output_flat_indices.to(dense_positions.device)
    if dense_positions.shape[0] == 3 and dense_positions.shape[1] == plan.original_token_count:
        return dense_positions.index_select(1, idx)
    if dense_positions.shape[1] == 3 and dense_positions.shape[0] == plan.original_token_count:
        return dense_positions.index_select(0, idx)
    raise ValueError(
        "dense_positions shape does not match TTF plan: "
        f"shape={tuple(dense_positions.shape)} expected token count={plan.original_token_count}"
    )


def verify_ttf_lengths(
    *,
    plan: TTFPlan,
    placeholder_count: int,
    embedding_rows: int,
    mrope_count: int,
    video_index: int = 0,
) -> None:
    expected = int(plan.retained_token_count)
    if placeholder_count != expected or embedding_rows != expected or mrope_count != expected:
        raise RuntimeError(
            "[TTF-V2-vLLM] invariant failed for "
            f"video[{video_index}]: placeholders={placeholder_count}, "
            f"embeds={embedding_rows}, mrope={mrope_count}, expected={expected}, "
            f"dense={plan.original_token_count}"
        )


def _row_to_thw(row: Any) -> tuple[int, int, int]:
    if isinstance(row, torch.Tensor):
        return tuple(int(x) for x in row.detach().cpu().reshape(-1).tolist())  # type: ignore[return-value]
    return tuple(int(x) for x in row)  # type: ignore[return-value]


def _tokens_per_frame_for_row(row: Any, spatial_merge_size: int) -> tuple[int, int, int, int]:
    t, h, w = _row_to_thw(row)
    m = max(1, int(spatial_merge_size))
    dense_h = h // m
    dense_w = w // m
    return t, dense_h, dense_w, dense_h * dense_w


def _target_count_for_row(row: Any, spatial_merge_size: int, retain_ratio: float) -> int:
    t, _, _, tokens_per_frame = _tokens_per_frame_for_row(row, spatial_merge_size)
    total = int(t * tokens_per_frame)
    if t <= 1:
        return total
    return _target_retained_count(total, tokens_per_frame, retain_ratio)


def _selected_video_sizes(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    retain_ratio: float,
) -> list[int]:
    return [
        _target_count_for_row(row, spatial_merge_size, retain_ratio)
        for row in grid_thw
    ]


def _dense_local_mrope_positions(
    t: int,
    dense_h: int,
    dense_w: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return local media M-RoPE positions shaped [T*H*W, 4].

    The last channel follows vLLM EVS convention and stores the media width so
    the generic recompute helper can shift trailing text positions.
    """
    ti = torch.arange(t, device=device, dtype=torch.long).view(-1, 1).expand(-1, dense_h * dense_w).flatten()
    hi = torch.arange(dense_h, device=device, dtype=torch.long).view(1, -1, 1).expand(t, -1, dense_w).flatten()
    wi = torch.arange(dense_w, device=device, dtype=torch.long).view(1, 1, -1).expand(t, dense_h, -1).flatten()
    width = torch.full_like(ti, int(dense_w))
    return torch.stack([ti, hi, wi, width], dim=1).to(dtype=dtype)


def _compress_video_outputs(
    main: torch.Tensor,
    deepstack: Sequence[torch.Tensor],
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    threshold: float,
    window_radius: int,
    anchor: str,
    order: str,
    budget_mode: str,
    retain_ratio: float,
    temporal_anchor_radius: int,
    debug_verify: bool,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    offset = 0
    for video_idx, row in enumerate(grid_thw):
        t, dense_h, dense_w, tokens_per_frame = _tokens_per_frame_for_row(
            row,
            spatial_merge_size,
        )
        dense_tokens = int(t * tokens_per_frame)
        main_chunk = main[offset : offset + dense_tokens]
        deep_chunks = [layer[offset : offset + dense_tokens] for layer in deepstack]

        if t <= 1 or _identity_only():
            plan = build_ttf_plan(
                main_chunk.reshape(t, dense_h, dense_w, -1),
                threshold=threshold,
                window_radius=window_radius,
                anchor=anchor,
                order="temporal",
                temporal_anchor_radius=temporal_anchor_radius,
                disable_fusion=True,
            )
        else:
            plan = build_ttf_plan(
                main_chunk.reshape(t, dense_h, dense_w, -1),
                threshold=threshold,
                window_radius=window_radius,
                anchor=anchor,
                order=order,
                budget_mode=budget_mode,
                retain_ratio=retain_ratio,
                temporal_anchor_radius=temporal_anchor_radius,
            )

        selected = plan.output_flat_indices.to(main.device)
        selected_main = main_chunk.index_select(0, selected)
        selected_deep = [layer.index_select(0, selected) for layer in deep_chunks]
        dense_pos = _dense_local_mrope_positions(
            t,
            dense_h,
            dense_w,
            device=main.device,
            dtype=main.dtype,
        )
        selected_pos = dense_pos.index_select(0, selected)
        out = torch.cat([selected_main] + selected_deep + [selected_pos], dim=1)
        chunks.append(out)

        if _verbose() or debug_verify:
            reduction = 1.0 - (plan.retained_token_count / max(plan.original_token_count, 1))
            eff = (
                "n/a"
                if plan.effective_threshold is None
                else f"{plan.effective_threshold:.4f}"
            )
            print(
                f"[TTF-V2-vLLM] video[{video_idx}] dynamic_radius={temporal_anchor_radius} "
                f"order=temporal budget_mode={budget_mode} "
                f"tokens={plan.original_token_count}->{plan.retained_token_count} "
                f"reduction={reduction:.2%} effective_threshold={eff}"
            )
        if debug_verify:
            verify_ttf_lengths(
                plan=plan,
                placeholder_count=_target_count_for_row(row, spatial_merge_size, retain_ratio)
                if budget_mode == "retain_ratio"
                else plan.retained_token_count,
                embedding_rows=out.shape[0],
                mrope_count=selected_pos.shape[0],
                video_index=video_idx,
            )

        offset += dense_tokens
    return torch.cat(chunks, dim=0) if chunks else main.new_empty((0, main.shape[-1] + 4))


def _append_dense_image_positions(
    image_embeds: torch.Tensor,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, ...]:
    chunks = []
    offset = 0
    for row in grid_thw:
        t, dense_h, dense_w, tokens_per_frame = _tokens_per_frame_for_row(
            row,
            spatial_merge_size,
        )
        dense_tokens = int(t * tokens_per_frame)
        emb = image_embeds[offset : offset + dense_tokens]
        pos = _dense_local_mrope_positions(
            t,
            dense_h,
            dense_w,
            device=emb.device,
            dtype=emb.dtype,
        )
        chunks.append(torch.cat([emb, pos], dim=1))
        offset += dense_tokens
    return tuple(chunks)


def _enabled_mode() -> str:
    return os.environ.get("QWEN3VL_TTF_V2_MODE", "none").strip().lower()


def _threshold() -> float:
    return float(os.environ.get("QWEN3VL_TTF_V2_THRESHOLD", "0.70"))


def _budget_mode() -> str:
    return os.environ.get("QWEN3VL_TTF_V2_BUDGET_MODE", "retain_ratio").strip().lower()


def _retain_ratio() -> float:
    return float(os.environ.get("QWEN3VL_TTF_V2_RETAIN_RATIO", "0.25"))


def _window_radius() -> int:
    return int(os.environ.get("QWEN3VL_TTF_V2_WINDOW_RADIUS", "1"))


def _anchor() -> str:
    return os.environ.get("QWEN3VL_TTF_V2_ANCHOR", "auto").strip().lower()


def _order() -> str:
    return os.environ.get("QWEN3VL_TTF_V2_ORDER", "paper").strip().lower()


def _temporal_anchor_radius() -> int:
    return int(os.environ.get("QWEN3VL_TTF_V2_TEMPORAL_ANCHOR_RADIUS", "2"))


def _debug_verify() -> bool:
    return os.environ.get("QWEN3VL_TTF_V2_DEBUG_VERIFY", "0") == "1"


def _verbose() -> bool:
    return os.environ.get("QWEN3VL_TTF_V2_QUIET", "0") != "1"


def _identity_only() -> bool:
    return os.environ.get("QWEN3VL_TTF_V2_DISABLE_FUSION", "0") == "1" or (
        _budget_mode() == "threshold" and _threshold() > 1.0
    )


def _unsupported_dynamic_message() -> str:
    return (
        "[TTF-V2-vLLM] Current local vLLM Qwen3-VL API expands video placeholders "
        "before projector/merger features are available. Threshold TTF needs "
        "post-ViT features to determine a dynamic retained length, so this "
        "raw-video path would require dense placeholders and would not shorten "
        "LLM prefill/KV. Refusing to run pseudo-pruning. Use identity mode "
        "(QWEN3VL_TTF_V2_DISABLE_FUSION=1 or threshold > 1) for parity checks, or "
        "add a request-local two-stage precompute path that supplies retained "
        "counts before prompt processing."
    )


def _feature_report(qwen3_vl: Any) -> dict[str, bool]:
    model_cls = getattr(qwen3_vl, "Qwen3VLForConditionalGeneration", None)
    processor_cls = getattr(qwen3_vl, "Qwen3VLMultiModalProcessor", None)
    visual_cls = getattr(qwen3_vl, "Qwen3_VisionTransformer", None)
    return {
        "SupportsMultiModalPruning": hasattr(qwen3_vl, "SupportsMultiModalPruning"),
        "module_is_multimodal_pruning_enabled": hasattr(qwen3_vl, "is_multimodal_pruning_enabled"),
        "_postprocess_video_embeds_evs": hasattr(model_cls, "_postprocess_video_embeds_evs") if model_cls else False,
        "_create_final_video_embeddings": hasattr(model_cls, "_create_final_video_embeddings") if model_cls else False,
        "recompute_mrope_positions": hasattr(model_cls, "recompute_mrope_positions") if model_cls else False,
        "get_mrope_input_positions": hasattr(model_cls, "get_mrope_input_positions") if model_cls else False,
        "compute_mrope_for_media": hasattr(qwen3_vl, "compute_mrope_for_media"),
        "Qwen3VLMultiModalProcessor._get_prompt_updates": hasattr(processor_cls, "_get_prompt_updates") if processor_cls else False,
        "Qwen3VLForConditionalGeneration._process_video_input": hasattr(model_cls, "_process_video_input") if model_cls else False,
        "Qwen3_VisionTransformer.forward": hasattr(visual_cls, "forward") if visual_cls else False,
        "iter_mm_grid_hw": hasattr(model_cls, "iter_mm_grid_hw") if model_cls else False,
    }


def apply_patch(
    mode: str = "post_vit",
    threshold: float = 0.70,
    budget_mode: str = "retain_ratio",
    retain_ratio: float = 0.25,
    window_radius: int = 1,
    anchor: str = "auto",
    order: str = "paper",
    temporal_anchor_radius: int = 2,
    debug_verify: bool = False,
) -> None:
    """Patch vLLM Qwen3-VL classes in the current process.

    ``budget_mode="retain_ratio"`` is the supported vLLM path for local
    Qwen3-VL.  It preserves the TTF local-anchor scoring, but fixes the total
    retained count before scheduler admission so prompt/KV length is truly
    compressed.  ``budget_mode="threshold"`` still needs a two-stage precompute
    path and fails fast except for explicit identity/parity checks.
    """
    global _PATCHED
    mode = (mode or "none").strip().lower()
    anchor = (anchor or "auto").strip().lower()
    order = (order or "paper").strip().lower()
    if mode in {"none", "off", ""}:
        return
    if mode != "post_vit":
        raise ValueError(f"Unsupported TTF mode for vLLM: {mode}")
    budget_mode = (budget_mode or "retain_ratio").strip().lower()
    _validate_params(
        float(threshold),
        int(window_radius),
        anchor,
        order,
        budget_mode,
        float(retain_ratio),
        int(temporal_anchor_radius),
    )

    os.environ["QWEN3VL_TTF_V2_MODE"] = mode
    os.environ["QWEN3VL_TTF_V2_THRESHOLD"] = str(float(threshold))
    os.environ["QWEN3VL_TTF_V2_BUDGET_MODE"] = budget_mode
    os.environ["QWEN3VL_TTF_V2_RETAIN_RATIO"] = str(float(retain_ratio))
    os.environ["QWEN3VL_TTF_V2_WINDOW_RADIUS"] = str(int(window_radius))
    os.environ["QWEN3VL_TTF_V2_ANCHOR"] = anchor
    os.environ["QWEN3VL_TTF_V2_ORDER"] = order
    os.environ["QWEN3VL_TTF_V2_TEMPORAL_ANCHOR_RADIUS"] = str(int(temporal_anchor_radius))
    os.environ["QWEN3VL_TTF_V2_DEBUG_VERIFY"] = "1" if debug_verify else "0"

    if budget_mode == "threshold" and threshold <= 1.0:
        raise NotImplementedError(_unsupported_dynamic_message())

    import vllm.model_executor.models.qwen3_vl as qwen3_vl

    if _PATCHED:
        return

    required = _feature_report(qwen3_vl)
    for name in (
        "Qwen3VLMultiModalProcessor._get_prompt_updates",
        "Qwen3VLForConditionalGeneration._process_video_input",
        "Qwen3_VisionTransformer.forward",
        "get_mrope_input_positions",
        "iter_mm_grid_hw",
    ):
        if not required.get(name, False):
            raise AttributeError(f"[TTF-V2-vLLM] Unsupported local Qwen3-VL API; missing {name}")

    model_cls = qwen3_vl.Qwen3VLForConditionalGeneration
    processor_cls = qwen3_vl.Qwen3VLMultiModalProcessor

    orig_get_prompt_updates = processor_cls._get_prompt_updates
    orig_process_image_input = model_cls._process_image_input
    orig_process_video_input = model_cls._process_video_input
    orig_iter_mm_grid_hw = model_cls.iter_mm_grid_hw
    orig_get_mrope_input_positions = model_cls.get_mrope_input_positions

    def patched_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        mode_now = _enabled_mode()
        if mode_now != "post_vit":
            return orig_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
        if _identity_only():
            return orig_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
        if _budget_mode() == "threshold":
            raise NotImplementedError(_unsupported_dynamic_message())

        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
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
            count = _target_count_for_row(
                grid_thw,
                int(image_processor.merge_size),
                _retain_ratio(),
            )
            placeholder = (
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
        if mode_now != "post_vit" or _identity_only() or _budget_mode() == "threshold":
            return _ORIGINALS["Qwen3_VisionTransformer.forward"](self, x, grid_thw)

        hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        hidden_states = self.patch_embed(hidden_states)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_np = np.array(grid_thw, dtype=np.int32)
            grid_tensor = torch.tensor(grid_thw, dtype=torch.int64)
        else:
            grid_thw_list = grid_thw.tolist()
            grid_np = (
                grid_thw.detach().cpu().numpy()
                if isinstance(grid_thw, torch.Tensor)
                else np.array(grid_thw, dtype=np.int32)
            )
            grid_tensor = (
                grid_thw.to(dtype=torch.int64)
                if isinstance(grid_thw, torch.Tensor)
                else torch.tensor(grid_thw, dtype=torch.int64)
            )

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw_list)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

        cu_np = np.repeat(grid_np[:, 1] * grid_np[:, 2], grid_np[:, 0]).cumsum(
            axis=0,
            dtype=np.int32,
        )
        cu_np = np.concatenate([np.zeros(1, dtype=np.int32), cu_np])
        cu_seqlens = torch.from_numpy(cu_np)

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

        main = self.merger(hidden_states)
        return _compress_video_outputs(
            main,
            deepstack_feature_lists,
            grid_tensor,
            int(self.spatial_merge_size),
            _threshold(),
            _window_radius(),
            _anchor(),
            _order(),
            _budget_mode(),
            _retain_ratio(),
            _temporal_anchor_radius(),
            _debug_verify(),
        )

    def patched_process_image_input(self, image_input):
        if _enabled_mode() != "post_vit" or _identity_only() or _budget_mode() == "threshold":
            return orig_process_image_input(self, image_input)
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            if self.use_data_parallel:
                raise RuntimeError("TTF V2 vLLM patch does not support mm_encoder_tp_mode=data yet.")
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
            # self.visual is patched for video only by caller context; image
            # inputs should still be dense, so strip no dimensions here.
            if image_embeds.shape[1] == self.visual.out_hidden_size + 4:
                return image_embeds.split((grid_thw.prod(-1) // self.visual.spatial_merge_size // self.visual.spatial_merge_size).tolist())

        return _append_dense_image_positions(
            image_embeds,
            grid_thw,
            int(self.visual.spatial_merge_size),
        )

    def patched_process_video_input(self, video_input):
        mode_now = _enabled_mode()
        if mode_now != "post_vit":
            return orig_process_video_input(self, video_input)
        if _identity_only():
            return orig_process_video_input(self, video_input)
        if _budget_mode() == "threshold":
            raise NotImplementedError(_unsupported_dynamic_message())

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        if video_input["type"] == "video_embeds":
            raise NotImplementedError("[TTF-V2-vLLM] budgeted TTF currently requires raw pixel_values_videos.")
        pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
        if self.use_data_parallel:
            raise RuntimeError("TTF V2 vLLM patch does not support mm_encoder_tp_mode=data yet.")
        video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)
        sizes = _selected_video_sizes(
            grid_thw,
            int(self.visual.spatial_merge_size),
            _retain_ratio(),
        )
        return video_embeds.split(sizes)

    def patched_iter_mm_grid_hw(self, input_tokens, mm_features):
        mode_now = _enabled_mode()
        if mode_now != "post_vit" or _identity_only() or _budget_mode() == "threshold":
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
                count = _target_count_for_row(grid, spatial_merge_size, _retain_ratio())
                offset = input_tokens.index(video_token_id, offset)
                yield offset, 1, int(count)
            else:
                raise ValueError(f"Unsupported modality: {mm_feature.modality}")

    def recompute_mrope_positions(self, input_ids, multimodal_embeddings, mrope_positions, num_computed_tokens):
        if _enabled_mode() != "post_vit":
            return multimodal_embeddings, mrope_positions, int((mrope_positions.max() + 1 - len(input_ids)).item())
        if _identity_only():
            return multimodal_embeddings, mrope_positions, int((mrope_positions.max() + 1 - len(input_ids)).item())
        if _budget_mode() == "threshold":
            raise NotImplementedError(_unsupported_dynamic_message())
        from vllm.multimodal.evs import recompute_mrope_positions as _recompute

        device = multimodal_embeddings[0].device if multimodal_embeddings else mrope_positions.device
        input_ids_t = torch.as_tensor(input_ids, device=device, dtype=torch.long)
        mm_embeddings_out = [mm[:, :-4] for mm in multimodal_embeddings]
        mm_positions = [mm[:, -4:].permute(1, 0).long() for mm in multimodal_embeddings]
        positions, delta = _recompute(
            input_ids_t,
            mm_positions,
            mrope_positions,
            num_computed_tokens,
            self.config.vision_start_token_id,
            self.config.image_token_id,
            self.config.video_token_id,
        )
        if _debug_verify():
            for idx, (emb, pos) in enumerate(zip(mm_embeddings_out, mm_positions)):
                if emb.shape[0] != pos.shape[1]:
                    raise RuntimeError(
                        f"[TTF-V2-vLLM] recompute invariant failed for mm[{idx}]: "
                        f"embeds={emb.shape[0]} positions={pos.shape[1]}"
                    )
        return tuple(mm_embeddings_out), positions, int(delta)

    _ORIGINALS["Qwen3VLMultiModalProcessor._get_prompt_updates"] = orig_get_prompt_updates
    _ORIGINALS["Qwen3_VisionTransformer.forward"] = qwen3_vl.Qwen3_VisionTransformer.forward
    _ORIGINALS["Qwen3VLForConditionalGeneration._process_image_input"] = orig_process_image_input
    _ORIGINALS["Qwen3VLForConditionalGeneration._process_video_input"] = orig_process_video_input
    _ORIGINALS["Qwen3VLForConditionalGeneration.iter_mm_grid_hw"] = orig_iter_mm_grid_hw
    _ORIGINALS["Qwen3VLForConditionalGeneration.get_mrope_input_positions"] = orig_get_mrope_input_positions
    _ORIGINALS["Qwen3VLForConditionalGeneration.recompute_mrope_positions"] = getattr(
        model_cls,
        "recompute_mrope_positions",
        None,
    )
    _ORIGINALS["Qwen3VLForConditionalGeneration.supports_multimodal_pruning"] = getattr(
        model_cls,
        "supports_multimodal_pruning",
        None,
    )

    processor_cls._get_prompt_updates = patched_get_prompt_updates
    qwen3_vl.Qwen3_VisionTransformer.forward = patched_vision_forward
    model_cls._process_image_input = patched_process_image_input
    model_cls._process_video_input = patched_process_video_input
    model_cls.iter_mm_grid_hw = patched_iter_mm_grid_hw
    model_cls.get_mrope_input_positions = orig_get_mrope_input_positions
    model_cls.recompute_mrope_positions = recompute_mrope_positions
    model_cls.supports_multimodal_pruning = True

    _PATCHED = True
    if _verbose():
        print(
            f"[TTF-V2-vLLM] enabled mode={mode} budget_mode={budget_mode} "
            f"retain_ratio={float(retain_ratio):.4f} threshold={float(threshold):.4f} "
            f"spatial_radius={int(window_radius)} temporal_anchor_radius={int(temporal_anchor_radius)} "
            f"anchor={anchor} order=temporal"
        )
        print(f"[TTF-V2-vLLM] local_api={required}")
