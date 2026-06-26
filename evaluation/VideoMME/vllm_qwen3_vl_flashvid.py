"""FlashVID utilities and vLLM Qwen3-VL patch hooks.

This module implements the vision-side FlashVID pipeline for Qwen3-VL video
tokens:

* DySeg dynamic temporal segmentation.
* ADTS attention/diversity selection.
* TSTM adjacent-frame full-spatial token merging.
* Frame-wise DPC-kNN budget alignment.

The pure PyTorch implementation at the top of this file is CPU-testable and
does not import vLLM.  The runtime patch follows the compact lifecycle already
used by the local TTF/EchoPrune/MMTok integrations: compressed video embeddings
carry representative sparse local M-RoPE coordinates in their last four columns,
and vLLM's multimodal-pruning recompute path builds a compact LLM sequence.

Official FlashVID reference: https://github.com/Fanziyang-v/FlashVID
Audited commit: 983cce6e30d7a8012442bfc7557d3afa61b3572d
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


_PATCHED = False
_ORIGINALS: dict[str, Any] = {}


@dataclass(frozen=True)
class FlashVIDAlgorithmConfig:
    profile: str = "official_qwen3"
    budget_mode: str = "direct"
    retention_ratio: float = 0.20
    expansion: float = 1.25
    alpha: float = 0.70
    token_selection_method: str = "attn_div"
    temporal_threshold: float = 0.80
    do_segment: bool = True
    segment_threshold: float = 0.90
    min_segment_num: int = 4
    complementary_segment: bool = True
    budget_correction: str = "official_ceil"
    dpc_k_max: int = 7
    deepstack_mode: str = "official_gather"
    temporal_match_chunk_size: int = 256
    force_exact_count: bool = False
    debug: bool = False


@dataclass(frozen=True)
class FlashVIDSegmentPlan:
    start_frame: int
    end_frame: int
    length: int
    adts_indices_per_frame: list[torch.LongTensor]
    tstm_parent_indices: torch.LongTensor | None
    tstm_merge_mask: torch.BoolTensor | None
    tstm_root_indices_per_frame: list[torch.LongTensor]
    dpc_cluster_assignments_per_frame: list[torch.LongTensor]
    dpc_center_indices_per_frame: list[torch.LongTensor]
    requested_threshold: float
    effective_threshold: float
    initial_root_count: int
    lower_bound: int
    final_root_count: int
    planned_adts_tokens: int
    planned_tstm_tokens: int
    actual_output_tokens: int


@dataclass(frozen=True)
class FlashVIDPlan:
    representative_flat_indices: torch.LongTensor
    num_tokens_per_frame: list[int]
    output_frame_indices: torch.LongTensor
    dense_token_count: int
    planned_token_count: int
    retained_token_count: int
    num_frames: int
    grid_h: int
    grid_w: int
    segment_lengths: list[int]
    segment_cut_indices: list[int]
    segment_plans: list[FlashVIDSegmentPlan]
    requested_profile: str
    effective_token_selection_method: str
    budget_mode: str
    budget_correction: str
    requested_retention_ratio: float
    planned_pre_llm_ratio: float
    actual_pre_llm_ratio: float
    alpha: float
    temporal_threshold: float
    segment_threshold: float
    cls_attention_stats: dict[str, float] | None
    timing_stats: dict[str, float] | None
    merging_stats: dict[str, float] | None


@dataclass(frozen=True)
class FlashVIDCompressionResult:
    compressed_main_embeddings: torch.Tensor
    plan: FlashVIDPlan


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-6)


def _stable_topk_largest(scores: torch.Tensor, k: int) -> torch.LongTensor:
    k = int(k)
    if k <= 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got {tuple(scores.shape)}")
    if k >= scores.numel():
        return torch.arange(scores.numel(), device=scores.device, dtype=torch.long)
    clean = torch.nan_to_num(scores.float(), nan=-torch.inf)
    order = torch.argsort(-clean, stable=True)
    return order[:k].long()


def _stable_argmax(scores: torch.Tensor) -> torch.LongTensor:
    return _stable_topk_largest(scores, 1)[0]


def pairwise_cosine_distances(features: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine distance, `1 - cosine`, with zero-norm safety."""
    if features.ndim != 2:
        raise ValueError(f"features must be [N,C], got {tuple(features.shape)}")
    x = _normalize(features)
    return 1.0 - x @ x.T


def compute_incoming_cls_attention_reference(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Exact incoming attention per key token.

    Args:
        query_states: [num_heads, num_queries, head_dim].
        key_states: [num_heads, num_keys, head_dim].
    """
    if query_states.ndim != 3 or key_states.ndim != 3:
        raise ValueError("query_states and key_states must be [H,L,D]")
    logits = torch.matmul(query_states.float(), key_states.float().transpose(-1, -2)) * float(scale)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.sum(dim=(0, 1)) / float(query_states.shape[0] * query_states.shape[1])


def compute_incoming_cls_attention_chunked(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    scale: float,
    query_chunk_size: int,
) -> torch.Tensor:
    """Memory-bounded incoming attention per key token."""
    if int(query_chunk_size) <= 0:
        raise ValueError(f"query_chunk_size must be > 0, got {query_chunk_size}")
    if query_states.ndim != 3 or key_states.ndim != 3:
        raise ValueError("query_states and key_states must be [H,L,D]")
    h, q_len, _ = query_states.shape
    out = torch.zeros((key_states.shape[1],), device=query_states.device, dtype=torch.float32)
    chunk = int(query_chunk_size)
    for start in range(0, q_len, chunk):
        q = query_states[:, start : start + chunk].float()
        logits = torch.matmul(q, key_states.float().transpose(-1, -2)) * float(scale)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        out += probs.sum(dim=(0, 1))
    return out / float(h * q_len)


def align_premerge_attention_to_merged_tokens(
    premerge_attention: torch.Tensor,
    *,
    merge_group_indices: torch.LongTensor,
) -> torch.Tensor:
    """Mean-pool pre-merge attention into post-merger token groups."""
    if premerge_attention.ndim != 1:
        raise ValueError("premerge_attention must be 1-D")
    groups = merge_group_indices.to(premerge_attention.device, dtype=torch.long)
    if groups.ndim != 2:
        raise ValueError("merge_group_indices must be [num_merged, group_size]")
    vals = premerge_attention.index_select(0, groups.reshape(-1)).reshape(groups.shape)
    return vals.float().mean(dim=1)


def build_default_merge_group_indices(total_pre_tokens: int, spatial_merge_unit: int, *, device: torch.device) -> torch.LongTensor:
    unit = int(spatial_merge_unit)
    if unit <= 0 or int(total_pre_tokens) % unit != 0:
        raise ValueError(f"total_pre_tokens={total_pre_tokens} not divisible by spatial_merge_unit={unit}")
    return torch.arange(total_pre_tokens, device=device, dtype=torch.long).reshape(-1, unit)


def _budget_split(num_tokens_per_frame: int, cfg: FlashVIDAlgorithmConfig) -> tuple[int, int, int, float]:
    ratio = float(cfg.retention_ratio)
    if not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"retention_ratio must be in (0,1], got {cfg.retention_ratio!r}")
    expansion = float(cfg.expansion) if cfg.budget_mode == "paper_hybrid" else 1.0
    if not math.isfinite(expansion) or expansion <= 0.0:
        raise ValueError(f"expansion must be finite and > 0, got {cfg.expansion!r}")
    pre_ratio = min(1.0, ratio * expansion)
    frame_budget = max(1, min(int(num_tokens_per_frame), math.ceil(int(num_tokens_per_frame) * pre_ratio)))
    alpha = min(1.0, max(0.0, float(cfg.alpha)))
    b_adts = min(frame_budget, math.ceil(frame_budget * alpha))
    b_tstm = max(0, frame_budget - b_adts)
    return frame_budget, b_adts, b_tstm, pre_ratio


def dynamic_segment_video(
    frame_features: torch.Tensor,
    *,
    threshold: float,
    min_segment_num: int,
    complementary: bool,
) -> tuple[list[int], list[int], list[int], torch.Tensor, list[int]]:
    """Dynamic segmentation using strict `< threshold` transition cuts."""
    if frame_features.ndim != 2:
        raise ValueError(f"frame_features must be [T,C], got {tuple(frame_features.shape)}")
    t = int(frame_features.shape[0])
    if t <= 0:
        raise ValueError("num_frames must be positive")
    if t == 1:
        empty = torch.empty((0,), device=frame_features.device, dtype=torch.float32)
        return [1], [0], [1], empty, []
    g = _normalize(frame_features)
    sims = (g[:-1] * g[1:]).sum(dim=-1)
    cuts = torch.where(sims < float(threshold))[0].tolist()
    max_segments = min(t, max(1, int(min_segment_num)))
    if complementary and len(cuts) + 1 < max_segments:
        used = set(int(x) for x in cuts)
        remaining = [i for i in range(t - 1) if i not in used]
        remaining.sort(key=lambda idx: (float(sims[idx].item()), idx))
        need = max_segments - (len(cuts) + 1)
        cuts.extend(remaining[:need])
    cuts = sorted(set(int(x) for x in cuts))
    boundaries = [-1] + cuts + [t - 1]
    lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
    starts = [boundaries[i] + 1 for i in range(len(boundaries) - 1)]
    ends = [boundaries[i + 1] + 1 for i in range(len(boundaries) - 1)]
    if any(x <= 0 for x in lengths) or sum(lengths) != t:
        raise RuntimeError(f"Invalid FlashVID segments: lengths={lengths}, frames={t}")
    return lengths, starts, ends, sims, cuts


def _event_relevance_for_segment(segment_features: torch.Tensor) -> torch.Tensor:
    pooled = segment_features.float().mean(dim=1)  # [S,C]
    return torch.einsum("snc,pc->snp", segment_features.float(), pooled).mean(dim=-1)


def select_adts_tokens(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    *,
    num_tokens: int,
    method: str,
    event_relevance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
    """ADTS and ablation token selection for one frame.

    Returns selected features, sorted selected indices, and greedy-order indices.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be [N,C], got {tuple(features.shape)}")
    n, c = features.shape
    k = int(num_tokens)
    if k <= 0:
        empty = torch.empty((0,), dtype=torch.long, device=features.device)
        return features.new_empty((0, c)), empty, empty
    if k >= n:
        idx = torch.arange(n, device=features.device, dtype=torch.long)
        return features, idx, idx
    method = (method or "attn_div").strip().lower()
    attn = torch.nan_to_num(cls_attention.to(features.device).float(), nan=0.0, posinf=0.0, neginf=0.0)
    if attn.ndim != 1 or attn.numel() != n:
        raise ValueError(f"cls_attention must be [N], got {tuple(cls_attention.shape)} for N={n}")

    if method == "attn":
        greedy = _stable_topk_largest(attn, k)
        idx = torch.sort(greedy).values
        return features.index_select(0, idx), idx, greedy

    dist = pairwise_cosine_distances(features)
    if method == "div":
        calibrated = dist
    elif method in {"attn_div", "attn_div_v2"}:
        calibrated = dist * (attn * 1e6).view(1, -1)
        if method == "attn_div_v2":
            if event_relevance is None:
                raise ValueError("event_relevance is required for attn_div_v2")
            rel = event_relevance.to(features.device).float().reshape(-1)
            if rel.numel() != n:
                raise ValueError("event_relevance length must match features")
            calibrated = calibrated * rel.view(1, -1)
    else:
        raise ValueError(f"Unsupported token selection method: {method}")

    if n == 1:
        greedy = torch.zeros((1,), device=features.device, dtype=torch.long)
        return features[:1], greedy, greedy
    first_scores = torch.topk(calibrated, k=2, dim=0, largest=False).values[1]
    selected: list[torch.Tensor] = [_stable_argmax(first_scores)]
    for _ in range(1, k):
        chosen = torch.stack(selected).long()
        min_dist = calibrated.index_select(0, chosen).min(dim=0).values
        min_dist = min_dist.clone()
        min_dist[chosen] = -torch.inf
        selected.append(_stable_argmax(min_dist))
    greedy = torch.stack(selected).long()
    if torch.unique(greedy).numel() != greedy.numel():
        raise RuntimeError(f"ADTS selected duplicate indices: {greedy.tolist()}")
    idx = torch.sort(greedy).values
    return features.index_select(0, idx), idx, greedy


def tstm_match_segment(
    segment_features: torch.Tensor,
    residual_mask: torch.BoolTensor,
    *,
    temporal_threshold: float,
    lower_bound: int,
    match_chunk_size: int,
) -> tuple[torch.BoolTensor, torch.LongTensor, float, int, int]:
    """Build TSTM child->parent links using adjacent full-frame matching."""
    if segment_features.ndim != 3:
        raise ValueError("segment_features must be [S,N,C]")
    s, n, _ = segment_features.shape
    if residual_mask.shape != (s, n):
        raise ValueError(f"residual_mask must be {(s, n)}, got {tuple(residual_mask.shape)}")
    merge_mask = torch.zeros((s, n), dtype=torch.bool, device=segment_features.device)
    parent = torch.full((s, n), -1, dtype=torch.long, device=segment_features.device)
    best = torch.full((s, n), -torch.inf, dtype=torch.float32, device=segment_features.device)
    if s > 1 and int(match_chunk_size) > 0:
        v = _normalize(segment_features)
        chunk = int(match_chunk_size)
        for frame in range(1, s):
            prev_valid = torch.where(residual_mask[frame - 1])[0]
            curr_valid = torch.where(residual_mask[frame])[0]
            if prev_valid.numel() == 0 or curr_valid.numel() == 0:
                continue
            prev = v[frame - 1].index_select(0, prev_valid)
            for start in range(0, curr_valid.numel(), chunk):
                cur_idx = curr_valid[start : start + chunk]
                sim = v[frame].index_select(0, cur_idx) @ prev.T
                # Stable tie: torch.max returns first occurrence, prev_valid is sorted.
                vals, loc = sim.max(dim=1)
                best[frame, cur_idx] = vals
                parent[frame, cur_idx] = prev_valid.index_select(0, loc)
        merge_mask = best > float(temporal_threshold)
        merge_mask &= residual_mask
        merge_mask[0] = False
    total_tokens = s * n
    initial_roots = int((~merge_mask).sum().item())
    effective = float(temporal_threshold)
    if initial_roots < int(lower_bound):
        allowed_merges = max(0, total_tokens - int(lower_bound))
        flat = best.reshape(-1)
        if allowed_merges <= 0:
            effective = float("inf")
        else:
            valid = torch.nan_to_num(flat, nan=-torch.inf)
            kth = _stable_topk_largest(valid, min(allowed_merges, valid.numel()))
            effective = max(float(valid[kth[-1]].item()), -1.0 + 1e-6)
        merge_mask = (best > effective) & residual_mask
        merge_mask[0] = False
    final_roots = int((~merge_mask).sum().item())
    return merge_mask, parent, effective, initial_roots, final_roots


def aggregate_tstm_roots(
    segment_features: torch.Tensor,
    residual_mask: torch.BoolTensor,
    merge_mask: torch.BoolTensor,
    parent_indices: torch.LongTensor,
    segment_global_indices: torch.LongTensor,
) -> tuple[list[torch.Tensor], list[torch.LongTensor], list[torch.LongTensor], torch.Tensor, torch.Tensor]:
    """Aggregate child trees into root means with accumulated subtree counts."""
    s, n, c = segment_features.shape
    sums = segment_features.float().clone()
    counts = torch.ones((s, n), device=segment_features.device, dtype=torch.float32)
    for frame in range(s - 1, 0, -1):
        children = torch.where(merge_mask[frame])[0]
        if children.numel() == 0:
            continue
        parents = parent_indices[frame, children]
        valid = parents >= 0
        if not bool(valid.all().item()):
            children = children[valid]
            parents = parents[valid]
        sums[frame - 1].index_add_(0, parents, sums[frame].index_select(0, children))
        counts[frame - 1].index_add_(0, parents, counts[frame].index_select(0, children))
        counts[frame, children] = 0.0
    root_features: list[torch.Tensor] = []
    root_reps: list[torch.LongTensor] = []
    root_local: list[torch.LongTensor] = []
    for frame in range(s):
        root_mask = residual_mask[frame] & (~merge_mask[frame]) & (counts[frame] > 0)
        idx = torch.where(root_mask)[0]
        if idx.numel() == 0:
            root_features.append(segment_features.new_empty((0, c)))
            root_reps.append(segment_global_indices.new_empty((0,), dtype=torch.long))
            root_local.append(idx)
            continue
        feat = sums[frame].index_select(0, idx) / counts[frame].index_select(0, idx).unsqueeze(1).clamp_min(1.0)
        root_features.append(feat.to(dtype=segment_features.dtype))
        root_reps.append(segment_global_indices[frame].index_select(0, idx).long())
        root_local.append(idx.long())
    return root_features, root_reps, root_local, sums, counts


def _allocate_exact_counts(root_counts: list[int], target_total: int) -> list[int]:
    total = sum(root_counts)
    target = max(0, min(int(target_total), total))
    if total == 0:
        return [0 for _ in root_counts]
    raw = [target * (r / total) for r in root_counts]
    counts = [min(r, int(math.floor(x))) for r, x in zip(root_counts, raw)]
    while sum(counts) < target:
        candidates = [i for i, r in enumerate(root_counts) if counts[i] < r]
        if not candidates:
            break
        candidates.sort(key=lambda i: (-(raw[i] - counts[i]), i))
        counts[candidates[0]] += 1
    while sum(counts) > target:
        candidates = [i for i, v in enumerate(counts) if v > 0]
        candidates.sort(key=lambda i: ((raw[i] - counts[i]), i))
        counts[candidates[0]] -= 1
    return counts


def dpc_knn_single_frame(
    features: torch.Tensor,
    representatives: torch.LongTensor,
    *,
    num_clusters: int,
    dpc_k_max: int,
) -> tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
    """DPC-kNN centers and unweighted cluster mean for one frame."""
    r = int(features.shape[0])
    c = int(features.shape[-1]) if features.ndim == 2 else 0
    k = int(num_clusters)
    if r == 0 or k <= 0:
        return (
            features.new_empty((0, c)),
            representatives.new_empty((0,), dtype=torch.long),
            torch.full((r,), -1, dtype=torch.long, device=features.device),
        )
    if k >= r:
        return features, representatives.long(), torch.arange(r, device=features.device, dtype=torch.long)
    dist = torch.cdist(features.float(), features.float()) / math.sqrt(max(c, 1))
    nn_k = min(max(1, int(dpc_k_max)), r)
    nearest = torch.topk(dist, k=nn_k, dim=-1, largest=False).values
    density = torch.exp((-(nearest**2)).mean(dim=-1))
    higher = density.view(1, -1) > density.view(-1, 1)
    max_dist = dist.max()
    delta = torch.where(higher, dist, max_dist).min(dim=-1).values
    score = density * delta
    centers = _stable_topk_largest(score, k)
    centers_sorted = torch.sort(centers).values
    center_dist = dist.index_select(1, centers_sorted)
    assign = torch.argmin(center_dist, dim=1)
    # Force centers into their own cluster after sorting centers by original index.
    for cluster_id, center_idx in enumerate(centers_sorted.tolist()):
        assign[center_idx] = cluster_id
    out = features.new_zeros((k, c))
    out.index_add_(0, assign, features)
    counts = torch.bincount(assign, minlength=k).to(device=features.device, dtype=features.dtype).clamp_min(1).unsqueeze(1)
    out = out / counts
    reps = representatives.index_select(0, centers_sorted).long()
    return out, reps, assign.long()


def dpc_knn_framewise(
    frame_root_features: list[torch.Tensor],
    frame_root_representatives: list[torch.LongTensor],
    *,
    target_total: int,
    dpc_k_max: int,
    budget_correction: str,
) -> tuple[list[torch.Tensor], list[torch.LongTensor], list[torch.LongTensor], list[torch.LongTensor]]:
    counts = [int(x.shape[0]) for x in frame_root_features]
    total = sum(counts)
    if total == 0 or int(target_total) >= total:
        assignments = [torch.arange(c, device=frame_root_features[i].device, dtype=torch.long) for i, c in enumerate(counts)]
        centers = [torch.arange(c, device=frame_root_features[i].device, dtype=torch.long) for i, c in enumerate(counts)]
        return frame_root_features, frame_root_representatives, assignments, centers
    ratio = max(0.0, float(target_total) / float(total))
    if budget_correction == "exact_total":
        clusters = _allocate_exact_counts(counts, int(target_total))
    elif budget_correction == "official_ceil":
        clusters = [min(c, math.ceil(c * ratio)) for c in counts]
    else:
        raise ValueError(f"budget_correction must be official_ceil/exact_total, got {budget_correction!r}")
    out_features: list[torch.Tensor] = []
    out_reps: list[torch.LongTensor] = []
    assignments: list[torch.LongTensor] = []
    centers: list[torch.LongTensor] = []
    for feats, reps, k in zip(frame_root_features, frame_root_representatives, clusters):
        feat, rep, assign = dpc_knn_single_frame(feats, reps, num_clusters=int(k), dpc_k_max=int(dpc_k_max))
        out_features.append(feat)
        out_reps.append(rep)
        assignments.append(assign)
        if k <= 0:
            centers.append(torch.empty((0,), device=feats.device, dtype=torch.long))
        else:
            # Map center representatives back to local root row positions.
            center_rows = []
            for rr in rep.tolist():
                loc = torch.where(reps == int(rr))[0]
                center_rows.append(int(loc[0].item()) if loc.numel() else 0)
            centers.append(torch.tensor(center_rows, device=feats.device, dtype=torch.long))
    return out_features, out_reps, assignments, centers


def _segment_compression(
    segment_features: torch.Tensor,
    segment_cls_attention: torch.Tensor,
    segment_global_indices: torch.LongTensor,
    cfg: FlashVIDAlgorithmConfig,
    *,
    start_frame: int,
    b_adts: int,
    b_tstm: int,
    frame_budget: int,
    event_relevance: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.LongTensor, FlashVIDSegmentPlan]:
    s, n, c = segment_features.shape
    adts_feats: list[torch.Tensor] = []
    adts_reps: list[torch.LongTensor] = []
    adts_indices: list[torch.LongTensor] = []
    residual = torch.ones((s, n), dtype=torch.bool, device=segment_features.device)
    for frame in range(s):
        rel = None if event_relevance is None else event_relevance[frame]
        feat, idx, _ = select_adts_tokens(
            segment_features[frame],
            segment_cls_attention[frame],
            num_tokens=b_adts,
            method=cfg.token_selection_method,
            event_relevance=rel,
        )
        adts_feats.append(feat)
        adts_reps.append(segment_global_indices[frame].index_select(0, idx).long())
        adts_indices.append(idx)
        if idx.numel() > 0:
            residual[frame, idx] = False

    lower_bound = int(frame_budget * s)
    merge_mask, parent, eff_th, initial_roots, final_roots = tstm_match_segment(
        segment_features,
        residual,
        temporal_threshold=float(cfg.temporal_threshold),
        lower_bound=lower_bound,
        match_chunk_size=int(cfg.temporal_match_chunk_size),
    )
    root_feats, root_reps, root_local, _sums, counts = aggregate_tstm_roots(
        segment_features,
        residual,
        merge_mask,
        parent,
        segment_global_indices,
    )
    target_residual = int(b_tstm * s)
    correction = "exact_total" if cfg.force_exact_count else cfg.budget_correction
    dpc_feats, dpc_reps, assignments, centers = dpc_knn_framewise(
        root_feats,
        root_reps,
        target_total=target_residual,
        dpc_k_max=int(cfg.dpc_k_max),
        budget_correction=correction,
    )
    all_feats = [x for x in adts_feats if x.numel() > 0] + [x for x in dpc_feats if x.numel() > 0]
    all_reps = [x for x in adts_reps if x.numel() > 0] + [x for x in dpc_reps if x.numel() > 0]
    if all_feats:
        feats = torch.cat(all_feats, dim=0)
        reps = torch.cat(all_reps, dim=0)
        order = torch.argsort(reps, stable=True)
        feats = feats.index_select(0, order)
        reps = reps.index_select(0, order)
    else:
        feats = segment_features.new_empty((0, c))
        reps = segment_global_indices.new_empty((0,), dtype=torch.long)
    plan = FlashVIDSegmentPlan(
        start_frame=int(start_frame),
        end_frame=int(start_frame + s),
        length=int(s),
        adts_indices_per_frame=adts_indices,
        tstm_parent_indices=parent,
        tstm_merge_mask=merge_mask,
        tstm_root_indices_per_frame=root_local,
        dpc_cluster_assignments_per_frame=assignments,
        dpc_center_indices_per_frame=centers,
        requested_threshold=float(cfg.temporal_threshold),
        effective_threshold=float(eff_th),
        initial_root_count=int(initial_roots),
        lower_bound=int(lower_bound),
        final_root_count=int(sum(x.shape[0] for x in root_feats) + sum(x.numel() for x in adts_indices)),
        planned_adts_tokens=int(b_adts * s),
        planned_tstm_tokens=int(b_tstm * s),
        actual_output_tokens=int(feats.shape[0]),
    )
    return feats, reps, plan


def _make_config_with_profile(
    *,
    profile: str,
    budget_mode: str,
    retention_ratio: float,
    expansion: float,
    alpha: float | None,
    token_selection_method: str | None,
    temporal_threshold: float | None,
    do_segment: bool,
    segment_threshold: float,
    min_segment_num: int,
    complementary_segment: bool,
    budget_correction: str,
    dpc_k_max: int,
    deepstack_mode: str,
    temporal_match_chunk_size: int,
    force_exact_count: bool,
    debug: bool = False,
) -> FlashVIDAlgorithmConfig:
    profile = (profile or "official_qwen3").strip().lower()
    if profile not in {"official_qwen3", "paper_adts_v2", "custom"}:
        raise ValueError(f"profile must be official_qwen3/paper_adts_v2/custom, got {profile!r}")
    default_method = "attn_div_v2" if profile == "paper_adts_v2" else "attn_div"
    return FlashVIDAlgorithmConfig(
        profile=profile,
        budget_mode=(budget_mode or "direct").strip().lower(),
        retention_ratio=float(retention_ratio),
        expansion=float(expansion),
        alpha=0.70 if alpha is None else float(alpha),
        token_selection_method=(token_selection_method or default_method).strip().lower(),
        temporal_threshold=0.80 if temporal_threshold is None else float(temporal_threshold),
        do_segment=bool(do_segment),
        segment_threshold=float(segment_threshold),
        min_segment_num=int(min_segment_num),
        complementary_segment=bool(complementary_segment),
        budget_correction=(budget_correction or "official_ceil").strip().lower(),
        dpc_k_max=int(dpc_k_max),
        deepstack_mode=(deepstack_mode or "official_gather").strip().lower(),
        temporal_match_chunk_size=int(temporal_match_chunk_size),
        force_exact_count=bool(force_exact_count),
        debug=bool(debug),
    )


def compress_video_flashvid(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
    config: FlashVIDAlgorithmConfig,
) -> FlashVIDCompressionResult:
    """Compress one video using FlashVID vision-side merging."""
    if video_features.ndim != 3:
        raise ValueError(f"video_features must be [T,N,C], got {tuple(video_features.shape)}")
    t, n, c = [int(x) for x in video_features.shape]
    if int(grid_h) * int(grid_w) != n:
        raise ValueError(f"grid_h*grid_w must equal N: grid={grid_h}x{grid_w}, N={n}")
    if cls_attention.shape != (t, n):
        raise ValueError(f"cls_attention must be {(t, n)}, got {tuple(cls_attention.shape)}")
    frame_budget, b_adts, b_tstm, pre_ratio = _budget_split(n, config)
    planned = int(frame_budget * t)
    if config.budget_mode == "direct" and float(config.retention_ratio) >= 1.0:
        reps = torch.arange(t * n, device=video_features.device, dtype=torch.long)
        counts = [n for _ in range(t)]
        plan = FlashVIDPlan(
            representative_flat_indices=reps,
            num_tokens_per_frame=counts,
            output_frame_indices=torch.div(reps, n, rounding_mode="floor"),
            dense_token_count=t * n,
            planned_token_count=t * n,
            retained_token_count=t * n,
            num_frames=t,
            grid_h=int(grid_h),
            grid_w=int(grid_w),
            segment_lengths=[t],
            segment_cut_indices=[],
            segment_plans=[],
            requested_profile=config.profile,
            effective_token_selection_method=config.token_selection_method,
            budget_mode=config.budget_mode,
            budget_correction=config.budget_correction,
            requested_retention_ratio=float(config.retention_ratio),
            planned_pre_llm_ratio=1.0,
            actual_pre_llm_ratio=1.0,
            alpha=float(config.alpha),
            temporal_threshold=float(config.temporal_threshold),
            segment_threshold=float(config.segment_threshold),
            cls_attention_stats=None,
            timing_stats=None,
            merging_stats={"identity": 1.0},
        )
        return FlashVIDCompressionResult(video_features.reshape(-1, c), plan)

    st_total = time.time()
    frame_features = video_features.float().mean(dim=1)
    if config.do_segment:
        lengths, starts, ends, sims, cuts = dynamic_segment_video(
            frame_features,
            threshold=float(config.segment_threshold),
            min_segment_num=int(config.min_segment_num),
            complementary=bool(config.complementary_segment),
        )
    else:
        lengths, starts, ends, sims, cuts = [t], [0], [t], torch.empty((0,), device=video_features.device), []

    all_features: list[torch.Tensor] = []
    all_reps: list[torch.LongTensor] = []
    segment_plans: list[FlashVIDSegmentPlan] = []
    event_rel_all = _event_relevance_for_segment(video_features) if config.token_selection_method == "attn_div_v2" else None
    for start, end in zip(starts, ends):
        seg = video_features[start:end]
        attn = cls_attention[start:end]
        global_idx = torch.arange(t * n, device=video_features.device, dtype=torch.long).reshape(t, n)[start:end]
        event_rel = None if event_rel_all is None else event_rel_all[start:end]
        feats, reps, seg_plan = _segment_compression(
            seg,
            attn,
            global_idx,
            config,
            start_frame=start,
            b_adts=b_adts,
            b_tstm=b_tstm,
            frame_budget=frame_budget,
            event_relevance=event_rel,
        )
        all_features.append(feats)
        all_reps.append(reps)
        segment_plans.append(seg_plan)
    compressed = torch.cat(all_features, dim=0) if all_features else video_features.new_empty((0, c))
    reps = torch.cat(all_reps, dim=0) if all_reps else torch.empty((0,), device=video_features.device, dtype=torch.long)
    if torch.unique(reps).numel() != reps.numel():
        raise RuntimeError(f"[FlashVID] duplicate representative indices: {reps.tolist()}")
    order = torch.argsort(reps, stable=True)
    compressed = compressed.index_select(0, order).to(dtype=video_features.dtype)
    reps = reps.index_select(0, order).long()

    if config.force_exact_count and reps.numel() != planned:
        # vLLM requires prompt placeholders to be known before the vision pass.
        # Use a deterministic representative-order correction only for runtime
        # shape alignment; pure algorithm calls can keep official ceil overshoot.
        if reps.numel() > planned:
            keep = torch.arange(planned, device=reps.device, dtype=torch.long)
            compressed = compressed.index_select(0, keep)
            reps = reps.index_select(0, keep)
        else:
            used = torch.zeros((t * n,), device=reps.device, dtype=torch.bool)
            used[reps] = True
            needed = planned - reps.numel()
            add = torch.where(~used)[0][:needed]
            compressed = torch.cat([compressed, video_features.reshape(-1, c).index_select(0, add)], dim=0)
            reps = torch.cat([reps, add.long()], dim=0)
            order = torch.argsort(reps, stable=True)
            compressed = compressed.index_select(0, order)
            reps = reps.index_select(0, order)

    frame_ids = torch.div(reps, n, rounding_mode="floor")
    counts = [int((frame_ids == i).sum().item()) for i in range(t)]
    retained = int(reps.numel())
    stats = {
        "adts_tokens": float(sum(p.planned_adts_tokens for p in segment_plans)),
        "tstm_roots": float(sum(sum(x.numel() for x in p.tstm_root_indices_per_frame) for p in segment_plans)),
        "dpc_clusters": float(sum(sum(x.numel() for x in p.dpc_center_indices_per_frame) for p in segment_plans)),
        "synthetic_merged_tokens": float(max(0, retained - sum(p.planned_adts_tokens for p in segment_plans))),
    }
    plan = FlashVIDPlan(
        representative_flat_indices=reps,
        num_tokens_per_frame=counts,
        output_frame_indices=frame_ids.long(),
        dense_token_count=int(t * n),
        planned_token_count=int(planned),
        retained_token_count=int(retained),
        num_frames=t,
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        segment_lengths=lengths,
        segment_cut_indices=cuts,
        segment_plans=segment_plans,
        requested_profile=config.profile,
        effective_token_selection_method=config.token_selection_method,
        budget_mode=config.budget_mode,
        budget_correction=config.budget_correction,
        requested_retention_ratio=float(config.retention_ratio),
        planned_pre_llm_ratio=float(pre_ratio),
        actual_pre_llm_ratio=float(retained / max(t * n, 1)),
        alpha=float(config.alpha),
        temporal_threshold=float(config.temporal_threshold),
        segment_threshold=float(config.segment_threshold),
        cls_attention_stats={
            "mean": float(cls_attention.float().mean().item()),
            "max": float(cls_attention.float().max().item()),
            "min": float(cls_attention.float().min().item()),
        },
        timing_stats={"total_flashvid_s": float(time.time() - st_total)},
        merging_stats=stats,
    )
    return FlashVIDCompressionResult(compressed, plan)


def verify_flashvid_lengths(
    *,
    plan: FlashVIDPlan,
    placeholder_count: int,
    embedding_rows: int,
    mrope_count: int,
    deepstack_rows: Sequence[int] | None = None,
    video_index: int = 0,
) -> None:
    expected = int(plan.retained_token_count)
    values = {
        "placeholder_count": int(placeholder_count),
        "embedding_rows": int(embedding_rows),
        "mrope_count": int(mrope_count),
        "sum_per_frame": int(sum(plan.num_tokens_per_frame)),
    }
    if deepstack_rows is not None:
        for i, row in enumerate(deepstack_rows):
            values[f"deepstack[{i}]"] = int(row)
    bad = {k: v for k, v in values.items() if v != expected}
    if bad:
        raise RuntimeError(
            f"[FlashVID-vLLM] invariant failed for video[{video_index}]: "
            f"expected={expected}, dense={plan.dense_token_count}, bad={bad}"
        )


def compute_hybrid_llm_retention_ratio(num_layers: int, pruning_layer: int, expansion: float) -> float:
    l = int(num_layers)
    k = int(pruning_layer)
    gamma = float(expansion)
    if l <= 0 or k < 0 or k >= l or gamma <= 0.0:
        raise ValueError(f"Invalid hybrid params: L={l}, K={k}, gamma={gamma}")
    numerator = l - k * gamma
    denominator = gamma * (l - k)
    if numerator <= 0 or denominator <= 0:
        raise ValueError(f"Invalid hybrid ratio: L - K*gamma must be > 0, got {numerator}")
    rho = numerator / denominator
    if rho <= 0.0 or rho > 1.0:
        raise ValueError(f"Invalid computed llm retention ratio: {rho}")
    return rho


# === vLLM patch layer ===


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _enabled_mode() -> str:
    return os.environ.get("QWEN3VL_FLASHVID_MODE", "none").strip().lower()


def _verbose() -> bool:
    return not _truthy(os.environ.get("QWEN3VL_FLASHVID_QUIET"), False)


def _debug_verify() -> bool:
    return _truthy(os.environ.get("QWEN3VL_FLASHVID_DEBUG_VERIFY"), False)


def _env_config(*, force_exact_count: bool = True) -> FlashVIDAlgorithmConfig:
    return _make_config_with_profile(
        profile=os.environ.get("QWEN3VL_FLASHVID_PROFILE", "official_qwen3"),
        budget_mode=os.environ.get("QWEN3VL_FLASHVID_BUDGET_MODE", "direct"),
        retention_ratio=float(os.environ.get("QWEN3VL_FLASHVID_RETENTION_RATIO", "0.20")),
        expansion=float(os.environ.get("QWEN3VL_FLASHVID_EXPANSION", "1.25")),
        alpha=float(os.environ.get("QWEN3VL_FLASHVID_ALPHA", "0.70")),
        token_selection_method=os.environ.get("QWEN3VL_FLASHVID_TOKEN_SELECTION_METHOD", "") or None,
        temporal_threshold=float(os.environ.get("QWEN3VL_FLASHVID_TEMPORAL_THRESHOLD", "0.80")),
        do_segment=_truthy(os.environ.get("QWEN3VL_FLASHVID_DO_SEGMENT"), True),
        segment_threshold=float(os.environ.get("QWEN3VL_FLASHVID_SEGMENT_THRESHOLD", "0.90")),
        min_segment_num=int(os.environ.get("QWEN3VL_FLASHVID_MIN_SEGMENT_NUM", "4")),
        complementary_segment=_truthy(os.environ.get("QWEN3VL_FLASHVID_COMPLEMENTARY_SEGMENT"), True),
        budget_correction=os.environ.get("QWEN3VL_FLASHVID_BUDGET_CORRECTION", "official_ceil"),
        dpc_k_max=int(os.environ.get("QWEN3VL_FLASHVID_DPC_K_MAX", "7")),
        deepstack_mode=os.environ.get("QWEN3VL_FLASHVID_DEEPSTACK_MODE", "official_gather"),
        temporal_match_chunk_size=int(os.environ.get("QWEN3VL_FLASHVID_TEMPORAL_MATCH_CHUNK_SIZE", "256")),
        force_exact_count=force_exact_count,
        debug=_debug_verify(),
    )


def _row_to_thw(row: Any) -> tuple[int, int, int]:
    if isinstance(row, torch.Tensor):
        vals = row.detach().cpu().tolist()
    else:
        vals = list(row)
    return int(vals[0]), int(vals[1]), int(vals[2])


def _tokens_per_frame_for_row(row: Any, spatial_merge_size: int) -> tuple[int, int, int, int]:
    t, h, w = _row_to_thw(row)
    m = max(1, int(spatial_merge_size))
    dense_h = h // m
    dense_w = w // m
    return t, dense_h, dense_w, dense_h * dense_w


def _target_count_for_row(row: Any, spatial_merge_size: int) -> int:
    t, _, _, tokens_per_frame = _tokens_per_frame_for_row(row, spatial_merge_size)
    frame_budget, _, _, _ = _budget_split(tokens_per_frame, _env_config(force_exact_count=True))
    return int(frame_budget * t)


def _selected_video_sizes(grid_thw: torch.Tensor, spatial_merge_size: int) -> list[int]:
    return [_target_count_for_row(row, spatial_merge_size) for row in grid_thw]


def _dense_local_mrope_positions(t: int, dense_h: int, dense_w: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ti = torch.arange(t, device=device, dtype=torch.long).view(-1, 1).expand(-1, dense_h * dense_w).flatten()
    hi = torch.arange(dense_h, device=device, dtype=torch.long).view(1, -1, 1).expand(t, -1, dense_w).flatten()
    wi = torch.arange(dense_w, device=device, dtype=torch.long).view(1, 1, -1).expand(t, dense_h, -1).flatten()
    width = torch.full_like(ti, int(dense_w))
    return torch.stack([ti, hi, wi, width], dim=1).to(dtype=dtype)


def _premerge_group_mean(hidden_states: torch.Tensor, spatial_merge_unit: int) -> torch.Tensor:
    x = hidden_states.squeeze(1) if hidden_states.ndim == 3 and hidden_states.shape[1] == 1 else hidden_states
    unit = int(spatial_merge_unit)
    if x.ndim != 2 or x.shape[0] % unit != 0:
        raise ValueError(f"Invalid pre-merge hidden shape={tuple(hidden_states.shape)} unit={unit}")
    return x.float().reshape(-1, unit, x.shape[-1]).mean(dim=1)


def _attention_group_mean(pre_attention: torch.Tensor, spatial_merge_unit: int) -> torch.Tensor:
    groups = build_default_merge_group_indices(pre_attention.numel(), spatial_merge_unit, device=pre_attention.device)
    return align_premerge_attention_to_merged_tokens(pre_attention, merge_group_indices=groups)


def _feature_report(qwen3_vl: Any) -> dict[str, bool]:
    model_cls = getattr(qwen3_vl, "Qwen3VLForConditionalGeneration", None)
    processor_cls = getattr(qwen3_vl, "Qwen3VLMultiModalProcessor", None)
    visual_cls = getattr(qwen3_vl, "Qwen3_VisionTransformer", None)
    return {
        "SupportsMultiModalPruning": hasattr(qwen3_vl, "SupportsMultiModalPruning"),
        "Qwen3VLMultiModalProcessor._get_prompt_updates": hasattr(processor_cls, "_get_prompt_updates") if processor_cls else False,
        "Qwen3VLForConditionalGeneration._parse_and_validate_video_input": hasattr(model_cls, "_parse_and_validate_video_input") if model_cls else False,
        "Qwen3VLForConditionalGeneration._process_video_input": hasattr(model_cls, "_process_video_input") if model_cls else False,
        "Qwen3_VisionTransformer.forward": hasattr(visual_cls, "forward") if visual_cls else False,
        "iter_mm_grid_hw": hasattr(model_cls, "iter_mm_grid_hw") if model_cls else False,
        "get_mrope_input_positions": hasattr(model_cls, "get_mrope_input_positions") if model_cls else False,
    }


def _extract_last_block_incoming_attention(
    blk: Any,
    normed_hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb_cos: torch.Tensor,
    rotary_pos_emb_sin: torch.Tensor,
    *,
    query_chunk_size: int,
) -> torch.Tensor:
    from vllm.model_executor.models.qwen2_vl import apply_rotary_pos_emb_vision

    qkv_out, _ = blk.attn.qkv(normed_hidden_states)
    seq_len, batch_size, _ = qkv_out.shape
    heads = blk.attn.num_attention_heads_per_partition
    head_dim = blk.attn.hidden_size_per_attention_head
    qkv = qkv_out.reshape(seq_len, batch_size, 3, heads, head_dim).permute(1, 0, 2, 3, 4)
    qk = qkv[:, :, :2]
    qk_reshaped = qk.permute(2, 0, 1, 3, 4).reshape(2 * batch_size, seq_len, heads, head_dim)
    qk_rotated = apply_rotary_pos_emb_vision(qk_reshaped, cos=rotary_pos_emb_cos, sin=rotary_pos_emb_sin)
    qk_rotated = qk_rotated.view(2, batch_size, seq_len, heads, head_dim)
    q, k = qk_rotated.unbind(dim=0)
    q = q[0].permute(1, 0, 2).contiguous()  # [heads, seq, head_dim]
    k = k[0].permute(1, 0, 2).contiguous()
    pieces = []
    scale = 1.0 / math.sqrt(float(head_dim))
    starts = cu_seqlens.detach().cpu().tolist()
    for start, end in zip(starts[:-1], starts[1:]):
        qs = q[:, int(start) : int(end)]
        ks = k[:, int(start) : int(end)]
        pieces.append(
            compute_incoming_cls_attention_chunked(
                qs,
                ks,
                scale=scale,
                query_chunk_size=int(query_chunk_size),
            )
        )
    return torch.cat(pieces, dim=0)


def _compress_video_outputs(
    main: torch.Tensor,
    deepstack: Sequence[torch.Tensor],
    cls_attention: torch.Tensor,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    debug_verify: bool,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    offset = 0
    for video_idx, row in enumerate(grid_thw):
        st = time.time()
        t, dense_h, dense_w, tokens_per_frame = _tokens_per_frame_for_row(row, spatial_merge_size)
        dense_tokens = int(t * tokens_per_frame)
        target = _target_count_for_row(row, spatial_merge_size)
        main_chunk = main[offset : offset + dense_tokens]
        deep_chunks = [layer[offset : offset + dense_tokens] for layer in deepstack]
        attn_chunk = cls_attention[offset : offset + dense_tokens].reshape(t, tokens_per_frame)
        cfg = _env_config(force_exact_count=True)
        result = compress_video_flashvid(
            main_chunk.reshape(t, tokens_per_frame, -1),
            attn_chunk,
            grid_h=dense_h,
            grid_w=dense_w,
            config=cfg,
        )
        plan = result.plan
        reps = plan.representative_flat_indices.to(main.device)
        selected_deep = [layer.index_select(0, reps) for layer in deep_chunks]
        dense_pos = _dense_local_mrope_positions(t, dense_h, dense_w, device=main.device, dtype=main.dtype)
        selected_pos = dense_pos.index_select(0, reps)
        out = torch.cat([result.compressed_main_embeddings.to(main.dtype)] + selected_deep + [selected_pos], dim=1)
        chunks.append(out)
        if _verbose() or debug_verify:
            extra = f" segments={plan.segment_lengths} per_frame={plan.num_tokens_per_frame}" if debug_verify else ""
            print(
                f"[FlashVID-vLLM] video[{video_idx}] profile={plan.requested_profile} "
                f"mode=post_vit budget={plan.budget_mode}/{plan.budget_correction} "
                f"frames={t} grid={dense_h}x{dense_w} dense={plan.dense_token_count} "
                f"planned={plan.planned_token_count} actual={plan.retained_token_count} "
                f"ratio={plan.actual_pre_llm_ratio:.4f} time={time.time() - st:.3f}s{extra}"
            )
        if debug_verify:
            verify_flashvid_lengths(
                plan=plan,
                placeholder_count=target,
                embedding_rows=out.shape[0],
                mrope_count=selected_pos.shape[0],
                deepstack_rows=[x.shape[0] for x in selected_deep],
                video_index=video_idx,
            )
        offset += dense_tokens
    return torch.cat(chunks, dim=0) if chunks else main.new_empty((0, main.shape[-1] + 4))


def _append_dense_image_positions(image_embeds: torch.Tensor, grid_thw: torch.Tensor, spatial_merge_size: int) -> tuple[torch.Tensor, ...]:
    chunks = []
    offset = 0
    for row in grid_thw:
        t, dense_h, dense_w, tokens_per_frame = _tokens_per_frame_for_row(row, spatial_merge_size)
        dense_tokens = int(t * tokens_per_frame)
        emb = image_embeds[offset : offset + dense_tokens]
        pos = _dense_local_mrope_positions(t, dense_h, dense_w, device=emb.device, dtype=emb.dtype)
        chunks.append(torch.cat([emb, pos], dim=1))
        offset += dense_tokens
    return tuple(chunks)


def _validate_patch_params(
    *,
    mode: str,
    profile: str,
    budget_mode: str,
    retention_ratio: float,
    expansion: float,
    alpha: float,
    token_selection_method: str,
    temporal_threshold: float,
    min_segment_num: int,
    budget_correction: str,
    dpc_k_max: int,
    deepstack_mode: str,
    cls_attention_query_chunk_size: int,
    temporal_match_chunk_size: int,
    pruning_layer: int,
    llm_retention_ratio: float | None,
) -> None:
    if mode not in {"post_vit", "hybrid"}:
        raise ValueError(f"mode must be post_vit/hybrid, got {mode!r}")
    if profile not in {"official_qwen3", "paper_adts_v2", "custom"}:
        raise ValueError(f"profile must be official_qwen3/paper_adts_v2/custom, got {profile!r}")
    if budget_mode not in {"direct", "paper_hybrid"}:
        raise ValueError(f"budget_mode must be direct/paper_hybrid, got {budget_mode!r}")
    if mode == "hybrid" and budget_mode != "paper_hybrid":
        raise ValueError("FlashVID hybrid requires budget_mode=paper_hybrid")
    if not math.isfinite(retention_ratio) or retention_ratio <= 0 or retention_ratio > 1:
        raise ValueError("retention_ratio must be in (0,1]")
    if not math.isfinite(expansion) or expansion <= 0:
        raise ValueError("expansion must be finite and > 0")
    if not math.isfinite(alpha) or alpha < 0 or alpha > 1:
        raise ValueError("alpha must be in [0,1]")
    if token_selection_method not in {"attn", "div", "attn_div", "attn_div_v2"}:
        raise ValueError(f"Unsupported token_selection_method={token_selection_method!r}")
    if not math.isfinite(temporal_threshold):
        raise ValueError("temporal_threshold must be finite")
    if int(min_segment_num) < 1:
        raise ValueError("min_segment_num must be >= 1")
    if budget_correction not in {"official_ceil", "exact_total"}:
        raise ValueError("budget_correction must be official_ceil/exact_total")
    if deepstack_mode not in {"official_gather", "hierarchical_mean"}:
        raise ValueError("deepstack_mode must be official_gather/hierarchical_mean")
    if int(dpc_k_max) < 1 or int(cls_attention_query_chunk_size) <= 0 or int(temporal_match_chunk_size) <= 0:
        raise ValueError("dpc_k_max and chunk sizes must be positive")
    if int(pruning_layer) < 0:
        raise ValueError("pruning_layer must be >= 0")
    if llm_retention_ratio is not None and (llm_retention_ratio <= 0 or llm_retention_ratio > 1):
        raise ValueError("llm_retention_ratio must be None or in (0,1]")


def apply_patch(
    mode: str = "post_vit",
    profile: str = "official_qwen3",
    budget_mode: str = "direct",
    retention_ratio: float = 0.20,
    expansion: float = 1.25,
    alpha: float = 0.70,
    token_selection_method: str | None = None,
    temporal_threshold: float = 0.80,
    do_segment: bool = True,
    segment_threshold: float = 0.90,
    min_segment_num: int = 4,
    complementary_segment: bool = True,
    budget_correction: str = "official_ceil",
    dpc_k_max: int = 7,
    deepstack_mode: str = "official_gather",
    cls_attention_query_chunk_size: int = 128,
    temporal_match_chunk_size: int = 256,
    pruning_layer: int = 28,
    llm_retention_ratio: float | None = None,
    debug_verify: bool = False,
) -> None:
    global _PATCHED
    mode = (mode or "none").strip().lower()
    if mode in {"none", "off", ""}:
        return
    profile = (profile or "official_qwen3").strip().lower()
    method = (token_selection_method or ("attn_div_v2" if profile == "paper_adts_v2" else "attn_div")).strip().lower()
    budget_mode = (budget_mode or "direct").strip().lower()
    budget_correction = (budget_correction or "official_ceil").strip().lower()
    deepstack_mode = (deepstack_mode or "official_gather").strip().lower()
    _validate_patch_params(
        mode=mode,
        profile=profile,
        budget_mode=budget_mode,
        retention_ratio=float(retention_ratio),
        expansion=float(expansion),
        alpha=float(alpha),
        token_selection_method=method,
        temporal_threshold=float(temporal_threshold),
        min_segment_num=int(min_segment_num),
        budget_correction=budget_correction,
        dpc_k_max=int(dpc_k_max),
        deepstack_mode=deepstack_mode,
        cls_attention_query_chunk_size=int(cls_attention_query_chunk_size),
        temporal_match_chunk_size=int(temporal_match_chunk_size),
        pruning_layer=int(pruning_layer),
        llm_retention_ratio=llm_retention_ratio,
    )
    if mode == "hybrid":
        raise NotImplementedError(
            "FlashVID hybrid is not enabled for local vLLM 0.12.0: layer-dependent "
            "sequence length and paged KV slot remapping are not exposed safely. "
            "Use --flashvid-mode post_vit."
        )
    if deepstack_mode != "official_gather":
        raise NotImplementedError(
            "FlashVID deepstack_mode=hierarchical_mean is not implemented in the "
            "vLLM runtime path. Use deepstack_mode=official_gather."
        )

    os.environ["QWEN3VL_FLASHVID_MODE"] = mode
    os.environ["QWEN3VL_FLASHVID_PROFILE"] = profile
    os.environ["QWEN3VL_FLASHVID_BUDGET_MODE"] = budget_mode
    os.environ["QWEN3VL_FLASHVID_RETENTION_RATIO"] = str(float(retention_ratio))
    os.environ["QWEN3VL_FLASHVID_EXPANSION"] = str(float(expansion))
    os.environ["QWEN3VL_FLASHVID_ALPHA"] = str(float(alpha))
    os.environ["QWEN3VL_FLASHVID_TOKEN_SELECTION_METHOD"] = method
    os.environ["QWEN3VL_FLASHVID_TEMPORAL_THRESHOLD"] = str(float(temporal_threshold))
    os.environ["QWEN3VL_FLASHVID_DO_SEGMENT"] = "1" if do_segment else "0"
    os.environ["QWEN3VL_FLASHVID_SEGMENT_THRESHOLD"] = str(float(segment_threshold))
    os.environ["QWEN3VL_FLASHVID_MIN_SEGMENT_NUM"] = str(int(min_segment_num))
    os.environ["QWEN3VL_FLASHVID_COMPLEMENTARY_SEGMENT"] = "1" if complementary_segment else "0"
    os.environ["QWEN3VL_FLASHVID_BUDGET_CORRECTION"] = budget_correction
    os.environ["QWEN3VL_FLASHVID_DPC_K_MAX"] = str(int(dpc_k_max))
    os.environ["QWEN3VL_FLASHVID_DEEPSTACK_MODE"] = deepstack_mode
    os.environ["QWEN3VL_FLASHVID_CLS_ATTN_CHUNK_SIZE"] = str(int(cls_attention_query_chunk_size))
    os.environ["QWEN3VL_FLASHVID_TEMPORAL_MATCH_CHUNK_SIZE"] = str(int(temporal_match_chunk_size))
    os.environ["QWEN3VL_FLASHVID_PRUNING_LAYER"] = str(int(pruning_layer))
    os.environ["QWEN3VL_FLASHVID_LLM_RETENTION_RATIO"] = "" if llm_retention_ratio is None else str(float(llm_retention_ratio))
    os.environ["QWEN3VL_FLASHVID_DEBUG_VERIFY"] = "1" if debug_verify else "0"

    import vllm.model_executor.models.qwen3_vl as qwen3_vl

    if _PATCHED:
        return
    required = _feature_report(qwen3_vl)
    for name in (
        "Qwen3VLMultiModalProcessor._get_prompt_updates",
        "Qwen3VLForConditionalGeneration._parse_and_validate_video_input",
        "Qwen3VLForConditionalGeneration._process_video_input",
        "Qwen3_VisionTransformer.forward",
        "iter_mm_grid_hw",
        "get_mrope_input_positions",
    ):
        if not required.get(name, False):
            raise AttributeError(f"[FlashVID-vLLM] Unsupported local Qwen3-VL API; missing {name}")

    model_cls = qwen3_vl.Qwen3VLForConditionalGeneration
    processor_cls = qwen3_vl.Qwen3VLMultiModalProcessor
    orig_get_prompt_updates = processor_cls._get_prompt_updates
    orig_parse_video_input = model_cls._parse_and_validate_video_input
    orig_process_image_input = model_cls._process_image_input
    orig_process_video_input = model_cls._process_video_input
    orig_iter_mm_grid_hw = model_cls.iter_mm_grid_hw

    def patched_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        if _enabled_mode() != "post_vit":
            return orig_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        hf_config = self.info.get_hf_config()
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        vision_end_token_id = hf_config.vision_end_token_id
        merge_length = image_processor.merge_size**2

        def get_image_replacement_qwen3vl(item_idx: int):
            grid = out_mm_kwargs["image"][item_idx]["image_grid_thw"].data
            return [hf_processor.image_token_id] * (int(grid.prod()) // merge_length)

        def get_video_replacement_qwen3vl(item_idx: int):
            grid = out_mm_kwargs["video"][item_idx]["video_grid_thw"].data
            count = _target_count_for_row(grid, int(image_processor.merge_size))
            return qwen3_vl.PromptUpdateDetails.select_token_id(
                [vision_start_token_id] + [video_token_id] * int(count) + [vision_end_token_id],
                video_token_id,
            )

        return [
            qwen3_vl.PromptReplacement(modality="image", target=hf_processor.image_token, replacement=get_image_replacement_qwen3vl),
            qwen3_vl.PromptReplacement(modality="video", target="<|vision_start|><|video_pad|><|vision_end|>", replacement=get_video_replacement_qwen3vl),
        ]

    def patched_vision_forward(self, x, grid_thw, flashvid_is_video=False):
        if _enabled_mode() != "post_vit" or not flashvid_is_video:
            return _ORIGINALS["Qwen3_VisionTransformer.forward"](self, x, grid_thw)
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
        hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid_thw_list)
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)
        cu_np = np.repeat(grid_np[:, 1] * grid_np[:, 2], grid_np[:, 0]).cumsum(axis=0, dtype=np.int32)
        cu_np = np.concatenate([np.zeros(1, dtype=np.int32), cu_np])
        cu_seqlens_cpu = torch.from_numpy(cu_np)
        hidden_states = hidden_states.unsqueeze(1)
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens_cpu)
        cu_seqlens = cu_seqlens_cpu.to(self.device, non_blocking=True)
        deepstack_feature_lists = []
        pre_attn = None
        for layer_num, blk in enumerate(self.blocks):
            if layer_num == len(self.blocks) - 1:
                pre_attn = _extract_last_block_incoming_attention(
                    blk,
                    blk.norm1(hidden_states),
                    cu_seqlens,
                    rotary_pos_emb_cos,
                    rotary_pos_emb_sin,
                    query_chunk_size=int(os.environ.get("QWEN3VL_FLASHVID_CLS_ATTN_CHUNK_SIZE", "128")),
                )
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
        if pre_attn is None:
            raise RuntimeError("[FlashVID-vLLM] failed to capture vision attention")
        cls_attention = _attention_group_mean(pre_attn, int(self.spatial_merge_unit))
        main = self.merger(hidden_states)
        if cls_attention.numel() != main.shape[0]:
            raise RuntimeError(f"[FlashVID-vLLM] attention/main row mismatch: attn={cls_attention.numel()} main={main.shape[0]}")
        return _compress_video_outputs(
            main,
            deepstack_feature_lists,
            cls_attention,
            grid_tensor,
            int(self.spatial_merge_size),
            _debug_verify(),
        )

    def patched_process_image_input(self, image_input):
        if _enabled_mode() != "post_vit":
            return orig_process_image_input(self, image_input)
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            if self.use_data_parallel:
                raise RuntimeError("FlashVID does not support mm_encoder_tp_mode=data yet.")
            image_embeds = _ORIGINALS["Qwen3_VisionTransformer.forward"](self.visual, pixel_values, grid_thw)
        return _append_dense_image_positions(image_embeds, grid_thw, int(self.visual.spatial_merge_size))

    def patched_parse_video_input(self, **kwargs):
        if _enabled_mode() != "post_vit":
            return orig_parse_video_input(self, **kwargs)
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)
        if pixel_values_videos is None and video_embeds is None:
            return None
        return {
            "type": "pixel_values_videos" if pixel_values_videos is not None else "video_embeds",
            "pixel_values_videos": pixel_values_videos,
            "video_embeds": video_embeds,
            "video_grid_thw": video_grid_thw,
            "second_per_grid_ts": second_per_grid_ts,
        }

    def patched_process_video_input(self, video_input):
        if _enabled_mode() != "post_vit":
            return orig_process_video_input(self, video_input)
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        if video_input["type"] == "video_embeds":
            raise NotImplementedError("[FlashVID-vLLM] raw pixel_values_videos are required to compute vision attention.")
        if self.use_data_parallel:
            raise RuntimeError("FlashVID does not support mm_encoder_tp_mode=data yet.")
        pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw, flashvid_is_video=True)
        return video_embeds.split(_selected_video_sizes(grid_thw, int(self.visual.spatial_merge_size)))

    def patched_iter_mm_grid_hw(self, input_tokens, mm_features):
        if _enabled_mode() != "post_vit":
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
                count = _target_count_for_row(grid, spatial_merge_size)
                offset = input_tokens.index(video_token_id, offset)
                yield offset, 1, int(count)
            else:
                raise ValueError(f"Unsupported modality: {mm_feature.modality}")

    def recompute_mrope_positions(self, input_ids, multimodal_embeddings, mrope_positions, num_computed_tokens):
        if _enabled_mode() != "post_vit":
            return multimodal_embeddings, mrope_positions, int((mrope_positions.max() + 1 - len(input_ids)).item())
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
        return tuple(mm_embeddings_out), positions, int(delta)

    _ORIGINALS["Qwen3VLMultiModalProcessor._get_prompt_updates"] = orig_get_prompt_updates
    _ORIGINALS["Qwen3_VisionTransformer.forward"] = qwen3_vl.Qwen3_VisionTransformer.forward
    _ORIGINALS["Qwen3VLForConditionalGeneration._parse_and_validate_video_input"] = orig_parse_video_input
    _ORIGINALS["Qwen3VLForConditionalGeneration._process_image_input"] = orig_process_image_input
    _ORIGINALS["Qwen3VLForConditionalGeneration._process_video_input"] = orig_process_video_input
    _ORIGINALS["Qwen3VLForConditionalGeneration.iter_mm_grid_hw"] = orig_iter_mm_grid_hw
    _ORIGINALS["Qwen3VLForConditionalGeneration.recompute_mrope_positions"] = getattr(model_cls, "recompute_mrope_positions", None)

    processor_cls._get_prompt_updates = patched_get_prompt_updates
    qwen3_vl.Qwen3_VisionTransformer.forward = patched_vision_forward
    model_cls._parse_and_validate_video_input = patched_parse_video_input
    model_cls._process_image_input = patched_process_image_input
    model_cls._process_video_input = patched_process_video_input
    model_cls.iter_mm_grid_hw = patched_iter_mm_grid_hw
    model_cls.recompute_mrope_positions = recompute_mrope_positions
    model_cls.supports_multimodal_pruning = True

    _PATCHED = True
    if _verbose():
        print(
            f"[FlashVID-vLLM] enabled mode={mode} profile={profile} budget_mode={budget_mode} "
            f"ratio={float(retention_ratio):.4f} alpha={float(alpha):.4f} method={method} "
            f"temporal_threshold={float(temporal_threshold):.4f} budget_correction={budget_correction}"
        )
        print(f"[FlashVID-vLLM] local_api={required}")
