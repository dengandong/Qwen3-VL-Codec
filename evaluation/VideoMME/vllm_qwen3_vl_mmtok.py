"""MMTok utilities and vLLM Qwen3-VL patch hooks.

This module implements MMTok-style multimodal maximum coverage for Qwen3-VL
video tokens.  The pure PyTorch implementation does not import vLLM.  The
runtime integration mirrors the local EchoPrune compact lifecycle: selected
embeddings carry sparse local M-RoPE coordinates in their last four columns,
and vLLM's multimodal-pruning recompute path builds the compact LLM sequence.

Official MMTok reference:
https://github.com/Ironieser/MMTok
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
class MMTokPlan:
    keep_flat_indices: torch.LongTensor
    keep_indices_per_frame: list[torch.LongTensor]
    num_tokens_per_frame: list[int]
    dense_token_count: int
    target_token_count: int
    retained_token_count: int
    num_frames: int
    grid_h: int
    grid_w: int
    query_token_count: int
    requested_profile: str
    effective_profile: str
    requested_greedy_mode: str
    effective_greedy_mode: str
    alpha: float
    requested_tv_temperature: float
    effective_tv_temperature: float
    requested_vv_temperature: float
    effective_vv_temperature: float
    temperature_mode: str
    vv_target_mode: str
    vv_target_indices: torch.LongTensor | None
    vv_target_weights: torch.Tensor | None
    vv_target_count: int
    objective_value: float | None
    text_coverage_value: float | None
    vision_coverage_value: float | None
    temperature_diagnostics: dict[str, float] | None


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-8)


def _validate_temperature(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return value


def _budget_from_inputs(
    total_tokens: int,
    *,
    retain_ratio: float | None,
    target_visual_tokens: int | None,
    budget_rounding: str = "floor",
) -> int:
    total = int(total_tokens)
    if total <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")
    if target_visual_tokens is not None:
        target = int(target_visual_tokens)
        if target <= 0:
            raise ValueError(f"target_visual_tokens must be positive, got {target_visual_tokens!r}")
    else:
        rr = 0.20 if retain_ratio is None else float(retain_ratio)
        if not math.isfinite(rr) or rr <= 0.0 or rr > 1.0:
            raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio!r}")
        mode = (budget_rounding or "floor").strip().lower()
        if mode == "floor":
            target = math.floor(total * rr)
        elif mode == "round":
            target = round(total * rr)
        else:
            raise ValueError(f"budget_rounding must be floor/round, got {budget_rounding!r}")
    return max(1, min(total, int(target)))


def _validate_plan_inputs(
    post_projector_tokens: torch.Tensor,
    pre_projector_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    *,
    num_frames: int,
    grid_h: int,
    grid_w: int,
    require_query: bool = True,
) -> tuple[int, int, int]:
    if post_projector_tokens.ndim != 2:
        raise ValueError(
            f"post_projector_tokens must be [N,D_lm], got {tuple(post_projector_tokens.shape)}"
        )
    if pre_projector_tokens.ndim != 2:
        raise ValueError(
            f"pre_projector_tokens must be [N,D_v], got {tuple(pre_projector_tokens.shape)}"
        )
    if query_tokens.ndim != 2:
        raise ValueError(f"query_tokens must be [M,D_lm], got {tuple(query_tokens.shape)}")
    n, d_lm = [int(x) for x in post_projector_tokens.shape]
    n_pre = int(pre_projector_tokens.shape[0])
    m, qd = [int(x) for x in query_tokens.shape]
    if n <= 0:
        raise ValueError("MMTok needs at least one visual token")
    if n_pre != n:
        raise ValueError(f"pre/post row mismatch: pre={n_pre}, post={n}")
    if require_query and m <= 0:
        raise ValueError("MMTok query is empty; question/options text is required")
    if m > 0 and qd != d_lm:
        raise ValueError(f"MMTok dimension mismatch: post D={d_lm}, query D={qd}")
    if int(num_frames) <= 0 or int(grid_h) <= 0 or int(grid_w) <= 0:
        raise ValueError(f"Invalid grid: T={num_frames}, H={grid_h}, W={grid_w}")
    if int(num_frames) * int(grid_h) * int(grid_w) != n:
        raise ValueError(
            f"T*H*W must equal N: T={num_frames}, H={grid_h}, W={grid_w}, N={n}"
        )
    return n, d_lm, int(pre_projector_tokens.shape[-1])


def build_stratified_video_target_coreset(
    *,
    num_frames: int,
    grid_h: int,
    grid_w: int,
    target_count: int,
    device: torch.device,
) -> tuple[torch.LongTensor, torch.Tensor]:
    """Build deterministic temporally balanced visual target rows.

    Candidates remain all dense video tokens.  The returned target representatives
    are only rows for visual coverage estimation.
    """
    t = int(num_frames)
    n_per_frame = int(grid_h) * int(grid_w)
    n = t * n_per_frame
    u = max(1, min(int(target_count), n))
    if u == n:
        return (
            torch.arange(n, device=device, dtype=torch.long),
            torch.full((n,), 1.0 / float(n), device=device, dtype=torch.float32),
        )

    temporal_bins = min(t, u)
    frame_bins = np.array_split(np.arange(t, dtype=np.int64), temporal_bins)
    bin_dense_counts = np.array([len(frames) * n_per_frame for frames in frame_bins], dtype=np.int64)
    raw = bin_dense_counts.astype(np.float64) / float(n) * float(u)
    counts = np.maximum(1, np.floor(raw).astype(np.int64))
    while counts.sum() > u:
        removable = np.where(counts > 1)[0]
        if removable.size == 0:
            break
        idx = int(removable[np.argmax(counts[removable] - raw[removable])])
        counts[idx] -= 1
    while counts.sum() < u:
        deficit = raw - counts
        idx = int(np.argmax(deficit))
        counts[idx] += 1

    selected: list[int] = []
    weights: list[float] = []
    for frames, count in zip(frame_bins, counts):
        start = int(frames[0]) * n_per_frame
        end = int(frames[-1] + 1) * n_per_frame
        dense_indices = np.arange(start, end, dtype=np.int64)
        buckets = np.array_split(dense_indices, int(count))
        for bucket in buckets:
            if bucket.size == 0:
                continue
            center = int(bucket[(bucket.size - 1) // 2])
            selected.append(center)
            weights.append(float(bucket.size) / float(n))

    if len(selected) != u:
        raise RuntimeError(f"MMTok coreset selected {len(selected)} rows, expected {u}")
    idx_t = torch.tensor(selected, device=device, dtype=torch.long)
    w_t = torch.tensor(weights, device=device, dtype=torch.float32)
    w_t = w_t / w_t.sum().clamp_min(1e-12)
    if torch.unique(idx_t).numel() != idx_t.numel():
        raise RuntimeError("MMTok coreset produced duplicate target indices")
    return idx_t, w_t


def compute_row_log_normalizers(
    target_features: torch.Tensor,
    candidate_features: torch.Tensor,
    *,
    temperature: float,
    candidate_chunk_size: int,
) -> torch.Tensor:
    """Streaming logsumexp over all candidate columns."""
    tau = _validate_temperature(temperature, "temperature")
    if int(candidate_chunk_size) <= 0:
        raise ValueError(f"candidate_chunk_size must be > 0, got {candidate_chunk_size}")
    if target_features.ndim != 2 or candidate_features.ndim != 2:
        raise ValueError("target_features and candidate_features must be rank-2")
    if target_features.shape[-1] != candidate_features.shape[-1]:
        raise ValueError(
            f"feature dim mismatch: target={target_features.shape[-1]}, "
            f"candidate={candidate_features.shape[-1]}"
        )
    logz = torch.full(
        (target_features.shape[0],),
        -torch.inf,
        device=target_features.device,
        dtype=torch.float32,
    )
    chunk = max(1, int(candidate_chunk_size))
    for start in range(0, candidate_features.shape[0], chunk):
        cand = candidate_features[start : start + chunk]
        logits = (target_features.float() @ cand.float().T) / tau
        logz = torch.logaddexp(logz, torch.logsumexp(logits, dim=1))
    return logz


def compute_calibrated_columns(
    target_features: torch.Tensor,
    candidate_features: torch.Tensor,
    candidate_indices: torch.LongTensor,
    *,
    temperature: float,
    row_weights: torch.Tensor,
    row_log_normalizers: torch.Tensor,
) -> torch.Tensor:
    """Return weighted global-softmax probabilities for candidate columns."""
    tau = _validate_temperature(temperature, "temperature")
    idx = candidate_indices.to(candidate_features.device, dtype=torch.long)
    if row_weights.shape[0] != target_features.shape[0]:
        raise ValueError("row_weights length must match target rows")
    cand = candidate_features.index_select(0, idx)
    logits = (target_features.float() @ cand.float().T) / tau
    probs = torch.exp(logits - row_log_normalizers.float().unsqueeze(1))
    return probs * row_weights.to(probs.device, dtype=torch.float32).unsqueeze(1)


def _probability_matrix(
    target_features: torch.Tensor,
    candidate_features: torch.Tensor,
    *,
    temperature: float,
    candidate_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logz = compute_row_log_normalizers(
        target_features,
        candidate_features,
        temperature=temperature,
        candidate_chunk_size=candidate_chunk_size,
    )
    all_idx = torch.arange(candidate_features.shape[0], device=candidate_features.device, dtype=torch.long)
    cols = []
    chunk = max(1, int(candidate_chunk_size))
    for start in range(0, all_idx.numel(), chunk):
        sub = all_idx[start : start + chunk]
        cols.append(
            compute_calibrated_columns(
                target_features,
                candidate_features,
                sub,
                temperature=temperature,
                row_weights=torch.ones(target_features.shape[0], device=target_features.device),
                row_log_normalizers=logz,
            )
        )
    return torch.cat(cols, dim=1), logz


def _probability_diagnostics(probs: torch.Tensor, prefix: str) -> dict[str, float]:
    if probs.numel() == 0:
        return {}
    p = probs.float().clamp_min(1e-45)
    entropy = -(p * p.log()).sum(dim=1)
    n = max(1, int(probs.shape[1]))
    topk = min(5, n)
    top_vals = torch.topk(probs.float(), k=topk, dim=1).values
    return {
        f"{prefix}_entropy_mean": float(entropy.mean().detach().cpu().item()),
        f"{prefix}_entropy_norm": float((entropy / math.log(max(n, 2))).mean().detach().cpu().item()),
        f"{prefix}_effective_support": float(torch.exp(entropy).mean().detach().cpu().item()),
        f"{prefix}_support_ratio": float((torch.exp(entropy) / float(n)).mean().detach().cpu().item()),
        f"{prefix}_top1": float(top_vals[:, 0].mean().detach().cpu().item()),
        f"{prefix}_top5_mass": float(top_vals.sum(dim=1).mean().detach().cpu().item()),
    }


def greedy_coverage_reference(combined_coverage: torch.Tensor, budget: int) -> torch.LongTensor:
    """Slow exact greedy reference for small matrices."""
    if combined_coverage.ndim != 2:
        raise ValueError("combined_coverage must be [rows, candidates]")
    rows, n = combined_coverage.shape
    b = max(1, min(int(budget), int(n)))
    best = torch.zeros(rows, dtype=combined_coverage.dtype, device=combined_coverage.device)
    selected: list[int] = []
    available = torch.ones(n, dtype=torch.bool, device=combined_coverage.device)
    for _ in range(b):
        gains = torch.empty(n, dtype=torch.float32, device=combined_coverage.device)
        for j in range(n):
            if not bool(available[j].item()):
                gains[j] = -torch.inf
            else:
                gains[j] = torch.clamp(combined_coverage[:, j] - best, min=0).sum()
        idx = int(torch.argmax(gains).item())
        selected.append(idx)
        available[idx] = False
        best = torch.maximum(best, combined_coverage[:, idx])
    return torch.tensor(selected, dtype=torch.long, device=combined_coverage.device)


def greedy_coverage_exact(
    combined_coverage: torch.Tensor,
    budget: int,
) -> tuple[torch.LongTensor, dict[str, Any]]:
    if combined_coverage.ndim != 2:
        raise ValueError("combined_coverage must be [rows, candidates]")
    rows, n = combined_coverage.shape
    b = max(1, min(int(budget), int(n)))
    best = torch.zeros(rows, dtype=torch.float32, device=combined_coverage.device)
    selected = torch.empty(b, dtype=torch.long, device=combined_coverage.device)
    score_mask = torch.zeros(n, dtype=torch.float32, device=combined_coverage.device)
    objective_trace: list[float] = []
    for i in range(b):
        delta = torch.clamp(combined_coverage.float() - best.unsqueeze(1), min=0).sum(dim=0)
        delta = delta + score_mask
        idx = torch.argmax(delta)
        selected[i] = idx
        best = torch.maximum(best, combined_coverage[:, idx].float())
        score_mask[idx] = -torch.inf
        objective_trace.append(float(best.sum().detach().cpu().item()))
    return selected, {"objective_trace": objective_trace}


def greedy_coverage_stochastic(
    combined_coverage: torch.Tensor,
    budget: int,
    *,
    epsilon: float,
    seed: int,
) -> tuple[torch.LongTensor, dict[str, Any]]:
    if combined_coverage.ndim != 2:
        raise ValueError("combined_coverage must be [rows, candidates]")
    eps = float(epsilon)
    if not math.isfinite(eps) or eps <= 0.0 or eps >= 1.0:
        raise ValueError(f"epsilon must be in (0,1), got {epsilon!r}")
    rows, n = combined_coverage.shape
    b = max(1, min(int(budget), int(n)))
    sample_size = max(1, math.ceil((float(n) / float(b)) * math.log(1.0 / eps)))
    gen = torch.Generator(device=combined_coverage.device)
    gen.manual_seed(int(seed))
    selected_mask = torch.zeros(n, dtype=torch.bool, device=combined_coverage.device)
    selected = torch.empty(b, dtype=torch.long, device=combined_coverage.device)
    best = torch.zeros(rows, dtype=torch.float32, device=combined_coverage.device)
    objective_trace: list[float] = []
    for i in range(b):
        remaining = torch.nonzero(~selected_mask, as_tuple=False).flatten()
        k = min(sample_size, int(remaining.numel()))
        perm = torch.randperm(remaining.numel(), generator=gen, device=remaining.device)
        sample = torch.sort(remaining.index_select(0, perm[:k])).values
        delta = torch.clamp(combined_coverage.index_select(1, sample).float() - best.unsqueeze(1), min=0).sum(dim=0)
        rel_idx = torch.argmax(delta)
        idx = sample[rel_idx]
        selected[i] = idx
        selected_mask[idx] = True
        best = torch.maximum(best, combined_coverage[:, idx].float())
        objective_trace.append(float(best.sum().detach().cpu().item()))
    return selected, {"sample_size": sample_size, "objective_trace": objective_trace}


def _build_combined_coverage(
    post_norm: torch.Tensor,
    pre_norm: torch.Tensor,
    query_norm: torch.Tensor,
    target_indices: torch.LongTensor,
    target_weights: torch.Tensor,
    *,
    alpha: float,
    tv_temperature: float,
    vv_temperature: float,
    candidate_chunk_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    m = int(query_norm.shape[0])
    n = int(post_norm.shape[0])
    text_probs, _ = _probability_matrix(
        query_norm,
        post_norm,
        temperature=tv_temperature,
        candidate_chunk_size=candidate_chunk_size,
    )
    text_rows = text_probs / float(m)
    visual_targets = pre_norm.index_select(0, target_indices.to(pre_norm.device))
    visual_probs, _ = _probability_matrix(
        visual_targets,
        pre_norm,
        temperature=vv_temperature,
        candidate_chunk_size=candidate_chunk_size,
    )
    visual_rows = visual_probs * (float(alpha) * target_weights.to(visual_probs.device).float()).unsqueeze(1)
    combined = torch.cat([text_rows, visual_rows], dim=0).contiguous()
    aux = {
        "text_probs": text_probs,
        "visual_probs": visual_probs,
        "text_peak": text_probs.max(dim=1).values.mean(),
        "visual_second_peak": _weighted_second_peak(visual_probs, target_weights.to(visual_probs.device)),
    }
    if combined.shape[1] != n:
        raise RuntimeError("combined coverage candidate dimension mismatch")
    return combined, aux


def _weighted_second_peak(probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if probs.shape[1] < 2:
        return torch.zeros((), dtype=torch.float32, device=probs.device)
    vals = torch.topk(probs.float(), k=2, dim=1).values[:, 1]
    return (vals * weights.to(vals.device).float()).sum()


def select_adaptive_vv_temperature(
    *,
    text_peak: torch.Tensor,
    pre_norm: torch.Tensor,
    target_indices: torch.LongTensor,
    target_weights: torch.Tensor,
    candidates: tuple[float, ...],
    candidate_chunk_size: int,
    fixed_fallback: float,
) -> tuple[float, dict[str, float]]:
    legal = tuple(float(x) for x in candidates if math.isfinite(float(x)) and float(x) > 0.0)
    if not legal:
        raise ValueError("adaptive_vv_candidates must contain at least one finite positive value")
    if pre_norm.shape[0] < 2:
        return float(fixed_fallback), {"adaptive_fallback_n_lt_2": 1.0}
    target = pre_norm.index_select(0, target_indices.to(pre_norm.device))
    text_peak_f = float(text_peak.detach().cpu().item())
    best_tau = legal[0]
    best_gap = float("inf")
    diagnostics: dict[str, float] = {"text_peak": text_peak_f}
    for tau in legal:
        probs, _ = _probability_matrix(
            target,
            pre_norm,
            temperature=tau,
            candidate_chunk_size=candidate_chunk_size,
        )
        second = float(_weighted_second_peak(probs, target_weights.to(probs.device)).detach().cpu().item())
        gap = abs(text_peak_f - second)
        diagnostics[f"adaptive_gap_tau_{tau:g}"] = gap
        diagnostics[f"adaptive_second_tau_{tau:g}"] = second
        if gap < best_gap - 1e-12 or (abs(gap - best_gap) <= 1e-12 and tau < best_tau):
            best_gap = gap
            best_tau = tau
    diagnostics["adaptive_selected_tau_v"] = float(best_tau)
    diagnostics["adaptive_selected_gap"] = float(best_gap)
    return float(best_tau), diagnostics


def _candidate_tuple(raw: str | Sequence[float] | None) -> tuple[float, ...]:
    if raw is None:
        return (0.05, 0.10, 0.15, 0.20)
    if isinstance(raw, str):
        return tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    return tuple(float(x) for x in raw)


def build_mmtok_plan(
    post_projector_tokens: torch.Tensor,
    pre_projector_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    *,
    num_frames: int,
    grid_h: int,
    grid_w: int,
    target_visual_tokens: int | None = None,
    retain_ratio: float | None = 0.20,
    budget_rounding: str = "floor",
    alpha: float = 0.5,
    tv_temperature: float = 0.01,
    vv_temperature: float = 0.20,
    temperature_mode: str = "fixed",
    adaptive_vv_candidates: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
    profile: str = "auto",
    vv_target_mode: str = "stratified_3d",
    vv_target_tokens: int = 1024,
    greedy_mode: str = "auto",
    stochastic_epsilon: float = 0.10,
    selection_seed: int = 3407,
    candidate_chunk_size: int = 512,
    target_chunk_size: int = 256,
    exact_max_tokens: int = 1024,
    debug: bool = False,
) -> MMTokPlan:
    del target_chunk_size
    requested_profile = (profile or "auto").strip().lower()
    requested_greedy = (greedy_mode or "auto").strip().lower()
    temperature_mode = (temperature_mode or "fixed").strip().lower()
    vv_target_mode = (vv_target_mode or "stratified_3d").strip().lower()
    budget_rounding = (budget_rounding or "floor").strip().lower()
    if requested_profile not in {"paper_exact", "video_scalable", "auto"}:
        raise ValueError(f"profile must be paper_exact/video_scalable/auto, got {profile!r}")
    if requested_greedy not in {"exact", "stochastic", "auto"}:
        raise ValueError(f"greedy_mode must be exact/stochastic/auto, got {greedy_mode!r}")
    if temperature_mode not in {"fixed", "adaptive_vv"}:
        raise ValueError(f"temperature_mode must be fixed/adaptive_vv, got {temperature_mode!r}")
    if vv_target_mode not in {"full", "stratified_3d"}:
        raise ValueError(f"vv_target_mode must be full/stratified_3d, got {vv_target_mode!r}")
    if not math.isfinite(float(alpha)) or float(alpha) < 0.0:
        raise ValueError(f"alpha must be finite and >= 0, got {alpha!r}")
    tau_t = _validate_temperature(tv_temperature, "tv_temperature")
    tau_v = _validate_temperature(vv_temperature, "vv_temperature")
    if int(vv_target_tokens) <= 0:
        raise ValueError(f"vv_target_tokens must be > 0, got {vv_target_tokens!r}")
    if int(candidate_chunk_size) <= 0:
        raise ValueError(f"candidate_chunk_size must be > 0, got {candidate_chunk_size!r}")
    if int(exact_max_tokens) <= 0:
        raise ValueError(f"exact_max_tokens must be > 0, got {exact_max_tokens!r}")

    n = int(post_projector_tokens.shape[0])
    budget = _budget_from_inputs(
        n,
        retain_ratio=retain_ratio,
        target_visual_tokens=target_visual_tokens,
        budget_rounding=budget_rounding,
    )
    require_query = budget < n
    _validate_plan_inputs(
        post_projector_tokens,
        pre_projector_tokens,
        query_tokens,
        num_frames=num_frames,
        grid_h=grid_h,
        grid_w=grid_w,
        require_query=require_query,
    )
    if budget == n:
        keep = torch.arange(n, device=post_projector_tokens.device, dtype=torch.long)
        return _make_plan(
            keep,
            n,
            budget,
            int(num_frames),
            int(grid_h),
            int(grid_w),
            int(query_tokens.shape[0]),
            requested_profile,
            "identity",
            requested_greedy,
            "identity",
            float(alpha),
            tau_t,
            tau_t,
            tau_v,
            tau_v,
            temperature_mode,
            "full",
            keep,
            torch.full((n,), 1.0 / float(n), device=keep.device),
            None,
            None,
            None,
            {"identity": 1.0},
        )

    effective_profile = requested_profile
    if effective_profile == "auto":
        effective_profile = "paper_exact" if n <= int(exact_max_tokens) else "video_scalable"
    if effective_profile == "paper_exact" and n > int(exact_max_tokens):
        raise RuntimeError(
            f"MMTok paper_exact requires N <= exact_max_tokens; N={n}, "
            f"exact_max_tokens={exact_max_tokens}. Use profile=video_scalable."
        )
    effective_vv_target_mode = "full" if effective_profile == "paper_exact" else vv_target_mode
    if effective_vv_target_mode == "full":
        target_indices = torch.arange(n, device=post_projector_tokens.device, dtype=torch.long)
        target_weights = torch.full((n,), 1.0 / float(n), device=post_projector_tokens.device)
    else:
        target_indices, target_weights = build_stratified_video_target_coreset(
            num_frames=int(num_frames),
            grid_h=int(grid_h),
            grid_w=int(grid_w),
            target_count=int(vv_target_tokens),
            device=post_projector_tokens.device,
        )

    post_norm = _normalize(post_projector_tokens)
    pre_norm = _normalize(pre_projector_tokens)
    query_norm = _normalize(query_tokens)

    text_probs, _ = _probability_matrix(
        query_norm,
        post_norm,
        temperature=tau_t,
        candidate_chunk_size=int(candidate_chunk_size),
    )
    text_peak = text_probs.max(dim=1).values.mean()
    diag: dict[str, float] = {}
    diag.update(_probability_diagnostics(text_probs, "tv"))
    diag["text_peak"] = float(text_peak.detach().cpu().item())

    effective_tau_v = tau_v
    if temperature_mode == "adaptive_vv":
        effective_tau_v, adaptive_diag = select_adaptive_vv_temperature(
            text_peak=text_peak,
            pre_norm=pre_norm,
            target_indices=target_indices,
            target_weights=target_weights,
            candidates=adaptive_vv_candidates,
            candidate_chunk_size=int(candidate_chunk_size),
            fixed_fallback=tau_v,
        )
        diag.update(adaptive_diag)

    combined, aux = _build_combined_coverage(
        post_norm,
        pre_norm,
        query_norm,
        target_indices,
        target_weights,
        alpha=float(alpha),
        tv_temperature=tau_t,
        vv_temperature=effective_tau_v,
        candidate_chunk_size=int(candidate_chunk_size),
    )
    if debug:
        diag.update(_probability_diagnostics(aux["visual_probs"], "vv"))
        diag["visual_second_peak"] = float(aux["visual_second_peak"].detach().cpu().item())

    effective_greedy = requested_greedy
    if effective_greedy == "auto":
        effective_greedy = "exact" if effective_profile == "paper_exact" else "stochastic"
    if effective_greedy == "exact":
        if n > int(exact_max_tokens):
            raise RuntimeError(
                f"MMTok exact greedy guard: N={n} exceeds exact_max_tokens={exact_max_tokens}"
            )
        selected_order, greedy_info = greedy_coverage_exact(combined, budget)
    else:
        selected_order, greedy_info = greedy_coverage_stochastic(
            combined,
            budget,
            epsilon=float(stochastic_epsilon),
            seed=int(selection_seed),
        )
    keep = torch.sort(selected_order.to(torch.long)).values

    covered = combined.index_select(1, keep).max(dim=1).values
    objective = float(covered.sum().detach().cpu().item())
    text_value = float(covered[: query_norm.shape[0]].sum().detach().cpu().item())
    visual_weighted = covered[query_norm.shape[0] :].sum()
    vision_value = (
        float((visual_weighted / float(alpha)).detach().cpu().item())
        if float(alpha) > 0.0
        else 0.0
    )
    if debug and "objective_trace" in greedy_info:
        trace = greedy_info["objective_trace"]
        diag["objective_final"] = float(trace[-1]) if trace else objective
    return _make_plan(
        keep,
        n,
        budget,
        int(num_frames),
        int(grid_h),
        int(grid_w),
        int(query_tokens.shape[0]),
        requested_profile,
        effective_profile,
        requested_greedy,
        effective_greedy,
        float(alpha),
        tau_t,
        tau_t,
        tau_v,
        effective_tau_v,
        temperature_mode,
        effective_vv_target_mode,
        target_indices,
        target_weights,
        objective,
        text_value,
        vision_value,
        diag if (debug or temperature_mode == "adaptive_vv") else None,
    )


def _make_plan(
    keep: torch.LongTensor,
    dense_count: int,
    target_count: int,
    num_frames: int,
    grid_h: int,
    grid_w: int,
    query_count: int,
    requested_profile: str,
    effective_profile: str,
    requested_greedy: str,
    effective_greedy: str,
    alpha: float,
    requested_tau_t: float,
    effective_tau_t: float,
    requested_tau_v: float,
    effective_tau_v: float,
    temperature_mode: str,
    vv_target_mode: str,
    target_indices: torch.LongTensor | None,
    target_weights: torch.Tensor | None,
    objective: float | None,
    text_value: float | None,
    vision_value: float | None,
    diagnostics: dict[str, float] | None,
) -> MMTokPlan:
    n_per_frame = int(grid_h) * int(grid_w)
    per_frame: list[torch.LongTensor] = []
    counts: list[int] = []
    for frame_idx in range(int(num_frames)):
        start = frame_idx * n_per_frame
        end = start + n_per_frame
        local = keep[(keep >= start) & (keep < end)] - start
        local = local.to(torch.long)
        per_frame.append(local)
        counts.append(int(local.numel()))
    return MMTokPlan(
        keep_flat_indices=keep.to(torch.long),
        keep_indices_per_frame=per_frame,
        num_tokens_per_frame=counts,
        dense_token_count=int(dense_count),
        target_token_count=int(target_count),
        retained_token_count=int(keep.numel()),
        num_frames=int(num_frames),
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        query_token_count=int(query_count),
        requested_profile=requested_profile,
        effective_profile=effective_profile,
        requested_greedy_mode=requested_greedy,
        effective_greedy_mode=effective_greedy,
        alpha=float(alpha),
        requested_tv_temperature=float(requested_tau_t),
        effective_tv_temperature=float(effective_tau_t),
        requested_vv_temperature=float(requested_tau_v),
        effective_vv_temperature=float(effective_tau_v),
        temperature_mode=temperature_mode,
        vv_target_mode=vv_target_mode,
        vv_target_indices=target_indices,
        vv_target_weights=target_weights,
        vv_target_count=0 if target_indices is None else int(target_indices.numel()),
        objective_value=objective,
        text_coverage_value=text_value,
        vision_coverage_value=vision_value,
        temperature_diagnostics=diagnostics,
    )


def apply_mmtok_plan(tensor: torch.Tensor, plan: MMTokPlan) -> torch.Tensor:
    if tensor.ndim < 1:
        raise ValueError(f"Expected tensor with token dimension, got {tuple(tensor.shape)}")
    if int(tensor.shape[0]) != int(plan.dense_token_count):
        raise ValueError(
            f"Tensor rows do not match MMTok plan: expected={plan.dense_token_count} "
            f"actual={tensor.shape[0]}"
        )
    return tensor.index_select(0, plan.keep_flat_indices.to(tensor.device))


def apply_mmtok_plan_to_deepstack(
    deepstack_features: Iterable[torch.Tensor],
    plan: MMTokPlan,
) -> list[torch.Tensor]:
    return [apply_mmtok_plan(layer, plan) for layer in deepstack_features]


def build_mmtok_query_text(
    annotation: Mapping[str, Any] | None = None,
    messages: Sequence[Mapping[str, Any]] | None = None,
    *,
    query_source: str = "question_options",
) -> str:
    query_source = (query_source or "question_options").strip().lower()
    if query_source not in {"question", "question_options", "user_text", "all_text"}:
        raise ValueError(f"Unsupported MMTok query_source={query_source!r}")

    def message_text(role_filter: set[str] | None) -> str:
        parts: list[str] = []
        for msg in messages or []:
            role = str(msg.get("role", "")).lower()
            if role_filter is not None and role not in role_filter:
                continue
            if role == "assistant":
                continue
            content = msg.get("content", "")
            items = content if isinstance(content, list) else [content]
            for item in items:
                if isinstance(item, str):
                    text = item
                elif isinstance(item, Mapping):
                    if item.get("type") in {"video", "image"}:
                        continue
                    text = str(item.get("text", ""))
                else:
                    text = str(item)
                text = text.strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()

    if query_source == "user_text":
        return message_text({"user"})
    if query_source == "all_text":
        return message_text(None)

    ann = dict(annotation or {})
    question = str(ann.get("question", ann.get("query", ""))).strip()
    if query_source == "question":
        return f"Question:\n{question}".strip() if question else message_text({"user"})
    raw_choices = (
        ann.get("choices")
        or ann.get("options")
        or ann.get("candidates")
        or ann.get("choice")
        or {}
    )
    choice_lines: list[str] = []
    if isinstance(raw_choices, Mapping):
        for key in sorted(raw_choices.keys(), key=lambda x: str(x)):
            choice_lines.append(f"({key}) {raw_choices[key]}")
    elif isinstance(raw_choices, Sequence) and not isinstance(raw_choices, (str, bytes)):
        for idx, value in enumerate(raw_choices):
            choice_lines.append(f"({chr(ord('A') + idx)}) {value}")

    parts = []
    if question:
        parts.append(f"Question:\n{question}")
    if choice_lines:
        parts.append("Options:\n" + "\n".join(choice_lines))
    query = "\n\n".join(parts).strip()
    return query if query else message_text({"user"})


def verify_mmtok_lengths(
    *,
    plan: MMTokPlan,
    placeholder_count: int,
    embedding_rows: int,
    mrope_count: int,
    deepstack_rows: Sequence[int] | None = None,
    dense_prompt_length: int | None = None,
    compact_prompt_length: int | None = None,
    video_index: int = 0,
) -> None:
    expected = int(plan.target_token_count)
    if (
        int(placeholder_count) != expected
        or int(embedding_rows) != expected
        or int(mrope_count) != expected
        or int(plan.retained_token_count) != expected
    ):
        raise RuntimeError(
            "[MMTok-vLLM] invariant failed for "
            f"video[{video_index}]: placeholders={placeholder_count}, embeds={embedding_rows}, "
            f"mrope={mrope_count}, retained={plan.retained_token_count}, expected={expected}, "
            f"dense={plan.dense_token_count}, query_tokens={plan.query_token_count}, "
            f"profile={plan.effective_profile}, greedy={plan.effective_greedy_mode}"
        )
    if deepstack_rows is not None:
        bad = [int(x) for x in deepstack_rows if int(x) != expected]
        if bad:
            raise RuntimeError(
                f"[MMTok-vLLM] deepstack invariant failed for video[{video_index}]: "
                f"rows={list(deepstack_rows)}, expected={expected}"
            )
    if dense_prompt_length is not None and compact_prompt_length is not None:
        expected_delta = int(plan.dense_token_count) - expected
        actual_delta = int(dense_prompt_length) - int(compact_prompt_length)
        if actual_delta != expected_delta:
            raise RuntimeError(
                f"[MMTok-vLLM] prompt delta invariant failed for video[{video_index}]: "
                f"delta={actual_delta}, expected_delta={expected_delta}"
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


def _target_count_for_row(row: Any, spatial_merge_size: int) -> int:
    t, _, _, tokens_per_frame = _tokens_per_frame_for_row(row, spatial_merge_size)
    return _budget_from_inputs(
        int(t * tokens_per_frame),
        retain_ratio=_retain_ratio(),
        target_visual_tokens=_target_visual_tokens(),
        budget_rounding=_budget_rounding(),
    )


def _selected_video_sizes(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> list[int]:
    return [_target_count_for_row(row, spatial_merge_size) for row in grid_thw]


def _dense_local_mrope_positions(
    t: int,
    dense_h: int,
    dense_w: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ti = torch.arange(t, device=device, dtype=torch.long).view(-1, 1).expand(-1, dense_h * dense_w).flatten()
    hi = torch.arange(dense_h, device=device, dtype=torch.long).view(1, -1, 1).expand(t, -1, dense_w).flatten()
    wi = torch.arange(dense_w, device=device, dtype=torch.long).view(1, 1, -1).expand(t, dense_h, -1).flatten()
    width = torch.full_like(ti, int(dense_w))
    return torch.stack([ti, hi, wi, width], dim=1).to(dtype=dtype)


def _pre_merger_group_mean(hidden_states: torch.Tensor, spatial_merge_unit: int) -> torch.Tensor:
    x = hidden_states.squeeze(1) if hidden_states.ndim == 3 and hidden_states.shape[1] == 1 else hidden_states
    if x.ndim != 2:
        raise ValueError(f"Expected pre-merger hidden states [seq,D], got {tuple(hidden_states.shape)}")
    unit = int(spatial_merge_unit)
    if x.shape[0] % unit != 0:
        raise ValueError(f"pre-merger rows {x.shape[0]} not divisible by spatial_merge_unit={unit}")
    return x.float().reshape(-1, unit, x.shape[-1]).mean(dim=1)


def _enabled_mode() -> str:
    return os.environ.get("QWEN3VL_MMTOK_MODE", "none").strip().lower()


def _retain_ratio() -> float:
    return float(os.environ.get("QWEN3VL_MMTOK_RETAIN_RATIO", "0.20"))


def _target_visual_tokens() -> int | None:
    raw = os.environ.get("QWEN3VL_MMTOK_TARGET_VISUAL_TOKENS", "").strip()
    if not raw or raw.lower() in {"none", "null", "0"}:
        return None
    return int(raw)


def _budget_rounding() -> str:
    return os.environ.get("QWEN3VL_MMTOK_BUDGET_ROUNDING", "floor").strip().lower()


def _profile() -> str:
    return os.environ.get("QWEN3VL_MMTOK_PROFILE", "auto").strip().lower()


def _alpha() -> float:
    return float(os.environ.get("QWEN3VL_MMTOK_ALPHA", "0.50"))


def _tv_temperature() -> float:
    return float(os.environ.get("QWEN3VL_MMTOK_TV_TEMPERATURE", "0.01"))


def _vv_temperature() -> float:
    return float(os.environ.get("QWEN3VL_MMTOK_VV_TEMPERATURE", "0.20"))


def _temperature_mode() -> str:
    return os.environ.get("QWEN3VL_MMTOK_TEMPERATURE_MODE", "fixed").strip().lower()


def _adaptive_vv_candidates() -> tuple[float, ...]:
    return _candidate_tuple(os.environ.get("QWEN3VL_MMTOK_ADAPTIVE_VV_CANDIDATES", "0.05,0.10,0.15,0.20"))


def _vv_target_mode() -> str:
    return os.environ.get("QWEN3VL_MMTOK_VV_TARGET_MODE", "stratified_3d").strip().lower()


def _vv_target_tokens() -> int:
    return int(os.environ.get("QWEN3VL_MMTOK_VV_TARGET_TOKENS", "1024"))


def _greedy_mode() -> str:
    return os.environ.get("QWEN3VL_MMTOK_GREEDY_MODE", "auto").strip().lower()


def _stochastic_epsilon() -> float:
    return float(os.environ.get("QWEN3VL_MMTOK_STOCHASTIC_EPSILON", "0.10"))


def _selection_seed() -> int:
    return int(os.environ.get("QWEN3VL_MMTOK_SELECTION_SEED", "3407"))


def _query_source() -> str:
    return os.environ.get("QWEN3VL_MMTOK_QUERY_SOURCE", "question_options").strip().lower()


def _candidate_chunk_size() -> int:
    return int(os.environ.get("QWEN3VL_MMTOK_CANDIDATE_CHUNK_SIZE", "512"))


def _target_chunk_size() -> int:
    return int(os.environ.get("QWEN3VL_MMTOK_TARGET_CHUNK_SIZE", "256"))


def _exact_max_tokens() -> int:
    return int(os.environ.get("QWEN3VL_MMTOK_EXACT_MAX_TOKENS", "1024"))


def _debug_verify() -> bool:
    return os.environ.get("QWEN3VL_MMTOK_DEBUG_VERIFY", "0") == "1"


def _verbose() -> bool:
    return os.environ.get("QWEN3VL_MMTOK_QUIET", "0") != "1"


def _feature_report(qwen3_vl: Any) -> dict[str, bool]:
    model_cls = getattr(qwen3_vl, "Qwen3VLForConditionalGeneration", None)
    processor_cls = getattr(qwen3_vl, "Qwen3VLMultiModalProcessor", None)
    visual_cls = getattr(qwen3_vl, "Qwen3_VisionTransformer", None)
    return {
        "SupportsMultiModalPruning": hasattr(qwen3_vl, "SupportsMultiModalPruning"),
        "Qwen3VLMultiModalProcessor._call_hf_processor": hasattr(processor_cls, "_call_hf_processor") if processor_cls else False,
        "Qwen3VLMultiModalProcessor._get_mm_fields_config": hasattr(processor_cls, "_get_mm_fields_config") if processor_cls else False,
        "Qwen3VLMultiModalProcessor._get_prompt_updates": hasattr(processor_cls, "_get_prompt_updates") if processor_cls else False,
        "Qwen3VLForConditionalGeneration._parse_and_validate_video_input": hasattr(model_cls, "_parse_and_validate_video_input") if model_cls else False,
        "Qwen3VLForConditionalGeneration._process_video_input": hasattr(model_cls, "_process_video_input") if model_cls else False,
        "Qwen3VLForConditionalGeneration.embed_input_ids": hasattr(model_cls, "embed_input_ids") if model_cls else False,
        "Qwen3VLForConditionalGeneration.get_language_model": hasattr(model_cls, "get_language_model") if model_cls else False,
        "Qwen3_VisionTransformer.forward": hasattr(visual_cls, "forward") if visual_cls else False,
        "iter_mm_grid_hw": hasattr(model_cls, "iter_mm_grid_hw") if model_cls else False,
        "get_mrope_input_positions": hasattr(model_cls, "get_mrope_input_positions") if model_cls else False,
    }


def _pop_mmtok_query_texts(mm_kwargs: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    clean = dict(mm_kwargs or {})
    raw = clean.pop("mmtok_query_texts", None)
    if raw is None:
        raw = clean.pop("mmtok_query_text", None)
    if raw is None:
        return clean, []
    if isinstance(raw, str):
        return clean, [raw]
    return clean, [str(x) for x in raw]


def _tokenize_query_texts(tokenizer: Any, query_texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[list[int]] = []
    for text in query_texts:
        if not str(text).strip():
            raise ValueError("[MMTok-vLLM] Empty mmtok_query_text")
        ids = tokenizer.encode(str(text), add_special_tokens=False)
        if not ids:
            raise ValueError("[MMTok-vLLM] Query text tokenized to zero tokens")
        rows.append([int(x) for x in ids])
    max_len = max(len(row) for row in rows)
    token_ids = torch.zeros((len(rows), max_len), dtype=torch.long)
    lengths = torch.zeros((len(rows),), dtype=torch.long)
    for idx, row in enumerate(rows):
        lengths[idx] = len(row)
        token_ids[idx, : len(row)] = torch.tensor(row, dtype=torch.long)
    return token_ids, lengths


def _query_rows(
    token_ids: torch.Tensor | None,
    lengths: torch.Tensor | None,
    n_videos: int,
    *,
    device: torch.device,
) -> list[torch.LongTensor]:
    if token_ids is None or lengths is None:
        raise ValueError("[MMTok-vLLM] Missing query token ids/lengths for video pruning")
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=device)
    lens = torch.as_tensor(lengths, dtype=torch.long, device=device).reshape(-1)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if lens.numel() == 1 and n_videos > 1:
        lens = lens.expand(n_videos)
    if ids.shape[0] == 1 and n_videos > 1:
        ids = ids.expand(n_videos, -1)
    if ids.shape[0] < n_videos or lens.numel() < n_videos:
        raise ValueError(
            f"[MMTok-vLLM] Query rows do not match videos: ids={tuple(ids.shape)} "
            f"lengths={tuple(lens.shape)} videos={n_videos}"
        )
    rows: list[torch.LongTensor] = []
    for idx in range(n_videos):
        length = int(lens[idx].item())
        if length <= 0:
            raise ValueError(f"[MMTok-vLLM] Query length for video[{idx}] is {length}")
        rows.append(ids[idx, :length].contiguous())
    return rows


def _embed_query_rows(model: Any, query_rows: Sequence[torch.LongTensor]) -> list[torch.Tensor]:
    embeds: list[torch.Tensor] = []
    language_model = model.get_language_model() if hasattr(model, "get_language_model") else getattr(model, "language_model", None)
    if language_model is None or not hasattr(language_model, "embed_input_ids"):
        raise AttributeError("[MMTok-vLLM] Could not find language_model.embed_input_ids")
    for row in query_rows:
        embeds.append(language_model.embed_input_ids(row))
    return embeds


def _compress_video_outputs(
    main: torch.Tensor,
    pre: torch.Tensor,
    deepstack: Sequence[torch.Tensor],
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    query_embeds: Sequence[torch.Tensor] | None,
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
        pre_chunk = pre[offset : offset + dense_tokens]
        deep_chunks = [layer[offset : offset + dense_tokens] for layer in deepstack]
        plan: MMTokPlan | None = None
        if query_embeds is None:
            selected = torch.arange(target, device=main.device, dtype=torch.long)
        else:
            query = query_embeds[video_idx]
            plan = build_mmtok_plan(
                main_chunk,
                pre_chunk,
                query,
                num_frames=t,
                grid_h=dense_h,
                grid_w=dense_w,
                target_visual_tokens=target,
                retain_ratio=None,
                budget_rounding=_budget_rounding(),
                alpha=_alpha(),
                tv_temperature=_tv_temperature(),
                vv_temperature=_vv_temperature(),
                temperature_mode=_temperature_mode(),
                adaptive_vv_candidates=_adaptive_vv_candidates(),
                profile=_profile(),
                vv_target_mode=_vv_target_mode(),
                vv_target_tokens=_vv_target_tokens(),
                greedy_mode=_greedy_mode(),
                stochastic_epsilon=_stochastic_epsilon(),
                selection_seed=_selection_seed(),
                candidate_chunk_size=_candidate_chunk_size(),
                target_chunk_size=_target_chunk_size(),
                exact_max_tokens=_exact_max_tokens(),
                debug=debug_verify,
            )
            selected = plan.keep_flat_indices.to(main.device)
        selected_main = main_chunk.index_select(0, selected)
        selected_deep = [layer.index_select(0, selected) for layer in deep_chunks]
        dense_pos = _dense_local_mrope_positions(t, dense_h, dense_w, device=main.device, dtype=main.dtype)
        selected_pos = dense_pos.index_select(0, selected)
        out = torch.cat([selected_main] + selected_deep + [selected_pos], dim=1)
        chunks.append(out)

        if _verbose() or debug_verify:
            if plan is None:
                print(
                    f"[MMTok-vLLM] profile_fallback video[{video_idx}] frames={t} "
                    f"grid={dense_h}x{dense_w} tokens={dense_tokens}->{target} "
                    f"reduction={1.0 - target / max(dense_tokens, 1):.2%}"
                )
            else:
                diag = plan.temperature_diagnostics or {}
                extra = ""
                if debug_verify:
                    extra = (
                        f" tau_t={plan.effective_tv_temperature:.4f} tau_v={plan.effective_vv_temperature:.4f}"
                        f" tv_entropy_norm={diag.get('tv_entropy_norm', float('nan')):.4f}"
                        f" vv_entropy_norm={diag.get('vv_entropy_norm', float('nan')):.4f}"
                        f" per_frame={plan.num_tokens_per_frame}"
                    )
                print(
                    f"[MMTok-vLLM] video[{video_idx}] profile={plan.effective_profile} "
                    f"greedy={plan.effective_greedy_mode} frames={t} grid={dense_h}x{dense_w} "
                    f"query_tokens={plan.query_token_count} vv_targets={plan.vv_target_count} "
                    f"tokens={plan.dense_token_count}->{plan.retained_token_count} "
                    f"reduction={1.0 - plan.retained_token_count / max(plan.dense_token_count, 1):.2%} "
                    f"time={time.time() - st:.3f}s{extra}"
                )
        if debug_verify:
            if plan is not None:
                verify_mmtok_lengths(
                    plan=plan,
                    placeholder_count=target,
                    embedding_rows=out.shape[0],
                    mrope_count=selected_pos.shape[0],
                    deepstack_rows=[x.shape[0] for x in selected_deep],
                    video_index=video_idx,
                )
            elif out.shape[0] != target or selected_pos.shape[0] != target:
                raise RuntimeError(
                    f"[MMTok-vLLM] profile fallback invariant failed for video[{video_idx}]: "
                    f"embeds={out.shape[0]}, mrope={selected_pos.shape[0]}, expected={target}"
                )
        offset += dense_tokens
    if not chunks:
        return main.new_empty((0, main.shape[-1] + 4))
    return torch.cat(chunks, dim=0)


def _append_dense_image_positions(
    image_embeds: torch.Tensor,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, ...]:
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
    profile: str,
    retain_ratio: float,
    target_visual_tokens: int | None,
    budget_rounding: str,
    alpha: float,
    tv_temperature: float,
    vv_temperature: float,
    temperature_mode: str,
    adaptive_vv_candidates: tuple[float, ...],
    vv_target_mode: str,
    vv_target_tokens: int,
    greedy_mode: str,
    stochastic_epsilon: float,
    query_source: str,
    candidate_chunk_size: int,
    target_chunk_size: int,
    exact_max_tokens: int,
) -> None:
    _budget_from_inputs(10, retain_ratio=retain_ratio, target_visual_tokens=target_visual_tokens, budget_rounding=budget_rounding)
    if profile not in {"paper_exact", "video_scalable", "auto"}:
        raise ValueError(f"profile must be paper_exact/video_scalable/auto, got {profile!r}")
    if budget_rounding not in {"floor", "round"}:
        raise ValueError(f"budget_rounding must be floor/round, got {budget_rounding!r}")
    if not math.isfinite(float(alpha)) or float(alpha) < 0.0:
        raise ValueError(f"alpha must be finite and >= 0, got {alpha!r}")
    _validate_temperature(tv_temperature, "tv_temperature")
    _validate_temperature(vv_temperature, "vv_temperature")
    if temperature_mode not in {"fixed", "adaptive_vv"}:
        raise ValueError(f"temperature_mode must be fixed/adaptive_vv, got {temperature_mode!r}")
    if not adaptive_vv_candidates or any((not math.isfinite(x) or x <= 0.0) for x in adaptive_vv_candidates):
        raise ValueError("adaptive_vv_candidates must all be finite and > 0")
    if vv_target_mode not in {"full", "stratified_3d"}:
        raise ValueError(f"vv_target_mode must be full/stratified_3d, got {vv_target_mode!r}")
    if int(vv_target_tokens) <= 0:
        raise ValueError(f"vv_target_tokens must be > 0, got {vv_target_tokens!r}")
    if greedy_mode not in {"exact", "stochastic", "auto"}:
        raise ValueError(f"greedy_mode must be exact/stochastic/auto, got {greedy_mode!r}")
    if not math.isfinite(float(stochastic_epsilon)) or not (0.0 < float(stochastic_epsilon) < 1.0):
        raise ValueError(f"stochastic_epsilon must be in (0,1), got {stochastic_epsilon!r}")
    if query_source not in {"question", "question_options", "user_text", "all_text"}:
        raise ValueError(f"query_source must be question/question_options/user_text/all_text, got {query_source!r}")
    if int(candidate_chunk_size) <= 0 or int(target_chunk_size) <= 0 or int(exact_max_tokens) <= 0:
        raise ValueError("chunk sizes and exact_max_tokens must be positive")


def apply_patch(
    mode: str = "post_vit",
    profile: str = "auto",
    retain_ratio: float = 0.20,
    target_visual_tokens: int | None = None,
    budget_rounding: str = "floor",
    alpha: float = 0.50,
    tv_temperature: float = 0.01,
    vv_temperature: float = 0.20,
    temperature_mode: str = "fixed",
    adaptive_vv_candidates: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
    vv_target_mode: str = "stratified_3d",
    vv_target_tokens: int = 1024,
    greedy_mode: str = "auto",
    stochastic_epsilon: float = 0.10,
    selection_seed: int = 3407,
    query_source: str = "question_options",
    candidate_chunk_size: int = 512,
    target_chunk_size: int = 256,
    exact_max_tokens: int = 1024,
    debug_verify: bool = False,
) -> None:
    global _PATCHED
    mode = (mode or "none").strip().lower()
    if mode in {"none", "off", ""}:
        return
    if mode != "post_vit":
        raise ValueError(f"Unsupported MMTok mode for vLLM: {mode}")
    profile = (profile or "auto").strip().lower()
    budget_rounding = (budget_rounding or "floor").strip().lower()
    temperature_mode = (temperature_mode or "fixed").strip().lower()
    vv_target_mode = (vv_target_mode or "stratified_3d").strip().lower()
    greedy_mode = (greedy_mode or "auto").strip().lower()
    query_source = (query_source or "question_options").strip().lower()
    adaptive_vv_candidates = tuple(float(x) for x in adaptive_vv_candidates)
    _validate_patch_params(
        profile=profile,
        retain_ratio=float(retain_ratio),
        target_visual_tokens=target_visual_tokens,
        budget_rounding=budget_rounding,
        alpha=float(alpha),
        tv_temperature=float(tv_temperature),
        vv_temperature=float(vv_temperature),
        temperature_mode=temperature_mode,
        adaptive_vv_candidates=adaptive_vv_candidates,
        vv_target_mode=vv_target_mode,
        vv_target_tokens=int(vv_target_tokens),
        greedy_mode=greedy_mode,
        stochastic_epsilon=float(stochastic_epsilon),
        query_source=query_source,
        candidate_chunk_size=int(candidate_chunk_size),
        target_chunk_size=int(target_chunk_size),
        exact_max_tokens=int(exact_max_tokens),
    )

    os.environ["QWEN3VL_MMTOK_MODE"] = mode
    os.environ["QWEN3VL_MMTOK_PROFILE"] = profile
    os.environ["QWEN3VL_MMTOK_RETAIN_RATIO"] = str(float(retain_ratio))
    os.environ["QWEN3VL_MMTOK_TARGET_VISUAL_TOKENS"] = "" if target_visual_tokens is None else str(int(target_visual_tokens))
    os.environ["QWEN3VL_MMTOK_BUDGET_ROUNDING"] = budget_rounding
    os.environ["QWEN3VL_MMTOK_ALPHA"] = str(float(alpha))
    os.environ["QWEN3VL_MMTOK_TV_TEMPERATURE"] = str(float(tv_temperature))
    os.environ["QWEN3VL_MMTOK_VV_TEMPERATURE"] = str(float(vv_temperature))
    os.environ["QWEN3VL_MMTOK_TEMPERATURE_MODE"] = temperature_mode
    os.environ["QWEN3VL_MMTOK_ADAPTIVE_VV_CANDIDATES"] = ",".join(str(x) for x in adaptive_vv_candidates)
    os.environ["QWEN3VL_MMTOK_VV_TARGET_MODE"] = vv_target_mode
    os.environ["QWEN3VL_MMTOK_VV_TARGET_TOKENS"] = str(int(vv_target_tokens))
    os.environ["QWEN3VL_MMTOK_GREEDY_MODE"] = greedy_mode
    os.environ["QWEN3VL_MMTOK_STOCHASTIC_EPSILON"] = str(float(stochastic_epsilon))
    os.environ["QWEN3VL_MMTOK_SELECTION_SEED"] = str(int(selection_seed))
    os.environ["QWEN3VL_MMTOK_QUERY_SOURCE"] = query_source
    os.environ["QWEN3VL_MMTOK_CANDIDATE_CHUNK_SIZE"] = str(int(candidate_chunk_size))
    os.environ["QWEN3VL_MMTOK_TARGET_CHUNK_SIZE"] = str(int(target_chunk_size))
    os.environ["QWEN3VL_MMTOK_EXACT_MAX_TOKENS"] = str(int(exact_max_tokens))
    os.environ["QWEN3VL_MMTOK_DEBUG_VERIFY"] = "1" if debug_verify else "0"

    import vllm.model_executor.models.qwen3_vl as qwen3_vl

    if _PATCHED:
        return

    required = _feature_report(qwen3_vl)
    for name in (
        "Qwen3VLMultiModalProcessor._call_hf_processor",
        "Qwen3VLMultiModalProcessor._get_mm_fields_config",
        "Qwen3VLMultiModalProcessor._get_prompt_updates",
        "Qwen3VLForConditionalGeneration._parse_and_validate_video_input",
        "Qwen3VLForConditionalGeneration._process_video_input",
        "Qwen3VLForConditionalGeneration.embed_input_ids",
        "Qwen3_VisionTransformer.forward",
        "iter_mm_grid_hw",
        "get_mrope_input_positions",
    ):
        if not required.get(name, False):
            raise AttributeError(f"[MMTok-vLLM] Unsupported local Qwen3-VL API; missing {name}")

    model_cls = qwen3_vl.Qwen3VLForConditionalGeneration
    processor_cls = qwen3_vl.Qwen3VLMultiModalProcessor

    orig_call_hf_processor = processor_cls._call_hf_processor
    orig_get_mm_fields_config = processor_cls._get_mm_fields_config
    orig_get_prompt_updates = processor_cls._get_prompt_updates
    orig_parse_video_input = model_cls._parse_and_validate_video_input
    orig_process_image_input = model_cls._process_image_input
    orig_process_video_input = model_cls._process_video_input
    orig_iter_mm_grid_hw = model_cls.iter_mm_grid_hw
    orig_get_mrope_input_positions = model_cls.get_mrope_input_positions

    def patched_call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        clean_mm_kwargs, query_texts = _pop_mmtok_query_texts(mm_kwargs)
        outputs = orig_call_hf_processor(self, prompt, mm_data, clean_mm_kwargs, tok_kwargs)
        if _enabled_mode() == "post_vit" and query_texts:
            grid = outputs.get("video_grid_thw")
            n_videos = int(grid.shape[0]) if isinstance(grid, torch.Tensor) and grid.ndim == 2 else 0
            if n_videos == 0:
                return outputs
            if len(query_texts) == 1 and n_videos > 1:
                query_texts = query_texts * n_videos
            if len(query_texts) != n_videos:
                raise ValueError(
                    f"[MMTok-vLLM] query/video count mismatch: queries={len(query_texts)} videos={n_videos}"
                )
            token_ids, lengths = _tokenize_query_texts(self.info.get_tokenizer(), query_texts)
            outputs["mmtok_query_token_ids"] = token_ids
            outputs["mmtok_query_lengths"] = lengths
        return outputs

    def patched_get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        clean_mm_kwargs, _ = _pop_mmtok_query_texts(hf_processor_mm_kwargs)
        config = dict(orig_get_mm_fields_config(self, hf_inputs, clean_mm_kwargs))
        if "mmtok_query_token_ids" in hf_inputs:
            config["mmtok_query_token_ids"] = qwen3_vl.MultiModalFieldConfig.batched("video")
        if "mmtok_query_lengths" in hf_inputs:
            config["mmtok_query_lengths"] = qwen3_vl.MultiModalFieldConfig.batched("video")
        return config

    def patched_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        if _enabled_mode() != "post_vit":
            return orig_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
        clean_mm_kwargs, _ = _pop_mmtok_query_texts(hf_processor_mm_kwargs)
        hf_processor = self.info.get_hf_processor(**clean_mm_kwargs)
        image_processor = self.info.get_image_processor(**clean_mm_kwargs)
        hf_config = self.info.get_hf_config()
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        vision_end_token_id = hf_config.vision_end_token_id
        merge_length = image_processor.merge_size**2

        def get_image_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            return [hf_processor.image_token_id] * (int(grid_thw.prod()) // merge_length)

        def get_video_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            count = _target_count_for_row(grid_thw, int(image_processor.merge_size))
            placeholder = [vision_start_token_id] + [video_token_id] * int(count) + [vision_end_token_id]
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

    def patched_vision_forward(self, x, grid_thw, mmtok_query_embeds=None):
        if _enabled_mode() != "post_vit":
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

        pre_features = _pre_merger_group_mean(hidden_states, int(self.spatial_merge_unit))
        main = self.merger(hidden_states)
        if pre_features.shape[0] != main.shape[0]:
            raise RuntimeError(
                f"[MMTok-vLLM] pre/post feature row mismatch: pre={pre_features.shape[0]} post={main.shape[0]}"
            )
        return _compress_video_outputs(
            main,
            pre_features,
            deepstack_feature_lists,
            grid_tensor,
            int(self.spatial_merge_size),
            mmtok_query_embeds,
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
                raise RuntimeError("MMTok vLLM patch does not support mm_encoder_tp_mode=data yet.")
            image_embeds = _ORIGINALS["Qwen3_VisionTransformer.forward"](self.visual, pixel_values, grid_thw)
            if image_embeds.shape[1] == self.visual.out_hidden_size + 4:
                return image_embeds.split((grid_thw.prod(-1) // self.visual.spatial_merge_size // self.visual.spatial_merge_size).tolist())
        return _append_dense_image_positions(image_embeds, grid_thw, int(self.visual.spatial_merge_size))

    def patched_parse_video_input(self, **kwargs):
        if _enabled_mode() != "post_vit":
            return orig_parse_video_input(self, **kwargs)
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)
        query_token_ids = kwargs.pop("mmtok_query_token_ids", None)
        query_lengths = kwargs.pop("mmtok_query_lengths", None)
        if pixel_values_videos is None and video_embeds is None:
            return None
        return {
            "type": "pixel_values_videos" if pixel_values_videos is not None else "video_embeds",
            "pixel_values_videos": pixel_values_videos,
            "video_embeds": video_embeds,
            "video_grid_thw": video_grid_thw,
            "second_per_grid_ts": second_per_grid_ts,
            "mmtok_query_token_ids": query_token_ids,
            "mmtok_query_lengths": query_lengths,
        }

    def patched_process_video_input(self, video_input):
        if _enabled_mode() != "post_vit":
            return orig_process_video_input(self, video_input)
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        if video_input["type"] == "video_embeds":
            raise NotImplementedError("[MMTok-vLLM] MMTok requires raw pixel_values_videos to compute query-dependent plans.")
        pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
        if self.use_data_parallel:
            raise RuntimeError("MMTok vLLM patch does not support mm_encoder_tp_mode=data yet.")
        query_token_ids = video_input.get("mmtok_query_token_ids")
        query_lengths = video_input.get("mmtok_query_lengths")
        if query_token_ids is None or query_lengths is None:
            query_embeds = None
        else:
            query_rows = _query_rows(query_token_ids, query_lengths, int(grid_thw.shape[0]), device=pixel_values_videos.device)
            query_embeds = _embed_query_rows(self, query_rows)
        video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw, mmtok_query_embeds=query_embeds)
        sizes = _selected_video_sizes(grid_thw, int(self.visual.spatial_merge_size))
        return video_embeds.split(sizes)

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

    _ORIGINALS["Qwen3VLMultiModalProcessor._call_hf_processor"] = orig_call_hf_processor
    _ORIGINALS["Qwen3VLMultiModalProcessor._get_mm_fields_config"] = orig_get_mm_fields_config
    _ORIGINALS["Qwen3VLMultiModalProcessor._get_prompt_updates"] = orig_get_prompt_updates
    _ORIGINALS["Qwen3_VisionTransformer.forward"] = qwen3_vl.Qwen3_VisionTransformer.forward
    _ORIGINALS["Qwen3VLForConditionalGeneration._parse_and_validate_video_input"] = orig_parse_video_input
    _ORIGINALS["Qwen3VLForConditionalGeneration._process_image_input"] = orig_process_image_input
    _ORIGINALS["Qwen3VLForConditionalGeneration._process_video_input"] = orig_process_video_input
    _ORIGINALS["Qwen3VLForConditionalGeneration.iter_mm_grid_hw"] = orig_iter_mm_grid_hw
    _ORIGINALS["Qwen3VLForConditionalGeneration.get_mrope_input_positions"] = orig_get_mrope_input_positions
    _ORIGINALS["Qwen3VLForConditionalGeneration.recompute_mrope_positions"] = getattr(model_cls, "recompute_mrope_positions", None)
    _ORIGINALS["Qwen3VLForConditionalGeneration.supports_multimodal_pruning"] = getattr(model_cls, "supports_multimodal_pruning", None)

    processor_cls._call_hf_processor = patched_call_hf_processor
    processor_cls._get_mm_fields_config = patched_get_mm_fields_config
    processor_cls._get_prompt_updates = patched_get_prompt_updates
    qwen3_vl.Qwen3_VisionTransformer.forward = patched_vision_forward
    model_cls._parse_and_validate_video_input = patched_parse_video_input
    model_cls._process_image_input = patched_process_image_input
    model_cls._process_video_input = patched_process_video_input
    model_cls.iter_mm_grid_hw = patched_iter_mm_grid_hw
    model_cls.get_mrope_input_positions = orig_get_mrope_input_positions
    model_cls.recompute_mrope_positions = recompute_mrope_positions
    model_cls.supports_multimodal_pruning = True

    _PATCHED = True
    if _verbose():
        print(
            f"[MMTok-vLLM] enabled mode={mode} profile={profile} retain_ratio={float(retain_ratio):.4f} "
            f"target_visual_tokens={target_visual_tokens} budget_rounding={budget_rounding} "
            f"alpha={float(alpha):.4f} tau_t={float(tv_temperature):.4f} tau_v={float(vv_temperature):.4f} "
            f"temperature_mode={temperature_mode} vv_target={vv_target_mode}/{int(vv_target_tokens)} "
            f"greedy={greedy_mode} epsilon={float(stochastic_epsilon):.4f} seed={int(selection_seed)} "
            f"query_source={query_source}"
        )
        print(f"[MMTok-vLLM] local_api={required}")
