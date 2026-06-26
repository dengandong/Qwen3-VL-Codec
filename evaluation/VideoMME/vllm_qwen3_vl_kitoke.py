"""KiToke utilities and vLLM Qwen3-VL patch hooks.

This module implements Kernel-based Interval-aware Token Compression for
Qwen3-VL video tokens.  The pure PyTorch implementation at the top is
CPU-testable and does not import vLLM.  The runtime patch follows the local
FlashVID/MMTok compact lifecycle: compressed video embeddings carry selected
representative sparse local M-RoPE coordinates in their last four columns, and
vLLM's multimodal-pruning recompute path builds the compact LLM sequence.

KiToke paper: https://arxiv.org/pdf/2604.03414

No public official code was available during implementation.  Engineering
fallbacks are marked where the paper leaves details unspecified.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


_PATCHED = False
_ORIGINALS: dict[str, Any] = {}


@dataclass(frozen=True)
class KiTokeAlgorithmConfig:
    retain_ratio: float = 0.10
    target_visual_tokens: int | None = None
    kernel_alpha: float = 800.0
    selection_method: str = "pivotal"
    selection_seed: int = 3407
    seed_policy: str = "stable_video"
    pivotal_pairing: str = "random_rounds"
    diff_threshold: float = 110.0
    delta_threshold: float = 70.0
    relative_delta_threshold: float = 0.40
    edge_policy: str = "absolute_only"
    empty_interval_policy: str = "repair_swap"
    merge_mode: str = "weighted"
    deepstack_mode: str = "same_weighted_merge"
    kernel_row_chunk_size: int = 256
    kernel_col_chunk_size: int = 512
    frame_match_chunk_size: int = 256
    interval_match_chunk_size: int = 256
    debug: bool = False


@dataclass
class KiTokeInterval:
    start_frame: int
    end_frame_exclusive: int
    start_flat_index: int
    end_flat_index_exclusive: int
    selected_flat_indices: torch.LongTensor
    unselected_flat_indices: torch.LongTensor
    empty_before_repair: bool = False
    repaired: bool = False


@dataclass(frozen=True)
class KiTokePlan:
    selected_flat_indices: torch.LongTensor
    num_tokens_per_frame: list[int]
    output_frame_indices: torch.LongTensor
    dense_to_output_slot: torch.LongTensor
    normalized_merge_weights: torch.Tensor
    dense_token_count: int
    target_token_count: int
    retained_token_count: int
    num_frames: int
    grid_h: int
    grid_w: int
    tokens_per_frame: int
    density: torch.Tensor | None
    diversity: torch.Tensor | None
    inclusion_probabilities: torch.Tensor | None
    diff_pos: torch.Tensor | None
    diff_match: torch.Tensor | None
    transition_diff: torch.Tensor | None
    transition_delta: torch.Tensor | None
    transition_delta_pct: torch.Tensor | None
    boundary_transition_indices: list[int]
    intervals: list[KiTokeInterval]
    selection_method: str
    pivotal_pairing: str
    effective_seed: int
    empty_interval_policy: str
    repaired_empty_interval_count: int
    promoted_indices: list[int]
    demoted_indices: list[int]
    merge_mode: str
    deepstack_mode: str
    timing_stats: dict[str, float] | None


@dataclass(frozen=True)
class KiTokeCompressionResult:
    compressed_main_embeddings: torch.Tensor
    plan: KiTokePlan


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype != torch.float32 else x


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    bad = ~torch.isfinite(tensor)
    if bool(bad.any().item()):
        idx = int(bad.nonzero(as_tuple=False)[0, 0].item())
        raise ValueError(f"{name} contains non-finite value at flat index {idx}")


def _budget_from_inputs(total_tokens: int, cfg: KiTokeAlgorithmConfig) -> int:
    n = int(total_tokens)
    if n <= 0:
        raise ValueError(f"dense token count must be positive, got {total_tokens}")
    if cfg.target_visual_tokens is not None:
        b = int(cfg.target_visual_tokens)
        if b <= 0:
            raise ValueError(f"target_visual_tokens must be positive, got {cfg.target_visual_tokens}")
    else:
        ratio = float(cfg.retain_ratio)
        if not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
            raise ValueError(f"retain_ratio must be in (0,1], got {cfg.retain_ratio!r}")
        b = math.floor(n * ratio)
    return max(1, min(int(b), n))


def _stable_topk_largest(scores: torch.Tensor, k: int) -> torch.LongTensor:
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got {tuple(scores.shape)}")
    k = int(k)
    if k <= 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    if k >= scores.numel():
        return torch.arange(scores.numel(), device=scores.device, dtype=torch.long)
    clean = torch.nan_to_num(scores.float(), nan=-torch.inf)
    order = torch.argsort(-clean, stable=True)
    return order[:k].long()


def compute_kernel_density_reference(
    features: torch.Tensor,
    *,
    kernel_alpha: float,
) -> torch.Tensor:
    """Full Gaussian KDE reference: exp(-||xi-xj||^2 / alpha), diagonal kept."""
    if features.ndim != 2:
        raise ValueError(f"features must be [N,C], got {tuple(features.shape)}")
    if not math.isfinite(float(kernel_alpha)) or float(kernel_alpha) <= 0:
        raise ValueError(f"kernel_alpha must be finite and > 0, got {kernel_alpha!r}")
    x = _as_float_tensor(features)
    _check_finite("features", x)
    d2 = torch.cdist(x, x, p=2.0).pow(2)
    return torch.exp(-d2 / float(kernel_alpha)).sum(dim=1).float()


def compute_kernel_density_chunked(
    features: torch.Tensor,
    *,
    kernel_alpha: float,
    row_chunk_size: int,
    col_chunk_size: int,
) -> torch.Tensor:
    """Memory-bounded exact global KDE over all video tokens."""
    if features.ndim != 2:
        raise ValueError(f"features must be [N,C], got {tuple(features.shape)}")
    if int(row_chunk_size) <= 0 or int(col_chunk_size) <= 0:
        raise ValueError("row_chunk_size and col_chunk_size must be positive")
    if not math.isfinite(float(kernel_alpha)) or float(kernel_alpha) <= 0:
        raise ValueError(f"kernel_alpha must be finite and > 0, got {kernel_alpha!r}")
    x = _as_float_tensor(features)
    _check_finite("features", x)
    n = x.shape[0]
    norms = (x * x).sum(dim=-1)
    density = torch.zeros((n,), dtype=torch.float32, device=x.device)
    for rs in range(0, n, int(row_chunk_size)):
        re = min(n, rs + int(row_chunk_size))
        xr = x[rs:re]
        nr = norms[rs:re]
        acc = torch.zeros((re - rs,), dtype=torch.float32, device=x.device)
        for cs in range(0, n, int(col_chunk_size)):
            ce = min(n, cs + int(col_chunk_size))
            xc = x[cs:ce]
            d2 = nr[:, None] + norms[cs:ce][None, :] - 2.0 * (xr @ xc.T)
            acc += torch.exp(-d2.clamp_min(0.0) / float(kernel_alpha)).sum(dim=1)
        density[rs:re] = acc
    if bool((density <= 0).any().item()) or bool((~torch.isfinite(density)).any().item()):
        raise RuntimeError("kernel density must be finite and strictly positive")
    return density


def diversity_to_inclusion_probabilities(
    diversity: torch.Tensor,
    budget: int,
    *,
    tolerance: float = 1e-10,
) -> torch.Tensor:
    """ENGINEERING FALLBACK: cap-and-redistribute pi with sum(pi)=budget."""
    if diversity.ndim != 1:
        raise ValueError(f"diversity must be 1-D, got {tuple(diversity.shape)}")
    n = diversity.numel()
    b = int(budget)
    if b < 0 or b > n:
        raise ValueError(f"budget must be in [0,N], got B={b}, N={n}")
    weights = diversity.to(dtype=torch.float64)
    bad = (~torch.isfinite(weights)) | (weights <= 0)
    if bool(bad.any().item()):
        idx = int(bad.nonzero(as_tuple=False)[0, 0].item())
        raise ValueError(f"diversity weights must be finite and >0; bad index={idx}")
    if b == 0:
        return torch.zeros_like(weights)
    if b == n:
        return torch.ones_like(weights)
    pi = torch.zeros_like(weights)
    active = torch.ones((n,), dtype=torch.bool, device=weights.device)
    remaining = float(b)
    while True:
        active_idx = active.nonzero(as_tuple=False).flatten()
        if active_idx.numel() == 0:
            break
        w_active = weights.index_select(0, active_idx)
        w_sum = float(w_active.sum().item())
        if not math.isfinite(w_sum) or w_sum <= 0:
            raise ValueError("sum of active diversity weights must be positive")
        provisional = remaining * w_active / w_sum
        saturated_mask = provisional >= (1.0 - float(tolerance))
        if not bool(saturated_mask.any().item()):
            pi.index_copy_(0, active_idx, provisional)
            break
        saturated = active_idx.index_select(0, saturated_mask.nonzero(as_tuple=False).flatten())
        pi.index_fill_(0, saturated, 1.0)
        active.index_fill_(0, saturated, False)
        remaining -= float(saturated.numel())
        if remaining <= float(tolerance):
            break
    total = pi.sum()
    if not torch.isclose(total, torch.tensor(float(b), dtype=total.dtype, device=total.device), atol=1e-7, rtol=1e-7):
        # Numerical correction only: distribute the tiny residual over fractional entries.
        residual = float(b) - float(total.item())
        frac = ((pi > 0) & (pi < 1)).nonzero(as_tuple=False).flatten()
        if frac.numel() > 0:
            pi[frac[0]] = (pi[frac[0]] + residual).clamp(0.0, 1.0)
    if bool((pi < -1e-8).any().item()) or bool((pi > 1 + 1e-8).any().item()):
        raise RuntimeError("inclusion probabilities outside [0,1]")
    return pi.clamp(0.0, 1.0)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        gen = torch.Generator(device=device)
    except TypeError:
        gen = torch.Generator()
    gen.manual_seed(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)
    return gen


def pivotal_sample_fixed_size(
    inclusion_probabilities: torch.Tensor,
    *,
    generator: torch.Generator,
    pairing: str,
    tolerance: float = 1e-7,
) -> torch.LongTensor:
    """Fixed-size pivotal sampling with local RNG and deterministic final snap."""
    if inclusion_probabilities.ndim != 1:
        raise ValueError("inclusion_probabilities must be 1-D")
    pairing = (pairing or "random_rounds").lower()
    if pairing not in {"random_rounds", "sequential"}:
        raise ValueError(f"Unsupported pivotal_pairing={pairing!r}")
    pi = inclusion_probabilities.to(dtype=torch.float64).clone()
    bad = (~torch.isfinite(pi)) | (pi < -tolerance) | (pi > 1.0 + tolerance)
    if bool(bad.any().item()):
        idx = int(bad.nonzero(as_tuple=False)[0, 0].item())
        raise ValueError(f"invalid inclusion probability at index={idx}")
    budget = int(round(float(pi.sum().item())))
    n = pi.numel()
    if budget <= 0:
        return torch.empty((0,), dtype=torch.long, device=pi.device)
    if budget >= n:
        return torch.arange(n, dtype=torch.long, device=pi.device)

    while True:
        pi[pi <= tolerance] = 0.0
        pi[pi >= 1.0 - tolerance] = 1.0
        frac = ((pi > tolerance) & (pi < 1.0 - tolerance)).nonzero(as_tuple=False).flatten()
        if frac.numel() <= 1:
            break
        if pairing == "random_rounds":
            perm = torch.randperm(frac.numel(), generator=generator, device=frac.device)
            frac = frac.index_select(0, perm)
        if frac.numel() % 2 == 1:
            frac = frac[:-1]
        i = frac[0::2]
        j = frac[1::2]
        a = pi.index_select(0, i)
        b = pi.index_select(0, j)
        s = a + b
        u = torch.rand((i.numel(),), generator=generator, device=pi.device, dtype=torch.float64)

        lt = s < (1.0 - tolerance)
        gt = s > (1.0 + tolerance)
        eq = ~(lt | gt)

        new_i = torch.empty_like(a)
        new_j = torch.empty_like(b)

        if bool(lt.any().item()):
            prob_i = a[lt] / s[lt].clamp_min(tolerance)
            choose_i = u[lt] < prob_i
            new_i[lt] = torch.where(choose_i, s[lt], torch.zeros_like(s[lt]))
            new_j[lt] = torch.where(choose_i, torch.zeros_like(s[lt]), s[lt])
        if bool(gt.any().item()):
            denom = (2.0 - s[gt]).clamp_min(tolerance)
            prob_i_one = (1.0 - b[gt]) / denom
            choose_i = u[gt] < prob_i_one
            new_i[gt] = torch.where(choose_i, torch.ones_like(s[gt]), s[gt] - 1.0)
            new_j[gt] = torch.where(choose_i, s[gt] - 1.0, torch.ones_like(s[gt]))
        if bool(eq.any().item()):
            choose_i = u[eq] < a[eq].clamp(0.0, 1.0)
            new_i[eq] = torch.where(choose_i, torch.ones_like(s[eq]), torch.zeros_like(s[eq]))
            new_j[eq] = torch.where(choose_i, torch.zeros_like(s[eq]), torch.ones_like(s[eq]))

        pi.index_copy_(0, i, new_i.clamp(0.0, 1.0))
        pi.index_copy_(0, j, new_j.clamp(0.0, 1.0))

    pi[pi <= tolerance] = 0.0
    pi[pi >= 1.0 - tolerance] = 1.0
    selected = (pi > 0.5).nonzero(as_tuple=False).flatten()
    # ENGINEERING FALLBACK: final numerical snap for one residual fractional token.
    if selected.numel() != budget:
        order = torch.argsort(-pi, stable=True)
        selected = torch.sort(order[:budget].long()).values
    else:
        selected = torch.sort(selected.long()).values
    if selected.numel() != budget or torch.unique(selected).numel() != budget:
        raise RuntimeError(f"pivotal sampling failed exact-size invariant: expected={budget} actual={selected.numel()}")
    return selected


def _stable_effective_seed(
    *,
    base_seed: int,
    seed_policy: str,
    identity: str | bytes | None,
    cfg: KiTokeAlgorithmConfig,
) -> int:
    if seed_policy == "fixed":
        return int(base_seed) & 0x7FFF_FFFF
    payload = identity if identity is not None else b"kitoke-no-identity"
    if isinstance(payload, str):
        payload_b = payload.encode("utf-8")
    else:
        payload_b = bytes(payload)
    h = hashlib.blake2b(digest_size=8)
    h.update(payload_b)
    h.update(str(int(base_seed)).encode())
    h.update(str(float(cfg.retain_ratio)).encode())
    h.update(str(cfg.target_visual_tokens).encode())
    h.update(str(float(cfg.kernel_alpha)).encode())
    h.update(cfg.selection_method.encode())
    h.update(cfg.pivotal_pairing.encode())
    return int.from_bytes(h.digest(), "little") & 0x7FFF_FFFF


def select_global_tokens(
    diversity: torch.Tensor,
    *,
    budget: int,
    method: str,
    seed: int,
    pivotal_pairing: str,
) -> tuple[torch.LongTensor, torch.Tensor | None]:
    method = (method or "pivotal").lower()
    b = int(budget)
    if b <= 0 or b > diversity.numel():
        raise ValueError(f"budget must be in [1,N], got B={b}, N={diversity.numel()}")
    if method == "topk":
        return torch.sort(_stable_topk_largest(diversity, b)).values, None
    gen = _make_generator(diversity.device, int(seed))
    if method == "multinomial":
        selected = torch.multinomial(diversity.float(), num_samples=b, replacement=False, generator=gen)
        return torch.sort(selected.long()).values, None
    if method == "pivotal":
        pi = diversity_to_inclusion_probabilities(diversity, b)
        return pivotal_sample_fixed_size(pi, generator=gen, pairing=pivotal_pairing), pi.float()
    raise ValueError(f"Unsupported selection_method={method!r}")


def compute_transition_metrics_reference(video_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference transition metrics for [T,M,C] using dense cdist."""
    if video_features.ndim != 3:
        raise ValueError(f"video_features must be [T,M,C], got {tuple(video_features.shape)}")
    x = _as_float_tensor(video_features)
    _check_finite("video_features", x)
    t = x.shape[0]
    if t <= 1:
        empty = torch.empty((0,), dtype=torch.float32, device=x.device)
        return empty, empty
    diff_pos = torch.linalg.vector_norm(x[1:] - x[:-1], dim=-1).mean(dim=1)
    diff_match = []
    for idx in range(1, t):
        dist = torch.cdist(x[idx - 1], x[idx], p=2.0)
        diff_match.append(dist.min(dim=1).values.mean())
    return diff_pos.float(), torch.stack(diff_match).float()


def compute_transition_metrics_chunked(
    video_features: torch.Tensor,
    *,
    match_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact previous-to-current full-spatial transition metrics."""
    if video_features.ndim != 3:
        raise ValueError(f"video_features must be [T,M,C], got {tuple(video_features.shape)}")
    if int(match_chunk_size) <= 0:
        raise ValueError("match_chunk_size must be positive")
    x = _as_float_tensor(video_features)
    _check_finite("video_features", x)
    t, m, _ = x.shape
    if t <= 1:
        empty = torch.empty((0,), dtype=torch.float32, device=x.device)
        return empty, empty
    diff_pos = torch.linalg.vector_norm(x[1:] - x[:-1], dim=-1).mean(dim=1)
    diff_match = torch.empty((t - 1,), dtype=torch.float32, device=x.device)
    for frame in range(1, t):
        prev = x[frame - 1]
        curr = x[frame]
        curr_norm = (curr * curr).sum(dim=-1)
        mins = []
        for start in range(0, m, int(match_chunk_size)):
            prev_chunk = prev[start : start + int(match_chunk_size)]
            d2 = (prev_chunk * prev_chunk).sum(dim=-1)[:, None] + curr_norm[None, :] - 2.0 * (prev_chunk @ curr.T)
            mins.append(d2.clamp_min(0.0).min(dim=1).values.sqrt())
        diff_match[frame - 1] = torch.cat(mins).mean()
    return diff_pos.float(), diff_match.float()


def construct_temporal_intervals(
    diff_pos: torch.Tensor,
    diff_match: torch.Tensor,
    *,
    tokens_per_frame: int,
    diff_threshold: float,
    delta_threshold: float,
    relative_delta_threshold: float,
    edge_policy: str,
) -> tuple[list[KiTokeInterval], list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build intervals. Boundary transition index t starts a new interval at frame t."""
    if diff_pos.shape != diff_match.shape:
        raise ValueError("diff_pos and diff_match must have the same shape")
    if int(tokens_per_frame) <= 0:
        raise ValueError("tokens_per_frame must be positive")
    edge_policy = (edge_policy or "absolute_only").lower()
    if edge_policy not in {"absolute_only", "one_sided"}:
        raise ValueError(f"Unsupported edge_policy={edge_policy!r}")
    diff = (diff_pos.float() + diff_match.float()).float()
    transitions = diff.numel()
    if transitions == 0:
        return [
            KiTokeInterval(
                0,
                1,
                0,
                int(tokens_per_frame),
                torch.empty((0,), dtype=torch.long, device=diff.device),
                torch.empty((0,), dtype=torch.long, device=diff.device),
            )
        ], [], diff, diff.new_empty((0,)), diff.new_empty((0,))
    delta = torch.full_like(diff, -torch.inf)
    delta_pct = torch.full_like(diff, -torch.inf)
    eps = 1e-6
    for i in range(transitions):
        vals = []
        pct_vals = []
        if i > 0:
            vals.append(diff[i] - diff[i - 1])
            pct_vals.append((diff[i] - diff[i - 1]) / diff[i - 1].clamp_min(eps))
        if i + 1 < transitions:
            vals.append(diff[i] - diff[i + 1])
            pct_vals.append((diff[i] - diff[i + 1]) / diff[i + 1].clamp_min(eps))
        if vals and (edge_policy == "one_sided" or (i > 0 and i + 1 < transitions)):
            delta[i] = torch.stack(vals).max()
            delta_pct[i] = torch.stack(pct_vals).max()
    absolute = diff > float(diff_threshold)
    local = (delta > float(delta_threshold)) & (delta_pct > float(relative_delta_threshold))
    boundary_mask = absolute | local
    boundaries = [int(i + 1) for i in boundary_mask.nonzero(as_tuple=False).flatten().tolist()]
    frame_count = transitions + 1
    starts = [0] + boundaries
    ends = boundaries + [frame_count]
    intervals = [
        KiTokeInterval(
            int(s),
            int(e),
            int(s * tokens_per_frame),
            int(e * tokens_per_frame),
            torch.empty((0,), dtype=torch.long, device=diff.device),
            torch.empty((0,), dtype=torch.long, device=diff.device),
        )
        for s, e in zip(starts, ends)
        if e > s
    ]
    return intervals, boundaries, diff, delta, delta_pct


def _interval_members(
    intervals: Sequence[KiTokeInterval],
    selected_indices: torch.LongTensor,
    *,
    total_tokens: int,
) -> list[KiTokeInterval]:
    device = selected_indices.device
    selected_mask = torch.zeros((total_tokens,), dtype=torch.bool, device=device)
    selected_mask.index_fill_(0, selected_indices.to(device=device, dtype=torch.long), True)
    out: list[KiTokeInterval] = []
    for interval in intervals:
        idx = torch.arange(interval.start_flat_index, interval.end_flat_index_exclusive, device=device, dtype=torch.long)
        sel = idx[selected_mask.index_select(0, idx)]
        unsel = idx[~selected_mask.index_select(0, idx)]
        out.append(
            KiTokeInterval(
                interval.start_frame,
                interval.end_frame_exclusive,
                interval.start_flat_index,
                interval.end_flat_index_exclusive,
                sel,
                unsel,
                empty_before_repair=interval.empty_before_repair,
                repaired=interval.repaired,
            )
        )
    return out


def _coarsen_intervals_to_budget(
    intervals: list[KiTokeInterval],
    boundaries: list[int],
    transition_diff: torch.Tensor,
    transition_delta: torch.Tensor,
    transition_delta_pct: torch.Tensor,
    *,
    budget: int,
    tokens_per_frame: int,
    diff_threshold: float,
    delta_threshold: float,
    relative_delta_threshold: float,
) -> tuple[list[KiTokeInterval], list[int]]:
    if len(intervals) <= int(budget):
        return intervals, boundaries
    keep = list(boundaries)
    while len(keep) + 1 > int(budget):
        strengths = []
        for b in keep:
            i = int(b) - 1
            abs_strength = float(transition_diff[i].item()) / max(float(diff_threshold), 1e-6)
            local_strength = min(
                float(transition_delta[i].item()) / max(float(delta_threshold), 1e-6),
                float(transition_delta_pct[i].item()) / max(float(relative_delta_threshold), 1e-6),
            )
            strengths.append((max(abs_strength, local_strength), b))
        _, drop = min(strengths, key=lambda x: (x[0], -x[1]))
        keep.remove(drop)
    starts = [0] + keep
    frame_count = intervals[-1].end_frame_exclusive
    ends = keep + [frame_count]
    new_intervals = [
        KiTokeInterval(
            int(s),
            int(e),
            int(s * tokens_per_frame),
            int(e * tokens_per_frame),
            torch.empty((0,), dtype=torch.long, device=transition_diff.device),
            torch.empty((0,), dtype=torch.long, device=transition_diff.device),
        )
        for s, e in zip(starts, ends)
    ]
    return new_intervals, keep


def repair_empty_intervals(
    selected_indices: torch.LongTensor,
    diversity: torch.Tensor,
    intervals: Sequence[KiTokeInterval],
    *,
    budget: int,
    policy: str,
) -> tuple[torch.LongTensor, list[KiTokeInterval], list[int], list[int], int]:
    """ENGINEERING FALLBACK: optional swap repair for intervals with no representative."""
    policy = (policy or "repair_swap").lower()
    if policy not in {"paper_strict", "repair_swap", "coarsen_then_repair"}:
        raise ValueError(f"Unsupported empty_interval_policy={policy!r}")
    n = diversity.numel()
    selected = torch.zeros((n,), dtype=torch.bool, device=diversity.device)
    selected.index_fill_(0, selected_indices.to(diversity.device), True)
    intervals_now = _interval_members(intervals, selected_indices.to(diversity.device), total_tokens=n)
    empty = [idx for idx, interval in enumerate(intervals_now) if interval.selected_flat_indices.numel() == 0]
    if not empty:
        return torch.sort(selected_indices.long()).values, intervals_now, [], [], 0
    if policy == "paper_strict":
        first = intervals_now[empty[0]]
        raise RuntimeError(
            "KiToke interval has no selected representative: "
            f"interval={empty[0]} range=[{first.start_frame},{first.end_frame_exclusive}) "
            f"dense_tokens={first.end_flat_index_exclusive - first.start_flat_index} B={budget} "
            f"intervals={len(intervals_now)}"
        )
    if int(budget) < len(intervals_now):
        raise RuntimeError(
            f"KiToke repair_swap cannot cover intervals: B={budget} interval_count={len(intervals_now)}. "
            "Use empty_interval_policy=coarsen_then_repair explicitly for this ablation."
        )
    promoted: list[int] = []
    demoted: list[int] = []
    repaired_count = 0
    for empty_idx in empty:
        intervals_now = _interval_members([i for i in intervals_now], selected.nonzero(as_tuple=False).flatten(), total_tokens=n)
        interval = intervals_now[empty_idx]
        candidates = torch.arange(interval.start_flat_index, interval.end_flat_index_exclusive, device=diversity.device, dtype=torch.long)
        unselected = candidates[~selected.index_select(0, candidates)]
        if unselected.numel() == 0:
            raise RuntimeError(f"KiToke empty interval {empty_idx} has no promote candidate")
        promote_order = torch.argsort(-diversity.index_select(0, unselected).float(), stable=True)
        promote = int(unselected[promote_order[0]].item())
        donor_idx = None
        donor_selected = None
        for idx, donor in enumerate(intervals_now):
            if donor.selected_flat_indices.numel() >= 2:
                donor_idx = idx
                donor_selected = donor.selected_flat_indices
                break
        if donor_idx is None or donor_selected is None:
            raise RuntimeError("KiToke repair_swap internal error: no donor interval with >=2 selected tokens")
        donor_div = diversity.index_select(0, donor_selected).float()
        # Demote smallest diversity; tie demotes later dense index.
        order = torch.argsort(donor_div, stable=True)
        min_val = donor_div[order[0]]
        tied = donor_selected[donor_div == min_val]
        demote = int(tied.max().item())
        selected[promote] = True
        selected[demote] = False
        promoted.append(promote)
        demoted.append(demote)
        repaired_count += 1
    out = selected.nonzero(as_tuple=False).flatten().long()
    if out.numel() != int(budget):
        raise RuntimeError(f"KiToke repair changed budget: expected={budget} actual={out.numel()}")
    intervals_repaired = _interval_members(intervals, out, total_tokens=n)
    for interval in intervals_repaired:
        interval.empty_before_repair = interval.selected_flat_indices.numel() == 0
        interval.repaired = repaired_count > 0
    if any(i.selected_flat_indices.numel() == 0 for i in intervals_repaired):
        raise RuntimeError("KiToke repair failed to give every interval a representative")
    return torch.sort(out).values, intervals_repaired, promoted, demoted, repaired_count


def build_interval_assignments(
    features: torch.Tensor,
    selected_indices: torch.LongTensor,
    intervals: Sequence[KiTokeInterval],
    *,
    match_chunk_size: int,
) -> torch.LongTensor:
    """Assign each dense token to a selected token in the same interval by cosine."""
    if features.ndim != 2:
        raise ValueError(f"features must be [N,C], got {tuple(features.shape)}")
    if int(match_chunk_size) <= 0:
        raise ValueError("match_chunk_size must be positive")
    x = F.normalize(features.float(), dim=-1, eps=1e-6)
    selected = torch.sort(selected_indices.to(device=features.device, dtype=torch.long)).values
    n = features.shape[0]
    dense_to_slot = torch.full((n,), -1, dtype=torch.long, device=features.device)
    selected_to_slot = torch.full((n,), -1, dtype=torch.long, device=features.device)
    selected_to_slot.index_copy_(0, selected, torch.arange(selected.numel(), device=features.device, dtype=torch.long))
    for interval_idx, interval in enumerate(intervals):
        reps = interval.selected_flat_indices.to(device=features.device, dtype=torch.long)
        if reps.numel() == 0:
            raise RuntimeError(f"KiToke interval {interval_idx} has no representative for assignment")
        reps = torch.sort(reps).values
        dense_to_slot.index_copy_(0, reps, selected_to_slot.index_select(0, reps))
        idx = torch.arange(interval.start_flat_index, interval.end_flat_index_exclusive, device=features.device, dtype=torch.long)
        is_rep = torch.zeros((idx.numel(),), dtype=torch.bool, device=features.device)
        rel = reps - int(interval.start_flat_index)
        is_rep.index_fill_(0, rel, True)
        unselected = idx[~is_rep]
        if unselected.numel() == 0:
            continue
        rep_features = x.index_select(0, reps)
        for start in range(0, unselected.numel(), int(match_chunk_size)):
            cur = unselected[start : start + int(match_chunk_size)]
            sim = x.index_select(0, cur) @ rep_features.T
            best = sim.argmax(dim=1)
            dense_to_slot.index_copy_(0, cur, selected_to_slot.index_select(0, reps.index_select(0, best)))
    if bool((dense_to_slot < 0).any().item()):
        bad = int((dense_to_slot < 0).nonzero(as_tuple=False)[0, 0].item())
        raise RuntimeError(f"KiToke assignment missed dense token index={bad}")
    return dense_to_slot


def build_merge_weights(
    diversity: torch.Tensor,
    dense_to_output_slot: torch.LongTensor,
    *,
    output_count: int,
    merge_mode: str,
    selected_indices: torch.LongTensor,
) -> torch.Tensor:
    mode = (merge_mode or "weighted").lower()
    if mode not in {"weighted", "uniform", "none"}:
        raise ValueError(f"Unsupported merge_mode={mode!r}")
    n = diversity.numel()
    if dense_to_output_slot.shape != (n,):
        raise ValueError("dense_to_output_slot shape mismatch")
    if mode == "weighted":
        base = diversity.float()
    else:
        base = torch.ones_like(diversity, dtype=torch.float32)
    if mode == "none":
        keep = torch.zeros((n,), dtype=torch.bool, device=diversity.device)
        keep.index_fill_(0, selected_indices.to(diversity.device, dtype=torch.long), True)
        base = torch.where(keep, torch.ones_like(base), torch.zeros_like(base))
    denom = torch.zeros((int(output_count),), dtype=torch.float32, device=diversity.device)
    denom.index_add_(0, dense_to_output_slot.long(), base)
    if bool((denom <= 0).any().item()):
        bad = int((denom <= 0).nonzero(as_tuple=False)[0, 0].item())
        raise RuntimeError(f"KiToke merge output slot has non-positive denominator: slot={bad}")
    return base / denom.index_select(0, dense_to_output_slot.long())


def apply_kitoke_plan(
    tensor: torch.Tensor,
    plan: KiTokePlan,
    *,
    mode: str,
) -> torch.Tensor:
    """Apply plan to main/deepstack tensors. `mode=gather` uses representatives."""
    if tensor.ndim != 2:
        raise ValueError(f"tensor must be [N,C], got {tuple(tensor.shape)}")
    if tensor.shape[0] != plan.dense_token_count:
        raise ValueError(f"tensor rows={tensor.shape[0]} dense_count={plan.dense_token_count}")
    if mode in {"representative_gather", "gather", "none"}:
        return tensor.index_select(0, plan.selected_flat_indices.to(tensor.device))
    if mode not in {"same_weighted_merge", "weighted", "uniform"}:
        raise ValueError(f"Unsupported apply mode={mode!r}")
    slots = plan.dense_to_output_slot.to(tensor.device)
    weights = plan.normalized_merge_weights.to(device=tensor.device, dtype=torch.float32)
    out = torch.zeros((plan.retained_token_count, tensor.shape[1]), dtype=torch.float32, device=tensor.device)
    out.index_add_(0, slots, tensor.float() * weights[:, None])
    return out.to(dtype=tensor.dtype)


def _feature_seed_identity(features: torch.Tensor, *, video_index: int, t: int, h: int, w: int) -> bytes:
    # ENGINEERING FALLBACK: vLLM worker does not expose original video path here.
    x = features.detach().float()
    sample = torch.stack(
        [
            x.reshape(-1)[0],
            x.reshape(-1)[x.numel() // 2],
            x.reshape(-1)[-1],
            x.mean(),
            x.std(unbiased=False),
        ]
    ).cpu().numpy().astype(np.float32).tobytes()
    return f"video={video_index};shape={t}x{h}x{w}x{features.shape[-1]}".encode() + sample


def compress_video_kitoke(
    video_features: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
    config: KiTokeAlgorithmConfig,
    stable_video_identity: str | bytes | None = None,
) -> KiTokeCompressionResult:
    """Compress one video `[T, H*W, C]` with KiToke."""
    if video_features.ndim != 3:
        raise ValueError(f"video_features must be [T,M,C], got {tuple(video_features.shape)}")
    t, m, c = video_features.shape
    if int(grid_h) * int(grid_w) != int(m):
        raise ValueError(f"grid_h*grid_w mismatch: {grid_h}*{grid_w} != {m}")
    n = int(t * m)
    flat = video_features.reshape(n, c)
    _check_finite("video_features", flat.float())
    budget = _budget_from_inputs(n, config)
    selected_all = torch.arange(n, dtype=torch.long, device=flat.device)
    timing: dict[str, float] = {}
    if budget == n:
        dense_to_slot = selected_all.clone()
        weights = torch.ones((n,), dtype=torch.float32, device=flat.device)
        intervals = [
            KiTokeInterval(
                0,
                int(t),
                0,
                n,
                selected_all,
                torch.empty((0,), dtype=torch.long, device=flat.device),
            )
        ]
        plan = KiTokePlan(
            selected_flat_indices=selected_all,
            num_tokens_per_frame=[int(m)] * int(t),
            output_frame_indices=torch.arange(t, device=flat.device, dtype=torch.long).repeat_interleave(m),
            dense_to_output_slot=dense_to_slot,
            normalized_merge_weights=weights,
            dense_token_count=n,
            target_token_count=budget,
            retained_token_count=n,
            num_frames=int(t),
            grid_h=int(grid_h),
            grid_w=int(grid_w),
            tokens_per_frame=int(m),
            density=None,
            diversity=None,
            inclusion_probabilities=None,
            diff_pos=None,
            diff_match=None,
            transition_diff=None,
            transition_delta=None,
            transition_delta_pct=None,
            boundary_transition_indices=[],
            intervals=intervals,
            selection_method=config.selection_method,
            pivotal_pairing=config.pivotal_pairing,
            effective_seed=int(config.selection_seed),
            empty_interval_policy=config.empty_interval_policy,
            repaired_empty_interval_count=0,
            promoted_indices=[],
            demoted_indices=[],
            merge_mode=config.merge_mode,
            deepstack_mode=config.deepstack_mode,
            timing_stats=timing,
        )
        return KiTokeCompressionResult(flat.to(video_features.dtype), plan)

    total_start = time.time()
    st = time.time()
    density = compute_kernel_density_chunked(
        flat,
        kernel_alpha=float(config.kernel_alpha),
        row_chunk_size=int(config.kernel_row_chunk_size),
        col_chunk_size=int(config.kernel_col_chunk_size),
    )
    diversity = 1.0 / density.clamp_min(1e-12)
    timing["kernel_density_ms"] = (time.time() - st) * 1000.0

    st = time.time()
    seed = _stable_effective_seed(
        base_seed=int(config.selection_seed),
        seed_policy=config.seed_policy,
        identity=stable_video_identity,
        cfg=config,
    )
    selected, pi = select_global_tokens(
        diversity,
        budget=budget,
        method=config.selection_method,
        seed=seed,
        pivotal_pairing=config.pivotal_pairing,
    )
    timing["sampling_ms"] = (time.time() - st) * 1000.0

    st = time.time()
    diff_pos, diff_match = compute_transition_metrics_chunked(
        video_features,
        match_chunk_size=int(config.frame_match_chunk_size),
    )
    intervals, boundaries, transition_diff, delta, delta_pct = construct_temporal_intervals(
        diff_pos,
        diff_match,
        tokens_per_frame=int(m),
        diff_threshold=float(config.diff_threshold),
        delta_threshold=float(config.delta_threshold),
        relative_delta_threshold=float(config.relative_delta_threshold),
        edge_policy=config.edge_policy,
    )
    if config.empty_interval_policy == "coarsen_then_repair" and len(intervals) > budget:
        intervals, boundaries = _coarsen_intervals_to_budget(
            intervals,
            boundaries,
            transition_diff,
            delta,
            delta_pct,
            budget=budget,
            tokens_per_frame=int(m),
            diff_threshold=float(config.diff_threshold),
            delta_threshold=float(config.delta_threshold),
            relative_delta_threshold=float(config.relative_delta_threshold),
        )
    timing["transition_interval_ms"] = (time.time() - st) * 1000.0

    st = time.time()
    selected, intervals, promoted, demoted, repaired_count = repair_empty_intervals(
        selected,
        diversity,
        intervals,
        budget=budget,
        policy=config.empty_interval_policy,
    )
    timing["empty_interval_repair_ms"] = (time.time() - st) * 1000.0

    st = time.time()
    dense_to_slot = build_interval_assignments(
        flat,
        selected,
        intervals,
        match_chunk_size=int(config.interval_match_chunk_size),
    )
    weights = build_merge_weights(
        diversity,
        dense_to_slot,
        output_count=budget,
        merge_mode=config.merge_mode,
        selected_indices=selected,
    )
    timing["assignment_ms"] = (time.time() - st) * 1000.0

    st = time.time()
    selected_sorted = torch.sort(selected.long()).values
    if config.merge_mode == "none":
        compressed = flat.index_select(0, selected_sorted)
    else:
        out = torch.zeros((budget, flat.shape[1]), dtype=torch.float32, device=flat.device)
        out.index_add_(0, dense_to_slot, flat.float() * weights[:, None])
        compressed = out.to(dtype=video_features.dtype)
    timing["weighted_merge_ms"] = (time.time() - st) * 1000.0
    timing["total_kitoke_ms"] = (time.time() - total_start) * 1000.0

    frame_indices = selected_sorted // int(m)
    counts = torch.bincount(frame_indices, minlength=int(t)).tolist()
    plan = KiTokePlan(
        selected_flat_indices=selected_sorted,
        num_tokens_per_frame=[int(x) for x in counts],
        output_frame_indices=frame_indices.long(),
        dense_to_output_slot=dense_to_slot.long(),
        normalized_merge_weights=weights.float(),
        dense_token_count=n,
        target_token_count=budget,
        retained_token_count=int(selected_sorted.numel()),
        num_frames=int(t),
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        tokens_per_frame=int(m),
        density=density if config.debug else None,
        diversity=diversity if config.debug else None,
        inclusion_probabilities=pi if config.debug else None,
        diff_pos=diff_pos if config.debug else None,
        diff_match=diff_match if config.debug else None,
        transition_diff=transition_diff if config.debug else None,
        transition_delta=delta if config.debug else None,
        transition_delta_pct=delta_pct if config.debug else None,
        boundary_transition_indices=boundaries,
        intervals=intervals,
        selection_method=config.selection_method,
        pivotal_pairing=config.pivotal_pairing,
        effective_seed=seed,
        empty_interval_policy=config.empty_interval_policy,
        repaired_empty_interval_count=int(repaired_count),
        promoted_indices=promoted,
        demoted_indices=demoted,
        merge_mode=config.merge_mode,
        deepstack_mode=config.deepstack_mode,
        timing_stats=timing,
    )
    if plan.retained_token_count != budget:
        raise RuntimeError(f"KiToke retained count mismatch: expected={budget} actual={plan.retained_token_count}")
    return KiTokeCompressionResult(compressed, plan)


def verify_kitoke_lengths(
    *,
    plan: KiTokePlan,
    placeholder_count: int,
    embedding_rows: int,
    mrope_count: int,
    deepstack_rows: Sequence[int],
    video_index: int,
) -> None:
    expected = int(plan.retained_token_count)
    if (
        int(placeholder_count) != expected
        or int(embedding_rows) != expected
        or int(mrope_count) != expected
        or any(int(x) != expected for x in deepstack_rows)
    ):
        raise RuntimeError(
            "[KiToke-vLLM] compact invariant failed: "
            f"video={video_index} dense={plan.dense_token_count} B={plan.target_token_count} "
            f"retained={expected} placeholders={placeholder_count} embeds={embedding_rows} "
            f"mrope={mrope_count} deepstack={list(deepstack_rows)} "
            f"frames={plan.num_frames} grid={plan.grid_h}x{plan.grid_w} "
            f"method={plan.selection_method} thresholds=({plan.selection_method}, {plan.empty_interval_policy})"
        )


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"Invalid boolean env {name}={raw!r}")


def _enabled_mode() -> str:
    return os.environ.get("QWEN3VL_KITOKE_MODE", "none").strip().lower()


def _debug_verify() -> bool:
    return _parse_bool_env("QWEN3VL_KITOKE_DEBUG_VERIFY", False)


def _verbose() -> bool:
    return not _parse_bool_env("QWEN3VL_KITOKE_QUIET", False)


def _env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name, "")
    if raw.strip() == "":
        return None
    return int(raw)


def _env_config() -> KiTokeAlgorithmConfig:
    return KiTokeAlgorithmConfig(
        retain_ratio=float(os.environ.get("QWEN3VL_KITOKE_RETAIN_RATIO", "0.10")),
        target_visual_tokens=_env_optional_int("QWEN3VL_KITOKE_TARGET_VISUAL_TOKENS"),
        kernel_alpha=float(os.environ.get("QWEN3VL_KITOKE_KERNEL_ALPHA", "800.0")),
        selection_method=os.environ.get("QWEN3VL_KITOKE_SELECTION_METHOD", "pivotal").strip().lower(),
        selection_seed=int(os.environ.get("QWEN3VL_KITOKE_SELECTION_SEED", "3407")),
        seed_policy=os.environ.get("QWEN3VL_KITOKE_SEED_POLICY", "stable_video").strip().lower(),
        pivotal_pairing=os.environ.get("QWEN3VL_KITOKE_PIVOTAL_PAIRING", "random_rounds").strip().lower(),
        diff_threshold=float(os.environ.get("QWEN3VL_KITOKE_DIFF_THRESHOLD", "110.0")),
        delta_threshold=float(os.environ.get("QWEN3VL_KITOKE_DELTA_THRESHOLD", "70.0")),
        relative_delta_threshold=float(os.environ.get("QWEN3VL_KITOKE_RELATIVE_DELTA_THRESHOLD", "0.40")),
        edge_policy=os.environ.get("QWEN3VL_KITOKE_EDGE_POLICY", "absolute_only").strip().lower(),
        empty_interval_policy=os.environ.get("QWEN3VL_KITOKE_EMPTY_INTERVAL_POLICY", "repair_swap").strip().lower(),
        merge_mode=os.environ.get("QWEN3VL_KITOKE_MERGE_MODE", "weighted").strip().lower(),
        deepstack_mode=os.environ.get("QWEN3VL_KITOKE_DEEPSTACK_MODE", "same_weighted_merge").strip().lower(),
        kernel_row_chunk_size=int(os.environ.get("QWEN3VL_KITOKE_KERNEL_ROW_CHUNK_SIZE", "256")),
        kernel_col_chunk_size=int(os.environ.get("QWEN3VL_KITOKE_KERNEL_COL_CHUNK_SIZE", "512")),
        frame_match_chunk_size=int(os.environ.get("QWEN3VL_KITOKE_FRAME_MATCH_CHUNK_SIZE", "256")),
        interval_match_chunk_size=int(os.environ.get("QWEN3VL_KITOKE_INTERVAL_MATCH_CHUNK_SIZE", "256")),
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
    return int(t), int(dense_h), int(dense_w), int(dense_h * dense_w)


def _target_count_for_row(row: Any, spatial_merge_size: int) -> int:
    t, _, _, tokens_per_frame = _tokens_per_frame_for_row(row, spatial_merge_size)
    return _budget_from_inputs(int(t * tokens_per_frame), _env_config())


def _selected_video_sizes(grid_thw: torch.Tensor, spatial_merge_size: int) -> list[int]:
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


def _compress_video_outputs(
    main: torch.Tensor,
    deepstack: Sequence[torch.Tensor],
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    debug_verify: bool,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    offset = 0
    cfg = _env_config()
    for video_idx, row in enumerate(grid_thw):
        st = time.time()
        t, dense_h, dense_w, tokens_per_frame = _tokens_per_frame_for_row(row, spatial_merge_size)
        dense_tokens = int(t * tokens_per_frame)
        target = _target_count_for_row(row, spatial_merge_size)
        main_chunk = main[offset : offset + dense_tokens]
        deep_chunks = [layer[offset : offset + dense_tokens] for layer in deepstack]
        identity = _feature_seed_identity(main_chunk, video_index=video_idx, t=t, h=dense_h, w=dense_w)
        result = compress_video_kitoke(
            main_chunk.reshape(t, tokens_per_frame, -1),
            grid_h=dense_h,
            grid_w=dense_w,
            config=cfg,
            stable_video_identity=identity,
        )
        plan = result.plan
        reps = plan.selected_flat_indices.to(main.device)
        if cfg.deepstack_mode == "representative_gather":
            selected_deep = [layer.index_select(0, reps) for layer in deep_chunks]
        else:
            selected_deep = [apply_kitoke_plan(layer, plan, mode="same_weighted_merge") for layer in deep_chunks]
        dense_pos = _dense_local_mrope_positions(t, dense_h, dense_w, device=main.device, dtype=main.dtype)
        selected_pos = dense_pos.index_select(0, reps)
        out = torch.cat([result.compressed_main_embeddings.to(main.dtype)] + selected_deep + [selected_pos], dim=1)
        chunks.append(out)
        if _verbose() or debug_verify:
            extra = f" intervals={len(plan.intervals)} per_frame={plan.num_tokens_per_frame}" if debug_verify else ""
            print(
                f"[KiToke-vLLM] video[{video_idx}] frames={t} grid={dense_h}x{dense_w} "
                f"dense={plan.dense_token_count} target={plan.target_token_count} "
                f"actual={plan.retained_token_count} ratio={plan.retained_token_count / max(plan.dense_token_count, 1):.4f} "
                f"selection={plan.selection_method} merge={plan.merge_mode} "
                f"deepstack={plan.deepstack_mode} seed={plan.effective_seed} "
                f"time={time.time() - st:.3f}s{extra}"
            )
        if debug_verify:
            verify_kitoke_lengths(
                plan=plan,
                placeholder_count=target,
                embedding_rows=out.shape[0],
                mrope_count=selected_pos.shape[0],
                deepstack_rows=[x.shape[0] for x in selected_deep],
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
        t, dense_h, dense_w, tokens_per_frame = _tokens_per_frame_for_row(row, spatial_merge_size)
        dense_tokens = int(t * tokens_per_frame)
        emb = image_embeds[offset : offset + dense_tokens]
        pos = _dense_local_mrope_positions(t, dense_h, dense_w, device=emb.device, dtype=emb.dtype)
        chunks.append(torch.cat([emb, pos], dim=1))
        offset += dense_tokens
    return tuple(chunks)


def _validate_patch_params(cfg: KiTokeAlgorithmConfig, mode: str) -> None:
    if mode not in {"post_vit"}:
        raise ValueError(f"mode must be post_vit/none, got {mode!r}")
    if not math.isfinite(float(cfg.retain_ratio)) or cfg.retain_ratio <= 0 or cfg.retain_ratio > 1:
        raise ValueError("retain_ratio must be in (0,1]")
    if cfg.target_visual_tokens is not None and int(cfg.target_visual_tokens) <= 0:
        raise ValueError("target_visual_tokens must be None or positive")
    if not math.isfinite(float(cfg.kernel_alpha)) or cfg.kernel_alpha <= 0:
        raise ValueError("kernel_alpha must be finite and > 0")
    if cfg.selection_method not in {"pivotal", "multinomial", "topk"}:
        raise ValueError("selection_method must be pivotal/multinomial/topk")
    if cfg.seed_policy not in {"stable_video", "fixed"}:
        raise ValueError("seed_policy must be stable_video/fixed")
    if cfg.pivotal_pairing not in {"random_rounds", "sequential"}:
        raise ValueError("pivotal_pairing must be random_rounds/sequential")
    if cfg.edge_policy not in {"absolute_only", "one_sided"}:
        raise ValueError("edge_policy must be absolute_only/one_sided")
    if cfg.empty_interval_policy not in {"paper_strict", "repair_swap", "coarsen_then_repair"}:
        raise ValueError("empty_interval_policy must be paper_strict/repair_swap/coarsen_then_repair")
    if cfg.merge_mode not in {"weighted", "uniform", "none"}:
        raise ValueError("merge_mode must be weighted/uniform/none")
    if cfg.deepstack_mode not in {"same_weighted_merge", "representative_gather"}:
        raise ValueError("deepstack_mode must be same_weighted_merge/representative_gather")
    for name in (
        "kernel_row_chunk_size",
        "kernel_col_chunk_size",
        "frame_match_chunk_size",
        "interval_match_chunk_size",
    ):
        if int(getattr(cfg, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("diff_threshold", "delta_threshold", "relative_delta_threshold"):
        if not math.isfinite(float(getattr(cfg, name))):
            raise ValueError(f"{name} must be finite")


def apply_patch(
    mode: str = "post_vit",
    retain_ratio: float = 0.10,
    target_visual_tokens: int | None = None,
    kernel_alpha: float = 800.0,
    selection_method: str = "pivotal",
    selection_seed: int = 3407,
    seed_policy: str = "stable_video",
    pivotal_pairing: str = "random_rounds",
    diff_threshold: float = 110.0,
    delta_threshold: float = 70.0,
    relative_delta_threshold: float = 0.40,
    edge_policy: str = "absolute_only",
    empty_interval_policy: str = "repair_swap",
    merge_mode: str = "weighted",
    deepstack_mode: str = "same_weighted_merge",
    kernel_row_chunk_size: int = 256,
    kernel_col_chunk_size: int = 512,
    frame_match_chunk_size: int = 256,
    interval_match_chunk_size: int = 256,
    debug_verify: bool = False,
) -> None:
    global _PATCHED
    mode = (mode or "none").strip().lower()
    if mode in {"none", "off", ""}:
        return
    cfg = KiTokeAlgorithmConfig(
        retain_ratio=float(retain_ratio),
        target_visual_tokens=target_visual_tokens,
        kernel_alpha=float(kernel_alpha),
        selection_method=(selection_method or "pivotal").strip().lower(),
        selection_seed=int(selection_seed),
        seed_policy=(seed_policy or "stable_video").strip().lower(),
        pivotal_pairing=(pivotal_pairing or "random_rounds").strip().lower(),
        diff_threshold=float(diff_threshold),
        delta_threshold=float(delta_threshold),
        relative_delta_threshold=float(relative_delta_threshold),
        edge_policy=(edge_policy or "absolute_only").strip().lower(),
        empty_interval_policy=(empty_interval_policy or "repair_swap").strip().lower(),
        merge_mode=(merge_mode or "weighted").strip().lower(),
        deepstack_mode=(deepstack_mode or "same_weighted_merge").strip().lower(),
        kernel_row_chunk_size=int(kernel_row_chunk_size),
        kernel_col_chunk_size=int(kernel_col_chunk_size),
        frame_match_chunk_size=int(frame_match_chunk_size),
        interval_match_chunk_size=int(interval_match_chunk_size),
        debug=bool(debug_verify),
    )
    _validate_patch_params(cfg, mode)
    os.environ["QWEN3VL_KITOKE_MODE"] = mode
    os.environ["QWEN3VL_KITOKE_RETAIN_RATIO"] = str(cfg.retain_ratio)
    os.environ["QWEN3VL_KITOKE_TARGET_VISUAL_TOKENS"] = "" if cfg.target_visual_tokens is None else str(int(cfg.target_visual_tokens))
    os.environ["QWEN3VL_KITOKE_KERNEL_ALPHA"] = str(cfg.kernel_alpha)
    os.environ["QWEN3VL_KITOKE_SELECTION_METHOD"] = cfg.selection_method
    os.environ["QWEN3VL_KITOKE_SELECTION_SEED"] = str(int(cfg.selection_seed))
    os.environ["QWEN3VL_KITOKE_SEED_POLICY"] = cfg.seed_policy
    os.environ["QWEN3VL_KITOKE_PIVOTAL_PAIRING"] = cfg.pivotal_pairing
    os.environ["QWEN3VL_KITOKE_DIFF_THRESHOLD"] = str(cfg.diff_threshold)
    os.environ["QWEN3VL_KITOKE_DELTA_THRESHOLD"] = str(cfg.delta_threshold)
    os.environ["QWEN3VL_KITOKE_RELATIVE_DELTA_THRESHOLD"] = str(cfg.relative_delta_threshold)
    os.environ["QWEN3VL_KITOKE_EDGE_POLICY"] = cfg.edge_policy
    os.environ["QWEN3VL_KITOKE_EMPTY_INTERVAL_POLICY"] = cfg.empty_interval_policy
    os.environ["QWEN3VL_KITOKE_MERGE_MODE"] = cfg.merge_mode
    os.environ["QWEN3VL_KITOKE_DEEPSTACK_MODE"] = cfg.deepstack_mode
    os.environ["QWEN3VL_KITOKE_KERNEL_ROW_CHUNK_SIZE"] = str(int(cfg.kernel_row_chunk_size))
    os.environ["QWEN3VL_KITOKE_KERNEL_COL_CHUNK_SIZE"] = str(int(cfg.kernel_col_chunk_size))
    os.environ["QWEN3VL_KITOKE_FRAME_MATCH_CHUNK_SIZE"] = str(int(cfg.frame_match_chunk_size))
    os.environ["QWEN3VL_KITOKE_INTERVAL_MATCH_CHUNK_SIZE"] = str(int(cfg.interval_match_chunk_size))
    os.environ["QWEN3VL_KITOKE_DEBUG_VERIFY"] = "1" if debug_verify else "0"

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
            raise AttributeError(f"[KiToke-vLLM] unsupported local Qwen3-VL API; missing {name}")

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

    def patched_vision_forward(self, x, grid_thw, kitoke_is_video=False):
        if _enabled_mode() != "post_vit" or not kitoke_is_video:
            return _ORIGINALS["Qwen3_VisionTransformer.forward"](self, x, grid_thw)
        hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        hidden_states = self.patch_embed(hidden_states)
        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_tensor = torch.tensor(grid_thw, dtype=torch.int64)
            grid_np = np.array(grid_thw, dtype=np.int32)
        else:
            grid_thw_list = grid_thw.tolist()
            grid_tensor = grid_thw.to(dtype=torch.int64) if isinstance(grid_thw, torch.Tensor) else torch.tensor(grid_thw, dtype=torch.int64)
            grid_np = grid_thw.detach().cpu().numpy() if isinstance(grid_thw, torch.Tensor) else np.array(grid_thw, dtype=np.int32)
        hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid_thw_list)
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)
        cu_np = np.repeat(grid_np[:, 1] * grid_np[:, 2], grid_np[:, 0]).cumsum(axis=0, dtype=np.int32)
        cu_np = np.concatenate([np.zeros(1, dtype=np.int32), cu_np])
        cu_seqlens_cpu = torch.from_numpy(cu_np)
        hidden_states = hidden_states.unsqueeze(1)
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens_cpu)
        cu_seqlens = cu_seqlens_cpu.to(self.device, non_blocking=True)
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
                raise RuntimeError("KiToke does not support mm_encoder_tp_mode=data yet.")
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
            raise NotImplementedError("[KiToke-vLLM] raw pixel_values_videos are required so merged embeddings can be constructed.")
        if self.use_data_parallel:
            raise RuntimeError("KiToke does not support mm_encoder_tp_mode=data yet.")
        pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw, kitoke_is_video=True)
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
            f"[KiToke-vLLM] enabled mode={mode} ratio={cfg.retain_ratio:.4f} "
            f"target={cfg.target_visual_tokens} alpha={cfg.kernel_alpha:.1f} "
            f"selection={cfg.selection_method} merge={cfg.merge_mode} deepstack={cfg.deepstack_mode}"
        )
        print(f"[KiToke-vLLM] local_api={required}")
