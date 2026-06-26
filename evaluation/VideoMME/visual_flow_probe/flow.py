from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class GridMapping:
    visual_seq_positions: torch.LongTensor
    visual_local_indices: torch.LongTensor
    video_indices: torch.LongTensor
    temporal_grid_indices: torch.LongTensor
    y_grid_indices: torch.LongTensor
    x_grid_indices: torch.LongTensor
    video_grid_thw: torch.LongTensor


def assert_causal_attention(attention: torch.Tensor, *, atol: float = 1e-4) -> None:
    """Validate A[q, k] has no mass on future keys k > q."""
    if attention.ndim != 2 or attention.shape[0] != attention.shape[1]:
        raise ValueError(f"attention must be square [seq, seq], got {tuple(attention.shape)}")
    future = torch.triu(attention.detach().float(), diagonal=1)
    max_future = float(future.abs().max().item()) if future.numel() else 0.0
    if max_future > atol:
        raise ValueError(f"attention is not causal: max future-key mass={max_future:.6g} > {atol}")


def compute_answer_reachability(
    attention: torch.Tensor,
    target_positions: Sequence[int],
    *,
    check_causal: bool = True,
) -> torch.Tensor:
    """Backward path-mass DP.

    Args:
        attention: Averaged decoder attention A[q, k] with query rows and key
            columns. A[later_query, earlier_key] represents earlier_key -> later_query.
        target_positions: Answer sink positions.

    Returns:
        h[i], the total attention-weighted path mass from source i to the target
        region through all later intermediate nodes.
    """
    if attention.ndim != 2 or attention.shape[0] != attention.shape[1]:
        raise ValueError(f"attention must be [seq, seq], got {tuple(attention.shape)}")
    seq_len = attention.shape[0]
    targets = sorted({int(p) for p in target_positions})
    if not targets:
        raise ValueError("target_positions is empty")
    bad = [p for p in targets if p < 0 or p >= seq_len]
    if bad:
        raise ValueError(f"target positions out of range for seq_len={seq_len}: {bad}")
    if check_causal:
        assert_causal_attention(attention)

    a = attention.detach().to(dtype=torch.float64, device="cpu")
    h = torch.zeros(seq_len, dtype=torch.float64)
    b = torch.zeros(seq_len, dtype=torch.float64)
    b[torch.tensor(targets, dtype=torch.long)] = 1.0 / float(len(targets))
    for i in range(seq_len - 1, -1, -1):
        # Only strict forward-in-time paths participate: source i -> later node j.
        h[i] = b[i]
        if i + 1 < seq_len:
            h[i] += torch.dot(a[i + 1 :, i], h[i + 1 :])
    if not torch.isfinite(h).all():
        raise ValueError("reachability contains non-finite values")
    return h


def normalize_visual_responsibility(
    reachability: torch.Tensor,
    visual_seq_positions: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Normalize h over visual source positions only."""
    positions = torch.as_tensor(visual_seq_positions, dtype=torch.long, device=reachability.device)
    vals = reachability.index_select(0, positions).to(torch.float64)
    denom = vals.sum()
    if not torch.isfinite(denom) or float(denom.item()) <= 0:
        raise ValueError(f"visual responsibility denominator is invalid: {float(denom.item())}")
    out = vals / denom
    if not torch.isfinite(out).all():
        raise ValueError("visual responsibility contains non-finite values")
    return out


def direct_target_attention(
    attention: torch.Tensor,
    target_positions: Sequence[int],
    visual_seq_positions: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Mean direct A[target, visual] normalized across visual tokens when possible."""
    targets = torch.as_tensor(list(target_positions), dtype=torch.long, device=attention.device)
    visuals = torch.as_tensor(visual_seq_positions, dtype=torch.long, device=attention.device)
    if targets.numel() == 0 or visuals.numel() == 0:
        raise ValueError("target_positions and visual_seq_positions must be non-empty")
    scores = attention.index_select(0, targets).index_select(1, visuals).float().mean(dim=0).to(torch.float64)
    denom = scores.sum()
    if torch.isfinite(denom) and float(denom.item()) > 0:
        scores = scores / denom
    return scores.cpu()


def _rank_average(values: torch.Tensor) -> torch.Tensor:
    """Average ranks for ties, 0-based."""
    x = values.detach().cpu().to(torch.float64)
    order = torch.argsort(x, stable=True)
    ranks = torch.empty_like(x)
    sorted_x = x[order]
    n = x.numel()
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_x[end] == sorted_x[start]:
            end += 1
        avg = (start + end - 1) / 2.0
        ranks[order[start:end]] = avg
        start = end
    return ranks


def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() != y.numel() or x.numel() < 2:
        return float("nan")
    rx = _rank_average(x)
    ry = _rank_average(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = torch.linalg.vector_norm(rx) * torch.linalg.vector_norm(ry)
    if float(denom.item()) == 0:
        return float("nan")
    return float(torch.dot(rx, ry).item() / denom.item())


def gini_coefficient(values: torch.Tensor) -> float:
    x = values.detach().cpu().to(torch.float64).flatten()
    if x.numel() == 0:
        return float("nan")
    if (x < 0).any():
        x = x - x.min()
    total = x.sum()
    if float(total.item()) <= 0:
        return 0.0
    x_sorted = torch.sort(x).values
    n = x_sorted.numel()
    idx = torch.arange(1, n + 1, dtype=torch.float64)
    return float(((2 * idx - n - 1) * x_sorted).sum().item() / (n * total.item()))


def concentration_statistics(
    scores: torch.Tensor,
    *,
    temporal_grid_indices: torch.Tensor | None = None,
) -> dict[str, float]:
    """Responsibility concentration diagnostics for a non-negative score vector."""
    x = scores.detach().cpu().to(torch.float64).flatten()
    if x.numel() == 0:
        raise ValueError("scores is empty")
    total = x.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0:
        raise ValueError("scores must have positive finite sum")
    p = x / total
    sorted_p = torch.sort(p, descending=True).values
    n = p.numel()
    stats: dict[str, float] = {}
    for frac in (0.01, 0.05, 0.10, 0.20):
        k = max(1, int(math.ceil(frac * n)))
        stats[f"top_{int(frac * 100)}pct_mass"] = float(sorted_p[:k].sum().item())
    entropy = -(p.clamp_min(1e-300) * p.clamp_min(1e-300).log()).sum()
    stats["normalized_entropy"] = float(entropy.item() / math.log(n)) if n > 1 else 0.0
    effective = float(torch.exp(entropy).item())
    stats["effective_token_count"] = effective
    stats["effective_token_fraction"] = effective / float(n)
    stats["gini"] = gini_coefficient(p)
    stats["max"] = float(p.max().item())
    stats["mean"] = float(p.mean().item())
    stats["std"] = float(p.std(unbiased=False).item())
    stats["spearman_local_index"] = spearman_corr(p, torch.arange(n, dtype=torch.float64))
    if temporal_grid_indices is not None:
        stats["spearman_temporal_index"] = spearman_corr(p, temporal_grid_indices.to(torch.float64).cpu())
    else:
        stats["spearman_temporal_index"] = float("nan")
    return stats


def map_video_tokens_to_grid(
    visual_seq_positions: Sequence[int] | torch.Tensor,
    video_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> GridMapping:
    """Map flattened video placeholder order to (video, t, merged_y, merged_x)."""
    positions = torch.as_tensor(visual_seq_positions, dtype=torch.long).cpu()
    grid = torch.as_tensor(video_grid_thw, dtype=torch.long).cpu()
    if grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError(f"video_grid_thw must be [num_videos, 3], got {tuple(grid.shape)}")
    if spatial_merge_size <= 0:
        raise ValueError(f"spatial_merge_size must be positive, got {spatial_merge_size}")

    video_ids: list[int] = []
    ts: list[int] = []
    ys: list[int] = []
    xs: list[int] = []
    expected = 0
    for vid, (t_raw, h_raw, w_raw) in enumerate(grid.tolist()):
        t, h, w = int(t_raw), int(h_raw), int(w_raw)
        if h % spatial_merge_size or w % spatial_merge_size:
            raise ValueError(
                f"grid {vid}={t,h,w} is not divisible by spatial_merge_size={spatial_merge_size}"
            )
        mh, mw = h // spatial_merge_size, w // spatial_merge_size
        count = t * mh * mw
        expected += count
        for local in range(count):
            frame_area = mh * mw
            tt = local // frame_area
            rem = local % frame_area
            yy = rem // mw
            xx = rem % mw
            video_ids.append(vid)
            ts.append(tt)
            ys.append(yy)
            xs.append(xx)

    if expected != positions.numel():
        raise ValueError(
            "video token/grid count mismatch: "
            f"visual_tokens={positions.numel()} expected={expected} "
            f"video_grid_thw={grid.tolist()} spatial_merge_size={spatial_merge_size}"
        )
    local_indices = torch.arange(positions.numel(), dtype=torch.long)
    return GridMapping(
        visual_seq_positions=positions,
        visual_local_indices=local_indices,
        video_indices=torch.tensor(video_ids, dtype=torch.long),
        temporal_grid_indices=torch.tensor(ts, dtype=torch.long),
        y_grid_indices=torch.tensor(ys, dtype=torch.long),
        x_grid_indices=torch.tensor(xs, dtype=torch.long),
        video_grid_thw=grid,
    )


def visual_positions_from_input_ids(
    input_ids: torch.Tensor,
    video_token_id: int,
    *,
    mm_token_type_ids: torch.Tensor | None = None,
) -> tuple[torch.LongTensor, dict[str, object]]:
    ids = input_ids[0] if input_ids.ndim == 2 else input_ids
    positions = torch.nonzero(ids == int(video_token_id), as_tuple=False).flatten().cpu()
    diag: dict[str, object] = {
        "video_token_id": int(video_token_id),
        "video_token_count": int(positions.numel()),
        "mm_token_type_crosscheck": "unavailable",
    }
    if mm_token_type_ids is not None:
        mm = mm_token_type_ids[0] if mm_token_type_ids.ndim == 2 else mm_token_type_ids
        # Local Qwen3VLProcessor marks image tokens as 1 and may leave videos as 0.
        mm_non_text = torch.nonzero(mm != 0, as_tuple=False).flatten().cpu()
        diag["mm_token_type_nonzero_count"] = int(mm_non_text.numel())
        if mm_non_text.numel() == positions.numel() and torch.equal(mm_non_text, positions):
            diag["mm_token_type_crosscheck"] = "agree"
        else:
            diag["mm_token_type_crosscheck"] = (
                "different_or_not_video_specific; using input_ids == video_token_id"
            )
    return positions, diag
