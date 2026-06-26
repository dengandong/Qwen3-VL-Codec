"""EchoPrune utilities and vLLM Qwen3-VL patch hooks.

The pure PyTorch implementation at the top of this file does not import vLLM
and is intended to be CPU-testable.  The integration layer below follows the
local Qwen3-VL/vLLM 0.12.0 monkey-patch style already used by the VideoMME
TTF and codec-guided experiments: it computes pruning decisions after the
vision merger, appends sparse local M-RoPE positions to selected embeddings,
and lets vLLM's multimodal-pruning recompute path build the compact LLM
sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


_PATCHED = False
_ORIGINALS: dict[str, Any] = {}


@dataclass(frozen=True)
class EchoPrunePlan:
    keep_flat_indices: torch.LongTensor
    keep_indices_per_frame: list[torch.LongTensor]
    num_tokens_per_frame: list[int]
    relevance: torch.Tensor | None
    delta_corr: torch.Tensor | None
    delta_echo: torch.Tensor | None
    score: torch.Tensor | None
    dense_token_count: int
    target_token_count: int
    retained_token_count: int
    first_frame_quota: int
    query_token_count: int
    num_frames: int
    grid_h: int
    grid_w: int


def _as_float32_normalized(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-6)


def _validate_video_query(
    video_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    grid_h: int | None = None,
    grid_w: int | None = None,
) -> tuple[int, int, int]:
    if video_tokens.ndim != 3:
        raise ValueError(f"video_tokens must be [T,N,C], got {tuple(video_tokens.shape)}")
    if query_tokens.ndim != 2:
        raise ValueError(f"query_tokens must be [M,C], got {tuple(query_tokens.shape)}")
    t, n, c = [int(v) for v in video_tokens.shape]
    m, qc = [int(v) for v in query_tokens.shape]
    if t <= 0 or n <= 0 or c <= 0:
        raise ValueError(f"video_tokens dimensions must be positive, got {tuple(video_tokens.shape)}")
    if m <= 0:
        raise ValueError("EchoPrune query is empty; question/options text is required")
    if c != qc:
        raise ValueError(f"EchoPrune dimension mismatch: video C={c}, query C={qc}")
    if grid_h is not None and grid_w is not None and int(grid_h) * int(grid_w) != n:
        raise ValueError(
            f"grid_h*grid_w must equal per-frame tokens: grid={grid_h}x{grid_w}, N={n}"
        )
    return t, n, c


def _valid_query_mask(query_tokens: torch.Tensor, query_mask: torch.Tensor | None) -> torch.BoolTensor:
    if query_mask is None:
        mask = torch.ones(query_tokens.shape[0], device=query_tokens.device, dtype=torch.bool)
    else:
        if query_mask.ndim != 1 or int(query_mask.shape[0]) != int(query_tokens.shape[0]):
            raise ValueError(
                f"query_mask must be [M] matching query_tokens; got {tuple(query_mask.shape)}"
            )
        mask = query_mask.to(device=query_tokens.device, dtype=torch.bool)
    if not bool(mask.any().item()):
        raise ValueError("EchoPrune query has no valid tokens after masking")
    return mask


def compute_crossmodal_relevance(
    video_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    query_mask: torch.Tensor | None = None,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Compute max text-token relevance for every visual token.

    Args:
        video_tokens: Projector/merger output, shaped [T, N, C].
        query_tokens: Same-language-model text embeddings, shaped [M, C].
        query_mask: Optional [M] bool mask; masked tokens do not participate.
    """
    _validate_video_query(video_tokens, query_tokens)
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    mask = _valid_query_mask(query_tokens, query_mask)
    q = _as_float32_normalized(query_tokens[mask])
    v = _as_float32_normalized(video_tokens).reshape(-1, video_tokens.shape[-1])
    out = torch.full((v.shape[0],), -torch.inf, device=v.device, dtype=torch.float32)
    chunk = max(1, int(chunk_size))
    for start in range(0, v.shape[0], chunk):
        sim = v[start : start + chunk] @ q.T
        out[start : start + chunk] = sim.max(dim=-1).values
    return out.reshape(video_tokens.shape[0], video_tokens.shape[1])


def _local_candidate_indices(
    grid_h: int,
    grid_w: int,
    window_size: int,
    *,
    device: torch.device,
) -> list[torch.LongTensor]:
    radius = int(window_size) // 2
    candidates: list[torch.LongTensor] = []
    for r in range(int(grid_h)):
        for c in range(int(grid_w)):
            idxs = []
            for rr in range(max(0, r - radius), min(int(grid_h), r + radius + 1)):
                for cc in range(max(0, c - radius), min(int(grid_w), c + radius + 1)):
                    idxs.append(rr * int(grid_w) + cc)
            candidates.append(torch.tensor(idxs, device=device, dtype=torch.long))
    return candidates


def compute_temporal_echo_scores(
    video_tokens: torch.Tensor,
    *,
    temperature: float,
    match_scope: str,
    grid_h: int,
    grid_w: int,
    window_size: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return adjacent correspondence and soft temporal echo scores.

    Both outputs are [T, N].  The first frame is filled with zero because it has
    no causal previous frame.  The implementation normalizes input visual
    features, computes echo_hat from normalized previous tokens, and does not
    normalize echo_hat again.
    """
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
    match_scope = (match_scope or "full").strip().lower()
    if match_scope not in {"full", "local"}:
        raise ValueError(f"match_scope must be full/local, got {match_scope!r}")
    if int(window_size) <= 0 or int(window_size) % 2 != 1:
        raise ValueError(f"window_size must be a positive odd integer, got {window_size}")
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    t, n, _ = _validate_video_query(
        video_tokens,
        torch.empty((1, video_tokens.shape[-1]), device=video_tokens.device, dtype=video_tokens.dtype),
        int(grid_h),
        int(grid_w),
    )
    v = _as_float32_normalized(video_tokens)
    delta_corr = torch.zeros((t, n), device=video_tokens.device, dtype=torch.float32)
    delta_echo = torch.zeros_like(delta_corr)
    if t == 1:
        return delta_corr, delta_echo

    chunk = max(1, int(chunk_size))
    tau = float(temperature)
    local_candidates = None
    if match_scope == "local":
        local_candidates = _local_candidate_indices(grid_h, grid_w, window_size, device=video_tokens.device)

    for frame_idx in range(1, t):
        curr = v[frame_idx]
        prev = v[frame_idx - 1]
        delta_corr[frame_idx] = (curr * prev).sum(dim=-1)

        if match_scope == "full":
            prev_t = prev.T
            for start in range(0, n, chunk):
                cur_chunk = curr[start : start + chunk]
                sim = cur_chunk @ prev_t
                probs = torch.softmax(sim / tau, dim=-1)
                echo_hat = probs @ prev
                delta_echo[frame_idx, start : start + chunk] = (cur_chunk * echo_hat).sum(dim=-1)
        else:
            assert local_candidates is not None
            for start in range(0, n, chunk):
                end = min(n, start + chunk)
                vals = []
                for pos in range(start, end):
                    idx = local_candidates[pos]
                    cand = prev.index_select(0, idx)
                    sim = curr[pos : pos + 1] @ cand.T
                    probs = torch.softmax(sim / tau, dim=-1)
                    echo_hat = probs @ cand
                    vals.append((curr[pos : pos + 1] * echo_hat).sum(dim=-1))
                delta_echo[frame_idx, start:end] = torch.cat(vals, dim=0)
    return delta_corr, delta_echo


def reference_compute_temporal_echo_scores(
    video_tokens: torch.Tensor,
    *,
    temperature: float,
    match_scope: str,
    grid_h: int,
    grid_w: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slow reference implementation for small CPU tests."""
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
    match_scope = (match_scope or "full").strip().lower()
    if match_scope not in {"full", "local"}:
        raise ValueError(f"match_scope must be full/local, got {match_scope!r}")
    t, n, _ = _validate_video_query(
        video_tokens,
        torch.empty((1, video_tokens.shape[-1]), device=video_tokens.device, dtype=video_tokens.dtype),
        int(grid_h),
        int(grid_w),
    )
    v = _as_float32_normalized(video_tokens)
    corr = torch.zeros((t, n), dtype=torch.float32, device=video_tokens.device)
    echo = torch.zeros_like(corr)
    radius = int(window_size) // 2
    for k in range(1, t):
        for i in range(n):
            corr[k, i] = torch.dot(v[k, i], v[k - 1, i])
            if match_scope == "full":
                idxs = list(range(n))
            else:
                r = i // int(grid_w)
                c = i % int(grid_w)
                idxs = [
                    rr * int(grid_w) + cc
                    for rr in range(max(0, r - radius), min(int(grid_h), r + radius + 1))
                    for cc in range(max(0, c - radius), min(int(grid_w), c + radius + 1))
                ]
            cand = v[k - 1, idxs]
            sim = v[k, i : i + 1] @ cand.T
            probs = torch.softmax(sim / float(temperature), dim=-1)
            echo_hat = probs @ cand
            echo[k, i] = torch.sum(v[k, i : i + 1] * echo_hat)
    return corr, echo


def _budget_from_inputs(
    total_tokens: int,
    *,
    retain_ratio: float | None,
    target_visual_tokens: int | None,
) -> int:
    if target_visual_tokens is not None:
        target = int(target_visual_tokens)
    else:
        rr = 0.20 if retain_ratio is None else float(retain_ratio)
        if not math.isfinite(rr) or rr <= 0.0 or rr > 1.0:
            raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio!r}")
        target = int(round(int(total_tokens) * rr))
    return max(1, min(int(total_tokens), int(target)))


def _topk_by_score_then_index(
    scores: torch.Tensor,
    candidate_flat_indices: torch.LongTensor,
    k: int,
) -> torch.LongTensor:
    if int(k) <= 0:
        return candidate_flat_indices.new_empty((0,), dtype=torch.long)
    if candidate_flat_indices.numel() == 0:
        return candidate_flat_indices.new_empty((0,), dtype=torch.long)
    # Stable descending score sort preserves ascending flat-index order for ties
    # because candidate_flat_indices are always supplied in ascending order.
    ordered = torch.argsort(scores, descending=True, stable=True)
    return candidate_flat_indices.index_select(0, ordered[: min(int(k), ordered.numel())])


def build_echoprune_plan(
    video_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    *,
    target_visual_tokens: int | None = None,
    retain_ratio: float | None = 0.20,
    query_mask: torch.Tensor | None = None,
    temperature: float = 0.5,
    match_scope: str = "full",
    grid_h: int,
    grid_w: int,
    window_size: int = 3,
    first_frame_policy: str = "paper",
    chunk_size: int = 256,
    debug: bool = False,
) -> EchoPrunePlan:
    """Build an EchoPrune selection plan for one video item."""
    del debug
    first_frame_policy = (first_frame_policy or "paper").strip().lower()
    if first_frame_policy not in {"paper", "global"}:
        raise ValueError(
            f"first_frame_policy must be paper/global, got {first_frame_policy!r}"
        )
    t, n, _ = _validate_video_query(video_tokens, query_tokens, int(grid_h), int(grid_w))
    mask = _valid_query_mask(query_tokens, query_mask)
    total = int(t * n)
    budget = _budget_from_inputs(
        total,
        retain_ratio=retain_ratio,
        target_visual_tokens=target_visual_tokens,
    )

    relevance = compute_crossmodal_relevance(video_tokens, query_tokens, mask, chunk_size)
    corr, echo = compute_temporal_echo_scores(
        video_tokens,
        temperature=temperature,
        match_scope=match_scope,
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        window_size=int(window_size),
        chunk_size=int(chunk_size),
    )
    score = relevance - corr - echo

    all_indices = torch.arange(total, device=video_tokens.device, dtype=torch.long)
    if budget == total:
        keep = all_indices
        first_quota = n if t > 0 else 0
    elif t == 1:
        keep = _topk_by_score_then_index(relevance[0], all_indices, budget)
        first_quota = budget
    elif first_frame_policy == "global":
        global_scores = score.reshape(-1).clone()
        global_scores[:n] = relevance[0]
        keep = _topk_by_score_then_index(global_scores, all_indices, budget)
        first_quota = int((keep < n).sum().item())
    else:
        first_quota = min(n, budget, max(1, budget // t))
        first_indices = torch.arange(n, device=video_tokens.device, dtype=torch.long)
        selected_first = _topk_by_score_then_index(relevance[0], first_indices, first_quota)
        remaining = budget - int(selected_first.numel())
        later_indices = torch.arange(n, total, device=video_tokens.device, dtype=torch.long)
        selected_later = _topk_by_score_then_index(score.reshape(-1)[n:], later_indices, remaining)
        keep_parts = [selected_first, selected_later]
        selected_count = int(selected_first.numel() + selected_later.numel())
        if selected_count < budget:
            first_mask = torch.ones(n, device=video_tokens.device, dtype=torch.bool)
            first_mask[selected_first] = False
            fill_candidates = first_indices[first_mask]
            fill_scores = relevance[0].index_select(0, fill_candidates)
            keep_parts.append(
                _topk_by_score_then_index(fill_scores, fill_candidates, budget - selected_count)
            )
        keep = torch.cat(keep_parts, dim=0)

    keep = torch.sort(keep.to(torch.long)).values
    per_frame: list[torch.LongTensor] = []
    counts: list[int] = []
    for frame_idx in range(t):
        start = frame_idx * n
        end = start + n
        local = keep[(keep >= start) & (keep < end)] - start
        local = local.to(torch.long)
        per_frame.append(local)
        counts.append(int(local.numel()))

    return EchoPrunePlan(
        keep_flat_indices=keep,
        keep_indices_per_frame=per_frame,
        num_tokens_per_frame=counts,
        relevance=relevance,
        delta_corr=corr,
        delta_echo=echo,
        score=score,
        dense_token_count=total,
        target_token_count=budget,
        retained_token_count=int(keep.numel()),
        first_frame_quota=int(first_quota),
        query_token_count=int(mask.sum().item()),
        num_frames=t,
        grid_h=int(grid_h),
        grid_w=int(grid_w),
    )


def apply_echoprune_plan(tensor: torch.Tensor, plan: EchoPrunePlan) -> torch.Tensor:
    if tensor.ndim < 1:
        raise ValueError(f"Expected tensor with token dimension, got {tuple(tensor.shape)}")
    if int(tensor.shape[0]) != int(plan.dense_token_count):
        raise ValueError(
            "Tensor rows do not match EchoPrune plan: "
            f"expected={plan.dense_token_count} actual={tensor.shape[0]}"
        )
    return tensor.index_select(0, plan.keep_flat_indices.to(tensor.device))


def apply_echoprune_plan_to_deepstack(
    deepstack_features: Iterable[torch.Tensor],
    plan: EchoPrunePlan,
) -> list[torch.Tensor]:
    return [apply_echoprune_plan(layer, plan) for layer in deepstack_features]


def build_echoprune_query_text(
    annotation: Mapping[str, Any] | None = None,
    messages: Sequence[Mapping[str, Any]] | None = None,
    *,
    query_source: str = "question_options",
) -> str:
    """Build EchoPrune semantic query text without ground-truth answer fields."""
    query_source = (query_source or "question_options").strip().lower()
    if query_source not in {"question_options", "user_text", "all_text"}:
        raise ValueError(f"Unsupported EchoPrune query_source={query_source!r}")

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
            label = chr(ord("A") + idx)
            choice_lines.append(f"({label}) {value}")

    parts = []
    if question:
        parts.append(f"Question:\n{question}")
    if choice_lines:
        parts.append("Options:\n" + "\n".join(choice_lines))
    query = "\n\n".join(parts).strip()
    if not query and messages is not None:
        query = message_text({"user"})
    return query


def verify_echoprune_lengths(
    *,
    plan: EchoPrunePlan,
    placeholder_count: int,
    embedding_rows: int,
    mrope_count: int,
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
            "[EchoPrune-vLLM] invariant failed for "
            f"video[{video_index}]: placeholders={placeholder_count}, embeds={embedding_rows}, "
            f"mrope={mrope_count}, retained={plan.retained_token_count}, expected={expected}, "
            f"dense={plan.dense_token_count}, query_tokens={plan.query_token_count}"
        )
    if dense_prompt_length is not None and compact_prompt_length is not None:
        expected_delta = int(plan.dense_token_count) - expected
        actual_delta = int(dense_prompt_length) - int(compact_prompt_length)
        if actual_delta != expected_delta:
            raise RuntimeError(
                "[EchoPrune-vLLM] prompt length invariant failed for "
                f"video[{video_index}]: dense_prompt={dense_prompt_length}, "
                f"compact_prompt={compact_prompt_length}, delta={actual_delta}, "
                f"expected_delta={expected_delta}"
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


def _target_count_for_row(
    row: Any,
    spatial_merge_size: int,
    retain_ratio: float,
    target_visual_tokens: int | None = None,
) -> int:
    t, _, _, tokens_per_frame = _tokens_per_frame_for_row(row, spatial_merge_size)
    total = int(t * tokens_per_frame)
    return _budget_from_inputs(
        total,
        retain_ratio=float(retain_ratio),
        target_visual_tokens=target_visual_tokens,
    )


def _selected_video_sizes(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    retain_ratio: float,
    target_visual_tokens: int | None,
) -> list[int]:
    return [
        _target_count_for_row(row, spatial_merge_size, retain_ratio, target_visual_tokens)
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
    """Return local media M-RoPE positions shaped [T*H*W, 4]."""
    ti = torch.arange(t, device=device, dtype=torch.long).view(-1, 1).expand(-1, dense_h * dense_w).flatten()
    hi = torch.arange(dense_h, device=device, dtype=torch.long).view(1, -1, 1).expand(t, -1, dense_w).flatten()
    wi = torch.arange(dense_w, device=device, dtype=torch.long).view(1, 1, -1).expand(t, dense_h, -1).flatten()
    width = torch.full_like(ti, int(dense_w))
    return torch.stack([ti, hi, wi, width], dim=1).to(dtype=dtype)


def _enabled_mode() -> str:
    return os.environ.get("QWEN3VL_ECHOPRUNE_MODE", "none").strip().lower()


def _retain_ratio() -> float:
    return float(os.environ.get("QWEN3VL_ECHOPRUNE_RETAIN_RATIO", "0.20"))


def _target_visual_tokens() -> int | None:
    raw = os.environ.get("QWEN3VL_ECHOPRUNE_TARGET_VISUAL_TOKENS", "").strip()
    if not raw or raw.lower() in {"none", "null", "0"}:
        return None
    return int(raw)


def _temperature() -> float:
    return float(os.environ.get("QWEN3VL_ECHOPRUNE_TEMPERATURE", "0.50"))


def _match_scope() -> str:
    return os.environ.get("QWEN3VL_ECHOPRUNE_MATCH_SCOPE", "full").strip().lower()


def _window_size() -> int:
    return int(os.environ.get("QWEN3VL_ECHOPRUNE_WINDOW_SIZE", "3"))


def _first_frame_policy() -> str:
    return os.environ.get("QWEN3VL_ECHOPRUNE_FIRST_FRAME_POLICY", "paper").strip().lower()


def _query_source() -> str:
    return os.environ.get("QWEN3VL_ECHOPRUNE_QUERY_SOURCE", "question_options").strip().lower()


def _match_chunk_size() -> int:
    return int(os.environ.get("QWEN3VL_ECHOPRUNE_MATCH_CHUNK_SIZE", "256"))


def _debug_verify() -> bool:
    return os.environ.get("QWEN3VL_ECHOPRUNE_DEBUG_VERIFY", "0") == "1"


def _verbose() -> bool:
    return os.environ.get("QWEN3VL_ECHOPRUNE_QUIET", "0") != "1"


def _validate_patch_params(
    *,
    retain_ratio: float,
    target_visual_tokens: int | None,
    temperature: float,
    match_scope: str,
    window_size: int,
    first_frame_policy: str,
    query_source: str,
    match_chunk_size: int,
) -> None:
    if not math.isfinite(float(retain_ratio)) or float(retain_ratio) <= 0.0 or float(retain_ratio) > 1.0:
        raise ValueError(f"retain_ratio must be in (0,1], got {retain_ratio!r}")
    if target_visual_tokens is not None and int(target_visual_tokens) <= 0:
        raise ValueError(f"target_visual_tokens must be positive, got {target_visual_tokens!r}")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
    if match_scope not in {"full", "local"}:
        raise ValueError(f"match_scope must be full/local, got {match_scope!r}")
    if int(window_size) <= 0 or int(window_size) % 2 != 1:
        raise ValueError(f"window_size must be a positive odd integer, got {window_size!r}")
    if first_frame_policy not in {"paper", "global"}:
        raise ValueError(f"first_frame_policy must be paper/global, got {first_frame_policy!r}")
    if query_source not in {"question_options", "user_text", "all_text"}:
        raise ValueError(f"query_source must be question_options/user_text/all_text, got {query_source!r}")
    if int(match_chunk_size) <= 0:
        raise ValueError(f"match_chunk_size must be > 0, got {match_chunk_size!r}")


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
        "Qwen3VLMultiModalProcessor._call_hf_processor": hasattr(processor_cls, "_call_hf_processor") if processor_cls else False,
        "Qwen3VLMultiModalProcessor._get_mm_fields_config": hasattr(processor_cls, "_get_mm_fields_config") if processor_cls else False,
        "Qwen3VLMultiModalProcessor._get_prompt_updates": hasattr(processor_cls, "_get_prompt_updates") if processor_cls else False,
        "Qwen3VLForConditionalGeneration._parse_and_validate_video_input": hasattr(model_cls, "_parse_and_validate_video_input") if model_cls else False,
        "Qwen3VLForConditionalGeneration._process_video_input": hasattr(model_cls, "_process_video_input") if model_cls else False,
        "Qwen3VLForConditionalGeneration.embed_input_ids": hasattr(model_cls, "embed_input_ids") if model_cls else False,
        "Qwen3VLForConditionalGeneration.get_language_model": hasattr(model_cls, "get_language_model") if model_cls else False,
        "Qwen3_VisionTransformer.forward": hasattr(visual_cls, "forward") if visual_cls else False,
        "iter_mm_grid_hw": hasattr(model_cls, "iter_mm_grid_hw") if model_cls else False,
    }


def _pop_echoprune_query_texts(mm_kwargs: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    clean = dict(mm_kwargs or {})
    raw = clean.pop("echoprune_query_texts", None)
    if raw is None:
        raw = clean.pop("echoprune_query_text", None)
    if raw is None:
        return clean, []
    if isinstance(raw, str):
        return clean, [raw]
    return clean, [str(x) for x in raw]


def _tokenize_query_texts(tokenizer: Any, query_texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[list[int]] = []
    for text in query_texts:
        if not str(text).strip():
            raise ValueError("[EchoPrune-vLLM] Empty echoprune_query_text")
        ids = tokenizer.encode(str(text), add_special_tokens=False)
        if not ids:
            raise ValueError("[EchoPrune-vLLM] Query text tokenized to zero tokens")
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
        raise ValueError("[EchoPrune-vLLM] Missing query token ids/lengths for video pruning")
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
            f"[EchoPrune-vLLM] Query rows do not match videos: ids={tuple(ids.shape)} "
            f"lengths={tuple(lens.shape)} videos={n_videos}"
        )
    rows: list[torch.LongTensor] = []
    for idx in range(n_videos):
        length = int(lens[idx].item())
        if length <= 0:
            raise ValueError(f"[EchoPrune-vLLM] Query length for video[{idx}] is {length}")
        rows.append(ids[idx, :length].contiguous())
    return rows


def _embed_query_rows(model: Any, query_rows: Sequence[torch.LongTensor]) -> list[torch.Tensor]:
    embeds: list[torch.Tensor] = []
    language_model = model.get_language_model() if hasattr(model, "get_language_model") else getattr(model, "language_model", None)
    if language_model is None or not hasattr(language_model, "embed_input_ids"):
        raise AttributeError("[EchoPrune-vLLM] Could not find language_model.embed_input_ids")
    for row in query_rows:
        embeds.append(language_model.embed_input_ids(row))
    return embeds


def _compress_video_outputs(
    main: torch.Tensor,
    deepstack: Sequence[torch.Tensor],
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    query_embeds: Sequence[torch.Tensor] | None,
    debug_verify: bool,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    offset = 0
    for video_idx, row in enumerate(grid_thw):
        t, dense_h, dense_w, tokens_per_frame = _tokens_per_frame_for_row(row, spatial_merge_size)
        dense_tokens = int(t * tokens_per_frame)
        target = _target_count_for_row(
            row,
            spatial_merge_size,
            _retain_ratio(),
            _target_visual_tokens(),
        )
        main_chunk = main[offset : offset + dense_tokens]
        deep_chunks = [layer[offset : offset + dense_tokens] for layer in deepstack]
        plan: EchoPrunePlan | None = None
        if query_embeds is None:
            # vLLM profiles maximum multimodal encoder memory during engine
            # startup using dummy video inputs before request-local query fields
            # exist.  Return a compact tensor with the exact budget so profiling
            # succeeds, but keep real request pruning query-dependent.
            selected = torch.arange(target, device=main.device, dtype=torch.long)
        else:
            query = query_embeds[video_idx]
            plan = build_echoprune_plan(
                main_chunk.reshape(t, tokens_per_frame, -1),
                query,
                target_visual_tokens=target,
                retain_ratio=None,
                temperature=_temperature(),
                match_scope=_match_scope(),
                grid_h=dense_h,
                grid_w=dense_w,
                window_size=_window_size(),
                first_frame_policy=_first_frame_policy(),
                chunk_size=_match_chunk_size(),
                debug=debug_verify,
            )
            selected = plan.keep_flat_indices.to(main.device)
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

        if plan is not None:
            try:
                from vllm_qwen3_vl_index_dump import dump_video_selection

                dump_video_selection(
                    method="echoprune",
                    video_index=video_idx,
                    grid_thw=row,
                    spatial_merge_size=spatial_merge_size,
                    dense_token_count=dense_tokens,
                    keep_indices=plan.keep_flat_indices,
                    output_indices=plan.keep_flat_indices,
                    num_tokens_per_frame=plan.num_tokens_per_frame,
                    extra={
                        "retain_ratio": float(_retain_ratio()),
                        "target_visual_tokens": _target_visual_tokens(),
                        "temperature": float(_temperature()),
                        "match_scope": str(_match_scope()),
                        "window_size": int(_window_size()),
                        "first_frame_policy": str(_first_frame_policy()),
                        "query_source": str(_query_source()),
                        "query_token_count": int(plan.query_token_count),
                    },
                )
            except Exception as exc:
                if os.environ.get("QWEN3VL_INDEX_DUMP_STRICT", "0") == "1":
                    raise
                if os.environ.get("QWEN3VL_INDEX_DUMP_DIR"):
                    print(f"[EchoPrune-vLLM] index dump failed for video[{video_idx}]: {exc}")

        if _verbose() or debug_verify:
            if plan is None:
                reduction = 1.0 - (int(target) / max(dense_tokens, 1))
                print(
                    f"[EchoPrune-vLLM] profile_fallback video[{video_idx}] "
                    f"frames={t} grid={dense_h}x{dense_w} tokens={dense_tokens}->{target} "
                    f"reduction={reduction:.2%}"
                )
            else:
                reduction = 1.0 - (plan.retained_token_count / max(plan.dense_token_count, 1))
                per_frame = "" if not debug_verify else f" per_frame={plan.num_tokens_per_frame}"
                print(
                    f"[EchoPrune-vLLM] video[{video_idx}] frames={t} grid={dense_h}x{dense_w} "
                    f"query_tokens={plan.query_token_count} scope={_match_scope()} tau={_temperature():.4f} "
                    f"tokens={plan.dense_token_count}->{plan.retained_token_count} "
                    f"reduction={reduction:.2%}{per_frame}"
                )
        if debug_verify:
            if plan is not None:
                verify_echoprune_lengths(
                    plan=plan,
                    placeholder_count=target,
                    embedding_rows=out.shape[0],
                    mrope_count=selected_pos.shape[0],
                    video_index=video_idx,
                )
            elif out.shape[0] != target or selected_pos.shape[0] != target:
                raise RuntimeError(
                    "[EchoPrune-vLLM] profile fallback invariant failed for "
                    f"video[{video_idx}]: embeds={out.shape[0]}, mrope={selected_pos.shape[0]}, "
                    f"expected={target}, dense={dense_tokens}"
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


def apply_patch(
    mode: str = "post_vit",
    retain_ratio: float = 0.20,
    target_visual_tokens: int | None = None,
    temperature: float = 0.50,
    match_scope: str = "full",
    window_size: int = 3,
    first_frame_policy: str = "paper",
    query_source: str = "question_options",
    match_chunk_size: int = 256,
    debug_verify: bool = False,
) -> None:
    """Patch local vLLM Qwen3-VL classes in the current process."""
    global _PATCHED
    mode = (mode or "none").strip().lower()
    if mode in {"none", "off", ""}:
        return
    if mode != "post_vit":
        raise ValueError(f"Unsupported EchoPrune mode for vLLM: {mode}")
    match_scope = (match_scope or "full").strip().lower()
    first_frame_policy = (first_frame_policy or "paper").strip().lower()
    query_source = (query_source or "question_options").strip().lower()
    _validate_patch_params(
        retain_ratio=float(retain_ratio),
        target_visual_tokens=target_visual_tokens,
        temperature=float(temperature),
        match_scope=match_scope,
        window_size=int(window_size),
        first_frame_policy=first_frame_policy,
        query_source=query_source,
        match_chunk_size=int(match_chunk_size),
    )

    os.environ["QWEN3VL_ECHOPRUNE_MODE"] = mode
    os.environ["QWEN3VL_ECHOPRUNE_RETAIN_RATIO"] = str(float(retain_ratio))
    os.environ["QWEN3VL_ECHOPRUNE_TARGET_VISUAL_TOKENS"] = (
        "" if target_visual_tokens is None else str(int(target_visual_tokens))
    )
    os.environ["QWEN3VL_ECHOPRUNE_TEMPERATURE"] = str(float(temperature))
    os.environ["QWEN3VL_ECHOPRUNE_MATCH_SCOPE"] = match_scope
    os.environ["QWEN3VL_ECHOPRUNE_WINDOW_SIZE"] = str(int(window_size))
    os.environ["QWEN3VL_ECHOPRUNE_FIRST_FRAME_POLICY"] = first_frame_policy
    os.environ["QWEN3VL_ECHOPRUNE_QUERY_SOURCE"] = query_source
    os.environ["QWEN3VL_ECHOPRUNE_MATCH_CHUNK_SIZE"] = str(int(match_chunk_size))
    os.environ["QWEN3VL_ECHOPRUNE_DEBUG_VERIFY"] = "1" if debug_verify else "0"

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
        "get_mrope_input_positions",
        "iter_mm_grid_hw",
    ):
        if not required.get(name, False):
            raise AttributeError(f"[EchoPrune-vLLM] Unsupported local Qwen3-VL API; missing {name}")

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
        clean_mm_kwargs, query_texts = _pop_echoprune_query_texts(mm_kwargs)
        outputs = orig_call_hf_processor(self, prompt, mm_data, clean_mm_kwargs, tok_kwargs)
        if _enabled_mode() == "post_vit" and query_texts:
            grid = outputs.get("video_grid_thw")
            n_videos = int(grid.shape[0]) if isinstance(grid, torch.Tensor) and grid.ndim == 2 else 0
            if n_videos == 0:
                return outputs
            if not query_texts:
                raise ValueError("[EchoPrune-vLLM] echoprune_query_texts is required for video requests")
            if len(query_texts) == 1 and n_videos > 1:
                query_texts = query_texts * n_videos
            if len(query_texts) != n_videos:
                raise ValueError(
                    f"[EchoPrune-vLLM] query/video count mismatch: queries={len(query_texts)} videos={n_videos}"
                )
            token_ids, lengths = _tokenize_query_texts(self.info.get_tokenizer(), query_texts)
            outputs["echoprune_query_token_ids"] = token_ids
            outputs["echoprune_query_lengths"] = lengths
            if _debug_verify():
                print(
                    f"[EchoPrune-vLLM] processor query fields videos={n_videos} "
                    f"query_rows={len(query_texts)} max_query_len={token_ids.shape[1]}"
                )
        return outputs

    def patched_get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        clean_mm_kwargs, _ = _pop_echoprune_query_texts(hf_processor_mm_kwargs)
        config = dict(orig_get_mm_fields_config(self, hf_inputs, clean_mm_kwargs))
        if "echoprune_query_token_ids" in hf_inputs:
            config["echoprune_query_token_ids"] = qwen3_vl.MultiModalFieldConfig.batched("video")
        if "echoprune_query_lengths" in hf_inputs:
            config["echoprune_query_lengths"] = qwen3_vl.MultiModalFieldConfig.batched("video")
        return config

    def patched_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        if _enabled_mode() != "post_vit":
            return orig_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
        clean_mm_kwargs, _ = _pop_echoprune_query_texts(hf_processor_mm_kwargs)
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
            num_tokens = int(grid_thw.prod()) // merge_length
            return [hf_processor.image_token_id] * num_tokens

        def get_video_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            count = _target_count_for_row(
                grid_thw,
                int(image_processor.merge_size),
                _retain_ratio(),
                _target_visual_tokens(),
            )
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

    def patched_vision_forward(self, x, grid_thw, echoprune_query_embeds=None):
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

        main = self.merger(hidden_states)
        return _compress_video_outputs(
            main,
            deepstack_feature_lists,
            grid_tensor,
            int(self.spatial_merge_size),
            echoprune_query_embeds,
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
                raise RuntimeError("EchoPrune vLLM patch does not support mm_encoder_tp_mode=data yet.")
            image_embeds = _ORIGINALS["Qwen3_VisionTransformer.forward"](
                self.visual,
                pixel_values,
                grid_thw,
            )
            if image_embeds.shape[1] == self.visual.out_hidden_size + 4:
                return image_embeds.split((grid_thw.prod(-1) // self.visual.spatial_merge_size // self.visual.spatial_merge_size).tolist())
        return _append_dense_image_positions(
            image_embeds,
            grid_thw,
            int(self.visual.spatial_merge_size),
        )

    def patched_parse_video_input(self, **kwargs):
        if _enabled_mode() != "post_vit":
            return orig_parse_video_input(self, **kwargs)
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)
        query_token_ids = kwargs.pop("echoprune_query_token_ids", None)
        query_lengths = kwargs.pop("echoprune_query_lengths", None)
        if pixel_values_videos is None and video_embeds is None:
            return None
        return {
            "type": "pixel_values_videos" if pixel_values_videos is not None else "video_embeds",
            "pixel_values_videos": pixel_values_videos,
            "video_embeds": video_embeds,
            "video_grid_thw": video_grid_thw,
            "second_per_grid_ts": second_per_grid_ts,
            "echoprune_query_token_ids": query_token_ids,
            "echoprune_query_lengths": query_lengths,
        }

    def patched_process_video_input(self, video_input):
        if _enabled_mode() != "post_vit":
            return orig_process_video_input(self, video_input)
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        if video_input["type"] == "video_embeds":
            raise NotImplementedError("[EchoPrune-vLLM] EchoPrune requires raw pixel_values_videos to compute query-dependent plans.")
        pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
        if self.use_data_parallel:
            raise RuntimeError("EchoPrune vLLM patch does not support mm_encoder_tp_mode=data yet.")
        query_token_ids = video_input.get("echoprune_query_token_ids")
        query_lengths = video_input.get("echoprune_query_lengths")
        if query_token_ids is None or query_lengths is None:
            query_embeds = None
        else:
            query_rows = _query_rows(
                query_token_ids,
                query_lengths,
                int(grid_thw.shape[0]),
                device=pixel_values_videos.device,
            )
            query_embeds = _embed_query_rows(self, query_rows)
        video_embeds = self.visual(
            pixel_values_videos,
            grid_thw=grid_thw,
            echoprune_query_embeds=query_embeds,
        )
        sizes = _selected_video_sizes(
            grid_thw,
            int(self.visual.spatial_merge_size),
            _retain_ratio(),
            _target_visual_tokens(),
        )
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
                count = _target_count_for_row(grid, spatial_merge_size, _retain_ratio(), _target_visual_tokens())
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
        if _debug_verify():
            for idx, (emb, pos) in enumerate(zip(mm_embeddings_out, mm_positions)):
                if emb.shape[0] != pos.shape[1]:
                    raise RuntimeError(
                        f"[EchoPrune-vLLM] recompute invariant failed for mm[{idx}]: "
                        f"embeds={emb.shape[0]} positions={pos.shape[1]}"
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
            f"[EchoPrune-vLLM] enabled mode={mode} retain_ratio={float(retain_ratio):.4f} "
            f"target_visual_tokens={target_visual_tokens} temperature={float(temperature):.4f} "
            f"scope={match_scope} window={int(window_size)} first_frame={first_frame_policy} "
            f"query_source={query_source} chunk={int(match_chunk_size)}"
        )
        print(f"[EchoPrune-vLLM] local_api={required}")
