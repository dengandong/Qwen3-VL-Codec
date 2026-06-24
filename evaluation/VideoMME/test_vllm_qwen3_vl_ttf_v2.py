from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vllm_qwen3_vl_ttf_v2 import (  # noqa: E402
    _select_dynamic_anchor_indices,
    apply_ttf_plan_to_flat_embeddings,
    build_ttf_plan,
    build_ttf_plans_for_videos,
    gather_dense_mrope_positions,
    verify_ttf_lengths,
)


def test_dynamic_anchor_uses_local_temporal_window() -> None:
    x = torch.zeros(5, 1, 1, 3)
    x[0, 0, 0, 0] = 1.0
    x[1, 0, 0, 0] = 1.0
    x[2, 0, 0, 1] = 1.0
    x[3, 0, 0, 1] = 1.0
    x[4, 0, 0, 1] = 1.0

    anchors = _select_dynamic_anchor_indices(x, temporal_anchor_radius=2)
    assert anchors.tolist() == [1, 0, 3, 2, 2]


def test_dynamic_anchor_excludes_self_and_respects_radius() -> None:
    x = torch.zeros(4, 1, 1, 3)
    x[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
    x[1, 0, 0] = torch.tensor([0.0, 1.0, 0.0])
    x[2, 0, 0] = torch.tensor([0.0, 1.0, 0.0])
    x[3, 0, 0] = torch.tensor([1.0, 0.0, 0.0])

    anchors = _select_dynamic_anchor_indices(x, temporal_anchor_radius=1)
    # Frame 0 cannot use the globally similar frame 3 because it is outside
    # the local temporal window.
    assert anchors.tolist() == [1, 2, 1, 2]


def test_retain_ratio_budget_can_keep_less_than_one_frame() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 2, 4, 8)
    plan = build_ttf_plan(
        x,
        budget_mode="retain_ratio",
        retain_ratio=0.125,
        temporal_anchor_radius=2,
    )
    assert plan.original_token_count == 32
    assert plan.retained_token_count == 4
    assert sum(plan.num_tokens_per_original_frame) == 4
    assert max(plan.num_tokens_per_original_frame) <= 4


def test_output_order_is_chronological_not_score_order() -> None:
    torch.manual_seed(1)
    x = torch.randn(5, 2, 2, 6)
    plan = build_ttf_plan(
        x,
        budget_mode="retain_ratio",
        retain_ratio=0.4,
        temporal_anchor_radius=2,
    )
    indices = plan.output_flat_indices.tolist()
    assert indices == sorted(indices)
    assert plan.output_frame_order == [0, 1, 2, 3, 4]


def test_mrope_and_embedding_gather_share_v2_indices() -> None:
    torch.manual_seed(2)
    x = torch.randn(3, 2, 2, 5)
    plan = build_ttf_plan(
        x,
        budget_mode="retain_ratio",
        retain_ratio=0.5,
        temporal_anchor_radius=2,
    )
    flat = x.reshape(-1, 5)
    dense_pos = torch.arange(3 * flat.shape[0]).reshape(3, flat.shape[0])
    out = apply_ttf_plan_to_flat_embeddings(flat, plan)
    sparse_pos = gather_dense_mrope_positions(dense_pos, plan)
    assert torch.equal(out, flat.index_select(0, plan.output_flat_indices))
    verify_ttf_lengths(
        plan=plan,
        placeholder_count=plan.retained_token_count,
        embedding_rows=out.shape[0],
        mrope_count=sparse_pos.shape[1],
    )


def test_single_frame_identity() -> None:
    x = torch.randn(1, 2, 3, 4)
    plan = build_ttf_plan(
        x,
        budget_mode="retain_ratio",
        retain_ratio=0.125,
        temporal_anchor_radius=2,
    )
    assert plan.retained_token_count == 6
    assert plan.output_flat_indices.tolist() == list(range(6))
    assert plan.anchor_indices_per_frame.tolist() == [0]


def test_independent_videos_have_independent_dynamic_anchors() -> None:
    x0 = torch.randn(4, 1, 2, 5)
    x1 = torch.randn(3, 2, 2, 5)
    plans = build_ttf_plans_for_videos(
        [x0, x1],
        budget_mode="retain_ratio",
        retain_ratio=0.5,
        temporal_anchor_radius=2,
    )
    assert len(plans) == 2
    assert plans[0].anchor_indices_per_frame.shape == (4,)
    assert plans[1].anchor_indices_per_frame.shape == (3,)
    assert plans[0].output_flat_indices.max().item() < plans[0].original_token_count
    assert plans[1].output_flat_indices.max().item() < plans[1].original_token_count
