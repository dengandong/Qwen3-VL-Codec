from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vllm_qwen3_vl_ttf import (  # noqa: E402
    _anchor_neighborhood_indices,
    apply_ttf_plan_to_deepstack,
    apply_ttf_plan_to_flat_embeddings,
    build_ttf_plan,
    build_ttf_plans_for_videos,
    gather_dense_mrope_positions,
    verify_ttf_lengths,
)


def _basis_video(t: int, h: int, w: int) -> torch.Tensor:
    n = h * w
    eye = torch.eye(n)
    return eye.reshape(1, h, w, n).repeat(t, 1, 1, 1)


def test_identical_frames_keep_only_anchor() -> None:
    torch.manual_seed(0)
    frame = torch.randn(2, 3, 5)
    x = frame.unsqueeze(0).repeat(4, 1, 1, 1)
    plan = build_ttf_plan(x, threshold=0.70, anchor="auto")
    assert plan.anchor_idx == 0
    assert plan.retained_token_count == 6
    assert plan.num_tokens_per_original_frame == [6, 0, 0, 0]
    assert plan.output_coords[:, 0].tolist() == [0] * 6


def test_dissimilar_source_tokens_are_retained() -> None:
    x = torch.zeros(2, 1, 2, 2)
    x[0, :, :, 0] = 1.0
    x[1, :, :, 1] = 1.0
    plan = build_ttf_plan(x, threshold=0.70, anchor="first", window_radius=1)
    assert plan.retained_token_count == 4
    assert torch.all(plan.best_similarity[1] < 0.70)


def test_one_cell_spatial_shift_needs_radius_one() -> None:
    x = _basis_video(2, 3, 3)
    # Interior token at (1, 1) in source matches anchor token at (1, 0).
    x[1, 1, 1] = x[0, 1, 0]
    plan_r0 = build_ttf_plan(x, threshold=0.99, anchor="first", window_radius=0)
    plan_r1 = build_ttf_plan(x, threshold=0.99, anchor="first", window_radius=1)
    source_center_flat = 1 * 9 + 1 * 3 + 1
    assert source_center_flat in plan_r0.output_flat_indices.tolist()
    assert source_center_flat not in plan_r1.output_flat_indices.tolist()
    assert plan_r1.retained_token_count < plan_r0.retained_token_count


def test_border_clipping_deduplicates_candidates() -> None:
    cand, mask = _anchor_neighborhood_indices(2, 2, 1, device=torch.device("cpu"))
    top_left_unique = cand[0][mask[0]].tolist()
    assert sorted(top_left_unique) == [0, 1, 2, 3]
    assert len(top_left_unique) == len(set(top_left_unique))

    x = torch.zeros(2, 2, 2, 4)
    x[0] = torch.eye(4).reshape(2, 2, 4)
    x[1, 0, 0] = x[0, 0, 0]
    plan = build_ttf_plan(x, threshold=0.99, anchor="first", window_radius=1)
    assert int(plan.matched_anchor_indices[1, 0].item()) == 0
    assert int(plan.best_similarity[1, 0].item()) == 1


def test_non_square_grid_coords_are_correct() -> None:
    x = torch.randn(2, 2, 5, 4)
    plan = build_ttf_plan(x, threshold=2.0, anchor="first", order="temporal")
    assert plan.retained_token_count == 20
    assert plan.output_coords.shape == (20, 3)
    assert plan.output_coords[:, 1].max().item() == 1
    assert plan.output_coords[:, 2].max().item() == 4


def test_single_frame_identity() -> None:
    x = torch.randn(1, 3, 2, 4)
    plan = build_ttf_plan(x, threshold=-1.0, anchor="last")
    out = apply_ttf_plan_to_flat_embeddings(x.reshape(-1, 4), plan)
    assert plan.anchor_idx == 0
    assert plan.retained_token_count == 6
    assert plan.output_flat_indices.tolist() == list(range(6))
    assert torch.equal(out, x.reshape(-1, 4))


def test_auto_anchor_tie_picks_first() -> None:
    x = torch.ones(3, 1, 1, 4)
    plan = build_ttf_plan(x, anchor="auto")
    assert plan.anchor_idx == 0


def test_threshold_monotonicity() -> None:
    torch.manual_seed(1)
    x = torch.randn(4, 2, 2, 8)
    low = build_ttf_plan(x, threshold=-0.5)
    high = build_ttf_plan(x, threshold=0.95)
    assert low.retained_token_count <= high.retained_token_count


def test_retain_ratio_budget_keeps_exact_target_count() -> None:
    torch.manual_seed(10)
    x = torch.randn(4, 2, 3, 8)
    plan = build_ttf_plan(
        x,
        budget_mode="retain_ratio",
        retain_ratio=0.5,
        anchor="first",
    )
    assert plan.original_token_count == 24
    assert plan.retained_token_count == 12
    assert plan.num_tokens_per_original_frame[0] == 6


def test_retain_ratio_budget_keeps_lowest_similarity_sources() -> None:
    x = torch.zeros(2, 1, 4, 5)
    x[0] = torch.eye(5)[:4].reshape(1, 4, 5)
    # Source tokens 0/1 match anchor and should be deleted first. Tokens 2/3
    # are orthogonal to all matching same-location anchors and should be kept.
    x[1, 0, 0] = x[0, 0, 0]
    x[1, 0, 1] = x[0, 0, 1]
    x[1, 0, 2, 4] = 1.0
    x[1, 0, 3, 4] = 1.0
    plan = build_ttf_plan(
        x,
        budget_mode="retain_ratio",
        retain_ratio=0.75,
        anchor="first",
        window_radius=0,
    )
    kept = set(plan.output_flat_indices.tolist())
    assert plan.retained_token_count == 6
    assert 6 in kept
    assert 7 in kept
    assert 4 not in kept
    assert 5 not in kept


def test_anchor_tokens_always_retained() -> None:
    x = torch.randn(3, 2, 2, 5)
    plan = build_ttf_plan(x, threshold=-1.0, anchor="last")
    anchor_indices = set(range(2 * 4, 3 * 4))
    assert anchor_indices.issubset(set(plan.output_flat_indices.tolist()))


def test_no_feature_averaging() -> None:
    torch.manual_seed(2)
    x = torch.randn(3, 2, 2, 6)
    flat = x.reshape(-1, 6)
    plan = build_ttf_plan(x, threshold=0.2)
    out = apply_ttf_plan_to_flat_embeddings(flat, plan)
    expected = flat.index_select(0, plan.output_flat_indices)
    assert torch.equal(out, expected)


def test_no_input_mutation() -> None:
    x = torch.randn(3, 2, 2, 5)
    before = x.clone()
    _ = build_ttf_plan(x, threshold=0.7)
    assert torch.equal(x, before)


def test_paper_order_moves_anchor_first_without_reindexing_time() -> None:
    x = torch.randn(3, 2, 2, 5)
    plan = build_ttf_plan(x, threshold=2.0, anchor="last", order="paper")
    assert plan.output_frame_order == [2, 0, 1]
    assert plan.output_coords[:4, 0].tolist() == [2, 2, 2, 2]
    assert plan.output_coords[4:8, 0].tolist() == [0, 0, 0, 0]


def test_temporal_order_keeps_original_time_order() -> None:
    x = torch.randn(3, 2, 2, 5)
    plan = build_ttf_plan(x, threshold=2.0, anchor="last", order="temporal")
    assert plan.output_frame_order == [0, 1, 2]
    assert plan.output_coords[:4, 0].tolist() == [0, 0, 0, 0]
    assert plan.output_coords[-4:, 0].tolist() == [2, 2, 2, 2]


def test_deepstack_alignment_uses_same_indices() -> None:
    x = _basis_video(2, 2, 2)
    plan = build_ttf_plan(x, threshold=-1.0, anchor="first")
    main = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    deep = [
        torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2),
        torch.arange(100, 116, dtype=torch.float32).reshape(8, 2),
    ]
    main_out = apply_ttf_plan_to_flat_embeddings(main, plan)
    deep_out = apply_ttf_plan_to_deepstack(deep, plan)
    assert main_out.shape[0] == plan.retained_token_count
    assert all(layer.shape[0] == plan.retained_token_count for layer in deep_out)
    assert torch.equal(deep_out[0], deep[0].index_select(0, plan.output_flat_indices))


def test_independent_videos_do_not_share_masks() -> None:
    x0 = _basis_video(2, 1, 2)
    x1 = torch.randn(3, 2, 2, 5)
    plans = build_ttf_plans_for_videos([x0, x1], threshold=0.7, anchor="first")
    assert len(plans) == 2
    assert plans[0].original_token_count == 4
    assert plans[1].original_token_count == 12
    assert plans[0].output_flat_indices.max().item() < 4
    assert plans[1].output_flat_indices.max().item() < 12


def test_baseline_parity_disable_fusion() -> None:
    x = torch.randn(3, 2, 2, 5)
    plan = build_ttf_plan(x, threshold=0.7, anchor="first", disable_fusion=True)
    flat = x.reshape(-1, 5)
    dense_pos = torch.arange(3 * flat.shape[0]).reshape(3, flat.shape[0])
    assert plan.retained_token_count == flat.shape[0]
    assert plan.output_flat_indices.tolist() == list(range(flat.shape[0]))
    assert torch.equal(apply_ttf_plan_to_flat_embeddings(flat, plan), flat)
    assert torch.equal(gather_dense_mrope_positions(dense_pos, plan), dense_pos)


def test_maximum_fusion_mock_invariant() -> None:
    x = _basis_video(4, 2, 2)
    plan = build_ttf_plan(x, threshold=-1.0, anchor="first")
    dense_pos = torch.arange(3 * plan.original_token_count).reshape(3, plan.original_token_count)
    sparse_pos = gather_dense_mrope_positions(dense_pos, plan)
    assert plan.retained_token_count == 4
    verify_ttf_lengths(
        plan=plan,
        placeholder_count=4,
        embedding_rows=4,
        mrope_count=sparse_pos.shape[1],
    )


def test_mixed_image_video_mock_only_video_is_compressed() -> None:
    image_embeds = torch.randn(6, 5)
    video = _basis_video(2, 2, 2)
    plan = build_ttf_plan(video, threshold=-1.0, anchor="first")
    video_embeds = torch.randn(plan.original_token_count, 5)
    compressed_video = apply_ttf_plan_to_flat_embeddings(video_embeds, plan)
    assert image_embeds.shape == (6, 5)
    assert compressed_video.shape[0] == plan.retained_token_count == 4
