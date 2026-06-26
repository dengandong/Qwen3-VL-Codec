from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vllm_qwen3_vl_kitoke import (  # noqa: E402
    KiTokeAlgorithmConfig,
    KiTokeInterval,
    apply_kitoke_plan,
    build_interval_assignments,
    build_merge_weights,
    compress_video_kitoke,
    compute_kernel_density_chunked,
    compute_kernel_density_reference,
    compute_transition_metrics_chunked,
    compute_transition_metrics_reference,
    construct_temporal_intervals,
    diversity_to_inclusion_probabilities,
    pivotal_sample_fixed_size,
    repair_empty_intervals,
    select_global_tokens,
    verify_kitoke_lengths,
)


def test_gaussian_kernel_formula_self_and_raw_scale() -> None:
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    alpha = 25.0
    density = compute_kernel_density_reference(x, kernel_alpha=alpha)
    k01 = torch.exp(torch.tensor(-25.0 / alpha))
    k02 = torch.exp(torch.tensor(-100.0 / alpha))
    expected0 = 1.0 + k01 + k02
    assert torch.allclose(density[0], expected0)
    assert torch.all(density >= 1.0)
    normalized_density = compute_kernel_density_reference(F.normalize(x, dim=-1), kernel_alpha=alpha)
    assert not torch.allclose(density, normalized_density)


def test_kernel_chunked_parity_and_no_mutation_alpha_behavior() -> None:
    torch.manual_seed(0)
    x = torch.randn(7, 4)
    before = x.clone()
    ref = compute_kernel_density_reference(x, kernel_alpha=4.0)
    got = compute_kernel_density_chunked(x, kernel_alpha=4.0, row_chunk_size=3, col_chunk_size=2)
    assert torch.allclose(got, ref, atol=1e-6)
    assert torch.equal(x, before)
    small = compute_kernel_density_reference(x[:2], kernel_alpha=1.0)
    large = compute_kernel_density_reference(x[:2], kernel_alpha=10.0)
    assert large[0] > small[0]


def test_density_global_duplicate_and_outlier_diversity() -> None:
    x = torch.tensor([[1.0, 0.0], [1.0, 0.0], [100.0, 0.0]])
    density = compute_kernel_density_reference(x, kernel_alpha=10.0)
    diversity = 1.0 / density
    assert density[0] > density[2]
    assert diversity[2] > diversity[0]


def test_budget_floor_identity_and_compression_count() -> None:
    x = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)
    cfg = KiTokeAlgorithmConfig(retain_ratio=0.25, selection_method="topk", empty_interval_policy="coarsen_then_repair")
    result = compress_video_kitoke(x, grid_h=1, grid_w=2, config=cfg, stable_video_identity="a")
    assert result.plan.target_token_count == 2
    assert result.compressed_main_embeddings.shape[0] == 2
    full = compress_video_kitoke(
        x,
        grid_h=1,
        grid_w=2,
        config=KiTokeAlgorithmConfig(retain_ratio=1.0),
        stable_video_identity="a",
    )
    assert torch.allclose(full.compressed_main_embeddings, x.reshape(10, 3))
    assert full.plan.selected_flat_indices.tolist() == list(range(10))


def test_inclusion_probabilities_equal_saturation_and_invariants() -> None:
    pi = diversity_to_inclusion_probabilities(torch.ones(5), 2)
    assert torch.allclose(pi, torch.full((5,), 0.4, dtype=torch.float64))
    skew = diversity_to_inclusion_probabilities(torch.tensor([100.0, 1.0, 1.0, 1.0]), 2)
    assert torch.allclose(skew.sum(), torch.tensor(2.0, dtype=torch.float64), atol=1e-8)
    assert skew[0] == 1.0
    assert torch.all((skew >= 0) & (skew <= 1))
    with pytest.raises(ValueError):
        diversity_to_inclusion_probabilities(torch.tensor([1.0, float("nan")]), 1)


def test_pivotal_fixed_size_seed_rng_and_pi_zeros_ones() -> None:
    pi = torch.tensor([1.0, 0.0, 0.3, 0.7, 0.4, 0.6], dtype=torch.float64)
    torch.manual_seed(123)
    before = torch.rand(4)
    torch.manual_seed(123)
    gen = torch.Generator().manual_seed(9)
    selected1 = pivotal_sample_fixed_size(pi, generator=gen, pairing="random_rounds")
    after = torch.rand(4)
    torch.manual_seed(123)
    assert torch.allclose(after, torch.rand(4))
    assert selected1.numel() == 3
    assert 0 in selected1.tolist()
    assert 1 not in selected1.tolist()
    gen = torch.Generator().manual_seed(9)
    selected2 = pivotal_sample_fixed_size(pi, generator=gen, pairing="random_rounds")
    assert torch.equal(selected1, selected2)
    assert selected1.unique().numel() == selected1.numel()
    _ = before  # keeps the intent explicit: global RNG was not consumed by pivotal.


def test_multinomial_topk_and_global_selection_ties() -> None:
    div = torch.tensor([1.0, 5.0, 5.0, 0.1])
    topk, _ = select_global_tokens(div, budget=2, method="topk", seed=1, pivotal_pairing="sequential")
    assert topk.tolist() == [1, 2]
    multi, _ = select_global_tokens(div, budget=3, method="multinomial", seed=1, pivotal_pairing="sequential")
    assert multi.numel() == 3 and multi.unique().numel() == 3


def test_transition_metrics_formula_direction_and_chunked() -> None:
    video = torch.tensor(
        [
            [[0.0, 0.0], [10.0, 0.0]],
            [[10.0, 0.0], [11.0, 0.0]],
            [[10.0, 0.0], [11.0, 0.0]],
        ]
    )
    ref_pos, ref_match = compute_transition_metrics_reference(video)
    got_pos, got_match = compute_transition_metrics_chunked(video, match_chunk_size=1)
    assert torch.allclose(got_pos, ref_pos)
    assert torch.allclose(got_match, ref_match)
    assert torch.allclose(got_pos[0], torch.tensor(5.5))
    assert torch.allclose(got_match[0], torch.tensor(5.0))
    changed_past = video.clone()
    changed_past[0] += 100
    _, changed_match = compute_transition_metrics_chunked(changed_past, match_chunk_size=1)
    assert not torch.allclose(changed_match[0], got_match[0])
    assert torch.allclose(changed_match[1], got_match[1])


def test_interval_boundaries_strict_or_and_no_abs_edge_policy() -> None:
    diff_pos = torch.tensor([1.0, 50.0, 1.0, 110.0])
    diff_match = torch.tensor([1.0, 30.0, 1.0, 0.0])
    intervals, boundaries, diff, delta, delta_pct = construct_temporal_intervals(
        diff_pos,
        diff_match,
        tokens_per_frame=2,
        diff_threshold=110.0,
        delta_threshold=70.0,
        relative_delta_threshold=0.4,
        edge_policy="absolute_only",
    )
    assert boundaries == [2]
    assert diff[-1] == 110.0  # strict equality is not a boundary
    assert torch.isneginf(delta[0]) and torch.isneginf(delta[-1])
    assert intervals[0].start_frame == 0 and intervals[-1].end_frame_exclusive == 5


def test_t1_interval_and_contiguous_ranges() -> None:
    intervals, boundaries, *_ = construct_temporal_intervals(
        torch.empty(0),
        torch.empty(0),
        tokens_per_frame=3,
        diff_threshold=1,
        delta_threshold=1,
        relative_delta_threshold=1,
        edge_policy="absolute_only",
    )
    assert boundaries == []
    assert len(intervals) == 1
    assert intervals[0].start_flat_index == 0
    assert intervals[0].end_flat_index_exclusive == 3


def test_empty_interval_paper_strict_repair_swap_and_coarsen_failure() -> None:
    diversity = torch.tensor([1.0, 2.0, 0.1, 0.2, 3.0, 0.5])
    intervals = [
        KiTokeInterval(0, 1, 0, 2, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
        KiTokeInterval(1, 2, 2, 4, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
        KiTokeInterval(2, 3, 4, 6, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
    ]
    selected = torch.tensor([1, 4, 5])
    with pytest.raises(RuntimeError):
        repair_empty_intervals(selected, diversity, intervals, budget=3, policy="paper_strict")
    repaired, repaired_intervals, promoted, demoted, count = repair_empty_intervals(
        selected,
        diversity,
        intervals,
        budget=3,
        policy="repair_swap",
    )
    assert repaired.numel() == 3
    assert count == 1
    assert promoted == [3]
    assert demoted == [5]
    assert all(i.selected_flat_indices.numel() >= 1 for i in repaired_intervals)
    with pytest.raises(RuntimeError):
        repair_empty_intervals(torch.tensor([1, 4]), diversity, intervals, budget=2, policy="repair_swap")


def test_assignment_cosine_no_cross_interval_and_tie() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    intervals = [
        KiTokeInterval(0, 1, 0, 2, torch.tensor([0]), torch.tensor([1])),
        KiTokeInterval(1, 2, 2, 4, torch.tensor([2]), torch.tensor([3])),
    ]
    slots = build_interval_assignments(features, torch.tensor([0, 2]), intervals, match_chunk_size=1)
    assert slots.tolist() == [0, 0, 1, 1]
    # Token 3 is closer to selected 0 globally, but must stay inside interval 1.
    assert int(slots[3]) == 1


def test_weighted_uniform_none_merge_and_selected_included() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    diversity = torch.tensor([2.0, 1.0, 1.0])
    slots = torch.tensor([0, 0, 0])
    weights = build_merge_weights(diversity, slots, output_count=1, merge_mode="weighted", selected_indices=torch.tensor([0]))
    expected = (2 * features[0] + features[1] + features[2]) / 4
    out = torch.zeros(1, 2)
    out.index_add_(0, slots, features * weights[:, None])
    assert torch.allclose(out[0], expected)
    assert not torch.allclose(out[0], features[0])
    uni = build_merge_weights(diversity, slots, output_count=1, merge_mode="uniform", selected_indices=torch.tensor([0]))
    assert torch.allclose(uni, torch.full((3,), 1 / 3))
    none = build_merge_weights(diversity, slots, output_count=1, merge_mode="none", selected_indices=torch.tensor([0]))
    assert none.tolist() == [1.0, 0.0, 0.0]


def test_compress_video_kitoke_weighted_merge_not_gather_and_order() -> None:
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    cfg = KiTokeAlgorithmConfig(
        target_visual_tokens=1,
        selection_method="topk",
        merge_mode="weighted",
        empty_interval_policy="repair_swap",
        kernel_alpha=1000.0,
        debug=True,
    )
    result = compress_video_kitoke(x, grid_h=1, grid_w=3, config=cfg, stable_video_identity="same")
    assert result.plan.retained_token_count == 1
    rep = result.plan.selected_flat_indices[0]
    assert not torch.allclose(result.compressed_main_embeddings[0], x.reshape(3, 2)[rep])
    assert result.plan.selected_flat_indices.tolist() == sorted(result.plan.selected_flat_indices.tolist())
    assert sum(result.plan.num_tokens_per_frame) == 1


def test_non_square_grid_deepstack_and_mrope_gather_mock_invariants() -> None:
    torch.manual_seed(2)
    x = torch.randn(3, 10, 4)
    cfg = KiTokeAlgorithmConfig(retain_ratio=0.2, selection_method="topk", empty_interval_policy="coarsen_then_repair")
    result = compress_video_kitoke(x, grid_h=2, grid_w=5, config=cfg, stable_video_identity="grid")
    plan = result.plan
    deep = torch.arange(plan.dense_token_count * 2, dtype=torch.float32).reshape(plan.dense_token_count, 2)
    weighted_deep = apply_kitoke_plan(deep, plan, mode="same_weighted_merge")
    gathered_deep = apply_kitoke_plan(deep, plan, mode="representative_gather")
    positions = torch.arange(plan.dense_token_count * 3).reshape(plan.dense_token_count, 3)
    selected_pos = positions.index_select(0, plan.selected_flat_indices)
    assert weighted_deep.shape[0] == plan.retained_token_count
    assert gathered_deep.shape[0] == plan.retained_token_count
    assert selected_pos.shape[0] == plan.retained_token_count
    verify_kitoke_lengths(
        plan=plan,
        placeholder_count=plan.retained_token_count,
        embedding_rows=result.compressed_main_embeddings.shape[0],
        mrope_count=selected_pos.shape[0],
        deepstack_rows=[weighted_deep.shape[0], gathered_deep.shape[0]],
        video_index=0,
    )
    dense_prompt = 100 + plan.dense_token_count
    compact_prompt = 100 + plan.retained_token_count
    assert dense_prompt - compact_prompt == plan.dense_token_count - plan.retained_token_count


def test_multiple_videos_and_stable_identity() -> None:
    torch.manual_seed(3)
    v1 = torch.randn(2, 4, 3)
    v2 = torch.randn(3, 2, 3)
    cfg = KiTokeAlgorithmConfig(retain_ratio=0.5, selection_method="pivotal", empty_interval_policy="coarsen_then_repair")
    a1 = compress_video_kitoke(v1, grid_h=2, grid_w=2, config=cfg, stable_video_identity="video-a").plan
    a2 = compress_video_kitoke(v1, grid_h=2, grid_w=2, config=cfg, stable_video_identity="video-a").plan
    b = compress_video_kitoke(v2, grid_h=1, grid_w=2, config=cfg, stable_video_identity="video-b").plan
    assert torch.equal(a1.selected_flat_indices, a2.selected_flat_indices)
    assert a1.target_token_count == 4
    assert b.target_token_count == 3


def test_zero_norm_finite_guard_and_assignment_no_nan() -> None:
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    density = compute_kernel_density_chunked(x, kernel_alpha=10.0, row_chunk_size=2, col_chunk_size=2)
    assert torch.isfinite(density).all()
    intervals = [KiTokeInterval(0, 1, 0, 3, torch.tensor([0]), torch.tensor([1, 2]))]
    slots = build_interval_assignments(x, torch.tensor([0]), intervals, match_chunk_size=1)
    assert slots.tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        compute_kernel_density_chunked(torch.tensor([[float("inf")]]), kernel_alpha=1.0, row_chunk_size=1, col_chunk_size=1)


def test_mode_none_apply_patch_no_side_effect() -> None:
    import vllm_qwen3_vl_kitoke as kitoke

    before = kitoke._PATCHED
    kitoke.apply_patch(mode="none")
    assert kitoke._PATCHED is before
