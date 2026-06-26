from __future__ import annotations

from pathlib import Path
import math
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vllm_qwen3_vl_flashvid import (  # noqa: E402
    FlashVIDAlgorithmConfig,
    aggregate_tstm_roots,
    align_premerge_attention_to_merged_tokens,
    apply_patch,
    build_default_merge_group_indices,
    compress_video_flashvid,
    compute_hybrid_llm_retention_ratio,
    compute_incoming_cls_attention_chunked,
    compute_incoming_cls_attention_reference,
    dpc_knn_framewise,
    dpc_knn_single_frame,
    dynamic_segment_video,
    pairwise_cosine_distances,
    select_adts_tokens,
    tstm_match_segment,
    verify_flashvid_lengths,
    _budget_split,
    _event_relevance_for_segment,
)


def _norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-6)


def test_pairwise_cosine_distance_and_zero_norm_safety() -> None:
    feats = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    dist = pairwise_cosine_distances(feats)
    assert torch.allclose(dist[0, 0], torch.tensor(0.0))
    assert torch.allclose(dist[0, 1], torch.tensor(1.0))
    assert torch.allclose(dist[0, 2], torch.tensor(2.0))
    assert torch.isfinite(dist).all()


def test_cls_attention_reference_and_chunked_parity() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 5, 4)
    k = torch.randn(2, 5, 4)
    ref = compute_incoming_cls_attention_reference(q, k, scale=0.5)
    got = compute_incoming_cls_attention_chunked(q, k, scale=0.5, query_chunk_size=2)
    assert torch.allclose(got, ref, atol=1e-6)
    assert torch.allclose(got.sum(), torch.tensor(1.0), atol=1e-6)


def test_cls_attention_per_frame_isolation() -> None:
    q0 = torch.eye(3).reshape(1, 3, 3)
    k0 = torch.eye(3).reshape(1, 3, 3)
    q1 = torch.flip(q0, dims=[1])
    k1 = torch.flip(k0, dims=[1])
    a0 = compute_incoming_cls_attention_reference(q0, k0, scale=1.0)
    a1 = compute_incoming_cls_attention_reference(q1, k1, scale=1.0)
    cat = compute_incoming_cls_attention_reference(torch.cat([q0, q1], dim=1), torch.cat([k0, k1], dim=1), scale=1.0)
    assert not torch.allclose(cat[:3], a0, atol=1e-4)
    assert torch.allclose(a0, a1.flip(0), atol=1e-6)


def test_attention_merge_group_alignment() -> None:
    attn = torch.arange(8, dtype=torch.float32)
    groups = build_default_merge_group_indices(8, 4, device=torch.device("cpu"))
    merged = align_premerge_attention_to_merged_tokens(attn, merge_group_indices=groups)
    assert torch.allclose(merged, torch.tensor([1.5, 5.5]))


def test_budget_split_direct_and_hybrid() -> None:
    direct = _budget_split(10, FlashVIDAlgorithmConfig(retention_ratio=0.25, budget_mode="direct", alpha=0.7))
    hybrid = _budget_split(10, FlashVIDAlgorithmConfig(retention_ratio=0.25, budget_mode="paper_hybrid", expansion=1.25, alpha=0.7))
    assert direct[:3] == (3, 3, 0)
    assert hybrid[0] == 4
    assert hybrid[3] == 0.3125


def test_dynamic_segmentation_strict_complementary_and_no_mutation() -> None:
    frames = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.8, 0.6], [0.0, 1.0]])
    before = frames.clone()
    lengths, starts, ends, sims, cuts = dynamic_segment_video(
        frames,
        threshold=float(F.cosine_similarity(frames[1], frames[2], dim=0)),
        min_segment_num=3,
        complementary=True,
    )
    assert torch.equal(frames, before)
    assert sum(lengths) == 4
    assert len(lengths) == 3
    assert starts[0] == 0 and ends[-1] == 4
    assert all(cuts[i] < cuts[i + 1] for i in range(len(cuts) - 1))
    assert 1 not in cuts  # equality with threshold is not a strict cut


def test_dynamic_segmentation_t_less_than_min_segments() -> None:
    lengths, *_ = dynamic_segment_video(
        torch.randn(3, 4),
        threshold=-2.0,
        min_segment_num=8,
        complementary=True,
    )
    assert lengths == [1, 1, 1]


def test_adts_attention_only_and_sorted_output() -> None:
    feats = torch.randn(4, 3)
    attn = torch.tensor([0.1, 0.9, 0.2, 0.8])
    _, idx, greedy = select_adts_tokens(feats, attn, num_tokens=2, method="attn")
    assert greedy.tolist() == [1, 3]
    assert idx.tolist() == [1, 3]


def test_adts_calibration_candidate_dimension_first_seed_and_iteration() -> None:
    feats = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    attn = torch.tensor([0.1, 0.1, 10.0, 0.1])
    _, idx, greedy = select_adts_tokens(feats, attn, num_tokens=2, method="attn_div")
    assert int(greedy[0]) == 2
    assert idx.tolist() == sorted(idx.tolist())
    assert len(set(greedy.tolist())) == 2


def test_adts_div_only_reference_and_ties() -> None:
    feats = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    attn = torch.ones(3)
    _, idx, greedy = select_adts_tokens(feats, attn, num_tokens=2, method="div")
    assert int(greedy[0]) == 0
    assert greedy.tolist()[1] in {0, 2}
    assert idx.tolist() == sorted(idx.tolist())


def test_adts_v2_event_relevance_and_uniqueness() -> None:
    segment = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.5, 0.5]],
        ]
    )
    relevance = _event_relevance_for_segment(segment)
    pooled = segment.float().mean(dim=1)
    expected = torch.einsum("snc,pc->snp", segment.float(), pooled).mean(dim=-1)
    assert torch.allclose(relevance, expected)
    _, idx, _ = select_adts_tokens(segment[0], torch.ones(2), num_tokens=2, method="attn_div_v2", event_relevance=relevance[0])
    assert idx.tolist() == [0, 1]


def test_tstm_full_spatial_adjacent_strict_threshold_and_no_future() -> None:
    seg = torch.zeros(3, 2, 2)
    seg[0, 0] = torch.tensor([1.0, 0.0])
    seg[0, 1] = torch.tensor([0.0, 1.0])
    seg[1, 0] = torch.tensor([0.0, 1.0])
    seg[1, 1] = torch.tensor([1.0, 0.0])
    seg[2] = torch.randn(2, 2)
    residual = torch.ones(3, 2, dtype=torch.bool)
    merge, parent, _, _, _ = tstm_match_segment(seg, residual, temporal_threshold=0.99, lower_bound=2, match_chunk_size=1)
    assert bool(merge[1, 0])
    assert int(parent[1, 0]) == 1
    changed = seg.clone()
    changed[2] = torch.randn_like(changed[2]) * 100
    merge_changed, parent_changed, *_ = tstm_match_segment(changed, residual, temporal_threshold=0.99, lower_bound=2, match_chunk_size=1)
    assert torch.equal(merge[1], merge_changed[1])
    assert torch.equal(parent[1], parent_changed[1])


def test_tstm_strict_threshold_equal_does_not_merge() -> None:
    seg = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    residual = torch.ones(2, 1, dtype=torch.bool)
    merge, *_ = tstm_match_segment(seg, residual, temporal_threshold=1.0, lower_bound=1, match_chunk_size=1)
    assert not bool(merge[1, 0])


def test_adts_protection_and_segment_boundary() -> None:
    seg = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    residual = torch.ones(2, 2, dtype=torch.bool)
    residual[:, 0] = False
    merge, parent, *_ = tstm_match_segment(seg, residual, temporal_threshold=0.5, lower_bound=2, match_chunk_size=2)
    assert not bool(merge[1, 0])
    assert int(parent[1, 1]) == 1

    first = seg[:1]
    second = seg[1:]
    m0, *_ = tstm_match_segment(first, residual[:1], temporal_threshold=0.5, lower_bound=1, match_chunk_size=2)
    m1, *_ = tstm_match_segment(second, residual[1:], temporal_threshold=0.5, lower_bound=1, match_chunk_size=2)
    assert not bool(m0.any()) and not bool(m1.any())


def test_tstm_chain_branching_and_unequal_subtree_mean() -> None:
    seg = torch.tensor(
        [
            [[0.0], [10.0]],
            [[2.0], [12.0]],
            [[4.0], [14.0]],
        ]
    )
    residual = torch.ones(3, 2, dtype=torch.bool)
    merge = torch.tensor([[False, False], [True, True], [True, False]])
    parent = torch.full((3, 2), -1, dtype=torch.long)
    parent[1, 0] = 0
    parent[1, 1] = 1
    parent[2, 0] = 0
    global_idx = torch.arange(6).reshape(3, 2)
    feats, reps, root_local, sums, counts = aggregate_tstm_roots(seg, residual, merge, parent, global_idx)
    assert reps[0].tolist() == [0, 1]
    assert torch.allclose(feats[0][0], torch.tensor([2.0]))
    assert torch.allclose(feats[0][1], torch.tensor([11.0]))
    assert torch.allclose(feats[2][0], torch.tensor([14.0]))
    assert counts[0, 0].item() == 3
    assert root_local[0].tolist() == [0, 1]


def test_tstm_lower_bound_adjustment_and_chunked_parity() -> None:
    torch.manual_seed(1)
    seg = torch.randn(3, 4, 5)
    residual = torch.ones(3, 4, dtype=torch.bool)
    a = tstm_match_segment(seg, residual, temporal_threshold=-1.0, lower_bound=8, match_chunk_size=1)
    b = tstm_match_segment(seg, residual, temporal_threshold=-1.0, lower_bound=8, match_chunk_size=4)
    assert int((~a[0]).sum()) >= 8
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_tstm_no_input_mutation_and_single_frame_roots() -> None:
    seg = torch.randn(1, 3, 2)
    before = seg.clone()
    residual = torch.ones(1, 3, dtype=torch.bool)
    merge, *_ = tstm_match_segment(seg, residual, temporal_threshold=0.8, lower_bound=1, match_chunk_size=2)
    assert torch.equal(seg, before)
    assert not bool(merge.any())


def test_dpc_density_center_assignment_ties_and_representative() -> None:
    feats = torch.tensor([[0.0], [1.0], [10.0], [11.0]])
    reps = torch.tensor([5, 6, 7, 8])
    out, out_reps, assign = dpc_knn_single_frame(feats, reps, num_clusters=2, dpc_k_max=2)
    assert out.shape == (2, 1)
    assert out_reps.numel() == 2
    for cid, rep in enumerate(out_reps.tolist()):
        center_row = int(torch.where(reps == rep)[0][0])
        assert int(assign[center_row]) == cid


def test_dpc_cluster_mean_and_hierarchical_mean_not_global_original_mean() -> None:
    root_feats = [torch.tensor([[10.0], [1.0]])]
    root_reps = [torch.tensor([0, 1])]
    out_feats, out_reps, *_ = dpc_knn_framewise(
        root_feats,
        root_reps,
        target_total=1,
        dpc_k_max=1,
        budget_correction="exact_total",
    )
    assert torch.allclose(out_feats[0][0], torch.tensor([5.5]))
    assert not torch.allclose(out_feats[0][0], torch.tensor([(10.0 * 10 + 1.0) / 11]))
    assert out_reps[0].numel() == 1


def test_official_ceil_budget_can_overshoot_and_exact_total_is_exact() -> None:
    feats = [torch.randn(3, 2), torch.randn(3, 2), torch.randn(3, 2)]
    reps = [torch.arange(3), torch.arange(3, 6), torch.arange(6, 9)]
    official, _, _, _ = dpc_knn_framewise(feats, reps, target_total=4, dpc_k_max=2, budget_correction="official_ceil")
    exact, _, _, _ = dpc_knn_framewise(feats, reps, target_total=4, dpc_k_max=2, budget_correction="exact_total")
    assert sum(x.shape[0] for x in official) >= 4
    assert sum(x.shape[0] for x in exact) == 4


def test_compress_flashvid_outputs_merged_features_not_representative_gather() -> None:
    video = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ]
    )
    attn = torch.ones(2, 2)
    cfg = FlashVIDAlgorithmConfig(
        retention_ratio=0.5,
        alpha=0.0,
        temporal_threshold=0.5,
        do_segment=False,
        budget_correction="exact_total",
        force_exact_count=True,
    )
    result = compress_video_flashvid(video, attn, grid_h=1, grid_w=2, config=cfg)
    assert result.plan.retained_token_count == 2
    assert result.plan.representative_flat_indices.tolist() == [0, 1]
    assert torch.allclose(result.compressed_main_embeddings, torch.tensor([[1.5, 0.0], [0.0, 1.5]]))
    dense = video.reshape(-1, 2).index_select(0, result.plan.representative_flat_indices)
    assert not torch.allclose(result.compressed_main_embeddings, dense)


def test_compress_flashvid_final_order_counts_non_square_and_identity() -> None:
    torch.manual_seed(2)
    video = torch.randn(3, 10, 4)
    attn = torch.ones(3, 10)
    cfg = FlashVIDAlgorithmConfig(retention_ratio=0.3, force_exact_count=True, temporal_match_chunk_size=3)
    result = compress_video_flashvid(video, attn, grid_h=2, grid_w=5, config=cfg)
    reps = result.plan.representative_flat_indices
    assert torch.equal(reps, torch.sort(reps).values)
    assert result.plan.retained_token_count == result.plan.planned_token_count
    assert sum(result.plan.num_tokens_per_frame) == result.plan.retained_token_count
    assert result.compressed_main_embeddings.shape[0] == result.plan.retained_token_count

    identity = compress_video_flashvid(
        video,
        attn,
        grid_h=2,
        grid_w=5,
        config=FlashVIDAlgorithmConfig(retention_ratio=1.0),
    )
    assert torch.equal(identity.plan.representative_flat_indices, torch.arange(video.numel() // 4))
    assert torch.allclose(identity.compressed_main_embeddings, video.reshape(-1, 4))


def test_deepstack_official_gather_and_mrope_representative_gather() -> None:
    reps = torch.tensor([0, 3, 5])
    deep = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    gathered = deep.index_select(0, reps)
    assert gathered.tolist() == [[0.0, 1.0], [6.0, 7.0], [10.0, 11.0]]
    dense_pos = torch.arange(24).reshape(6, 4)
    assert torch.equal(dense_pos.index_select(0, reps), dense_pos[reps])


def test_placeholder_embedding_position_invariant_and_prompt_delta() -> None:
    video = torch.randn(2, 4, 3)
    attn = torch.ones(2, 4)
    cfg = FlashVIDAlgorithmConfig(retention_ratio=0.5, force_exact_count=True)
    result = compress_video_flashvid(video, attn, grid_h=2, grid_w=2, config=cfg)
    verify_flashvid_lengths(
        plan=result.plan,
        placeholder_count=result.plan.retained_token_count,
        embedding_rows=result.compressed_main_embeddings.shape[0],
        mrope_count=result.plan.retained_token_count,
        deepstack_rows=[result.plan.retained_token_count],
    )
    dense_prompt = 100
    compact_prompt = dense_prompt - (result.plan.dense_token_count - result.plan.retained_token_count)
    assert dense_prompt - compact_prompt == result.plan.dense_token_count - result.plan.retained_token_count


def test_multiple_videos_independent_and_mixed_image_unchanged() -> None:
    a = compress_video_flashvid(
        torch.randn(2, 4, 3),
        torch.ones(2, 4),
        grid_h=2,
        grid_w=2,
        config=FlashVIDAlgorithmConfig(retention_ratio=0.5, force_exact_count=True),
    )
    b = compress_video_flashvid(
        torch.randn(3, 6, 3),
        torch.ones(3, 6),
        grid_h=2,
        grid_w=3,
        config=FlashVIDAlgorithmConfig(retention_ratio=0.5, force_exact_count=True),
    )
    assert a.plan.dense_token_count == 8
    assert b.plan.dense_token_count == 18
    image_embeds = torch.randn(5, 3)
    assert torch.equal(image_embeds, image_embeds.clone())


def test_empty_frame_count_possible_without_dummy() -> None:
    video = torch.randn(4, 6, 3)
    attn = torch.zeros(4, 6)
    cfg = FlashVIDAlgorithmConfig(retention_ratio=0.2, alpha=0.0, force_exact_count=True)
    result = compress_video_flashvid(video, attn, grid_h=2, grid_w=3, config=cfg)
    assert sum(result.plan.num_tokens_per_frame) == result.plan.retained_token_count
    assert result.plan.retained_token_count > 0


def test_hybrid_ratio_formula_and_invalid_params() -> None:
    rho = compute_hybrid_llm_retention_ratio(36, 28, 1.25)
    assert math.isclose(rho, 0.1, rel_tol=1e-6)
    with pytest.raises(ValueError):
        compute_hybrid_llm_retention_ratio(36, 30, 1.25)


def test_apply_patch_mode_none_and_hybrid_fail_fast() -> None:
    apply_patch(mode="none")
    with pytest.raises(NotImplementedError):
        apply_patch(mode="hybrid", budget_mode="paper_hybrid")


def test_no_input_mutation_end_to_end() -> None:
    video = torch.randn(2, 4, 3)
    attn = torch.rand(2, 4)
    before_video = video.clone()
    before_attn = attn.clone()
    compress_video_flashvid(
        video,
        attn,
        grid_h=2,
        grid_w=2,
        config=FlashVIDAlgorithmConfig(retention_ratio=0.5, force_exact_count=True),
    )
    assert torch.equal(video, before_video)
    assert torch.equal(attn, before_attn)


def test_length_verifier_raises_on_mismatch() -> None:
    video = torch.randn(2, 4, 3)
    attn = torch.ones(2, 4)
    result = compress_video_flashvid(
        video,
        attn,
        grid_h=2,
        grid_w=2,
        config=FlashVIDAlgorithmConfig(retention_ratio=0.5, force_exact_count=True),
    )
    with pytest.raises(RuntimeError):
        verify_flashvid_lengths(
            plan=result.plan,
            placeholder_count=result.plan.retained_token_count + 1,
            embedding_rows=result.plan.retained_token_count,
            mrope_count=result.plan.retained_token_count,
        )


@pytest.mark.parametrize(
    ("num_tokens", "ratio", "mode", "expansion", "alpha", "expected_budget"),
    [
        (8, 0.125, "direct", 1.25, 0.70, 1),
        (8, 0.50, "direct", 1.25, 0.70, 4),
        (8, 1.00, "direct", 1.25, 0.70, 8),
        (8, 0.20, "paper_hybrid", 1.25, 0.70, 2),
        (9, 0.20, "direct", 1.25, 0.00, 2),
        (9, 0.20, "direct", 1.25, 1.00, 2),
    ],
)
def test_budget_split_parameter_grid(
    num_tokens: int,
    ratio: float,
    mode: str,
    expansion: float,
    alpha: float,
    expected_budget: int,
) -> None:
    budget, b_adts, b_tstm, _ = _budget_split(
        num_tokens,
        FlashVIDAlgorithmConfig(
            retention_ratio=ratio,
            budget_mode=mode,
            expansion=expansion,
            alpha=alpha,
        ),
    )
    assert budget == expected_budget
    assert b_adts + b_tstm == budget
    assert 0 <= b_adts <= budget


@pytest.mark.parametrize(
    ("method", "budget"),
    [
        ("attn", 0),
        ("attn", 2),
        ("div", 1),
        ("div", 3),
        ("attn_div", 2),
        ("attn_div", 4),
        ("attn_div_v2", 2),
    ],
)
def test_adts_methods_parameter_grid(method: str, budget: int) -> None:
    feats = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    attn = torch.tensor([0.4, 0.3, 0.2, 0.1])
    event = torch.ones(4)
    _, idx, greedy = select_adts_tokens(
        feats,
        attn,
        num_tokens=budget,
        method=method,
        event_relevance=event if method == "attn_div_v2" else None,
    )
    assert idx.numel() == min(budget, feats.shape[0])
    assert torch.equal(idx, torch.sort(idx).values)
    assert torch.unique(greedy).numel() == greedy.numel()


@pytest.mark.parametrize(
    ("frames", "threshold", "min_segments"),
    [
        (1, 0.9, 4),
        (2, -2.0, 4),
        (2, 2.0, 4),
        (5, -2.0, 3),
        (5, 2.0, 8),
    ],
)
def test_segmentation_parameter_grid(frames: int, threshold: float, min_segments: int) -> None:
    x = torch.arange(frames * 3, dtype=torch.float32).reshape(frames, 3)
    lengths, starts, ends, _, cuts = dynamic_segment_video(
        x,
        threshold=threshold,
        min_segment_num=min_segments,
        complementary=True,
    )
    assert sum(lengths) == frames
    assert len(lengths) <= frames
    assert starts[0] == 0
    assert ends[-1] == frames
    assert len(cuts) == len(lengths) - 1


@pytest.mark.parametrize("clusters", [0, 1, 2, 4])
def test_dpc_single_frame_parameter_grid(clusters: int) -> None:
    feats = torch.tensor([[0.0, 0.0], [0.1, 0.0], [2.0, 2.0], [2.1, 2.0]])
    reps = torch.arange(10, 14)
    out, out_reps, assign = dpc_knn_single_frame(feats, reps, num_clusters=clusters, dpc_k_max=2)
    assert out.shape[0] == min(max(clusters, 0), feats.shape[0])
    assert out_reps.numel() == out.shape[0]
    assert assign.numel() == feats.shape[0]
    if clusters <= 0:
        assert torch.all(assign == -1)


@pytest.mark.parametrize("ratio", [0.125, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_compression_ratio_parameter_grid(ratio: float) -> None:
    torch.manual_seed(12)
    video = torch.randn(4, 8, 6)
    attn = torch.ones(4, 8)
    result = compress_video_flashvid(
        video,
        attn,
        grid_h=2,
        grid_w=4,
        config=FlashVIDAlgorithmConfig(retention_ratio=ratio, force_exact_count=True),
    )
    assert result.plan.retained_token_count == result.plan.planned_token_count
    assert result.compressed_main_embeddings.shape[0] == result.plan.retained_token_count
    assert torch.equal(result.plan.representative_flat_indices, torch.sort(result.plan.representative_flat_indices).values)
    assert result.plan.dense_token_count == 32


@pytest.mark.parametrize("deepstack_levels", [0, 1, 4])
def test_length_verifier_parameter_grid(deepstack_levels: int) -> None:
    video = torch.randn(2, 4, 3)
    attn = torch.ones(2, 4)
    result = compress_video_flashvid(
        video,
        attn,
        grid_h=2,
        grid_w=2,
        config=FlashVIDAlgorithmConfig(retention_ratio=0.5, force_exact_count=True),
    )
    rows = [result.plan.retained_token_count for _ in range(deepstack_levels)]
    verify_flashvid_lengths(
        plan=result.plan,
        placeholder_count=result.plan.retained_token_count,
        embedding_rows=result.plan.retained_token_count,
        mrope_count=result.plan.retained_token_count,
        deepstack_rows=rows,
    )


@pytest.mark.parametrize(
    ("mode", "profile", "method"),
    [
        ("direct", "official_qwen3", "attn_div"),
        ("direct", "paper_adts_v2", "attn_div_v2"),
        ("paper_hybrid", "official_qwen3", "attn_div"),
    ],
)
def test_profile_parameter_grid(mode: str, profile: str, method: str) -> None:
    video = torch.randn(3, 6, 5)
    attn = torch.ones(3, 6)
    result = compress_video_flashvid(
        video,
        attn,
        grid_h=2,
        grid_w=3,
        config=FlashVIDAlgorithmConfig(
            profile=profile,
            budget_mode=mode,
            token_selection_method=method,
            retention_ratio=0.5,
            force_exact_count=True,
        ),
    )
    assert result.plan.requested_profile == profile
    assert result.plan.effective_token_selection_method == method
    assert result.plan.retained_token_count == result.plan.planned_token_count
