from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vllm_qwen3_vl_mmtok import (  # noqa: E402
    _budget_from_inputs,
    _pre_merger_group_mean,
    apply_mmtok_plan,
    apply_mmtok_plan_to_deepstack,
    build_mmtok_plan,
    build_mmtok_query_text,
    build_stratified_video_target_coreset,
    compute_calibrated_columns,
    compute_row_log_normalizers,
    greedy_coverage_exact,
    greedy_coverage_reference,
    greedy_coverage_stochastic,
    select_adaptive_vv_temperature,
    verify_mmtok_lengths,
)


def _norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-8)


def _toy_video(t: int = 2, h: int = 2, w: int = 2, d: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    n = t * h * w
    post = torch.randn(n, d)
    pre = torch.randn(n, d + 1)
    query = torch.randn(3, d)
    return post, pre, query


def _full_combined(post: torch.Tensor, pre: torch.Tensor, query: torch.Tensor, tau_t: float, tau_v: float, alpha: float) -> torch.Tensor:
    p = torch.softmax(_norm(query) @ _norm(post).T / tau_t, dim=1) / query.shape[0]
    q = torch.softmax(_norm(pre) @ _norm(pre).T / tau_v, dim=1) / pre.shape[0]
    return torch.cat([p, alpha * q], dim=0)


def test_text_vision_formula() -> None:
    post = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pre = torch.randn(2, 3)
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    plan = build_mmtok_plan(
        post,
        pre,
        query,
        num_frames=1,
        grid_h=1,
        grid_w=2,
        target_visual_tokens=1,
        profile="paper_exact",
        greedy_mode="exact",
        tv_temperature=1.0,
        vv_temperature=1.0,
        exact_max_tokens=4,
    )
    combined = _full_combined(post, pre, query, 1.0, 1.0, 0.5)
    assert torch.allclose(combined[:2], torch.softmax(query @ post.T, dim=1) / 2, atol=1e-6)
    assert plan.retained_token_count == 1


def test_vision_vision_formula() -> None:
    post = torch.randn(2, 2)
    pre = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    query = torch.randn(1, 2)
    combined = _full_combined(post, pre, query, 1.0, 1.0, 0.5)
    expected = 0.5 * torch.softmax(pre @ pre.T, dim=1) / 2
    assert torch.allclose(combined[1:], expected, atol=1e-6)


def test_official_style_greedy_reference_parity() -> None:
    combined = torch.tensor(
        [
            [0.9, 0.1, 0.4],
            [0.1, 0.8, 0.4],
            [0.2, 0.2, 0.7],
        ]
    )
    ref = greedy_coverage_reference(combined, 2)
    opt, info = greedy_coverage_exact(combined, 2)
    assert torch.equal(ref, opt)
    assert torch.equal(torch.sort(opt).values, torch.tensor([0, 2]))
    assert info["objective_trace"][-1] >= info["objective_trace"][0]


def test_combined_objective_exactness() -> None:
    post, pre, query = _toy_video(t=1, h=2, w=2, d=3)
    combined = _full_combined(post, pre, query, 0.7, 0.9, 0.5)
    selected = torch.tensor([0, 2])
    obj = combined[:, selected].max(dim=1).values.sum()
    text = combined[: query.shape[0], selected].max(dim=1).values.sum()
    visual = combined[query.shape[0] :, selected].max(dim=1).values.sum() / 0.5
    assert torch.allclose(obj, text + 0.5 * visual)


def test_marginal_gain_exactness() -> None:
    combined = torch.rand(4, 5)
    current = torch.tensor([1, 3])
    best = combined[:, current].max(dim=1).values
    for j in range(5):
        gain = torch.clamp(combined[:, j] - best, min=0).sum()
        direct = combined[:, torch.cat([current, torch.tensor([j])])].max(dim=1).values.sum() - best.sum()
        assert torch.allclose(gain, direct, atol=1e-6)


def test_text_and_visual_feature_spaces_are_separate() -> None:
    post = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    pre = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    query = torch.tensor([[0.0, 1.0]])
    plan = build_mmtok_plan(post, pre, query, num_frames=1, grid_h=1, grid_w=3, target_visual_tokens=2, profile="paper_exact", exact_max_tokens=8)
    wrong = build_mmtok_plan(post, post, query, num_frames=1, grid_h=1, grid_w=3, target_visual_tokens=2, profile="paper_exact", exact_max_tokens=8)
    assert plan.keep_flat_indices.tolist() != wrong.keep_flat_indices.tolist()


def test_global_logsumexp_matches_full_softmax() -> None:
    torch.manual_seed(0)
    target = _norm(torch.randn(3, 4))
    cand = _norm(torch.randn(7, 4))
    logz = compute_row_log_normalizers(target, cand, temperature=0.3, candidate_chunk_size=2)
    expected = torch.logsumexp(target @ cand.T / 0.3, dim=1)
    assert torch.allclose(logz, expected, atol=1e-6)


def test_calibrated_columns_use_global_denominator_not_chunk_local() -> None:
    target = _norm(torch.tensor([[1.0, 0.0]]))
    cand = _norm(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    logz = compute_row_log_normalizers(target, cand, temperature=1.0, candidate_chunk_size=1)
    cols = compute_calibrated_columns(target, cand, torch.tensor([0]), temperature=1.0, row_weights=torch.ones(1), row_log_normalizers=logz)
    assert cols.item() < 1.0
    assert torch.allclose(cols[0, 0], torch.softmax(target @ cand.T, dim=1)[0, 0])


def test_temperature_entropy_and_independent_temperatures() -> None:
    logits = torch.tensor([[3.0, 1.0, 0.0]])
    cold = torch.softmax(logits / 0.1, dim=1)
    warm = torch.softmax(logits / 1.0, dim=1)
    ent_cold = -(cold * cold.clamp_min(1e-12).log()).sum()
    ent_warm = -(warm * warm.clamp_min(1e-12).log()).sum()
    assert ent_cold <= ent_warm
    assert cold.max() >= warm.max()

    post, pre, query = _toy_video(t=1, h=2, w=2, d=3)
    a = build_mmtok_plan(post, pre, query, num_frames=1, grid_h=2, grid_w=2, target_visual_tokens=2, tv_temperature=0.01, vv_temperature=0.2, profile="paper_exact", exact_max_tokens=8, debug=True)
    b = build_mmtok_plan(post, pre, query, num_frames=1, grid_h=2, grid_w=2, target_visual_tokens=2, tv_temperature=0.5, vv_temperature=0.2, profile="paper_exact", exact_max_tokens=8, debug=True)
    assert a.effective_vv_temperature == b.effective_vv_temperature == 0.2


def test_adaptive_vv_temperature_selects_closest_second_peak_and_tie_smaller() -> None:
    pre = _norm(torch.eye(3))
    target_indices = torch.arange(3)
    weights = torch.full((3,), 1 / 3)
    tau, diag = select_adaptive_vv_temperature(
        text_peak=torch.tensor(0.30),
        pre_norm=pre,
        target_indices=target_indices,
        target_weights=weights,
        candidates=(0.1, 1.0),
        candidate_chunk_size=3,
        fixed_fallback=0.2,
    )
    assert tau in {0.1, 1.0}
    assert "adaptive_selected_gap" in diag
    tau_tie, _ = select_adaptive_vv_temperature(
        text_peak=torch.tensor(0.0),
        pre_norm=pre,
        target_indices=target_indices,
        target_weights=weights,
        candidates=(0.2, 0.1),
        candidate_chunk_size=3,
        fixed_fallback=0.2,
    )
    assert tau_tie == 0.1


def test_adaptive_uses_second_largest_and_n1_fallback() -> None:
    post = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pre = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    query = torch.tensor([[1.0, 0.0]])
    plan = build_mmtok_plan(
        post,
        pre,
        query,
        num_frames=1,
        grid_h=1,
        grid_w=2,
        target_visual_tokens=1,
        temperature_mode="adaptive_vv",
        adaptive_vv_candidates=(0.05, 0.2),
        profile="paper_exact",
        exact_max_tokens=4,
        debug=True,
    )
    assert plan.temperature_diagnostics is not None
    assert "adaptive_second_tau_0.05" in plan.temperature_diagnostics
    single = build_mmtok_plan(
        post[:1],
        pre[:1],
        query,
        num_frames=1,
        grid_h=1,
        grid_w=1,
        target_visual_tokens=1,
        temperature_mode="adaptive_vv",
    )
    assert single.effective_profile == "identity"


def test_coreset_invariants_temporal_coverage_and_full_equivalence() -> None:
    idx, weights = build_stratified_video_target_coreset(num_frames=4, grid_h=2, grid_w=3, target_count=8, device=torch.device("cpu"))
    assert len(idx) == 8
    assert torch.unique(idx).numel() == 8
    assert torch.all(weights > 0)
    assert torch.allclose(weights.sum(), torch.tensor(1.0))
    frames = torch.div(idx, 6, rounding_mode="floor")
    assert set(frames.tolist()) == {0, 1, 2, 3}

    full_idx, full_weights = build_stratified_video_target_coreset(num_frames=2, grid_h=2, grid_w=2, target_count=8, device=torch.device("cpu"))
    assert full_idx.tolist() == list(range(8))
    assert torch.allclose(full_weights, torch.full((8,), 1 / 8))


def test_weighted_coreset_objective_not_uniform() -> None:
    combined_visual = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    weights = torch.tensor([0.9, 0.1])
    weighted = (weights[:, None] * combined_visual)[:, [0]].max(dim=1).values.sum()
    uniform = (torch.full((2,), 0.5)[:, None] * combined_visual)[:, [0]].max(dim=1).values.sum()
    assert not torch.allclose(weighted, uniform)


def test_stochastic_determinism_budget_tie_and_rng_isolation() -> None:
    combined = torch.ones(4, 8)
    state = torch.random.get_rng_state()
    a, info_a = greedy_coverage_stochastic(combined, 3, epsilon=0.2, seed=123)
    state_after = torch.random.get_rng_state()
    b, info_b = greedy_coverage_stochastic(combined, 3, epsilon=0.2, seed=123)
    assert torch.equal(a, b)
    assert torch.equal(state, state_after)
    assert a.numel() == 3
    assert torch.unique(a).numel() == 3
    assert info_a["sample_size"] == info_b["sample_size"]


def test_global_selection_not_per_frame_and_rare_event() -> None:
    post = torch.zeros(12, 4)
    pre = torch.zeros(12, 4)
    post[8:12] = torch.eye(4)
    query = torch.eye(4)
    plan = build_mmtok_plan(post, pre, query, num_frames=3, grid_h=2, grid_w=2, target_visual_tokens=4, alpha=0.0, profile="paper_exact", exact_max_tokens=20)
    assert plan.num_tokens_per_frame[2] == 4
    assert sum(plan.num_tokens_per_frame) == 4


def test_text_guided_rare_event_and_alpha_extremes() -> None:
    post = torch.eye(4)
    pre = torch.ones(4, 3)
    query = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    plan = build_mmtok_plan(post, pre, query, num_frames=1, grid_h=2, grid_w=2, target_visual_tokens=1, alpha=0.0, profile="paper_exact", exact_max_tokens=8)
    assert plan.keep_flat_indices.tolist() == [2]


def test_single_frame_non_square_floor_budget_identity_and_b1() -> None:
    post, pre, query = _toy_video(t=1, h=2, w=5, d=4)
    assert _budget_from_inputs(10, retain_ratio=0.25, target_visual_tokens=None, budget_rounding="floor") == 2
    plan = build_mmtok_plan(post, pre, query, num_frames=1, grid_h=2, grid_w=5, retain_ratio=1.0)
    assert plan.keep_flat_indices.tolist() == list(range(10))
    one = build_mmtok_plan(post, pre, query, num_frames=1, grid_h=2, grid_w=5, target_visual_tokens=1, profile="paper_exact", exact_max_tokens=16)
    assert one.retained_token_count == 1
    assert one.keep_flat_indices.tolist() == sorted(one.keep_flat_indices.tolist())


def test_deepstack_alignment_no_input_mutation_and_apply() -> None:
    post, pre, query = _toy_video(t=2, h=2, w=2, d=4)
    post_before, pre_before, query_before = post.clone(), pre.clone(), query.clone()
    plan = build_mmtok_plan(post, pre, query, num_frames=2, grid_h=2, grid_w=2, target_visual_tokens=3, profile="paper_exact", exact_max_tokens=16)
    selected = apply_mmtok_plan(post, plan)
    deep = [torch.arange(8 * 2).reshape(8, 2).float(), torch.arange(8 * 3).reshape(8, 3).float()]
    selected_deep = apply_mmtok_plan_to_deepstack(deep, plan)
    assert selected.shape[0] == 3
    assert all(x.shape[0] == 3 for x in selected_deep)
    assert torch.equal(selected_deep[0], deep[0].index_select(0, plan.keep_flat_indices))
    assert torch.equal(post, post_before)
    assert torch.equal(pre, pre_before)
    assert torch.equal(query, query_before)


def test_dimension_mismatch_pre_post_mismatch_empty_query() -> None:
    post, pre, query = _toy_video(t=1, h=2, w=2, d=4)
    with pytest.raises(ValueError, match="dimension mismatch"):
        build_mmtok_plan(post, pre, torch.randn(2, 5), num_frames=1, grid_h=2, grid_w=2, target_visual_tokens=2)
    with pytest.raises(ValueError, match="pre/post row mismatch"):
        build_mmtok_plan(post, pre[:3], query, num_frames=1, grid_h=2, grid_w=2, target_visual_tokens=2)
    with pytest.raises(ValueError, match="query is empty"):
        build_mmtok_plan(post, pre, torch.empty(0, 4), num_frames=1, grid_h=2, grid_w=2, target_visual_tokens=2)


def test_multiple_videos_independent_plans() -> None:
    a_post, a_pre, q = _toy_video(t=1, h=2, w=2, d=3)
    b_post, b_pre, _ = _toy_video(t=3, h=1, w=2, d=3)
    a = build_mmtok_plan(a_post, a_pre, q, num_frames=1, grid_h=2, grid_w=2, target_visual_tokens=2, profile="paper_exact", exact_max_tokens=8)
    b = build_mmtok_plan(b_post, b_pre, q, num_frames=3, grid_h=1, grid_w=2, target_visual_tokens=3, profile="paper_exact", exact_max_tokens=8)
    assert a.retained_token_count == 2
    assert b.retained_token_count == 3
    assert a.num_tokens_per_frame != b.num_tokens_per_frame


def test_no_answer_leakage_query_builder() -> None:
    annotation = {
        "question": "What is shown?",
        "choices": {"A": "cat", "B": "dog"},
        "answer": "SECRET_LABEL",
        "answer_id": "B",
    }
    query = build_mmtok_query_text(annotation, query_source="question_options")
    assert "What is shown" in query
    assert "(A) cat" in query
    assert "SECRET_LABEL" not in query
    assert "answer_id" not in query


def test_mixed_message_query_skips_video_items() -> None:
    messages = [{"role": "user", "content": [{"type": "video", "video": "x.mp4"}, {"type": "text", "text": "Use this"}]}]
    assert build_mmtok_query_text(None, messages, query_source="user_text") == "Use this"


def test_mock_true_compression_invariants_and_uneven_counts() -> None:
    post, pre, query = _toy_video(t=4, h=2, w=5, d=4)
    plan = build_mmtok_plan(post, pre, query, num_frames=4, grid_h=2, grid_w=5, target_visual_tokens=20, profile="paper_exact", exact_max_tokens=64)
    dense_prompt = 123
    compact_prompt = dense_prompt - (plan.dense_token_count - plan.target_token_count)
    verify_mmtok_lengths(
        plan=plan,
        placeholder_count=20,
        embedding_rows=20,
        mrope_count=20,
        deepstack_rows=[20, 20],
        dense_prompt_length=dense_prompt,
        compact_prompt_length=compact_prompt,
    )
    assert sum(plan.num_tokens_per_frame) == 20


def test_pre_merger_group_mean_mapping() -> None:
    hidden = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 1, 3)
    pre = _pre_merger_group_mean(hidden, 4)
    assert torch.equal(pre[0], hidden[:4, 0].mean(dim=0))
    assert torch.equal(pre[1], hidden[4:, 0].mean(dim=0))


def test_exact_memory_guard() -> None:
    post, pre, query = _toy_video(t=2, h=2, w=2, d=4)
    with pytest.raises(RuntimeError, match="exact_max_tokens"):
        build_mmtok_plan(post, pre, query, num_frames=2, grid_h=2, grid_w=2, target_visual_tokens=2, profile="paper_exact", exact_max_tokens=4)


def test_objective_monotonicity_and_bound() -> None:
    combined = torch.rand(5, 6)
    _, info = greedy_coverage_exact(combined, 4)
    trace = info["objective_trace"]
    assert all(a <= b + 1e-6 for a, b in zip(trace, trace[1:]))

    post, pre, query = _toy_video(t=1, h=2, w=2, d=3)
    plan = build_mmtok_plan(post, pre, query, num_frames=1, grid_h=2, grid_w=2, target_visual_tokens=2, alpha=0.5, profile="paper_exact", exact_max_tokens=8)
    assert plan.objective_value is not None
    assert 0.0 <= plan.objective_value <= 1.5 + 1e-5


def test_query_and_temperature_dependent_selection() -> None:
    post = torch.eye(4)
    pre = torch.eye(4)
    q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q1 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    a = build_mmtok_plan(post, pre, q0, num_frames=1, grid_h=2, grid_w=2, target_visual_tokens=1, alpha=0.0, profile="paper_exact", exact_max_tokens=8)
    b = build_mmtok_plan(post, pre, q1, num_frames=1, grid_h=2, grid_w=2, target_visual_tokens=1, alpha=0.0, profile="paper_exact", exact_max_tokens=8)
    assert a.keep_flat_indices.tolist() != b.keep_flat_indices.tolist()
    c = build_mmtok_plan(post, pre, q0, num_frames=1, grid_h=2, grid_w=2, target_visual_tokens=2, tv_temperature=0.01, vv_temperature=0.05, profile="paper_exact", exact_max_tokens=8, debug=True)
    d = build_mmtok_plan(post, pre, q0, num_frames=1, grid_h=2, grid_w=2, target_visual_tokens=2, tv_temperature=0.01, vv_temperature=1.0, profile="paper_exact", exact_max_tokens=8, debug=True)
    assert c.temperature_diagnostics is not None
    assert d.temperature_diagnostics is not None


def test_mode_none_mock_identity() -> None:
    post, pre, query = _toy_video(t=1, h=2, w=2, d=4)
    plan = build_mmtok_plan(post, pre, query, num_frames=1, grid_h=2, grid_w=2, retain_ratio=1.0)
    assert torch.equal(apply_mmtok_plan(post, plan), post)


def test_budget_conservation_parametrized() -> None:
    for t, h, w, b in [(2, 2, 2, 3), (3, 1, 5, 7), (1, 1, 2, 1)]:
        post, pre, query = _toy_video(t=t, h=h, w=w, d=4)
        plan = build_mmtok_plan(post, pre, query, num_frames=t, grid_h=h, grid_w=w, target_visual_tokens=b, profile="paper_exact", exact_max_tokens=32)
        assert len(plan.keep_flat_indices) == min(max(1, b), t * h * w)
        assert sum(plan.num_tokens_per_frame) == plan.retained_token_count
