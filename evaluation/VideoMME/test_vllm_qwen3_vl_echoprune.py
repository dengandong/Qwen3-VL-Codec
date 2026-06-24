from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vllm_qwen3_vl_echoprune import (  # noqa: E402
    _local_candidate_indices,
    apply_echoprune_plan,
    apply_echoprune_plan_to_deepstack,
    build_echoprune_plan,
    build_echoprune_query_text,
    compute_crossmodal_relevance,
    compute_temporal_echo_scores,
    reference_compute_temporal_echo_scores,
    verify_echoprune_lengths,
)


def test_crossmodal_relevance_and_mask() -> None:
    video = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    rel = compute_crossmodal_relevance(video, query)
    expected = torch.tensor([[1.0, 1.0, 2**-0.5]])
    assert torch.allclose(rel, expected, atol=1e-5)

    masked = compute_crossmodal_relevance(video, query, torch.tensor([True, False]))
    assert torch.allclose(masked, torch.tensor([[1.0, 0.0, 2**-0.5]]), atol=1e-5)


def test_correspondence_exactness() -> None:
    video = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )
    corr, _ = compute_temporal_echo_scores(
        video,
        temperature=0.5,
        match_scope="full",
        grid_h=1,
        grid_w=2,
        window_size=3,
        chunk_size=2,
    )
    assert torch.allclose(corr[1], torch.tensor([1.0, 0.0]), atol=1e-5)


def test_echo_reconstruction_exactness_and_no_renormalize() -> None:
    video = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )
    _, echo = compute_temporal_echo_scores(
        video,
        temperature=1.0,
        match_scope="full",
        grid_h=1,
        grid_w=2,
        window_size=3,
        chunk_size=1,
    )
    probs = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)
    echo_hat = probs[0] * torch.tensor([1.0, 0.0]) + probs[1] * torch.tensor([0.0, 1.0])
    expected = torch.dot(torch.tensor([1.0, 0.0]), echo_hat)
    normalized_echo_hat = torch.nn.functional.normalize(echo_hat, dim=0)
    normalized_value = torch.dot(torch.tensor([1.0, 0.0]), normalized_echo_hat)
    assert torch.allclose(echo[1, 0], expected, atol=1e-5)
    assert not torch.allclose(echo[1, 0], normalized_value, atol=1e-5)


def test_temperature_effect() -> None:
    video = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    _, cold = compute_temporal_echo_scores(
        video,
        temperature=0.1,
        match_scope="full",
        grid_h=1,
        grid_w=2,
        window_size=3,
        chunk_size=2,
    )
    _, warm = compute_temporal_echo_scores(
        video,
        temperature=2.0,
        match_scope="full",
        grid_h=1,
        grid_w=2,
        window_size=3,
        chunk_size=2,
    )
    assert cold[1, 0] > warm[1, 0]


def test_full_vs_local_scope() -> None:
    video = torch.zeros((2, 4, 4))
    video[0, 3, 0] = 1.0
    video[0, 0, 1] = 1.0
    video[1, 0, 0] = 1.0
    _, full = compute_temporal_echo_scores(
        video,
        temperature=0.1,
        match_scope="full",
        grid_h=1,
        grid_w=4,
        window_size=3,
        chunk_size=2,
    )
    _, local = compute_temporal_echo_scores(
        video,
        temperature=0.1,
        match_scope="local",
        grid_h=1,
        grid_w=4,
        window_size=3,
        chunk_size=2,
    )
    assert full[1, 0] > 0.9
    assert local[1, 0] < 0.1


def test_local_boundary_has_no_clamped_duplicates() -> None:
    candidates = _local_candidate_indices(2, 2, 3, device=torch.device("cpu"))
    assert [int(x.numel()) for x in candidates] == [4, 4, 4, 4]
    assert candidates[0].tolist() == [0, 1, 2, 3]


def test_causal_adjacent_only() -> None:
    video = torch.randn(3, 3, 5)
    changed = video.clone()
    changed[0] = torch.randn_like(changed[0])
    _, echo = compute_temporal_echo_scores(
        video,
        temperature=0.5,
        match_scope="full",
        grid_h=1,
        grid_w=3,
        window_size=3,
        chunk_size=2,
    )
    _, echo_changed = compute_temporal_echo_scores(
        changed,
        temperature=0.5,
        match_scope="full",
        grid_h=1,
        grid_w=3,
        window_size=3,
        chunk_size=2,
    )
    assert torch.allclose(echo[2], echo_changed[2], atol=1e-6)


def test_identical_frames_have_high_redundancy() -> None:
    frame = torch.eye(3)
    video = torch.stack([frame, frame], dim=0)
    corr, echo = compute_temporal_echo_scores(
        video,
        temperature=0.1,
        match_scope="full",
        grid_h=1,
        grid_w=3,
        window_size=3,
        chunk_size=3,
    )
    assert torch.all(corr[1] > 0.99)
    assert torch.all(echo[1] > 0.99)


def test_novel_query_relevant_token_is_selected() -> None:
    video = torch.zeros((2, 3, 4))
    video[0, :, 1] = 1.0
    video[1, 0, 1] = 1.0
    video[1, 1, 0] = 1.0
    video[1, 2, 1] = 1.0
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    plan = build_echoprune_plan(
        video,
        query,
        target_visual_tokens=2,
        grid_h=1,
        grid_w=3,
        first_frame_policy="paper",
    )
    assert 4 in plan.keep_flat_indices.tolist()


def test_exact_score_equation() -> None:
    video = torch.randn(3, 4, 6)
    query = torch.randn(5, 6)
    plan = build_echoprune_plan(video, query, target_visual_tokens=5, grid_h=2, grid_w=2)
    assert plan.score is not None
    assert torch.allclose(
        plan.score,
        plan.relevance - plan.delta_corr - plan.delta_echo,
        atol=1e-6,
    )


def test_global_topk_not_uniform_per_frame() -> None:
    video = torch.zeros((3, 4, 4))
    video[0, :, 1] = 1.0
    video[1, :, 1] = 1.0
    video[2, :, 0] = 1.0
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    plan = build_echoprune_plan(video, query, target_visual_tokens=4, grid_h=2, grid_w=2)
    assert plan.retained_token_count == 4
    assert plan.num_tokens_per_frame[2] >= 2


def test_first_frame_paper_policy_quota() -> None:
    video = torch.randn(4, 5, 3)
    query = torch.randn(2, 3)
    plan = build_echoprune_plan(video, query, target_visual_tokens=7, grid_h=1, grid_w=5)
    assert plan.first_frame_quota == min(5, 7, max(1, 7 // 4))
    assert sum(plan.num_tokens_per_frame) == 7


def test_single_frame_selects_budget_by_relevance() -> None:
    video = torch.eye(4).unsqueeze(0)
    query = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    plan = build_echoprune_plan(video, query, target_visual_tokens=2, grid_h=1, grid_w=4)
    assert plan.num_frames == 1
    assert plan.retained_token_count == 2
    assert 2 in plan.keep_flat_indices.tolist()


def test_retain_ratio_one_identity_and_apply() -> None:
    video = torch.randn(2, 3, 4)
    query = torch.randn(2, 4)
    plan = build_echoprune_plan(video, query, retain_ratio=1.0, grid_h=1, grid_w=3)
    assert plan.keep_flat_indices.tolist() == list(range(6))
    flat = video.reshape(6, 4)
    assert torch.equal(apply_echoprune_plan(flat, plan), flat)


def test_budget_one_and_deterministic_ties() -> None:
    video = torch.zeros((2, 3, 4))
    query = torch.ones((1, 4))
    plan_one = build_echoprune_plan(video, query, target_visual_tokens=1, grid_h=1, grid_w=3)
    assert plan_one.keep_flat_indices.tolist() == [0]

    plan_global = build_echoprune_plan(
        video,
        query,
        target_visual_tokens=3,
        grid_h=1,
        grid_w=3,
        first_frame_policy="global",
    )
    assert plan_global.keep_flat_indices.tolist() == [0, 1, 2]


def test_non_square_grid_and_chronological_output() -> None:
    video = torch.randn(3, 10, 4)
    query = torch.randn(2, 4)
    plan = build_echoprune_plan(video, query, target_visual_tokens=8, grid_h=2, grid_w=5)
    assert plan.grid_h == 2
    assert plan.grid_w == 5
    assert plan.keep_flat_indices.tolist() == sorted(plan.keep_flat_indices.tolist())


def test_no_input_mutation() -> None:
    video = torch.randn(2, 4, 3)
    query = torch.randn(3, 3)
    video_before = video.clone()
    query_before = query.clone()
    build_echoprune_plan(video, query, target_visual_tokens=3, grid_h=2, grid_w=2)
    assert torch.equal(video, video_before)
    assert torch.equal(query, query_before)


def test_slow_reference_parity() -> None:
    video = torch.randn(3, 4, 5)
    fast = compute_temporal_echo_scores(
        video,
        temperature=0.7,
        match_scope="full",
        grid_h=2,
        grid_w=2,
        window_size=3,
        chunk_size=2,
    )
    slow = reference_compute_temporal_echo_scores(
        video,
        temperature=0.7,
        match_scope="full",
        grid_h=2,
        grid_w=2,
        window_size=3,
    )
    assert torch.allclose(fast[0], slow[0], atol=1e-6)
    assert torch.allclose(fast[1], slow[1], atol=1e-6)


def test_deepstack_alignment() -> None:
    video = torch.randn(2, 4, 3)
    query = torch.randn(2, 3)
    plan = build_echoprune_plan(video, query, target_visual_tokens=3, grid_h=2, grid_w=2)
    main = video.reshape(8, 3)
    deep = [torch.arange(8 * 2).reshape(8, 2).float(), torch.arange(8 * 5).reshape(8, 5).float()]
    selected = apply_echoprune_plan(main, plan)
    selected_deep = apply_echoprune_plan_to_deepstack(deep, plan)
    assert selected.shape[0] == 3
    assert all(layer.shape[0] == 3 for layer in selected_deep)
    assert torch.equal(selected_deep[0], deep[0].index_select(0, plan.keep_flat_indices))


def test_independent_videos_and_multiple_queries() -> None:
    video_a = torch.randn(2, 4, 3)
    video_b = torch.randn(3, 4, 3)
    query_a = torch.tensor([[1.0, 0.0, 0.0]])
    query_b = torch.tensor([[0.0, 1.0, 0.0]])
    plan_a = build_echoprune_plan(video_a, query_a, target_visual_tokens=3, grid_h=2, grid_w=2)
    plan_b = build_echoprune_plan(video_b, query_b, target_visual_tokens=5, grid_h=2, grid_w=2)
    assert plan_a.retained_token_count == 3
    assert plan_b.retained_token_count == 5
    assert plan_a.query_token_count == plan_b.query_token_count == 1


def test_empty_query_and_dimension_mismatch_fail_fast() -> None:
    video = torch.randn(2, 4, 3)
    with pytest.raises(ValueError, match="empty"):
        build_echoprune_plan(video, torch.empty(0, 3), target_visual_tokens=2, grid_h=2, grid_w=2)
    with pytest.raises(ValueError, match="dimension mismatch"):
        build_echoprune_plan(video, torch.randn(2, 4), target_visual_tokens=2, grid_h=2, grid_w=2)


def test_query_padding_mask_excludes_padding() -> None:
    video = torch.tensor([[[0.0, 1.0]]])
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    rel = compute_crossmodal_relevance(video, query, torch.tensor([True, False]))
    assert torch.allclose(rel, torch.tensor([[0.0]]), atol=1e-6)


def test_no_answer_leakage_in_query_builder() -> None:
    annotation = {
        "question": "What color is the car?",
        "choices": {"A": "red", "B": "blue"},
        "answer": "SECRET_ANSWER_FIELD",
        "answer_id": "B",
    }
    query = build_echoprune_query_text(annotation, query_source="question_options")
    assert "What color" in query
    assert "(A) red" in query
    assert "SECRET_ANSWER_FIELD" not in query
    assert "answer_id" not in query


def test_mixed_message_query_skips_video_items() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "file:///tmp/x.mp4"},
                {"type": "text", "text": "Only this text"},
            ],
        }
    ]
    assert build_echoprune_query_text(None, messages, query_source="user_text") == "Only this text"


@pytest.mark.parametrize("t,n,b", [(2, 4, 3), (5, 6, 8), (1, 5, 2), (3, 4, 12)])
def test_budget_conservation(t: int, n: int, b: int) -> None:
    video = torch.randn(t, n, 4)
    query = torch.randn(2, 4)
    plan = build_echoprune_plan(video, query, target_visual_tokens=b, grid_h=1, grid_w=n)
    assert len(plan.keep_flat_indices) == min(t * n, max(1, b))
    assert sum(plan.num_tokens_per_frame) == plan.retained_token_count


def test_mock_true_compression_invariants() -> None:
    video = torch.randn(5, 20, 6)
    query = torch.randn(3, 6)
    plan = build_echoprune_plan(video, query, target_visual_tokens=20, grid_h=4, grid_w=5)
    dense_prompt = 130
    compact_prompt = dense_prompt - (plan.dense_token_count - plan.target_token_count)
    verify_echoprune_lengths(
        plan=plan,
        placeholder_count=20,
        embedding_rows=20,
        mrope_count=20,
        dense_prompt_length=dense_prompt,
        compact_prompt_length=compact_prompt,
    )


def test_mock_uneven_per_frame_selection_can_have_empty_frame() -> None:
    video = torch.zeros(4, 10, 4)
    video[2, :, 0] = 1.0
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    plan = build_echoprune_plan(video, query, target_visual_tokens=10, grid_h=2, grid_w=5)
    assert sum(plan.num_tokens_per_frame) == 10
    assert any(count == 0 for count in plan.num_tokens_per_frame[1:])
