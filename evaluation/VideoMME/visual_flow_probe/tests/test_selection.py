from __future__ import annotations

import torch

from evaluation.VideoMME.visual_flow_probe.interventions import (
    build_selection_sets,
    random_global_indices,
    random_temporal_matched_indices,
    select_ranked_indices,
)


def test_exact_budget_top_bottom_random() -> None:
    scores = torch.tensor([0.1, 0.5, 0.2, 0.4, 0.3])
    assert select_ranked_indices(scores, 0.4, condition="top").tolist() == [1, 3]
    assert select_ranked_indices(scores, 0.4, condition="bottom").tolist() == [0, 2]
    rand = random_global_indices(5, 0.4, seed=123)
    assert rand.numel() == 2
    assert len(set(rand.tolist())) == 2


def test_stable_random_selection() -> None:
    a = random_global_indices(20, 0.2, seed=99)
    b = random_global_indices(20, 0.2, seed=99)
    c = random_global_indices(20, 0.2, seed=100)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_temporally_matched_random_counts_match_top() -> None:
    temporal = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    top = torch.tensor([0, 3, 5, 6])
    matched = random_temporal_matched_indices(top, temporal, seed=7)
    for bin_id in torch.unique(temporal):
        assert int((temporal[matched] == bin_id).sum()) == int((temporal[top] == bin_id).sum())


def test_build_selection_sets_all_conditions_exact_budget() -> None:
    scores = torch.linspace(0, 1, 10)
    temporal = torch.tensor([0] * 5 + [1] * 5)
    out = build_selection_sets(
        {"responsibility": scores},
        [0.2],
        temporal,
        question_id="q",
        seed=42,
        random_repeats=3,
    )
    by_cond = out["responsibility"][0.2]
    assert by_cond["top"].numel() == 2
    assert by_cond["bottom"].numel() == 2
    assert len(by_cond["random_global"]) == 3
    assert all(x.numel() == 2 for x in by_cond["random_global"])
