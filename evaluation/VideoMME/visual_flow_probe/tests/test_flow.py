from __future__ import annotations

import pytest
import torch

from evaluation.VideoMME.visual_flow_probe.flow import (
    compute_answer_reachability,
    direct_target_attention,
    normalize_visual_responsibility,
)


def test_flow_dp_hand_constructed_dag() -> None:
    # Edges: 0->2 (0.5), 1->2 (0.2), 2->4 (0.4), 3 cannot reach target.
    a = torch.zeros(5, 5)
    a[2, 0] = 0.5
    a[2, 1] = 0.2
    a[4, 2] = 0.4
    h = compute_answer_reachability(a, [4])
    assert h[4].item() == pytest.approx(1.0)
    assert h[2].item() == pytest.approx(0.4)
    assert h[0].item() == pytest.approx(0.2)
    assert h[1].item() == pytest.approx(0.08)
    assert h[3].item() == pytest.approx(0.0)


def test_strict_chronology_future_and_diagonal_do_not_contribute() -> None:
    a = torch.zeros(4, 4)
    a[1, 0] = 0.1
    a[2, 1] = 0.5
    a[3, 2] = 0.5
    a[0, 3] = 100.0  # future-to-past, should fail causal validation
    with pytest.raises(ValueError):
        compute_answer_reachability(a, [3])
    a[0, 3] = 0.0
    a[2, 2] = 100.0  # diagonal ignored by recurrence
    h = compute_answer_reachability(a, [3])
    assert h[2].item() == pytest.approx(0.5)
    assert h[1].item() == pytest.approx(0.25)


def test_source_responsibility_normalization() -> None:
    h = torch.tensor([2.0, 3.0, 5.0])
    r = normalize_visual_responsibility(h, torch.tensor([0, 2]))
    assert r.tolist() == pytest.approx([2 / 7, 5 / 7])
    assert r.sum().item() == pytest.approx(1.0)


def test_multihop_ranks_above_direct_attention() -> None:
    # Visual 0 has weaker direct target attention than visual 1, but strong
    # multihop path through node 2.
    a = torch.zeros(4, 4)
    a[2, 0] = 0.9
    a[3, 2] = 0.9
    a[3, 1] = 0.2
    h = compute_answer_reachability(a, [3])
    r = normalize_visual_responsibility(h, torch.tensor([0, 1]))
    d = direct_target_attention(a, [3], torch.tensor([0, 1]))
    assert r[0] > r[1]
    assert d[1] > d[0]
