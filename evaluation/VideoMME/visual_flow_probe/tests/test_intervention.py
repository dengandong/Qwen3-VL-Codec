from __future__ import annotations

import torch

from evaluation.VideoMME.visual_flow_probe.interventions import zero_selected_visual_values


class FakeAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.v_proj = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v_proj(x)


class FakeLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = FakeAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn(x)


class FakeLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeLayer()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers[0](x)


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = FakeLM()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.language_model(x)


def test_value_zero_hook_prefill_positions_and_cleanup() -> None:
    model = FakeModel()
    x = torch.arange(1 * 5 * 3, dtype=torch.float32).reshape(1, 5, 3)
    with zero_selected_visual_values(model, torch.tensor([1, 3])) as state:
        out = model(x)
        assert torch.equal(out[:, 1], torch.zeros_like(out[:, 1]))
        assert torch.equal(out[:, 3], torch.zeros_like(out[:, 3]))
        assert torch.equal(out[:, 0], x[:, 0])
        assert state.modified_calls == 1
    out_after = model(x)
    assert torch.equal(out_after, x)


def test_value_zero_hook_decode_q_len_one_not_indexed_as_full_position() -> None:
    model = FakeModel()
    x = torch.ones(1, 1, 4)
    with zero_selected_visual_values(model, torch.tensor([0, 2])) as state:
        out = model(x)
        assert torch.equal(out, x)
        assert state.modified_calls == 0
