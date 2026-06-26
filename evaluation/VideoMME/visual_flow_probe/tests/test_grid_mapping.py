from __future__ import annotations

import pytest
import torch

from evaluation.VideoMME.visual_flow_probe.flow import map_video_tokens_to_grid


def test_grid_mapping_frame_major_non_square() -> None:
    positions = torch.arange(30) + 10
    grid = torch.tensor([[3, 4, 10]])  # merge=2 => T=3, H'=2, W'=5
    mapping = map_video_tokens_to_grid(positions, grid, spatial_merge_size=2)
    assert mapping.temporal_grid_indices[:10].tolist() == [0] * 10
    assert mapping.temporal_grid_indices[10:20].tolist() == [1] * 10
    assert mapping.y_grid_indices[:5].tolist() == [0] * 5
    assert mapping.x_grid_indices[:5].tolist() == [0, 1, 2, 3, 4]
    assert mapping.y_grid_indices[5:10].tolist() == [1] * 5


def test_grid_mapping_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="count mismatch"):
        map_video_tokens_to_grid(torch.arange(3), torch.tensor([[1, 4, 4]]), spatial_merge_size=2)
