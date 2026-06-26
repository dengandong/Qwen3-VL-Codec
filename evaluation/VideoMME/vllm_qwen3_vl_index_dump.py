"""Request-scoped visual token index dump helpers for Qwen3-VL vLLM probes.

This module is intentionally inert unless ``QWEN3VL_INDEX_DUMP_DIR`` is set and
the runner sets a current request context through ``set_index_dump_context``.
Compression modules call ``dump_video_selection`` after building their normal
plan.  The helper writes local dense visual-token indices only; it does not
change model outputs or compression behavior.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


_CURRENT_CONTEXT: dict[str, Any] = {}


def _safe_name(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _tensor_list(values: torch.Tensor | Sequence[int] | None) -> list[int]:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return [int(x) for x in values.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in values]


def set_index_dump_context(
    model: Any,
    question_id: str,
    *,
    method: str,
    drop_ratio: float,
    duration: str = "short",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del model
    _CURRENT_CONTEXT.clear()
    _CURRENT_CONTEXT.update(
        {
            "question_id": str(question_id),
            "method": str(method),
            "drop_ratio": float(drop_ratio),
            "duration": str(duration),
            "extra": dict(extra or {}),
        }
    )
    return {"ok": True, "context": dict(_CURRENT_CONTEXT)}


def clear_index_dump_context(model: Any) -> dict[str, Any]:
    del model
    _CURRENT_CONTEXT.clear()
    return {"ok": True}


def dump_video_selection(
    *,
    method: str,
    video_index: int,
    grid_thw: Any,
    spatial_merge_size: int,
    dense_token_count: int,
    keep_indices: torch.Tensor | Sequence[int],
    output_indices: torch.Tensor | Sequence[int] | None = None,
    num_tokens_per_frame: Sequence[int] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    dump_dir_raw = os.environ.get("QWEN3VL_INDEX_DUMP_DIR", "").strip()
    if not dump_dir_raw or not _CURRENT_CONTEXT:
        return

    qid = _CURRENT_CONTEXT.get("question_id")
    if not qid:
        return

    dense = int(dense_token_count)
    keep = sorted(set(_tensor_list(keep_indices)))
    if any(idx < 0 or idx >= dense for idx in keep):
        raise RuntimeError(
            f"[Qwen3VL-index-dump] invalid keep index for {method} qid={qid}: "
            f"dense={dense} keep_minmax={(min(keep), max(keep)) if keep else None}"
        )
    keep_set = set(keep)
    drop = [idx for idx in range(dense) if idx not in keep_set]

    if isinstance(grid_thw, torch.Tensor):
        grid = [int(x) for x in grid_thw.detach().cpu().reshape(-1).tolist()]
    else:
        grid = [int(x) for x in grid_thw]

    record_extra: dict[str, Any] = {}
    record_extra.update(dict(_CURRENT_CONTEXT.get("extra") or {}))
    record_extra.update(dict(extra or {}))

    record = {
        "question_id": str(qid),
        "duration": _CURRENT_CONTEXT.get("duration"),
        "method": str(method),
        "context_method": _CURRENT_CONTEXT.get("method"),
        "drop_ratio": float(_CURRENT_CONTEXT.get("drop_ratio", float("nan"))),
        "video_index": int(video_index),
        "grid_thw": grid,
        "spatial_merge_size": int(spatial_merge_size),
        "dense_token_count": dense,
        "keep_count": len(keep),
        "drop_count": len(drop),
        "actual_drop_ratio": (len(drop) / dense) if dense else 0.0,
        "keep_indices": keep,
        "drop_indices": drop,
        "output_indices": _tensor_list(output_indices),
        "num_tokens_per_frame": [int(x) for x in (num_tokens_per_frame or [])],
        "extra": record_extra,
    }

    root = Path(dump_dir_raw)
    subdir = root / _safe_name(str(method)) / f"drop_{float(record['drop_ratio']):.4f}".replace(".", "p")
    subdir.mkdir(parents=True, exist_ok=True)
    suffix = "" if int(video_index) == 0 else f"_video{int(video_index)}"
    path = subdir / f"{_safe_name(qid)}{suffix}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)
