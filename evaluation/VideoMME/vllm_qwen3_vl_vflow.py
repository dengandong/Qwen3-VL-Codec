"""VFlow-guided Qwen3-VL video-token pruning for vLLM.

VFlow is an offline diagnostic signal here, not an online attention pass.  This
patch consumes per-question visual-flow responsibility ``.npz`` files and keeps
the highest-responsibility post-merger visual tokens.  The implementation uses
the same compact prompt lifecycle as the repository's existing vLLM methods:
placeholder counts, post-ViT main/deepstack rows, and sparse M-RoPE positions
are all built from one deterministic selection plan.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


_PATCHED = False
_ORIGINALS: dict[str, Any] = {}

_RESPONSIBILITY_INDEX: dict[int, Path] | None = None
_RESPONSIBILITY_ROOT: Path | None = None
_PLAN_CACHE: dict[tuple[int | None, int, int, int, int, str, str, int | None], "VFlowPlan"] = {}


@dataclass(frozen=True)
class VFlowPlan:
    keep_indices: np.ndarray
    counts_per_frame: tuple[int, ...]
    dense_token_count: int
    target_token_count: int
    source_token_count: int | None
    source_path: str | None
    score_min: float
    score_max: float
    score_mean: float


def _enabled_mode() -> str:
    return os.environ.get("QWEN3VL_VFLOW_MODE", "none").strip().lower()


def _retain_ratio() -> float:
    return float(os.environ.get("QWEN3VL_VFLOW_RETAIN_RATIO", "0.125"))


def _target_visual_tokens() -> int | None:
    raw = os.environ.get("QWEN3VL_VFLOW_TARGET_VISUAL_TOKENS", "").strip()
    if not raw:
        return None
    value = int(raw)
    return value if value > 0 else None


def _responsibility_dir() -> Path:
    raw = os.environ.get("QWEN3VL_VFLOW_RESPONSIBILITY_DIR", "").strip()
    if not raw:
        raise RuntimeError("[VFlow-vLLM] --vflow-responsibility-dir is required when VFlow is enabled")
    return Path(raw)


def _signal_name() -> str:
    return os.environ.get("QWEN3VL_VFLOW_SIGNAL", "responsibility").strip()


def _keep_policy() -> str:
    return os.environ.get("QWEN3VL_VFLOW_KEEP", "high").strip().lower()


def _debug_verify() -> bool:
    return os.environ.get("QWEN3VL_VFLOW_DEBUG_VERIFY", "0") == "1"


def _verbose() -> bool:
    return os.environ.get("QWEN3VL_VFLOW_QUIET", "0") != "1"


def _question_hash(question_id: str) -> int:
    digest = hashlib.blake2b(str(question_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def _hash_tensor_from_ids(ids: Sequence[Any]) -> torch.Tensor:
    return torch.tensor([_question_hash(str(x)) for x in ids], dtype=torch.long)


def _pop_vflow_ids(mm_kwargs: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    clean = dict(mm_kwargs or {})
    raw = clean.pop("vflow_question_ids", None)
    clean.pop("vflow_video_ids", None)
    clean.pop("vflow_durations", None)
    if raw is None:
        return clean, []
    if isinstance(raw, (str, int)):
        return clean, [str(raw)]
    return clean, [str(x) for x in raw]


def _grid_tuple(row: Any, spatial_merge_size: int) -> tuple[int, int, int, int]:
    if isinstance(row, torch.Tensor):
        values = [int(x) for x in row.detach().cpu().reshape(-1).tolist()]
    else:
        values = [int(x) for x in row]
    if len(values) != 3:
        raise ValueError(f"[VFlow-vLLM] expected video_grid_thw row [T,H,W], got {values}")
    t, h, w = values
    m = int(spatial_merge_size)
    if m <= 0:
        raise ValueError(f"[VFlow-vLLM] invalid spatial_merge_size={m}")
    if h % m != 0 or w % m != 0:
        raise RuntimeError(f"[VFlow-vLLM] grid {values} is not divisible by spatial_merge_size={m}")
    gh = h // m
    gw = w // m
    return int(t), int(gh), int(gw), int(gh * gw)


def _budget(n_tokens: int) -> int:
    explicit = _target_visual_tokens()
    if explicit is not None:
        b = int(explicit)
    else:
        ratio = _retain_ratio()
        if not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
            raise ValueError(f"[VFlow-vLLM] retain_ratio must be in (0,1], got {ratio}")
        b = int(math.floor(float(n_tokens) * ratio))
    return max(1, min(int(n_tokens), int(b)))


def _index_responsibilities() -> dict[int, Path]:
    global _RESPONSIBILITY_INDEX, _RESPONSIBILITY_ROOT
    root = _responsibility_dir().expanduser().resolve()
    if _RESPONSIBILITY_INDEX is not None and _RESPONSIBILITY_ROOT == root:
        return _RESPONSIBILITY_INDEX
    if not root.exists():
        raise FileNotFoundError(f"[VFlow-vLLM] responsibility dir not found: {root}")

    paths: list[Path]
    if root.is_file():
        paths = [root]
    else:
        direct = list(root.glob("*.npz"))
        nested = list((root / "responsibilities").glob("*.npz")) if (root / "responsibilities").exists() else []
        paths = direct + nested
        if not paths:
            paths = list(root.rglob("*.npz"))

    index: dict[int, Path] = {}
    for path in paths:
        qid = path.stem
        qhash = _question_hash(qid)
        old = index.get(qhash)
        if old is not None and old != path:
            raise RuntimeError(f"[VFlow-vLLM] responsibility id hash collision: {old} vs {path}")
        index[qhash] = path

    if not index:
        raise RuntimeError(f"[VFlow-vLLM] no .npz responsibility files found under {root}")
    _RESPONSIBILITY_ROOT = root
    _RESPONSIBILITY_INDEX = index
    if _verbose():
        print(f"[VFlow-vLLM] indexed {len(index)} responsibility files from {root}")
    return index


def _load_source_scores(question_hash: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    index = _index_responsibilities()
    path = index.get(int(question_hash))
    if path is None:
        raise FileNotFoundError(
            f"[VFlow-vLLM] no responsibility .npz for question_hash={int(question_hash)} "
            f"under {_responsibility_dir()}"
        )

    with np.load(path) as arr:
        signal = _signal_name()
        if signal not in arr:
            raise KeyError(f"[VFlow-vLLM] signal '{signal}' not found in {path}; keys={list(arr.keys())}")
        scores = np.asarray(arr[signal], dtype=np.float64)
        if scores.ndim == 2:
            scores = scores.mean(axis=0)
        if scores.ndim != 1:
            raise RuntimeError(f"[VFlow-vLLM] expected 1D or [layers,N] scores in {path}, got {scores.shape}")
        required = ("temporal_grid_indices", "y_grid_indices", "x_grid_indices")
        missing = [key for key in required if key not in arr]
        if missing:
            raise KeyError(f"[VFlow-vLLM] missing grid arrays {missing} in {path}")
        t_idx = np.asarray(arr["temporal_grid_indices"], dtype=np.int64)
        y_idx = np.asarray(arr["y_grid_indices"], dtype=np.int64)
        x_idx = np.asarray(arr["x_grid_indices"], dtype=np.int64)

    n = int(scores.shape[0])
    if any(int(a.shape[0]) != n for a in (t_idx, y_idx, x_idx)):
        raise RuntimeError(f"[VFlow-vLLM] score/grid length mismatch in {path}")
    if not np.isfinite(scores).all():
        raise RuntimeError(f"[VFlow-vLLM] non-finite responsibility values in {path}")
    return scores, t_idx, y_idx, x_idx, path


def _map_scores_to_grid(question_hash: int | None, t: int, gh: int, gw: int) -> tuple[np.ndarray, int | None, str | None]:
    n = int(t * gh * gw)
    if question_hash is None:
        # Startup profiling and dummy multimodal requests do not carry the
        # VideoMME question id.  Use a deterministic uniform profile-only plan;
        # real requests pass a hash and missing files fail loudly.
        return np.arange(n, dtype=np.float64), None, None

    src_scores, src_t, src_y, src_x, path = _load_source_scores(int(question_hash))
    src_n = int(src_scores.shape[0])
    src_t_count = int(src_t.max()) + 1 if src_n else 0
    src_h = int(src_y.max()) + 1 if src_n else 0
    src_w = int(src_x.max()) + 1 if src_n else 0
    if src_t_count <= 0 or src_h <= 0 or src_w <= 0:
        raise RuntimeError(f"[VFlow-vLLM] invalid source grid metadata in {path}")

    score_grid = np.full((src_t_count, src_h, src_w), np.nan, dtype=np.float64)
    seen = np.zeros_like(score_grid, dtype=np.bool_)
    for score, ti, yi, xi in zip(src_scores, src_t, src_y, src_x):
        if ti < 0 or yi < 0 or xi < 0 or ti >= src_t_count or yi >= src_h or xi >= src_w:
            raise RuntimeError(f"[VFlow-vLLM] invalid source grid index in {path}")
        if seen[ti, yi, xi]:
            raise RuntimeError(f"[VFlow-vLLM] duplicate source grid index {(int(ti), int(yi), int(xi))} in {path}")
        score_grid[ti, yi, xi] = float(score)
        seen[ti, yi, xi] = True
    if not seen.all():
        raise RuntimeError(f"[VFlow-vLLM] source grid in {path} is incomplete")

    local = np.arange(n, dtype=np.int64)
    tpf = gh * gw
    cur_t = local // tpf
    cur_y = (local % tpf) // gw
    cur_x = local % gw

    def scale_axis(values: np.ndarray, src_size: int, dst_size: int) -> np.ndarray:
        if dst_size <= 1 or src_size <= 1:
            return np.zeros_like(values)
        scaled = np.rint(values.astype(np.float64) * float(src_size - 1) / float(dst_size - 1))
        return np.clip(scaled.astype(np.int64), 0, src_size - 1)

    mapped = score_grid[
        scale_axis(cur_t, src_t_count, t),
        scale_axis(cur_y, src_h, gh),
        scale_axis(cur_x, src_w, gw),
    ]
    if not np.isfinite(mapped).all():
        raise RuntimeError(f"[VFlow-vLLM] mapped responsibility contains non-finite values for {path}")
    return mapped, src_n, str(path)


def _select_indices_from_scores(scores: np.ndarray, budget: int) -> np.ndarray:
    n = int(scores.shape[0])
    if int(budget) >= n:
        return np.arange(n, dtype=np.int64)
    local = np.arange(n, dtype=np.int64)
    if _keep_policy() in {"high", "top"}:
        order = np.lexsort((local, -scores))
    elif _keep_policy() in {"low", "bottom"}:
        order = np.lexsort((local, scores))
    else:
        raise ValueError(f"[VFlow-vLLM] unsupported keep policy: {_keep_policy()}")
    keep = np.sort(order[: int(budget)].astype(np.int64))
    if int(np.unique(keep).size) != int(budget):
        raise RuntimeError("[VFlow-vLLM] internal duplicate VFlow keep indices")
    return keep


def _plan_for_row(row: Any, question_hash: int | None, spatial_merge_size: int) -> VFlowPlan:
    t, gh, gw, tpf = _grid_tuple(row, spatial_merge_size)
    n = int(t * tpf)
    if n <= 0:
        raise RuntimeError(f"[VFlow-vLLM] invalid dense token count for grid={row}")
    key = (
        None if question_hash is None else int(question_hash),
        int(t),
        int(gh),
        int(gw),
        int(spatial_merge_size),
        _signal_name(),
        _keep_policy(),
        _target_visual_tokens(),
    )
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    scores, src_n, src_path = _map_scores_to_grid(question_hash, t, gh, gw)
    b = _budget(n)
    keep = _select_indices_from_scores(scores, b)
    counts = np.bincount(keep // tpf, minlength=t).astype(np.int64)
    if int(counts.sum()) != int(keep.size):
        raise RuntimeError("[VFlow-vLLM] per-frame counts do not match selected indices")
    plan = VFlowPlan(
        keep_indices=keep,
        counts_per_frame=tuple(int(x) for x in counts.tolist()),
        dense_token_count=n,
        target_token_count=int(keep.size),
        source_token_count=src_n,
        source_path=src_path,
        score_min=float(np.min(scores)),
        score_max=float(np.max(scores)),
        score_mean=float(np.mean(scores)),
    )
    _PLAN_CACHE[key] = plan
    return plan


def _hash_values(raw: Any) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, torch.Tensor):
        return [int(x) for x in raw.detach().cpu().reshape(-1).tolist()]
    if isinstance(raw, np.ndarray):
        return [int(x) for x in raw.reshape(-1).tolist()]
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    return [int(raw)]


def _hash_from_out_item(out_item: Mapping[str, Any]) -> int | None:
    if "vflow_question_hashes" not in out_item:
        return None
    data = out_item["vflow_question_hashes"].data
    values = _hash_values(data)
    return values[0] if values else None


def _hash_from_feature(mm_feature: Any) -> int | None:
    data = getattr(mm_feature, "data", {})
    if "vflow_question_hashes" not in data:
        return None
    values = _hash_values(data["vflow_question_hashes"].data)
    return values[0] if values else None


def _hashes_for_grid(question_hashes: Any, rows: int) -> list[int | None]:
    values = _hash_values(question_hashes)
    if not values:
        return [None] * int(rows)
    if len(values) < rows:
        values = values + [values[-1]] * (int(rows) - len(values))
    return values[: int(rows)]


def _selected_indices_for_grid(
    grid_thw: torch.Tensor,
    question_hashes: Any,
    spatial_merge_size: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, list[int], list[VFlowPlan]]:
    hashes = _hashes_for_grid(question_hashes, int(grid_thw.shape[0]))
    chunks: list[torch.Tensor] = []
    sizes: list[int] = []
    plans: list[VFlowPlan] = []
    offset = 0
    for row_idx, row in enumerate(grid_thw):
        plan = _plan_for_row(row, hashes[row_idx], spatial_merge_size)
        local = torch.as_tensor(plan.keep_indices, device=device, dtype=torch.long)
        chunks.append(local + int(offset))
        sizes.append(int(local.numel()))
        plans.append(plan)
        offset += int(plan.dense_token_count)
        if _verbose() and plan.target_token_count < plan.dense_token_count:
            qtag = "profile" if hashes[row_idx] is None else str(hashes[row_idx])
            print(
                f"[VFlow-vLLM] video[{row_idx}] qhash={qtag} "
                f"tokens={plan.dense_token_count}->{plan.target_token_count} "
                f"source={plan.source_token_count} score=({plan.score_min:.4g},{plan.score_max:.4g})"
            )
    if not chunks:
        return torch.zeros((0,), device=device, dtype=torch.long), sizes, plans
    return torch.cat(chunks, dim=0), sizes, plans


def _selected_sizes(grid_thw: torch.Tensor, question_hashes: Any, spatial_merge_size: int) -> list[int]:
    hashes = _hashes_for_grid(question_hashes, int(grid_thw.shape[0]))
    sizes = []
    for row_idx, row in enumerate(grid_thw):
        sizes.append(int(_plan_for_row(row, hashes[row_idx], spatial_merge_size).target_token_count))
    return sizes


def apply_patch(
    mode: str = "post_vit",
    retain_ratio: float = 0.125,
    responsibility_dir: str | os.PathLike[str] | None = None,
    target_visual_tokens: int | None = None,
    signal: str = "responsibility",
    keep: str = "high",
    debug_verify: bool = False,
    quiet: bool = False,
) -> None:
    """Patch vLLM Qwen3-VL classes in the current process."""
    global _PATCHED
    mode = (mode or "none").strip().lower()
    if mode in {"none", "off", ""}:
        return
    if mode != "post_vit":
        raise ValueError(f"[VFlow-vLLM] unsupported mode: {mode}")
    ratio = float(retain_ratio)
    if not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"[VFlow-vLLM] retain_ratio must be in (0,1], got {retain_ratio}")
    if target_visual_tokens is not None and int(target_visual_tokens) <= 0:
        target_visual_tokens = None
    if target_visual_tokens is not None and ratio != 0.125:
        raise ValueError("[VFlow-vLLM] retain_ratio and target_visual_tokens are mutually exclusive")
    if not responsibility_dir:
        raise ValueError("[VFlow-vLLM] responsibility_dir is required")
    keep = (keep or "high").strip().lower()
    if keep not in {"high", "top", "low", "bottom"}:
        raise ValueError(f"[VFlow-vLLM] unsupported keep policy: {keep}")

    os.environ["QWEN3VL_VFLOW_MODE"] = mode
    os.environ["QWEN3VL_VFLOW_RETAIN_RATIO"] = str(ratio)
    os.environ["QWEN3VL_VFLOW_TARGET_VISUAL_TOKENS"] = "" if target_visual_tokens is None else str(int(target_visual_tokens))
    os.environ["QWEN3VL_VFLOW_RESPONSIBILITY_DIR"] = str(responsibility_dir)
    os.environ["QWEN3VL_VFLOW_SIGNAL"] = str(signal)
    os.environ["QWEN3VL_VFLOW_KEEP"] = keep
    os.environ["QWEN3VL_VFLOW_DEBUG_VERIFY"] = "1" if debug_verify else "0"
    os.environ["QWEN3VL_VFLOW_QUIET"] = "1" if quiet else "0"

    import vllm.model_executor.models.qwen3_vl as qwen3_vl

    if _PATCHED:
        return

    processor_cls = qwen3_vl.Qwen3VLMultiModalProcessor
    model_cls = qwen3_vl.Qwen3VLForConditionalGeneration
    vision_cls = qwen3_vl.Qwen3_VisionTransformer

    orig_call_hf_processor = processor_cls._call_hf_processor
    orig_get_mm_fields_config = processor_cls._get_mm_fields_config
    orig_get_prompt_updates = processor_cls._get_prompt_updates
    orig_parse_video_input = model_cls._parse_and_validate_video_input
    orig_process_video_input = model_cls._process_video_input
    orig_iter_mm_grid_hw = model_cls.iter_mm_grid_hw
    orig_get_mrope_input_positions = model_cls.get_mrope_input_positions
    orig_vision_forward = vision_cls.forward

    def patched_call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        clean_mm_kwargs, question_ids = _pop_vflow_ids(mm_kwargs)
        outputs = orig_call_hf_processor(self, prompt, mm_data, clean_mm_kwargs, tok_kwargs)
        if question_ids:
            outputs["vflow_question_hashes"] = _hash_tensor_from_ids(question_ids)
        return outputs

    def patched_get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        clean_mm_kwargs, _ = _pop_vflow_ids(hf_processor_mm_kwargs)
        config = dict(orig_get_mm_fields_config(self, hf_inputs, clean_mm_kwargs))
        if "vflow_question_hashes" in hf_inputs:
            config["vflow_question_hashes"] = qwen3_vl.MultiModalFieldConfig.batched("video")
        return config

    def patched_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        if _enabled_mode() != "post_vit":
            return orig_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
        clean_mm_kwargs, _ = _pop_vflow_ids(hf_processor_mm_kwargs)
        hf_processor = self.info.get_hf_processor(**clean_mm_kwargs)
        image_processor = self.info.get_image_processor(**clean_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        hf_config = self.info.get_hf_config()

        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        vision_end_token_id = hf_config.vision_end_token_id
        merge_length = image_processor.merge_size**2

        def get_image_replacement_qwen3vl(item_idx: int):
            grid = out_mm_kwargs["image"][item_idx]["image_grid_thw"].data
            return [hf_processor.image_token_id] * (int(grid.prod()) // merge_length)

        def get_video_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            grid = out_item["video_grid_thw"].data
            video, metadata = mm_items["video"][item_idx]
            del video
            do_sample_frames = clean_mm_kwargs.get("do_sample_frames")
            sampled_fps = clean_mm_kwargs.get("fps")
            if qwen3_vl.is_list_of(sampled_fps, float):
                sampled_fps = sampled_fps[item_idx]
            timestamps = self.info._get_video_second_idx(metadata, out_item, do_sample_frames, sampled_fps)
            assert len(timestamps) == int(grid[0])

            qhash = _hash_from_out_item(out_item)
            plan = _plan_for_row(grid, qhash, int(image_processor.merge_size))
            placeholder = []
            for curr_time, count in zip(timestamps, plan.counts_per_frame):
                frame_idx = tokenizer.encode(f"<{curr_time:.1f} seconds>", add_special_tokens=False)
                placeholder.extend(frame_idx)
                placeholder.extend([vision_start_token_id] + [video_token_id] * int(count) + [vision_end_token_id])
            return qwen3_vl.PromptUpdateDetails.select_token_id(placeholder, video_token_id)

        return [
            qwen3_vl.PromptReplacement(modality="image", target=hf_processor.image_token, replacement=get_image_replacement_qwen3vl),
            qwen3_vl.PromptReplacement(
                modality="video",
                target="<|vision_start|><|video_pad|><|vision_end|>",
                replacement=get_video_replacement_qwen3vl,
            ),
        ]

    def patched_vision_forward(self, x, grid_thw, vflow_question_hashes=None):
        if _enabled_mode() != "post_vit":
            return orig_vision_forward(self, x, grid_thw)

        hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        if isinstance(grid_thw, list):
            grid_tensor = torch.tensor(grid_thw, dtype=torch.int64)
            grid_list = grid_thw
        else:
            grid_tensor = grid_thw.to(dtype=torch.int64) if isinstance(grid_thw, torch.Tensor) else torch.tensor(grid_thw, dtype=torch.int64)
            grid_list = grid_tensor.detach().cpu().tolist()

        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid_list)
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_list)
        grid_np = np.array(grid_list, dtype=np.int32)
        cu_np = np.repeat(grid_np[:, 1] * grid_np[:, 2], grid_np[:, 0]).cumsum(axis=0, dtype=np.int32)
        cu_np = np.concatenate([np.zeros(1, dtype=np.int32), cu_np])
        cu_seqlens_cpu = torch.from_numpy(cu_np)
        hidden_states = hidden_states.unsqueeze(1)
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens_cpu)
        cu_seqlens = cu_seqlens_cpu.to(self.device, non_blocking=True)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen,
            )
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature_lists.append(self.deepstack_merger_list[idx](hidden_states))

        main = self.merger(hidden_states)
        keep, _, plans = _selected_indices_for_grid(
            grid_tensor,
            vflow_question_hashes,
            int(self.spatial_merge_size),
            device=main.device,
        )
        if keep.numel() != main.shape[0]:
            main = main.index_select(0, keep)
            deepstack_feature_lists = [d.index_select(0, keep) for d in deepstack_feature_lists]

        try:
            from vllm_qwen3_vl_index_dump import dump_video_selection

            offset = 0
            for row_idx, plan in enumerate(plans):
                local = torch.as_tensor(plan.keep_indices, device=main.device, dtype=torch.long)
                dump_video_selection(
                    method="vflow",
                    video_index=row_idx,
                    grid_thw=grid_tensor[row_idx],
                    spatial_merge_size=int(self.spatial_merge_size),
                    dense_token_count=plan.dense_token_count,
                    keep_indices=local,
                    output_indices=local,
                    num_tokens_per_frame=plan.counts_per_frame,
                    extra={
                        "retain_ratio": float(_retain_ratio()),
                        "signal": _signal_name(),
                        "keep": _keep_policy(),
                        "source_path": plan.source_path,
                        "source_token_count": plan.source_token_count,
                    },
                )
                offset += plan.dense_token_count
            del offset
        except Exception as exc:
            if os.environ.get("QWEN3VL_INDEX_DUMP_STRICT", "0") == "1":
                raise
            if os.environ.get("QWEN3VL_INDEX_DUMP_DIR"):
                print(f"[VFlow-vLLM] index dump failed: {exc}")

        return torch.cat([main] + deepstack_feature_lists, dim=1)

    def patched_parse_video_input(self, **kwargs):
        if _enabled_mode() != "post_vit":
            return orig_parse_video_input(self, **kwargs)
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)
        vflow_question_hashes = kwargs.pop("vflow_question_hashes", None)
        if pixel_values_videos is None and video_embeds is None:
            return None
        return {
            "type": "pixel_values_videos" if pixel_values_videos is not None else "video_embeds",
            "pixel_values_videos": pixel_values_videos,
            "video_embeds": video_embeds,
            "video_grid_thw": video_grid_thw,
            "second_per_grid_ts": second_per_grid_ts,
            "vflow_question_hashes": vflow_question_hashes,
        }

    def patched_process_video_input(self, video_input):
        if _enabled_mode() != "post_vit":
            return orig_process_video_input(self, video_input)
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        qhashes = video_input.get("vflow_question_hashes")
        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
            if self.use_data_parallel:
                raise RuntimeError("[VFlow-vLLM] mm_encoder_tp_mode=data is not supported")
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw, vflow_question_hashes=qhashes)
        sizes = _selected_sizes(grid_thw, qhashes, int(self.visual.spatial_merge_size))
        return video_embeds.split(sizes)

    def patched_iter_mm_grid_hw(self, input_tokens, mm_features):
        if _enabled_mode() != "post_vit":
            yield from orig_iter_mm_grid_hw(self, input_tokens, mm_features)
            return
        video_token_id = self.config.video_token_id
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            offset = mm_feature.mm_position.offset
            if mm_feature.modality == "image":
                t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
                assert t == 1
                yield offset, h // spatial_merge_size, w // spatial_merge_size
            elif mm_feature.modality == "video":
                grid = mm_feature.data["video_grid_thw"].data
                qhash = _hash_from_feature(mm_feature)
                plan = _plan_for_row(grid, qhash, spatial_merge_size)
                for count in plan.counts_per_frame:
                    if int(count) <= 0:
                        continue
                    offset = input_tokens.index(video_token_id, offset)
                    yield offset, 1, int(count)
                    offset += int(count)
            else:
                raise ValueError(f"Unsupported modality: {mm_feature.modality}")

    def patched_get_mrope_input_positions(self, input_tokens, mm_features):
        if _enabled_mode() != "post_vit":
            return orig_get_mrope_input_positions(self, input_tokens, mm_features)
        video_token_id = self.config.video_token_id
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        llm_pos_ids_list = []
        st = 0
        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            if mm_feature.modality == "image":
                offset, llm_grid_h, llm_grid_w = next(self.iter_mm_grid_hw(input_tokens, [mm_feature]))
                text_len = offset - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)
                grid_indices = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
                llm_pos_ids_list.append(grid_indices + text_len + st_idx)
                st = offset + llm_grid_h * llm_grid_w
                continue

            if mm_feature.modality != "video":
                raise ValueError(f"Unsupported modality: {mm_feature.modality}")

            grid = mm_feature.data["video_grid_thw"].data
            t, _, gw, tpf = _grid_tuple(grid, spatial_merge_size)
            plan = _plan_for_row(grid, _hash_from_feature(mm_feature), spatial_merge_size)
            selected = plan.keep_indices
            offset = mm_feature.mm_position.offset
            for ti in range(t):
                frame_count = int(plan.counts_per_frame[ti])
                if frame_count <= 0:
                    continue
                offset = input_tokens.index(video_token_id, offset)
                text_len = offset - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)
                local = selected[selected // tpf == ti] % tpf
                if int(local.size) != frame_count:
                    raise RuntimeError(
                        f"[VFlow-vLLM] selected local count mismatch for frame {ti}: "
                        f"positions={int(local.size)} prompt_count={frame_count}"
                    )
                gh_idx = (local // gw).astype(np.int64)
                gw_idx = (local % gw).astype(np.int64)
                grid_indices = np.stack([np.zeros_like(gh_idx), gh_idx, gw_idx], axis=0)
                llm_pos_ids_list.append(grid_indices + text_len + st_idx)
                st = offset + int(local.size)
                offset = st

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
        return torch.from_numpy(llm_positions), mrope_position_delta

    _ORIGINALS["Qwen3VLMultiModalProcessor._call_hf_processor"] = orig_call_hf_processor
    _ORIGINALS["Qwen3VLMultiModalProcessor._get_mm_fields_config"] = orig_get_mm_fields_config
    _ORIGINALS["Qwen3VLMultiModalProcessor._get_prompt_updates"] = orig_get_prompt_updates
    _ORIGINALS["Qwen3_VisionTransformer.forward"] = orig_vision_forward
    _ORIGINALS["Qwen3VLForConditionalGeneration._parse_and_validate_video_input"] = orig_parse_video_input
    _ORIGINALS["Qwen3VLForConditionalGeneration._process_video_input"] = orig_process_video_input
    _ORIGINALS["Qwen3VLForConditionalGeneration.iter_mm_grid_hw"] = orig_iter_mm_grid_hw
    _ORIGINALS["Qwen3VLForConditionalGeneration.get_mrope_input_positions"] = orig_get_mrope_input_positions

    processor_cls._call_hf_processor = patched_call_hf_processor
    processor_cls._get_mm_fields_config = patched_get_mm_fields_config
    processor_cls._get_prompt_updates = patched_get_prompt_updates
    vision_cls.forward = patched_vision_forward
    model_cls._parse_and_validate_video_input = patched_parse_video_input
    model_cls._process_video_input = patched_process_video_input
    model_cls.iter_mm_grid_hw = patched_iter_mm_grid_hw
    model_cls.get_mrope_input_positions = patched_get_mrope_input_positions

    _PATCHED = True
    if _verbose():
        print(
            f"[VFlow-vLLM] enabled mode={mode} retain_ratio={ratio:.4f} "
            f"target={target_visual_tokens} signal={signal} keep={keep} dir={responsibility_dir}"
        )
