"""Codec-guided Qwen3-VL patches for vLLM.

This module monkey-patches vLLM's Qwen3-VL implementation at runtime.  It
keeps the repository self-contained while avoiding edits to site-packages.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _enabled_mode() -> str:
    return os.environ.get("QWEN3VL_CODEC_GUIDED_MODE", "off").strip().lower()


def _verbose() -> bool:
    return os.environ.get("QWEN3VL_CODEC_GUIDED_QUIET", "0") != "1"


class CodecGuideStore:
    def __init__(self, source: str | None = None) -> None:
        self.source = source or os.environ.get("QWEN3VL_CODEC_GUIDE_ZIP", "")
        self._indexed = False
        self._by_video_id: dict[str, str] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self._video_id_to_index: dict[str, int] = {}
        self._index_to_video_id: dict[int, str] = {}
        self._warned: set[str] = set()

    def set_source(self, source: str) -> None:
        if source != self.source:
            self.source = source
            self._indexed = False
            self._by_video_id.clear()
            self._meta.clear()
            self._video_id_to_index.clear()
            self._index_to_video_id.clear()
            self._warned.clear()

    def _warn_once(self, key: str, msg: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        print(msg)

    @staticmethod
    def _video_id_from_meta(meta: dict[str, Any]) -> str:
        sample_id = meta.get("sample_id")
        if sample_id:
            return str(sample_id)
        video = meta.get("video")
        return Path(str(video)).stem if video else ""

    def _index(self) -> None:
        if self._indexed:
            return
        src = Path(self.source)
        if not src.exists():
            self._warn_once("missing_source", f"[CodecGuided-vLLM] guidance source not found: {src}")
            self._indexed = True
            return

        if src.is_file() and src.suffix == ".zip":
            with zipfile.ZipFile(src) as zf:
                names = sorted(n for n in zf.namelist() if n.endswith(".meta.json"))
                for name in names:
                    with zf.open(name) as f:
                        meta = json.load(f)
                    video_id = self._video_id_from_meta(meta)
                    if not video_id:
                        continue
                    self._by_video_id.setdefault(video_id, name)
                    self._meta[name] = meta
        elif src.is_dir():
            for path in sorted(src.rglob("*.meta.json")):
                with open(path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                video_id = self._video_id_from_meta(meta)
                if not video_id:
                    continue
                self._by_video_id.setdefault(video_id, str(path))
                self._meta[str(path)] = meta
        else:
            self._warn_once("bad_source", f"[CodecGuided-vLLM] unsupported guidance source: {src}")

        for idx, video_id in enumerate(sorted(self._by_video_id)):
            self._video_id_to_index[video_id] = idx
            self._index_to_video_id[idx] = video_id

        self._indexed = True
        if _verbose():
            print(f"[CodecGuided-vLLM] indexed {len(self._by_video_id)} videos from {src}")

    def get_video_index(self, video_id: str | None) -> int:
        self._index()
        if not video_id:
            return -1
        return self._video_id_to_index.get(Path(str(video_id)).stem, -1)

    def _load_npy(self, meta_name: str, file_key: str) -> np.ndarray:
        meta = self._meta[meta_name]
        rel = meta["files"][file_key]
        src = Path(self.source)
        if src.is_file() and src.suffix == ".zip":
            base = str(Path(meta_name).parent)
            member = str(Path(base) / rel) if base != "." else rel
            with zipfile.ZipFile(src) as zf:
                return np.load(io.BytesIO(zf.read(member)), allow_pickle=False)
        return np.load(Path(meta_name).parent / rel, allow_pickle=False)

    def selected_groups_np(
        self,
        codec_video_index: int,
        grid_thw: list[int] | tuple[int, int, int] | torch.Tensor | np.ndarray,
    ) -> np.ndarray | None:
        self._index()
        video_id = self._index_to_video_id.get(int(codec_video_index))
        if video_id is None:
            return None
        meta_name = self._by_video_id.get(video_id)
        if meta_name is None:
            return None
        meta = self._meta[meta_name]
        actual = [int(x) for x in (grid_thw.detach().cpu().tolist() if isinstance(grid_thw, torch.Tensor) else grid_thw)]
        expected = meta.get("qwen_group_selection", {}).get("qwen_video_grid_thw")
        if expected is not None and [int(x) for x in expected] != actual:
            self._warn_once(
                f"grid_mismatch:{video_id}:{actual}",
                f"[CodecGuided-vLLM] grid mismatch for {video_id}: guidance={expected}, runtime={actual}; using dense tokens.",
            )
            return None
        t, h, w = actual
        m = int(meta.get("spatial_merge_size", 2))
        total_groups = int(t * (h // m) * (w // m))
        group_idx = self._load_npy(meta_name, "qwen_groupidx").astype(np.int64)
        group_idx = np.unique(group_idx[(group_idx >= 0) & (group_idx < total_groups)])
        return group_idx if group_idx.size else None


_STORE = CodecGuideStore()
_PATCHED = False


def _to_codec_indices_tensor(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32)


def _codec_index_from_feature(mm_feature: Any) -> int:
    try:
        field = mm_feature.data.get("codec_video_indices")
        data = field.data
        if isinstance(data, torch.Tensor):
            return int(data.reshape(-1)[0].item())
        return int(data)
    except Exception:
        return -1


def _selected_groups_for_row(row: Any, codec_idx: int, merge_size: int) -> np.ndarray | None:
    if codec_idx < 0:
        return None
    row_list = [int(x) for x in (row.detach().cpu().tolist() if isinstance(row, torch.Tensor) else row)]
    if row_list[0] <= 1:
        return None
    selected = _STORE.selected_groups_np(codec_idx, row_list)
    if selected is None:
        return None
    t, h, w = row_list
    groups_per_t = int((h // merge_size) * (w // merge_size))
    total_groups = int(t * groups_per_t)
    if groups_per_t <= 0:
        return None
    temporal_idx = selected // groups_per_t
    missing = np.setdiff1d(np.arange(t, dtype=np.int64), np.unique(temporal_idx), assume_unique=False)
    if missing.size:
        selected = np.unique(np.concatenate([selected, missing * groups_per_t]))
    if selected.size >= total_groups:
        return None
    return selected


def _group_counts_per_t(row: Any, codec_idx: int, merge_size: int) -> list[int] | None:
    selected = _selected_groups_for_row(row, codec_idx, merge_size)
    if selected is None:
        return None
    row_list = [int(x) for x in (row.detach().cpu().tolist() if isinstance(row, torch.Tensor) else row)]
    t, h, w = row_list
    groups_per_t = int((h // merge_size) * (w // merge_size))
    temporal_idx = selected // groups_per_t
    return np.bincount(temporal_idx, minlength=t).astype(np.int64).tolist()


def _expand_group_indices_to_patch_indices(group_idx: torch.Tensor, row: torch.Tensor, merge_size: int) -> torch.Tensor:
    m = int(merge_size)
    t, h, w = [int(x) for x in row.detach().cpu().tolist()]
    del t
    gh_total = h // m
    gw_total = w // m
    group_idx = group_idx.to(dtype=torch.long)
    gt = group_idx // (gh_total * gw_total)
    rem = group_idx % (gh_total * gw_total)
    gh = rem // gw_total
    gw = rem % gw_total
    base = gt * h * w + (gh * gw_total + gw) * (m * m)
    inner = torch.arange(m * m, dtype=torch.long, device=group_idx.device)
    return (base[:, None] + inner[None, :]).reshape(-1)


def _select_for_grid(
    grid_thw: torch.Tensor,
    codec_video_indices: torch.Tensor | None,
    merge_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    patch_chunks = []
    group_chunks = []
    cu_parts = [torch.zeros((), device=device, dtype=torch.int32)]
    patch_offset = 0
    group_offset = 0

    if codec_video_indices is None:
        codec_values = [-1] * int(grid_thw.shape[0])
    else:
        codec_values = [int(x) for x in codec_video_indices.detach().cpu().reshape(-1).tolist()]

    for row_idx, row in enumerate(grid_thw):
        t, h, w = [int(x) for x in row.detach().cpu().tolist()]
        group_num = int(t * (h // merge_size) * (w // merge_size))
        patch_num = int(t * h * w)
        codec_idx = codec_values[row_idx] if row_idx < len(codec_values) else -1
        selected_np = _selected_groups_for_row(row, codec_idx, merge_size)
        if selected_np is None:
            local_group = torch.arange(group_num, device=device, dtype=torch.long)
            local_patch = torch.arange(patch_num, device=device, dtype=torch.long)
        else:
            local_group = torch.from_numpy(selected_np).to(device=device, dtype=torch.long)
            local_patch = _expand_group_indices_to_patch_indices(local_group, row.to(device=device), merge_size)

        patch_chunks.append(local_patch + patch_offset)
        group_chunks.append(local_group + group_offset)

        if selected_np is None:
            per_t_patches = [int(h * w)] * t
        else:
            groups_per_t = int((h // merge_size) * (w // merge_size))
            temporal_idx = selected_np // groups_per_t
            per_t_groups = np.bincount(temporal_idx, minlength=t).astype(np.int64)
            per_t_patches = (per_t_groups * (merge_size * merge_size)).tolist()
        for count in per_t_patches:
            cu_parts.append(cu_parts[-1] + int(count))

        if _verbose() and selected_np is not None:
            print(
                f"[CodecGuided-vLLM] video[{row_idx}] keep_groups={local_group.numel()}/{group_num} "
                f"keep_patches={local_patch.numel()}/{patch_num}"
            )

        patch_offset += patch_num
        group_offset += group_num

    return torch.cat(patch_chunks), torch.cat(group_chunks), torch.stack(cu_parts).to(torch.int32)


def apply_patch(mode: str, guide_zip: str) -> None:
    """Patch vLLM Qwen3-VL classes in the current process."""
    global _PATCHED
    mode = (mode or "off").strip().lower()
    if mode in {"none", "off", ""}:
        return
    if mode not in {"pre_vit", "post_vit"}:
        raise ValueError(f"Unsupported codec-guided mode: {mode}")

    os.environ["QWEN3VL_CODEC_GUIDED_MODE"] = mode
    os.environ["QWEN3VL_CODEC_GUIDE_ZIP"] = guide_zip
    _STORE.set_source(guide_zip)

    import vllm.model_executor.models.qwen3_vl as qwen3_vl

    if _PATCHED:
        return

    orig_call_hf_processor = qwen3_vl.Qwen3VLMultiModalProcessor._call_hf_processor
    orig_get_mm_fields_config = qwen3_vl.Qwen3VLMultiModalProcessor._get_mm_fields_config

    def _pop_codec_video_ids(mm_kwargs):
        clean_kwargs = dict(mm_kwargs or {})
        raw_ids = clean_kwargs.pop("codec_video_ids", None)
        if raw_ids is None:
            return clean_kwargs, []
        if isinstance(raw_ids, (str, int)):
            return clean_kwargs, [str(raw_ids)]
        return clean_kwargs, [str(x) for x in raw_ids]

    def patched_call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        clean_mm_kwargs, codec_video_ids = _pop_codec_video_ids(mm_kwargs)
        codec_indices = [_STORE.get_video_index(video_id) for video_id in codec_video_ids]
        outputs = orig_call_hf_processor(self, prompt, mm_data, clean_mm_kwargs, tok_kwargs)
        if codec_indices:
            outputs["codec_video_indices"] = _to_codec_indices_tensor(codec_indices)
        return outputs

    def patched_get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        clean_mm_kwargs, _ = _pop_codec_video_ids(hf_processor_mm_kwargs)
        config = dict(orig_get_mm_fields_config(self, hf_inputs, clean_mm_kwargs))
        if "codec_video_indices" in hf_inputs:
            config["codec_video_indices"] = qwen3_vl.MultiModalFieldConfig.batched("video")
        return config

    def patched_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        clean_mm_kwargs, _ = _pop_codec_video_ids(hf_processor_mm_kwargs)
        hf_processor = self.info.get_hf_processor(**clean_mm_kwargs)
        image_processor = self.info.get_image_processor(**clean_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        hf_config = self.info.get_hf_config()

        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        vision_end_token_id = hf_config.vision_end_token_id
        merge_length = image_processor.merge_size**2

        def get_image_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            num_tokens = int(grid_thw.prod()) // merge_length
            return [hf_processor.image_token_id] * num_tokens

        def get_video_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            video, metadata = mm_items["video"][item_idx]
            del video
            do_sample_frames = clean_mm_kwargs.get("do_sample_frames")
            sampled_fps = clean_mm_kwargs.get("fps")
            if qwen3_vl.is_list_of(sampled_fps, float):
                sampled_fps = sampled_fps[item_idx]
            timestamps = self.info._get_video_second_idx(metadata, out_item, do_sample_frames, sampled_fps)
            assert len(timestamps) == int(grid_thw[0])

            codec_idx = -1
            if "codec_video_indices" in out_item:
                data = out_item["codec_video_indices"].data
                codec_idx = int(data.reshape(-1)[0].item()) if isinstance(data, torch.Tensor) else int(data)
            counts = _group_counts_per_t(grid_thw, codec_idx, int(image_processor.merge_size))
            if counts is None:
                dense_per_frame = int(grid_thw[1:].prod()) // merge_length
                counts = [dense_per_frame] * int(grid_thw[0])

            placeholder = []
            for curr_time, count in zip(timestamps, counts):
                frame_idx = tokenizer.encode(f"<{curr_time:.1f} seconds>", add_special_tokens=False)
                placeholder.extend(frame_idx)
                placeholder.extend(
                    [vision_start_token_id]
                    + [video_token_id] * int(count)
                    + [vision_end_token_id]
                )
            return qwen3_vl.PromptUpdateDetails.select_token_id(placeholder, video_token_id)

        return [
            qwen3_vl.PromptReplacement(
                modality="image",
                target=hf_processor.image_token,
                replacement=get_image_replacement_qwen3vl,
            ),
            qwen3_vl.PromptReplacement(
                modality="video",
                target="<|vision_start|><|video_pad|><|vision_end|>",
                replacement=get_video_replacement_qwen3vl,
            ),
        ]

    def patched_vision_forward(self, x, grid_thw, codec_video_indices=None):
        mode_now = _enabled_mode()
        hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)

        if isinstance(grid_thw, list):
            grid_thw_tensor = torch.tensor(grid_thw, dtype=torch.int64)
            grid_thw_list = grid_thw
        else:
            grid_thw_tensor = grid_thw.to(dtype=torch.int64) if isinstance(grid_thw, torch.Tensor) else torch.tensor(grid_thw, dtype=torch.int64)
            grid_thw_list = grid_thw_tensor.detach().cpu().tolist()

        patch_keep = None
        group_keep = None
        if mode_now == "pre_vit" and codec_video_indices is not None:
            patch_keep, group_keep, cu_seqlens = _select_for_grid(
                grid_thw_tensor,
                codec_video_indices,
                int(self.spatial_merge_size),
                hidden_states.device,
            )
            if patch_keep.numel() != hidden_states.shape[0]:
                hidden_states = hidden_states.index_select(0, patch_keep)

        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw_list)
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

        if patch_keep is not None and patch_keep.numel() != pos_embeds.shape[0]:
            pos_embeds = pos_embeds.index_select(0, patch_keep)
            rotary_pos_emb_cos = rotary_pos_emb_cos.index_select(0, patch_keep)
            rotary_pos_emb_sin = rotary_pos_emb_sin.index_select(0, patch_keep)
        else:
            grid_np = np.array(grid_thw_list, dtype=np.int32)
            cu_np = np.repeat(grid_np[:, 1] * grid_np[:, 2], grid_np[:, 0]).cumsum(axis=0, dtype=np.int32)
            cu_np = np.concatenate([np.zeros(1, dtype=np.int32), cu_np])
            cu_seqlens = torch.from_numpy(cu_np)

        hidden_states = hidden_states + pos_embeds
        hidden_states = hidden_states.unsqueeze(1)
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)
        cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)

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

        hidden_states = self.merger(hidden_states)
        if mode_now == "post_vit" and codec_video_indices is not None:
            _, group_keep, _ = _select_for_grid(
                grid_thw_tensor,
                codec_video_indices,
                int(self.spatial_merge_size),
                hidden_states.device,
            )
            if group_keep.numel() != hidden_states.shape[0]:
                hidden_states = hidden_states.index_select(0, group_keep)
                deepstack_feature_lists = [x.index_select(0, group_keep) for x in deepstack_feature_lists]

        hidden_states = torch.cat([hidden_states] + deepstack_feature_lists, dim=1)
        return hidden_states

    def patched_parse_video_input(self, **kwargs):
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)
        codec_video_indices = kwargs.pop("codec_video_indices", None)
        if pixel_values_videos is None and video_embeds is None:
            return None
        if pixel_values_videos is not None:
            return {
                "type": "pixel_values_videos",
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
                "second_per_grid_ts": second_per_grid_ts,
                "codec_video_indices": codec_video_indices,
            }
        return {
            "type": "video_embeds",
            "video_embeds": video_embeds,
            "video_grid_thw": video_grid_thw,
            "codec_video_indices": codec_video_indices,
        }

    def selected_video_sizes(grid_thw, codec_video_indices, merge_size):
        if codec_video_indices is None:
            return (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        sizes = []
        codec_values = [int(x) for x in codec_video_indices.detach().cpu().reshape(-1).tolist()]
        for row_idx, row in enumerate(grid_thw):
            dense = int(row.prod().item()) // merge_size // merge_size
            codec_idx = codec_values[row_idx] if row_idx < len(codec_values) else -1
            selected = _selected_groups_for_row(row, codec_idx, merge_size)
            sizes.append(int(selected.size) if selected is not None else dense)
        return sizes

    def patched_process_video_input(self, video_input):
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        codec_video_indices = video_input.get("codec_video_indices")
        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
            if self.use_data_parallel:
                raise RuntimeError("Codec-guided Qwen3-VL vLLM patch does not support mm_encoder_tp_mode=data yet.")
            video_embeds = self.visual(
                pixel_values_videos,
                grid_thw=grid_thw,
                codec_video_indices=codec_video_indices,
            )
        merge_size = self.visual.spatial_merge_size
        sizes = selected_video_sizes(grid_thw, codec_video_indices, merge_size)
        return video_embeds.split(sizes)

    def _iter_pruned_mm_grid_hw(input_tokens, mm_features, video_token_id, spatial_merge_size):
        """Yield pruned multimodal grid info in vLLM's newer 4-field shape."""
        mode_now = _enabled_mode()
        if mode_now not in {"pre_vit", "post_vit"}:
            raise RuntimeError("_iter_pruned_mm_grid_hw should only be used in codec-guided modes")
        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            offset = mm_feature.mm_position.offset
            if mm_feature.modality == "image":
                t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
                assert t == 1
                llm_grid_h = h // spatial_merge_size
                llm_grid_w = w // spatial_merge_size
                yield offset, llm_grid_h, llm_grid_w, llm_grid_h * llm_grid_w
            elif mm_feature.modality == "video":
                grid = mm_feature.data["video_grid_thw"].data
                t, h, w = grid.tolist()
                dense_h = h // spatial_merge_size
                dense_w = w // spatial_merge_size
                codec_idx = _codec_index_from_feature(mm_feature)
                counts = _group_counts_per_t(grid, codec_idx, spatial_merge_size)
                if counts is None:
                    counts = [dense_h * dense_w] * int(t)
                for count in counts:
                    offset = input_tokens.index(video_token_id, offset)
                    yield offset, 1, int(count), int(count)
                    offset += int(count)
            else:
                raise ValueError(f"Unsupported modality: {mm_feature.modality}")

    def patched_iter_mm_grid_hw(self, input_tokens, mm_features):
        mode_now = _enabled_mode()
        if mode_now not in {"pre_vit", "post_vit"}:
            yield from orig_iter_mm_grid_hw(self, input_tokens, mm_features)
            return
        video_token_id = self.config.video_token_id
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        for offset, llm_grid_h, llm_grid_w, _ in _iter_pruned_mm_grid_hw(
            input_tokens,
            mm_features,
            video_token_id,
            spatial_merge_size,
        ):
            yield offset, llm_grid_h, llm_grid_w

    def patched_static_iter_mm_grid_hw(
        input_tokens,
        mm_features,
        video_token_id,
        vision_start_token_id,
        vision_end_token_id,
        spatial_merge_size,
    ):
        mode_now = _enabled_mode()
        if mode_now not in {"pre_vit", "post_vit"}:
            yield from orig_static_iter_mm_grid_hw(
                input_tokens,
                mm_features,
                video_token_id,
                vision_start_token_id,
                vision_end_token_id,
                spatial_merge_size,
            )
            return
        del vision_start_token_id, vision_end_token_id
        yield from _iter_pruned_mm_grid_hw(
            input_tokens,
            mm_features,
            video_token_id,
            spatial_merge_size,
        )

    def patched_get_mrope_input_positions(self, input_tokens, mm_features):
        mode_now = _enabled_mode()
        if mode_now not in {"pre_vit", "post_vit"}:
            return orig_get_mrope_input_positions(self, input_tokens, mm_features)
        video_token_id = self.config.video_token_id
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        llm_pos_ids_list = []
        st = 0
        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            if mm_feature.modality == "image":
                offset, llm_grid_h, llm_grid_w, _ = next(
                    _iter_pruned_mm_grid_hw(
                        input_tokens,
                        [mm_feature],
                        video_token_id,
                        spatial_merge_size,
                    )
                )
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
            t, h, w = [int(x) for x in grid.tolist()]
            dense_h = h // spatial_merge_size
            dense_w = w // spatial_merge_size
            codec_idx = _codec_index_from_feature(mm_feature)
            selected = _selected_groups_for_row(grid, codec_idx, spatial_merge_size)
            offset = mm_feature.mm_position.offset
            if selected is None:
                for _ in range(t):
                    offset = input_tokens.index(video_token_id, offset)
                    text_len = offset - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)
                    grid_indices = np.indices((1, dense_h, dense_w)).reshape(3, -1)
                    llm_pos_ids_list.append(grid_indices + text_len + st_idx)
                    st = offset + dense_h * dense_w
                    offset = st
                continue

            groups_per_t = dense_h * dense_w
            for ti in range(t):
                offset = input_tokens.index(video_token_id, offset)
                text_len = offset - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)
                local = selected[selected // groups_per_t == ti] % groups_per_t
                gh = (local // dense_w).astype(np.int64)
                gw = (local % dense_w).astype(np.int64)
                grid_indices = np.stack([np.zeros_like(gh), gh, gw], axis=0)
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

    model_cls = qwen3_vl.Qwen3VLForConditionalGeneration
    orig_iter_mm_grid_hw = getattr(model_cls, "iter_mm_grid_hw", None)
    orig_static_iter_mm_grid_hw = getattr(model_cls, "_iter_mm_grid_hw", None)
    orig_get_mrope_input_positions = getattr(model_cls, "get_mrope_input_positions", None)
    if orig_iter_mm_grid_hw is None and orig_static_iter_mm_grid_hw is None:
        raise AttributeError(
            "Unsupported vLLM Qwen3-VL API: neither iter_mm_grid_hw nor "
            "_iter_mm_grid_hw exists on Qwen3VLForConditionalGeneration."
        )

    qwen3_vl.Qwen3VLMultiModalProcessor._call_hf_processor = patched_call_hf_processor
    qwen3_vl.Qwen3VLMultiModalProcessor._get_mm_fields_config = patched_get_mm_fields_config
    qwen3_vl.Qwen3VLMultiModalProcessor._get_prompt_updates = patched_get_prompt_updates
    qwen3_vl.Qwen3_VisionTransformer.forward = patched_vision_forward
    model_cls._parse_and_validate_video_input = patched_parse_video_input
    model_cls._process_video_input = patched_process_video_input
    if orig_iter_mm_grid_hw is not None:
        model_cls.iter_mm_grid_hw = patched_iter_mm_grid_hw
    if orig_static_iter_mm_grid_hw is not None:
        model_cls._iter_mm_grid_hw = staticmethod(patched_static_iter_mm_grid_hw)
    if orig_get_mrope_input_positions is not None:
        model_cls.get_mrope_input_positions = patched_get_mrope_input_positions

    _PATCHED = True
    grid_api = "iter_mm_grid_hw" if orig_iter_mm_grid_hw is not None else "_iter_mm_grid_hw"
    print(f"[CodecGuided-vLLM] enabled mode={mode} guide={guide_zip} grid_api={grid_api}")
