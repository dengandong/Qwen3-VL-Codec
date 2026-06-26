from __future__ import annotations

import copy
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


@dataclass
class PreparedInputs:
    text: str
    inputs: dict[str, torch.Tensor]
    video_metadata: dict[str, Any]


@dataclass
class TeacherForcedInputs:
    inputs: dict[str, torch.Tensor]
    prompt_length: int
    answer_token_ids: torch.LongTensor
    answer_positions: list[int]
    decision_target_positions: list[int]
    post_answer_target_positions: list[int]


def load_hf_model_and_processor(model_path: str, *, device: str | None = None, dtype: str = "auto"):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if dtype == "auto":
        kwargs["torch_dtype"] = "auto"
    elif dtype:
        kwargs["torch_dtype"] = getattr(torch, dtype)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, processor


def move_to_device(inputs: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def prepare_prompt_inputs(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    return_mm_token_type_ids: bool = True,
) -> PreparedInputs:
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=getattr(processor.image_processor, "patch_size", None),
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    processor_kwargs = dict(video_kwargs or {})
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=False,
        return_mm_token_type_ids=return_mm_token_type_ids,
        **processor_kwargs,
    )
    metadata = {}
    if "video_metadata" in inputs:
        metadata["video_metadata"] = inputs["video_metadata"]
    return PreparedInputs(text=text, inputs=dict(inputs), video_metadata=metadata)


def clone_model_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.clone()
        else:
            out[key] = copy.deepcopy(value)
    return out


def strip_generated_special_tokens(
    generated_ids: torch.Tensor,
    *,
    eos_token_id: int | Sequence[int] | None,
    pad_token_id: int | None,
) -> torch.LongTensor:
    ids = generated_ids.detach().cpu().to(torch.long).flatten().tolist()
    eos_ids: set[int] = set()
    if eos_token_id is not None:
        if isinstance(eos_token_id, Sequence) and not isinstance(eos_token_id, (str, bytes)):
            eos_ids = {int(x) for x in eos_token_id}
        else:
            eos_ids = {int(eos_token_id)}
    trimmed: list[int] = []
    for tok in ids:
        if pad_token_id is not None and int(tok) == int(pad_token_id):
            continue
        if int(tok) in eos_ids:
            break
        trimmed.append(int(tok))
    return torch.tensor(trimmed, dtype=torch.long)


def deterministic_generate(
    model: torch.nn.Module,
    processor: Any,
    prompt_inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
) -> tuple[str, torch.LongTensor, torch.Tensor]:
    with torch.inference_mode():
        output_ids = model.generate(
            **prompt_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    raw_generated = output_ids[0, prompt_len:].detach().cpu()
    answer_ids = strip_generated_special_tokens(
        raw_generated,
        eos_token_id=getattr(processor.tokenizer, "eos_token_id", None),
        pad_token_id=getattr(processor.tokenizer, "pad_token_id", None),
    )
    text = processor.decode(answer_ids, skip_special_tokens=True).strip()
    return text, answer_ids, output_ids.detach().cpu()


def build_teacher_forced_inputs(
    prompt_inputs: dict[str, Any],
    answer_token_ids: torch.Tensor,
    *,
    max_answer_tokens: int | None = None,
) -> TeacherForcedInputs:
    answer = answer_token_ids.detach().to(prompt_inputs["input_ids"].device, dtype=torch.long).flatten()
    if max_answer_tokens is not None:
        answer = answer[: int(max_answer_tokens)]
    if answer.numel() == 0:
        raise ValueError("baseline generated no usable answer tokens")
    inputs = clone_model_inputs(prompt_inputs)
    prompt_len = int(inputs["input_ids"].shape[1])
    answer_2d = answer.view(1, -1)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], answer_2d], dim=1)
    if "attention_mask" in inputs and isinstance(inputs["attention_mask"], torch.Tensor):
        ones = torch.ones((1, answer.numel()), dtype=inputs["attention_mask"].dtype, device=inputs["attention_mask"].device)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], ones], dim=1)
    if "mm_token_type_ids" in inputs and isinstance(inputs["mm_token_type_ids"], torch.Tensor):
        zeros = torch.zeros((1, answer.numel()), dtype=inputs["mm_token_type_ids"].dtype, device=inputs["mm_token_type_ids"].device)
        inputs["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], zeros], dim=1)
    inputs.pop("position_ids", None)
    answer_positions = list(range(prompt_len, prompt_len + int(answer.numel())))
    return TeacherForcedInputs(
        inputs=inputs,
        prompt_length=prompt_len,
        answer_token_ids=answer.detach().cpu(),
        answer_positions=answer_positions,
        decision_target_positions=[p - 1 for p in answer_positions],
        post_answer_target_positions=answer_positions,
    )


def select_target_positions(teacher: TeacherForcedInputs, mode: str) -> list[int]:
    if mode == "decision":
        return teacher.decision_target_positions
    if mode == "post_answer":
        return teacher.post_answer_target_positions
    raise ValueError(f"unsupported target mode: {mode}")


def get_video_token_id(model: torch.nn.Module, processor: Any) -> int:
    for obj in [getattr(model, "config", None), processor]:
        val = getattr(obj, "video_token_id", None)
        if val is not None:
            return int(val)
    tokenizer = getattr(processor, "tokenizer", processor)
    token = getattr(processor, "video_token", "<|video_pad|>")
    return int(tokenizer.convert_tokens_to_ids(token))


def get_spatial_merge_size(model: torch.nn.Module, processor: Any) -> int:
    cfg = getattr(getattr(model, "config", None), "vision_config", None)
    if cfg is not None and getattr(cfg, "spatial_merge_size", None) is not None:
        return int(cfg.spatial_merge_size)
    for obj in [getattr(processor, "video_processor", None), getattr(processor, "image_processor", None)]:
        val = getattr(obj, "merge_size", None)
        if val is not None:
            return int(val)
    raise RuntimeError("could not determine Qwen3-VL spatial merge size")


def model_config_summary(model: torch.nn.Module) -> dict[str, Any]:
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", None)
    vision_cfg = getattr(cfg, "vision_config", None)
    return {
        "model_class": type(model).__name__,
        "num_hidden_layers": getattr(text_cfg, "num_hidden_layers", None),
        "hidden_size": getattr(text_cfg, "hidden_size", None),
        "num_attention_heads": getattr(text_cfg, "num_attention_heads", None),
        "attn_implementation": getattr(text_cfg, "_attn_implementation", getattr(cfg, "_attn_implementation", None)),
        "video_token_id": getattr(cfg, "video_token_id", None),
        "vision_spatial_merge_size": getattr(vision_cfg, "spatial_merge_size", None),
    }


def answer_logprob(
    model: torch.nn.Module,
    teacher_inputs: dict[str, torch.Tensor],
    *,
    prompt_length: int,
    answer_token_ids: torch.Tensor,
) -> tuple[float, float]:
    with torch.inference_mode():
        outputs = model(**teacher_inputs, use_cache=False, logits_to_keep=0)
        logits = outputs.logits
    answer = answer_token_ids.to(logits.device, dtype=torch.long).flatten()
    positions = torch.arange(prompt_length - 1, prompt_length - 1 + answer.numel(), device=logits.device)
    selected_logits = logits[0].index_select(0, positions)
    log_probs = torch.log_softmax(selected_logits.float(), dim=-1)
    token_lp = log_probs.gather(1, answer.view(-1, 1)).squeeze(1)
    total = float(token_lp.sum().item())
    mean = float(token_lp.mean().item())
    return total, mean


def input_without_mm_token_type(inputs: dict[str, Any]) -> dict[str, Any]:
    # HF Qwen3VL model forward does not consume processor-side metadata fields.
    allowed = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "position_ids",
        "cache_position",
    }
    return {k: v for k, v in inputs.items() if k in allowed}
