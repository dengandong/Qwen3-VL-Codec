from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from .flow import assert_causal_attention
from .vllm_attention_patch import (
    clear_attention_capture_on_model,
    enable_attention_capture_on_model,
    get_and_clear_attention_capture_on_model,
    get_and_clear_layer_attention_capture_on_model,
    get_and_clear_zero_state_on_model,
    install_probe_patch_on_model,
    set_zero_positions_on_model,
)


@dataclass
class VLLMPreparedPrompt:
    text: str
    vllm_input: dict[str, Any]
    input_ids: torch.LongTensor
    video_grid_thw: torch.LongTensor
    video_metadata: list[dict[str, Any]]


@dataclass
class VLLMGeneration:
    text: str
    token_ids: torch.LongTensor
    prompt_token_ids: torch.LongTensor
    raw_output: Any


@dataclass
class VLLMTeacherForced:
    text: str
    vllm_input: dict[str, Any]
    prompt_length: int
    answer_token_ids: torch.LongTensor
    answer_positions: list[int]
    decision_target_positions: list[int]
    post_answer_target_positions: list[int]
    input_ids: torch.LongTensor
    video_grid_thw: torch.LongTensor


@dataclass
class VLLMAttentionCapture:
    attention: torch.Tensor
    layer_indices: list[int]
    seq_len: int
    captured_layers: list[int]
    captured_calls: int
    prompt_token_ids: torch.LongTensor


@dataclass
class VLLMLayerAttentionCapture:
    attentions_by_layer: dict[int, torch.Tensor]
    layer_indices: list[int]
    seq_len: int
    captured_layers: list[int]
    captured_calls: int
    capture_counts_by_layer: dict[int, int]
    prompt_token_ids: torch.LongTensor


def _first_worker_result(results: Any) -> Any:
    if isinstance(results, list):
        if not results:
            raise RuntimeError("vLLM worker RPC returned an empty result list")
        return results[0]
    return results


def load_vllm_and_processor(args: Any):
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # The probe uses LLM.apply_model to install request-scoped diagnostics in
    # the local offline worker. vLLM requires this opt-in for callable RPCs.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from transformers import AutoProcessor
    from vllm import LLM

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    llm_kwargs = dict(
        model=args.model_path,
        tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.8),
        trust_remote_code=True,
        max_model_len=getattr(args, "max_model_len", 131072),
        limit_mm_per_prompt={"video": 1},
        seed=args.seed,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    llm = LLM(**llm_kwargs)
    install_info = _first_worker_result(llm.apply_model(install_probe_patch_on_model))
    if not install_info.get("attention_modules"):
        raise RuntimeError(f"vLLM Qwen3 attention patch installed no modules: {install_info}")
    return llm, processor


def prepare_vllm_prompt(processor: Any, messages: list[dict[str, Any]]) -> VLLMPreparedPrompt:
    """Prepare vLLM multimodal input and local token/grid diagnostics."""
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=getattr(processor.image_processor, "patch_size", None),
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    video_kwargs = dict(video_kwargs or {})
    if not video_inputs:
        raise RuntimeError("qwen_vl_utils returned no decoded video inputs")

    # qwen-vl-utils returns [(frames, metadata)] for videos. vLLM consumes that
    # directly. The HF processor metadata path in this environment cannot,
    # but it can compute local input_ids/grid from the frames tensor alone.
    frames_only = []
    metadata: list[dict[str, Any]] = []
    for item in video_inputs:
        if isinstance(item, tuple) and len(item) == 2:
            frames_only.append(item[0])
            metadata.append(dict(item[1]))
        else:
            frames_only.append(item)
            metadata.append({})

    diag_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=frames_only,
        return_tensors="pt",
        padding=False,
        return_mm_token_type_ids=True,
        do_resize=False,
        **video_kwargs,
    )
    video_grid_thw = diag_inputs.get("video_grid_thw")
    if video_grid_thw is None:
        raise RuntimeError("processor did not return video_grid_thw for vLLM probe")
    vllm_video_kwargs = dict(video_kwargs)
    vllm_video_kwargs["do_resize"] = False
    mm_data: dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    mm_data["video"] = video_inputs
    return VLLMPreparedPrompt(
        text=text,
        vllm_input={
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": vllm_video_kwargs,
        },
        input_ids=diag_inputs["input_ids"].detach().cpu().to(torch.long),
        video_grid_thw=video_grid_thw.detach().cpu().to(torch.long),
        video_metadata=metadata,
    )


def _make_sampling_params(*, max_tokens: int, prompt_logprobs: int | None = None):
    from vllm import SamplingParams

    kwargs: dict[str, Any] = dict(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop_token_ids=[],
    )
    if prompt_logprobs is not None:
        kwargs["prompt_logprobs"] = prompt_logprobs
    return SamplingParams(**kwargs)


def deterministic_vllm_generate(llm: Any, prompt_input: dict[str, Any], *, max_new_tokens: int) -> VLLMGeneration:
    outputs = llm.generate([prompt_input], sampling_params=_make_sampling_params(max_tokens=max_new_tokens))
    out = outputs[0]
    comp = out.outputs[0]
    token_ids = torch.tensor(list(getattr(comp, "token_ids", []) or []), dtype=torch.long)
    prompt_ids = torch.tensor(list(getattr(out, "prompt_token_ids", []) or []), dtype=torch.long)
    return VLLMGeneration(
        text=str(getattr(comp, "text", "")).split("</think>")[-1].strip(),
        token_ids=token_ids,
        prompt_token_ids=prompt_ids,
        raw_output=out,
    )


def build_vllm_teacher_forced(
    prepared: VLLMPreparedPrompt,
    baseline_text: str,
    prompt_token_ids: torch.Tensor,
    answer_token_ids: torch.Tensor,
    *,
    max_answer_tokens: int | None,
) -> VLLMTeacherForced:
    answer = answer_token_ids.detach().cpu().to(torch.long).flatten()
    if max_answer_tokens is not None:
        answer = answer[: int(max_answer_tokens)]
    if answer.numel() == 0:
        raise ValueError("baseline generated no usable answer tokens")
    prompt_len = int(prompt_token_ids.numel())
    teacher_text = prepared.text + str(baseline_text)
    teacher_input = dict(prepared.vllm_input)
    teacher_input["prompt"] = teacher_text
    answer_positions = list(range(prompt_len, prompt_len + int(answer.numel())))
    input_ids = torch.cat([prompt_token_ids.detach().cpu().to(torch.long), answer], dim=0).view(1, -1)
    return VLLMTeacherForced(
        text=teacher_text,
        vllm_input=teacher_input,
        prompt_length=prompt_len,
        answer_token_ids=answer,
        answer_positions=answer_positions,
        decision_target_positions=[p - 1 for p in answer_positions],
        post_answer_target_positions=answer_positions,
        input_ids=input_ids,
        video_grid_thw=prepared.video_grid_thw,
    )


def select_vllm_target_positions(teacher: VLLMTeacherForced, mode: str) -> list[int]:
    if mode == "decision":
        return teacher.decision_target_positions
    if mode == "post_answer":
        return teacher.post_answer_target_positions
    raise ValueError(f"unsupported target mode: {mode}")


def capture_vllm_decoder_attention(
    llm: Any,
    teacher_input: dict[str, Any],
    *,
    layer_indices: list[int],
    max_seq_len: int,
    causal_atol: float,
) -> VLLMAttentionCapture:
    llm.apply_model(lambda model: enable_attention_capture_on_model(model, list(layer_indices), int(max_seq_len)))
    try:
        # One generated token is enough to force a full prompt prefill.
        outputs = llm.generate([teacher_input], sampling_params=_make_sampling_params(max_tokens=1))
        prompt_token_ids = torch.tensor(list(getattr(outputs[0], "prompt_token_ids", []) or []), dtype=torch.long)
        result = _first_worker_result(llm.apply_model(get_and_clear_attention_capture_on_model))
    except Exception:
        try:
            llm.apply_model(clear_attention_capture_on_model)
        finally:
            pass
        raise
    attention = result["attention"].detach().cpu().float()
    assert_causal_attention(attention, atol=causal_atol)
    return VLLMAttentionCapture(
        attention=attention,
        layer_indices=list(layer_indices),
        seq_len=int(result.get("seq_len", attention.shape[0])),
        captured_layers=[int(x) for x in result.get("captured_layers", [])],
        captured_calls=int(result.get("captured_calls", len(result.get("captured_layers", [])))),
        prompt_token_ids=prompt_token_ids,
    )


def capture_vllm_decoder_attention_by_layer(
    llm: Any,
    teacher_input: dict[str, Any],
    *,
    layer_indices: list[int],
    max_seq_len: int,
    causal_atol: float,
) -> VLLMLayerAttentionCapture:
    """Capture head-averaged decoder attention separately for each layer."""
    llm.apply_model(lambda model: enable_attention_capture_on_model(model, list(layer_indices), int(max_seq_len)))
    try:
        outputs = llm.generate([teacher_input], sampling_params=_make_sampling_params(max_tokens=1))
        prompt_token_ids = torch.tensor(list(getattr(outputs[0], "prompt_token_ids", []) or []), dtype=torch.long)
        result = _first_worker_result(llm.apply_model(get_and_clear_layer_attention_capture_on_model))
    except Exception:
        try:
            llm.apply_model(clear_attention_capture_on_model)
        finally:
            pass
        raise
    raw = result["attentions_by_layer"]
    attentions: dict[int, torch.Tensor] = {}
    for layer, attention in raw.items():
        layer_idx = int(layer)
        attn = attention.detach().cpu().float()
        assert_causal_attention(attn, atol=causal_atol)
        attentions[layer_idx] = attn
    return VLLMLayerAttentionCapture(
        attentions_by_layer=attentions,
        layer_indices=list(layer_indices),
        seq_len=int(result.get("seq_len", next(iter(attentions.values())).shape[0])),
        captured_layers=[int(x) for x in result.get("captured_layers", sorted(attentions))],
        captured_calls=int(result.get("captured_calls", len(attentions))),
        capture_counts_by_layer={int(k): int(v) for k, v in result.get("capture_counts_by_layer", {}).items()},
        prompt_token_ids=prompt_token_ids,
    )


def _prompt_logprob_for_answer(
    output: Any,
    *,
    prompt_length: int,
    answer_token_ids: torch.Tensor,
) -> tuple[float, float]:
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if prompt_logprobs is None:
        raise RuntimeError("vLLM output did not include prompt_logprobs")
    answer = answer_token_ids.detach().cpu().to(torch.long).flatten().tolist()
    vals: list[float] = []
    for offset, tok in enumerate(answer):
        pos = prompt_length + offset
        if pos >= len(prompt_logprobs):
            raise RuntimeError(
                f"prompt_logprobs too short for answer pos={pos}, len={len(prompt_logprobs)}"
            )
        entry = prompt_logprobs[pos]
        lp = None
        if isinstance(entry, dict):
            item = entry.get(int(tok))
            if item is not None:
                lp = getattr(item, "logprob", item)
        if lp is None:
            raise RuntimeError(f"vLLM prompt_logprobs missing answer token {tok} at prompt position {pos}")
        vals.append(float(lp))
    if not vals:
        raise RuntimeError("no answer logprobs extracted")
    total = float(sum(vals))
    return total, total / float(len(vals))


def vllm_answer_logprob(
    llm: Any,
    teacher_input: dict[str, Any],
    *,
    prompt_length: int,
    answer_token_ids: torch.Tensor,
) -> tuple[float, float]:
    outputs = llm.generate([teacher_input], sampling_params=_make_sampling_params(max_tokens=1, prompt_logprobs=1))
    return _prompt_logprob_for_answer(outputs[0], prompt_length=prompt_length, answer_token_ids=answer_token_ids)


def vllm_zero_generate_and_logprob(
    llm: Any,
    prompt_input: dict[str, Any],
    teacher_input: dict[str, Any],
    selected_seq_positions: torch.Tensor,
    *,
    max_new_tokens: int,
    prompt_length: int,
    answer_token_ids: torch.Tensor,
) -> tuple[VLLMGeneration, float, float, dict[str, Any]]:
    positions = [int(x) for x in selected_seq_positions.detach().cpu().flatten().tolist()]
    llm.apply_model(lambda model: set_zero_positions_on_model(model, positions))
    try:
        gen = deterministic_vllm_generate(llm, prompt_input, max_new_tokens=max_new_tokens)
        lp_total, lp_mean = vllm_answer_logprob(
            llm,
            teacher_input,
            prompt_length=prompt_length,
            answer_token_ids=answer_token_ids,
        )
    finally:
        state = _first_worker_result(llm.apply_model(get_and_clear_zero_state_on_model))
    if int(state.get("modified_calls", 0)) <= 0:
        raise RuntimeError("vLLM value-zero patch installed but modified no qkv V slices")
    return gen, lp_total, lp_mean, state


def get_vllm_model_config_summary(llm: Any) -> dict[str, Any]:
    cfg = getattr(getattr(llm, "llm_engine", None), "model_config", None)
    hf_cfg = getattr(cfg, "hf_config", None)
    text_cfg = getattr(hf_cfg, "text_config", None)
    vision_cfg = getattr(hf_cfg, "vision_config", None)
    return {
        "backend": "vllm",
        "model_class": getattr(hf_cfg, "architectures", None),
        "num_hidden_layers": getattr(text_cfg, "num_hidden_layers", None),
        "hidden_size": getattr(text_cfg, "hidden_size", None),
        "num_attention_heads": getattr(text_cfg, "num_attention_heads", None),
        "video_token_id": getattr(hf_cfg, "video_token_id", None),
        "vision_spatial_merge_size": getattr(vision_cfg, "spatial_merge_size", None),
    }


def get_vllm_video_token_id(processor: Any, llm: Any | None = None) -> int:
    if llm is not None:
        cfg = getattr(getattr(getattr(llm, "llm_engine", None), "model_config", None), "hf_config", None)
        val = getattr(cfg, "video_token_id", None)
        if val is not None:
            return int(val)
    tokenizer = getattr(processor, "tokenizer", processor)
    token = getattr(processor, "video_token", "<|video_pad|>")
    return int(tokenizer.convert_tokens_to_ids(token))


def get_vllm_spatial_merge_size(processor: Any, llm: Any | None = None) -> int:
    if llm is not None:
        cfg = getattr(getattr(getattr(llm, "llm_engine", None), "model_config", None), "hf_config", None)
        vision_cfg = getattr(cfg, "vision_config", None)
        val = getattr(vision_cfg, "spatial_merge_size", None)
        if val is not None:
            return int(val)
    for obj in [getattr(processor, "video_processor", None), getattr(processor, "image_processor", None)]:
        val = getattr(obj, "merge_size", None)
        if val is not None:
            return int(val)
    raise RuntimeError("could not determine Qwen3-VL spatial merge size")
