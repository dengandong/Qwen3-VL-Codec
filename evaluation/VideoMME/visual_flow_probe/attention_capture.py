from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch

from .flow import assert_causal_attention


def get_language_model(model: torch.nn.Module) -> torch.nn.Module:
    language_model = getattr(model, "language_model", None)
    if language_model is None and hasattr(model, "model"):
        language_model = getattr(model.model, "language_model", None)
    if language_model is None:
        raise RuntimeError("could not locate Qwen3-VL language_model")
    return language_model


def find_decoder_self_attention_modules(model: torch.nn.Module) -> list[tuple[int, str, torch.nn.Module]]:
    language_model = get_language_model(model)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise RuntimeError("language_model.layers is unavailable")
    modules: list[tuple[int, str, torch.nn.Module]] = []
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, torch.nn.Module):
            modules.append((idx, f"language_model.layers.{idx}.self_attn", attn))
    if not modules:
        raise RuntimeError("no decoder self-attention modules found")
    return modules


def default_layer_indices(num_layers: int, *, start: int | None = None, end: int | None = None, stride: int = 1) -> list[int]:
    if stride <= 0:
        raise ValueError("layer stride must be positive")
    lo = num_layers // 3 if start is None else int(start)
    hi = (2 * num_layers) // 3 if end is None else int(end)
    lo = max(0, min(num_layers, lo))
    hi = max(lo + 1, min(num_layers, hi))
    return list(range(lo, hi, stride))


@contextlib.contextmanager
def force_eager_attention(model: torch.nn.Module) -> Iterator[None]:
    """Temporarily force eager attention so attention probabilities are returned."""
    configs = []
    for obj in [getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)]:
        if obj is not None and hasattr(obj, "_attn_implementation"):
            configs.append((obj, getattr(obj, "_attn_implementation")))
            setattr(obj, "_attn_implementation", "eager")
    try:
        yield
    finally:
        for obj, value in configs:
            setattr(obj, "_attn_implementation", value)


@dataclass
class AttentionCaptureResult:
    attention: torch.Tensor
    layer_indices: list[int]
    seq_len: int
    captured_layers: int


class _AttentionAccumulator:
    def __init__(self) -> None:
        self.sum_attention: torch.Tensor | None = None
        self.count = 0
        self.non_none_seen = False

    def add(self, attn: torch.Tensor) -> None:
        if attn.ndim != 4:
            raise RuntimeError(f"attention hook expected [batch, heads, query, key], got {tuple(attn.shape)}")
        if attn.shape[0] != 1:
            raise RuntimeError(f"attention capture requires batch size 1, got {attn.shape[0]}")
        mean_attn = attn.detach().float().mean(dim=1).squeeze(0).cpu()
        if mean_attn.shape[0] != mean_attn.shape[1]:
            raise RuntimeError(f"flow pass expected square attention, got {tuple(mean_attn.shape)}")
        if self.sum_attention is None:
            self.sum_attention = torch.zeros_like(mean_attn)
        self.sum_attention += mean_attn
        self.count += 1
        self.non_none_seen = True

    def result(self) -> torch.Tensor:
        if self.sum_attention is None or self.count == 0:
            raise RuntimeError(
                "decoder attention hooks did not receive attention weights. "
                "Use eager attention and output_attentions=True."
            )
        return self.sum_attention / float(self.count)


def capture_averaged_decoder_attention(
    model: torch.nn.Module,
    model_inputs: dict[str, torch.Tensor],
    *,
    layer_indices: Sequence[int],
    causal_atol: float = 1e-4,
) -> AttentionCaptureResult:
    """Capture averaged decoder self-attention A[q, k]."""
    modules = find_decoder_self_attention_modules(model)
    want = set(int(i) for i in layer_indices)
    selected = [(idx, name, mod) for idx, name, mod in modules if idx in want]
    if not selected:
        raise RuntimeError(f"no selected attention layers found for indices={sorted(want)}")

    accumulator = _AttentionAccumulator()
    handles = []

    def hook(_module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        attn = None
        if isinstance(output, tuple) and len(output) >= 2:
            attn = output[1]
        elif hasattr(output, "attentions"):
            attn = output.attentions
        if attn is not None:
            accumulator.add(attn)

    try:
        for _, _, module in selected:
            handles.append(module.register_forward_hook(hook))
        with force_eager_attention(model):
            with torch.inference_mode():
                _ = model(**model_inputs, use_cache=False, output_attentions=True, logits_to_keep=1)
        attention = accumulator.result()
        assert_causal_attention(attention, atol=causal_atol)
        return AttentionCaptureResult(
            attention=attention,
            layer_indices=[idx for idx, _, _ in selected],
            seq_len=int(attention.shape[0]),
            captured_layers=accumulator.count,
        )
    finally:
        for handle in handles:
            handle.remove()
