from __future__ import annotations

import contextlib
import hashlib
import math
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch


def stable_int_seed(*parts: object, modulo: int = 2**63 - 1) -> int:
    h = hashlib.blake2b(digest_size=16)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "big") % modulo


def _budget_count(n: int, ratio: float) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    return max(1, min(n, int(math.floor(ratio * n + 0.5))))


def _stable_descending_topk(scores: torch.Tensor, k: int, *, largest: bool) -> torch.LongTensor:
    x = scores.detach().cpu().to(torch.float64)
    order = sorted(range(x.numel()), key=lambda i: ((-float(x[i]) if largest else float(x[i])), i))
    return torch.tensor(sorted(order[:k]), dtype=torch.long)


def select_ranked_indices(scores: torch.Tensor, ratio: float, *, condition: str) -> torch.LongTensor:
    k = _budget_count(int(scores.numel()), float(ratio))
    if condition == "top":
        return _stable_descending_topk(scores, k, largest=True)
    if condition == "bottom":
        return _stable_descending_topk(scores, k, largest=False)
    raise ValueError(f"unsupported ranked condition: {condition}")


def random_global_indices(n: int, ratio: float, *, seed: int) -> torch.LongTensor:
    k = _budget_count(n, ratio)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) % (2**63 - 1))
    idx = torch.randperm(n, generator=gen)[:k]
    return torch.sort(idx).values.to(torch.long)


def random_temporal_matched_indices(
    top_indices: torch.Tensor,
    temporal_grid_indices: torch.Tensor,
    *,
    seed: int,
) -> torch.LongTensor:
    top = top_indices.detach().cpu().to(torch.long)
    temporal = temporal_grid_indices.detach().cpu().to(torch.long)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) % (2**63 - 1))
    selected: list[torch.Tensor] = []
    for bin_id in torch.unique(temporal[top], sorted=True).tolist():
        bin_positions = torch.nonzero(temporal == int(bin_id), as_tuple=False).flatten()
        need = int((temporal[top] == int(bin_id)).sum().item())
        if need > bin_positions.numel():
            raise RuntimeError(
                f"temporal matched random cannot sample {need} tokens from bin {bin_id} "
                f"with only {bin_positions.numel()} candidates"
            )
        perm = torch.randperm(bin_positions.numel(), generator=gen)[:need]
        selected.append(bin_positions[perm])
    if not selected:
        return torch.empty(0, dtype=torch.long)
    return torch.sort(torch.cat(selected).to(torch.long)).values


def build_selection_sets(
    scores_by_type: dict[str, torch.Tensor],
    ratios: Sequence[float],
    temporal_grid_indices: torch.Tensor,
    *,
    question_id: str,
    seed: int,
    random_repeats: int,
) -> dict[str, dict[float, dict[str, list[torch.LongTensor] | torch.LongTensor]]]:
    out: dict[str, dict[float, dict[str, list[torch.LongTensor] | torch.LongTensor]]] = {}
    n = int(temporal_grid_indices.numel())
    for score_type, scores in scores_by_type.items():
        out[score_type] = {}
        for ratio in ratios:
            top = select_ranked_indices(scores, ratio, condition="top")
            bottom = select_ranked_indices(scores, ratio, condition="bottom")
            globals_: list[torch.LongTensor] = []
            matched: list[torch.LongTensor] = []
            for rep in range(random_repeats):
                globals_.append(
                    random_global_indices(
                        n,
                        ratio,
                        seed=stable_int_seed(seed, question_id, ratio, rep, "random_global"),
                    )
                )
                matched.append(
                    random_temporal_matched_indices(
                        top,
                        temporal_grid_indices,
                        seed=stable_int_seed(seed, question_id, ratio, rep, "random_temporal_matched"),
                    )
                )
            out[score_type][float(ratio)] = {
                "top": top,
                "bottom": bottom,
                "random_global": globals_,
                "random_temporal_matched": matched,
            }
    return out


def find_decoder_v_proj_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    modules: list[tuple[str, torch.nn.Module]] = []
    language_model = getattr(model, "language_model", None)
    if language_model is None and hasattr(model, "model"):
        language_model = getattr(model.model, "language_model", None)
    if language_model is not None and hasattr(language_model, "layers"):
        for idx, layer in enumerate(language_model.layers):
            attn = getattr(layer, "self_attn", None)
            v_proj = getattr(attn, "v_proj", None)
            if isinstance(v_proj, torch.nn.Module):
                modules.append((f"language_model.layers.{idx}.self_attn.v_proj", v_proj))
    if modules:
        return modules
    for name, module in model.named_modules():
        if name.endswith("self_attn.v_proj") or name.endswith(".v_proj"):
            if ".visual." not in name and not name.startswith("visual."):
                modules.append((name, module))
    return modules


@dataclass
class ZeroHookState:
    module_count: int
    modified_calls: int = 0


def _zero_tensor_positions(
    output: torch.Tensor,
    selected_positions: torch.LongTensor,
) -> tuple[torch.Tensor, bool]:
    if output.ndim < 3 or selected_positions.numel() == 0:
        return output, False
    seq_len = int(output.shape[1])
    # Cached decode usually calls v_proj with q_len=1. Do not reinterpret that
    # local index 0 as full sequence position 0.
    valid = selected_positions[selected_positions < seq_len]
    if seq_len == 1 or valid.numel() == 0:
        return output, False
    cloned = output.clone()
    cloned[:, valid.to(output.device), ...] = 0
    return cloned, True


@contextlib.contextmanager
def zero_selected_visual_values(
    model: torch.nn.Module,
    selected_seq_positions: Sequence[int] | torch.Tensor,
) -> Iterator[ZeroHookState]:
    """Temporarily zero v_proj outputs for selected full-sequence positions."""
    selected = torch.as_tensor(selected_seq_positions, dtype=torch.long, device="cpu")
    modules = find_decoder_v_proj_modules(model)
    if not modules:
        raise RuntimeError("no decoder v_proj modules found for value-zero intervention")
    state = ZeroHookState(module_count=len(modules))
    handles = []

    def hook(_module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> object:
        if isinstance(output, torch.Tensor):
            new_out, changed = _zero_tensor_positions(output, selected)
            state.modified_calls += int(changed)
            return new_out
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            new_first, changed = _zero_tensor_positions(output[0], selected)
            state.modified_calls += int(changed)
            return (new_first, *output[1:])
        return output

    try:
        for _, module in modules:
            handles.append(module.register_forward_hook(hook))
        yield state
    finally:
        for handle in handles:
            handle.remove()
