from __future__ import annotations

from typing import Any

import torch


_PATCHED = False
_ORIGINAL_FORWARD = None


def _repeat_kv_for_query_heads(k: torch.Tensor, num_heads: int) -> torch.Tensor:
    if k.shape[1] == num_heads:
        return k
    if num_heads % k.shape[1] != 0:
        raise RuntimeError(
            f"cannot repeat kv heads: query_heads={num_heads} kv_heads={k.shape[1]}"
        )
    return k.repeat_interleave(num_heads // k.shape[1], dim=1)


def compute_causal_attention_from_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    scaling: float,
) -> torch.Tensor:
    """Compute mean causal attention A[q, k] from flattened q/k.

    Shapes:
        q: [seq, num_heads * head_dim]
        k: [seq, num_kv_heads * head_dim]
        return: [seq, seq], averaged over query heads.
    """
    if q.ndim != 2 or k.ndim != 2:
        raise ValueError(f"q/k must be [seq, hidden], got {tuple(q.shape)} {tuple(k.shape)}")
    if q.shape[0] != k.shape[0]:
        raise ValueError(f"q/k sequence mismatch: {q.shape[0]} != {k.shape[0]}")
    seq_len = int(q.shape[0])
    qh = q.float().view(seq_len, num_heads, head_dim)
    kh = k.float().view(seq_len, k.shape[-1] // head_dim, head_dim)
    kh = _repeat_kv_for_query_heads(kh, num_heads)
    logits = torch.einsum("qhd,khd->hqk", qh, kh) * float(scaling)
    mask = torch.triu(torch.ones((seq_len, seq_len), device=logits.device, dtype=torch.bool), diagonal=1)
    logits = logits.masked_fill(mask.unsqueeze(0), torch.finfo(logits.dtype).min)
    return torch.softmax(logits, dim=-1).mean(dim=0).detach().cpu()


def _get_probe_state(module: torch.nn.Module) -> dict[str, Any]:
    state = getattr(module, "_visual_flow_probe_state", None)
    if state is None:
        state = {}
        setattr(module, "_visual_flow_probe_state", state)
    return state


def install_vllm_qwen3_probe_patch() -> bool:
    """Patch vLLM Qwen3Attention.forward in the current worker process.

    This is runtime-only and stores all mutable state on individual attention
    modules. It is intentionally disabled unless enable_* helpers set module
    state through LLM.apply_model.
    """
    global _PATCHED, _ORIGINAL_FORWARD
    if _PATCHED:
        return False

    from vllm.model_executor.models.qwen3 import Qwen3Attention

    _ORIGINAL_FORWARD = Qwen3Attention.forward

    def patched_forward(self: torch.nn.Module, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        state = _get_probe_state(self)
        zero_positions = state.get("zero_positions")
        if zero_positions is not None and v.ndim == 2 and int(v.shape[0]) > 1:
            selected = torch.as_tensor(zero_positions, dtype=torch.long, device=v.device)
            selected = selected[(selected >= 0) & (selected < int(v.shape[0]))]
            if selected.numel() > 0:
                v = v.clone()
                v.index_fill_(0, selected, 0)
                state["zero_modified_calls"] = int(state.get("zero_modified_calls", 0)) + 1

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)

        if state.get("capture_enabled") and q.ndim == 2 and int(q.shape[0]) > 1:
            seq_len = int(q.shape[0])
            max_seq_len = int(state.get("capture_max_seq_len", 0) or 0)
            if max_seq_len <= 0 or seq_len <= max_seq_len:
                attn = compute_causal_attention_from_qk(
                    q,
                    k,
                    num_heads=int(self.num_heads),
                    head_dim=int(self.head_dim),
                    scaling=float(self.scaling),
                )
                acc = state.get("capture_sum")
                if acc is None:
                    state["capture_sum"] = attn
                else:
                    if tuple(acc.shape) != tuple(attn.shape):
                        raise RuntimeError(
                            f"attention capture shape changed: {tuple(acc.shape)} -> {tuple(attn.shape)}"
                        )
                    state["capture_sum"] = acc + attn
                state["capture_count"] = int(state.get("capture_count", 0)) + 1
                state["capture_seq_len"] = seq_len

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    Qwen3Attention.forward = patched_forward
    _PATCHED = True
    return True


def _iter_qwen3_attn_modules(model: torch.nn.Module):
    lm = getattr(model, "language_model", None)
    if lm is not None:
        lm_model = getattr(lm, "model", None)
        layers = getattr(lm_model, "layers", None)
        if layers is not None:
            for idx, layer in enumerate(layers):
                attn = getattr(layer, "self_attn", None)
                if isinstance(attn, torch.nn.Module):
                    yield idx, attn
            return
    for name, module in model.named_modules():
        if name.endswith(".self_attn") and hasattr(module, "qkv_proj") and hasattr(module, "attn"):
            idx = -1
            parts = name.split(".")
            for part in parts:
                if part.isdigit():
                    idx = int(part)
            yield idx, module


def install_probe_patch_on_model(model: torch.nn.Module) -> dict[str, Any]:
    patched_now = install_vllm_qwen3_probe_patch()
    count = 0
    for idx, module in _iter_qwen3_attn_modules(model):
        state = _get_probe_state(module)
        state["layer_idx"] = int(idx)
        count += 1
    return {"patched_now": patched_now, "attention_modules": count}


def enable_attention_capture_on_model(
    model: torch.nn.Module,
    layer_indices: list[int],
    max_seq_len: int,
) -> dict[str, Any]:
    install_vllm_qwen3_probe_patch()
    wanted = {int(i) for i in layer_indices}
    enabled = []
    for idx, module in _iter_qwen3_attn_modules(model):
        state = _get_probe_state(module)
        state["capture_enabled"] = int(idx) in wanted
        state["capture_max_seq_len"] = int(max_seq_len)
        state["capture_sum"] = None
        state["capture_count"] = 0
        state["capture_seq_len"] = None
        if state["capture_enabled"]:
            enabled.append(int(idx))
    if not enabled:
        raise RuntimeError(f"no vLLM Qwen3 attention layers matched {sorted(wanted)}")
    return {"enabled_layers": enabled}


def get_and_clear_attention_capture_on_model(model: torch.nn.Module) -> dict[str, Any]:
    total = None
    count = 0
    layers = []
    seq_len = None
    for idx, module in _iter_qwen3_attn_modules(model):
        state = _get_probe_state(module)
        if state.get("capture_enabled") and state.get("capture_count", 0):
            attn = state.get("capture_sum")
            if attn is not None:
                total = attn if total is None else total + attn
                count += int(state.get("capture_count", 0))
                layers.append(int(idx))
                seq_len = state.get("capture_seq_len", seq_len)
        state["capture_enabled"] = False
        state["capture_sum"] = None
        state["capture_count"] = 0
    if total is None or count <= 0:
        raise RuntimeError("vLLM attention patch captured no decoder attention")
    return {
        "attention": (total / float(count)).cpu(),
        "captured_layers": layers,
        "captured_calls": count,
        "seq_len": int(seq_len) if seq_len is not None else int(total.shape[0]),
    }


def get_and_clear_layer_attention_capture_on_model(model: torch.nn.Module) -> dict[str, Any]:
    """Return one averaged attention matrix per captured decoder layer.

    The existing aggregate getter intentionally averages all selected layers.
    Layer-dynamics diagnostics need the same head-averaged A[q, k] matrices
    before that cross-layer average. Tensors are CPU tensors so vLLM worker RPC
    does not keep GPU memory alive after the prefill pass.
    """
    by_layer: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    seq_len = None
    for idx, module in _iter_qwen3_attn_modules(model):
        state = _get_probe_state(module)
        if state.get("capture_enabled") and state.get("capture_count", 0):
            attn = state.get("capture_sum")
            count = int(state.get("capture_count", 0))
            if attn is not None and count > 0:
                layer = int(idx)
                by_layer[layer] = (attn / float(count)).cpu()
                counts[layer] = count
                seq_len = state.get("capture_seq_len", seq_len)
        state["capture_enabled"] = False
        state["capture_sum"] = None
        state["capture_count"] = 0
    if not by_layer:
        raise RuntimeError("vLLM attention patch captured no per-layer decoder attention")
    return {
        "attentions_by_layer": by_layer,
        "capture_counts_by_layer": counts,
        "captured_layers": sorted(by_layer),
        "captured_calls": int(sum(counts.values())),
        "seq_len": int(seq_len) if seq_len is not None else int(next(iter(by_layer.values())).shape[0]),
    }


def clear_attention_capture_on_model(model: torch.nn.Module) -> dict[str, Any]:
    for _, module in _iter_qwen3_attn_modules(model):
        state = _get_probe_state(module)
        state["capture_enabled"] = False
        state["capture_sum"] = None
        state["capture_count"] = 0
    return {"cleared": True}


def set_zero_positions_on_model(model: torch.nn.Module, positions: list[int]) -> dict[str, Any]:
    install_vllm_qwen3_probe_patch()
    positions = [int(p) for p in positions]
    count = 0
    for _, module in _iter_qwen3_attn_modules(model):
        state = _get_probe_state(module)
        state["zero_positions"] = positions
        state["zero_modified_calls"] = 0
        count += 1
    return {"module_count": count, "positions": len(positions)}


def get_and_clear_zero_state_on_model(model: torch.nn.Module) -> dict[str, Any]:
    total_modified = 0
    module_count = 0
    for _, module in _iter_qwen3_attn_modules(model):
        state = _get_probe_state(module)
        total_modified += int(state.get("zero_modified_calls", 0))
        state["zero_positions"] = None
        state["zero_modified_calls"] = 0
        module_count += 1
    return {"module_count": module_count, "modified_calls": total_modified}
