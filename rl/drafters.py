"""Drafters for the harness besides the EAGLE-3 head: an independent Llama used as an
argmax chain / tree proposer (PLAN D8 `ModelDrafter`). The "frozen" draft is a bf16 copy
of the step-0 policy, i.e. the static-draft baseline whose acceptance decays as the
policy drifts; `refresh` copies the current policy into it (the fixed-interval baseline
in its cheapest form: the draft is re-derived, not retrained).

Same call protocol as spec/drafter.py::Eagle3Head: (ids, hidden, kv, meta[, backend]) ->
(logits over the target vocab, hidden for the next chain step). The hidden input is
ignored (the draft model runs on its own KV) and a [N, 1] placeholder is returned.
"""

from __future__ import annotations

import torch
from torch import nn

from model.loader import load_eagle3, load_model
from model.llama import LlamaForCausalLM


class _Cfg:
    num_tapped = 3

    def __init__(self, c) -> None:
        self.c = c
        self.hidden_size, self.head_dim = c.hidden_size, c.head_dim
        self.num_attention_heads, self.num_key_value_heads = c.num_attention_heads, c.num_key_value_heads

    def rope_config(self):
        return self.c


class ModelDrafter(nn.Module):
    def __init__(self, model: LlamaForCausalLM) -> None:
        super().__init__()
        self.model = model
        self.config = _Cfg(model.config)

    @torch.no_grad()
    def forward(self, ids, hidden, kv, meta, backend=None, logits_idx=None):
        logits, _ = self.model(ids, kv, meta, backend=backend, logits_idx=logits_idx)
        return logits, torch.zeros(len(ids), 1, device=ids.device, dtype=ids.dtype)

    def to_target_ids(self, x):
        return x


def load_drafter(spec: str, target: LlamaForCausalLM, model_id: str, device, backend: str = "auto"):
    """spec: none | frozen (copy of the target's checkpoint) | model:<hf id> | eagle:<hf id>"""
    if spec == "none":
        return None
    if spec == "frozen":
        return ModelDrafter(load_model(model_id, device=device, dtype=target.lm_head.weight.dtype, backend=backend))
    kind, _, ident = spec.partition(":")
    if kind == "model":
        return ModelDrafter(load_model(ident, device=device, dtype=target.lm_head.weight.dtype, backend=backend))
    if kind == "eagle":
        return load_eagle3(ident, target, backend=backend)
    raise ValueError(f"unknown drafter {spec!r}")


def refresh(engine, drafter, state: dict[str, torch.Tensor]) -> None:
    """Copy the policy's weights into a frozen ModelDrafter (restarts running requests)."""
    assert isinstance(drafter, ModelDrafter), "only a ModelDrafter can be refreshed from policy weights"
    have = dict(drafter.named_parameters()) | dict(drafter.named_buffers())  # a tied lm_head is not listed twice
    engine.swap_drafter({f"model.{k}": v for k, v in state.items() if f"model.{k}" in have})
