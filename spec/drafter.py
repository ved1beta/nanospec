"""EAGLE-3 draft head (yuhuili/EAGLE3-LLaMA3.1-Instruct-8B layout).

One modified Llama layer. Row i of a draft call pairs the target's residual stream at
position i (three tapped layers -> `fc` -> 4096) with the embedding of the token at
position i+1, and predicts the token at position i+2 over a 32k draft vocab that
`d2t` maps back into the target vocab (target_id = draft_id + d2t[draft_id]).

Chain steps feed the layer's own pre-norm output back in as the hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from kv.cache import AttnMeta, PagedKVCache
from model.llama import MLP, LlamaConfig, RMSNorm, RotaryEmbedding, apply_rope


@dataclass(frozen=True)
class Eagle3Config:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    draft_vocab_size: int
    num_tapped: int = 3

    @classmethod
    def from_hf(cls, cfg: dict, target: LlamaConfig) -> "Eagle3Config":
        heads = cfg["num_attention_heads"]
        return cls(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_attention_heads=heads,
            num_key_value_heads=cfg.get("num_key_value_heads", heads),
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // heads,
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
            rope_theta=float(cfg.get("rope_theta", 10000.0)),  # the head was trained with plain RoPE
            max_position_embeddings=target.max_position_embeddings,
            draft_vocab_size=cfg["draft_vocab_size"],
        )

    def rope_config(self) -> LlamaConfig:
        """A LlamaConfig-shaped view for RotaryEmbedding (theta, head_dim, no scaling)."""
        return LlamaConfig(
            vocab_size=0, hidden_size=self.hidden_size, intermediate_size=0, num_hidden_layers=1,
            num_attention_heads=self.num_attention_heads, num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim, rms_norm_eps=self.rms_norm_eps, rope_theta=self.rope_theta,
            max_position_embeddings=self.max_position_embeddings,
        )


class DraftAttention(nn.Module):
    """Same as the target's Attention but the projections read the 2*hidden concat."""

    def __init__(self, cfg: Eagle3Config) -> None:
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.q_proj = nn.Linear(2 * cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(2 * cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(2 * cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, kv: PagedKVCache, meta: AttnMeta, backend) -> torch.Tensor:
        N = x.shape[0]
        q = self.q_proj(x).view(N, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(N, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(N, self.n_kv_heads, self.head_dim)
        q, k = apply_rope(q, k, cos, sin)
        kv.write(0, meta.slots, k, v)
        return self.o_proj(backend.run(q, 0, kv).reshape(N, -1))


class DraftLayer(nn.Module):
    def __init__(self, cfg: Eagle3Config) -> None:
        super().__init__()
        self.self_attn = DraftAttention(cfg)
        self.mlp = MLP(cfg)  # gate/up/down over hidden_size / intermediate_size
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)  # on the embedding
        self.hidden_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)  # on the target hidden
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, embeds, hidden, cos, sin, kv, meta, backend) -> torch.Tensor:
        x = torch.cat([self.input_layernorm(embeds), self.hidden_norm(hidden)], dim=-1)
        h = hidden + self.self_attn(x, cos, sin, kv, meta, backend)
        return h + self.mlp(self.post_attention_layernorm(h))


class Eagle3Head(nn.Module):
    def __init__(self, cfg: Eagle3Config) -> None:
        super().__init__()
        self.config = cfg
        self.fc = nn.Linear(cfg.num_tapped * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.midlayer = DraftLayer(cfg)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.draft_vocab_size, bias=False)
        self.d2t = nn.Buffer(torch.zeros(cfg.draft_vocab_size, dtype=torch.int64))
        self.rotary_emb = RotaryEmbedding(cfg.rope_config())
        self.embed_tokens: nn.Embedding | None = None  # shared with the target
        self.backend = None

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor, hidden: torch.Tensor, kv: PagedKVCache, meta: AttnMeta, backend=None, logits_idx=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """input_ids [N]; hidden [N, 3*H] (target taps, first call) or [N, H] (chain).
        -> (draft logits [N or len(logits_idx), draft_vocab], pre-norm hidden [N, H] for the next chain step)"""
        if hidden.shape[-1] != self.config.hidden_size:
            hidden = self.fc(hidden)
        embeds = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb.tables(embeds.device, embeds.dtype)
        cos, sin = cos[meta.positions], sin[meta.positions]
        backend = backend or self.backend
        backend.plan(meta)
        h = self.midlayer(embeds, hidden, cos, sin, kv, meta, backend)
        hn = self.norm(h)
        return self.lm_head(hn if logits_idx is None else hn[logits_idx]), h

    def to_target_ids(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids + self.d2t[draft_ids]
