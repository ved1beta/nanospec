"""Llama-3 decoder.

Module names and op order match HF `transformers` so weights load without remapping
and greedy decode is bit-identical (G1). Attention goes through `attention()`; v0.0 is
SDPA over a contiguous `KVCache`, the paged FlashInfer path replaces that one function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    rope_scaling: dict | None = None  # llama3 params, None for plain RoPE
    tie_word_embeddings: bool = False
    bos_token_id: int | None = None
    eos_token_ids: tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def from_hf(cls, cfg: dict, generation_cfg: dict | None = None) -> "LlamaConfig":
        if cfg.get("model_type") != "llama":
            raise ValueError(f"not a llama config: model_type={cfg.get('model_type')!r}")
        if cfg.get("attention_bias") or cfg.get("mlp_bias"):
            raise ValueError("biased projections are not supported")

        # transformers>=5 writes `rope_parameters`; older configs use `rope_theta` + `rope_scaling`
        rope = dict(cfg.get("rope_parameters") or cfg.get("rope_scaling") or {})
        rope_theta = float(rope.pop("rope_theta", cfg.get("rope_theta", 10000.0)))
        rope_type = rope.pop("rope_type", rope.pop("type", "default"))
        if rope_type == "default":
            rope_scaling = None
        elif rope_type == "llama3":
            rope_scaling = rope
        else:
            raise ValueError(f"unsupported rope_type {rope_type!r}")

        heads = cfg["num_attention_heads"]
        eos = cfg.get("eos_token_id")
        if generation_cfg and generation_cfg.get("eos_token_id") is not None:
            eos = generation_cfg["eos_token_id"]
        eos_ids = tuple(eos) if isinstance(eos, (list, tuple)) else ((eos,) if eos is not None else ())

        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=heads,
            num_key_value_heads=cfg.get("num_key_value_heads", heads),
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // heads,
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
            rope_theta=rope_theta,
            max_position_embeddings=cfg["max_position_embeddings"],
            rope_scaling=rope_scaling,
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            bos_token_id=cfg.get("bos_token_id"),
            eos_token_ids=eos_ids,
        )


def eagle3_aux_layers(num_layers: int) -> tuple[int, int, int]:
    """Layers whose input residual stream EAGLE-3 taps (SGLang default; verify vs head config)."""
    return (2, num_layers // 2, num_layers - 3)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(x.dtype)


def compute_inv_freq(config: LlamaConfig) -> torch.Tensor:
    """fp32 inverse frequencies, same expression as HF. Explicit cpu so meta-device init works."""
    dim = config.head_dim
    ar = torch.arange(0, dim, 2, dtype=torch.int64, device="cpu").float()
    inv_freq = 1.0 / (config.rope_theta ** (ar / dim))
    if config.rope_scaling is None:
        return inv_freq

    s = config.rope_scaling
    factor = s["factor"]
    low_freq_factor, high_freq_factor = s["low_freq_factor"], s["high_freq_factor"]
    old_context_len = s["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama
    is_medium = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium, smoothed, inv_freq_llama)


class RotaryEmbedding(nn.Module):
    """cos/sin tables [max_pos, head_dim], built lazily per (device, dtype)."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.inv_freq = compute_inv_freq(config)
        self.max_pos = config.max_position_embeddings
        self._tables: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def tables(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        if key not in self._tables:
            t = torch.arange(self.max_pos, dtype=torch.float32, device=device)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._tables[key] = (emb.cos().to(dtype), emb.sin().to(dtype))
        return self._tables[key]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [B, H, S, D]; cos, sin: [S, D]
    cos, sin = cos[None, None], sin[None, None]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class KVCache:
    """Contiguous K/V, per layer [B, kv_heads, max_len, head_dim]. Batch-1, no paging (G1 only)."""

    def __init__(self, config: LlamaConfig, batch_size: int, max_len: int, device, dtype) -> None:
        shape = (batch_size, config.num_key_value_heads, max_len, config.head_dim)
        self.k = [torch.empty(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.v = [torch.empty(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.max_len = max_len
        self.seq_len = 0

    def rollback(self, seq_len: int) -> None:
        assert 0 <= seq_len <= self.seq_len
        self.seq_len = seq_len


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    """q: [B, H, S, D]; k, v: [B, H_kv, L, D] -> [B, H, S, D]"""
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)


class Attention(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        hs = config.hidden_size
        self.q_proj = nn.Linear(hs, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hs, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hs, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, hs, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)

        end = start_pos + S
        k_cache[:, :, start_pos:end] = k
        v_cache[:, :, start_pos:end] = v

        # is_causal is top-left aligned; multi-token appends past 0 need an explicit mask
        assert S == 1 or start_pos == 0
        out = attention(q, k_cache[:, :, :end], v_cache[:, :, :end], causal=S > 1)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, -1))


class MLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, cos, sin, k_cache, v_cache, start_pos) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, k_cache, v_cache, start_pos)
        return x + self.mlp(self.post_attention_layernorm(x))


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(
        self, input_ids: torch.Tensor, kv: KVCache, aux_layers: tuple[int, ...] = ()
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """-> (normed hidden, residual stream entering each layer in aux_layers)"""
        B, S = input_ids.shape
        start_pos = kv.seq_len
        if start_pos + S > kv.max_len:
            raise ValueError(f"kv cache full: {start_pos}+{S} > {kv.max_len}")

        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb.tables(x.device, x.dtype)
        cos, sin = cos[start_pos : start_pos + S], sin[start_pos : start_pos + S]

        aux: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            if i in aux_layers:
                aux.append(x)
            x = layer(x, cos, sin, kv.k[i], kv.v[i], start_pos)
        kv.seq_len = start_pos + S
        return self.norm(x), aux


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        kv: KVCache,
        aux_layers: tuple[int, ...] = (),
        last_only: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """input_ids [B, S] go at positions kv.seq_len..+S. -> (logits [B, S, V], aux hidden states)"""
        h, aux = self.model(input_ids, kv, aux_layers)
        if last_only:
            h = h[:, -1:]
        return self.lm_head(h), aux

    def new_kv(self, batch_size: int, max_len: int) -> KVCache:
        p = self.lm_head.weight
        return KVCache(self.config, batch_size, max_len, p.device, p.dtype)
