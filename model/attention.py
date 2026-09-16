"""Attention backends over the paged cache. `plan()` once per step, `run()` per layer.

q: [N, H, D] -> [N, H, D]. K/V for this step are already written to the cache.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kv.cache import AttnMeta, PagedKVCache


class SdpaBackend:
    """Reference path: gather each request's blocks and run torch SDPA. Any device."""

    def plan(self, meta: AttnMeta) -> None:
        self.meta = meta

    def run(self, q: torch.Tensor, layer: int, kv: PagedKVCache) -> torch.Tensor:
        m = self.meta
        out = torch.empty_like(q)
        for i, (kv_len, q_len) in enumerate(zip(m.seq_lens, m.qo_lens)):
            qs, qe = int(m.qo_indptr[i]), int(m.qo_indptr[i + 1])
            blocks = m.kv_indices[m.kv_indptr[i] : m.kv_indptr[i + 1]].long()
            k, v = kv.gather(layer, blocks, kv_len)
            qi = q[qs:qe].transpose(0, 1)[None]  # [1, H, S, D]
            ki, vi = k.transpose(0, 1)[None], v.transpose(0, 1)[None]
            if q_len == 1:
                o = F.scaled_dot_product_attention(qi, ki, vi, enable_gqa=True)
            elif q_len == kv_len:
                o = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True, enable_gqa=True)
            else:  # append in the middle of a sequence: keys up to kv_len - q_len + j
                qpos = torch.arange(kv_len - q_len, kv_len, device=q.device)[:, None]
                kpos = torch.arange(kv_len, device=q.device)[None, :]
                o = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=kpos <= qpos, enable_gqa=True)
            out[qs:qe] = o[0].transpose(0, 1)
        return out


class FlashInferBackend:
    """Paged prefill / decode wrappers. CUDA only."""

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, dtype, device) -> None:
        import flashinfer

        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.dtype = dtype
        self.workspace = torch.empty(128 << 20, dtype=torch.uint8, device=device)
        self.prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, "NHD")
        self.decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace, "NHD", use_tensor_cores=num_heads // num_kv_heads >= 4
        )

    def plan(self, meta: AttnMeta) -> None:
        self.meta = meta
        common = dict(
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            page_size=meta.block_size,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )
        if meta.is_decode:
            self.decode.plan(
                meta.kv_indptr, meta.kv_indices, meta.kv_last_page_len, head_dim=self.head_dim, **common
            )
            self.wrapper = self.decode
        else:
            self.prefill.plan(
                meta.qo_indptr,
                meta.kv_indptr,
                meta.kv_indices,
                meta.kv_last_page_len,
                head_dim_qk=self.head_dim,
                causal=True,
                **common,
            )
            self.wrapper = self.prefill

    def run(self, q: torch.Tensor, layer: int, kv: PagedKVCache) -> torch.Tensor:
        return self.wrapper.run(q, (kv.k[layer], kv.v[layer]))


def default_backend(config, device: torch.device, dtype):
    if device.type == "cuda":
        return FlashInferBackend(config.num_attention_heads, config.num_key_value_heads, config.head_dim, dtype, device)
    return SdpaBackend()
