"""Paged K/V storage and the per-step attention metadata built from block tables."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kv.allocator import BlockTable
from model.llama import LlamaConfig


class PagedKVCache:
    """Per layer K and V of shape [num_blocks, block_size, kv_heads, head_dim] (FlashInfer NHD)."""

    def __init__(self, config: LlamaConfig, num_blocks: int, block_size: int, device, dtype) -> None:
        shape = (config.num_hidden_layers, num_blocks, block_size, config.num_key_value_heads, config.head_dim)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty(shape, device=device, dtype=dtype)
        self.num_blocks = num_blocks
        self.block_size = block_size

    def write(self, layer: int, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v: [N, kv_heads, head_dim] into flat slot = block * block_size + offset."""
        self.k[layer].view(-1, *self.k.shape[-2:])[slots] = k
        self.v[layer].view(-1, *self.v.shape[-2:])[slots] = v

    def gather(self, layer: int, blocks: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Contiguous [seq_len, kv_heads, head_dim] for one request (SDPA path only)."""
        k = self.k[layer][blocks].flatten(0, 1)[:seq_len]
        v = self.v[layer][blocks].flatten(0, 1)[:seq_len]
        return k, v


@dataclass
class AttnMeta:
    """One step's batch, flattened: N query tokens over B requests."""

    qo_indptr: torch.Tensor  # [B+1] int32
    kv_indptr: torch.Tensor  # [B+1] int32, into kv_indices
    kv_indices: torch.Tensor  # [sum blocks] int32
    kv_last_page_len: torch.Tensor  # [B] int32
    positions: torch.Tensor  # [N] int64, logical position of each query token
    slots: torch.Tensor  # [N] int64, flat cache slot each new token is written to
    seq_lens: list[int]  # kv length per request after this step
    qo_lens: list[int]
    block_size: int

    @property
    def is_decode(self) -> bool:
        return all(n == 1 for n in self.qo_lens)

    @property
    def last_token_idx(self) -> torch.Tensor:
        return self.qo_indptr[1:].long() - 1

    @classmethod
    def build(cls, tables: list[BlockTable], n_new: list[int], block_size: int, device) -> "AttnMeta":
        """Tables must already include the new tokens (allocator.append was called)."""
        qo_indptr, kv_indptr, kv_indices, last_len, positions, slots = [0], [0], [], [], [], []
        for t, n in zip(tables, n_new):
            assert 1 <= n <= t.seq_len
            qo_indptr.append(qo_indptr[-1] + n)
            kv_indptr.append(kv_indptr[-1] + len(t.blocks))
            kv_indices.extend(t.blocks)
            last_len.append(t.seq_len - (len(t.blocks) - 1) * block_size)
            for p in range(t.seq_len - n, t.seq_len):
                positions.append(p)
                slots.append(t.blocks[p // block_size] * block_size + p % block_size)
        i32 = lambda x: torch.tensor(x, dtype=torch.int32, device=device)
        i64 = lambda x: torch.tensor(x, dtype=torch.int64, device=device)
        return cls(
            qo_indptr=i32(qo_indptr),
            kv_indptr=i32(kv_indptr),
            kv_indices=i32(kv_indices),
            kv_last_page_len=i32(last_len),
            positions=i64(positions),
            slots=i64(slots),
            seq_lens=[t.seq_len for t in tables],
            qo_lens=list(n_new),
            block_size=block_size,
        )
