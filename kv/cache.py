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
        self.k = torch.zeros(shape, device=device, dtype=dtype)  # zeros, not empty: masked-out slots are still read
        self.v = torch.zeros(shape, device=device, dtype=dtype)
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

    def move(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Copy K/V at flat slots src -> dst in every layer (tree compaction)."""
        k = self.k.view(self.k.shape[0], -1, *self.k.shape[-2:])
        v = self.v.view(self.v.shape[0], -1, *self.v.shape[-2:])
        k[:, dst] = k[:, src]
        v[:, dst] = v[:, src]


@dataclass
class AttnMeta:
    """One step's batch, flattened: N query tokens over B requests."""

    qo_indptr: torch.Tensor  # [B+1] int32
    kv_indptr: torch.Tensor  # [B+1] int32, into kv_indices
    kv_indices: torch.Tensor  # [sum blocks] int32
    kv_last_page_len: torch.Tensor  # [B] int32
    positions: torch.Tensor  # [N] int64, logical position of each query token
    slots: torch.Tensor  # [N] int64, flat cache slot each new token is written to
    seq_lens: list[int]  # kv length attended per request (start + n)
    qo_lens: list[int]
    block_size: int
    # host copies, so consumers that need them (graph padding, FlashInfer plan) never sync
    kv_indptr_host: list[int] = None
    kv_indices_host: list[int] = None
    kv_last_page_len_host: list[int] = None
    positions_host: list[int] = None
    slots_host: list[int] = None
    masks: list[torch.Tensor | None] = None  # per request bool [qo, kv]; None = causal

    @staticmethod
    def causal_mask(q: int, k: int, device) -> torch.Tensor:
        """[q, k] bool, bottom-right aligned: the last query row sees every key."""
        qi = torch.arange(k - q, k, device=device)[:, None]
        return torch.arange(k, device=device)[None, :] <= qi

    @property
    def custom_mask(self) -> torch.Tensor | None:
        """All requests' masks flattened (FlashInfer's custom_mask), causal ones made
        explicit. None when no request has a mask, so eager callers can use the plain
        causal kernels; graph runners must materialise masks themselves."""
        if self.masks is None or all(m is None for m in self.masks):
            return None
        dev = self.qo_indptr.device
        return torch.cat([
            (m if m is not None else self.causal_mask(q, k, dev)).reshape(-1)
            for m, q, k in zip(self.masks, self.qo_lens, self.seq_lens)
        ])

    @property
    def is_decode(self) -> bool:
        return all(n == 1 for n in self.qo_lens)

    @property
    def last_token_idx(self) -> torch.Tensor:
        return self.qo_indptr[1:].long() - 1

    @classmethod
    def build(
        cls, tables: list[BlockTable], n_new: list[int], block_size: int, device, starts: list[int] | None = None
    ) -> "AttnMeta":
        """n_new query tokens per request at positions [start, start+n), attending to
        kv [0, start+n). Default start = the last n reserved positions."""
        rows = []
        for i, (t, n) in enumerate(zip(tables, n_new)):
            start = t.seq_len - n if starts is None else starts[i]
            rows.append((list(range(start, start + n)), list(range(start, start + n)), start + n))
        return cls.from_rows(tables, rows, block_size, device)

    @classmethod
    def from_rows(
        cls,
        tables: list[BlockTable],
        rows: list[tuple[list[int], list[int], int]],
        block_size: int,
        device,
        masks: list[torch.Tensor | None] | None = None,
    ) -> "AttnMeta":
        """Per request: (RoPE position per query row, logical slot per query row, kv_len).
        Query rows attend logical slots [0, kv_len) under masks[i] (None = causal, aligned
        so the last row sees everything)."""
        qo_indptr, kv_indptr, kv_indices, last_len, positions, slots, kv_lens = [0], [0], [], [], [], [], []
        for t, (qpos, qslot, kv_len) in zip(tables, rows):
            n = len(qpos)
            assert 1 <= n and kv_len <= t.seq_len and max(qslot) < kv_len, (qslot, kv_len, t.seq_len)
            nb = -(-kv_len // block_size)
            qo_indptr.append(qo_indptr[-1] + n)
            kv_indptr.append(kv_indptr[-1] + nb)
            kv_indices.extend(t.blocks[:nb])
            last_len.append(kv_len - (nb - 1) * block_size)
            kv_lens.append(kv_len)
            positions.extend(qpos)
            slots.extend(t.blocks[p // block_size] * block_size + p % block_size for p in qslot)
        n_new = [len(r[0]) for r in rows]
        i32 = lambda x: torch.tensor(x, dtype=torch.int32, device=device)
        i64 = lambda x: torch.tensor(x, dtype=torch.int64, device=device)
        return cls(
            qo_indptr=i32(qo_indptr),
            kv_indptr=i32(kv_indptr),
            kv_indices=i32(kv_indices),
            kv_last_page_len=i32(last_len),
            positions=i64(positions),
            slots=i64(slots),
            seq_lens=kv_lens,
            qo_lens=list(n_new),
            block_size=block_size,
            kv_indptr_host=kv_indptr,
            kv_indices_host=kv_indices,
            kv_last_page_len_host=last_len,
            positions_host=positions,
            slots_host=slots,
            masks=masks,
        )
