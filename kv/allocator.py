"""Block pool + per-request block tables. Pure Python, no tensors.

Decides *which* physical blocks a request owns; kv/cache.py owns the memory.
Invariant: every block is in exactly one place -- the free list or one live table.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class PoolExhausted(Exception):
    pass


@dataclass
class BlockTable:
    blocks: list[int] = field(default_factory=list)
    seq_len: int = 0


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        assert num_blocks > 0 and block_size > 0
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))  # pop() hands out block 0 first
        self._live: dict[int, BlockTable] = {}

    @property
    def num_free(self) -> int:
        return len(self._free)

    def blocks_for(self, seq_len: int) -> int:
        return -(-seq_len // self.block_size)

    def last_block_len(self, table: BlockTable) -> int:
        """Valid tokens in the last block, 1..block_size (FlashInfer's kv_last_page_len)."""
        return table.seq_len - (len(table.blocks) - 1) * self.block_size if table.seq_len else 0

    def alloc(self, req_id: int, seq_len: int = 0) -> BlockTable:
        assert req_id not in self._live
        need = self.blocks_for(seq_len)
        if need > len(self._free):
            raise PoolExhausted(f"need {need} blocks, {len(self._free)} free")
        table = BlockTable([self._free.pop() for _ in range(need)], seq_len)
        self._live[req_id] = table
        return table

    def append(self, req_id: int, n_tokens: int) -> BlockTable:
        table = self._live[req_id]
        need = self.blocks_for(table.seq_len + n_tokens) - len(table.blocks)
        if need > len(self._free):
            raise PoolExhausted(f"need {need} blocks, {len(self._free)} free")
        for _ in range(need):
            table.blocks.append(self._free.pop())
        table.seq_len += n_tokens
        return table

    def rollback(self, req_id: int, seq_len: int) -> BlockTable:
        """Truncate the tail; whole blocks past the new end go back to the pool."""
        table = self._live[req_id]
        assert 0 <= seq_len <= table.seq_len
        keep = self.blocks_for(seq_len)
        self._free.extend(table.blocks[keep:])
        del table.blocks[keep:]
        table.seq_len = seq_len
        return table

    def free(self, req_id: int) -> None:
        table = self._live.pop(req_id)
        self._free.extend(table.blocks)
        table.blocks.clear()
        table.seq_len = 0

    def get(self, req_id: int) -> BlockTable:
        return self._live[req_id]

    def check(self) -> None:
        """Every block accounted for exactly once."""
        seen = list(self._free)
        for t in self._live.values():
            seen.extend(t.blocks)
            assert len(t.blocks) == self.blocks_for(t.seq_len), (t.blocks, t.seq_len)
        assert sorted(seen) == list(range(self.num_blocks)), "block leaked or double-owned"
