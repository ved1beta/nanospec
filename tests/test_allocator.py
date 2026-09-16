import random

import pytest

from kv.allocator import BlockAllocator, PoolExhausted


def test_alloc_append_free_roundtrip():
    a = BlockAllocator(num_blocks=8, block_size=4)
    t = a.alloc(0, seq_len=5)
    assert len(t.blocks) == 2 and a.last_block_len(t) == 1
    a.append(0, 3)  # 8 tokens: exactly two full blocks, no new block
    assert len(t.blocks) == 2 and a.last_block_len(t) == 4
    a.append(0, 1)  # crosses the boundary
    assert len(t.blocks) == 3 and a.last_block_len(t) == 1
    a.free(0)
    assert a.num_free == 8
    a.check()


def test_rollback_frees_only_whole_blocks_past_tail():
    a = BlockAllocator(num_blocks=8, block_size=4)
    t = a.alloc(0, seq_len=10)  # blocks 0,1,2 ; last has 2 valid
    b = list(t.blocks)
    a.rollback(0, 9)
    assert t.blocks == b and a.last_block_len(t) == 1
    a.rollback(0, 8)
    assert t.blocks == b[:2] and a.last_block_len(t) == 4
    a.rollback(0, 1)
    assert t.blocks == b[:1] and a.num_free == 7
    a.rollback(0, 0)
    assert t.blocks == [] and a.num_free == 8
    a.check()


def test_exhaustion_raises_and_leaves_state_intact():
    a = BlockAllocator(num_blocks=4, block_size=2)
    a.alloc(0, seq_len=6)
    with pytest.raises(PoolExhausted):
        a.alloc(1, seq_len=3)
    a.check()
    assert a.num_free == 1
    a.alloc(1, seq_len=2)
    a.check()


def test_churn_keeps_invariant_and_fragments():
    rng = random.Random(0)
    a = BlockAllocator(num_blocks=256, block_size=16)
    live = []
    for i in range(400):
        op = rng.random()
        if live and op < 0.3:
            a.free(live.pop(rng.randrange(len(live))))
        elif live and op < 0.5:
            r = rng.choice(live)
            a.rollback(r, rng.randint(0, a.get(r).seq_len))
        elif live and op < 0.7:
            try:
                a.append(rng.choice(live), rng.randint(1, 40))
            except PoolExhausted:
                pass
        else:
            try:
                a.alloc(i, rng.randint(1, 100))
                live.append(i)
            except PoolExhausted:
                pass
        a.check()
    # a request allocated now should land on scattered blocks, not a contiguous run
    for r in live[: len(live) // 2]:
        a.free(r)
    t = a.alloc(10_000, seq_len=16 * 8)
    assert any(b2 != b1 + 1 for b1, b2 in zip(t.blocks, t.blocks[1:])), t.blocks
    a.check()
