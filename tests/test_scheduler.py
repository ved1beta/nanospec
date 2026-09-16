from kv.allocator import BlockAllocator
from sched.scheduler import Request, SamplingParams, Scheduler


def _req(i, n):
    return Request(i, list(range(n)), SamplingParams(max_tokens=4))


def test_staggered_admission_and_decode_order():
    s = Scheduler(BlockAllocator(num_blocks=64, block_size=4), max_admit=1)
    for i in range(3):
        s.add(_req(i, 5))
    b = s.next_batch()
    assert [(r.id, n) for r, n in b] == [(0, 5)]
    b = s.next_batch()  # running decode first, then one admission
    assert [(r.id, n) for r, n in b] == [(0, 1), (1, 5)]
    b = s.next_batch()
    assert [(r.id, n) for r, n in b] == [(0, 1), (1, 1), (2, 5)]
    assert [r.seq_len for r in s.running] == [5, 5, 5]  # out_tokens are the engine's job
    assert [s.alloc.get(r.id).seq_len for r in s.running] == [7, 6, 5]


def test_admission_waits_for_blocks_and_finish_frees():
    alloc = BlockAllocator(num_blocks=4, block_size=4)
    s = Scheduler(alloc, max_admit=8)
    s.add(_req(0, 12))  # 3 blocks
    s.add(_req(1, 8))  # 2 blocks: doesn't fit alongside
    s.add(_req(2, 4))  # 1 block: would fit, but FIFO -- stays behind 1
    assert [r.id for r, _ in s.next_batch()] == [0]
    assert [r.id for r in s.waiting] == [1, 2]
    s.finish(s.running[0])
    assert alloc.num_free == 4
    assert [r.id for r, _ in s.next_batch()] == [1, 2]
    for r in list(s.running):
        s.finish(r)
    assert not s.has_work and alloc.num_free == 4
    alloc.check()
