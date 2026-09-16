"""Request lifecycle and per-step batch composition. No tensors.

The scheduler decides what runs (admission, who decodes, who finishes) and owns the
block accounting; the engine turns the batch into tensors and runs the model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from kv.allocator import BlockAllocator, BlockTable


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 128  # greedy only in v0.1


@dataclass
class Request:
    id: int
    prompt_ids: list[int]
    params: SamplingParams
    out_tokens: list[int] = field(default_factory=list)
    state: str = "waiting"  # waiting | running | done
    table: BlockTable | None = None
    committed: int = 0  # KV positions written; out_tokens[-1] is the next input token
    # speculative state, owned by the engine
    drafts: list[int] = field(default_factory=list)  # chain
    tree: object = None  # spec.tree.Tree
    ext_ids: list[int] = field(default_factory=list)
    ext_hidden: object = None
    accepted: list[int] = field(default_factory=list)  # accepted draft count per verify step
    step_logits: list = field(default_factory=list)  # only with EngineConfig.record_logits

    @property
    def seq_len(self) -> int:
        return len(self.prompt_ids) + len(self.out_tokens)


class Scheduler:
    def __init__(self, alloc: BlockAllocator, max_admit: int = 32) -> None:
        self.alloc = alloc
        self.max_admit = max_admit
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def add(self, req: Request) -> None:
        assert req.state == "waiting"
        self.waiting.append(req)

    def next_batch(self) -> tuple[list[Request], list[Request]]:
        """(running, newly admitted). Admission is FIFO, up to max_admit per step, only if
        the prompt fits in the pool. The engine reserves per-step KV for running requests."""
        running = list(self.running)
        admitted: list[Request] = []
        while self.waiting and len(admitted) < self.max_admit:
            req = self.waiting[0]
            if self.alloc.blocks_for(len(req.prompt_ids)) > self.alloc.num_free:
                break
            self.waiting.popleft()
            req.table = self.alloc.alloc(req.id, len(req.prompt_ids))
            req.state = "running"
            self.running.append(req)
            admitted.append(req)
        return running, admitted

    def finish(self, req: Request) -> None:
        self.running.remove(req)
        self.alloc.free(req.id)
        req.table = None
        req.state = "done"
