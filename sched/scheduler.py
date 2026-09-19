"""Request lifecycle and per-step batch composition. No tensors (bar the RNG).

The scheduler decides what runs (admission, who decodes, who finishes) and owns the
block accounting; the engine turns the batch into tensors and runs the model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch

from kv.allocator import BlockAllocator, BlockTable


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 0.0  # 0 = greedy
    top_k: int = 0  # 0 = off
    top_p: float = 1.0
    seed: int | None = None  # per-request RNG; None = random
    # exact: lossless (the emitted distribution equals plain sampling). relaxed: accept a
    # draft whenever p(draft) >= tau or the draft is in the target's top accept_topk --
    # faster, and NOT the target's distribution any more.
    acceptance: str = "exact"
    tau: float = 1.0
    accept_topk: int = 0
    # argmax: point-mass proposals (chain and tree). sample: chain drafts drawn from the
    # head; the Leviathan min(1, p/q) test with residual norm(max(0, p - q)).
    draft_sampling: str = "argmax"
    logprobs: bool = False  # fill Request.logprobs


@dataclass(slots=True)
class TokenInfo:
    token: int
    logprob: float  # target log p(token) at temperature 1, no top-k / top-p
    sample_logprob: float  # log-prob under the (T, top-k, top-p) distribution it was drawn from; 0 when greedy
    draft_logprob: float | None  # the draft's log q(token) for accepted drafts
    source: str  # draft (accepted) | resample (row where a draft was rejected) | bonus (no draft tested at the row)


@dataclass
class Request:
    id: int
    prompt_ids: list[int]
    params: SamplingParams
    out_tokens: list[int] = field(default_factory=list)
    state: str = "waiting"  # waiting | running | done
    table: BlockTable | None = None
    committed: int = 0  # KV positions written; out_tokens[-1] is the next input token
    gen: torch.Generator = None  # CPU RNG, seeded per request: results do not depend on batch composition
    logprobs: list[TokenInfo] = field(default_factory=list)  # only with params.logprobs
    # speculative state, owned by the engine
    tree: object = None  # spec.tree.Tree (a chain is a tree with linear parents)
    draft_q: object = None  # [D, draft_vocab] proposal probs when params.draft_sampling == "sample"
    ext_ids: list[int] = field(default_factory=list)
    ext_hidden: object = None
    accepted: list[int] = field(default_factory=list)  # accepted draft count per verify step
    step_logits: list = field(default_factory=list)  # only with EngineConfig.record_logits

    def __post_init__(self) -> None:
        self.gen = torch.Generator()
        self.gen.manual_seed(self.params.seed) if self.params.seed is not None else self.gen.seed()

    @property
    def seq_len(self) -> int:
        return len(self.prompt_ids) + len(self.out_tokens)


class Scheduler:
    """Admission is FIFO, up to max_admit per step and max_running live. A request is
    admitted only if the pool can hold every live request at its maximum length
    (prompt + max_tokens + spare), so append() never fails mid-decode."""

    def __init__(self, alloc: BlockAllocator, max_admit: int = 32, max_running: int | None = None, spare: int = 0) -> None:
        self.alloc = alloc
        self.max_admit, self.max_running, self.spare = max_admit, max_running, spare
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def add(self, req: Request) -> None:
        assert req.state == "waiting"
        self.waiting.append(req)

    def _max_blocks(self, req: Request) -> int:
        return self.alloc.blocks_for(len(req.prompt_ids) + req.params.max_tokens + self.spare)

    def next_batch(self) -> tuple[list[Request], list[Request]]:
        """(running, newly admitted). The engine reserves per-step KV for running requests."""
        running = list(self.running)
        headroom = self.alloc.num_free - sum(self._max_blocks(r) - len(r.table.blocks) for r in running)
        admitted: list[Request] = []
        while self.waiting and len(admitted) < self.max_admit and len(self.running) != self.max_running:
            req = self.waiting[0]
            if self._max_blocks(req) > headroom:
                break
            headroom -= self._max_blocks(req)
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

    def restart(self, req: Request) -> None:
        """Back to the head of the queue with prompt + output as the prompt to prefill:
        its KV is stale (weights changed). Emitted tokens and records are kept."""
        self.finish(req)
        req.prompt_ids = req.prompt_ids + req.out_tokens
        req.state, req.committed, req.tree, req.draft_q = "waiting", 0, None, None
        self.waiting.appendleft(req)
