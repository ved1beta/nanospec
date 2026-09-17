"""Offline, in-process engine: prefill + decode steps over the paged cache (PLAN D6),
optionally with EAGLE-3 chain speculation (PLAN D2/D4: greedy acceptance).

Position bookkeeping: `req.committed` is the number of KV positions actually written;
the allocator's table.seq_len is the reservation, kept at committed + spec_depth + 1
for running requests so verify / draft rows always have slots.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kv.allocator import BlockAllocator
from kv.cache import AttnMeta, PagedKVCache
from model.llama import LlamaForCausalLM, eagle3_aux_layers
from sched.scheduler import Request, SamplingParams, Scheduler
from spec.tree import Tree, draft_mask, longest_accepted, verify_mask


@dataclass(frozen=True)
class EngineConfig:
    num_blocks: int
    block_size: int = 16
    max_admit: int = 32
    cuda_graphs: bool = True  # decode step via engine/graphs.py; only takes effect on CUDA
    spec_depth: int = 0  # EAGLE-3 draft depth; 0 = no speculation
    spec_topk: int = 1  # 1 = chain; >1 = static tree of topk * depth nodes (PLAN D5)
    record_logits: bool = False  # keep each request's per-step next-token logits on CPU (tests)

    @property
    def num_draft(self) -> int:
        return self.spec_depth * self.spec_topk


class Engine:
    def __init__(self, model: LlamaForCausalLM, config: EngineConfig, drafter=None) -> None:
        self.model = model
        self.config = config
        p = model.lm_head.weight
        self.device, self.dtype = p.device, p.dtype
        self.kv = PagedKVCache(model.config, config.num_blocks, config.block_size, self.device, self.dtype)
        self.alloc = BlockAllocator(config.num_blocks, config.block_size)
        self.sched = Scheduler(self.alloc, config.max_admit)
        self.eos = set(model.config.eos_token_ids)
        self._next_id = 0

        self.depth = config.spec_depth
        self.topk = config.spec_topk
        self.drafter = drafter
        self.taps = eagle3_aux_layers(model.config.num_hidden_layers)
        if self.depth:
            assert drafter is not None, "spec_depth > 0 needs an EAGLE-3 drafter"
            self.draft_kv = PagedKVCache(drafter.config.rope_config(), config.num_blocks, config.block_size, self.device, self.dtype)

        self.graphs = self.verify_graphs = self.draft_graphs = None
        self.usable_blocks = config.num_blocks  # minus the graph pad block, if any
        if config.cuda_graphs and self.device.type == "cuda":
            self._capture_graphs()

    def _capture_graphs(self) -> None:
        from engine.graphs import GraphRunner

        c, cfg = self.config, self.model.config
        pad_block = self.alloc.alloc(-1, 1).blocks[0]  # reserved forever for padded rows
        self.usable_blocks -= 1
        heads = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, self.dtype, self.device)
        if not self.depth:
            self.graphs = GraphRunner(self.kv, pad_block, 1, False, *heads)
            self.graphs.capture(lambda ids, _h, meta, be: self.model(ids, self.kv, meta, backend=be)[0])
            return
        R = c.num_draft + 1
        self.verify_graphs = GraphRunner(self.kv, pad_block, R, True, *heads, buckets=(1, 2, 4, 8, 16, 32))

        def verify(ids, _h, meta, be):
            logits, aux = self.model(ids, self.kv, meta, aux_layers=self.taps, backend=be)
            return logits, torch.cat(aux, dim=-1)

        self.verify_graphs.capture(verify)
        d = self.drafter.config
        dheads = (d.num_attention_heads, d.num_key_value_heads, d.head_dim, self.dtype, self.device)
        self.draft_graphs = GraphRunner(self.draft_kv, pad_block, self.topk, self.topk > 1, *dheads,
                                        hidden_dim=d.hidden_size, buckets=(1, 2, 4, 8, 16, 32))
        self.draft_graphs.capture(lambda ids, h, meta, be: self.drafter(ids, h, self.draft_kv, meta, backend=be))

    # ------------------------------------------------------------------ requests

    def add(self, prompt_ids: list[int], params: SamplingParams = SamplingParams()) -> Request:
        req = Request(self._next_id, list(prompt_ids), params)
        self._next_id += 1
        self.sched.add(req)
        return req

    def generate(self, prompts: list[list[int]], params: SamplingParams = SamplingParams()) -> list[list[int]]:
        reqs = [self.add(p, params) for p in prompts]
        while self.sched.has_work:
            self.step()
        return [r.out_tokens for r in reqs]

    def _emit(self, req: Request, tokens: list[int]) -> bool:
        """Append tokens, truncating at eos / max_tokens. Returns True if the request is done."""
        for t in tokens:
            req.out_tokens.append(t)
            if t in self.eos or len(req.out_tokens) >= req.params.max_tokens:
                return True
        return False

    # ------------------------------------------------------------------ steps

    def step(self) -> list[Request]:
        """One target forward over this step's batch; returns requests that finished."""
        running, new = self.sched.next_batch()
        if not running and not new:
            return []
        if not self.depth:
            return self._plain_step(running, new)
        return self._tree_step(running, new) if self.topk > 1 else self._spec_step(running, new)

    def _plain_step(self, running: list[Request], new: list[Request]) -> list[Request]:
        batch = running + new
        ids, n_new, starts = [], [], []
        for req in running:
            self.alloc.append(req.id, 1)
            ids.append(req.out_tokens[-1])
            n_new.append(1)
            starts.append(req.committed)
        for req in new:
            ids.extend(req.prompt_ids)
            n_new.append(len(req.prompt_ids))
            starts.append(0)
        input_ids = torch.tensor(ids, dtype=torch.int64, device=self.device)
        meta = AttnMeta.build([r.table for r in batch], n_new, self.config.block_size, self.device, starts)

        if self.graphs is not None and self.graphs.can_run(meta):
            (logits,) = self.graphs.run(meta, ids)
        else:
            logits, _ = self.model(input_ids, self.kv, meta, logits_idx=meta.last_token_idx)
        next_tokens = logits.argmax(-1).tolist()
        if self.config.record_logits:
            rows = logits.float().cpu()
            for req, row in zip(batch, rows):
                req.step_logits.append(row)

        finished = []
        for req, n, tok in zip(batch, n_new, next_tokens):
            req.committed += n
            if self._emit(req, [tok]):
                self.sched.finish(req)
                finished.append(req)
        return finished

    def _spec_step(self, running: list[Request], new: list[Request]) -> list[Request]:
        D = self.depth
        batch = running + new
        ids, n_new, starts = [], [], []
        for req in running:  # verify rows: [next input token, d_1 .. d_D]
            ids.append(req.out_tokens[-1])
            ids.extend(req.drafts)
            n_new.append(D + 1)
            starts.append(req.committed)
        for req in new:
            ids.extend(req.prompt_ids)
            n_new.append(len(req.prompt_ids))
            starts.append(0)
            self.alloc.append(req.id, D + 1)
        meta = AttnMeta.build([r.table for r in batch], n_new, self.config.block_size, self.device, starts)
        logits, aux = self._verify_forward(ids, meta)
        argmax = logits.argmax(-1).tolist()

        finished, alive = [], []
        qs = 0
        for req, n in zip(batch, n_new):
            g = argmax[qs : qs + n]
            if req.drafts:  # verify
                a = 0
                while a < D and req.drafts[a] == g[a]:
                    a += 1
                req.accepted.append(a)
                done = self._emit(req, req.drafts[:a] + [g[a]])
                req.committed += a + 1
                ext_ids = req.drafts[:a] + [g[a]]
                ext_hidden = aux[qs : qs + a + 1]
            else:  # prefill
                done = self._emit(req, [g[-1]])
                req.committed += n
                ext_ids = req.prompt_ids[1:] + [g[-1]]
                ext_hidden = aux[qs : qs + n]
            qs += n
            if done:
                self.sched.finish(req)
                finished.append(req)
                continue
            self.alloc.rollback(req.id, req.committed)
            self.alloc.append(req.id, D + 1)
            req.ext_ids, req.ext_hidden = ext_ids, ext_hidden
            alive.append(req)

        if alive:
            self._draft(alive)
        return finished

    def _verify_forward(self, ids: list[int], meta: AttnMeta) -> tuple[torch.Tensor, torch.Tensor]:
        """Target forward with aux taps: captured graph when the batch is pure verify."""
        if self.verify_graphs is not None and self.verify_graphs.can_run(meta):
            return self.verify_graphs.run(meta, ids)
        input_ids = torch.tensor(ids, dtype=torch.int64, device=self.device)
        logits, aux = self.model(input_ids, self.kv, meta, aux_layers=self.taps)
        return logits, torch.cat(aux, dim=-1)

    def _draft_forward(self, ids, hidden: torch.Tensor, meta: AttnMeta) -> tuple[torch.Tensor, torch.Tensor]:
        if self.draft_graphs is not None and self.draft_graphs.can_run(meta):
            return self.draft_graphs.run(meta, ids, hidden)
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids, dtype=torch.int64, device=self.device)
        return self.drafter(ids, hidden, self.draft_kv, meta)

    def _draft(self, reqs: list[Request]) -> None:
        """Extend the draft cache over the newly committed tokens, then chain D-1 steps."""
        D = self.depth
        tables = [r.table for r in reqs]
        n = [len(r.ext_ids) for r in reqs]
        starts = [r.committed - len(r.ext_ids) for r in reqs]
        ids = torch.tensor([t for r in reqs for t in r.ext_ids], dtype=torch.int64, device=self.device)
        hidden = torch.cat([r.ext_hidden for r in reqs])
        meta = AttnMeta.build(tables, n, self.config.block_size, self.device, starts)
        logits, h = self.drafter(ids, hidden, self.draft_kv, meta)
        last = meta.last_token_idx
        tok = self.drafter.to_target_ids(logits[last].argmax(-1))
        h = h[last]
        drafts = [tok]
        for j in range(1, D):
            starts = [r.committed + j - 1 for r in reqs]
            meta = AttnMeta.build(tables, [1] * len(reqs), self.config.block_size, self.device, starts)
            logits, h = self._draft_forward(tok, h, meta)
            tok = self.drafter.to_target_ids(logits.argmax(-1))
            drafts.append(tok)
        drafts = torch.stack(drafts, dim=1).tolist()  # [B, D]
        for r, d in zip(reqs, drafts):
            r.drafts = d
            r.ext_ids, r.ext_hidden = [], None

    # ------------------------------------------------------------------ tree speculation (G6)

    def _tree_step(self, running: list[Request], new: list[Request]) -> list[Request]:
        N = self.config.num_draft
        bs = self.config.block_size
        batch = running + new
        ids, rows, masks = [], [], []
        for req in running:  # verify rows: root (next input token) + the N tree nodes
            L, t = req.committed, req.tree
            ids.append(req.out_tokens[-1])
            ids.extend(t.tokens)
            rows.append(([L] + [L + d for d in t.depths], [L] + [L + 1 + i for i in range(N)], L + 1 + N))
            masks.append(verify_mask(L, t.parents, self.device))
        for req in new:
            n = len(req.prompt_ids)
            ids.extend(req.prompt_ids)
            rows.append((list(range(n)), list(range(n)), n))
            masks.append(None)
            self.alloc.append(req.id, N + 1)
        meta = AttnMeta.from_rows([r.table for r in batch], rows, bs, self.device, masks)
        logits, aux = self._verify_forward(ids, meta)
        argmax = logits.argmax(-1).tolist()

        finished, alive = [], []
        qs = 0
        for req, (qpos, _, _) in zip(batch, rows):
            n = len(qpos)
            g = argmax[qs : qs + n]
            if req.tree is not None:
                L = req.committed
                path, bonus = longest_accepted(req.tree, g)
                a = len(path)
                req.accepted.append(a)
                if a and path != list(range(a)):  # compact the accepted path into chain order
                    blocks = req.table.blocks
                    phys = lambda p: blocks[p // bs] * bs + p % bs
                    src = torch.tensor([phys(L + 1 + p) for p in path], device=self.device)
                    dst = torch.tensor([phys(L + 1 + k) for k in range(a)], device=self.device)
                    self.kv.move(src, dst)
                done = self._emit(req, [req.tree.tokens[p] for p in path] + [bonus])
                req.committed += a + 1
                ext_ids = req.out_tokens[-(a + 1) :] if not done else []
                ext_hidden = aux[[qs] + [qs + 1 + p for p in path]]
            else:
                done = self._emit(req, [g[-1]])
                req.committed += n
                ext_ids = req.prompt_ids[1:] + [g[-1]]
                ext_hidden = aux[qs : qs + n]
            qs += n
            if done:
                self.sched.finish(req)
                finished.append(req)
                continue
            self.alloc.rollback(req.id, req.committed)
            self.alloc.append(req.id, N + 1)
            req.ext_ids, req.ext_hidden = ext_ids, ext_hidden
            alive.append(req)

        if alive:
            self._draft_tree(alive)
        return finished

    def _draft_tree(self, reqs: list[Request]) -> None:
        """Extend the draft cache over the committed tokens, then grow topk nodes per level.
        Level selection is batched on the GPU: one topk over the K*K candidates per request
        and one host sync per level."""
        K, D, N, bs = self.topk, self.depth, self.config.num_draft, self.config.block_size
        B = len(reqs)
        tables = [r.table for r in reqs]
        n = [len(r.ext_ids) for r in reqs]
        starts = [r.committed - len(r.ext_ids) for r in reqs]
        ids = torch.tensor([t for r in reqs for t in r.ext_ids], dtype=torch.int64, device=self.device)
        hidden = torch.cat([r.ext_hidden for r in reqs])
        meta = AttnMeta.build(tables, n, bs, self.device, starts)
        logits, h = self.drafter(ids, hidden, self.draft_kv, meta)
        last = meta.last_token_idx
        top = torch.log_softmax(logits[last].float(), dim=-1).topk(K, dim=-1)

        tok_dev = self.drafter.to_target_ids(top.indices)  # [B, K] level-1 tokens (target ids)
        tokens = tok_dev.tolist()  # per request, depth-major node lists
        parents = [[-1] * K for _ in range(B)]
        depths = [[1] * K for _ in range(B)]
        scores = top.values  # [B, K] cumulative log-prob of the current level's nodes
        h_parent = h[last][:, None, :].expand(B, K, -1)  # hidden feeding each current node

        for d in range(1, D):
            level = list(range((d - 1) * K, d * K))  # node indices of the current level
            rows = [([r.committed + d - 1] * K, [r.committed + i for i in level], r.committed + N) for r in reqs]
            masks = [draft_mask(r.committed, parents[b], level, N, self.device) for b, r in enumerate(reqs)]
            meta = AttnMeta.from_rows(tables, rows, bs, self.device, masks)
            logits, h = self._draft_forward(tok_dev.reshape(-1), h_parent.reshape(B * K, -1), meta)
            child = torch.log_softmax(logits.float(), dim=-1).view(B, K, -1).topk(K, dim=-1)  # [B, K, K]
            cand_scores = (scores[:, :, None] + child.values).reshape(B, K * K)
            best = cand_scores.topk(K, dim=-1).indices  # [B, K] global top-k children per request
            parent_local = best // K  # which current-level node each kept child hangs off
            tok_dev = self.drafter.to_target_ids(child.indices.reshape(B, K * K).gather(1, best))
            scores = cand_scores.gather(1, best)
            h_parent = h.view(B, K, -1).gather(1, parent_local[:, :, None].expand(B, K, h.shape[-1]))
            new_tokens, new_parents = torch.stack([tok_dev, parent_local]).tolist()  # one sync per level
            for b in range(B):
                tokens[b] += new_tokens[b]
                parents[b] += [level[p] for p in new_parents[b]]
                depths[b] += [d + 1] * K

        for b, r in enumerate(reqs):
            r.tree = Tree(tokens[b], parents[b], depths[b])
            r.ext_ids, r.ext_hidden = [], None
