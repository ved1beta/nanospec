"""Offline, in-process engine: prefill + decode steps over the paged cache (PLAN D6),
optionally with EAGLE-3 chain / tree speculation and greedy or stochastic acceptance.

Position bookkeeping: `req.committed` is the number of KV positions actually written;
the allocator's table.seq_len is the reservation, kept at committed + num_draft + 1
for running requests so verify / draft rows always have slots.

One step: rows for the running requests (verify: root + tree nodes; plain: one token)
and the prompts of the newly admitted ones, one target forward each (a pure verify or
decode batch fits its graph), then `_accept` scores every row's draw and every node's
probability in one chunked pass over lm_head, syncs once, and walks each tree on the
host (spec/sampling.py). A chain is a tree with linear parents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from kv.allocator import BlockAllocator
from kv.cache import AttnMeta, PagedKVCache
from model.llama import LlamaForCausalLM, eagle3_aux_layers
from sched.scheduler import Request, SamplingParams, Scheduler, TokenInfo
from spec.sampling import draw, process, walk
from spec.tree import Tree, ancestors, flat_masks


@dataclass(frozen=True)
class EngineConfig:
    num_blocks: int
    block_size: int = 16
    max_admit: int = 32  # admissions per step
    max_running: int | None = None  # concurrent sequences; None = whatever the pool holds
    cuda_graphs: bool = True  # decode / verify / draft steps via engine/graphs.py; only takes effect on CUDA
    spec_depth: int = 0  # EAGLE-3 draft depth; 0 = no speculation
    spec_topk: int = 1  # 1 = chain; >1 = static tree of topk * depth nodes (PLAN D5)
    spec_row_budget: int = 0  # >0: chain depth per step = min(spec_depth, budget // batch); compute-bound batches draft less
    record_logits: bool = False  # keep each request's per-step next-token logits on CPU (tests)
    profile: bool = False  # per-phase CUDA-event timings in Engine.prof (tests / bench)
    telemetry_path: str | None = None  # spec/telemetry.py JSON lines, one per verify step per request
    head_chunk: int = 256  # rows per lm_head + log-softmax chunk in _accept (fp32 [chunk, V])

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
        self.sched = Scheduler(self.alloc, config.max_admit, config.max_running, config.num_draft)
        self.eos = set(model.config.eos_token_ids)
        self._next_id = 0

        self.depth = config.spec_depth
        self.topk = config.spec_topk
        self.drafter = drafter
        self.taps = eagle3_aux_layers(model.config.num_hidden_layers)
        if self.depth:
            assert drafter is not None, "spec_depth > 0 needs an EAGLE-3 drafter"
            self.draft_kv = PagedKVCache(drafter.config.rope_config(), config.num_blocks, config.block_size, self.device, self.dtype)
        self._draft_ids = None  # target ids of the draft vocab, for sampled proposals

        self.graphs = self.verify_graphs = self.draft_graphs = self.extend_graphs = None
        self.usable_blocks = config.num_blocks  # minus the graph pad block, if any
        if config.cuda_graphs and self.device.type == "cuda":
            self._capture_graphs()
        self.prof: dict[str, list[float]] = {}
        self._events: list[tuple[str, torch.cuda.Event]] = []
        self.tele = None
        if config.telemetry_path:
            from spec.telemetry import Telemetry

            self.tele = Telemetry(config.telemetry_path)

    def _mark(self, name: str) -> None:
        if self.config.profile:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._events.append((name, e))

    def _flush_prof(self) -> None:
        if not self._events:
            return
        torch.cuda.synchronize()
        for (a, ea), (b, eb) in zip(self._events, self._events[1:]):
            self.prof.setdefault(b, []).append(ea.elapsed_time(eb))
        self._events.clear()

    def _capture_graphs(self) -> None:
        from engine.graphs import GraphRunner

        c, cfg = self.config, self.model.config
        pad_block = self.alloc.alloc(-1, 1).blocks[0]  # reserved forever for padded rows
        self.usable_blocks -= 1
        heads = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, self.dtype, self.device)
        if not self.depth:
            self.graphs = GraphRunner(self.kv, pad_block, 1, False, *heads)
            self.graphs.capture(lambda ids, _h, meta, be: self.model(ids, self.kv, meta, backend=be, head=False)[0])
            return
        def verify(ids, _h, meta, be):
            h, aux = self.model(ids, self.kv, meta, aux_layers=self.taps, backend=be, head=False)
            return h, torch.cat(aux, dim=-1)

        # the chain's verify rows are causal: the unmasked prefill wrapper, batched up to 512;
        # a runner per depth when the per-step depth varies (spec_row_budget)
        depths = range(1, self.depth + 1) if c.spec_row_budget and self.topk == 1 else [self.depth]
        self.verify_graphs = {}
        for d in depths:
            self.verify_graphs[d * self.topk + 1] = g = GraphRunner(self.kv, pad_block, d * self.topk + 1, self.topk > 1, *heads, buckets=self._buckets)
            g.capture(verify)
        self._capture_draft_graphs(pad_block)

    @property
    def _buckets(self) -> tuple[int, ...]:
        return tuple(b for b in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if b <= (512 if self.topk == 1 else 32))

    def _capture_draft_graphs(self, pad_block: int) -> None:
        from engine.graphs import GraphRunner

        d = self.drafter.config
        dheads = (d.num_attention_heads, d.num_key_value_heads, d.head_dim, self.dtype, self.device)
        self.draft_graphs = GraphRunner(self.draft_kv, pad_block, self.topk, self.topk > 1, *dheads,
                                        hidden_dim=d.hidden_size, buckets=self._buckets)
        self.draft_graphs.capture(lambda ids, h, meta, be: self.drafter(ids, h, self.draft_kv, meta, backend=be))
        # the extend has a+1 <= depth+1 rows: pad to depth+1 with a causal mask; pad rows
        # write K/V into the tree region, which the draft levels overwrite and mask out
        self.extend_graphs = GraphRunner(self.draft_kv, pad_block, self.depth + 1, True, *dheads,
                                         hidden_dim=d.num_tapped * d.hidden_size, buckets=self._buckets)
        self.extend_graphs.capture(lambda ids, h, meta, be: self.drafter(ids, h, self.draft_kv, meta, backend=be))

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
        if self.tele:
            self.tele.f.flush()
        return [r.out_tokens for r in reqs]

    def _emit(self, req: Request, infos: list[TokenInfo]) -> bool:
        """Append tokens, truncating at eos / max_tokens. Returns True if the request is done."""
        for t in infos:
            req.out_tokens.append(t.token)
            if req.params.logprobs:
                req.logprobs.append(t)
            if t.token in self.eos or len(req.out_tokens) >= req.params.max_tokens:
                return True
        return False

    # ------------------------------------------------------------------ weights

    @torch.no_grad()
    def update_weights(self, state: dict[str, torch.Tensor], module: torch.nn.Module | None = None) -> None:
        """Copy tensors in place (captured graphs hold the parameter pointers; the tied
        lm_head follows embed_tokens). Running requests restart from prompt + output:
        their KV was computed by the old weights."""
        params = dict((module or self.model).named_parameters()) | dict((module or self.model).named_buffers())
        for name, t in state.items():
            params[name].copy_(t)
        for r in list(self.sched.running):
            self.sched.restart(r)

    def swap_drafter(self, head) -> None:
        """A state dict, or an Eagle3Head: same architecture -> weights copied in place;
        otherwise the head is rebound and its graphs recaptured. Restarts running requests
        (their draft KV is stale)."""
        if isinstance(head, dict):
            return self.update_weights(head, self.drafter)
        if head.config == self.drafter.config:
            return self.update_weights(head.state_dict(), self.drafter)
        self.drafter, self._draft_ids = head, None
        self.draft_kv = PagedKVCache(head.config.rope_config(), self.config.num_blocks, self.config.block_size, self.device, self.dtype)
        if self.draft_graphs is not None:
            self._capture_draft_graphs(self.draft_graphs.pad_block)
        for r in list(self.sched.running):
            self.sched.restart(r)

    # ------------------------------------------------------------------ steps

    def step(self) -> list[Request]:
        """Admit, then run one step; returns requests that finished."""
        running, new = self.sched.next_batch()
        return self._run(running, new) if running or new else []

    def _run(self, running: list[Request], new: list[Request]) -> list[Request]:
        bs, N = self.config.block_size, self.config.num_draft
        dev = None if (self.graphs or self.verify_graphs) is not None else self.device  # host-only meta for the runners
        ids_r, rows_r, masks = [], [], None
        for req in running:
            L = req.committed
            if self.depth:  # verify rows: root (next input token) + the tree nodes
                t = req.tree
                ids_r += [req.out_tokens[-1]] + t.tokens
                rows_r.append(([L] + [L + d for d in t.depths], [L] + [L + 1 + i for i in range(t.n)], L + 1 + t.n))
            else:
                self.alloc.append(req.id, 1)
                ids_r.append(req.out_tokens[-1])
                rows_r.append(([L], [L], L + 1))
        ids_n, rows_n = [], []
        for req in new:
            n = len(req.prompt_ids)
            ids_n += req.prompt_ids
            rows_n.append((list(range(n)), list(range(n)), n))
            if self.depth:
                self.alloc.append(req.id, N + 1)
        self._mark("start")
        outs = []
        if running and self.topk > 1:  # rows [root] + nodes over kv = prefix + root + slots
            A = ancestors(torch.tensor([r.tree.parents for r in running], device=self.device), self.depth)
            masks = flat_masks([r.committed for r in running], torch.cat([torch.zeros_like(A[:, :1]), A], 1), 1)
        if running:
            outs.append(self._verify_forward(ids_r, AttnMeta.from_rows([r.table for r in running], rows_r, bs, dev, masks)))
        if new:
            outs.append(self._verify_forward(ids_n, AttnMeta.from_rows([r.table for r in new], rows_n, bs, dev)))
        h = torch.cat([o[0] for o in outs])
        aux = torch.cat([o[1] for o in outs]) if self.depth else None
        self._mark("verify")
        finished, alive = self._accept(h, aux, running + new, [len(r[0]) for r in rows_r + rows_n])
        self._mark("accept")
        if alive:
            (self._draft_tree if self.topk > 1 else self._draft)(alive)
        self._flush_prof()
        if self.tele:
            self.tele.write(self._tele_rows, {k: v[-1] for k, v in self.prof.items()} if self.config.profile else None)
        return finished

    def _verify_forward(self, ids: list[int], meta: AttnMeta) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Target forward -> (normed hidden [N, H], aux taps [N, 3H] when speculating);
        the captured graph when the batch is uniform."""
        runner = self.verify_graphs.get(meta.qo_lens[0]) if self.verify_graphs else self.graphs
        if runner is not None and runner.can_run(meta):
            out = runner.run(meta, ids)
            return out if self.depth else (out[0], None)
        meta = self._materialize(meta)
        input_ids = torch.tensor(ids, dtype=torch.int64, device=self.device)
        h, aux = self.model(input_ids, self.kv, meta, aux_layers=self.taps if self.depth else (), head=False)
        return h, torch.cat(aux, dim=-1) if self.depth else None

    def _materialize(self, meta: AttnMeta) -> AttnMeta:
        """Host-only meta (built for a graph runner) -> device tensors for an eager forward."""
        if meta.qo_indptr is not None:
            return meta
        d = self.device
        i32 = lambda x: torch.tensor(x, dtype=torch.int32, device=d)
        i64 = lambda x: torch.tensor(x, dtype=torch.int64, device=d)
        import dataclasses

        return dataclasses.replace(
            meta,
            qo_indptr=i32([0] + list(torch.tensor(meta.qo_lens).cumsum(0).tolist())),
            kv_indptr=i32(meta.kv_indptr_host), kv_indices=i32(meta.kv_indices_host),
            kv_last_page_len=i32(meta.kv_last_page_len_host),
            positions=i64(meta.positions_host), slots=i64(meta.slots_host),
        )

    # ------------------------------------------------------------------ accept

    def _accept(self, h: torch.Tensor, aux: torch.Tensor | None, batch: list[Request], spans: list[int]):
        """spans[b] rows of request b in h; a request with a tree has 1 + tree.n verify rows,
        otherwise only its last row matters. Scores, walks, emits; -> (finished, alive)."""
        P = [r.params for r in batch]
        sel, tmp, tk, tp, u_row, first = [], [], [], [], [], []  # per selected row
        nrow, ntok, nq, mass, u_node, qrows, qprobs = [], [], [], [], [], [], []  # per node; sampled-q rows
        s = 0
        for req, n, p in zip(batch, spans, P):
            t = req.tree
            rows = list(range(s, s + n)) if t is not None else [s + n - 1]
            u = torch.rand((t.n if t else 0) + len(rows), generator=req.gen).tolist()
            first.append(len(sel))
            if t is not None:
                base = len(sel)
                nrow += [base + 1 + q for q in t.parents]  # parent -1 -> the root row
                ntok += t.tokens
                u_node += u[: t.n]
                if req.draft_q is not None:  # sampled chain proposals: q is the head's distribution at row j
                    nq += [math.exp(x) for x in t.q]
                    qrows += [base + j for j in range(t.n)]
                    qprobs.append(req.draft_q)
                else:  # argmax proposals: a point mass on each node's token
                    nq += [1.0] * t.n
                mass += [float(req.draft_q is None)] * t.n
            sel += rows
            u_row += u[-len(rows):]
            tmp += [p.temperature] * len(rows)
            tk += [p.top_k] * len(rows)
            tp += [p.top_p] * len(rows)
            s += n
        dev = self.device
        f32 = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)
        i64 = lambda x: torch.tensor(x, dtype=torch.int64, device=dev)
        nucleus = any(p.top_k > 0 or p.top_p < 1 for p in P)
        want_rank = any(p.acceptance == "relaxed" and p.accept_topk > 0 for p in P)
        if qrows and self._draft_ids is None:
            self._draft_ids = self.drafter.to_target_ids(torch.arange(qprobs[0].shape[-1], device=dev))
        qs = (i64(qrows), torch.cat(qprobs)) if qrows else None
        tok, lpr, lpp, pn, lprn, rank, raw = self._score(
            h[i64(sel)], f32(tmp), i64(tk), f32(tp), f32(u_row), i64(nrow), i64(ntok), f32(mass), qs, nucleus, want_rank)

        finished, alive, tele, m, N, src, dst = [], [], [], 0, self.config.num_draft, [], []
        for b, (req, n, p) in enumerate(zip(batch, spans, P)):
            t, f, L = req.tree, first[b], req.committed
            if t is None:  # plain decode / prefill: the last row's draw
                j, a, path = f, 0, []
                infos = [TokenInfo(tok[j], lpr[j], lpp[j], None, "bonus")]
            else:
                ch = t.children()
                relaxed = (p.tau, p.accept_topk) if p.acceptance == "relaxed" else None
                path, dec = walk(ch, pn[m : m + t.n], nq[m : m + t.n], u_node[m : m + t.n],
                                 rank[m : m + t.n] if want_rank else None, relaxed)
                a, last = len(path), path[-1] if path else -1
                j = f + 1 + last
                final = "resample" if ch.get(last) else "bonus"
                infos = [TokenInfo(t.tokens[i], lprn[m + i], math.log(pn[m + i]) if pn[m + i] < 1 else 0.0,
                                   t.q[i] if t.q else None, "draft") for i in path]
                infos.append(TokenInfo(tok[j], lpr[j], lpp[j], None, final))
                req.accepted.append(a)
                if self.tele:
                    nodes = [dict(tok=t.tokens[i], parent=t.parents[i], p=pn[m + i], q=t.q[i] if t.q else None,
                                  u=u_node[m + i], d=dec[i]) for i in range(t.n)]
                    tele.append(dict(req=req.id, pos=len(req.out_tokens), depth=self.depth, topk=self.topk,
                                     accepted=a, final=final, nodes=nodes))
                if a and path != list(range(a)):  # compact the accepted path into chain order
                    bs, blocks = self.config.block_size, req.table.blocks
                    phys = lambda q: blocks[q // bs] * bs + q % bs
                    src += [phys(L + 1 + q) for q in path]
                    dst += [phys(L + 1 + k) for k in range(a)]
                m += t.n
            if raw is not None:
                req.step_logits.append(raw[j].cpu())
            done = self._emit(req, infos)
            req.committed += a + 1 if t is not None else n
            if done:
                self.sched.finish(req)
                finished.append(req)
                continue
            if self.depth:
                self.alloc.rollback(req.id, req.committed)
                self.alloc.append(req.id, N + 1)
                if t is None:
                    s0 = sel[f] - n + 1
                    req.ext_ids, req.ext_hidden = req.prompt_ids[1:] + [infos[0].token], aux[s0 : s0 + n]
                else:
                    s0 = sel[f]
                    req.ext_ids, req.ext_hidden = req.out_tokens[-(a + 1) :], aux[[s0] + [s0 + 1 + i for i in path]]
                alive.append(req)
        if src:  # depth-major nodes: every source slot is >= its destination, one move for the batch
            self.kv.move(i64(src), i64(dst))
        self._tele_rows = tele
        return finished, alive

    def _score(self, hs, T, top_k, top_p, u, nrow, ntok, mass, qs, nucleus: bool, want_rank: bool):
        """Chunked over rows: raw logits -> per row (draw, its raw and processed log-prob)
        and per node (target prob of its token at its parent row, raw log-prob, rank).
        The draw is from the residual after the proposal mass tested at that row: a point
        mass per argmax node (`mass`) or the sampled proposal's distribution (`qs`). One sync."""
        R, M, dev = hs.shape[0], nrow.numel(), hs.device
        tok = torch.empty(R, dtype=torch.int64, device=dev)
        lpr, lpp = torch.empty(R, device=dev), torch.empty(R, device=dev)
        pn, lprn, rank = (torch.zeros(M, device=dev) for _ in range(3))
        raw = [] if self.config.record_logits else None
        for s in range(0, R, self.config.head_chunk):
            e = min(s + self.config.head_chunk, R)
            z = self.model.lm_head(hs[s:e]).float()
            if raw is not None:
                raw.append(z)
            lse, am, g = z.logsumexp(-1), z.argmax(-1), T[s:e] == 0
            lp = process(z, T[s:e], top_k[s:e], top_p[s:e], nucleus)
            inn, idx = (nrow >= s) & (nrow < e), (nrow - s).clamp(0, e - s - 1)
            val = lp[idx, ntok]
            pn = torch.where(inn, torch.where(g[idx], (am[idx] == ntok).float(), val.exp()), pn)
            lprn = torch.where(inn, z[idx, ntok] - lse[idx], lprn)
            if want_rank:
                rank = torch.where(inn, (lp[idx] > val[:, None]).sum(-1).float(), rank)
            q = torch.zeros_like(z).index_put_((idx, ntok), inn.float() * mass, accumulate=True)
            if qs is not None:
                qr, qp = qs
                inq = ((qr >= s) & (qr < e)).float()
                q.index_put_(((qr - s).clamp(0, e - s - 1)[:, None], self._draft_ids[None, :]), qp * inq[:, None], accumulate=True)
            d = torch.where(g, am, draw(lp, q, u[s:e]))
            tok[s:e] = d
            lpr[s:e] = z.gather(1, d[:, None]).squeeze(1) - lse
            lpp[s:e] = lp.gather(1, d[:, None]).squeeze(1).masked_fill(g, 0.0)
        vals = torch.cat([tok.float(), lpr, lpp, pn, lprn, rank]).tolist()
        tok, lpr, lpp = [int(x) for x in vals[:R]], vals[R : 2 * R], vals[2 * R : 3 * R]
        pn, lprn, rank = vals[3 * R : 3 * R + M], vals[3 * R + M : 3 * R + 2 * M], vals[3 * R + 2 * M :]
        return tok, lpr, lpp, pn, lprn, rank, torch.cat(raw) if raw else None

    # ------------------------------------------------------------------ draft

    def _extend(self, reqs: list[Request]) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft extend over each request's newly committed tokens. -> (logits, h) at the
        last real row of each request. Captured when every request fits depth+1 rows."""
        bs = self.config.block_size
        tables = [r.table for r in reqs]
        n = [len(r.ext_ids) for r in reqs]
        starts = [r.committed - k for r, k in zip(reqs, n)]
        Re = self.depth + 1
        g = self.extend_graphs
        if g is not None and max(n) <= Re and g.bucket(len(reqs)) is not None:
            rows = [(list(range(st, st + Re)), list(range(st, st + Re)), st + Re) for st in starts]
            meta = AttnMeta.from_rows(tables, rows, bs, None)
            ids = [t for r, k in zip(reqs, n) for t in r.ext_ids + [0] * (Re - k)]
            for b, r in enumerate(reqs):  # straight into the runner's static hidden buffer
                g.hidden[b * Re : b * Re + n[b]].copy_(r.ext_hidden)
            logits, h = g.run(meta, ids, g.hidden)
            last = torch.tensor([b * Re + k - 1 for b, k in enumerate(n)], device=self.device)
            return logits[last], h[last]
        ids = torch.tensor([t for r in reqs for t in r.ext_ids], dtype=torch.int64, device=self.device)
        hidden = torch.cat([r.ext_hidden for r in reqs])
        meta = AttnMeta.build(tables, n, bs, self.device, starts)
        logits, h = self.drafter(ids, hidden, self.draft_kv, meta)
        last = meta.last_token_idx
        return logits[last], h[last]

    def _draft_forward(self, ids, hidden: torch.Tensor, meta: AttnMeta) -> tuple[torch.Tensor, torch.Tensor]:
        if self.draft_graphs is not None and self.draft_graphs.can_run(meta):
            return self.draft_graphs.run(meta, ids, hidden)
        meta = self._materialize(meta)
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids, dtype=torch.int64, device=self.device)
        return self.drafter(ids, hidden, self.draft_kv, meta)

    def _draft(self, reqs: list[Request]) -> None:
        """Extend the draft cache over the newly committed tokens, then chain D-1 steps.
        Proposals are the head's argmax, or for draft_sampling="sample" a draw from its
        (temperature-scaled) distribution, which is then kept for the Leviathan test."""
        B, dev, c = len(reqs), self.device, self.config
        D = min(self.depth, max(1, c.spec_row_budget // B)) if c.spec_row_budget else self.depth
        tables = [r.table for r in reqs]
        mdev = None if self.draft_graphs is not None else dev
        T = torch.tensor([r.params.temperature if r.params.draft_sampling == "sample" else 0.0 for r in reqs], device=dev)
        sample = [r.params.draft_sampling == "sample" and r.params.temperature > 0 for r in reqs]
        U = torch.tensor([torch.rand(D, generator=r.gen).tolist() if s else [0.0] * D for r, s in zip(reqs, sample)], device=dev)
        zero = torch.zeros((), device=dev)
        logits, h = self._extend(reqs)
        self._mark("extend")
        toks, lqs, qs = [], [], []
        for j in range(D):
            if j:
                meta = AttnMeta.build(tables, [1] * B, self.config.block_size, mdev, [r.committed + j - 1 for r in reqs])
                logits, h = self._draft_forward(toks[-1], h, meta)
            lq = process(logits.float(), T, torch.zeros(B, dtype=torch.int64, device=dev), torch.ones(B, device=dev), False)
            d = lq.argmax(-1)
            if any(sample):
                d = torch.where(T > 0, draw(lq, zero, U[:, j]), d)
                qs.append(lq.exp())
            toks.append(self.drafter.to_target_ids(d))
            lqs.append(lq.gather(1, d[:, None]).squeeze(1))
        out = torch.stack([torch.stack(toks, 1).float(), torch.stack(lqs, 1)]).tolist()  # [2, B, D], one sync
        qs = torch.stack(qs, 1) if qs else None
        self._mark("levels")
        for b, r in enumerate(reqs):
            r.tree = Tree.chain([int(x) for x in out[0][b]], out[1][b])
            r.draft_q = qs[b] if sample[b] else None
            r.ext_ids, r.ext_hidden = [], None

    def _draft_tree(self, reqs: list[Request]) -> None:
        """Extend the draft cache over the committed tokens, then grow topk nodes per level:
        one topk over the K*K candidates per request. Parents and the level masks live on
        the device (spec/tree.py), so the whole tree syncs to the host once at the end."""
        K, D, N, bs, dev = self.topk, self.depth, self.config.num_draft, self.config.block_size, self.device
        B = len(reqs)
        tables, prefix = [r.table for r in reqs], [r.committed for r in reqs]
        mdev = None if self.draft_graphs is not None else dev
        logits, h_last = self._extend(reqs)
        self._mark("extend")
        top = torch.log_softmax(logits.float(), dim=-1).topk(K, dim=-1)
        tok = self.drafter.to_target_ids(top.indices)  # [B, K] the current level's tokens (target ids)
        tokens, qs, scores = [tok], [top.values], top.values  # per level [B, K]; cumulative log-prob
        parents = torch.full((B, N), -1, dtype=torch.int64, device=dev)  # filled level by level
        h_parent = h_last[:, None, :].expand(B, K, -1)  # hidden feeding each current node

        for d in range(1, D):
            lo = (d - 1) * K  # the current level's nodes are lo .. lo+K
            rows = [([L + d - 1] * K, [L + lo + i for i in range(K)], L + N) for L in prefix]
            masks = flat_masks(prefix, ancestors(parents, d)[:, lo : lo + K], 0)
            meta = AttnMeta.from_rows(tables, rows, bs, mdev, masks)
            logits, h = self._draft_forward(tok.reshape(-1), h_parent.reshape(B * K, -1), meta)
            child = torch.log_softmax(logits.float(), dim=-1).view(B, K, -1).topk(K, dim=-1)  # [B, K, K]
            cand = (scores[:, :, None] + child.values).reshape(B, K * K)
            best = cand.topk(K, dim=-1).indices  # [B, K] global top-k children per request
            pl = best // K  # which current-level node each kept child hangs off
            tok = self.drafter.to_target_ids(child.indices.reshape(B, K * K).gather(1, best))
            scores = cand.gather(1, best)
            h_parent = h.view(B, K, -1).gather(1, pl[:, :, None].expand(B, K, h.shape[-1]))
            parents[:, lo + K : lo + 2 * K] = lo + pl
            tokens.append(tok)
            qs.append(child.values.reshape(B, K * K).gather(1, best))

        out = torch.stack([torch.cat(tokens, 1).float(), parents.float(), torch.cat(qs, 1)]).tolist()  # one sync
        self._mark("levels")
        depths = [1 + i // K for i in range(N)]
        for b, r in enumerate(reqs):
            r.tree = Tree([int(x) for x in out[0][b]], [int(x) for x in out[1][b]], depths, out[2][b])
            r.ext_ids, r.ext_hidden = [], None
