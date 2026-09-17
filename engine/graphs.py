"""CUDA-graph replay of fixed-shape steps, one graph per batch bucket.

Every request contributes exactly R rows, so a bucket of B requests is B*R rows. Static
input buffers (token ids, positions, cache slots, optional hidden states) and FlashInfer's
own indptr / indices / last-page / packed-mask buffers are rewritten in place each step;
`plan()` runs outside the graph, `run()` inside. A batch is padded up to its bucket with
rows that attend to one reserved pad block and write their K/V there.

Four uses (engine/engine.py): target decode (R=1), target verify (R=num_draft+1, masked,
with EAGLE-3 aux outputs), draft chain step (R=1, hidden input), draft tree level
(R=topk, masked, hidden input). The ragged draft *extend* stays eager.
"""

from __future__ import annotations

from typing import Callable

import torch

from kv.cache import AttnMeta, PagedKVCache

BUCKETS = (1, 2, 4, 8, 16, 32, 64)


class _GraphBackend:
    """Attention backend used inside a captured graph: already planned, just run."""

    fused = True

    def __init__(self, wrapper) -> None:
        self.wrapper = wrapper

    def plan(self, meta: AttnMeta) -> None:
        pass

    def run(self, q: torch.Tensor, layer: int, kv: PagedKVCache) -> torch.Tensor:
        return self.wrapper.run(q, (kv.k[layer], kv.v[layer]))


class GraphRunner:
    def __init__(
        self,
        kv: PagedKVCache,
        pad_block: int,
        rows_per_req: int,
        masked: bool,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype,
        device,
        hidden_dim: int | None = None,
        buckets=BUCKETS,
    ) -> None:
        import flashinfer

        self.kv, self.pad_block, self.R, self.masked = kv, pad_block, rows_per_req, masked
        self.device, self.dtype = device, dtype
        self.buckets = tuple(sorted(buckets))
        max_bs = self.buckets[-1]
        max_rows = max_bs * self.R
        max_kv = kv.num_blocks * kv.block_size

        self.inputs = torch.zeros(3, max_rows, dtype=torch.int64, device=device)  # ids, positions, slots
        self.input_ids, self.positions, self.slots = self.inputs[0], self.inputs[1], self.inputs[2]
        self.stage = torch.zeros(3, max_rows, dtype=torch.int64).pin_memory()  # host staging, allocated once
        self.hidden = torch.zeros(max_rows, hidden_dim, dtype=dtype, device=device) if hidden_dim else None
        self.kv_indices = torch.zeros(kv.num_blocks + max_bs, dtype=torch.int32, device=device)
        self.workspace = torch.empty(128 << 20, dtype=torch.uint8, device=device)

        self.wrappers = {}
        for bs in self.buckets:
            indptr_buf = torch.zeros(bs + 1, dtype=torch.int32, device=device)
            last_buf = torch.zeros(bs, dtype=torch.int32, device=device)
            # the two wrappers spell their buffer kwargs differently (_buf vs _buffer)
            common = dict(paged_kv_indptr_buf=indptr_buf, paged_kv_indices_buf=self.kv_indices, paged_kv_last_page_len_buf=last_buf)
            if masked:  # custom masks are FA2-only
                w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace, "NHD", backend="fa2", use_cuda_graph=True,
                    qo_indptr_buf=torch.arange(0, (bs + 1) * self.R, self.R, dtype=torch.int32, device=device),
                    custom_mask_buf=torch.zeros((bs * self.R * max_kv + 7) // 8, dtype=torch.uint8, device=device),
                    mask_indptr_buf=torch.zeros(bs + 1, dtype=torch.int32, device=device),
                    **common,
                )
            elif self.R == 1:
                w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace, "NHD", use_cuda_graph=True,
                    use_tensor_cores=num_heads // num_kv_heads >= 4,
                    paged_kv_indptr_buffer=indptr_buf, paged_kv_indices_buffer=self.kv_indices,
                    paged_kv_last_page_len_buffer=last_buf,
                )
            else:
                w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace, "NHD", use_cuda_graph=True,
                    qo_indptr_buf=torch.arange(0, (bs + 1) * self.R, self.R, dtype=torch.int32, device=device),
                    **common,
                )
            self.wrappers[bs] = w
        self.plan_kwargs = dict(
            num_qo_heads=num_heads, num_kv_heads=num_kv_heads, page_size=kv.block_size,
            q_data_type=dtype, kv_data_type=dtype,
        )
        self.head_dim = head_dim
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.outputs: dict[int, tuple] = {}

    # ------------------------------------------------------------------ batch -> static buffers

    def bucket(self, batch_size: int) -> int | None:
        return next((b for b in self.buckets if b >= batch_size), None)

    def can_run(self, meta: AttnMeta) -> bool:
        return all(n == self.R for n in meta.qo_lens) and self.bucket(len(meta.qo_lens)) is not None

    def _load(self, meta: AttnMeta, ids, bs: int, hidden: torch.Tensor | None = None) -> AttnMeta:
        """Plan the bucket's wrapper and fill the static buffers; returns the padded meta."""
        B, R = len(meta.qo_lens), self.R
        n_pad = bs - B
        base = meta.kv_indptr_host[-1]
        indptr = meta.kv_indptr_host + list(range(base + 1, base + n_pad + 1))
        indices = meta.kv_indices_host + [self.pad_block] * n_pad
        last = meta.kv_last_page_len_host + [1] * n_pad
        i32 = lambda x: torch.tensor(x, dtype=torch.int32)

        masks = None
        if self.masked:
            # every row gets an explicit mask: an all-causal batch must not fall back to
            # "no mask" (custom_mask -> None) on the masked wrapper
            masks = list(meta.masks) if meta.masks is not None else [None] * B
            masks = [m if m is not None else AttnMeta.causal_mask(R, k, self.device) for m, k in zip(masks, meta.seq_lens)]
            masks += [torch.ones(R, 1, dtype=torch.bool, device=self.device)] * n_pad
        pad_meta = AttnMeta(  # only positions / slots are read inside the graph
            qo_indptr=None, kv_indptr=None, kv_indices=None, kv_last_page_len=None,
            positions=self.positions[: bs * R],
            slots=self.slots[: bs * R],
            seq_lens=list(meta.seq_lens) + [1] * n_pad,
            qo_lens=[R] * bs,
            block_size=meta.block_size,
            masks=masks,
        )
        w = self.wrappers[bs]
        if self.masked:
            w.plan(i32(list(range(0, (bs + 1) * R, R))), i32(indptr), i32(indices), i32(last),
                   head_dim_qk=self.head_dim, custom_mask=pad_meta.flat_mask(self.device), **self.plan_kwargs)
        elif R == 1:
            w.plan(i32(indptr), i32(indices), i32(last), head_dim=self.head_dim, **self.plan_kwargs)
        else:
            w.plan(i32(list(range(0, (bs + 1) * R, R))), i32(indptr), i32(indices), i32(last),
                   head_dim_qk=self.head_dim, causal=True, **self.plan_kwargs)

        pad_slot = self.pad_block * self.kv.block_size
        n = bs * R
        stage = self.stage[:, :n]
        stage[1] = torch.tensor(meta.positions_host + [0] * (n_pad * R))
        stage[2] = torch.tensor(meta.slots_host + [pad_slot] * (n_pad * R))
        if torch.is_tensor(ids):  # already on device (draft tokens): copy, pad rows keep zeros
            self.input_ids[: B * R].copy_(ids)
            self.input_ids[B * R : n].zero_()
            self.inputs[1:, :n].copy_(stage[1:], non_blocking=True)
        else:
            stage[0] = torch.tensor(list(ids) + [0] * (n_pad * R))
            self.inputs[:, :n].copy_(stage, non_blocking=True)
        if hidden is not None and hidden.data_ptr() != self.hidden.data_ptr():
            self.hidden[: B * R].copy_(hidden)
        return pad_meta

    # ------------------------------------------------------------------ capture / replay

    def capture(self, forward: Callable) -> None:
        """forward(ids, hidden_or_None, meta, backend) -> tuple of tensors; captured per
        bucket, largest first so the shared pool is sized by the biggest."""
        pool = torch.cuda.graph_pool_handle()
        R = self.R
        for bs in reversed(self.buckets):
            dummy = AttnMeta(
                qo_indptr=None, kv_indptr=None, kv_indices=None, kv_last_page_len=None, positions=None, slots=None,
                seq_lens=[1] * bs, qo_lens=[R] * bs, block_size=self.kv.block_size,
                kv_indptr_host=list(range(bs + 1)), kv_indices_host=[self.pad_block] * bs,
                kv_last_page_len_host=[1] * bs, positions_host=[0] * (bs * R),
                slots_host=[self.pad_block * self.kv.block_size] * (bs * R),
                masks=[torch.ones(R, 1, dtype=torch.bool, device=self.device)] * bs if self.masked else None,
            )
            static = self._load(dummy, [0] * (bs * R), bs, self.hidden[: bs * R] * 0 if self.hidden is not None else None)
            backend = _GraphBackend(self.wrappers[bs])
            ids = self.input_ids[: bs * R]
            hid = self.hidden[: bs * R] if self.hidden is not None else None

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    forward(ids, hid, static, backend)
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                out = forward(ids, hid, static, backend)
            self.graphs[bs] = g
            self.outputs[bs] = tuple(out) if isinstance(out, (tuple, list)) else (out,)
        torch.cuda.synchronize()

    def run(self, meta: AttnMeta, ids, hidden: torch.Tensor | None = None) -> tuple:
        """-> the captured outputs, each sliced to the B*R real rows."""
        B = len(meta.qo_lens)
        bs = self.bucket(B)
        self._load(meta, ids, bs, hidden)
        self.graphs[bs].replay()
        return tuple(o[: B * self.R] for o in self.outputs[bs])
