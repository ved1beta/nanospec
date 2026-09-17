"""FlashInfer vs SDPA on the same paged cache: prefill from 0, decode, and the
append-in-the-middle case (qo_len < kv_len) that speculative verify relies on."""

from __future__ import annotations

import pytest
import torch

from kv.allocator import BlockAllocator
from kv.cache import AttnMeta, PagedKVCache
from model.attention import SdpaBackend
from model.llama import LlamaConfig
from tests.conftest import DEVICE

pytestmark = pytest.mark.skipif(DEVICE.type != "cuda", reason="FlashInfer needs CUDA")

CFG = LlamaConfig(vocab_size=1, hidden_size=1024, intermediate_size=1, num_hidden_layers=1, num_attention_heads=8,
                  num_key_value_heads=2, head_dim=128, rms_norm_eps=1e-5, rope_theta=1e4, max_position_embeddings=4096)


def _ulps(a, b):
    return ((a.float() - b.float()).abs() / (2.0**-7 * b.float().abs().amax(-1, keepdim=True).clamp_min(1e-3))).max().item()


@pytest.mark.parametrize("case", ["prefill", "decode", "append6", "mixed"])
def test_flashinfer_matches_sdpa(case):
    from model.attention import FlashInferBackend

    torch.manual_seed(0)
    bs = 16
    kv = PagedKVCache(CFG, 256, bs, DEVICE, torch.bfloat16)
    alloc = BlockAllocator(256, bs)
    fi = FlashInferBackend(8, 2, 128, torch.bfloat16, DEVICE)
    sd = SdpaBackend()

    def fill(rid, L):
        t = alloc.alloc(rid, L + 8)
        meta = AttnMeta.build([t], [L], bs, DEVICE, [0])
        k = torch.randn(L, 2, 128, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        kv.write(0, meta.slots, k, v)
        return t

    if case == "prefill":
        t = fill(0, 0) if False else alloc.alloc(0, 40)
        meta = AttnMeta.build([t], [40], bs, DEVICE, [0])
        tables, rows = [t], [([*range(40)], [*range(40)], 40)]
    elif case == "decode":
        t = fill(0, 37)
        rows, tables = [([37], [37], 38)], [t]
    elif case == "append6":
        t = fill(0, 37)
        rows, tables = [([*range(37, 43)], [*range(37, 43)], 43)], [t]
    else:  # a decode row, an append row, and a fresh prefill in one batch
        t0, t1 = fill(0, 21), fill(1, 50)
        t2 = alloc.alloc(2, 9)
        rows = [([21], [21], 22), ([*range(50, 56)], [*range(50, 56)], 56), ([*range(9)], [*range(9)], 9)]
        tables = [t0, t1, t2]
    meta = AttnMeta.from_rows(tables, rows, bs, DEVICE)
    N = sum(len(r[0]) for r in rows)
    q = torch.randn(N, 8, 128, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(N, 2, 128, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kv.write(0, meta.slots, k, v)
    sd.plan(meta)
    ref = sd.run(q, 0, kv)
    fi.plan(meta)
    out = fi.run(q, 0, kv)
    u = _ulps(out, ref)
    print(f"\n[backends {case}] flashinfer vs sdpa {u:.2f} ulps")
    assert u <= 4, f"{case}: {u:.2f} ulps"
