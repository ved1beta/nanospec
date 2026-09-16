"""H100 numerics diagnostic: determinism and kernel noise, not a test.

    modal run modal_app.py::diag
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv.allocator import BlockAllocator
from kv.cache import PagedKVCache
from model.loader import load_model
from tests.conftest import BF16_ULP, DEVICE, DTYPE, MODEL
from tests.decode import greedy
from tests.prompts import G1_PROMPTS

N_STEPS = 64
PROMPTS = [0, 14, 19]  # early divergers on the H100


def run_ours(model, ids):
    kv = PagedKVCache(model.config, 512, 16, DEVICE, DTYPE)
    return greedy(model, kv, BlockAllocator(512, 16), 0, ids, N_STEPS)


def run_hf(model, ids):
    r = model.generate(
        ids[None], attention_mask=torch.ones_like(ids[None]), do_sample=False, max_new_tokens=N_STEPS,
        output_logits=True, return_dict_in_generate=True, pad_token_id=0,
    )
    return r.sequences[0, ids.shape[0]:].tolist(), torch.stack([l[0] for l in r.logits])


def ulps_of_rowmax(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs() / (BF16_ULP * b.abs().amax(-1, keepdim=True))).amax(-1)  # per step


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, attn_implementation="sdpa").to(DEVICE).eval()
    fi = load_model(MODEL, DEVICE, DTYPE, backend="flashinfer")
    sd = load_model(MODEL, DEVICE, DTYPE, backend="sdpa")

    for p in PROMPTS:
        ids = tok(G1_PROMPTS[p], return_tensors="pt").input_ids[0].to(DEVICE)
        ht, hl = run_hf(hf, ids)
        ht2, hl2 = run_hf(hf, ids)
        ft, fl = run_ours(fi, ids)
        ft2, fl2 = run_ours(fi, ids)
        st, sl = run_ours(sd, ids)
        n = min(len(ht), len(ft), len(st), len(ht2), len(ft2))
        print(f"\n== prompt {p} ({ids.shape[0]} tokens, {N_STEPS} steps)")
        print(f"  HF deterministic:         tokens {ht == ht2}, logits bitwise {torch.equal(hl[:n], hl2[:n])}")
        print(f"  flashinfer deterministic: tokens {ft == ft2}, logits bitwise {torch.equal(fl[:n], fl2[:n])}")
        print(f"  sdpa vs HF:               tokens {st[:n] == ht[:n]}, logits bitwise {torch.equal(sl[:n], hl[:n])}, "
              f"max ulps-of-rowmax {ulps_of_rowmax(sl[:n], hl[:n]).max().item():.2f}")
        u = ulps_of_rowmax(fl[:n], hl[:n])
        k = next((i for i in range(n) if ft[i] != ht[i]), None)
        print(f"  flashinfer vs HF:         tokens {ft[:n] == ht[:n]}, ulps-of-rowmax per step: "
              f"median {u.median().item():.2f} p90 {u.quantile(0.9).item():.2f} max {u.max().item():.2f}"
              + (f"; first divergence step {k}" if k is not None else ""))
        if k is not None:
            top2 = hl[k].float().topk(2)
            gap = (top2.values[0] - top2.values[1]).item()
            print(f"    HF top-2 at step {k}: {top2.indices.tolist()} gap {gap:.4f} = "
                  f"{gap / (BF16_ULP * hl[k].float().abs().max().item()):.2f} ulps-of-rowmax; ours picked {ft[k]}")
        # prefill-only noise at every position of the HF continuation
        full = torch.cat([ids, torch.tensor(ht, device=DEVICE)])
        with torch.no_grad():
            ref = hf(full[None], attention_mask=torch.ones_like(full[None])).logits[0]
        for name, m in (("sdpa", sd), ("flashinfer", fi)):
            from kv.cache import AttnMeta
            kv = PagedKVCache(m.config, 512, 16, DEVICE, DTYPE)
            al = BlockAllocator(512, 16)
            t = al.alloc(0, full.shape[0])
            meta = AttnMeta.build([t], [full.shape[0]], 16, DEVICE)
            ours, _ = m(full, kv, meta)
            u = ulps_of_rowmax(ours, ref)
            print(f"  prefill {name:10s} vs HF:  bitwise {torch.equal(ours, ref)}, ulps-of-rowmax per position: "
                  f"median {u.median().item():.2f} p90 {u.quantile(0.9).item():.2f} max {u.max().item():.2f}")


if __name__ == "__main__":
    main()
