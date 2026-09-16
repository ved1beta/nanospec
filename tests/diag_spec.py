"""H100 diagnostics for the speculative path. Not a test.

1. lockstep: FlashInfer engine vs SDPA engine with identical drafts each step; report the
   first step where the verify logits differ beyond noise and which rows.
2. drafter probe: teacher-forced accuracy of the EAGLE-3 head on a chat prompt + the
   target's greedy continuation (extend rows), and the chain-step accuracy.
"""

from __future__ import annotations

import os
import sys

import torch
from transformers import AutoTokenizer

from engine.engine import Engine, EngineConfig
from kv.allocator import BlockAllocator
from kv.cache import AttnMeta, PagedKVCache
from model.loader import load_eagle3, load_model
from sched.scheduler import SamplingParams
from tests.conftest import DEVICE, DTYPE, MODEL
from tests.prompts import G1_PROMPTS
from tests.test_g5_spec_chain import DEPTH, SelfDrafter

EAGLE = os.environ.get("NANOSPEC_EAGLE") or "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
NB, BS = 1024, 16


def ulps(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs() / (2.0**-7 * b.abs().amax(-1, keepdim=True))).amax(-1)


def lockstep(tok, prompt_idx=8, steps=60):
    fi = load_model(MODEL, DEVICE, DTYPE, backend="flashinfer")
    sd = load_model(MODEL, DEVICE, DTYPE, backend="sdpa")
    cfg = EngineConfig(NB, BS, max_admit=1, cuda_graphs=False, spec_depth=DEPTH)
    ef, es = Engine(fi, cfg, SelfDrafter(fi)), Engine(sd, cfg, SelfDrafter(sd))
    cap = {}
    for name, e in (("fi", ef), ("sd", es)):
        orig = e._verify_forward

        def hook(ids, meta, _o=orig, _n=name):
            out = _o(ids, meta)
            cap[_n] = (out[0].clone(), list(ids), meta)
            return out

        e._verify_forward = hook
    ids = tok(G1_PROMPTS[prompt_idx], return_tensors="pt").input_ids[0].tolist()
    rf, rs = ef.add(ids, SamplingParams(steps * 3)), es.add(ids, SamplingParams(steps * 3))
    print(f"\n== lockstep prompt {prompt_idx}")
    for step in range(steps):
        if rf.state == "done" or rs.state == "done":
            break
        rf.drafts = list(rs.drafts)  # identical verify inputs
        ef.step()
        es.step()
        lf, idf, mf = cap["fi"]
        ls, ids_, ms = cap["sd"]
        assert idf == ids_, (idf, ids_)
        u = ulps(lf, ls)
        af, as_ = lf.argmax(-1).tolist(), ls.argmax(-1).tolist()
        flag = "" if u.max() <= 4 else "  <-- DIFF"
        print(f"step {step:3d} rows {len(idf)} kv_len {mf.seq_lens} start {mf.seq_lens[0] - mf.qo_lens[0]} "
              f"ulps per row {[round(x, 1) for x in u.tolist()]} argmax eq {[a == b for a, b in zip(af, as_)]} "
              f"accepted fi {rf.accepted[-1] if rf.accepted else '-'} sd {rs.accepted[-1] if rs.accepted else '-'}{flag}")
        if rf.out_tokens != rs.out_tokens:
            print(f"  out_tokens differ: fi {rf.out_tokens[-8:]} sd {rs.out_tokens[-8:]}")
            break
        rf.drafts = list(rs.drafts)


def drafter_probe(tok):
    """Teacher-forced top-1 / top-5 accuracy of the head's extend rows on the target's
    own greedy continuation, for a few plumbing variants."""
    import dataclasses

    model = load_model(MODEL, DEVICE, DTYPE, backend="sdpa")
    head = load_eagle3(EAGLE, model, backend="sdpa")
    L = model.config.num_hidden_layers
    msgs = [{"role": "user", "content": "Explain, step by step, how paged attention keeps memory fragmentation low in a serving engine."}]
    ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
    eng = Engine(model, EngineConfig(NB, BS, cuda_graphs=False))
    out = eng.generate([ids], SamplingParams(96))[0]
    seq = ids + out
    n = len(seq)
    print(f"\n== drafter probe: {len(ids)} prompt + {len(out)} generated tokens")
    print("  sample: " + tok.decode(out[:40]).replace("\n", " "))

    def target_aux(taps):
        kv = PagedKVCache(model.config, NB, BS, DEVICE, DTYPE)
        t = BlockAllocator(NB, BS).alloc(0, n)
        meta = AttnMeta.build([t], [n], BS, DEVICE, [0])
        _, aux = model(torch.tensor(seq, device=DEVICE), kv, meta, aux_layers=taps)
        return torch.cat(aux, -1)

    def run_head(aux, theta):
        h = head
        if theta != h.config.rope_theta:
            from model.llama import RotaryEmbedding

            h.rotary_emb = RotaryEmbedding(dataclasses.replace(h.config, rope_theta=theta).rope_config())
        dkv = PagedKVCache(h.config.rope_config(), NB, BS, DEVICE, DTYPE)
        dt = BlockAllocator(NB, BS).alloc(0, n)
        m = AttnMeta.build([dt], [n - 1], BS, DEVICE, [0])
        dlog, _ = h(torch.tensor(seq[1:], device=DEVICE), aux[: n - 1], dkv, m)
        pred = h.to_target_ids(dlog.argmax(-1)).tolist()
        top5 = h.to_target_ids(dlog.topk(5, dim=-1).indices).tolist()
        gen = list(range(len(ids) - 1, n - 2))  # rows whose answer is a generated token
        acc1 = sum(pred[i] == seq[i + 2] for i in gen) / len(gen)
        acc5 = sum(seq[i + 2] in top5[i] for i in gen) / len(gen)
        return acc1, acc5, pred

    for taps in [(2, L // 2, L - 3), (2, L // 2, L - 2), (1, L // 2 - 1, L - 4)]:
        aux = target_aux(taps)
        for theta in (10000.0, 500000.0):
            a1, a5, pred = run_head(aux, theta)
            print(f"  taps {taps} theta {theta:>8.0f}: extend top-1 {a1:.2f} top-5 {a5:.2f}   draft: "
                  + tok.decode(pred[len(ids) - 1 : len(ids) + 19]).replace("\n", " "))


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "lockstep"):
        lockstep(tok)
    if what in ("all", "drafter"):
        drafter_probe(tok)
