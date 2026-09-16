"""One harness, three engines. Same prompts, same max_tokens, greedy.

    python -m bench.run nanospec --config nospec|chain|tree --bs 1 8 32
    python -m bench.run vllm     --config nospec|chain
    python -m bench.run sglang   --config nospec|chain|tree

Each run prints one JSON line per (engine, config, bs): output tok/s, TPOT ms, mean
accepted length per step, plus versions and commit hash. `bench/table.py` folds them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

TARGET = os.environ.get("NANOSPEC_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
EAGLE = os.environ.get("NANOSPEC_EAGLE", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
CONFIGS = {  # (depth, topk); SGLang's EAGLE-3 defaults for this model are steps=5, topk=4, 32 draft tokens
    "nospec": (0, 1),
    "chain": (5, 1),
    "tree": (5, 4),
}


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def report(engine: str, config: str, bs: int, n_prompts: int, out_tokens: int, seconds: float, accepted: float | None, **extra):
    row = dict(
        engine=engine, config=config, bs=bs, prompts=n_prompts, out_tokens=out_tokens,
        tok_s=round(out_tokens / seconds, 1),
        tpot_ms=round(1000 * seconds / (out_tokens / bs), 2) if bs else None,
        accepted_len=None if accepted is None else round(accepted, 2),
        gpu=torch_gpu_name(), commit=commit(), **extra,
    )
    print(json.dumps(row), flush=True)
    return row


def torch_gpu_name() -> str:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------- nanospec


def run_nanospec(config: str, batch_sizes: list[int], max_tokens: int, runs: int):
    import torch
    from transformers import AutoTokenizer

    from bench.prompts import chat_prompts
    from engine.engine import Engine, EngineConfig
    from model.loader import load_eagle3, load_model
    from sched.scheduler import SamplingParams

    depth, topk = CONFIGS[config]
    tok = AutoTokenizer.from_pretrained(TARGET)
    prompts = [tok(p, add_special_tokens=False).input_ids for p in chat_prompts(tok)]
    model = load_model(TARGET, "cuda")
    drafter = load_eagle3(EAGLE, model) if depth else None
    num_blocks = 64 * (2048 // 16)
    for bs in batch_sizes:
        for run in range(runs):
            eng = Engine(model, EngineConfig(num_blocks, 16, max_admit=bs, cuda_graphs=not depth, spec_depth=depth, spec_topk=topk), drafter)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            reqs = [eng.add(p, SamplingParams(max_tokens)) for p in prompts]
            while eng.sched.has_work:
                eng.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            n_out = sum(len(r.out_tokens) for r in reqs)
            steps = sum(len(r.accepted) for r in reqs)
            acc = (sum(sum(r.accepted) for r in reqs) / steps + 1) if steps else None
            report("nanospec", config, bs, len(prompts), n_out, dt, acc, run=run, torch=torch.__version__)
            del eng
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------- vLLM


def run_vllm(config: str, batch_sizes: list[int], max_tokens: int, runs: int):
    import vllm
    from vllm import LLM, SamplingParams

    from bench.prompts import chat_prompts

    depth, topk = CONFIGS[config]
    if topk > 1:
        raise SystemExit("vLLM's EAGLE-3 path is chain-only at this version; run --config chain")
    spec = {"method": "eagle3", "model": EAGLE, "num_speculative_tokens": depth} if depth else None
    for bs in batch_sizes:
        llm = LLM(model=TARGET, dtype="bfloat16", speculative_config=spec, max_num_seqs=bs, enforce_eager=False,
                  gpu_memory_utilization=0.85, disable_log_stats=False)
        prompts = chat_prompts(llm.get_tokenizer())
        sp = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=False)
        for run in range(runs):
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            dt = time.perf_counter() - t0
            n_out = sum(len(o.outputs[0].token_ids) for o in outs)
            acc = None
            if depth:
                try:  # per-request spec metrics are not exposed; use the aggregate if available
                    m = llm.get_metrics()
                    d = {x.name: x for x in m}
                    if "vllm:spec_decode_num_accepted_tokens" in d and "vllm:spec_decode_num_drafts" in d:
                        acc = d["vllm:spec_decode_num_accepted_tokens"].value / d["vllm:spec_decode_num_drafts"].value + 1
                except Exception:
                    pass
            report("vllm", config, bs, len(prompts), n_out, dt, acc, run=run, vllm=vllm.__version__)
        del llm


# ---------------------------------------------------------------------------- SGLang


def run_sglang(config: str, batch_sizes: list[int], max_tokens: int, runs: int):
    import sglang
    from bench.prompts import chat_prompts

    depth, topk = CONFIGS[config]
    for bs in batch_sizes:
        kw = dict(model_path=TARGET, dtype="bfloat16", max_running_requests=bs, mem_fraction_static=0.8, log_level="error")
        if depth:
            kw.update(speculative_algorithm="EAGLE3", speculative_draft_model_path=EAGLE, speculative_num_steps=depth,
                      speculative_eagle_topk=topk, speculative_num_draft_tokens=depth * topk + 1 if topk > 1 else depth + 1)
        llm = sglang.Engine(**kw)
        prompts = chat_prompts(llm.tokenizer_manager.tokenizer)
        for run in range(runs):
            t0 = time.perf_counter()
            outs = llm.generate(prompts, {"temperature": 0, "max_new_tokens": max_tokens})
            dt = time.perf_counter() - t0
            n_out = sum(o["meta_info"]["completion_tokens"] for o in outs)
            acc = None
            if depth:
                spec = [o["meta_info"].get("spec_verify_ct") for o in outs]
                if all(spec):
                    acc = n_out / sum(spec)
            report("sglang", config, bs, len(prompts), n_out, dt, acc, run=run, sglang=sglang.__version__)
        llm.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["nanospec", "vllm", "sglang"])
    ap.add_argument("--config", default="nospec", choices=list(CONFIGS))
    ap.add_argument("--bs", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    {"nanospec": run_nanospec, "vllm": run_vllm, "sglang": run_sglang}[a.engine](a.config, a.bs, a.max_tokens, a.runs)
