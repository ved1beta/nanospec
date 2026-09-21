"""GRPO over nanospec rollouts, with the trainer-side correction for lossy speculation.

    python -m rl.grpo --model axolotl-ai-co/tiny-llama-50m --task toy --steps 20 --draft frozen \\
        --acceptance power --alpha 0.3 --correction exact --out runs/toy

One step: sample P prompts x G completions from the engine (T = 1, per-token target and
behavior log-probs), verifiable rewards, group-normalised advantages, then the token-level
PPO-clip surrogate with a per-token off-policy weight w_t = pi_old(a_t|s_t) / mu(a_t|s_t)
(spec/sampling.py::behavior_logprob), applied as one of:

    none     w = 1 (pretend the rollouts are on-policy)
    exact    w
    snis     w * N / sum(w): the same relative weighting with mean 1, so the gradient scale
             matches the uncorrected run (the control for "the weights just shrink the step")
    shuffle  the exact weights permuted at random across the batch's tokens: same mean, same
             ESS, no information about which token they belong to (the control for "the
             corrected run drifts less because of the weights' scale or variance")
    clipped  min(w, C)                      (truncated IS, Yao et al. 2025)
    icepop   1 where |log w| <= delta, else the token is masked  (Ring-1T / IcePop)
    m2po     mask the largest |log w| tokens until the batch's E[(w - 1)^2] <= delta, w on the rest

After the update the engine gets the new weights in place (Engine.update_weights); a frozen
ModelDrafter can be refreshed from the policy every K steps (rl/drafters.py). Every step
logs one JSON line (rewards, rollout throughput, accepted length, IS statistics, the
missing-mass estimate E[prod_t w_t], KL(mu || pi_old), KL(pi_old || pi_0) on the rollouts).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from engine.engine import Engine, EngineConfig
from model.loader import load_model
from rl.drafters import ModelDrafter, load_drafter, refresh
from rl.tasks import TASKS
from sched.scheduler import SamplingParams


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="axolotl-ai-co/tiny-llama-50m")
    ap.add_argument("--task", default="toy", choices=list(TASKS))
    ap.add_argument("--out", default="runs/dev")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--prompts", type=int, default=8, help="prompts per step")
    ap.add_argument("--group", type=int, default=4, help="completions per prompt (G)")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-prompt", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--micro", type=int, default=8, help="sequences per trainer forward")
    ap.add_argument("--opt-steps", type=int, default=1, help="optimizer steps per rollout batch (minibatches)")
    ap.add_argument("--beta", type=float, default=0.0, help="KL(pi || pi_0) penalty (k3); pi_0 kept only if > 0 or --kl-every")
    ap.add_argument("--kl-every", type=int, default=1, help="log KL(pi_old || pi_0) on the rollouts every k steps (0 = never)")
    ap.add_argument("--correction", default="none", choices=["none", "exact", "snis", "shuffle", "clipped", "icepop", "m2po"])
    ap.add_argument("--is-clip", type=float, default=2.0)
    ap.add_argument("--icepop-delta", type=float, default=0.5, help="|log w| beyond which a token is masked")
    ap.add_argument("--m2po-delta", type=float, default=0.04, help="bound on E[(w-1)^2] over the kept tokens")
    # rollout engine
    ap.add_argument("--draft", default="none", help="none | frozen | model:<id> | eagle:<id>")
    ap.add_argument("--draft-refresh", type=int, default=0, help="copy the policy into a frozen draft every K steps")
    ap.add_argument("--spec-depth", type=int, default=5)
    ap.add_argument("--spec-topk", type=int, default=1)
    ap.add_argument("--acceptance", default="exact", choices=["exact", "relaxed", "power"])
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--accept-topk", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-running", type=int, default=512)
    ap.add_argument("--kv-blocks", type=int, default=0, help="0 = enough for max-running sequences at full length")
    ap.add_argument("--cuda-graphs", action="store_true")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0, help="save the policy (bf16 state dict) every k steps to <out>/policy_<step>.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fp32", action="store_true", help="train in fp32 (default: bf16 params with fp32 Adam via autocast)")
    return ap.parse_args(argv)


def device_of(args) -> torch.device:
    if args.device:
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------- weights


def token_weights(log_w: torch.Tensor, mask: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """log_w [N, L] = log pi_old - log mu per rollout token -> (multiplier, mask, stats)."""
    w = log_w.exp() * mask
    n = mask.sum().clamp_min(1)
    stats = {"w_mean": float((w.sum() / n)), "w_max": float(w.max()), "logw_min": float(log_w[mask > 0].min()) if n else 0.0,
             "ess": float(w.sum() ** 2 / (w.pow(2).sum().clamp_min(1e-12) * n)), "kl_mu_pi": float(-(log_w * mask).sum() / n)}
    keep = mask.clone()
    if args.correction == "none":
        w = mask.clone()
    elif args.correction == "snis":
        w = w * n / w.sum().clamp_min(1e-12)
    elif args.correction == "shuffle":
        live = mask.flatten().nonzero().squeeze(1)
        flat = w.flatten().clone()
        flat[live] = flat[live[torch.randperm(len(live))]]
        w = flat.view_as(w)
    elif args.correction == "clipped":
        w = w.clamp_max(args.is_clip)
    elif args.correction == "icepop":
        keep = mask * (log_w.abs() <= args.icepop_delta)
        w = keep.clone()
    elif args.correction == "m2po":
        dev = (w - 1).pow(2) * mask
        order = torch.argsort((log_w.abs() * mask).flatten(), descending=True)  # drop the most off-policy tokens first
        flat_keep = mask.flatten().clone()
        for i in order.tolist():
            if (dev.flatten() * flat_keep).sum() / flat_keep.sum().clamp_min(1) <= args.m2po_delta:
                break
            flat_keep[i] = 0
        keep = flat_keep.view_as(mask)
        w = w * keep
    stats.update(clip_frac=float(((log_w.exp() > args.is_clip) * mask).sum() / n) if args.correction == "clipped" else 0.0,
                 mask_frac=float(1 - keep.sum() / n))
    return w, keep, stats


# ---------------------------------------------------------------------------- trainer


class Trainer:
    def __init__(self, args, tok, device):
        from transformers import AutoModelForCausalLM

        dtype = torch.float32 if args.fp32 or device.type != "cuda" else torch.bfloat16
        self.args, self.device, self.dtype = args, device, dtype
        self.policy = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
        self.policy.gradient_checkpointing_enable() if device.type == "cuda" else None
        self.ref = None
        if args.beta > 0 or args.kl_every:
            self.ref = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval().requires_grad_(False)
        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def _logps(self, model, ids: torch.Tensor, attn: torch.Tensor, n_prompt: list[int]) -> torch.Tensor:
        """log pi(a_t | s_t) for every position; [N, L-1], row i valid from n_prompt[i]-1 on."""
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
        V = logits.shape[-1]  # cross_entropy upcasts internally without keeping a second full-vocab fp32 copy
        return -F.cross_entropy(logits.reshape(-1, V), ids[:, 1:].reshape(-1), reduction="none").view(ids.shape[0], -1)

    def batch(self, seqs: list[list[int]], n_prompt: list[int]):
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), self.pad, dtype=torch.int64)
        attn = torch.zeros(len(seqs), L, dtype=torch.int64)
        for i, s in enumerate(seqs):
            ids[i, : len(s)], attn[i, : len(s)] = torch.tensor(s), 1
        mask = torch.zeros(len(seqs), L - 1)
        for i, (s, n) in enumerate(zip(seqs, n_prompt)):
            mask[i, n - 1 : len(s) - 1] = 1  # output tokens are predicted from position n-1 on
        return ids.to(self.device), attn.to(self.device), mask.to(self.device)

    def step(self, seqs, n_prompt, old_lp, log_w, adv) -> dict:
        """One rollout batch: opt_steps minibatches, each accumulated over micro-batches.
        seqs: prompt + output ids; old_lp / log_w: [N, L-1] aligned with the output mask."""
        a = self.args
        N = len(seqs)
        per = math.ceil(N / a.opt_steps)
        tot: dict[str, float] = {}
        for mb in range(0, N, per):
            self.opt.zero_grad(set_to_none=True)
            idx = list(range(mb, min(mb + per, N)))
            ids, attn, mask = self.batch([seqs[i] for i in idx], [n_prompt[i] for i in idx])
            lw, olp = log_w[idx, : mask.shape[1]].to(self.device), old_lp[idx, : mask.shape[1]].to(self.device)
            w, keep, st = token_weights(lw, mask, a)
            denom = keep.sum().clamp_min(1)
            for s in range(0, len(idx), a.micro):
                sl = slice(s, s + a.micro)
                lp = self._logps(self.policy, ids[sl], attn[sl], [n_prompt[i] for i in idx[sl]])
                ratio = (lp - olp[sl]).exp()
                A = adv[idx[sl]].to(self.device)[:, None]
                surr = torch.minimum(ratio * A, ratio.clamp(1 - a.clip_eps, 1 + a.clip_eps) * A)
                loss = -(w[sl] * keep[sl] * surr).sum() / denom
                if self.ref is not None and a.beta > 0:
                    with torch.no_grad():
                        rlp = self._logps(self.ref, ids[sl], attn[sl], [n_prompt[i] for i in idx[sl]])
                    d = rlp - lp
                    loss = loss + a.beta * ((d.exp() - d - 1) * keep[sl]).sum() / denom  # k3 estimator
                loss.backward()
                tot["loss"] = tot.get("loss", 0.0) + loss.detach().item()
                tot["ratio_clipped"] = tot.get("ratio_clipped", 0.0) + float((((ratio - 1).abs() > a.clip_eps) * keep[sl]).sum())
            gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()
            tot["grad_norm"] = max(tot.get("grad_norm", 0.0), float(gn))
            for k, v in st.items():
                tot[k] = tot.get(k, 0.0) + v / a.opt_steps
        tot["ratio_clipped"] /= max(sum(len(s) - n for s, n in zip(seqs, n_prompt)), 1)
        return tot

    @torch.no_grad()
    def kl_to_ref(self, seqs, n_prompt, old_lp) -> float:
        """KL(pi_old || pi_0) on the rollout tokens (k1: mean log pi_old - log pi_0)."""
        tot, n = 0.0, 0
        for s in range(0, len(seqs), self.args.micro):
            ids, attn, mask = self.batch(seqs[s : s + self.args.micro], n_prompt[s : s + self.args.micro])
            rlp = self._logps(self.ref, ids, attn, n_prompt[s : s + self.args.micro])
            tot += float(((old_lp[s : s + self.args.micro, : mask.shape[1]].to(self.device) - rlp) * mask).sum())
            n += int(mask.sum())
        return tot / max(n, 1)

    def state_bf16(self, keys) -> dict[str, torch.Tensor]:
        return {k: v.detach().to(torch.bfloat16) for k, v in self.policy.state_dict().items() if k in keys}


# ---------------------------------------------------------------------------- rollouts


def rollout(engine: Engine, prompts, params: SamplingParams, step: int):
    """-> requests (with logprobs) in prompt-major order, and timing stats."""
    t0 = time.perf_counter()
    reqs = [engine.add(ids, dataclasses.replace(params, seed=step * 1_000_003 + i)) for i, ids in enumerate(prompts)]
    while engine.sched.has_work:
        engine.step()
    if engine.device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    toks = sum(len(r.out_tokens) for r in reqs)
    steps = sum(len(r.accepted) for r in reqs)
    acc = sum(sum(r.accepted) for r in reqs) / steps if steps else 0.0
    return reqs, {"rollout_s": dt, "tok_s": toks / dt, "out_tokens": toks, "accepted": acc, "mean_len": toks / len(reqs)}


def main(argv=None) -> None:
    args = parse(argv)
    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = device_of(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    tok = AutoTokenizer.from_pretrained(args.model)
    task = TASKS[args.task](tok, args.max_prompt)

    target = load_model(args.model, device=device, dtype=torch.bfloat16, backend=args.backend)
    drafter = load_drafter(args.draft, target, args.model, device, args.backend)
    n_seq = args.prompts * args.group
    blocks = args.kv_blocks or (min(n_seq, args.max_running) * math.ceil((args.max_prompt + args.max_tokens + args.spec_depth * args.spec_topk + 2) / 16) + 2)
    cfg = EngineConfig(blocks, 16, max_admit=n_seq, max_running=args.max_running, cuda_graphs=args.cuda_graphs,
                       spec_depth=args.spec_depth if drafter else 0, spec_topk=args.spec_topk if drafter else 1,
                       telemetry_path=str(out / "telemetry.jsonl") if drafter else None)
    engine = Engine(target, cfg, drafter)
    keys = set(dict(target.named_parameters()) | dict(target.named_buffers()))
    params = SamplingParams(args.max_tokens, args.temperature, acceptance=args.acceptance, tau=args.tau,
                            accept_topk=args.accept_topk, alpha=args.alpha, logprobs=True)
    trainer = Trainer(args, tok, device)
    engine.update_weights(trainer.state_bf16(keys))  # the engine serves exactly the trainer's weights from step 0
    log = open(out / "log.jsonl", "a")
    seqlog = open(out / "seqs.jsonl", "a")
    print(f"[grpo] {args.task} on {args.model} | {args.prompts}x{args.group} x {args.max_tokens} tok | draft={args.draft} "
          f"depth={cfg.spec_depth} topk={cfg.spec_topk} acceptance={args.acceptance} tau={args.tau} topk={args.accept_topk} alpha={args.alpha} "
          f"| correction={args.correction} | {device}", flush=True)

    for step in range(args.steps):
        t_step = time.perf_counter()
        batch = task.sample(args.prompts, rng)
        prompts = [ids for ids, _ in batch for _ in range(args.group)]
        reqs, rs = rollout(engine, prompts, params, step)
        texts = [tok.decode(r.out_tokens, skip_special_tokens=True) for r in reqs]
        rewards = torch.tensor([task.reward(t, batch[i // args.group][1]) for i, t in enumerate(texts)])
        R = rewards.view(args.prompts, args.group)
        adv = ((R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-4)).flatten()

        seqs = [r.prompt_ids + r.out_tokens for r in reqs]
        n_prompt = [len(r.prompt_ids) for r in reqs]
        L = max(map(len, seqs)) - 1
        old_lp, log_w = torch.zeros(len(seqs), L), torch.zeros(len(seqs), L)
        for i, r in enumerate(reqs):
            n = n_prompt[i]
            old_lp[i, n - 1 : n - 1 + len(r.logprobs)] = torch.tensor([t.logprob for t in r.logprobs])
            log_w[i, n - 1 : n - 1 + len(r.logprobs)] = torch.tensor([t.sample_logprob - t.behavior_logprob for t in r.logprobs])
        # E[prod_t w_t] over the first 8 output tokens: 1 under full support, the reachable share of the
        # target's mass under the deterministic relaxed rule (the full-length product is too heavy-tailed to read)
        mass = float(torch.stack([log_w[i, n_prompt[i] - 1 : n_prompt[i] + 7].sum().exp() for i in range(len(reqs))]).mean())

        # per-sequence record: reward, output length, sum of log w (the sequence ratio pi_old / mu), accepted / step
        seqlog.write(json.dumps({"step": step, "reward": rewards.tolist(), "len": [len(r.out_tokens) for r in reqs],
                                 "logw": [float(log_w[i, n_prompt[i] - 1 : n_prompt[i] - 1 + len(r.logprobs)].sum()) for i, r in enumerate(reqs)],
                                 "acc": [sum(r.accepted) / max(len(r.accepted), 1) for r in reqs]}) + "\n")
        rec = {"step": step, "reward": float(rewards.mean()), "reward_std": float(rewards.std()), "frac_solved": float((rewards > 0.999).float().mean()),
               "groups_with_signal": float((R.std(1) > 0).float().mean()), "mass": mass, **rs}
        if trainer.ref is not None and args.kl_every and step % args.kl_every == 0:
            rec["kl_pi_pi0"] = trainer.kl_to_ref(seqs, n_prompt, old_lp)
        t_train = time.perf_counter()
        free_cache = torch.cuda.empty_cache if device.type == "cuda" else (lambda: None)
        free_cache()  # the engine's and the trainer's peak allocations do not overlap in time; do not let them fragment
        if float((R.std(1) > 0).float().sum()) > 0:
            rec.update(trainer.step(seqs, n_prompt, old_lp, log_w, adv))
        rec["train_s"] = time.perf_counter() - t_train
        t_sync = time.perf_counter()
        free_cache()
        engine.update_weights(trainer.state_bf16(keys))
        if args.draft_refresh and isinstance(drafter, ModelDrafter) and (step + 1) % args.draft_refresh == 0:
            refresh(engine, drafter, trainer.state_bf16(keys))
            rec["draft_refreshed"] = True
        rec["sync_s"] = time.perf_counter() - t_sync
        rec["step_s"] = time.perf_counter() - t_step
        if args.eval_every and (step + 1) % args.eval_every == 0:
            rec["eval_acc"] = evaluate(engine, task, args)
        if args.save_every and (step + 1) % args.save_every == 0:
            torch.save({k: v.cpu() for k, v in trainer.state_bf16(keys).items()}, out / f"policy_{step + 1}.pt")
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"[step {step:4d}] reward {rec['reward']:.3f} solved {rec['frac_solved']:.2f} | {rs['tok_s']:.0f} tok/s acc {rs['accepted']:.2f} "
              f"len {rs['mean_len']:.0f} | mass {mass:.3f} ess {rec.get('ess', 1.0):.2f} kl(mu|pi) {rec.get('kl_mu_pi', 0.0):.4f} "
              f"kl(pi|pi0) {rec.get('kl_pi_pi0', float('nan')):.4f} | mask {rec.get('mask_frac', 0.0):.2f} loss {rec.get('loss', 0.0):+.4f} "
              f"| {rec['step_s']:.1f}s", flush=True)
    (out / "sample.txt").write_text("\n---\n".join(t for t in texts[:8]))
    log.close()
    seqlog.close()


@torch.no_grad()
def evaluate(engine: Engine, task, args) -> float:
    prompts = [task.encode(q) for q, _ in task.test]
    params = SamplingParams(args.max_tokens, 0.0)
    reqs = [engine.add(p, params) for p in prompts]
    while engine.sched.has_work:
        engine.step()
    texts = [task.tok.decode(r.out_tokens, skip_special_tokens=True) for r in reqs]
    return sum(task.reward(t, a) for t, (_, a) in zip(texts, task.test)) / len(reqs)


if __name__ == "__main__":
    main()
