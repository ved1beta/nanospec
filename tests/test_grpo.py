"""The GRPO harness end to end on the tiny model: two steps of the toy task with a frozen
draft and the power rule, every correction mode's weights, and the log's contents."""

from __future__ import annotations

import json

import pytest
import torch

from rl.grpo import main, token_weights


class _A:
    correction, is_clip, icepop_delta, m2po_delta = "none", 2.0, 0.5, 0.04


@pytest.mark.parametrize("mode", ["none", "exact", "snis", "shuffle", "clipped", "icepop", "m2po"])
def test_token_weights(mode):
    a = _A()
    a.correction = mode
    log_w = torch.tensor([[0.0, -0.3, 1.5, 0.0], [0.2, -2.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1, 1, 0], [1, 1, 1, 0]])
    w, keep, st = token_weights(log_w, mask, a)
    assert (w[mask == 0] == 0).all() and (keep <= mask).all() and 0 < st["ess"] <= 1
    if mode == "none":
        assert torch.equal(w, mask)
    elif mode == "exact":
        assert torch.allclose(w, log_w.exp() * mask)
    elif mode == "snis":
        assert w.sum() == pytest.approx(mask.sum()) and torch.allclose(w / (log_w.exp() * mask).clamp_min(1e-9) * mask, (w > 0) * w.sum() / (log_w.exp() * mask).sum())
    elif mode == "shuffle":
        assert sorted(w[mask > 0].tolist()) == sorted((log_w.exp() * mask)[mask > 0].tolist())
    elif mode == "clipped":
        assert w.max() <= 2.0 and st["clip_frac"] == pytest.approx(1 / 6)
    elif mode == "icepop":
        assert keep.sum() == 4 and torch.equal(w, keep) and st["mask_frac"] == pytest.approx(2 / 6)
    else:
        dev = ((w - 1) ** 2 * keep).sum() / keep.sum()
        assert dev <= a.m2po_delta and 0 < st["mask_frac"] < 1


def test_harness_runs(tmp_path):
    out = tmp_path / "run"
    main(["--model", "axolotl-ai-co/tiny-llama-50m", "--task", "toy", "--steps", "2", "--prompts", "4", "--group", "2",
          "--max-tokens", "12", "--lr", "1e-5", "--draft", "frozen", "--draft-refresh", "2", "--acceptance", "power", "--alpha", "0.3",
          "--correction", "exact", "--out", str(out)])
    rows = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[1].get("draft_refreshed")
    for r in rows:
        assert {"reward", "tok_s", "accepted", "mass", "kl_pi_pi0", "step_s"} <= set(r)
        assert 0 <= r["reward"] <= 1 and r["accepted"] >= 0 and r["kl_pi_pi0"] >= -0.05
    assert (out / "telemetry.jsonl").exists() and (out / "args.json").exists()
    tele = [json.loads(l) for l in (out / "telemetry.jsonl").read_text().splitlines()]
    assert all("emitted" in t and all(e["mu"] <= 1e-9 for e in t["emitted"]) for t in tele)
