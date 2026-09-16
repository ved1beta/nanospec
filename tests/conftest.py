"""Shared fixtures: model/device selection, HF reference, nanospec model.

    NANOSPEC_MODEL=meta-llama/Llama-3.1-8B-Instruct NANOSPEC_DEVICE=cuda pytest tests/
"""

from __future__ import annotations

import os

import pytest
import torch

from model.loader import load_model
from tests.prompts import G1_PROMPTS

MODEL = os.environ.get("NANOSPEC_MODEL", "axolotl-ai-co/tiny-llama-50m")
MAX_NEW_TOKENS = int(os.environ.get("NANOSPEC_MAX_NEW", "128"))
DTYPE = torch.bfloat16

# bf16 has an 8-bit significand: one ulp is 2^-7 of the value. Different attention
# kernels (FlashInfer vs SDPA) reorder accumulations, so logits wobble by a few ulps of
# the *row's* scale -- not of each element, which for near-zero logits would be nothing.
# Measured on the H100 (tests/diag_numerics.py): SDPA is bit-exact vs HF; FlashInfer is
# ~1 ulp median, <2 ulps max per step, compounding to ~7 over 128 steps through the cache.
BF16_ULP = 2.0**-7
LOGIT_TOL_ULPS = 8
BACKEND = os.environ.get("NANOSPEC_BACKEND", "auto")  # auto | sdpa | flashinfer


def logit_tol(ref: torch.Tensor, ulps: float = LOGIT_TOL_ULPS) -> torch.Tensor:
    """Per-row tolerance, broadcastable against ref: [..., V]."""
    return ulps * BF16_ULP * ref.float().abs().amax(-1, keepdim=True)


def first_divergence(ours: list[int], ref: list[int]) -> int | None:
    for k, (a, b) in enumerate(zip(ours, ref)):
        if a != b:
            return k
    return None if len(ours) == len(ref) else min(len(ours), len(ref))


def assert_tokens_match(ours, ref, ref_logits, strict: bool, where: str) -> tuple[int, float] | None:
    """Token-for-token equality. With strict=False a divergence is tolerated only if the
    reference's logit for our token is within the kernel-noise tolerance of its best
    logit at that step; returns (step, gap) then."""
    k = first_divergence(ours, ref)
    if k is None:
        return None
    if strict or k >= len(ref_logits) or k >= len(ours):
        raise AssertionError(f"{where}: diverged at step {k}\nours={ours}\nref ={ref}")
    row = ref_logits[k].float()
    gap = (row.max() - row[ours[k]]).item()
    if gap > logit_tol(row).item():
        raise AssertionError(f"{where}: diverged at step {k} with a clear gap {gap:.4f} to the reference's choice\nours={ours}\nref ={ref}")
    return k, gap


def assert_logits_close(ours: torch.Tensor, ref: torch.Tensor, where: str, strict: bool = True) -> None:
    """strict: every logit within LOGIT_TOL_ULPS of its row max. Otherwise (a different
    attention kernel) the per-position worst case is judged by distribution -- median
    <= 2 ulps, p90 <= LOGIT_TOL_ULPS -- since degenerate repetitive contexts amplify
    kernel noise at a few positions without changing the argmax."""
    ours, ref = ours.float(), ref.float()
    diff = (ours - ref).abs()
    if not strict:
        u = (diff / (BF16_ULP * ref.abs().amax(-1, keepdim=True))).amax(-1)
        med, p90, mx = u.median().item(), u.quantile(0.9).item(), u.max().item()
        assert med <= 2 and p90 <= LOGIT_TOL_ULPS, f"{where}: ulps-of-rowmax median {med:.2f} p90 {p90:.2f} max {mx:.2f}"
        return
    bad = diff > logit_tol(ref)
    if bad.any():
        pos = tuple(bad.nonzero()[0].tolist())
        ulps = (diff / (BF16_ULP * ref.abs().amax(-1, keepdim=True))).max().item()
        raise AssertionError(
            f"{where}: {int(bad.sum())} logits outside tolerance; worst |diff|={diff.max().item():.4f} "
            f"= {ulps:.1f} bf16 ulps of the row max; first at {pos}: ours={ours[pos].item():.4f} ref={ref[pos].item():.4f}"
        )


def _device() -> torch.device:
    if d := os.environ.get("NANOSPEC_DEVICE"):
        return torch.device(d)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _device()


@pytest.fixture(scope="session")
def hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, attn_implementation="sdpa").to(DEVICE).eval()
    return tok, model


@pytest.fixture(scope="session")
def ns():
    return load_model(MODEL, device=DEVICE, dtype=DTYPE, backend=BACKEND)


@pytest.fixture(scope="session")
def strict(ns) -> bool:
    """SDPA is bit-exact vs HF; other backends get the tie allowance."""
    from model.attention import SdpaBackend

    return isinstance(ns.backend, SdpaBackend)


@pytest.fixture(scope="session")
def encoded(hf):
    tok, _ = hf
    return [tok(p, return_tensors="pt").input_ids[0].to(DEVICE) for p in G1_PROMPTS]


@pytest.fixture(scope="session")
def hf_ref(hf, encoded):
    """HF greedy tokens + per-step logits for every prompt, computed once."""
    _, model = hf
    eos = model.config.eos_token_id
    out = []
    for ids in encoded:
        r = model.generate(
            ids[None],
            attention_mask=torch.ones_like(ids[None]),
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=eos if isinstance(eos, int) else eos[0],
        )
        out.append((r.sequences[0, ids.shape[0] :].tolist(), torch.stack([l[0] for l in r.logits])))
    return out
