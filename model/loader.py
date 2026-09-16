"""HF checkpoint -> LlamaForCausalLM.

Resolves a local directory or a Hub repo id (downloading only config + safetensors),
parses ``config.json`` / ``generation_config.json`` into :class:`LlamaConfig`, builds
the model on the meta device and assigns the safetensors tensors straight in, so
weights are materialised exactly once, in the target dtype, on the target device.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from model.attention import default_backend
from model.llama import LlamaConfig, LlamaForCausalLM

_PATTERNS = ["config.json", "generation_config.json", "*.safetensors", "*.safetensors.index.json"]


def resolve_checkpoint(model: str | os.PathLike) -> Path:
    path = Path(model)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model), allow_patterns=_PATTERNS))


def load_config(path: str | os.PathLike) -> LlamaConfig:
    path = resolve_checkpoint(path)
    cfg = json.loads((path / "config.json").read_text())
    gen_path = path / "generation_config.json"
    gen = json.loads(gen_path.read_text()) if gen_path.exists() else None
    return LlamaConfig.from_hf(cfg, gen)


def load_model(
    model: str | os.PathLike,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    backend: str = "auto",  # "auto" | "sdpa" | "flashinfer"
) -> LlamaForCausalLM:
    path = resolve_checkpoint(model)
    config = load_config(path)
    device = torch.device(device)

    with torch.device("meta"):
        net = LlamaForCausalLM(config)

    # safetensors can only map straight onto cuda/cpu; everything else goes via cpu.
    open_device = str(device) if device.type in ("cuda", "cpu") else "cpu"
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device=open_device) as f:
            for name in f.keys():
                state[name] = f.get_tensor(name).to(device=device, dtype=dtype)

    tied = "lm_head.weight" not in state
    if tied and not config.tie_word_embeddings:
        raise KeyError("checkpoint has no lm_head.weight and config does not tie embeddings")
    if tied:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]

    net.load_state_dict(state, strict=True, assign=True)
    if tied:
        net.tie_weights()
    if backend == "auto":
        net.backend = default_backend(config, device, dtype)
    else:
        from model.attention import FlashInferBackend, SdpaBackend

        net.backend = (
            SdpaBackend()
            if backend == "sdpa"
            else FlashInferBackend(
                config.num_attention_heads, config.num_key_value_heads, config.head_dim, dtype, device
            )
        )
    return net.eval()


def load_eagle3(model: str | os.PathLike, target: LlamaForCausalLM, backend: str = "auto"):
    """EAGLE-3 head from a HF repo/dir holding config.json + pytorch_model.bin."""
    from model.attention import FlashInferBackend, SdpaBackend, default_backend
    from spec.drafter import Eagle3Config, Eagle3Head

    path = Path(model)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(str(model), allow_patterns=["config.json", "*.bin", "*.safetensors"]))
    cfg = Eagle3Config.from_hf(json.loads((path / "config.json").read_text()), target.config)
    p = target.lm_head.weight
    device, dtype = p.device, p.dtype

    bins = sorted(path.glob("*.safetensors"))
    if bins:
        state = {}
        for shard in bins:
            with safe_open(shard, framework="pt", device="cpu") as f:
                state.update({k: f.get_tensor(k) for k in f.keys()})
    else:
        state = torch.load(next(path.glob("*.bin")), map_location="cpu", weights_only=True)
    state.pop("t2d", None)
    state = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype) for k, v in state.items()}

    with torch.device("meta"):
        head = Eagle3Head(cfg)
    head.load_state_dict(state, strict=True, assign=True)
    head.embed_tokens = target.model.embed_tokens
    if backend == "auto":
        head.backend = default_backend(cfg, device, dtype)
    elif backend == "sdpa":
        head.backend = SdpaBackend()
    else:
        head.backend = FlashInferBackend(cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, dtype, device)
    return head.eval()
