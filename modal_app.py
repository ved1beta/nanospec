"""Modal entry points. One H100; the image is torch + flashinfer + transformers .

    modal run modal_app.py::gates                    # G1 + G2 on Llama-3.1-8B-Instruct, 32 x 128, vs HF
    modal run modal_app.py::gates --model NousResearch/Meta-Llama-3.1-8B-Instruct   # ungated mirror
    modal run modal_app.py::gates --extra "-k g2"

Needs a Modal secret named `huggingface` with HF_TOKEN for gated meta-llama repos.
Weights are cached in the `nanospec-hf-cache` volume across runs.
"""

import os
import subprocess

import modal

REPO = "/root/nanospec"
HF_CACHE = "/root/.cache/huggingface"

# CUDA devel base so flashinfer can JIT its kernels (needs nvcc); torch from the cu128 index.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("flashinfer-python")
    .pip_install("transformers>=4.55", "safetensors", "huggingface_hub[hf_transfer]", "pytest", "accelerate")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": HF_CACHE})
    .add_local_dir(".", remote_path=REPO, ignore=["**/.git", "**/__pycache__", "**/.venv", "**/*.pyc"])
)

app = modal.App("nanospec", image=image)
hf_cache = modal.Volume.from_name("nanospec-hf-cache", create_if_missing=True)


@app.function(
    gpu="H100",
    timeout=60 * 60,
    volumes={HF_CACHE: hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
)
def gates(model: str = "meta-llama/Llama-3.1-8B-Instruct", max_new: int = 128, extra: str = ""):
    """Run the correctness gates (G1, G2) on an H100."""
    env = {"NANOSPEC_MODEL": model, "NANOSPEC_DEVICE": "cuda", "NANOSPEC_MAX_NEW": str(max_new)}
    cmd = ["python", "-m", "pytest", "tests/", "-q", "-x", *extra.split()]
    rc = subprocess.call(cmd, cwd=REPO, env={**os.environ, **env})
    hf_cache.commit()
    if rc != 0:
        raise SystemExit(f"gates failed (pytest exit {rc})")
    print(f"gates green: {model}, 32 prompts x {max_new} tokens")
