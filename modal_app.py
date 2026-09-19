"""Modal entry points. One H100; the image is torch + flashinfer + transformers .

    modal run modal_app.py::test                    # full suite on Llama-3.1-8B-Instruct, 32 x 128, vs HF
    modal run modal_app.py::test --extra "-k fragmented"

Default model is the ungated mirror of Llama-3.1-8B-Instruct (identical weights). For the
gated meta-llama repo, export HF_TOKEN locally; it is passed through to the container.
Weights are cached in the `nanospec-hf-cache` volume across runs.
"""

import os
import shlex
import subprocess

import modal

REPO = "/root/nanospec"
HF_CACHE = "/root/.cache/huggingface"

# CUDA devel base so flashinfer can JIT its kernels (needs nvcc); torch from the cu128 index.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("flashinfer-python")
    .pip_install("transformers>=4.55", "safetensors", "huggingface_hub[hf_transfer]", "pytest", "accelerate", "datasets")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": HF_CACHE})
    .add_local_dir(".", remote_path=REPO, ignore=["**/.git", "**/__pycache__", "**/.venv", "**/*.pyc"])
)

app = modal.App("nanospec", image=image)
hf_cache = modal.Volume.from_name("nanospec-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("nanospec-runs", create_if_missing=True)  # rl/grpo.py logs, telemetry, args
RUNS = "/root/runs"


@app.function(
    gpu="H100",
    timeout=60 * 60,
    volumes={HF_CACHE: hf_cache},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def test(
    model: str = "NousResearch/Meta-Llama-3.1-8B-Instruct",
    eagle: str = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    max_new: int = 128,
    extra: str = "",
    backend: str = "auto",
):
    """Run the test suite on an H100. backend=sdpa is the strict bit-exact oracle."""
    env = {"NANOSPEC_MODEL": model, "NANOSPEC_EAGLE": eagle, "NANOSPEC_DEVICE": "cuda",
           "NANOSPEC_MAX_NEW": str(max_new), "NANOSPEC_BACKEND": backend}
    cmd = ["python", "-m", "pytest", "tests/", "-q", "-s", "-p", "no:warnings", *shlex.split(extra)]
    rc = subprocess.call(cmd, cwd=REPO, env={**os.environ, **env})
    hf_cache.commit()
    if rc != 0:
        raise SystemExit(f"tests failed (pytest exit {rc})")
    print(f"tests green: {model}, 32 prompts x {max_new} tokens")


@app.function(gpu="H100", timeout=30 * 60, volumes={HF_CACHE: hf_cache},
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def diag(model: str = "NousResearch/Meta-Llama-3.1-8B-Instruct", script: str = "tests/diag_numerics.py", args: str = ""):
    """Diagnostics: tests/diag_numerics.py (kernel noise) or tests/diag_spec.py (spec path)."""
    env = {**os.environ, "NANOSPEC_MODEL": model, "NANOSPEC_DEVICE": "cuda", "PYTHONPATH": REPO}
    subprocess.check_call(["python", script, *shlex.split(args)], cwd=REPO, env=env)
    hf_cache.commit()


# ---------------------------------------------------------------------------- benchmark (PLAN §5)

BENCH_ENV = {"NANOSPEC_MODEL": "NousResearch/Meta-Llama-3.1-8B-Instruct", "PYTHONPATH": REPO}
_secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})]


def _bench(engine: str, config: str, bs: str, max_tokens: int, runs: int) -> str:
    cmd = ["python", "-m", "bench.run", engine, "--config", config, "--bs", *bs.split(), "--max-tokens", str(max_tokens), "--runs", str(runs)]
    out = subprocess.run(cmd, cwd=REPO, env={**os.environ, **BENCH_ENV}, capture_output=True, text=True)
    rows = "\n".join(l for l in out.stdout.splitlines() if l.startswith("{"))
    print(rows or out.stdout[-4000:], out.stderr[-4000:] if out.returncode else "")
    hf_cache.commit()
    return rows


@app.function(gpu="H100", timeout=2 * 60 * 60, volumes={HF_CACHE: hf_cache}, secrets=_secrets)
def bench_nanospec(config: str = "nospec", bs: str = "1 8 32", max_tokens: int = 256, runs: int = 3) -> str:
    return _bench("nanospec", config, bs, max_tokens, runs)


vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "transformers", "huggingface_hub")
    .env({"HF_HOME": HF_CACHE, "VLLM_USE_V1": "1"})
    .add_local_dir(".", remote_path=REPO, ignore=["**/.git", "**/__pycache__", "**/.venv", "**/*.pyc"])
)


@app.function(gpu="H100", timeout=2 * 60 * 60, image=vllm_image, volumes={HF_CACHE: hf_cache}, secrets=_secrets)
def bench_vllm(config: str = "nospec", bs: str = "1 8 32", max_tokens: int = 256, runs: int = 3) -> str:
    return _bench("vllm", config, bs, max_tokens, runs)


sglang_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("sglang[all]", "transformers", "huggingface_hub")
    .env({"HF_HOME": HF_CACHE})
    .add_local_dir(".", remote_path=REPO, ignore=["**/.git", "**/__pycache__", "**/.venv", "**/*.pyc"])
)


@app.function(gpu="H100", timeout=2 * 60 * 60, image=sglang_image, volumes={HF_CACHE: hf_cache}, secrets=_secrets)
def bench_sglang(config: str = "nospec", bs: str = "1 8 32", max_tokens: int = 256, runs: int = 3) -> str:
    return _bench("sglang", config, bs, max_tokens, runs)


@app.local_entrypoint()
def bench(engines: str = "nanospec,vllm,sglang", configs: str = "nospec,chain,tree", bs: str = "1 8 32", runs: int = 3):
    """modal run modal_app.py::bench  -> bench/results/*.jsonl, then `python -m bench.table bench/results/*.jsonl`."""
    from pathlib import Path

    fns = {"nanospec": bench_nanospec, "vllm": bench_vllm, "sglang": bench_sglang}
    out = Path("bench/results")
    out.mkdir(exist_ok=True)
    for e in engines.split(","):
        for c in configs.split(","):
            if e == "vllm" and c == "tree":
                continue  # chain-only EAGLE-3 in vLLM; the table says so
            rows = fns[e].remote(config=c, bs=bs, runs=runs)
            (out / f"{e}-{c}.jsonl").write_text(rows + "\n")
            print(f"{e}/{c}: {rows.count(chr(10)) + 1} rows")


# ---------------------------------------------------------------------------- GRPO harness (rl/grpo.py)


@app.function(gpu="H100", timeout=6 * 60 * 60, volumes={HF_CACHE: hf_cache, RUNS: runs}, secrets=_secrets)
def grpo(name: str, args: str = "", model: str = "unsloth/Llama-3.2-1B-Instruct"):
    """modal run --detach modal_app.py::grpo --name e0-static --args "--task gsm8k --steps 300 --draft frozen ..."
    Logs land in the nanospec-runs volume under <name>/ (log.jsonl, telemetry.jsonl, args.json)."""
    cmd = ["python", "-m", "rl.grpo", "--model", model, "--out", f"{RUNS}/{name}", *shlex.split(args)]
    env = {**os.environ, "PYTHONPATH": REPO, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    print("$", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=REPO, env=env)
    runs.commit()
    hf_cache.commit()
    if rc != 0:
        raise SystemExit(f"rl.grpo exited {rc}")
