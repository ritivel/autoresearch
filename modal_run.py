"""
Run autoresearch on Modal cloud GPUs — no local GPU needed.

This is a thin wrapper that runs prepare.py and train.py on Modal's
serverless infrastructure. The existing local workflow is unchanged:
people with GPUs keep using `uv run train.py` directly.

Setup (one-time):
    pip install modal
    modal setup

Usage:
    modal run modal_run.py                      # prepare data + train (H100)
    modal run modal_run.py --gpu A100-80GB      # use a different GPU
    modal run modal_run.py --prepare-only       # just download data + tokenizer
    modal run modal_run.py --skip-prepare       # skip data prep (already cached)
    modal run modal_run.py --num-shards 5       # fewer shards for quick testing

Available GPUs: H100, A100-80GB, A100, L40S, A10G, L4, T4
Full list and pricing: https://modal.com/docs/guide/gpu
"""

import modal

DEFAULT_GPU = "H100"

app = modal.App("autoresearch")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "kernels>=0.11.7",
        "numpy>=2.2.6",
        "pyarrow>=21.0.0",
        "requests>=2.32.0",
        "rustbpe>=0.1.0",
        "tiktoken>=0.11.0",
    )
    .add_local_file("prepare.py", "/root/autoresearch/prepare.py")
    .add_local_file("train.py", "/root/autoresearch/train.py")
)

vol = modal.Volume.from_name("autoresearch-cache", create_if_missing=True)
CACHE_PATH = "/root/.cache/autoresearch"


@app.function(image=image, volumes={CACHE_PATH: vol}, timeout=1200)
def prepare(num_shards: int = 10):
    """Download data shards and train tokenizer (CPU-only, no GPU needed)."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "prepare.py", "--num-shards", str(num_shards)],
        cwd="/root/autoresearch",
    )
    vol.commit()
    if result.returncode != 0:
        raise RuntimeError(f"prepare.py failed (exit code {result.returncode})")


@app.cls(image=image, gpu=DEFAULT_GPU, volumes={CACHE_PATH: vol}, timeout=600)
class Trainer:
    """Runs train.py on a cloud GPU. Use Trainer.with_options(gpu=...) to change GPU type."""

    @modal.method()
    def run(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "train.py"],
            cwd="/root/autoresearch",
        )
        if result.returncode != 0:
            raise RuntimeError(f"train.py failed (exit code {result.returncode})")


@app.local_entrypoint()
def main(
    gpu: str = DEFAULT_GPU,
    prepare_only: bool = False,
    num_shards: int = 10,
    skip_prepare: bool = False,
):
    if not skip_prepare:
        print("=== Preparing data and tokenizer ===")
        prepare.remote(num_shards=num_shards)
        print("Data preparation complete.\n")

    if prepare_only:
        return

    print(f"=== Training on {gpu} ===")
    Trainer.with_options(gpu=gpu)().run.remote()
