"""
Profiling Script 07: Checkpoint Disk Loading & Deserialization Benchmark
========================================================================

Purpose:
--------
Measures the disk read and unpickling duration of the 4.58 GB MultiBandDiffusion
checkpoint (`mbd_comp_8.pt`), comparing:
1. Standard `torch.load(..., weights_only=False)`
2. Memory-mapped `torch.load(..., weights_only=False, mmap=True)`

Establishes the physical hardware lower bound for model initialization.

Usage:
------
docker run --gpus all --rm \\
    -v "${PWD}:/app" \\
    -v "${PWD}/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints" \\
    -v "${PWD}/.cache/huggingface:/root/.cache/huggingface" \\
    -w /app hebtts:latest python profiling/07_checkpoint_loading.py
"""

import glob
import os
import time
from config import utils
import torch


def run_benchmark():
    print("======================================================================")
    print("HebTTS Profiling: Checkpoint Disk Read & Deserialization Benchmark")
    print("======================================================================")

    # Locate the MBD blob in huggingface cache
    pattern = "/root/.cache/huggingface/hub/models--facebook--multiband-diffusion/blobs/*"
    files = glob.glob(pattern)

    if not files:
        # Fallback to local Windows path if running outside Docker
        pattern = ".cache/huggingface/hub/models--facebook--multiband-diffusion/blobs/*"
        files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"Could not locate MBD blob file in huggingface cache.")

    blob_path = files[0]
    size_gb = os.path.getsize(blob_path) / 1e9
    print(f"Target File: {blob_path}")
    print(f"File Size:   {size_gb:.2f} GB ({os.path.getsize(blob_path):,} bytes)\n")

    # 1. Standard torch.load
    print("Testing standard torch.load...")
    t0 = time.time()
    d1 = torch.load(blob_path, weights_only=False)
    t1 = time.time()
    dur_standard = t1 - t0
    print(f"Standard torch.load:  {dur_standard:.2f}s ({size_gb / dur_standard:.2f} GB/s)")

    # 2. torch.load with mmap=True
    print("\nTesting torch.load with mmap=True...")
    t2 = time.time()
    d2 = torch.load(blob_path, weights_only=False, mmap=True)
    t3 = time.time()
    dur_mmap = t3 - t2
    print(f"mmap=True torch.load: {dur_mmap:.2f}s ({size_gb / dur_mmap:.2f} GB/s)")


if __name__ == "__main__":
    run_benchmark()
