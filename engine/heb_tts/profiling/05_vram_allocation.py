"""
Profiling Script 05: VRAM Memory Allocation & Capacity Exhaustion
================================================================

Purpose:
--------
Measures exact CUDA memory allocated and reserved under three conditions:
1. Baseline: Empty CUDA context.
2. VALL-E alone: After loading `checkpoint_01.pt` with MBD disabled.
3. Co-existence: After loading both VALL-E and Multi-Band Diffusion simultaneously.

Compares memory footprints against the physical 8,192 MB VRAM limit of the
RTX 5070 Laptop GPU to detect shared memory thrashing thresholds.

Usage:
------
docker run --gpus all --rm \\
    -v "${PWD}:/app" \\
    -v "${PWD}/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints" \\
    -v "${PWD}/.cache/huggingface:/root/.cache/huggingface" \\
    -w /app hebtts:latest python profiling/05_vram_allocation.py
"""

from config import utils
import torch
from GradioPage import GradioPage


def run_benchmark():
    print("======================================================================")
    print("HebTTS Profiling: VRAM Footprint & Capacity Exhaustion Inspection")
    print("======================================================================")

    # 1. Baseline
    torch.cuda.empty_cache()
    dev_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e6
    print(f"GPU Hardware: {dev_name}")
    print(f"Total Physical VRAM: {total_mem:.1f} MB")
    print(f"Initial Allocated:   {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    # 2. VALL-E Alone
    page_valle = GradioPage()
    page_valle.load_model_if_needed("checkpoint_01.pt", mbd=0)
    valle_alloc = torch.cuda.memory_allocated() / 1e6
    valle_res = torch.cuda.memory_reserved() / 1e6
    print(f"\n[VALL-E Alone] Allocated: {valle_alloc:.1f} MB | Reserved: {valle_res:.1f} MB")

    # Cleanup VALL-E alone
    del page_valle
    torch.cuda.empty_cache()

    # 3. VALL-E + MBD Loaded Together
    page_both = GradioPage()
    page_both.load_model_if_needed("checkpoint_01.pt", mbd=1)
    both_alloc = torch.cuda.memory_allocated() / 1e6
    both_res = torch.cuda.memory_reserved() / 1e6
    print(f"\n[VALL-E + MBD Together] Allocated: {both_alloc:.1f} MB | Reserved: {both_res:.1f} MB")
    print(f"Physical Headroom Remaining: {total_mem - both_res:.1f} MB (out of {total_mem:.1f} MB)")


if __name__ == "__main__":
    run_benchmark()
