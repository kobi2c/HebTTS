"""
Profiling Script 03: Raw GPU Compute Throughput of U-Net Forward Pass
=====================================================================

Purpose:
--------
Measures the raw GPU forward-pass latency of a single MultiBandDiffusion U-Net
model (286M parameters) on realistic audio shapes:
- Waveform size: 4 seconds @ 24 kHz = 96,000 samples.
- Conditioning size: 4 seconds @ 75 Hz = 300 frames, embedding dimension = 128.
- Step index: single integer step scalar.

Calculates the theoretical baseline GPU duration for all 80 diffusion steps
(4 frequency bands x 20 steps per band) under zero-overhead conditions.

Usage:
------
docker run --gpus all --rm \\
    -v "${PWD}:/app" \\
    -v "${PWD}/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints" \\
    -v "${PWD}/.cache/huggingface:/root/.cache/huggingface" \\
    -w /app hebtts:latest python profiling/03_unet_raw_compute.py
"""

import time
import utils
import torch
from engine.heb_tts.valle.data.tokenizer import AudioTokenizer


def run_benchmark():
    print("======================================================================")
    print("HebTTS Profiling: Single U-Net Forward Pass Raw GPU Compute")
    print("======================================================================")

    tok = AudioTokenizer(mbd=True, device="cuda:0")
    dp = tok.mbd.DPs[0]

    # Input dimensions for 4.0 seconds of audio
    x = torch.randn(1, 1, 96000, device="cuda:0")
    cond = torch.randn(1, 128, 300, device="cuda:0")

    # Warmup pass
    with torch.no_grad():
        _ = dp.model(x, 500, condition=cond)
    torch.cuda.synchronize()

    # Timed run: 10 forward passes
    iterations = 10
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = dp.model(x, 500, condition=cond)
    torch.cuda.synchronize()
    t1 = time.time()

    avg_pass = (t1 - t0) / iterations
    expected_80_passes = 80 * avg_pass

    print(f"Audio Length:                        4.0s (96,000 samples)")
    print(f"Conditioning Frames:                 300 frames (128-dim)")
    print(f"U-Net Parameters:                    286,000,000")
    print(f"Average time per single pass:        {avg_pass:.4f}s")
    print(f"Theoretical total for 80 passes:     {expected_80_passes:.2f}s (4 bands x 20 steps)")


if __name__ == "__main__":
    run_benchmark()
