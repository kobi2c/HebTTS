"""
Profiling Script 08: End-to-End SampleMaker CLI Benchmark with MBD
==================================================================

Purpose:
--------
Benchmarks the full end-to-end execution of `SampleMaker` from the CLI
with Multi-Band Diffusion (`--mbd`) enabled, including:
1. Command line parsing.
2. Cold start model initialization (VALL-E + MBD).
3. Text normalization and Hebrew nikud conversion.
4. Reference speaker prompt encoding.
5. VALL-E autoregressive token generation.
6. Optimized MBD diffusion decode with VRAM offloading.
7. Saving output audio to disk.

Usage:
------
docker run --gpus all --rm \\
    -v "${PWD}:/app" \\
    -v "${PWD}/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints" \\
    -v "${PWD}/.cache/huggingface:/root/.cache/huggingface" \\
    -w /app hebtts:latest python profiling/08_sample_maker_e2e.py
"""

import time
import utils
from make_sample import SampleMaker


def run_benchmark():
    print("======================================================================")
    print("HebTTS Profiling: End-to-End SampleMaker CLI with MBD")
    print("======================================================================")

    t0 = time.time()
    sm = SampleMaker()
    args = [
        "--text", "שלום עולם, אנחנו בודקים את הביצועים של מערכת הדיבור.",
        "--speaker", "1",
        "--mbd",
        "--output", "/tmp/profile_sample_out.wav",
    ]
    sm.run(args)
    t1 = time.time()

    total_dur = t1 - t0
    print(f"\n======================================================================")
    print(f"Total End-to-End CLI Duration (with Cold Start Init): {total_dur:.2f}s")
    print(f"======================================================================")


if __name__ == "__main__":
    run_benchmark()
