"""
Profiling Script 04: Sub-component Breakdown of tokens_to_wav (Standalone MBD)
==============================================================================

Purpose:
--------
Times each sub-component of `MultiBandDiffusion.tokens_to_wav` when MBD is
loaded ALONE in GPU memory (without VALL-E):
1. `codec_model.decode(tokens)`: EnCodec neural decoder.
2. `get_emb(tokens)`: Acoustic embedding projection.
3. `DP[0].generate()`: Band 0 diffusion (0-3 kHz).
4. `DP[1].generate()`: Band 1 diffusion (3-6 kHz).
5. `DP[2].generate()`: Band 2 diffusion (6-9 kHz).
6. `DP[3].generate()`: Band 3 diffusion (9-12 kHz).
7. `re_eq()`: Multiband equalization and band summation.

Usage:
------
docker run --gpus all --rm \\
    -v "${PWD}:/app" \\
    -v "${PWD}/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints" \\
    -v "${PWD}/.cache/huggingface:/root/.cache/huggingface" \\
    -w /app hebtts:latest python profiling/04_mbd_tokens_to_wav.py
"""

import time
import utils
import torch
from engine.heb_tts.valle.data.tokenizer import AudioTokenizer


def run_benchmark():
    print("======================================================================")
    print("HebTTS Profiling: MBD tokens_to_wav Sub-component Breakdown")
    print("======================================================================")

    tok = AudioTokenizer(mbd=True, device="cuda:0")
    mbd = tok.mbd

    # Realistic tokens shape: 8 codebooks, 250 time steps (~3.33s audio)
    tokens = torch.randint(0, 1024, (1, 8, 250), device="cuda:0")

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        wav_encodec = mbd.codec_model.decode(tokens)
    torch.cuda.synchronize()
    t1 = time.time()

    with torch.no_grad():
        condition = mbd.get_emb(tokens)
    torch.cuda.synchronize()
    t2 = time.time()

    size = wav_encodec.size()
    out = torch.zeros(size, device=mbd.device)
    dp_times = []

    for i, DP in enumerate(mbd.DPs):
        t_dp_start = time.time()
        with torch.no_grad():
            out += DP.generate(condition=condition, initial_noise=torch.randn_like(out))
        torch.cuda.synchronize()
        t_dp_end = time.time()
        dp_times.append(t_dp_end - t_dp_start)

    t3 = time.time()
    with torch.no_grad():
        res = mbd.re_eq(wav=out, ref=wav_encodec, n_bands=32)
    torch.cuda.synchronize()
    t4 = time.time()

    print("\n--- Sub-component Timing Results ---")
    print(f"1. EnCodec decode:           {t1 - t0:.4f}s")
    print(f"2. Embedding projection:     {t2 - t1:.4f}s")
    for i, dur in enumerate(dp_times):
        print(f"3. DP[{i}] generate:           {dur:.4f}s")
    print(f"4. Multiband Re-EQ:          {t4 - t3:.4f}s")
    print(f"----------------------------------------")
    print(f"Total Standalone Decode:     {t4 - t0:.4f}s")


if __name__ == "__main__":
    run_benchmark()
