"""
Profiling Script 02: MBD Device Assignment & Buffer Inspection
==============================================================

Purpose:
--------
Verifies whether any parameters, submodules, or buffers within the
MultiBandDiffusion instance remain on the CPU after initialization on CUDA.

Checks:
1. Top-level `mbd.device` attribute.
2. Device of the underlying `codec_model` (EnCodec).
3. Parameter devices across all 4 `DiffusionProcess` (DP) models.
4. Tensor buffer devices (`schedule.betas`) for each frequency band.
5. Counts any lingering CPU parameters or CPU buffers.

Usage:
------
docker run --gpus all --rm \\
    -v "${PWD}:/app" \\
    -v "${PWD}/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints" \\
    -v "${PWD}/.cache/huggingface:/root/.cache/huggingface" \\
    -w /app hebtts:latest python profiling/02_mbd_device_check.py
"""

import utils
import torch
from valle.data.tokenizer import AudioTokenizer


def run_benchmark():
    print("======================================================================")
    print("HebTTS Profiling: MBD Parameter & Buffer Device Inspection")
    print("======================================================================")

    tok = AudioTokenizer(mbd=True, device="cuda:0")
    mbd = tok.mbd

    print(f"Top-level mbd.device: {getattr(mbd, 'device', None)}")
    print(f"codec_model device:   {next(mbd.codec_model.parameters()).device}")

    for i, dp in enumerate(mbd.DPs):
        model_dev = next(dp.model.parameters()).device
        beta_dev = dp.schedule.betas.device if hasattr(dp.schedule, "betas") else "no betas"
        print(f"DP[{i}] model device: {model_dev} | schedule.betas device: {beta_dev}")

    cpu_params = [name for name, p in mbd.codec_model.named_parameters() if p.device.type == "cpu"]
    print(f"\ncodec_model CPU parameters: {len(cpu_params)}")

    for i, dp in enumerate(mbd.DPs):
        cpu_p = [name for name, p in dp.model.named_parameters() if p.device.type == "cpu"]
        cpu_b = [name for name, b in dp.model.named_buffers() if b.device.type == "cpu"]
        print(f"DP[{i}] CPU params: {len(cpu_p)}, CPU buffers: {len(cpu_b)}")


if __name__ == "__main__":
    run_benchmark()
