"""
Profiling Script 01: Pipeline Stages & Cached Chunks Benchmark
=============================================================

Purpose:
--------
Measures the end-to-end latency of each stage in the HebTTS pipeline when
Multi-Band Diffusion (MBD) is enabled:
1. GradioPage() instance initialization.
2. Model loading (VALL-E checkpoint + MultiBandDiffusion weights).
3. Text tokenization and reference speaker prompt encoding.
4. VALL-E autoregressive acoustic token generation (model.inference).
5. MBD diffusion vocoder waveform generation (audio_tokenizer.mbd_decode).
6. Performance of a second (cached) chunk to measure warm execution latency.

Usage:
------
docker run --gpus all --rm \\
    -v "${PWD}:/app" \\
    -v "${PWD}/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints" \\
    -v "${PWD}/.cache/huggingface:/root/.cache/huggingface" \\
    -w /app hebtts:latest python profiling/01_pipeline_stages.py
"""

import time
import utils
import torch
from make_voice import VoiceMaker
from config import Config
from valle.data import tokenize_audio


def run_benchmark():
    print("======================================================================")
    print("HebTTS Profiling: Pipeline Stages & Cached Chunks Benchmark")
    print("======================================================================")

    t0 = time.time()
    vm = VoiceMaker()
    t1 = time.time()

    cfg = Config().reset()
    cfg.mbd = 1
    cfg.speaker = "1"

    t_load_start = time.time()
    model, audio_tokenizer, text_collater, alef_bert_tokenizer = vm.page.load_model_if_needed(
        cfg.checkpoint, mbd=cfg.mbd
    )
    t_load_end = time.time()

    text = "היום אנחנו הולכים ללמוד על העולם הקסום של כללי ההתחלקות."
    prompt_audio, prompt_text = vm.resolve_speaker_prompt(cfg)
    encoded_frames_prompt = tokenize_audio(audio_tokenizer, prompt_audio)
    audio_prompts = torch.concat([encoded_frames_prompt[0][0]], dim=-1).transpose(2, 1).to(vm.page.device)

    text_without_space = [f"{prompt_text} {text}".strip().replace(" ", "_")]
    tokens = alef_bert_tokenizer._tokenize(text_without_space)
    prompt_text_without_space = [f"{prompt_text}".strip().replace(" ", "_")]
    prompt_tokens = alef_bert_tokenizer._tokenize(prompt_text_without_space)
    tokens = [t for t in tokens if t in text_collater.token2idx]
    prompt_tokens = [t for t in prompt_tokens if t in text_collater.token2idx]
    text_tokens, text_tokens_lens = text_collater([tokens])
    _, enroll_x_lens = text_collater([prompt_tokens])

    # --- Chunk 1 (Cold Start Synthesis) ---
    t_inf_start = time.time()
    with torch.no_grad():
        encoded_frames = model.inference(
            text_tokens.to(vm.page.device),
            text_tokens_lens.to(vm.page.device),
            audio_prompts,
            enroll_x_lens=enroll_x_lens,
            top_k=50,
            temperature=0.95,
        )
    t_inf_end = time.time()

    t_mbd_start = time.time()
    with torch.no_grad():
        samples = audio_tokenizer.mbd_decode(encoded_frames.transpose(2, 1))
    t_mbd_end = time.time()

    # --- Chunk 2 (Cached Warm Synthesis) ---
    t_inf2_start = time.time()
    with torch.no_grad():
        encoded_frames2 = model.inference(
            text_tokens.to(vm.page.device),
            text_tokens_lens.to(vm.page.device),
            audio_prompts,
            enroll_x_lens=enroll_x_lens,
            top_k=50,
            temperature=0.95,
        )
    t_inf2_end = time.time()

    t_mbd2_start = time.time()
    with torch.no_grad():
        samples2 = audio_tokenizer.mbd_decode(encoded_frames2.transpose(2, 1))
    t_mbd2_end = time.time()

    print("\n--- Benchmark Results ---")
    print(f"1. Init VoiceMaker/GradioPage:        {t1 - t0:.2f}s")
    print(f"2. Load Models (VALL-E + MBD):        {t_load_end - t_load_start:.2f}s")
    print(f"3. Chunk 1 VALL-E Inference:          {t_inf_end - t_inf_start:.2f}s ({encoded_frames.shape[-1]} frames)")
    print(f"4. Chunk 1 MBD Decode:                {t_mbd_end - t_mbd_start:.2f}s")
    print(f"Total Chunk 1 Time:                   {t_mbd_end - t0:.2f}s")
    print(f"5. Chunk 2 VALL-E Inference (Warm):   {t_inf2_end - t_inf2_start:.2f}s")
    print(f"6. Chunk 2 MBD Decode (Warm):         {t_mbd2_end - t_mbd2_start:.2f}s")
    print(f"Total Chunk 2 Time (Warm):            {t_mbd2_end - t_inf2_start:.2f}s")


if __name__ == "__main__":
    run_benchmark()
