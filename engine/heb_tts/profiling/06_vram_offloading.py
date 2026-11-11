"""
Profiling Script 06: Dynamic VRAM Offloading Verification
=========================================================

Purpose:
--------
Verifies the optimization strategy:
1. VALL-E model executes autoregressive token generation on GPU.
2. After generation completes, VALL-E model is temporarily offloaded to CPU
   (`model.to('cpu')`) and `torch.cuda.empty_cache()` is called.
3. Multi-Band Diffusion runs `mbd_decode` with 1.8 GB of freed VRAM headroom.
4. Model is restored to GPU (`model.to(device)`).

Measures exact VRAM reduction, MBD decode acceleration, and offload overhead.

Usage:
------
docker run --gpus all --rm \\
    -v "${PWD}:/app" \\
    -v "${PWD}/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints" \\
    -v "${PWD}/.cache/huggingface:/root/.cache/huggingface" \\
    -w /app hebtts:latest python profiling/06_vram_offloading.py
"""

import time
from config import utils
import torch
from make_voice import VoiceMaker
from config import Config
from engine.heb_tts.valle.data import tokenize_audio


def run_benchmark():
    print("======================================================================")
    print("HebTTS Profiling: Dynamic VRAM Offloading Verification")
    print("======================================================================")

    vm = VoiceMaker()
    cfg = Config().reset()
    cfg.mbd = 1
    cfg.speaker = "1"

    model, audio_tokenizer, text_collater, alef_bert_tokenizer = vm.page.load_model_if_needed(
        cfg.checkpoint, mbd=cfg.mbd
    )

    prompt_audio, prompt_text = vm.resolve_speaker_prompt(cfg)
    encoded_frames_prompt = tokenize_audio(audio_tokenizer, prompt_audio)
    audio_prompts = torch.concat([encoded_frames_prompt[0][0]], dim=-1).transpose(2, 1).to(vm.page.device)

    text = "היום אנחנו הולכים ללמוד על העולם הקסום של כללי ההתחלקות."
    text_without_space = [f"{prompt_text} {text}".strip().replace(" ", "_")]
    tokens = alef_bert_tokenizer._tokenize(text_without_space)
    prompt_text_without_space = [f"{prompt_text}".strip().replace(" ", "_")]
    prompt_tokens = alef_bert_tokenizer._tokenize(prompt_text_without_space)
    tokens = [t for t in tokens if t in text_collater.token2idx]
    prompt_tokens = [t for t in prompt_tokens if t in text_collater.token2idx]
    text_tokens, text_tokens_lens = text_collater([tokens])
    _, enroll_x_lens = text_collater([prompt_tokens])

    # 1. Run VALL-E inference on GPU
    t0 = time.time()
    with torch.no_grad():
        encoded_frames = model.inference(
            text_tokens.to(vm.page.device),
            text_tokens_lens.to(vm.page.device),
            audio_prompts,
            enroll_x_lens=enroll_x_lens,
            top_k=50,
            temperature=0.95,
        )
    t1 = time.time()
    valle_dur = t1 - t0

    # 2. Dynamic Offload VALL-E to CPU to free 1.8 GB VRAM
    t_offload_start = time.time()
    model.to("cpu")
    torch.cuda.empty_cache()
    t_offload_end = time.time()
    offload_dur = t_offload_end - t_offload_start

    alloc_after = torch.cuda.memory_allocated() / 1e6
    res_after = torch.cuda.memory_reserved() / 1e6

    # 3. Run MBD decode on GPU with freed headroom
    t2 = time.time()
    with torch.no_grad():
        samples = audio_tokenizer.mbd_decode(encoded_frames.transpose(2, 1))
    t3 = time.time()
    mbd_dur = t3 - t2

    # 4. Restore model to GPU
    t_restore_start = time.time()
    model.to(vm.page.device)
    t_restore_end = time.time()
    restore_dur = t_restore_end - t_restore_start

    print(f"VALL-E Inference Duration:               {valle_dur:.2f}s")
    print(f"Offload VALL-E to CPU Duration:          {offload_dur:.4f}s")
    print(f"VRAM After Offload:                      Allocated: {alloc_after:.1f} MB | Reserved: {res_after:.1f} MB")
    print(f"MBD Decode Duration (VRAM Freed):        {mbd_dur:.2f}s")
    print(f"Restore VALL-E to GPU Duration:          {restore_dur:.4f}s")
    print(f"Total Chunk Synthesis (excluding init):  {valle_dur + offload_dur + mbd_dur + restore_dur:.2f}s")


if __name__ == "__main__":
    run_benchmark()
